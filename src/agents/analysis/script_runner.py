#!/usr/bin/env python3
"""
Constrained execution for generated analysis scripts
src/agents/analysis/script_runner.py

    in   a Python source string + an AnalysisContext
    out  {ok, result, figure, script, script_sha256, error}
         — and the script saved to disk

NOT A PIPELINE STEP, deliberately unnumbered. Running generated code against a
dataframe is infrastructure, not a stage: step 2 writes the scripts today, and
if step 3 ever re-runs one to verify a claim it uses this same runner.

WHAT THIS IS NOT — READ THIS BEFORE TRUSTING IT.

This module was called "sandboxed" until 2026-08-14 and it is not a sandbox.
The script runs as the invoking user, on the real filesystem, with no
namespace, no seccomp filter and no container. What is enforced:

    process isolation   a segfault in a native extension cannot take the
                        Analyzer with it
    a wall timeout      DEFAULT_TIMEOUT_S, enforced by the subprocess, so it
                        interrupts C-level pandas that SIGALRM would not
    resource ceilings   address space, CPU seconds, maximum file size, no core
                        dump — see _limits()
    an environment      an ALLOWLIST, not the inherited environment. This
                        deployment exports a USGS API key, AmeriFlux
                        credentials and a HydroFrame PIN before anything runs;
                        none of them now reach a figure script
    a scope guardrail   inspect_code() refuses a script that imports
                        subprocess, socket, requests and friends, or calls
                        eval/exec/os.system

What is NOT enforced: network access (blocking sockets needs namespaces or
root; the proxy variables are pointed at a dead port, which stops the libraries
that honour them and nothing else), filesystem reads outside the run directory,
and anything a determined bypass of the AST check would do. THE THREAT MODEL IS
CARELESSNESS, NOT MALICE — a model doing something out of scope, not a model
attacking the host. Treat this as running trusted-but-unreviewed code.

WHY A SUBPROCESS AND NOT exec(). Two things exec() in-process cannot give:

    a real timeout   a script with `while True` or a pathological groupby hangs
                     the whole Analyzer. SIGALRM only interrupts between
                     bytecodes and will not break out of a C-level pandas call.
    crash isolation  a segfault in a native extension takes the parent with it.

The subprocess also makes the saved script genuinely re-runnable by hand, which
is the point of keeping it: a reviewer can open the file and execute it. That
became true on 2026-08-14 — the preamble had loaded a pickle from a temporary
directory this module deletes, so every saved script died on a missing file the
moment its round ended. It now rebuilds the payload from the run directory when
the pickle is gone.

WHAT THE SCRIPT SEES. A namespace assembled here, and nothing else it did not
import itself:

    df        the tidy long frame: date | entity | variable | value | units | source
              None when the run has no daily series (standalone PFLOTRAN)
    prof      the tidy DEPTH frame, for backends with a vertical axis:
              entity | time_y | depth_m | variable | value | units | source
              None when the run has no profiles (ELM)
    soil      the tidy SOIL frame — one row per day per LAYER, with each
              layer's own geometry:
              entity | date | layer | depth_m | thickness_m | depth_top_m |
              depth_bottom_m | variable | value | units | source
              None when the run has no layered output (PFLOTRAN)
    columns   per-column metadata (lat, lon, elevation_m, band, soil_profile, ...)
    caveats   the constraint records — so a script can read what it must respect
    out_path  where to save the figure

A run supplies df + soil (ELM) or prof (PFLOTRAN); a script must check which
before using it. All three being None is refused below rather than passed
through.

A COLUMN MEAN OVER `soil` IS THICKNESS-WEIGHTED, always: ELM's layers span
1.75 cm to 13.85 m, so an unweighted mean reports the bedrock.

WHAT IT MUST RETURN. A dict named `result`, carrying at minimum an `n`: how many
data points the claim rests on. That requirement is the whole reason this file
exists. `soil_attribution` filtered on a key extraction never populated, returned
{}, and the figure simply did not appear — for the entire history of the study,
with nothing anywhere recording that a planned analysis had produced nothing. A
generated script reintroduces that failure fresh on every run: a filter that
matches nothing, a merge on a misaligned key, a .dropna() that removes
everything. Each returns a clean, plausible, empty answer.

So an empty or too-small result is NOT a silent skip here. It comes back with
ok=False and a reason, and the caller turns it into a caveat.
"""
import ast
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# Below this a "finding" is not a finding. Matches the threshold step2_derive
# already used for drivers ("fewer than 3 columns had a value") — with 19
# columns and one year, two points are an anecdote.
MIN_N = 3

DEFAULT_TIMEOUT_S = 180

# ─────────────────────────────────────────────────────────────────────
# THE LIMITS
# ─────────────────────────────────────────────────────────────────────
MAX_MEMORY_BYTES = 8 * 1024 ** 3      # 8 GB address space
MAX_CPU_SECONDS  = 300                 # complements the wall timeout below
MAX_FILE_BYTES   = 512 * 1024 ** 2     # any single file the script writes

# THE ENVIRONMENT IS AN ALLOWLIST (2026-08-14). It used to be
# `dict(os.environ, MPLBACKEND="Agg")` — the whole environment, inherited. On
# this deployment that includes a USGS API key, AmeriFlux credentials and a
# HydroFrame PIN, all exported by env_compy.sh before anything runs. Generated
# analysis code reads its data from a pickle and has no business seeing any of
# them.
#
# DENY BY DEFAULT, because a denylist of credential names only blocks the
# secrets someone remembered. What a pandas/matplotlib script legitimately
# needs is small and knowable, so that is what it gets. Prefixes cover the
# families whose exact names vary by machine (CONDA_*, LC_*, the geospatial
# stack's own configuration).
_ENV_KEEP = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP",
    "LANG", "TZ", "TERM",
    "LD_LIBRARY_PATH", "LD_PRELOAD",
    "PYTHONPATH", "PYTHONHOME", "PYTHONUNBUFFERED", "PYTHONIOENCODING",
    "MPLBACKEND", "MPLCONFIGDIR", "MATPLOTLIBRC",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
}
_ENV_KEEP_PREFIX = ("LC_", "CONDA_", "PROJ_", "GDAL_", "HDF5_", "NETCDF",
                    "CARTOPY_", "XDG_")

# WHAT A FIGURE SCRIPT HAS NO REASON TO TOUCH. This is a GUARDRAIL, NOT A
# SANDBOX — see the module docstring. It catches the careless case (a script
# that shells out, opens a socket, or writes outside out_path) and it is
# trivially bypassed by anything trying to. It is here because the realistic
# failure is a model doing something out of scope, not a model attacking the
# host, and a refusal naming the import is a far better diagnostic than
# whatever that import would have done.
_FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "shutil", "requests", "urllib", "urllib3",
    "http", "ftplib", "smtplib", "telnetlib", "paramiko", "ctypes",
    "multiprocessing", "importlib", "pty", "pickle", "shelve", "marshal",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint",
                    "input"}
# `os` and `sys` are allowed — os.path is ordinary in plotting code — but
# these members of them are not.
_FORBIDDEN_ATTRS = {
    "os": {"system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve",
           "execl", "fork", "forkpty", "remove", "unlink", "rmdir", "removedirs",
           "rename", "chmod", "chown", "setuid", "setgid", "environ"},
    "sys": {"exit", "_exit"},
    "shutil": {"rmtree", "move", "copy", "copytree"},
}


def save_exchange(out_dir, step: str, round_no: int,
                  prompt: str, reply: str) -> Dict[str, str]:
    """What the model was shown and what it said, verbatim, beside the run.

    NEITHER SURVIVED BEFORE 2026-08-14. Step 2's reply came back from propose()
    as `raw`, was parsed, and the original was dropped when the record was
    written; step 3 did not even keep a name for it. So two questions about a
    finished run had no answer:

        "the model proposed five figures and four appeared — what happened to
         the fifth?"      the four that parsed are on record. One that was
                          mangled on the way in left no trace it was proposed.

        "did the parser change what the model meant?"
                          _parse() repairs replies that are not quite valid
                          JSON — escaping raw newlines inside generated Python,
                          stripping comments. Usually right. When one is wrong
                          the original is already gone.

    THE PROMPT IS KEPT TOO, and that is not symmetry for its own sake. The
    worst bug this pipeline has had was a silent 700-character cap on each
    finding: the model was shown a fifth of a result and told to quote from it
    exactly, and the audit then struck nine of fourteen true claims. Nobody
    could see that because nobody could see what was sent.

    Writing these is also what makes an end-to-end test possible without
    calling a model: a recorded reply replays identically and for nothing,
    where a live call costs money and answers differently every time.

    Never fatal. A run that produced figures is not lost because a log could
    not be written.
    """
    out: Dict[str, str] = {}
    try:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for kind, body in (("prompt", prompt), ("reply", reply)):
            if body is None:
                continue
            p = d / f"{step}_round{round_no}_{kind}.txt"
            p.write_text(str(body))
            out[kind] = str(p)
    except OSError as e:                                        # noqa: BLE001
        print(f"   ⚠️  could not save the {step} exchange ({e}) — the run "
              f"stands, but this round cannot be replayed")
    return out


def inspect_code(code: str) -> List[str]:
    """Objections to a generated script, as a list of reasons. Empty is fine.

    Parsed with `ast`, not matched with regexes: `import subprocess` inside a
    string literal or a comment is not an import, and a regex cannot tell.
    A file that does not parse is reported as one objection, which is a better
    failure than handing unparseable source to a subprocess.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"the generated code does not parse: {e}"]

    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _FORBIDDEN_IMPORTS:
                    out.append(f"imports {a.name!r}, which a figure script has "
                               f"no use for")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                out.append(f"imports from {node.module!r}, which a figure "
                           f"script has no use for")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                out.append(f"calls {f.id}(), which is not analysis")
            elif (isinstance(f, ast.Attribute)
                  and isinstance(f.value, ast.Name)
                  and f.attr in _FORBIDDEN_ATTRS.get(f.value.id, ())):
                out.append(f"calls {f.value.id}.{f.attr}(), which is not "
                           f"analysis")
        elif (isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name)
              and node.attr in _FORBIDDEN_ATTRS.get(node.value.id, ())):
            out.append(f"uses {node.value.id}.{node.attr}, which is not "
                       f"analysis")
    return sorted(set(out))


def _child_env() -> Dict[str, str]:
    """The environment the script runs in: allowlisted, plus what it needs."""
    env = {k: v for k, v in os.environ.items()
           if k in _ENV_KEEP or k.startswith(_ENV_KEEP_PREFIX)}
    env["MPLBACKEND"] = "Agg"
    # NOT A NETWORK JAIL, and labelled as such. Blocking sockets needs
    # namespaces or root, neither of which is available here. Pointing the
    # proxy variables at a dead port stops the libraries that honour them
    # (requests, urllib) and does nothing to a raw socket. The real protection
    # is above: with no credentials in the environment, a script that does
    # reach the network has nothing to authenticate with.
    env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:1"
    env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:1"
    env["no_proxy"] = ""
    return env


def _limits() -> None:
    """Applied in the child between fork and exec.

    Each is a ceiling the analysis has no legitimate reason to reach, and each
    turns a hang or a runaway into a clean non-zero exit the caller reports as
    a caveat. RLIMIT_CPU complements the wall-clock timeout rather than
    duplicating it: a script blocked on I/O burns wall time and no CPU, and one
    spinning in C burns both.
    """
    for what, limit in ((resource.RLIMIT_AS, MAX_MEMORY_BYTES),
                        (resource.RLIMIT_CPU, MAX_CPU_SECONDS),
                        (resource.RLIMIT_FSIZE, MAX_FILE_BYTES),
                        (resource.RLIMIT_CORE, 0)):
        try:
            soft, hard = resource.getrlimit(what)
            ceiling = limit if hard in (resource.RLIM_INFINITY,) \
                else min(limit, hard)
            resource.setrlimit(what, (ceiling, hard))
        except (ValueError, OSError):
            # A limit the platform will not set is not a reason to refuse the
            # run — the timeout and the process boundary still hold.
            pass

# RE-RUNNABLE AFTER THE FACT (2026-08-14). The payload is a pickle in a
# temporary directory this module deletes in its `finally`, so a saved script
# was readable provenance and nothing more: run it tomorrow and it died on a
# missing file. The point of keeping the script is that a reviewer can execute
# it, and that was never true.
#
# It now falls back to REBUILDING the payload from the run directory — the same
# path step 0 takes, so a script re-run in a month reads what the figure was
# drawn from rather than an approximation of it. The pickle stays as the fast
# path because it is already in hand while the round is running.
_PREAMBLE = '''\
import json, pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PAYLOAD = {payload!r}
_RUN_DIR = {run_dir!r}
_FRAMEWORK = {framework!r}

if Path(_PAYLOAD).is_file():
    with open(_PAYLOAD, "rb") as _f:
        _ctx = pickle.load(_f)
else:
    # The temporary payload is gone; rebuild it from the run directory.
    sys.path.insert(0, str(Path(_FRAMEWORK) / "src"))
    from agents.analysis import step0_context as _s0
    from agents.analysis import script_runner as _sr
    _ctx = _sr._payload(_s0.load(_RUN_DIR))

df       = _ctx["df"]
prof     = _ctx["prof"]
soil     = _ctx["soil"]
columns  = _ctx["columns"]
caveats  = _ctx["caveats"]
out_path = {out_path!r}

result = None
# ── generated ────────────────────────────────────────────────────────────
'''

_POSTAMBLE = '''
# ── /generated ───────────────────────────────────────────────────────────
try:
    plt.close("all")
except Exception:
    pass
with open({result_path!r}, "w") as _f:
    json.dump(result, _f, default=str)
'''


SECONDS_PER_DAY = 86400.0


def _to_daily_rates(df, units: Dict[str, str]):
    """RELABEL per-second fluxes as per-day. No arithmetic — the values are
    already correct.

    `variable_units` says `mm/s` because that is the unit of the RAW ELM
    variable. The daily series it labels are not raw: the extractor's _daily()
    resampled 3-hourly output and converted fluxes to mm/day on the way into
    experiment.json. So the numbers are per-day and the label says per-second,
    and everything downstream trusts the label.

    That mislabel is the bug, and it is a live one for humans too — a generated
    script read `mm/s`, dutifully multiplied a year of correct daily values by
    86400, and plotted an annual runoff of 1.1e6 mm/yr. My first fix multiplied
    them a second time, which is how the mislabel was finally found: the label
    was wrong, not the data.

    Relabelling here keeps the frame self-describing for whatever reads it. The
    real repair belongs upstream, where _daily() should write the unit it
    actually produced.
    """
    per_sec = [v for v, u in (units or {}).items() if str(u).endswith("/s")]
    if df is None or not per_sec or "variable" not in getattr(df, "columns", []):
        return df, []
    df = df.copy()
    hit = df["variable"].isin(per_sec)
    if "units" in df.columns:
        df.loc[hit, "units"] = (df.loc[hit, "units"].astype(str)
                                .str.replace("/s", "/day", regex=False))
    return df, sorted(per_sec)


def _payload(ctx) -> Dict[str, Any]:
    """What the script is allowed to see.

    ctx.series() is the tidy frame; when pandas is missing it returns None and
    the runner refuses rather than handing the script a namespace with `df`
    silently absent — a NameError deep in generated code is a worse diagnostic
    than a stated precondition.
    """
    df, converted = _to_daily_rates(ctx.series(),
                                    (ctx.data or {}).get("variable_units"))

    # The depth frame, for backends whose output has a vertical axis rather
    # than a daily one. `prof` is None for an ELM run and `df` is None for a
    # standalone PFLOTRAN run; a script gets whichever the run actually has.
    try:
        prof = ctx.profiles()
    except Exception:
        prof = None

    # The soil column through time — one row per day per LAYER, each carrying
    # its own thickness. ELM has this and no `prof`; PFLOTRAN is the other way
    # round. Bound unconditionally so a script can test it rather than meet a
    # NameError.
    try:
        soil = ctx.soil()
    except Exception:
        soil = None

    # THE RAW `variables` BLOB IS WITHHELD, and this is the second half of the
    # unit fix rather than a size optimisation. Each column carries its daily
    # series twice: once here in mm/s, and once in `df` — which _to_daily_rates
    # has corrected. Converting only `df` left the other copy open, and the very
    # next generated script reached straight past the corrected frame into
    # columns[i]["variables"] for ET and produced 2.8e7 mm/yr.
    #
    # One path to the daily data, already unit-correct. `metrics` stays: those
    # are the precomputed per-column aggregates, and field_semantics records
    # what each was derived from.
    cols = [{k: v for k, v in c.items() if k != "variables"} for c in ctx.columns]

    return {"df": df,
            "prof": prof,
            "soil": soil,
            "columns": cols,
            "caveats": list(getattr(ctx, "caveats", []) or []),
            "converted_to_daily": converted}


def run(code: str, ctx, out_path, script_path=None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        min_n: int = MIN_N) -> Dict[str, Any]:
    """Execute `code`, returning a verdict rather than raising.

    Every failure mode is a returned reason, never an exception: the caller
    turns each into a caveat, and a step that crashed on one bad script would
    lose the four good ones alongside it.
    """
    # ABSOLUTE, because the script runs with cwd=tmp. A relative out_path or
    # script_path silently stops resolving the moment the working directory
    # changes: every figure in the first real Analyzer run failed with "can't
    # open file", and every earlier test passed only because it happened to
    # hand in absolute scratchpad paths. The entry point that matters passes a
    # relative run_dir.
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # REFUSED BEFORE IT RUNS. A figure script that shells out or opens a socket
    # is out of scope whatever it then does, and naming the import is a far
    # better diagnostic — for the caller's caveat and for round 2's feedback —
    # than whatever the import would have produced.
    objections = inspect_code(textwrap.dedent(code))
    if objections:
        return {"ok": False,
                "error": "refused before running — " + "; ".join(objections),
                "result": None, "figure": None, "script": code,
                "objections": objections}

    payload = _payload(ctx)
    if all(payload[k] is None for k in ("df", "prof", "soil")):
        return {"ok": False,
                "error": "no tidy frame to run against — the run has no daily "
                         "series, no soil layers and no depth profiles (or "
                         "pandas is unavailable)",
                "result": None, "figure": None, "script": code}

    tmp = Path(tempfile.mkdtemp(prefix="step2_"))
    pkl, res = tmp / "ctx.pkl", tmp / "result.json"
    try:
        import pickle
        with open(pkl, "wb") as f:
            pickle.dump(payload, f)

        full = (_PREAMBLE.format(
                    payload=str(pkl), out_path=str(out_path),
                    # RESOLVED, for the same reason out_path is: the script
                    # runs with cwd=tmp, and a relative run_dir stops resolving
                    # the moment the working directory changes. The entry point
                    # that matters passes a relative one.
                    run_dir=str(Path(getattr(ctx, "run_dir", "") or ".")
                                .resolve()),
                    framework=str(Path(__file__).resolve().parents[3]))
                + textwrap.dedent(code).rstrip() + "\n"
                + _POSTAMBLE.format(result_path=str(res)))
        script_path = (Path(script_path).resolve() if script_path
                       else tmp / "script.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(full)
        # THE GENERATED BODY, not the assembled file. The preamble
        # embeds the payload path, the run directory and out_path, all
        # of which differ between two runs of the SAME analysis — so
        # hashing the whole file answers "is this the same file" when
        # the question is "is this the same analysis".
        digest = hashlib.sha256(
            textwrap.dedent(code).strip().encode()).hexdigest()[:16]

        try:
            proc = subprocess.run([sys.executable, str(script_path)],
                                  capture_output=True, text=True,
                                  timeout=timeout_s, env=_child_env(),
                                  cwd=str(tmp), preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout_s}s",
                    "result": None, "figure": None, "script": full,
                    "script_path": str(script_path)}

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            return {"ok": False,
                    "error": "script failed: " + (tail[-1] if tail else "no stderr"),
                    "stderr": proc.stderr, "result": None, "figure": None,
                    "script": full, "script_path": str(script_path)}

        # ── the guards ───────────────────────────────────────────────────
        if not res.exists():
            return {"ok": False, "error": "script never assigned `result`",
                    "result": None, "figure": None, "script": full,
                    "script_path": str(script_path)}
        try:
            value = json.loads(res.read_text())
        except Exception as e:
            return {"ok": False, "error": f"`result` was not JSON-serialisable: {e}",
                    "result": None, "figure": None, "script": full,
                    "script_path": str(script_path)}

        if not isinstance(value, dict):
            return {"ok": False,
                    "error": f"`result` must be a dict, got {type(value).__name__}",
                    "result": value, "figure": None, "script": full,
                    "script_path": str(script_path)}

        n = value.get("n")
        if not isinstance(n, (int, float)):
            return {"ok": False,
                    "error": "`result` has no numeric `n` — a finding must say "
                             "how many data points it rests on",
                    "result": value, "figure": None, "script": full,
                    "script_path": str(script_path)}
        if n < min_n:
            return {"ok": False,
                    "error": f"n={n} is below the floor of {min_n}. This is the "
                             f"soil_attribution failure: a filter that matched "
                             f"nothing returns a clean, plausible, empty answer.",
                    "result": value, "figure": None, "script": full,
                    "script_path": str(script_path)}

        if not out_path.exists() or out_path.stat().st_size < 3000:
            return {"ok": False,
                    "error": "no figure was written to out_path",
                    "result": value, "figure": None, "script": full,
                    "script_path": str(script_path)}

        return {"ok": True, "error": None, "result": value,
                "figure": str(out_path), "script": full,
                # THE SCRIPT THAT PRODUCED THIS RESULT, identified. Two runs of
                # the same study can now be compared on whether the analysis
                # was the same analysis, which "it drew a figure with the same
                # name" does not answer.
                "script_sha256": digest,
                "script_path": str(script_path), "stdout": proc.stdout}
    finally:
        for p in (pkl, res):
            try:
                p.unlink()
            except Exception:
                pass
