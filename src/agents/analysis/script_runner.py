#!/usr/bin/env python3
"""
Sandboxed execution for generated analysis scripts
src/agents/analysis/script_runner.py

    in   a Python source string + an AnalysisContext
    out  {ok, result, figure, script, error}  — and the script saved to disk

NOT A PIPELINE STEP, deliberately unnumbered. Running generated code against a
dataframe is infrastructure, not a stage: step 2 writes the scripts today, and
if step 3 ever re-runs one to verify a claim it uses this same runner.

WHY A SUBPROCESS AND NOT exec(). Two things exec() in-process cannot give:

    a real timeout   a script with `while True` or a pathological groupby hangs
                     the whole Analyzer. SIGALRM only interrupts between
                     bytecodes and will not break out of a C-level pandas call.
    crash isolation  a segfault in a native extension takes the parent with it.

The subprocess also makes the saved script genuinely re-runnable by hand, which
is the point of keeping it: a reviewer can open the file and execute it.

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
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

# Below this a "finding" is not a finding. Matches the threshold step2_derive
# already used for drivers ("fewer than 3 columns had a value") — with 19
# columns and one year, two points are an anecdote.
MIN_N = 3

DEFAULT_TIMEOUT_S = 180

_PREAMBLE = '''\
import json, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open({payload!r}, "rb") as _f:
    _ctx = pickle.load(_f)
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

        full = (_PREAMBLE.format(payload=str(pkl), out_path=str(out_path))
                + textwrap.dedent(code).rstrip() + "\n"
                + _POSTAMBLE.format(result_path=str(res)))
        script_path = (Path(script_path).resolve() if script_path
                       else tmp / "script.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(full)

        # Inherit the env: the analysis stack (pandas, matplotlib, cartopy)
        # lives in the conda env the Analyzer is already running in, and a
        # scrubbed environment would just fail to import it.
        env = dict(os.environ, MPLBACKEND="Agg")
        try:
            proc = subprocess.run([sys.executable, str(script_path)],
                                  capture_output=True, text=True,
                                  timeout=timeout_s, env=env, cwd=str(tmp))
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
                "script_path": str(script_path), "stdout": proc.stdout}
    finally:
        for p in (pkl, res):
            try:
                p.unlink()
            except Exception:
                pass
