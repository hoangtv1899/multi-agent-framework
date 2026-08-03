#!/usr/bin/env python3
"""
ELM MCP Server — compile and run E3SM Land Model cases.

Phase 4 of the resumability work; design in docs/ELM_MCP_PLAN.md.

ONE RULE, AND THE WHOLE SURFACE FOLLOWS FROM IT
-----------------------------------------------
**This server compiles and runs the model. The framework decides what to run
and reads what came out.**

So it does NOT generate inputs and does NOT extract results. It takes the case
list the framework wrote — one entry per column, carrying that column's file
paths and run settings — builds CIME cases against it, runs them, and reports
what the scheduler is doing.

That boundary is where it is because of what the inputs actually are. A
column's surface data and its FINIDAT come from the WARM START: the framework
subsets the CONUS restart, snaps the column to its donor gridcell and takes
that cell's soil. Those are the two inputs that make a column credible, and
they are decisions, not compilation. A server that generated inputs itself
would offer a path that produces a cold, template-soil column at a coordinate
— runnable, and quietly worse science.

Everything the warm start decided therefore crosses as DATA, in each case's
runtime_config: FSURDAT, FINIDAT, LND_DOMAIN_*, ATM_DOMAIN_*, STOP_N,
RUN_STARTDATE, DATM_CLMNCEP_YR_*, REST_*. This side never opens a surfdata
file.

EVERY TOOL RETURNS QUICKLY OR RETURNS A JOB ID
-----------------------------------------------
Not a style preference — a consequence of how MCP works. `MCPManager` opens a
fresh stdio session per call and tearing it down kills the server and its
children. Against a CIME case build (~8-10 min, measured) under a 300 s timeout
that does not produce a slow call, it produces a case directory killed halfway.
So both slow stages submit to SLURM and hand back an id; the framework's
_advance/_poll machinery (Phase 3b) knows what to do with one.

THE ENVIRONMENT PROBLEM — read this before adding a tool
--------------------------------------------------------
`mcp.client.stdio.get_default_environment()` forwards **only** HOME, LOGNAME,
PATH, SHELL and USER. Our own MCPManager happens to forward os.environ.copy()
(src/core/mcp_client.py:78); a standard client does not. That difference is why
every LAMBDA tool in the reaction MCP worked through this framework and failed
under Claude Code — same server, same machine, different launcher.

**So every path this server needs is resolved HERE, with a default.** The list
is short precisely because inputs are somebody else's job: E3SM/CIME, a scratch
filesystem, and a UTF-8 locale CIME's python refuses to run without.

Registered in mcp_config.json as `elm`.

Tools:
    describe_elm_capabilities()  -> what this does, needs, and does NOT do
    build_elm_cases(...)         -> a JOB ID; CIME cases from the case list
    submit_elm_ensemble(...)     -> a JOB ID; the ensemble as one batch job
    check_elm_job(...)           -> what SLURM is doing, and the built case
                                    directories once a build job has landed
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT — every one of these has a default (see the header)
# ─────────────────────────────────────────────────────────────────────
FRAMEWORK = Path(os.getenv(
    "IDEAS_FRAMEWORK_DIR",
    str(Path(__file__).resolve().parents[2])))

os.environ.setdefault("PSCRATCH", "/compyfs/tran289")
os.environ.setdefault("E3SM_SRC_DIR", "/qfs/people/tran289/E3SM")
# CIME's python assumes a UTF-8 locale and dies on the C locale. Set here
# rather than per-subprocess so a tool added later cannot forget it —
# src/core/elm_wrapper.py learned this the hard way.
os.environ.setdefault("LC_ALL", "en_US.utf8")
os.environ.setdefault("LANG",   "en_US.utf8")
# SLURM defaults for the jobs this server submits.
os.environ.setdefault("IDEAS_SLURM_ACCOUNT", "e3sm")
os.environ.setdefault("IDEAS_SLURM_QUEUE",   "short")

# The framework's own modules. This server is a thin front for them rather than
# a reimplementation — the case build is ELMExperimentBuilder either way.
sys.path.insert(0, str(FRAMEWORK / "src"))

from mcp.server.fastmcp import FastMCP                          # noqa: E402

mcp = FastMCP("elm")

SUBMIT_SCRIPT = FRAMEWORK / "tools" / "submit_cases.sh"
BUILD_JOB     = Path(__file__).resolve().parent / "build_cases_job.py"

# What the framework writes, and what this server reads.
CASE_INPUTS  = "case_inputs.json"
BUILT_CASES  = "built_cases.json"


# ─────────────────────────────────────────────────────────────────────
# WHAT IS ACTUALLY HERE
# ─────────────────────────────────────────────────────────────────────
def _requirements() -> dict:
    """Every external thing this server needs, and whether it is present.

    CHECKED, not asserted. A capabilities tool that lists what it assumes is
    worth nothing on the machine where one of those assumptions is false —
    which is exactly the case it exists to diagnose, since a wrong path here
    surfaces eight minutes into a CIME build as an unrelated-looking error.

    Short, because generating inputs is not this server's job: no CONUS
    surfdata, no ELM input-file templates. Those belong to whoever decides
    what the columns are.
    """
    def _dir(p):  return {"path": str(p), "present": Path(p).is_dir()}
    def _file(p): return {"path": str(p), "present": Path(p).is_file()}

    reqs = {
        "framework":     _dir(FRAMEWORK),
        "submit_script": _file(SUBMIT_SCRIPT),
        "build_job":     _file(BUILD_JOB),
        "e3sm_source":   _dir(os.environ["E3SM_SRC_DIR"]),
        "cime":          _dir(Path(os.environ["E3SM_SRC_DIR"]) / "cime"),
        "scratch":       _dir(os.environ["PSCRATCH"]),
    }
    for name in ("sbatch", "squeue", "sacct"):
        reqs[name] = {"path": shutil.which(name) or "",
                      "present": bool(shutil.which(name))}
    return reqs


def _imports() -> dict:
    """Whether the framework modules this server fronts can be imported. They
    pull in CIME's own tree, which can be missing in a differently-launched
    process."""
    out = {}
    for mod in ("core.elm_experiment_builder", "core.elm_input_agent"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception as e:                                  # noqa: BLE001
            out[mod] = f"{type(e).__name__}: {e}"
    return out


@mcp.tool()
def describe_elm_capabilities() -> str:
    """What this server can do, what it needs, and whether that is present.

    Call this FIRST. It reports the workflow in order, and a checked inventory
    of every external dependency — so a missing E3SM tree or an absent sbatch
    is a one-line answer here rather than a confusing failure eight minutes
    into a case build.

    It also states plainly what this server does NOT do, which is most of what
    makes an ELM study a study: the sampling design, the warm start, the input
    generation, and reading the results.
    """
    reqs = _requirements()
    missing = sorted(k for k, v in reqs.items() if not v["present"])
    imports = _imports()
    broken = sorted(k for k, v in imports.items() if v is not True)

    return json.dumps({
        "server": "elm",
        "model": "E3SM Land Model (ELM), 1-D columns",
        "ready": not missing and not broken,
        "missing_requirements": missing,
        "broken_imports": broken,

        "rule": (
            "This server compiles and runs the model. The caller decides what "
            "to run and reads what came out."
        ),
        "contract": (
            "Every tool returns quickly or returns a job id. No tool blocks on "
            "the science: an MCP client opens a fresh session per call and "
            "tearing it down kills this server's children, so a 10-minute CIME "
            "build under a 300 s timeout is not a slow call, it is a half-built "
            "case directory."
        ),

        "workflow": [
            {"step": 1, "tool": "build_elm_cases",
             "does": f"CIME cases from 01_inputs/{CASE_INPUTS} — reference "
                     f"case compiled, the rest cloned with --keepexe",
             "returns": "a JOB ID; the case directories come back from "
                        "check_elm_job once it lands"},
            {"step": 2, "tool": "check_elm_job",
             "does": "ask SLURM what a job from step 1 or 3 is doing",
             "returns": "state, whether it is still active, and for a build "
                        "job the case directories it produced"},
            {"step": 3, "tool": "submit_elm_ensemble",
             "does": "run every column concurrently as one batch job",
             "returns": "a JOB ID"},
        ],

        "inputs_expected": {
            "file": f"<run_dir>/01_inputs/{CASE_INPUTS}",
            "shape": "[{case_name, runtime_config: {FSURDAT, FINIDAT, "
                     "LND_DOMAIN_FILE, LND_DOMAIN_PATH, ATM_DOMAIN_FILE, "
                     "ATM_DOMAIN_PATH, STOP_N, STOP_OPTION, RUN_STARTDATE, "
                     "DATM_CLMNCEP_YR_START, DATM_CLMNCEP_YR_END, REST_N, "
                     "REST_OPTION}}, ...]",
            "note": "absolute paths to files that already exist. This server "
                    "does not generate them.",
        },

        "does_not": [
            "choose where to put columns — the sampling design belongs to the "
            "caller, and is shared with PFLOTRAN so the two are comparable",
            "warm-start anything. FINIDAT and the donor soil come from "
            "subsetting the CONUS restart, which is a decision about the "
            "science rather than a compilation step",
            "generate surface or domain files",
            "read history files or compute any result",
            "decide whether a study is worth running, record caveats, or write "
            "experiment.json",
        ],

        "requirements": reqs,
        "imports": imports,
        "environment": {k: os.environ.get(k) for k in (
            "PSCRATCH", "E3SM_SRC_DIR", "LC_ALL",
            "IDEAS_SLURM_ACCOUNT", "IDEAS_SLURM_QUEUE")},
        "note": (
            "Every path above is resolved by this server with a default, not "
            "inherited: an MCP client forwards only HOME, LOGNAME, PATH, SHELL "
            "and USER. Override any of them by exporting it before the server "
            "starts."
        ),
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────
# BUILD THE CASES  (a job — D1)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def build_elm_cases(run_dir:  str,
                    queue:    str = "",
                    walltime: str = "02:00:00",
                    account:  str = "") -> str:
    """Compile the CIME cases. SUBMITS a job and returns its id.

    Reads <run_dir>/01_inputs/case_inputs.json — the case list the caller
    wrote — and builds a case per entry: the reference case compiled from
    scratch (~8-10 min, measured), the rest cloned with --keepexe (~30 s each).

    sbatch'd rather than run inline for two reasons, both measurements: no MCP
    client's timeout tolerates a ten-minute call, and compiling on a login node
    is antisocial.

    Poll the returned job_id with check_elm_job; it returns the case
    directories once the job is no longer active.

    Walltime defaults to two hours because a cold compile of a large ensemble
    is the case that must not be cut off, and an unused reservation costs
    nothing.
    """
    rd = Path(run_dir).resolve()
    src = rd / "01_inputs" / CASE_INPUTS
    if not src.is_file():
        return json.dumps({
            "error": f"no {CASE_INPUTS} in {rd / '01_inputs'} — this server "
                     f"does not generate inputs; the caller writes them"})
    if not BUILD_JOB.is_file():
        return json.dumps({"error": f"missing {BUILD_JOB}"})

    try:
        n_cases = len(json.loads(src.read_text()))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"unreadable {CASE_INPUTS}: {e}"})
    if not n_cases:
        return json.dumps({"error": f"{CASE_INPUTS} is empty"})

    # A stale result from an earlier attempt would be read as this job's answer
    # the moment it is polled.
    (rd / "01_inputs" / BUILT_CASES).unlink(missing_ok=True)

    q = queue or os.environ["IDEAS_SLURM_QUEUE"]
    acct = account or os.environ["IDEAS_SLURM_ACCOUNT"]
    sb = rd / "build_cases.sbatch"
    # sys.executable, not "python": the job gets a login shell with no conda
    # environment, and CIME's scripts are particular about the interpreter.
    sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_build
#SBATCH -N 1
#SBATCH -p {q}
#SBATCH -A {acct}
#SBATCH -t {walltime}
#SBATCH -o {rd}/build_cases.log
export IDEAS_FRAMEWORK_DIR={FRAMEWORK}
export PSCRATCH={os.environ['PSCRATCH']}
export LC_ALL=en_US.utf8
export LANG=en_US.utf8
cd {FRAMEWORK}
echo "building {n_cases} case(s) for {rd} on $(hostname)"
{sys.executable} {BUILD_JOB} {rd}
""")

    try:
        jid = subprocess.check_output(
            ["sbatch", "--parsable", str(sb)], text=True, timeout=120).strip()
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"sbatch failed: {e}"})

    return json.dumps({
        "job_id":   jid.split(";")[0],
        "n_cases":  n_cases,
        "stage":    "build",
        "queue":    q,
        "walltime": walltime,
        "log_path": str(rd / "build_cases.log"),
        "next":     "check_elm_job",
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────
# RUN THE ENSEMBLE  (a job)
# ─────────────────────────────────────────────────────────────────────
def _exeroot(case_dir: str) -> Optional[str]:
    """Where CIME put the executable, asked of the case itself.

    xmlquery rather than a constructed path: EXEROOT depends on the machine
    config and on whether the case was cloned with --keepexe, and a guess that
    is wrong produces a job that starts and immediately finds no binary.
    """
    try:
        out = subprocess.check_output(
            ["./xmlquery", "EXEROOT", "--value"], cwd=case_dir,
            env={**os.environ, "LC_ALL": "en_US.utf8", "LANG": "en_US.utf8"},
            text=True, timeout=120)
        return out.strip() or None
    except Exception:                                           # noqa: BLE001
        return None


@mcp.tool()
def submit_elm_ensemble(run_dir:   str,
                        case_dirs: List[str],
                        queue:     str = "",
                        walltime:  str = "00:40:00",
                        email:     str = "") -> str:
    """Run every column concurrently as ONE batch job. Returns a JOB ID.

    RETURNS IMMEDIATELY — it does not wait for the ensemble. That is the point:
    19 columns took 2406 s through SLURM, and a tool that blocked for that
    would be killed by its own client's timeout long before finishing.

    Poll the returned job_id with check_elm_job. Reading the results is the
    caller's job, not this server's.
    """
    rd = Path(run_dir)
    cases = [c for c in (case_dirs or []) if c and Path(c).is_dir()]
    if not cases:
        return json.dumps({"error": "no existing case directories were given"})
    if not SUBMIT_SCRIPT.is_file():
        return json.dumps({"error": f"missing {SUBMIT_SCRIPT}"})
    rd.mkdir(parents=True, exist_ok=True)

    exeroot = _exeroot(cases[0])
    if not exeroot:
        return json.dumps({
            "error": f"could not resolve EXEROOT from {cases[0]} — the case "
                     f"may not be built yet (run build_elm_cases first)"})

    # submit_cases.sh reads these two out of the run dir.
    (rd / "cases.json").write_text(json.dumps(cases, indent=1))
    (rd / "exe_path.txt").write_text(str(Path(exeroot) / "e3sm.exe") + "\n")

    cmd = ["bash", str(SUBMIT_SCRIPT), str(rd),
           "-q", queue or os.environ["IDEAS_SLURM_QUEUE"],
           "-t", walltime]
    if email:
        cmd += ["-m", email]

    try:
        proc = subprocess.run(cmd, cwd=str(FRAMEWORK), check=False,
                              capture_output=True, text=True, timeout=300)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"submission failed: {e}"})

    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"submitted job (\d+)", out)
    if not m:
        # No id means nothing to poll. Say so rather than return a success the
        # caller can never follow up on.
        return json.dumps({"error": "submitted but no job id in the output",
                           "output": out[-2000:]})

    return json.dumps({
        "job_id":   m.group(1),
        "n_cases":  len(cases),
        "stage":    "run",
        "queue":    queue or os.environ["IDEAS_SLURM_QUEUE"],
        "walltime": walltime,
        "log_path": str(rd / "run.log"),
        "next":     "check_elm_job",
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────
# WHAT IS THE SCHEDULER DOING
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def check_elm_job(job_id: str, run_dir: str = "") -> str:
    """What is SLURM doing with this job? Works for build and run alike.

    `active` is the field to branch on. It is false only when the scheduler has
    genuinely finished with the job — not when the scheduler could not be
    reached, which is reported as state=null and active=true, because a squeue
    timeout must never be read as a completed ensemble.

    For a BUILD job this also returns `cases`: where each case landed, once the
    job has written them. That is why there is no separate collect tool for the
    build — a stage that hands back a job id needs somewhere to hand back its
    answer, and this is it. The case directories are a few short strings, so
    they come back inline; results never would, which is why reading those is
    the caller's job.

    `ready` is reported separately from `active`. The scheduler finishing and
    the result file becoming readable here are different instants — the job
    writes to a parallel filesystem from a compute node and the submitting
    host's view of it lags.
    """
    from core.exp_manager_base import ExperimentManagerBase as _B

    state = _B._slurm_state(job_id)
    known = state is not None
    active = (state in _B.ACTIVE_JOB_STATES) if known else True

    out: Dict[str, Any] = {"job_id": str(job_id), "state": state,
                           "active": active, "scheduler_answered": known}
    if not known:
        out["note"] = ("the scheduler will not say — treat as still running "
                       "unless the output says otherwise")

    if not run_dir:
        return json.dumps(out, indent=2)

    rd = Path(run_dir)
    built = rd / "01_inputs" / BUILT_CASES
    if built.is_file():
        try:
            d = json.loads(built.read_text())
            out["stage"] = "build"
            out["ready"] = True
            out["ok"] = d.get("ok")
            out["cases"] = d.get("cases") or []
            out["n_ok"] = d.get("n_ok")
            out["n_total"] = d.get("n_total")
            if not d.get("ok"):
                out["error"] = d.get("error")
            if not known:
                out["active"] = False
                out["note"] = "gone from SLURM, but the build wrote its result"
            return json.dumps(out, indent=2)
        except Exception as e:                                  # noqa: BLE001
            out["error"] = f"unreadable {BUILT_CASES}: {e}"

    # A run job: run.log is the only evidence once sacct has purged the record.
    log = rd / "run.log"
    if log.is_file():
        txt = log.read_text(errors="replace")
        out["stage"] = out.get("stage", "run")
        out["log_says_all_done"] = "ALL_DONE" in txt
        out["log_tail"] = "\n".join(txt.strip().splitlines()[-8:])
        if not known and out["log_says_all_done"]:
            out["active"] = False
            out["note"] = "gone from SLURM, but the log says ALL_DONE"
    out.setdefault("ready", not out["active"])
    return json.dumps(out, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
