#!/usr/bin/env python3
"""
ELM MCP Server — E3SM Land Model ensembles as tools.

Phase 4 of the resumability work; design in docs/ELM_MCP_PLAN.md.

WHAT THIS IS FOR
----------------
The framework runs on a login node and ELM's stages run inside the framework
process, which couples where the science is decided to where the compute is
arranged. This server holds the second half: case building, submission and
collection. The framework keeps the ledger, the strategy gate and the packaging.

THE RULE THIS SERVER OBEYS
--------------------------
**Every tool either returns quickly, or returns a job id. No tool blocks on the
science.**

Not a style preference — a consequence of how MCP works. `MCPManager` opens a
fresh stdio session per call and tearing it down kills the server and its
children. Against a CIME case build (~8-10 min, measured) under a 300 s timeout
that does not produce a slow call, it produces a case directory killed halfway.
So the two slow stages, `prepare` and `run`, both submit to SLURM and hand back
an id; the framework's `_advance`/`_poll` machinery (Phase 3b) already knows
what to do with one.

THE ENVIRONMENT PROBLEM — read this before adding a tool
--------------------------------------------------------
`mcp.client.stdio.get_default_environment()` forwards **only** HOME, LOGNAME,
PATH, SHELL and USER. Our own MCPManager happens to forward os.environ.copy()
(src/core/mcp_client.py:78); a standard client does not. That difference is why
every LAMBDA tool in the reaction MCP worked through this framework and failed
under Claude Code — same server, same machine, different launcher.

ELM needs far more than LAMBDA did: the E3SM source tree, CIME, $PSCRATCH, the
input files, and a UTF-8 locale that CIME's python refuses to run without.

**So every path this server needs is resolved HERE, with a default.** Anything
read from the ambient environment is a tool that works for us and fails for
anyone else — and fails differently rather than loudly, which is worse.

Registered in mcp_config.json as `elm`.

WHAT CANNOT CROSS THE BOUNDARY
------------------------------
`ELMExperimentBuilder` returns experiment dicts carrying a live `ELMAgentAdapter`
under `elm_agent`. An object cannot be JSON, and the framework already learned
what happens when one is serialised anyway: it comes back as its repr — truthy,
attribute-free, and happy to be passed around until something calls a method on
it (Phase 3, `_rehydrate_handles`).

So the manifest carries the adapter's INPUTS — `case_name` and `runtime_config`,
both plain data — and `prepare_elm_cases` reconstructs the adapter on this side.
Nothing is smuggled across as a repr.

Tools:
    describe_elm_capabilities()  -> what this server does, what it needs, and
                                    whether each of those is actually present
    build_elm_cases(...)         -> per-column inputs + a case manifest
    prepare_elm_cases(...)       -> a JOB ID; CIME case build, sbatch'd (D1)
    collect_prepared_cases(...)  -> where that job put each case
    submit_elm_ensemble(...)     -> a JOB ID; the ensemble, one batch job
    check_elm_job(...)           -> what SLURM is doing with either job
    collect_elm_results(...)     -> a PATH to the extraction + a summary
"""
import json
import os
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
os.environ.setdefault("ELM_INPUT_FILES_DIR",
                      "/qfs/people/tran289/IDEAS/1d_elm/input_files")
os.environ.setdefault("CONUS_SURFDATA_NC",
                      "/qfs/people/tran289/IDEAS/1d_elm/conus_surfdata/"
                      "surfdata_0.5x0.5_rcp3.0_simyr2015_c231113.nc")
# CIME's python calls sys.setdefaultencoding paths that assume a UTF-8 locale
# and dies on the C locale. Set here rather than per-subprocess so a tool added
# later cannot forget it — src/core/elm_wrapper.py learned this the hard way.
os.environ.setdefault("LC_ALL", "en_US.utf8")
os.environ.setdefault("LANG",   "en_US.utf8")
# SLURM defaults for the jobs this server submits.
os.environ.setdefault("IDEAS_SLURM_ACCOUNT", "e3sm")
os.environ.setdefault("IDEAS_SLURM_QUEUE",   "short")

# The framework's own modules — the builder and the wrapper live there, and
# this server is a thin front for them rather than a reimplementation.
sys.path.insert(0, str(FRAMEWORK / "src"))

from mcp.server.fastmcp import FastMCP                          # noqa: E402

mcp = FastMCP("elm")

SUBMIT_SCRIPT = FRAMEWORK / "tools" / "submit_cases.sh"


# ─────────────────────────────────────────────────────────────────────
# WHAT IS ACTUALLY HERE
# ─────────────────────────────────────────────────────────────────────
def _requirements() -> dict:
    """Every external thing this server needs, and whether it is present.

    CHECKED, not asserted. A capabilities tool that lists what it assumes is
    worth nothing on a machine where one of those assumptions is false — which
    is exactly the case this exists to diagnose, since a wrong path here
    surfaces eight minutes into a CIME build as an unrelated-looking error.
    """
    def _dir(p):  return {"path": str(p), "present": Path(p).is_dir()}
    def _file(p): return {"path": str(p), "present": Path(p).is_file()}

    reqs = {
        "framework":        _dir(FRAMEWORK),
        "submit_script":    _file(SUBMIT_SCRIPT),
        "e3sm_source":      _dir(os.environ["E3SM_SRC_DIR"]),
        "cime":             _dir(Path(os.environ["E3SM_SRC_DIR"]) / "cime"),
        "scratch":          _dir(os.environ["PSCRATCH"]),
        "elm_input_files":  _dir(os.environ["ELM_INPUT_FILES_DIR"]),
        "conus_surfdata":   _file(os.environ["CONUS_SURFDATA_NC"]),
    }
    for name in ("sbatch", "squeue", "sacct"):
        reqs[name] = {"path": shutil.which(name) or "",
                      "present": bool(shutil.which(name))}
    return reqs


def _imports() -> dict:
    """Whether the framework modules this server fronts can actually be
    imported. They pull in xarray, netCDF4 and CIME's own tree, any of which
    can be missing in a differently-launched process."""
    out = {}
    for mod in ("core.elm_experiment_builder", "core.elm_wrapper",
                "core.elm_results_analyzer", "core.columns_to_plan"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception as e:                                  # noqa: BLE001
            out[mod] = f"{type(e).__name__}: {e}"
    return out


@mcp.tool()
def describe_elm_capabilities() -> str:
    """What this server can do, what it needs, and whether that is present.

    Call this FIRST. It reports the intended workflow in order, which tools
    exist today versus which are still planned, and a checked inventory of
    every external dependency — so a missing E3SM tree or an absent sbatch is a
    one-line answer here rather than a confusing failure eight minutes into a
    case build.

    It also states plainly what this server does NOT do, because an ensemble is
    not a study: there is no sampling design, no strategy gate, no caveat
    record and no experiment.json unless the framework is driving.
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

        "contract": (
            "Every tool either returns quickly or returns a job id. No tool "
            "blocks on the science: an MCP client opens a fresh session per "
            "call and tearing it down kills the server's children, so a "
            "10-minute CIME build under a 300 s timeout is not a slow call, it "
            "is a half-built case directory."
        ),

        "workflow": [
            {"step": 1, "tool": "build_elm_cases",
             "does": "per-column surface/domain inputs + a case manifest",
             "returns": "manifest path and the case list",
             "status": "available"},
            {"step": 2, "tool": "prepare_elm_cases",
             "does": "CIME case build (reference case, then --keepexe clones)",
             "returns": "a JOB ID — this is sbatch'd, not run inline",
             "status": "available"},
            {"step": "2b", "tool": "collect_prepared_cases",
             "does": "read where the prepare job put each case",
             "returns": "case_dir per case, or the reason the build failed",
             "status": "available"},
            {"step": 3, "tool": "submit_elm_ensemble",
             "does": "run every column concurrently as one batch job",
             "returns": "a JOB ID",
             "status": "available"},
            {"step": 4, "tool": "check_elm_job",
             "does": "ask SLURM what a job from step 2 or 3 is doing",
             "returns": "state and whether it is still active",
             "status": "available"},
            {"step": 5, "tool": "collect_elm_results",
             "does": "read the history files into rows",
             "returns": "a PATH to the analysis plus a compact summary — not "
                        "the series, which for 19 columns over 20 years is "
                        "tens of megabytes and would not survive stdio",
             "status": "available"},
        ],

        "available_now": ["describe_elm_capabilities", "build_elm_cases",
                          "prepare_elm_cases", "collect_prepared_cases",
                          "submit_elm_ensemble", "check_elm_job",
                          "collect_elm_results"],

        "does_not": [
            "choose where to put columns (that is the framework's sampling "
            "design, shared with PFLOTRAN so the two are comparable)",
            "decide whether a study is worth running (the strategy gate)",
            "record caveats or limitations",
            "write experiment.json, or track a run across sessions — that is "
            "the framework's ledger",
        ],

        "requirements": reqs,
        "imports": imports,
        "environment": {k: os.environ.get(k) for k in (
            "PSCRATCH", "E3SM_SRC_DIR", "ELM_INPUT_FILES_DIR",
            "CONUS_SURFDATA_NC", "LC_ALL", "IDEAS_SLURM_ACCOUNT",
            "IDEAS_SLURM_QUEUE")},
        "note": (
            "Every path above is resolved by this server with a default, not "
            "inherited: an MCP client forwards only HOME, LOGNAME, PATH, SHELL "
            "and USER. Override any of them by exporting it before the server "
            "starts."
        ),
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD
# ─────────────────────────────────────────────────────────────────────
MANIFEST_NAME = "elm_mcp_manifest.json"

# Keys of an experiment dict that are plain data and mean something to a
# caller. Listed rather than "everything except elm_agent" so a new object
# added upstream cannot silently become a repr in the manifest.
_MANIFEST_KEYS = (
    "scenario_index", "scenario_name", "case_name", "forcing_period",
    "soil_config", "substrate", "forcing_start", "forcing_end", "stop_n",
    "start_date", "description", "lat", "lon", "elevation_m", "band",
    "case_dir",
)


def _serialisable(exp: Dict[str, Any]) -> Dict[str, Any]:
    """One experiment, with the live adapter replaced by what built it.

    `runtime_config` is the adapter's whole input — STOP_N, the forcing years,
    FINIDAT, and the generated domain/surface paths — so prepare_elm_cases can
    rebuild an identical adapter without the object ever being serialised.
    """
    out = {k: exp.get(k) for k in _MANIFEST_KEYS if k in exp}
    agent = exp.get("elm_agent")
    rc = getattr(agent, "runtime_config", None)
    if isinstance(rc, dict):
        out["runtime_config"] = dict(rc)
    return out


def _resolve_plan(run_dir: Path,
                  run_plan: Optional[Dict[str, Any]],
                  columns:  Optional[List[Dict[str, Any]]],
                  soil_config: str, substrate: str,
                  yr_start: int, yr_end: int,
                  forcing_period: str) -> Dict[str, Any]:
    """D2: one tool, both input shapes.

    The exp manager has a materialized run_plan and passes it. An agent driving
    this server has coordinates and does not — `columns_to_elm_plan` is what
    turns the second into the first, and it is the same converter the framework
    uses, so the two paths cannot drift.
    """
    if run_plan:
        return run_plan
    if columns:
        from core.columns_to_plan import columns_to_elm_plan
        return columns_to_elm_plan(
            columns, forcing_period=forcing_period,
            yr_start=int(yr_start), yr_end=int(yr_end),
            soil_config=soil_config, substrate=substrate)
    p = run_dir / "run_plan.json"
    if p.is_file():
        return json.loads(p.read_text())
    raise ValueError(
        "no plan: pass run_plan, or columns, or put run_plan.json in run_dir")


@mcp.tool()
def build_elm_cases(run_dir:        str,
                    run_plan:       Optional[Dict[str, Any]] = None,
                    columns:        Optional[List[Dict[str, Any]]] = None,
                    soil_config:    str = "native",
                    substrate:      str = "extrapolate",
                    yr_start:       int = 1995,
                    yr_end:         int = 1999,
                    forcing_period: str = "baseline") -> str:
    """Generate each column's ELM inputs and write a case manifest.

    Takes EITHER a materialized `run_plan` (what the framework has) OR a list
    of `columns` with lat/lon/elevation (what an agent has); with neither it
    reads run_plan.json from run_dir. Columns are converted with the framework's
    own converter, so the two entry points cannot drift apart.

    This does NOT build CIME cases — that is prepare_elm_cases, which is
    sbatch'd because it takes ~8-10 minutes. This step writes domain and
    surface files and works out what each case will be.

    Returns the manifest path, the per-case list, and which columns failed to
    get inputs. A column that fails here is reported rather than dropped: an
    ensemble quietly one column short is a different study from the one asked
    for.
    """
    rd = Path(run_dir)
    if not rd.is_dir():
        return json.dumps({"error": f"run_dir does not exist: {rd}"})
    (rd / "01_inputs").mkdir(parents=True, exist_ok=True)

    # build_all reads columns.json from the run dir — that file is the SNAPPED
    # coordinates, written after any warm start, which is the whole reason it
    # is read rather than the plan.
    if columns and not (rd / "columns.json").is_file():
        (rd / "columns.json").write_text(
            json.dumps({"columns": columns}, indent=2, default=str))

    try:
        plan = _resolve_plan(rd, run_plan, columns, soil_config, substrate,
                             yr_start, yr_end, forcing_period)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": str(e)})

    # Pre-build the per-column inputs. Non-fatal by design upstream: the
    # builder retains its own generation path, so a failure here costs the
    # early warning and the manifest, not the run.
    inputs = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_column_inputs", FRAMEWORK / "tools" / "build_column_inputs.py")
        bci = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bci)
        inputs = bci.build_all(rd, soil_config=soil_config,
                               substrate=substrate, quiet=True) or {}
    except Exception as e:                                      # noqa: BLE001
        inputs = {"error": f"{type(e).__name__}: {e}"}

    try:
        from core.elm_experiment_builder import ELMExperimentBuilder
        builder = ELMExperimentBuilder(plan)
        experiments = builder.build_experiments()
        summary = builder.get_experiment_summary()
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"building experiments failed: "
                                    f"{type(e).__name__}: {e}"})

    # The prepare job runs on a compute node with none of this call's
    # arguments, so the resolved plan is written down here. Without it,
    # prepare_elm_cases has nothing to rebuild the experiment list from.
    plan_file = rd / "run_plan.json"
    if not plan_file.is_file():
        plan_file.write_text(json.dumps(plan, indent=2, default=str))

    cases = [_serialisable(e) for e in experiments]
    manifest = rd / "01_inputs" / MANIFEST_NAME
    manifest.write_text(json.dumps(cases, indent=2, default=str))
    (rd / "01_inputs" / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    return json.dumps({
        "manifest_path": str(manifest),
        "n_cases":       len(cases),
        "cases":         cases,
        "inputs_built":  len(inputs.get("built") or {}),
        "inputs_failed": inputs.get("failed") or {},
        "warm_started":  bool(inputs.get("warm_started")),
        "next":          "prepare_elm_cases",
    }, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — PREPARE (D1: sbatch'd, job-shaped)
# ─────────────────────────────────────────────────────────────────────
PREPARE_JOB = Path(__file__).resolve().parent / "prepare_job.py"
PREPARED_NAME = "prepared_cases.json"


@mcp.tool()
def prepare_elm_cases(run_dir:  str,
                      queue:    str = "",
                      walltime: str = "02:00:00",
                      account:  str = "") -> str:
    """Build the CIME cases. SUBMITS a job and returns its id.

    D1: this is sbatch'd rather than run inline. Two reasons and both are
    measurements — the reference case takes ~8-10 minutes to compile, which no
    MCP client's timeout will tolerate, and doing that on a login node is
    antisocial. Clones are ~30 s each with --keepexe.

    Poll the returned job_id with check_elm_job, then collect_prepared_cases
    once it is no longer active.

    Needs run_plan.json in run_dir — build_elm_cases writes it. Walltime
    defaults to two hours because a cold compile of a large ensemble is the
    case that must not be cut off, and an unused reservation costs nothing.
    """
    rd = Path(run_dir).resolve()
    if not (rd / "run_plan.json").is_file():
        return json.dumps({
            "error": f"no run_plan.json in {rd} — run build_elm_cases first"})
    if not PREPARE_JOB.is_file():
        return json.dumps({"error": f"missing {PREPARE_JOB}"})
    (rd / "01_inputs").mkdir(parents=True, exist_ok=True)

    # A stale result from an earlier attempt would be read as this job's
    # answer the moment it is polled.
    stale = rd / "01_inputs" / PREPARED_NAME
    if stale.exists():
        stale.unlink()

    q = queue or os.environ["IDEAS_SLURM_QUEUE"]
    acct = account or os.environ["IDEAS_SLURM_ACCOUNT"]
    sb = rd / "prepare_cases.sbatch"
    # sys.executable, not "python": the job inherits a login shell with no
    # conda environment, and CIME's own scripts are picky about which
    # interpreter finds them.
    sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_prep
#SBATCH -N 1
#SBATCH -p {q}
#SBATCH -A {acct}
#SBATCH -t {walltime}
#SBATCH -o {rd}/prepare.log
export IDEAS_FRAMEWORK_DIR={FRAMEWORK}
export PSCRATCH={os.environ['PSCRATCH']}
export ELM_INPUT_FILES_DIR={os.environ['ELM_INPUT_FILES_DIR']}
export CONUS_SURFDATA_NC={os.environ['CONUS_SURFDATA_NC']}
export LC_ALL=en_US.utf8
export LANG=en_US.utf8
cd {FRAMEWORK}
echo "preparing cases for {rd} on $(hostname)"
{sys.executable} {PREPARE_JOB} {rd}
""")

    try:
        jid = subprocess.check_output(
            ["sbatch", "--parsable", str(sb)], text=True, timeout=120).strip()
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"sbatch failed: {e}"})

    return json.dumps({
        "job_id":   jid.split(";")[0],
        "queue":    q,
        "walltime": walltime,
        "log_path": str(rd / "prepare.log"),
        "result_path": str(rd / "01_inputs" / PREPARED_NAME),
        "next":     "check_elm_job, then collect_prepared_cases",
    }, indent=2)


@mcp.tool()
def collect_prepared_cases(run_dir: str) -> str:
    """Where the prepare job put each case. Call once its job is inactive.

    A seventh tool, where the design said six: that was written when prepare
    was going to block, and a stage that hands back a job id needs somewhere to
    hand back its ANSWER. The alternative was the framework reading the
    server's files directly, which is not a boundary.

    Returns ok=false with the reason when the build failed — a case that did
    not compile is a failure to report, not an ensemble to run one column short.
    """
    p = Path(run_dir) / "01_inputs" / PREPARED_NAME
    if not p.is_file():
        log = Path(run_dir) / "prepare.log"
        return json.dumps({
            "ok": False,
            "error": "no result file — the prepare job has not finished, or "
                     "it died before writing one",
            "log_tail": ("\n".join(log.read_text(errors="replace")
                                   .strip().splitlines()[-15:])
                         if log.is_file() else ""),
        }, indent=2)
    try:
        return json.dumps(json.loads(p.read_text()), indent=2)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "error": f"unreadable result: {e}"})


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — SUBMIT, and ask what happened
# ─────────────────────────────────────────────────────────────────────
def _exeroot(case_dir: str) -> Optional[str]:
    """Where CIME put the executable for this case, asked of the case itself.

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

    RETURNS IMMEDIATELY — it does not wait for the ensemble. That is the whole
    point: 19 columns took 2406 s through SLURM, and a tool that blocked for
    that would be killed by its own client's timeout long before finishing.

    Poll the returned job_id with check_elm_job, then collect_elm_results once
    it is no longer active.
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
                     f"may not be built yet (run prepare_elm_cases first)"})

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
    import re
    m = re.search(r"submitted job (\d+)", out)
    if not m:
        # No id means nothing to poll. Say so rather than returning a success
        # the caller can never follow up on.
        return json.dumps({"error": "submitted but no job id in the output",
                           "output": out[-2000:]})

    return json.dumps({
        "job_id":   m.group(1),
        "n_cases":  len(cases),
        "queue":    queue or os.environ["IDEAS_SLURM_QUEUE"],
        "walltime": walltime,
        "log_path": str(rd / "run.log"),
        "next":     "check_elm_job",
    }, indent=2)


@mcp.tool()
def check_elm_job(job_id: str, run_dir: str = "") -> str:
    """What is SLURM doing with this job? Works for prepare and run alike.

    `active` is the field to branch on. It is false only when the scheduler has
    genuinely finished with the job — not when the scheduler could not be
    reached, which is reported as state=null and active=true, because a squeue
    timeout must never be read as a completed ensemble.

    With run_dir given, also reports whether the job's log says ALL_DONE — the
    only evidence available when sacct has purged the record.
    """
    # The FRAMEWORK's implementation, imported rather than reimplemented. Two
    # versions of "is this job finished" that could disagree is a worse problem
    # than the coupling.
    from core.exp_manager_base import ExperimentManagerBase as _B

    state = _B._slurm_state(job_id)
    known = state is not None
    active = (state in _B.ACTIVE_JOB_STATES) if known else True

    out = {"job_id": str(job_id), "state": state, "active": active,
           "scheduler_answered": known}

    if not known:
        out["note"] = ("the scheduler will not say — treat as still running "
                       "unless the log says otherwise")

    if run_dir:
        log = Path(run_dir) / "run.log"
        if log.is_file():
            txt = log.read_text(errors="replace")
            out["log_says_all_done"] = "ALL_DONE" in txt
            out["log_tail"] = "\n".join(txt.strip().splitlines()[-8:])
            if not known and out["log_says_all_done"]:
                out["active"] = False
                out["note"] = "gone from SLURM, but the log says ALL_DONE"
        else:
            out["log_says_all_done"] = False

    return json.dumps(out, indent=2)


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — COLLECT
# ─────────────────────────────────────────────────────────────────────
RESULTS_NAME = "elm_mcp_results.json"


@mcp.tool()
def collect_elm_results(run_dir:       str,
                        manifest_path: str = "",
                        last_year_only: bool = False) -> str:
    """Read the ensemble's history files. Returns a PATH plus a summary.

    NOT the series. Nineteen columns of daily output over twenty years is tens
    of megabytes, and an MCP response is JSON over stdio — inlining it would
    pass every small test and fail on a real watershed. The full extraction is
    written to run_dir/04_analysis/ and to a results file whose path comes back
    here; the caller reads it.

    A column with no history files is reported as failed, with a reason. The
    scheduler having finished is what ends the wait; the FILES decide the
    outcome, and an ensemble quietly one column short is a different study from
    the one that was asked for.
    """
    rd = Path(run_dir)
    mp = Path(manifest_path) if manifest_path else rd / "01_inputs" / MANIFEST_NAME
    if not mp.is_file():
        return json.dumps({"ok": False,
                           "error": f"no manifest at {mp} — run build_elm_cases"})

    try:
        cases = json.loads(mp.read_text())
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "error": f"unreadable manifest: {e}"})

    # Case dirs come from the prepare job when there is one; the manifest is
    # written before the cases exist, so it usually has none.
    prepared = rd / "01_inputs" / PREPARED_NAME
    if prepared.is_file():
        try:
            got = {c.get("case_name"): c.get("case_dir")
                   for c in (json.loads(prepared.read_text()).get("cases") or [])}
            for c in cases:
                c.setdefault("case_dir", None)
                if got.get(c.get("case_name")):
                    c["case_dir"] = got[c["case_name"]]
        except Exception:                                       # noqa: BLE001
            pass

    missing = [c["case_name"] for c in cases if not c.get("case_dir")]
    if len(missing) == len(cases):
        return json.dumps({
            "ok": False,
            "error": "no case directories are known — has prepare_elm_cases "
                     "run, and collect_prepared_cases been called?"})

    try:
        from core.elm_results_analyzer import ELMResultsAnalyzer
        an = ELMResultsAnalyzer(experiments=cases,
                                analysis_dir=str(rd / "04_analysis"),
                                last_year_only=bool(last_year_only))
        rows = an.extract_all()
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False,
                           "error": f"extraction failed: {type(e).__name__}: {e}"})

    payload = rd / "04_analysis" / RESULTS_NAME
    payload.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload.write_text(json.dumps(an.get_llm_analysis_input(),
                                      indent=2, default=str))
    except Exception as e:                                      # noqa: BLE001
        payload = None
        print(f"could not write {RESULTS_NAME}: {e}", file=sys.stderr)

    # The COMPACT summary — what a caller needs to decide what to do next,
    # without the series.
    columns = []
    for name, r in (rows or {}).items():
        r = r or {}
        columns.append({
            "case_name": name,
            "ok":        r.get("status") == "ok",
            "status":    r.get("status"),
            "reason":    r.get("reason") or r.get("error"),
            "n_history_files": len(r.get("history_files") or []),
        })
    n_ok = sum(1 for c in columns if c["ok"])

    return json.dumps({
        "ok":             n_ok > 0,
        "n_ok":           n_ok,
        "n_total":        len(columns),
        "columns":        columns,
        "results_path":   str(payload) if payload else None,
        "analysis_dir":   str(rd / "04_analysis"),
        "cases_without_directories": missing,
        "note": ("results_path holds the full extraction. It is a path and not "
                 "inline data on purpose: the series for a real ensemble does "
                 "not fit in an MCP response."),
    }, indent=2, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
