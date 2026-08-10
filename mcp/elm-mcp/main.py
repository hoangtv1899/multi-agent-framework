#!/usr/bin/env python3
"""
ELM MCP Server — compile and run E3SM Land Model cases.

Phase 4 of the resumability work; design in docs/ELM_MCP_PLAN.md.

ONE RULE, AND THE WHOLE SURFACE FOLLOWS FROM IT
-----------------------------------------------
**Anything that requires knowing what ELM is lives here.**

The caller decides WHERE the columns go — that is watershed geometry, shared
with PFLOTRAN so the two are comparable. Everything downstream of a coordinate
is this server's: the warm start, the donor soil, the surfaces, the thirteen
CIME keys, the build, the run, and reading what came out.

THE RULE THIS REPLACED, and why. The boundary used to be "does it need the
scheduler?", which kept input generation on the framework side. That rule is
coherent and it is also what forced a per-model manager class to exist: ELM
knowledge barred from the server needs a home, and that home was
ELMExpManager. Three attempts to delete that class failed until the rule moved.

WHAT THE OLD RULE WAS RIGHT ABOUT. A tool taking bare coordinates and inventing
the rest produces a cold, template-soil column — runnable, and quietly worse
science. That is an argument about a SIGNATURE, not a location:
build_elm_inputs_from_location reads the columns the caller sampled, requires
the CONUS restarts, and is fatal when any column cannot be warm-started.

EVERY TOOL RETURNS QUICKLY OR RETURNS A JOB ID
-----------------------------------------------
Input generation is fast and local and returns DATA. Building and running are
slow and return an ID. That is the sharpest line in this server, and the second
half is not a style preference — a consequence of how MCP works. `MCPManager` opens a
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

Tools — all seven, and describe_elm_capabilities advertises all seven:
    describe_elm_capabilities()          -> what this does, needs, and does NOT do
    build_elm_inputs_from_location(...)  -> DATA; snapped columns + case_inputs.json
    get_column_metadata(...)             -> DATA; the columns as they will be RUN
    build_elm_cases(...)                 -> a JOB ID; CIME cases from the case list
    submit_elm_ensemble(...)             -> a JOB ID; the ensemble as one batch job
    run_elm_study(...)                   -> a JOB ID; build + run + analyze, one job
    check_elm_job(...)                   -> what SLURM is doing, and the built case
                                            directories once a build job has landed
"""
import contextlib
import functools
import io
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

# A WORKING python3 ON PATH, which is not the same thing as an interpreter.
# CIME's create_newcase is `#!/usr/bin/env python3` and tools/submit_cases.sh
# parses its case list with one, so both take whatever PATH offers. A client
# forwards PATH but not LD_LIBRARY_PATH (see the header), and the first python3
# on a stock PATH here is a module build that cannot load libpython3.11.so
# without it — so the build died 60 s in with a shared-library error inside
# create_newcase, which reads as anything but "the launcher dropped a variable"
# (job 770824). This interpreter is self-contained; put its directory first.
os.environ["PATH"] = (str(Path(sys.executable).parent) + os.pathsep
                      + os.environ.get("PATH", ""))

# THE MODULE SYSTEM, for the same reason. CIME's env_mach_specific runs
# `modulecmd python load cmake gcc intel intelmpi netcdf pnetcdf mkl` to build
# the case, and modulecmd finds nothing without MODULEPATH — a variable the
# login shell sets and no MCP client forwards. Symptom, on job 770825: "ERROR:
# No module path defined", ~30 s into a build that had already passed
# create_newcase. Compy's five module trees; export MODULEPATH before starting
# the server on any other machine.
os.environ.setdefault("MODULESHOME", "/share/apps/modules")
os.environ.setdefault("MODULEPATH", ":".join(
    f"/share/apps/modules/modulefiles/{d}" for d in (
        "environment", "development/mpi", "development/mlib",
        "development/compilers", "development/tools")))

# The framework's own modules. This server is a thin front for them rather than
# a reimplementation — the case build is ELMExperimentBuilder either way.
sys.path.insert(0, str(FRAMEWORK / "src"))
# This server's own library. Inserted after the framework's so a name that
# exists in both resolves HERE — the ELM code is migrating into this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mcp.server.fastmcp import FastMCP                          # noqa: E402
import paths                                                    # noqa: E402

mcp = FastMCP("elm")


# ─────────────────────────────────────────────────────────────────────────────
# STDOUT IS THE PROTOCOL. Every tool body runs with stdout redirected to stderr.
#
# This is a stdio MCP server: the client reads JSON-RPC frames from our stdout,
# so ANY print() inside a tool corrupts the transport. The moved modules print
# freely — 54 calls across inputs.py and elm_exp_manager.py, progress lines that
# are genuinely useful when the same code is driven from a terminal — and on
# 2026-08-07 the first call to build_elm_inputs_from_location through the
# protocol produced a stream of
#
#     Failed to parse JSONRPC message from server
#     Invalid JSON: input_value='🌡️  STEP 0b: Warm Start (CONUS subset)'
#
# and then timed out, because the reply was lost in the noise.
#
# Phase 1a checked this tool by calling the functions DIRECTLY and comparing 285
# fields. That is why it missed the defect entirely: parity across a function
# call says nothing about a transport. A decorator rather than 54 edits, so a
# print added later cannot reintroduce it, and the messages still reach a human
# on stderr.
# REDIRECTED TO A FILE, NOT TO STDERR. The first version of this sent tool
# output to sys.stderr and every MCP call then hung to the 600 s timeout while
# the same work took 8 s called directly. redirect_stdout swaps sys.stdout
# GLOBALLY, so in an async server the JSON-RPC reply can be written while the
# swap is active and disappear down the same pipe as the progress text — the
# client waits for a frame that was never delivered. A file has neither problem:
# nothing shares it, nothing parses it, and it cannot fill and block.
_LOG_PATH = os.environ.get(
    "IDEAS_ELM_MCP_LOG",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "elm_mcp_tools.log"))


def _stdout_to_stderr(fn):
    """Keep tool print() off the protocol channel.

    stdout IS the JSON-RPC transport for a stdio server, and the moved modules
    print 54 progress lines. They go to a log file so a human can still read
    them, and the transport stays clean.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            log = open(_LOG_PATH, "a", buffering=1)
        except Exception:                       # noqa: BLE001 - never fail a tool
            log = io.StringIO()                 # discard rather than corrupt
        try:
            with contextlib.redirect_stdout(log):
                return fn(*args, **kwargs)
        finally:
            log.close()
    return wrapper

SUBMIT_SCRIPT = FRAMEWORK / "tools" / "submit_cases.sh"
STUDY_SCRIPT  = FRAMEWORK / "tools" / "run_study.sh"
BUILD_JOB     = Path(__file__).resolve().parent / "scripts" / "ensemble_job.py"

# What the framework writes, and what this server reads.
CASE_INPUTS  = "case_inputs.json"
BUILT_CASES  = "built_cases.json"
# The MCP's own column metadata, and the FINAL one. The caller's columns.json is
# a temporary input: the warm start MOVES every column (snapping it to its CONUS
# donor gridcell and adopting that cell's soil), so the sampled coordinates
# describe a run that will not happen. This used to be written back into the
# caller's file, which gave one path two producers and no owner.
COLUMN_META  = "elm_columns.json"


# ─────────────────────────────────────────────────────────────────────
# WHAT IS ACTUALLY HERE
# ─────────────────────────────────────────────────────────────────────
def _requirements() -> dict:
    """Every external thing this server needs, and whether it is present.

    CHECKED, not asserted. A capabilities tool that lists what it assumes is
    worth nothing on the machine where one of those assumptions is false —
    which is exactly the case it exists to diagnose, since a wrong path here
    surfaces eight minutes into a CIME build as an unrelated-looking error.

    IT INCLUDES THE BULK DATA NOW. This used to say "generating inputs is not
    this server's job: no CONUS surfdata, no ELM input-file templates" — true
    until build_elm_inputs_from_location landed. The warm start is what makes a
    column credible, so a missing restart is not a degraded run, it is a
    different experiment; it belongs where a missing E3SM tree already is.

    The manifest EXISTING is not the restarts existing. paths.describe() checks
    that at least one band resolves, because the file that lists them is a few
    kilobytes and the data it points at is ~43 GB in somebody else's scratch.
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
    # Same shape as the rest, so `ready` and `missing_requirements` need no
    # special case — plus the extra fields describe() adds (source, owner,
    # bands_resolving, warnings), which callers may ignore.
    reqs.update(paths.describe_all())
    return reqs


def _imports() -> dict:
    """Whether the framework modules this server fronts can be imported. They
    pull in CIME's own tree, which can be missing in a differently-launched
    process."""
    out = {}
    for mod in ("elm_experiment_builder", "elm_input_agent"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception as e:                                  # noqa: BLE001
            out[mod] = f"{type(e).__name__}: {e}"
    return out


@mcp.tool()
@_stdout_to_stderr
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
            "Anything that requires knowing what ELM is lives here. The caller "
            "decides WHERE the columns go; this server decides what ELM sees "
            "at those locations, compiles, and runs."
        ),
        "contract": (
            "Every tool returns quickly or returns a job id. No tool blocks on "
            "the science: an MCP client opens a fresh session per call and "
            "tearing it down kills this server's children, so a 10-minute CIME "
            "build under a 300 s timeout is not a slow call, it is a half-built "
            "case directory."
        ),

        # EXHAUSTIVE, and a test enforces that against the tool registry. A
        # partial list here is worse than none: this is the tool an agent calls
        # FIRST, so anything missing from it effectively does not exist, and
        # the omission looks like an absent capability rather than a stale doc.
        "workflow": [
            {"step": 1, "tool": "describe_elm_capabilities",
             "does": "this — the workflow, a checked inventory of every "
                     "external dependency, and what this server does NOT do",
             "returns": "DATA"},
            {"step": 2, "tool": "build_elm_inputs_from_location",
             "does": "columns (passed as data, or read from "
                     "01_inputs/columns.json) — warm start, donor soil, "
                     f"surfaces, domains, and 01_inputs/{CASE_INPUTS}",
             "returns": "DATA; the SNAPPED columns, which are not the ones "
                        "you passed in"},
            {"step": 3, "tool": "get_column_metadata",
             "does": f"the columns as they will be RUN, from 01_inputs/"
                     f"{COLUMN_META} — post warm start, with donor soil",
             "returns": "DATA; ask here rather than reading the columns.json "
                        "you sampled, which is what you ASKED FOR"},
            {"step": 4, "tool": "build_elm_cases",
             "does": f"CIME cases from 01_inputs/{CASE_INPUTS} — reference "
                     f"case compiled, the rest cloned with --keepexe",
             "returns": "a JOB ID; the case directories come back from "
                        "check_elm_job once it lands"},
            {"step": 5, "tool": "submit_elm_ensemble",
             "does": "run every column concurrently as one batch job",
             "returns": "a JOB ID"},
            {"step": 6, "tool": "run_elm_study",
             "does": "steps 4 and 5 and the analysis as ONE job, so a study "
                     "finishes unattended instead of costing three "
                     "invocations each waiting on a queue",
             "returns": "a JOB ID"},
            {"step": 7, "tool": "check_elm_job",
             "does": "ask SLURM what a job from step 4, 5 or 6 is doing",
             "returns": "state, whether it is still active, and for a build "
                        "job the case directories it produced"},
        ],

        "inputs_expected": {
            "file": f"<run_dir>/01_inputs/{CASE_INPUTS}",
            "shape": "[{case_name, runtime_config: {FSURDAT, FINIDAT, "
                     "LND_DOMAIN_FILE, LND_DOMAIN_PATH, ATM_DOMAIN_FILE, "
                     "ATM_DOMAIN_PATH, STOP_N, STOP_OPTION, RUN_STARTDATE, "
                     "DATM_CLMNCEP_YR_START, DATM_CLMNCEP_YR_END, REST_N, "
                     "REST_OPTION}}, ...]",
            "note": "absolute paths to files that already exist. "
                    "build_elm_inputs_from_location WRITES this file — the "
                    "claim that this server does not generate inputs was true "
                    "under the superseded rule and is not true now. A caller "
                    "holding one already can go straight to build_elm_cases.",
        },

        "does_not": [
            "choose where to put columns — the sampling design belongs to the "
            "caller, and is shared with PFLOTRAN so the two are comparable",
            "read history files or compute any result",
            "decide whether a study is worth running, record caveats, or write "
            "experiment.json",
        ],

        "requirements": reqs,
        "imports": imports,

        "data_paths_note": (
            "The CONUS entries in `requirements` are bulk data, not code. Each "
            "is overridable three ways, in precedence order: a tool argument, "
            "its environment variable, or a paths.json beside this server. "
            "Prefer paths.json when a standard MCP client launches the server: "
            "it forwards only HOME, LOGNAME, PATH, SHELL and USER, so an "
            "exported variable reaches this process only under a client that "
            "copies the whole environment."
        ),

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
# LOCATIONS → RUNNABLE INPUTS  (not a job — fast, local, returns DATA)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
def build_elm_inputs_from_location(run_dir:         str,
                                   soil_config:     str = "native",
                                   substrate:       str = "extrapolate",
                                   yr_start:        int = 0,
                                   yr_end:          int = 0,
                                   conus_restart:   str = "",
                                   columns:         Optional[List[Dict]] = None,
                                   ) -> str:
    """Turn sampled column locations into everything ELM needs to run them.

    Reads <run_dir>/01_inputs/columns.json — the locations the caller sampled —
    and returns the SNAPPED columns plus a written case_inputs.json.

    `columns` PASSES THEM AS DATA INSTEAD, and is preferred by any caller that
    already has them in hand. The file form exists for an agent driving this
    server from a run directory; it is not the better interface. A caller whose
    warm start must happen before it persists anything cannot use the file form
    at all — the framework's own materialize step is exactly that case, since
    the snap moves the columns and so columns.json cannot be written until
    after this call. Passing them as data removes the ordering problem and the
    temporary file with it.

    Not a job: no scheduler, no compile, seconds to minutes. It is the counterpart
    of run_elm_ensemble, which is a job and returns an id instead of data.

    WHAT IT DOES, and the order matters:

      warm start          a finidat per column, subset from the CONUS restarts,
                          SNAPPING each column to its donor gridcell (~250-400 m)
      donor soil          that gridcell's own soil — the only soil the run has
      surfaces + domains  domain.nc and surface.nc per column
      runtime_config      the 13 CIME keys naming every file the build needs
      case_inputs.json    written to <run_dir>/01_inputs/

    THE RETURNED COLUMNS ARE NOT THE ONES YOU PASSED IN. The warm start moves
    them, so persist what comes back: anything written from the pre-snap
    coordinates describes a run that will not happen.

    conus_restart overrides which restart source is used, for this call only.
    Whatever is used is reported in data_provenance and written into the run,
    because which restart a column warm-started from is a fact about the science
    rather than a configuration detail.

    FATAL if any column cannot be warm-started. A partial warm start puts columns
    with different initial states and different soil datasets in one ensemble,
    and every cross-column comparison then spans two experiments.
    """
    rd = Path(run_dir).resolve()
    doc, src = {}, None
    if columns is None:
        src = rd / "01_inputs" / "columns.json"
        if not src.is_file():
            # The legacy location, which older runs wrote before 01_inputs existed.
            src = rd / "columns.json"
        if not src.is_file():
            return json.dumps({
                "ok": False,
                "error": f"no columns.json in {rd / '01_inputs'} or {rd} and no "
                         f"`columns` passed — this server does not sample; the "
                         f"caller either writes the columns or hands them over"})
        try:
            doc = json.loads(src.read_text())
        except Exception as e:                                  # noqa: BLE001
            return json.dumps({"ok": False,
                               "error": f"unreadable columns.json: {e}"})
        columns = doc.get("columns") if isinstance(doc, dict) else doc

    if not columns:
        return json.dumps({"ok": False,
                           "error": f"no columns to build from"
                                    + (f" — {src} carries none" if src else "")})

    cfg = {"soil_config": soil_config, "substrate": substrate}
    if yr_start:
        cfg["yr_start"] = yr_start
        cfg["yr_end"] = yr_end or yr_start
    if conus_restart:
        cfg["warm_start"] = {"conus_restart": conus_restart}

    try:
        import inputs
        out = inputs.build_from_location(rd, columns, cfg)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False,
                           "error": f"{type(e).__name__}: {e}"})

    # The snapped columns are OURS and are written where we own them, keeping
    # the caller's design metadata (bands, priors, the sampling grid) alongside
    # so the file is self-contained. The caller's columns.json is left exactly
    # as it was written — it is an input, and inputs are not edited in place.
    meta = dict(doc) if isinstance(doc, dict) else {}
    meta["columns"] = out["columns"]
    meta["source_columns"] = str(src) if src else "passed as data"
    meta_path = rd / "01_inputs" / COLUMN_META
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    out["column_metadata_path"] = str(meta_path)

    out["next"] = "run_elm_ensemble"
    return json.dumps(out, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────
# THE COLUMNS AS THEY WILL BE RUN  (data)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
def get_column_metadata(run_dir: str) -> str:
    """The FINAL columns — post warm start, with their donor soil.

    Ask here rather than reading the columns.json you sampled. That file is
    what you ASKED FOR; this is what will run. The warm start snaps every
    column to its CONUS donor gridcell and adopts that cell's TOPO and soil
    profile, so the two disagree by design — by up to MAX_SNAP_KM, and in
    elevation by whatever the donor's TOPO differs from the sampled 3DEP value.

    Anything that describes the ensemble — a design figure, a table of what was
    run, an area weighting — wants these. Reading the sampled file instead
    produces a picture of a run that did not happen, and nothing about it looks
    wrong.
    """
    rd = Path(run_dir).resolve()
    p = rd / "01_inputs" / COLUMN_META
    if not p.is_file():
        return json.dumps({
            "ok": False,
            "error": f"no {COLUMN_META} in {rd / '01_inputs'} — "
                     f"build_elm_inputs_from_location has not run for this run "
                     f"directory, so no column has a donor yet"})
    try:
        doc = json.loads(p.read_text())
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False,
                           "error": f"unreadable {COLUMN_META}: {e}"})
    cols = doc.get("columns") or []
    return json.dumps({
        "ok": True,
        "n_columns": len(cols),
        "path": str(p),
        "columns": cols,
        # The design metadata the caller sampled with, carried through so one
        # fetch answers "what ran" and "what was it meant to represent".
        "bands": doc.get("bands"),
        "sampling_design": doc.get("sampling_design"),
        "source_columns": doc.get("source_columns"),
    }, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────
# BUILD THE CASES  (a job — D1)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
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
# Written out rather than inherited: sbatch propagates this server's PATH
# today, but a site that defaults to --export=NONE, or anyone re-running this
# script by hand, would otherwise get the python3 that create_newcase cannot
# load (see the PATH note at the top of this server).
export PATH={Path(sys.executable).parent}:$PATH
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
        "stage":    "build_cases",
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
@_stdout_to_stderr
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
# THE WHOLE STUDY  (one job)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
def run_elm_study(run_dir:  str,
                  queue:    str = "",
                  walltime: str = "02:00:00",
                  email:    str = "") -> str:
    """Build the cases, run every column, and analyze — as ONE job.

    Use this instead of build_elm_cases + submit_elm_ensemble when you want
    the study to finish unattended. Those two are still the right tools when
    you want to inspect the cases between building and running, or to run an
    ensemble whose cases already exist.

    The difference is what the caller has to do. Split, a study costs three
    separate invocations, each one waiting on a queue: submit the build, come
    back and submit the run, come back and analyze. This costs one — the job
    builds, runs, and then invokes the framework's own tail on the same
    allocation, so by the time Slurm sends its END mail the analysis is
    already written to 04_analysis/.

    Reads <run_dir>/01_inputs/case_inputs.json, like build_elm_cases; this
    server still generates no inputs.

    email: an address for Slurm's END,FAIL notification. Without it the job
    runs identically and nobody is told when it lands.

    Walltime must cover build + run + analysis, not just the run: ~8-10 min
    for a cold compile, then the columns, then ~1-2 min of Analyzer. The
    default two hours is the `short` partition's limit and fits a 19-column
    study (measured: 2406 s of run time). Pass -q slurm and a longer walltime
    for anything bigger.
    """
    rd = Path(run_dir).resolve()
    src = rd / "01_inputs" / CASE_INPUTS
    if not src.is_file():
        return json.dumps({
            "error": f"no {CASE_INPUTS} in {rd / '01_inputs'} — this server "
                     f"does not generate inputs; the caller writes them"})
    if not STUDY_SCRIPT.is_file():
        return json.dumps({"error": f"missing {STUDY_SCRIPT}"})
    try:
        n_cases = len(json.loads(src.read_text()))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"unreadable {CASE_INPUTS}: {e}"})
    if not n_cases:
        return json.dumps({"error": f"{CASE_INPUTS} is empty"})

    cmd = ["bash", str(STUDY_SCRIPT), str(rd),
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
        return json.dumps({"error": "submitted but no job id in the output",
                           "output": out[-2000:]})

    return json.dumps({
        "job_id":   m.group(1),
        "n_cases":  n_cases,
        "stage":    "study",
        "queue":    queue or os.environ["IDEAS_SLURM_QUEUE"],
        "walltime": walltime,
        "email":    email or None,
        "log_path": str(rd / "study.log"),
        "produces": "04_analysis/ — the analysis is written INSIDE this job",
        "next":     "check_elm_job (optional — the email is the signal)",
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────
# WHAT IS THE SCHEDULER DOING
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
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
            out["stage"] = "build_cases"
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
