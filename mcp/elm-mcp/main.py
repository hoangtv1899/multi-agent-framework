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

THE TOOL LIST IS NOT WRITTEN DOWN HERE, deliberately. It used to be, and it
said "all seven" while the registry held eight — extract_elm_output was
registered, undescribed, and therefore invisible to the planner that reads
describe_elm_capabilities() to find out what exists. A capability the caller
cannot see is the same as one that does not exist.

So the authority is the live registry (_registered_tools), the ORDER and the
reasons are in _workflow(), and _coverage() compares them on every call and
reports both failure directions: registered_but_not_described (a real
capability reading as absent) and described_but_not_registered (the caller sent
at a name that will fail). Ask the server:

    describe_elm_capabilities() -> tools_registered, workflow, tool_coverage

TWO WENT AWAY on 2026-08-10, both because they shelled out to scripts in the
FRAMEWORK — a server running its client's code, which is the dependency the
boundary rule exists to forbid:

    run_elm_study        ran tools/run_study.sh, whose job ended with
                         `workflow.py --finalize`. run_elm_ensemble + job B is
                         the same study with the callback inverted.
    submit_elm_ensemble  ran tools/submit_cases.sh to run already-built cases.
                         run_elm_ensemble does exactly that now: job A verifies
                         built_cases.json against the case directories on disk
                         and skips the compile when they are still there.
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
import time
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

# EVERY SCRIPT THIS SERVER RUNS LIVES BESIDE IT. There were two more here —
# FRAMEWORK/tools/submit_cases.sh and FRAMEWORK/tools/run_study.sh — and they
# were the server reaching back across the boundary to shell out to the client.
# run_study.sh was the worse of the two: its job ended by running
# `workflow.py --finalize`, so the server's job executed the framework's code.
# Jobs A and B invert that (2026-08-10) and both scripts are gone from this side.
BUILD_JOB     = Path(__file__).resolve().parent / "scripts" / "ensemble_job.py"
ENSEMBLE_AB   = Path(__file__).resolve().parent / "scripts" / "ensemble_ab.sh"

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
        "build_job":     _file(BUILD_JOB),
        "ensemble_ab":   _file(ENSEMBLE_AB),
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


def _registered_tools() -> List[str]:
    """The tool names this server actually exposes, from the live registry.

    The authority on what exists is the registry, never a list in a docstring.
    """
    try:
        return sorted(t.name for t in mcp._tool_manager.list_tools())
    except Exception:                                           # noqa: BLE001
        return []


def _referenced_names() -> set:
    """Every global name the CODE in this module actually references.

    From code objects, not from the file's text, and the distinction is the
    whole point. A text scan matches the claim strings themselves — the first
    version of this check read the sentence "does not ... write
    experiment.json", found "experiment.json" in the file, and declared its own
    claim stale. co_names holds the names the bytecode reaches for; docstrings
    and comments are not in it, and string constants are co_consts.
    """
    seen: set = set()

    def walk(code) -> None:
        seen.update(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                walk(const)

    for obj in list(globals().values()):
        code = getattr(obj, "__code__", None)
        if code is not None:
            try:
                walk(code)
            except Exception:                                   # noqa: BLE001
                continue
    return seen


def _does_not(names: set) -> List[Dict[str, Any]]:
    """What this server does not do — each claim WITH THE CHECK THAT WOULD FALSIFY IT.

    A hand-written list of negative claims is the most rot-prone thing in a
    capability report, because nothing fails when one goes stale. This list
    claimed the server "does not read history files or compute any result" for
    as long as compare_to_obs has existed — false the whole time, and found
    only on 2026-08-11 by someone reading it.

    So each claim now names the symbols whose PRESENCE IN THE CODE would
    contradict it, and one that has been contradicted reports itself as STALE
    rather than lying quietly. The checks are deliberately coarse: a name is
    referenced or it is not. A check subtle enough to be wrong is a second
    thing to maintain, and this one only has to be harder to ignore than a
    sentence nobody re-reads.

    A claim with no checkable symbol says so — `checked: false` — rather than
    borrowing the credibility of the ones that are checked.
    """
    claims = [
        {"claim": "choose where to put columns — the sampling design belongs "
                  "to the caller, and is shared with PFLOTRAN so the two are "
                  "comparable",
         "contradicted_by": ("expand_sampling", "sample_columns",
                             "choose_columns", "place_columns")},
        {"claim": "GATHER observations. It reads the ones reception already "
                  "fetched and persisted in reception.json; refreshing them "
                  "means re-running reception, the one component that reaches "
                  "outside",
         "contradicted_by": ("requests", "urlopen", "urllib", "httpx",
                             "MCPManager", "MCPClient")},
        {"claim": "decide whether a study is worth running, or write "
                  "experiment.json and its caveats",
         "contradicted_by": ("Analyzer", "limitations", "write_experiment")},
        {"claim": "offer a verdict — compare_to_obs returns measurements, and "
                  "what they mean belongs to whoever reads them",
         "contradicted_by": ("verdict", "grade", "skill_score")},
    ]
    out = []
    for c in claims:
        hits = sorted(n for n in c["contradicted_by"] if n in names)
        rec: Dict[str, Any] = {
            "claim": c["claim"],
            "checked": True,
            "would_contradict": list(c["contradicted_by"])}
        if hits:
            rec["STALE"] = (f"no longer true — this server's code references "
                            f"{', '.join(hits)}")
        out.append(rec)
    return out


def _constraints(reqs: Dict[str, Any]) -> Dict[str, Any]:
    """What a study using this model may NOT vary, and why — the design fence.

    THE SIBLING OF _pinning, AND A DIFFERENT QUESTION. `pinning` says what a
    finished column may be COMPARED against. This says what a plan may CHOOSE
    before any column exists. Both were prose in planner.txt, stated as facts
    about "the framework" — "FORCING is NLDAS-2 only", "SOIL comes from the
    CONUS 1 km surface dataset", "WARM START is the default". Every one of
    those is a fact about ELM. A second model reading them is told its soil is
    unchoosable while its profiles sit in the reception package.

    WHY THE PLANNER NEEDS IT AT ALL. It is not decoration and it does not move
    a single column. It is what makes "how does soil texture control recharge
    here?" come back `partial` with `requires: [soil control]` instead of a
    confident design that cannot answer the question. Delete the fence and
    every study is judged feasible; freeze the wrong fence into the prompt and
    a study is downgraded for a limit it does not have.

    MEASURED WHERE IT CAN BE. `fixed` is the model — the compset is compiled
    that way and no path check changes it. Soil and the warm start are
    CONDITIONAL: both come from bulk data in somebody else's scratch, and when
    that data is absent the fence moves rather than vanishing — a study with no
    restart manifest must budget its own spin-up, which is a different design,
    not a degraded one. `reqs` is the same inventory `ready` is derived from,
    passed in so the two cannot disagree.

    THE FORCING YEARS ARE SCANNED, NOT DECLARED (2026-08-17). The dataset and
    resolution are compiled in (DATM_MODE=CLMMOSARTTEST); the runnable YEARS
    are whatever sits in the DATM directory, so forcing.scan_window() counts
    the months present. It used to be counted framework-side by
    core/forcing_availability.py, which meant the framework opened ELM's input
    tree and pasted an ELM answer into every request — a PFLOTRAN study of 2024
    was refused for a gap in ELM's forcing. That module is gone.

    THE SPAN IS THE LONGEST UNBROKEN RUN, not first..last. A gap year mid-record
    aborts the simulation partway rather than at submit time, so a complete year
    outside the contiguous span is reported in `years_excluded` and NOT offered.
    """
    def _has(name: str) -> bool:
        return bool((reqs.get(name) or {}).get("present"))

    restart_ok = _has("conus_restart_manifest")
    bands = (reqs.get("conus_restart_manifest") or {}).get("bands_resolving")
    surf_ok = _has("conus_surfdata")
    try:
        import forcing as _forcing                 # src/ is on sys.path
        w = _forcing.scan_window()
    except Exception as e:                         # noqa: BLE001 — reported
        w = {"exists": False, "error": f"{type(e).__name__}: {e}"[:120]}
    span = ([w["yr_first"], w["yr_last"]]
            if w.get("exists") and w.get("yr_first") else None)

    return {
        "model": "E3SM Land Model (ELM), 1-D columns",
        # NOT CONDITIONAL ON ANYTHING. The compset is fixed in code, so no
        # path check and no argument makes these available.
        "fixed": [
            {"what": "lateral flow between columns",
             "why": "the columns are independent 1-D land units. Nothing "
                    "moves water sideways from one to another",
             "consequence": "a stream gauge integrates and ROUTES a catchment; "
                            "a column produces a point flux. Comparing them is "
                            "a first-order water-balance check, never a "
                            "hydrograph match"},
            {"what": "river routing",
             "why": "DATM_MODE=CLMMOSARTTEST is fixed in code and the compset "
                    "is compiled without it",
             "consequence": "it cannot be added, so do not propose it as a "
                            "next step"},
        ],
        "forcing": {
            "chosen_by": "the model, not the plan",
            "dataset": "NLDAS-2",
            "resolution_m": 12000,
            "path": w.get("path"),
            # THE ONE HARD CONSTRAINT ON A REQUEST. Without forcing there is no
            # run at all; every other data gap only costs a comparison
            # afterwards. Counted on disk at the moment of asking.
            "years": span,
            "n_years": w.get("n_years"),
            "years_partial": w.get("partial_years") or {},
            "years_excluded": w.get("excluded") or [],
            "available": bool(span),
            "consequence": ("columns at different elevations inside ONE 12 km "
                            "cell share their weather exactly. An elevation "
                            "gradient can come out flat for that reason alone, "
                            "with nothing downstream saying so — spread bands "
                            "wider than the cell where the question turns on "
                            "climate. A period outside `years` cannot be run: "
                            "refuse it, do not shift it to a decade nobody "
                            "asked for"
                            if span else
                            "the forcing tree could not be read (" +
                            str(w.get("error") or w.get("path")) + "), so no "
                            "year can be confirmed runnable. Do NOT state a "
                            "range; ask which period is wanted and record that "
                            "forcing could not be verified"),
            "exception": ("a CONCEPTUAL sweep may PRESCRIBE weather: with "
                          "held_fixed.weather set, this server WRITES the "
                          "forcing for every column instead of reading a real "
                          "cell, so no cell's climate bounds the result and "
                          "the 12 km sharing above does not apply. The year "
                          "range still does — the written series is built from "
                          "a real one"),
        },
        "soil": {
            "chosen_by": "the model, not the plan",
            "source": "CONUS 1 km surface dataset, at each column's donor "
                      "gridcell",
            "available": surf_ok,
            "consequence": ("it varies column to column, but a site plan does "
                            "not choose it. A question that turns on soil "
                            "texture cannot be answered by a site run — say so "
                            "in `requires` rather than designing bands around "
                            "it"
                            if surf_ok else
                            "the CONUS surface dataset is NOT present on this "
                            "machine, so no column can be given a donor soil "
                            "at all. A site run cannot be built until it is"),
            "exception": ("a CONCEPTUAL sweep PRESCRIBES soil — that is what a "
                          "texture sweep is — and this server builds a "
                          "synthetic profile per level. The levels choose it, "
                          "not the planner and not the sampler"),
        },
        "initial_state": {
            "chosen_by": "the model, not the plan",
            "default": "warm start from the CONUS restart",
            "available": bool(restart_ok and bands),
            "bands_resolving": bands,
            "consequence": ("no prior run and no separate spin-up switch is "
                            "needed; a longer study is made by naming a longer "
                            "period"
                            if restart_ok and bands else
                            "the CONUS restart is NOT resolvable here, so a "
                            "run starts COLD. That is a different experiment, "
                            "not a degraded one: the study must budget its own "
                            "spin-up and say so in `requires`"),
        },
    }


def _pinning() -> Dict[str, Any]:
    """Which observed variables a column of THIS model may be pinned to.

    A pinned column exists so a simulated value and an observed one describe
    the SAME place. Whether that is possible is a fact about the model — what
    it computes, and where — so it is answered here rather than asserted by a
    prompt or by a tuple in the sampler. It was in both, in two vocabularies,
    and neither knew what ELM is.

    THAT MATTERS NOW BECAUSE THERE IS ABOUT TO BE A SECOND MODEL. "This model
    has no lateral transport" is true of a 1-D ELM column and false of a 3-D
    PFLOTRAN domain, so a rule frozen into planner.txt would either be copied
    for the second server or silently govern it.

    `reason` is required on every entry, pinnable or not, and it must say WHOSE
    fact it is. A design decision dressed as physics is the failure this
    replaces: the water table was excluded because recorder wells cluster and
    are scarce, which is a property of the observation network, and reading it
    in a list of model limits invited the conclusion that ELM cannot compute a
    water table at a point. It can.

    THE NAMES ARE THE COMPARISON'S NAMES (compare/*.py SPEC.name), because a
    pin is honoured only for the observable it names. When the sampler wrote
    "water_table" and the comparison read "wtd", pair_stations compared the two
    spellings, found them unequal, and silently refused every well pin ever
    designed.
    """
    return {
        "model": "1-D ELM column, no lateral transport",
        "vocabulary": "compare/*.py SPEC.name — the comparison honours a pin "
                      "only for the observable named here",
        "pinnable": [
            {"variable": "swe",
             "reason": "vertical and local — the column computes snow water "
                       "equivalent where it stands, and a pillow measures it "
                       "where it stands",
             "basis": "model"},
            {"variable": "et",
             "reason": "vertical and local — the column's evapotranspiration "
                       "is a flux through the surface above it",
             "basis": "model"},
        ],
        "not_pinnable": [
            {"variable": "streamflow",
             "reason": "a gauge measures discharge integrated and ROUTED over "
                       "its upstream area, and this model has no lateral "
                       "transport, so a column at the gauge produces a point "
                       "runoff flux and never the quantity the gauge "
                       "recorded. Structural: true of every gauge in every "
                       "basin, which is why it cannot be waived by calling a "
                       "column co-located",
             "basis": "model",
             "instead": "basin-aggregate comparison against the ensemble — "
                        "still validated, it simply stops costing a column"},
            {"variable": "water_table",
             # NOT A MODEL LIMIT, AND IT MUST NOT READ AS ONE. ELM computes ZWT
             # at the column. This is a standing decision about how columns are
             # spent, and it is reported here only because this report is the
             # single source the planner and the sampler both read.
             "reason": "excluded by a standing design decision (2026-08-12), "
                       "NOT by the model — an ELM column does compute a water "
                       "table where it stands",
             "basis": "design decision",
             # SAYS WHAT THE COMPARISON BECOMES, AND NOTHING ABOUT THE BUDGET.
             # An earlier wording here added "columns are spent on placement at
             # documented water tables instead". True, but planner.txt already
             # owns that arithmetic — `n_columns = n_bands*per_band +
             # n_validation + 2`, where the 2 is the sampler's water-table
             # anchors — and a capability report that restates a budget rule is
             # a second place for it to drift.
             "instead": "distance-matched — recorder wells are still fetched "
                        "and still compared; they simply do not cost a column"},
        ],
    }


def _workflow() -> List[Dict[str, Any]]:
    """The tools in the order a study uses them, with why each exists.

    Hand-written because it carries what the registry cannot: the ORDER,
    and the reason a step is there. _coverage() checks it against the
    registry so the prose cannot quietly drift out of step with the code.

    ONLY `step` AND `tool` LEAVE THIS SERVER (2026-08-17). The `does` and
    `returns` sentences below are for whoever reads this file; the report
    carries the order and the names, which is what _coverage() and the tests
    check. Keeping the prose here rather than deleting it costs nothing on the
    wire and keeps the reason a step exists next to the step.
    """
    return [
            {"step": 1, "tool": "describe_elm_capabilities",
             "does": "this — the workflow, a checked inventory of every "
                     "external dependency, and what this server does NOT do",
             "returns": "DATA"},
            # ── the conceptual archetype only ────────────────────────────
            # A site study skips these three entirely: its columns come from
            # the framework's spatial sampler. A controlled sweep has no basin
            # to sample, so the columns are DESIGNED, and designing them needs
            # to know what ELM can be told to vary — which is here.
            {"step": "1a (sweeps only)", "tool": "describe_conceptual_factors",
             "does": "the menu for a CONTROLLED SWEEP — what ELM can vary, "
                     "the ranges that make sense, the fixed soil grid a study "
                     "does not get to choose, and what cannot be varied at all",
             "returns": "DATA"},
            {"step": "1b (sweeps only)", "tool": "check_conceptual_design",
             "does": "judge a proposed sweep BEFORE compute — what will not "
                     "build, what will build but is far from anything "
                     "measured, and facts worth knowing (two levels in one "
                     "NLDAS cell share their weather exactly, so the sweep "
                     "would be flat and nothing downstream would say so)",
             "returns": "DATA; three lists and no severities — this server "
                        "reports what is true, it does not grade"},
            {"step": "1c (sweeps only)", "tool": "build_conceptual_columns",
             "does": "the checked design becomes the same columns.json a "
                     "sampled run produces — the point where the two "
                     "archetypes merge and everything below is identical",
             "returns": "DATA; raises rather than building a partial sweep"},
            {"step": "1d (coupled follow-ups only)",
             "tool": "set_initial_water_table",
             "does": "write a water table into a single-column finidat — ZWT "
                     "and WA kept consistent by ELM's own relation, clamps at "
                     "the 28.802 m aquifer bottom reported. The framework "
                     "drives this THROUGH step 2: a column carrying "
                     "initial_water_table_m is stamped right after its CONUS "
                     "subset; this tool is the same edit for a file in hand",
             "returns": "DATA: requested vs written, regime, clamped, note"},
            {"step": 2, "tool": "build_elm_inputs_from_location",
             "does": "columns (passed as data, or read from "
                     "01_inputs/columns.json) — warm start, donor soil, "
                     f"surfaces, domains, and 01_inputs/{CASE_INPUTS}. A "
                     f"column carrying a PRESCRIBED soil keeps it: the donor "
                     f"profile is not applied to a controlled sweep, or every "
                     f"column at one location would get the same soil",
             "returns": "DATA; the SNAPPED columns, which are not the ones "
                        "you passed in"},
            {"step": 3, "tool": "get_column_metadata",
             "does": f"the columns as they will be RUN, from 01_inputs/"
                     f"{COLUMN_META} — post warm start, with donor soil",
             "returns": "DATA; ask here rather than reading the columns.json "
                        "you sampled, which is what you ASKED FOR"},
            {"step": 4, "tool": "run_elm_ensemble",
             "does": "JOB A — build the cases from 01_inputs/"
                     f"{CASE_INPUTS} (reference case compiled, the rest cloned "
                     "with --keepexe) and run every column concurrently, then "
                     "stop. Submit job B yourself with "
                     "--dependency=afterany:<id> for the analysis, never "
                     "afterok (a failed ensemble would then never report). "
                     "Re-running on a directory whose cases still exist skips "
                     "the compile",
             "returns": "a JOB ID"},
            {"step": "4 (alt)", "tool": "build_elm_cases",
             "does": "the BUILD ALONE, when you want to inspect the cases "
                     "before spending node time on them. run_elm_ensemble is "
                     "the normal path and does this too",
             "returns": "a JOB ID; the case directories come back from "
                        "check_elm_job once it lands"},
            {"step": 5, "tool": "check_elm_job",
             "does": "ask SLURM what a job from step 4 is doing",
             "returns": "state, whether it is still active, and for a build "
                        "job the case directories it produced"},
            {"step": 6, "tool": "extract_elm_output",
             "does": "read each column's *.elm.h0.*.nc into daily series on "
                     "disk. IT IS ALSO WHAT ESTABLISHES WHAT RAN — a "
                     "submission receipt is written before the model starts, "
                     "and RUN_SUMMARY.json has said 'pending, 0 success' for a "
                     "run where every column finished. The history files are "
                     "the ground truth. RAW SERIES ONLY: daily means, fluxes "
                     "in mm/day, no fractions, ratios or budgets — those carry "
                     "semantics that stay framework-side",
             "returns": "a SUMMARY and the path to 03_results/extracted.json; "
                        "the series never travel inline"},
            {"step": 7, "tool": "compare_to_obs",
             "does": "pair each column's daily series against the observations "
                     "reception already gathered and persisted in "
                     "reception.json — swe, water_table, streamflow, et — and report "
                     "metrics plus how much of the observation was measured "
                     "rather than gap-filled. Reuses the extraction when it is "
                     "already on disk; re-run RECEPTION to refresh the "
                     "observations",
             "returns": "MEASUREMENTS, never a verdict; a summary inline and "
                        "the full record in 04_analysis/comparison.json"},
    ]

def _coverage(registered: List[str], wf: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Does the prose above still describe the tools that actually exist?

    Two failures, opposite and both silent. A tool registered with no workflow
    entry is INVISIBLE to the agent that calls this first, so a real capability
    reads as an absent one. An entry naming a tool that is no longer registered
    sends that agent at a name that will fail. Neither breaks anything at
    import, which is exactly why it has to be reported at call time.
    """
    described = {e["tool"] for e in wf if e.get("tool")}
    undescribed = sorted(set(registered) - described)
    phantom = sorted(described - set(registered))
    out: Dict[str, Any] = {
        "n_registered": len(registered), "n_described": len(described),
        "complete": not undescribed and not phantom}
    if undescribed:
        out["registered_but_not_described"] = undescribed
    if phantom:
        out["described_but_not_registered"] = phantom
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
    names = _referenced_names()
    registered = _registered_tools()
    wf = _workflow()

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
        #
        # ORDER AND NAMES ONLY, since 2026-08-17. The `does` and `returns`
        # sentences stay in _workflow() where a person reading this file can
        # see why each step is there; they no longer travel on the wire. They
        # were 3,112 of this report's 11,003 characters — 28% — and NOTHING
        # reads them: the framework takes five keys (model, ready,
        # missing_requirements, broken_imports, pinning) and discards the rest,
        # and the only readers of `workflow` are this server's own tests, which
        # read `step["tool"]`. The cost was real for a reader that has to fit
        # inside a 12,000-character tool result.
        "workflow": [{"step": s["step"], "tool": s["tool"]} for s in wf],

        # WHAT A COLUMN OF THIS MODEL MAY BE COMPARED AGAINST AT A POINT. The
        # planner reads it to decide which stations to name, and the sampler
        # reads the same answer to enforce it — one source, so the two cannot
        # drift into disagreeing about what a pin means.
        "pinning": _pinning(),

        # THE FENCE AROUND A DESIGN — what a study using this model may NOT
        # vary, and why. Its sibling `pinning` says what a column may be
        # COMPARED against; this says what a plan may CHOOSE. Added 2026-08-17.
        "constraints": _constraints(reqs),

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

        "does_not": _does_not(names),

        # THE REGISTRY IS THE AUTHORITY, this report is a description of it.
        # The workflow above is prose — it carries the ORDER and the reason,
        # which no registry knows — so it is written by hand and then checked
        # against what is actually exposed. A tool added without a workflow
        # entry is invisible to the agent that calls this first, and an entry
        # for a tool that no longer exists sends it at a name that will fail.
        "tools_registered": registered,
        "tool_coverage": _coverage(registered, wf),

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
                                   warm_start:      bool = True,
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

    # THE FLAG HAS TO CROSS THE WIRE, and until 2026-08-15 it could not. Only
    # `conus_restart` was forwarded — a STRING — so the far side's
    # config.get("warm_start", True) always saw the default. A conceptual run
    # that set warm_start=False got a warm build anyway, and because
    # conceptual.py had already stamped warm_start=False on every column, the
    # record said cold while the cases carried a FINIDAT each. That is worse
    # than either alone: a wrong run is recoverable, a wrong run that describes
    # itself correctly is not.
    # THE COLUMNS MAY ALREADY SAY. `warm_start` is a tool argument, so a caller
    # driving this server FROM A RUN DIRECTORY — passing no columns and letting
    # the file be read — cannot express "cold" through the file at all: the
    # argument defaults to True and the columns are never consulted. Every
    # conceptual column carries `warm_start: False`, so the file already knows,
    # and ignoring it would rebuild exactly the divergence fixed above with the
    # file path instead of the wire.
    #
    # ONLY UNANIMOUS FALSE COUNTS. A mixed list is the case warm_start() calls
    # fatal — some columns warm on donor soil, others cold on another dataset,
    # inside one ensemble — so silence from any column leaves the argument in
    # charge rather than guessing for it.
    stated = [c.get("warm_start") for c in columns if isinstance(c, dict)]
    columns_say_cold = bool(stated) and all(s is False for s in stated)
    if not warm_start or columns_say_cold:
        cfg["warm_start"] = False
        if warm_start and columns_say_cold:
            print(f"cold start: all {len(stated)} column(s) carry "
                  f"warm_start=False, so the restart is not used even though "
                  f"the caller did not say so", flush=True)
    elif conus_restart:
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
# CONTROLLED SWEEPS  (the conceptual archetype)
@mcp.tool()
@_stdout_to_stderr
def set_initial_water_table(finidat: str, water_table_m: float) -> str:
    """Write an initial water table into a single-column ELM finidat.

    THE EDIT TWO-WAY COUPLING NEEDED — the one `cannot_vary` used to name as
    missing. ZWT and WA are written mutually consistent by ELM's own relation
    (zwt = 28.802 - wa/200, recovered from the CONUS restart itself); a target
    below the soil column (> 3.802 m) is EXACT, one inside it is approximate
    (ELM re-diagnoses in-soil tables from the moisture profile, which is left
    untouched) and says so; a target deeper than 28.802 m is clamped there and
    the clamp reported. Edits the file IN PLACE; only ZWT and WA change, only
    in the unmasked entries.

    Driven by the framework this happens inside build_elm_inputs_from_location:
    a column carrying `initial_water_table_m` gets its fresh finidat stamped
    right after the CONUS subset. This tool is the same edit for a file you
    already have.

    Args:
        finidat: the single-column restart to edit (make_finidat_subset output).
        water_table_m: depth below ground, metres, positive down.

    Returns:
        JSON: finidat, requested_m, written_m, old_zwt_m, new_wa_mm, regime
        ("aquifer" exact / "in-soil" approximate), clamped, note.
    """
    import set_water_table
    try:
        return json.dumps(set_water_table.apply(finidat, water_table_m,
                                                quiet=True), indent=2)
    except Exception as e:                                  # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "validation_status": "failed"})


# ─────────────────────────────────────────────────────────────────────
# Three tools in the order a study uses them: ask what can be varied, check the
# design that comes back, then turn it into columns. The check and the builder
# share one validation, so a design that clears the check cannot fail the build
# for a reason the check knew about.
@mcp.tool()
@_stdout_to_stderr
def describe_conceptual_factors() -> str:
    """What a controlled ELM sweep can vary, and what it cannot.

    Call this BEFORE designing a conceptual study — it is the menu, and a
    factor that is not on it cannot be built however reasonable it sounds.
    Derived from this server's own RUNTIME_KEYS rather than written out beside
    them, so it cannot offer a knob the wrapper would refuse.

    Also states ELM's fixed soil grid, because the commonest misunderstanding
    in a conceptual soil study is that a study chooses the column depth. It
    does not.
    """
    import conceptual
    return json.dumps(conceptual.declare(), indent=2)


@mcp.tool()
@_stdout_to_stderr
def check_conceptual_design(design: dict) -> str:
    """Will this sweep do what it looks like it does? Ask before spending compute.

    `design` is the sampling block of a factor sweep: a list of factors with
    their levels, and what the design holds fixed.

    Returns three separate lists rather than a verdict:

        wont_build  arithmetic or buildability failures — turn each into a
                    question for the user, never a silent correction
        unusual     it will build, but is far from anything measured. Each
                    carries a caveat id for the finished study
        facts       measurements about the design that may be fine or may be
                    the whole problem

    THIS SERVER DOES NOT GRADE. It reports what is true; how much each thing
    matters is decided where every other caveat's severity is decided.

    The check worth having is the third kind: two locations that fall in one
    NLDAS cell share their weather exactly, so the sweep is flat by
    construction — every column runs, the count is right, and the figures come
    out on top of each other with nothing downstream reporting it.
    """
    import conceptual
    return json.dumps(conceptual.check(design or {}), indent=2)


@mcp.tool()
@_stdout_to_stderr
def build_conceptual_columns(design: dict) -> str:
    """Turn a checked sweep into the same columns.json a sampled run produces.

    THE MERGE POINT between the two archetypes. Everything downstream of this
    file is identical for a sweep and a site study, because the Experiment
    Manager does not care why two columns differ.

    Raises on an unbuildable design rather than building part of it: a sweep
    missing one level is not a smaller sweep, it is a different experiment
    wearing the same name.
    """
    import conceptual
    return json.dumps(conceptual.as_columns_file(design or {}), indent=2)


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


def _shared_fs(path: Path) -> Optional[str]:
    """None if `path` is on a shared mount, else why it is not.

    A compute node cannot see this login node's /tmp. The first jobs-A-and-B
    attempt (770938/770939) put the sbatch scripts and logs under node-local
    /tmp and both died in TWO SECONDS WITH COMPLETELY EMPTY LOGS — no error, no
    message, nothing to diagnose, because the shell that would have written the
    error could not read the script it was told to run.

    That is the worst failure shape there is: instant, silent, and identical to
    a scheduler problem. So it is refused at submit time, where the message can
    name the actual cause.
    """
    try:
        fs = subprocess.check_output(
            ["df", "-P", str(path)], text=True, timeout=20).splitlines()[-1].split()[0]
    except Exception:                                           # noqa: BLE001
        return None                     # cannot tell — do not block on a guess
    # Lustre and NFS present as host:/export or 10.0.0.1@tcp:/fs; a local disk
    # presents as a /dev node.
    if fs.startswith("/dev/"):
        return (f"{path} is on {fs}, a node-local filesystem — a compute node "
                f"cannot see it. Put the run directory on shared storage "
                f"($PSCRATCH).")
    return None


# ─────────────────────────────────────────────────────────────────────
# JOB A — BUILD AND RUN, one job, nothing else
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
def run_elm_ensemble(run_dir:  str,
                     queue:    str = "",
                     walltime: str = "02:00:00",
                     account:  str = "",
                     in_allocation: bool = False) -> str:
    """Build the cases and run every column, as ONE job. Returns its id.

    `in_allocation=True` RUNS IN PLACE INSTEAD OF SUBMITTING: the same
    ensemble script, executed synchronously inside the SLURM allocation this
    server already lives in (the wrapper is srun-based, so the columns land
    on the allocation's own cores). For the coupling iteration — ELM,
    PFLOTRAN, ELM again inside ONE job, one queue wait for the whole loop —
    rather than one sbatch per leg. Refused, by name, when there is no
    allocation (no SLURM_JOB_ID): srun from a login node would hang.
    Returns the built cases instead of a job id; there is no job B — the
    caller is already running and simply continues.

    This is JOB A. It does everything that needs the compute node and stops
    there. The analysis is job B's: submit it yourself with
    `--dependency=afterany:<this id>` and exit — SLURM starts it when this ends,
    and nothing here calls back into the framework to trigger it.

    USE `afterany`, NEVER `afterok`. With afterok a failed ensemble means B never
    runs and no mail is ever sent, which is the silent failure that happened
    twice on 2026-08-06. afterany means B always runs and always reports,
    including "the ensemble failed, here is why".

    Replaces run_elm_study, which ended by running the framework's
    `workflow.py --finalize` — the server's job executing the client's code.
    """
    rd = Path(run_dir).resolve()
    src = rd / "01_inputs" / CASE_INPUTS
    if not src.is_file():
        return json.dumps({
            "error": f"no {CASE_INPUTS} in {rd / '01_inputs'} — call "
                     f"build_elm_inputs_from_location first"})
    if not ENSEMBLE_AB.is_file():
        return json.dumps({"error": f"missing {ENSEMBLE_AB}"})
    local = _shared_fs(rd)
    if local:
        return json.dumps({"error": local})
    try:
        n_cases = len(json.loads(src.read_text()))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"unreadable {CASE_INPUTS}: {e}"})
    if not n_cases:
        return json.dumps({"error": f"{CASE_INPUTS} is empty"})

    # NOT unlinked. It used to be, copied from build_elm_cases below, where the
    # file IS the job's answer channel and a stale one would be read as this
    # job's result. Here it is the opposite: ensemble_ab.sh checks it to decide
    # whether the ~7 min CIME compile can be skipped, so deleting it guaranteed
    # a full rebuild of cases that were sitting right there, every single time.
    # Job A owns that decision — it validates the case directories and the
    # executable and rebuilds if either is missing, which is a check this side
    # cannot make anyway (the paths are only known once the build has run).
    if in_allocation:
        jid = os.environ.get("SLURM_JOB_ID")
        if not jid:
            return json.dumps({
                "error": "in_allocation=True but there is no SLURM_JOB_ID in "
                         "the environment — this process is not inside an "
                         "allocation, and srun from a login node would hang. "
                         "Submit normally, or run inside salloc/sbatch."})
        log = rd / "ensemble_A.log"
        env = dict(os.environ,
                   IDEAS_FRAMEWORK_DIR=str(FRAMEWORK),
                   LC_ALL="en_US.utf8", LANG="en_US.utf8",
                   PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH','')}")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.run(
                ["bash", str(ENSEMBLE_AB), str(rd), sys.executable],
                stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        cases = []
        built = rd / "01_inputs" / "built_cases.json"
        if built.is_file():
            try:
                cases = (json.loads(built.read_text()) or {}).get("cases") or []
            except Exception:                                   # noqa: BLE001
                pass
        return json.dumps({
            "ran_in_allocation": True,
            "slurm_job_id": jid,
            "returncode": rc,
            "n_cases": n_cases,
            "n_built": sum(1 for c in cases if c.get("case_dir")),
            "cases": cases,
            "elapsed_s": round(time.time() - t0, 1),
            "log_path": str(log),
        }, indent=2)

    q = queue or os.environ["IDEAS_SLURM_QUEUE"]
    acct = account or os.environ["IDEAS_SLURM_ACCOUNT"]
    sb = rd / "ensemble_A.sbatch"
    sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_A
#SBATCH -N 1
#SBATCH -p {q}
#SBATCH -A {acct}
#SBATCH -t {walltime}
#SBATCH -o {rd}/ensemble_A.log
export IDEAS_FRAMEWORK_DIR={FRAMEWORK}
export PSCRATCH={os.environ['PSCRATCH']}
export LC_ALL=en_US.utf8
export LANG=en_US.utf8
export PATH={Path(sys.executable).parent}:$PATH
bash {ENSEMBLE_AB} {rd} {sys.executable}
""")
    try:
        jid = subprocess.check_output(
            ["sbatch", "--parsable", str(sb)], text=True, timeout=120).strip()
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"error": f"sbatch failed: {e}"})

    jid = jid.split(";")[0]
    return json.dumps({
        "job_id":   jid,
        "n_cases":  n_cases,
        "stage":    "ensemble",
        "queue":    q,
        "walltime": walltime,
        "log_path": str(rd / "ensemble_A.log"),
        "next":     f"submit job B with --dependency=afterany:{jid}, then exit",
    }, indent=2)


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

    local = _shared_fs(rd)
    if local:
        return json.dumps({"error": local})

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
# READ WHAT THE MODEL WROTE  (no job — reads files already on disk)
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
@_stdout_to_stderr
def extract_elm_output(run_dir:   str,
                       columns:   str = "",
                       variables: str = "",
                       overwrite: bool = False) -> str:
    """Every column's daily series, out of the history files and onto disk.

    RUN THIS AFTER THE ENSEMBLE FINISHES. It reads `*.elm.h0.*.nc` from each
    case directory named in built_cases.json, so it needs the model to have run;
    a column whose files are absent is reported, not skipped.

    IT IS ALSO WHAT ESTABLISHES WHAT RAN. Nothing else in a run directory
    reliably says so. `run_elm_ensemble` returns a submission receipt written
    before the model starts. RUN_SUMMARY.json still read "pending, 0 success"
    for a run where all 17 columns finished. `check_elm_job` reports an ensemble
    job as a build job whenever built_cases.json exists. The history files are
    the only ground truth, and this reads them.

    RAW SERIES ONLY. Daily means from the native 3-hourly output, fluxes
    converted to mm/day, states in their own units. No `recharge_fraction`, no
    `water_budget`, no ratios of any kind — those carry semantics that must not
    cross this boundary (decision of 2026-08-06), and they stay framework-side.

    THE START-UP TRANSIENT IS TRIMMED BEFORE ANY OF IT, and how much depends on
    how the run STARTED: a fortnight if the cases carry a restart file, a year
    if they do not, read off case_inputs.json rather than asked for. The
    returned summary names both the number and the basis, and a run too short
    for its own window is trimmed by nothing and says so — see
    extract.resolve_spinup. The series therefore does not begin on the run's
    start date, which is why `date_range` travels with every column.

    columns:   comma-separated subset, e.g. "col_07". Empty means all. With an
               artifact already present this MERGES, which is how one failed
               column is redone without re-reading the other sixteen.
    variables: comma-separated subset of the 15 extracted by default.
    overwrite: re-read even if 03_results/extracted.json is already there.
               Without it, an existing artifact is reused untouched.

    Writes 03_results/extracted.json — {metadata, data}, one date list per
    column shared by its variables, layered variables keeping their soil
    layers. Returns a SUMMARY and the path; the series never travel inline.
    """
    from extract import extract_run          # src/ is on sys.path (line 137)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    vars_ = [v.strip().upper() for v in variables.split(",") if v.strip()]
    try:
        out = extract_run(run_dir, columns=cols or None,
                          variables=vars_ or None, overwrite=bool(overwrite))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False,
                           "error": f"{type(e).__name__}: {str(e)[:300]}"},
                          indent=2)
    return json.dumps(out, indent=2)


# ─────────────────────────────────────────────────────────────────────
# MODEL vs OBSERVATIONS
# ─────────────────────────────────────────────────────────────────────
# The comparison's own file name lives in the compare package (compare.FILENAME)
# — one name, next to the code that writes it.


def _rows_from_extracted(payload: Dict[str, Any]) -> List[Dict]:
    """extracted.json -> the row shape model_series reads.

    One date list per column becomes a `daily` block per variable, which is the
    shape the comparison has always spoken. Layered variables (SOILLIQ, H2OSOI)
    are SKIPPED: their values are one row per day per soil layer, no SPEC
    compares them, and summing them as if they were scalars would be silent
    nonsense rather than an error.
    """
    cols = (payload.get("metadata") or {}).get("columns") or {}
    rows: List[Dict] = []
    for name, block in (payload.get("data") or {}).items():
        dates = block.get("dates") or []
        variables = {}
        for var, v in (block.get("variables") or {}).items():
            if "n_layers" in v:
                continue
            variables[var] = {"daily": {"dates": dates,
                                        "values": v.get("values"),
                                        "units": v.get("units")}}
        m = cols.get(name) or {}
        rows.append({"case_name": name, "variables": variables,
                     **{k: m.get(k) for k in
                        ("lat", "lon", "elevation_m", "pinned", "station_id",
                         "station_variable", "forcing_start", "forcing_end")}})
    return rows


def _extracted_rows(rd: Path) -> tuple:
    """The per-column daily series. Returns (rows, how).

    ONE EXTRACTOR, AND IT IS extract_elm_output (2026-08-11). This used to read
    hydro_summary.json or, failing that, open the NetCDF itself — a second
    implementation of extraction living inside the comparison, with its own
    variable list, its own spin-up handling and its own idea of what a column
    is. Two extractors in one tree is one too many: they drifted the same
    afternoon extract.py gained QSNOMELT and H2OSOI.

    Extraction is not silently a side effect of comparing, either. If the
    artefact is missing this CALLS the tool, and says so in `how`, so the
    caller can see that reading NetCDF happened rather than wondering why one
    comparison took six minutes and the next took twenty seconds.
    """
    from extract import EXTRACTED, RESULTS_DIR, extract_run
    art = rd / RESULTS_DIR / EXTRACTED
    how = f"reused {RESULTS_DIR}/{EXTRACTED}"
    if not art.is_file():
        res = extract_run(str(rd))
        if not res.get("ok"):
            return [], (f"nothing to compare: {res.get('error')}")
        how = (f"ran extract_elm_output — {res.get('n_ok')} of "
               f"{res.get('n_columns')} column(s)")
    try:
        payload = json.loads(art.read_text())
    except Exception as e:                                      # noqa: BLE001
        return [], f"unreadable {RESULTS_DIR}/{EXTRACTED}: {e}"
    rows = _rows_from_extracted(payload)
    return rows, f"{how} ({len(rows)} column(s))"


@mcp.tool()
@_stdout_to_stderr
def compare_to_obs(run_dir: str,
                   observations_json: str = "",
                   observables: str = "") -> str:
    """Model against observations for this study. MEASUREMENTS ONLY.

    Compares each column's daily series to the observations the caller supplies,
    for any of four observables:

        swe          H2OSNO                  vs snow pillow      mm
        water_table  ZWT                     vs well             m below surface
        streamflow   QOVER + QDRAI           vs gauge            mm/day
        et           QSOIL + QVEGE + QVEGT   vs flux tower       mm/day

    RETURNS NUMBERS, NOT VERDICTS. Per-station metrics (bias, MAE, RMSE, r,
    NSE, KGE) and diagnostics about the matching itself: how many days, over
    which window, and how much of the observation was actually measured rather
    than gap-filled. It does not say whether the model is good. That reading is
    the caller's, and it needs the diagnostics to make it.

    MATCHED FIRST, THEN COMPARED. swe, water_table and et each match ONE column
    to ONE
    station — on elevation for snow, on distance for the rest — and compare
    only those. Comparing everything against everything and choosing afterwards
    leaves every discarded comparison in the record, where the flattering one is
    always available.

    STREAMFLOW MATCHES NOTHING, on purpose. A gauge measures an AREA, not a
    point, so there is no nearest column to find; the ensemble mean over every
    column stands against each in-basin gauge, which is the basin-aggregate
    comparison the planner has always specified. Its record carries `gauges`
    rather than `pairs`.

    EVERY COMPARISON HERE IS CONTEXT, NOT A SKILL CLAIM. Decided 2026-08-10.
    It is what makes streamflow admissible: a mean over sampled columns is not
    routed discharge, so no metric between them scores the model — but the
    hydrograph shape and the timing are worth seeing. Each record carries
    `model_comparand`, `obs_quantity` and `colocated` so a reader can see what
    was put beside what.

    AN OBSERVABLE WITH NO STATION STILL REPORTS. Most basins have no flux tower
    and many have no recorder well, and what the model itself did is true
    regardless: where each column's water left (over the surface or through the
    soil), which water tables never moved, where the ET came from, and where
    the basin's water table sits against Fan and ParFlow CONUS2. Those come
    back beside the reason there was nothing to compare, never instead of it.

    observations_json: the run's reception.json, which is where the observations
        already live — reception queries the four data servers once, after the
        period is fixed, and persists the whole payload (streamflow, water
        table, SWE, ET; every station with coordinates and daily series) under
        `observations`. Defaults to <run_dir>/reception.json. TO REFRESH THE
        OBSERVATIONS, RE-RUN RECEPTION: it is the component that reaches
        outside, and routing the refresh through it is what keeps the numbers a
        run is judged against identical to the ones its brief was written from.

        Point this at another run's reception.json to compare these columns
        against that domain's observations.

    observables: comma-separated subset, or empty for all four.

    Writes 04_analysis/comparison.json and a figure per observable, and returns
    a SUMMARY plus their paths — the full record is per station per column per
    observable and does not belong inline, the same rule that keeps this server
    from returning results.
    """
    rd = Path(run_dir).resolve()
    if not rd.is_dir():
        return json.dumps({"ok": False, "error": f"no such run dir: {rd}"})
    adir = rd / "04_analysis"
    obs_path = Path(observations_json or (rd / "reception.json"))
    if not obs_path.is_file():
        return json.dumps({
            "ok": False,
            "error": f"no observations at {obs_path}. This server compares what "
                     f"it is given; GATHERING observations is reception's job, "
                     f"and reception persists them under `observations` in its "
                     f"reception.json. Re-run reception for this domain and "
                     f"period, then call this again.",
            "expected_blocks": ["streamflow", "water_table", "swe", "et"]},
            indent=2)

    rows, how = _extracted_rows(rd)
    if not rows:
        return json.dumps({"ok": False, "error": how,
                           "note": "no model output to compare — this is not "
                                   "a disagreement with the observations"},
                          indent=2)

    import compare as _cmp
    want = [s.strip().lower() for s in observables.split(",") if s.strip()]
    try:
        # COMPARE, DRAW, WRITE, SUMMARISE — one call, in the package (2026-08-12).
        # This file used to do those four itself, which made it a second reader
        # of a shape only that package defines: the summary loop broke on a
        # KeyError the moment every observable matched first and stopped nesting
        # columns under a station, and nothing over there could have known. The
        # Analyzer's step 1 now calls the same function, so the two entry points
        # cannot drift.
        done = _cmp.compare_run(rows, str(obs_path), str(adir),
                                observables=want or None)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "model_rows": how,
                           "error": f"{type(e).__name__}: {e}"[:300]}, indent=2)

    out = done["comparison"]
    return json.dumps({
        "ok": True,
        "model_rows": how,
        "observation_rows_dropped": out.get("n_observation_rows_dropped"),
        "comparison_json": done["path"],
        "figures": done["figures"],
        "summary": done["summary"],
        "note": out.get("note"),
    }, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────
# WHAT IS THE SCHEDULER DOING
# ─────────────────────────────────────────────────────────────────────
# States in which SLURM still owns the job. Anything else — COMPLETED, FAILED,
# TIMEOUT, CANCELLED, NODE_FAIL — means the scheduler is finished with it,
# whatever it did, and the caller should go look at the output.
ACTIVE_JOB_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
    "STAGE_OUT", "RESV_DEL_HOLD", "STOPPED",
}


def _slurm_state(job_id: Any) -> Optional[str]:
    """SLURM's word for what job_id is doing, or None if it will not say.

    squeue first — it is cheap and it is the only one that sees a job that has
    not started. Then sacct, which is the only one that remembers a job that has
    already left the queue.

    None means NO ANSWER, not "finished". A squeue that times out or a cluster
    without sacct must not be read as a completed ensemble; the caller decides
    what other evidence it trusts.

    A COPY of ExperimentManagerBase._slurm_state, and deliberately so: this used
    to import it, which made a SERVER depend on its CLIENT for something that is
    not framework knowledge at all — it is thirty lines about squeue. Duplicated
    across the boundary rather than shared through it.
    """
    jid = str(job_id).split("_")[0].split(".")[0]
    if not jid.isdigit():
        return None

    def _ask(cmd) -> Optional[str]:
        if not shutil.which(cmd[0]):
            return None
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception:                                       # noqa: BLE001
            return None
        line = (out.stdout or "").strip().splitlines()
        return line[0].strip() if line and line[0].strip() else None

    st = _ask(["squeue", "-h", "-j", jid, "-o", "%T"])
    if st:
        return st.upper()
    # -X so a job's steps do not shadow the job itself; the step lines come back
    # first and a step can read COMPLETED while the job is still going.
    st = _ask(["sacct", "-n", "-X", "-j", jid, "-o", "State"])
    if st:
        return st.split()[0].upper()    # sacct spells it "CANCELLED by 12345"
    return None


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
    state = _slurm_state(job_id)
    known = state is not None
    active = (state in ACTIVE_JOB_STATES) if known else True

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
