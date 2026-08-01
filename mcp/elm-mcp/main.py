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

Tools:
    describe_elm_capabilities()  -> what this server does, what it needs, and
                                    whether each of those is actually present
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
             "status": "planned"},
            {"step": 2, "tool": "prepare_elm_cases",
             "does": "CIME case build (reference case, then --keepexe clones)",
             "returns": "a JOB ID — this is sbatch'd, not run inline",
             "status": "planned"},
            {"step": 3, "tool": "submit_elm_ensemble",
             "does": "run every column concurrently as one batch job",
             "returns": "a JOB ID",
             "status": "planned"},
            {"step": 4, "tool": "check_elm_job",
             "does": "ask SLURM what a job from step 2 or 3 is doing",
             "returns": "state and whether it is still active",
             "status": "planned"},
            {"step": 5, "tool": "collect_elm_results",
             "does": "read the history files into rows",
             "returns": "a PATH to the analysis plus a compact summary — not "
                        "the series, which for 19 columns over 20 years is "
                        "tens of megabytes and would not survive stdio",
             "status": "planned"},
        ],

        "available_now": ["describe_elm_capabilities"],

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
