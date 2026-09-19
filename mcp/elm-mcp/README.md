# elm-mcp

An MCP server that fronts the **E3SM Land Model (ELM)**: it turns a latitude and longitude into a
configured single-column CIME case, builds it, submits it, and reads the output back as numbers
an agent can reason over.

Twelve tools cover the whole arc, from asking what the server can do, through building surface
and domain files for a point, to submitting an ensemble and extracting its water fluxes. Rather
than repeat a list that drifts, call **`describe_elm_capabilities`**, which reports the live
surface.

## Read this before cloning

**This repository is a component of [multi-agent-framework](https://github.com/hoangtv1899/multi-agent-framework),
not a standalone package.** The server imports from the framework's `src/`: the experiment
lifecycle (`core.exp_manager_base`), the key-copying contract (`core.keyset`), the model-agent
contract (`core.model_agent_base`), the caveat catalogue (`core.limitations`), the observation
comparison layer (`agents.analysis.compare_common`) and several more, plus `figstyle` from its
`tools/`. That is roughly 4,000 lines at import time and more behind lazy imports.

So point it at a framework checkout:

```bash
export IDEAS_FRAMEWORK_DIR=/path/to/multi-agent-framework
python3 src/framework_path.py        # prints what resolved, and whether it is usable
```

If that variable is unset and this directory is not sitting at
`multi-agent-framework/mcp/elm-mcp/`, every entry point fails immediately with a message naming
`IDEAS_FRAMEWORK_DIR`. That is deliberate. The failure used to be a silent `sys.path` entry
pointing at a directory that did not exist, surfacing much later as
`ModuleNotFoundError: No module named 'core'`.

**This repository is a mirror. The parent is the source of truth.** It is published with
`git subtree split` from `multi-agent-framework/mcp/elm-mcp`. Commit here directly and the next
mirror push is rejected as non-fast-forward. Send changes to the parent repository instead.

The dependency also runs the other way: the framework imports back into this directory in half a
dozen places (its model dispatch, its analyzer's comparison loader, the coupling walk, its test
suite). The two are a mirror pair, not a separation.

## Setup

```bash
# 1. Point at the framework (see above).
export IDEAS_FRAMEWORK_DIR=/path/to/multi-agent-framework

# 2. Point at your bulk data. Every key is optional.
cp paths.json.example paths.json
python3 src/paths.py                 # prints each path and which layer supplied it
```

Bulk-data paths resolve through four layers, in order: an explicit tool argument, then
`IDEAS_CONUS_RESTART` / `IDEAS_CONUS_SURFDATA` / `IDEAS_ELM_INPUT_FILES`, then `paths.json`, then
a built-in default.

**Prefer `paths.json` over the environment variables.** A standard MCP client forwards only
`HOME`, `LOGNAME`, `PATH`, `SHELL` and `USER` to a server process, so an exported variable may
never arrive. `paths.json` sits beside the server and is always read.

### Registering the server

```jsonc
{
  "mcpServers": {
    "elm": {
      "command": "/path/to/python3",
      "args": ["/path/to/elm-mcp/main.py"],
      "env": {
        "IDEAS_FRAMEWORK_DIR": "/path/to/multi-agent-framework",
        "E3SM_SRC_DIR": "/path/to/E3SM",
        "PSCRATCH": "/path/to/scratch",
        "IDEAS_SLURM_ACCOUNT": "e3sm",
        "IDEAS_SLURM_QUEUE": "short"
      }
    }
  }
}
```

### What else you need

Running ELM takes more than this server: an E3SM source checkout at `$E3SM_SRC_DIR`, a scratch
filesystem, a working `sbatch`, and the CONUS 1 km restart and surface files plus NLDAS-2 forcing
for the years you want. No web credentials are needed.

Python dependencies: `mcp`, `numpy`, `pandas`, `scipy`, `xarray`, `netCDF4`, `matplotlib`.

## Layout

```
main.py                the server: twelve tools, and every environment default it needs
src/framework_path.py  where the framework checkout is. The ONE place that decides.
src/paths.py           where the bulk data is, with four-layer precedence
src/                   input building, output reading, the conceptual sweep,
                       the ELM experiment manager, figures
src/compare/           observation comparison: ET, SWE, streamflow, water table, maps
scripts/               batch entry points: build_cases, analyze_run, analyze_agentic,
                       ensemble_job, slice_proof_job
```

## How it works with the scheduler

Every MCP tool call spawns a fresh server process, so nothing can be cached between calls and no
tool may block longer than its client's timeout. The slow stages therefore do not wait: a build
or an ensemble run submits to SLURM and hands back a job id, and `check_elm_job` polls it. A tool
that returned only when ELM finished would have its session torn down mid-build.

## Boundary

The rule this server exists to enforce, from the parent's design documents: *anything that
requires knowing what ELM is lives in elm-mcp.* Its companion: *the framework decides what to ask
for and what the answer means; the server decides everything about how the model produces it.*

## Portability

`main.py` sets a number of defaults for the machine it was built on (PNNL's Compy): scratch
paths, an E3SM location, a SLURM account and queue, a UTF-8 locale that Intel compilers require,
and module-system variables. Every one has an environment override, but the defaults are not
portable and this server has not been run anywhere else.

## License

BSD 3-Clause. Copyright (c) 2026, Hoang Tran. See [LICENSE](LICENSE).
