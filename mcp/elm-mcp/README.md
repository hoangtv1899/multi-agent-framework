# elm-mcp

An MCP server that fronts the **E3SM Land Model (ELM)**: it turns a latitude and longitude into a
configured single-column CIME case, builds it, submits it, and reads the output back as numbers
an agent can reason over.

Twelve tools cover the whole arc, from asking what the server can do, through building surface
and domain files for a point, to submitting an ensemble and extracting its water fluxes. Rather
than repeat a list that drifts, call **`describe_elm_capabilities`**, which reports the live
surface.

## Does it run on its own?

**The server does.** All twelve tools work from a bare clone with nothing beside it:

```
$ python3 src/framework_path.py
{ "framework_present": false, "mode": "vendored (standalone)",
  "server_tools": "available",
  "experiment_manager": "unavailable: needs a framework checkout" }
```

`describe_elm_capabilities` then reports `ready: true` with no missing requirements. The server
borrows only six small, self-contained modules from
[multi-agent-framework](https://github.com/hoangtv1899/multi-agent-framework/tree/compy-port-agentic-pipeline), and verbatim
copies of them ship in [src/_vendor/](src/_vendor/): `core.keyset`, `core.model_agent_base`,
`core.basemap`, `core.static_wtd`, `agents.analysis.compare_common` and `figstyle`, 1,796 lines
in total.

**Two things do not, and deliberately so:**

| Needs a framework checkout | Why |
|---|---|
| `src/elm_exp_manager.py` | It subclasses `core.exp_manager_base`, a 2,000-line base class that ELM **and** PFLOTRAN share. Copying it here would fork a contract two models depend on. |
| `scripts/analyze_run.py`, `scripts/analyze_agentic.py` | They drive the framework's Analyzer (`agents.analysis.step2_derive`, `core.figure_registry`, `agents.analyzer_agent`). |

Both are driven from the framework side rather than through MCP, so they do not affect anyone
using this as a server. They fail with a message naming `IDEAS_FRAMEWORK_DIR` rather than an
obscure `ModuleNotFoundError`.

To use a real framework instead of the bundled copies:

```bash
git clone -b compy-port-agentic-pipeline \
    https://github.com/hoangtv1899/multi-agent-framework.git
export IDEAS_FRAMEWORK_DIR=$PWD/multi-agent-framework
```

**Mind the branch.** The framework's default branch does not yet carry this work, and a checkout
of it has none of the modules named above. Point `IDEAS_FRAMEWORK_DIR` at such a tree and the
server says so and keeps using its bundled copies, rather than claiming to be fine and failing
at the first import.

When that points at a checkout, the framework's own files are used and the vendored copies never
execute. That ordering is what keeps one source of truth.

### About the vendored copies

They are refreshed, never hand-edited:

```bash
python3 scripts/sync_vendor.py --check    # has anything drifted?
python3 scripts/sync_vendor.py --write    # refresh from the framework
```

The framework's test suite runs that `--check`, so a drift is a test failure rather than a silent
divergence.

### This repository is a mirror

It is published with `git subtree split` from `multi-agent-framework/mcp/elm-mcp`, and **the
parent is the source of truth.** Commit here directly and the next mirror push overwrites it.
Send changes to the parent instead.

The dependency also runs the other way: the framework imports back into this directory for its
model dispatch, its analyzer's comparison loader, the coupling walk and its test suite. The two
are a mirror pair, not a separation.

## Setup

```bash
# 1. Point at your bulk data. Every key is optional.
cp paths.json.example paths.json
python3 src/paths.py                 # prints each path and which layer supplied it

# 2. OPTIONAL: use a real framework checkout instead of the bundled copies.
export IDEAS_FRAMEWORK_DIR=/path/to/multi-agent-framework
python3 src/framework_path.py        # confirms which mode you are in
```

Bulk-data paths resolve through four layers, in order: an explicit tool argument, then
`IDEAS_CONUS_RESTART` / `IDEAS_CONUS_SURFDATA` / `IDEAS_ELM_INPUT_FILES`, then `paths.json`, then
a built-in default.

**Prefer `paths.json` over the environment variables.** A standard MCP client forwards only
`HOME`, `LOGNAME`, `PATH`, `SHELL` and `USER` to a server process, so an exported variable may
never arrive. `paths.json` sits beside the server and is always read.

### Registering the server

Write this to `.mcp.json` **in the directory you will start Claude Code from**, which is usually
this clone's root. It is gitignored, because it names your interpreter and your directories.

```json
{
  "mcpServers": {
    "elm": {
      "command": "/path/to/python3",
      "args": ["/path/to/elm-mcp/main.py"],
      "env": {
        "E3SM_SRC_DIR": "/path/to/E3SM",
        "PSCRATCH": "/path/to/scratch",
        "IDEAS_SLURM_ACCOUNT": "e3sm",
        "IDEAS_SLURM_QUEUE": "short"
      }
    }
  }
}
```

Claude Code parses `.mcp.json` as **strict** JSON, so do not add `//` comments to it: one comment
and the whole file is rejected and the server never appears at all. To use a real framework
checkout rather than the bundled copies, add one more entry to `env`, remembering the comma:
`"IDEAS_FRAMEWORK_DIR": "/path/to/multi-agent-framework"`.

Then start Claude Code in that directory once and approve the server:

```bash
claude            # approve the project server when asked, then quit
claude mcp list   # elm: ... Connected
```

The first `claude mcp list` before that approval reads `Pending approval`, which is normal rather
than a failure: Claude Code only reads a project's `.mcp.json` after you have trusted the folder
interactively. Committing a settings file does not shortcut it.

To confirm the server is genuinely healthy rather than merely connected, ask it for
`describe_elm_capabilities`; a good setup answers `"ready": true` with
`"missing_requirements": []`.

### What else you need

Running ELM takes more than this server: an E3SM source checkout at `$E3SM_SRC_DIR`, a scratch
filesystem, a working `sbatch`, and the CONUS 1 km restart and surface files plus NLDAS-2 forcing
for the years you want. No web credentials are needed.

Python dependencies: `mcp`, `numpy`, `pandas`, `scipy`, `xarray`, `netCDF4`, `matplotlib`.

## Layout

```
main.py                the server: twelve tools, and every environment default it needs
src/framework_path.py  framework or vendored copies. The ONE place that decides.
src/_vendor/           verbatim copies of the six borrowed framework modules,
                       used only when no framework checkout is present
src/paths.py           where the bulk data is, with four-layer precedence
src/                   input building, output reading, the conceptual sweep,
                       the ELM experiment manager, figures
src/compare/           observation comparison: ET, SWE, streamflow, water table, maps
scripts/               batch entry points: build_cases, analyze_run, analyze_agentic,
                       ensemble_job, slice_proof_job, sync_vendor
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
