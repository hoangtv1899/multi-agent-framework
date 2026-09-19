# multi-agent-framework

> **This work lives on the `compy-port-agentic-pipeline` branch.** The default
> branch is an older tree without this README or most of what it describes, so
> clone the branch:
>
> ```bash
> git clone -b compy-port-agentic-pipeline https://github.com/hoangtv1899/multi-agent-framework.git
> ```

Ask a watershed question in plain English; get a real land-model ensemble and a written answer
whose every number traces back to a file on disk.

```
You: How does water partition between runoff and recharge in the Naches
     sub-watershed (HUC8 17030002)? Use ELM, 1995.
```

The framework reads that sentence, works out where the basin is and what data exists there,
designs a sampling strategy, builds and submits one single-column simulation per sampled
location, then compares the result against gauges and wells and writes up what it found.

Large language models make the judgement calls: what is being asked, which model can answer it,
what is worth plotting, what the result means. Code does every fetch, every arithmetic
operation and every audit. That split is the point of the design, and it is why a claim in the
final report can always be walked back to a number in a JSON file.

## How it works

Four stages run in order, wired by [workflow.py](workflow.py):

| Stage | What it decides | Writes |
|---|---|---|
| **Reception** ([src/agents/reception_llm.py](src/agents/reception_llm.py)) | What, where, when, and which model. Drives the read-only data servers itself. | `reception.json` |
| **Planner** ([src/agents/planner.py](src/agents/planner.py)) | How to sample the basin, how many columns, and whether the question is answerable at all. Emits rules, never coordinates. | `strategy.json` |
| **Experiment Manager** ([src/core/exp_manager_base.py](src/core/exp_manager_base.py)) | Turns those rules into real columns, builds the cases, submits them, reads the output back. | `experiment.json` |
| **Analyzer** ([src/agents/analyzer.py](src/agents/analyzer.py)) | Compares against observations, picks figures, interprets, reports. | `04_analysis/analysis.json`, `REPORT.md` |

Three things about that pipeline are deliberate and worth knowing before you read the code.

**No stage invents a coordinate.** The Planner is only allowed to emit stratification rules
(for example "sample five elevation bands"). [tools/expand_sampling.py](tools/expand_sampling.py)
then materializes those rules against the real digital elevation model. The per-column model
configuration is never touched by a language model.

**There is no `--model` flag.** Reception reads the available servers and names the model in its
brief; `_adopt_model` ([workflow.py:254](workflow.py#L254)) checks that name against the
`RUNNABLE` set in [src/core/model_servers.py](src/core/model_servers.py) and refuses outright
rather than quietly falling back to a default. The flag was removed in August 2026 because it
fixed the model before Reception had read the request.

**A queued run is not a failed run.** A study that goes to the scheduler returns in about three
seconds with status `pending` and zero successes. Job A runs the ensemble and job B, chained
with `--dependency=afterany`, analyses it, so you can close the terminal. The answer appears
later in `04_analysis/analysis.json`.

## Quick start

This project runs on PNNL's Compy cluster and is not portable as it stands; see
[Not in this repository](#not-in-this-repository).

```bash
# 1. Environment. Every shell, and the first line of every job script.
#    This script lives OUTSIDE the repo and holds live credentials; never copy it in.
source /qfs/people/tran289/IDEAS/env_compy.sh

# 2. Server registry. The template lists all eight servers with placeholder
#    paths; replace every /path/to/... with your own.
cp mcp_config.template.json mcp_config.json

# 3. Point the ELM server at your bulk data (all keys optional).
cp mcp/elm-mcp/paths.json.example mcp/elm-mcp/paths.json
python3 mcp/elm-mcp/src/paths.py      # prints what resolved, and from which layer

# 4. Run a study.
python workflow.py --interactive
```

There is no `setup.py` and no `pyproject.toml`. Imports rely on `sys.path.insert(0, "src")`
([workflow.py:26](workflow.py#L26)), so **every command must be typed from the repository root.**

### The eight flags

`workflow.py` has exactly eight ([workflow.py:1304-1362](workflow.py#L1304-L1362)):

| Flag | Effect |
|---|---|
| `--interactive` / `-i` | Reception may ask clarifying questions, and the run pauses at `DESIGN_REVIEW.md` for your yes before anything is built. |
| `--request TEXT` / `-r` | One study, unattended, from a single sentence. Put the basin, the model and the years in the text. |
| `--no-ask` | Reception resolves gaps itself and records them as conflicts instead of asking. |
| `--resume [RUN_DIR]` | With no argument, lists what can be resumed and why. The listing alone is a pure filesystem scan (no language model, no servers started), so it still works when the gateway is down. With a directory, continues that run, which does start the servers and reach the Analyzer. |
| `--finalize RUN_DIR` | Runs only the tail (extract, package, analyze) against output already on disk, without polling the scheduler. The last line of a combined study job. From a login node use `--resume` instead. |
| `--output-dir DIR` / `-o` | Where run directories are created (default `./workflow_outputs`). |
| `--mcp-config FILE` / `-m` | Which server registry to load (default `mcp_config.json`). |
| `--ask` / `-a` | Hidden; implied by `--interactive` and kept only so older commands keep working. |

Note that `--columns` and `--return` are **not** `workflow.py` flags even though they are part of
the project's vocabulary. They belong to the coupled walk, [tools/walk_setup.py](tools/walk_setup.py).

## What a run produces

One directory per run, named `<model>_run_<YYYYmmdd_HHMMSS>`. Three of the four boundary files
sit at the surface (the Analyzer's lands in `04_analysis/`), each written only when its stage
finishes, so a failure inside the Experiment Manager still leaves the reception and strategy
records intact:

```
elm_run_20260825_160608/
├── reception.json        what/where/when, and the data that was found
├── strategy.json         the sampling rules and the feasibility verdict
├── experiment.json       what actually ran, per column
├── run_plan.json         the executable plan: strategy plus backend refinement
├── sampling_design.png   where the columns landed, drawn before compute is spent
├── DESIGN_REVIEW.md      the plan on one page, plus the decision that was taken
├── run_state.json        which stages are done, and any outstanding job id
├── 01_inputs/            columns.json, case_inputs.json, built_cases.json
├── 02_setup_plots/       column_surfaces.png, the per-column soil panels
├── 03_results/           model output
└── 04_analysis/          analysis.json, REPORT.md, figures
```

A real run leaves more than this at the surface (`assumptions.json`, `plan.json`,
`reception_brief.json` and others). [docs/RUN_LAYOUT.md](docs/RUN_LAYOUT.md) describes the
intended contract, which the code has not fully converged on; the files above are the ones worth
opening.

`run_state.json` is a handoff note rather than a progress bar: it carries the finished stages and
the outstanding job id across process and session boundaries, which is what makes `--resume` work
from disk alone. The contract is written up in [docs/RUN_LAYOUT.md](docs/RUN_LAYOUT.md).

## Repository map

```
workflow.py              the only entry point for a full study
src/core/                everything true of ANY model: the experiment lifecycle, the
                         data fetch, the MCP plumbing. Names no model, by design.
src/agents/              the reasoning boxes: reception, planner, design review,
                         analyzer, and the analyzer's five-step chain in analysis/
mcp/                     the data and model servers (see below)
tools/                   command-line drivers for individual stages, plus the
                         coupled ELM-PFLOTRAN walk
tests/                   the test suite, offline by default
eval/                    the frozen agent-evaluation harness
docs/                    design documents and contracts (see Where to read next)
```

The rule that shapes the layout is one sentence, from
[docs/ELM_MCP_PLAN.md](docs/ELM_MCP_PLAN.md): *anything that requires knowing what ELM is lives
in elm-mcp.* Its companion, from [docs/EXP_MANAGER_ELM.md](docs/EXP_MANAGER_ELM.md): *the
framework decides what to ask for and what the answer means; the server decides everything about
how the model produces it.*

That is why the two experiment managers live beside their servers
([mcp/elm-mcp/src/elm_exp_manager.py](mcp/elm-mcp/src/elm_exp_manager.py),
[mcp/pflotran-mcp/pflotran_exp_manager.py](mcp/pflotran-mcp/pflotran_exp_manager.py)) rather than
in `src/core/`. One function, `_manager_for` in [src/core/resumable.py](src/core/resumable.py),
is the single place a model name maps to a class. Adding a third model means adding one branch
there.

## The MCP servers

Every data source and every model is reached over the Model Context Protocol on stdio. Browsing
`mcp/` you will count ten directories; the framework registers eight, and one of those is not a
server at all.

| Server | Source | Credentials |
|---|---|---|
| `usgs_water` | USGS site discovery, streamflow, groundwater levels (OGC API) | none (`USGS_API_KEY` optional, raises the rate limit) |
| `terrain` | USGS 3DEP elevation and watershed boundaries | none |
| `snotel` | NRCS SNOTEL snow water equivalent | none |
| `daymet` | Daymet V4 daily 1 km weather | none |
| `ameriflux` | AmeriFlux eddy-covariance towers, for ET comparison | discovery open; **downloads need** `AMERIFLUX_USER_ID`, `AMERIFLUX_EMAIL` |
| `hydrodata` | ParFlow CONUS2 water table and Fan 2013 wells | **a free Princeton email and PIN, registered once per machine** |
| `elm` | The E3SM Land Model itself: builds and submits CIME column cases | none for the web, but needs an E3SM tree, scratch and `sbatch` |
| `pflotran` | PFLOTRAN, via a console script installed from another repository | none |

`geology` (SSURGO soils) is present in the tree but **retired**: a soil survey stops at about
1.5 m, so a deep column built from it was 88 to 97 percent extrapolated. `weather` is present but
never registered. Both still run if launched by hand.

The ELM server registers twelve tools. Rather than repeat a list that drifts, call
`describe_elm_capabilities`, which reports the live surface.

Two registry files wire two different clients, and **they are not the same file and do not list
the same servers**. `mcp_config.json` is what the Python framework reads (the eight above;
gitignored because it holds absolute paths). `.mcp.json` is what Claude Code reads (two servers,
`elm` and `PFLOTRAN`). Changing one does not change the other.

[mcp_config.template.json](mcp_config.template.json) carries all eight with placeholder paths.
Two spellings matter and neither is checked for you: the top-level key must be `mcp_servers`
(not `mcpServers`, which is Claude Code's spelling and loads zero servers in silence), and at
least one of `elm` or `pflotran` must be present, or reception finds no model in `RUNNABLE` and
the study stops before it starts.

One consequence worth internalising: every tool call spawns a **fresh** server process, so
nothing can be cached in server memory between calls and no tool may block longer than its
timeout. That is why the ELM server's build and run stages submit to SLURM and hand back a job id
instead of waiting. It is also why you should ask for many points in one call rather than looping
per point; each call pays a process spawn of a second or two.

## The coupled walk

ELM and PFLOTRAN can march through the same year together, exchanging at window boundaries. Each
window: ELM runs the window, its daily drainage or recharge flux drives the PFLOTRAN column's top
face, PFLOTRAN continues from its checkpoint, and the solved water table is stamped onto ELM's
next restart.

```bash
# 1. Measure the two numbers the outlet needs, from the run's own records.
python tools/recession_tau.py RUN_DIR --out tau.json
python tools/drainage_datum.py RUN_DIR --source hand --out datum.json

# 2. Build the walk directory. Login-node safe: it runs nothing.
python tools/walk_setup.py --elm-run ELM_RUN --pf-run PF_RUN --out WALK_DIR \
       --months 1 --bottom none --forward qcharge \
       --sink-datum hand --sink-tau-days 42.7

# 3. Submit as ONE job. Ask first, and announce the job id.
tools/walk_sbatch.sh WALK_DIR 2 short

# 4. Read what it did.
python tools/walk_report.py WALK_DIR
python tools/walk_figures.py WALK_DIR
```

The physics that makes the exchange meaningful is the **lateral sink**. Sealing the bottom of the
PFLOTRAN column is what lets the water table actually move (anchored, it barely responded: a
hundredfold increase in recharge moved it 11 cm). But a sealed column needs somewhere for water to
go, so a thin band of side faces above a drainage datum leaks outward, outflow only, while the
water table stands above that datum. The conductance is not guessed from permeability; it is
mapped from the basin's own gauge recession, which makes the column a linear reservoir and
produces a per-column baseflow comparable against USGS gauges in basin sum.

Measured at Naches for 1979 over 18 columns: sealed without a sink, five melt-fed columns filled
to the surface and dropped out between days 90 and 151. With the sink, all 18 walked the full
year, none reached the surface, and the unfed deep columns drained with e-folding times of 32 to
66 days against a 42.7 day dial.

The full contract, with every physical claim cited to a line in the PFLOTRAN v7.0 source, is
[docs/coupling/lateral_sink_design.md](docs/coupling/lateral_sink_design.md). What the experiment
measured is [docs/coupling/naches_sink_results_2026-09-13.md](docs/coupling/naches_sink_results_2026-09-13.md).

A few traps that cost real debugging time:

- The walk **refuses a leap year** outright, because every daily clock in the pipeline divides by 365.
- A walk directory is built **once**; re-running setup would rewrite the spin decks under a live job. Resuming is `walk_job.py`'s business, from `walk_state.json`.
- `--sink-tau-days` is **required** whenever a sink is requested, and has no hidden default.
- `--return` defaults to `wt` (the water table alone) as of 2026-09-13, but a walk directory written before that flip carries no `return_leg` and still means `wt+profile`. Read `walk.json`, not the current default, when interpreting an old walk.
- PFLOTRAN's exit code is **not** the verdict: it exits 0 when it gives up at its timestep cap.

## Tests

```bash
pytest                 # offline, fast, deterministic. The default.
pytest --runlive       # adds live MCP data sources (network)
pytest --runllm        # adds real LLM round-trips (needs PNNL_API_KEY)
pytest --runcompute    # adds ELM builds and runs (needs an salloc node)
```

**Plain `pytest` does not currently pass**, which you should know before reading a failure as
something you caused. Two files, `tests/test_backends.py` and `tests/test_lambda_pflotran.py`,
import names deleted in August 2026 and abort collection before a single test runs. Excluding
them, about 70 of roughly 1,060 tests fail on stale expectations, and a few more fail in a fresh
clone because they want bulk data that is not in git. Cleaning that up is outstanding work.

The tiers are defined in [conftest.py](conftest.py) and declared as markers in
[pytest.ini](pytest.ini). Plain `pytest` staying offline is a property worth protecting: it is
what makes a green suite meaningful without a cluster or a gateway. See
[tests/README.md](tests/README.md), whose setup lines are NERSC-era and stale, though its
description of the tiers is current.

## Where to read next

`docs/` holds eighteen Markdown files and had no index until this README. Start here:

**Current and authoritative**

- [ARCHITECTURE.md](ARCHITECTURE.md) is the structural reference: one section per stage, with the files that implement it. Read it before the code. Three things in it are stale: it names `src/core/backends.py` as the model table (deleted 2026-08-18, now `src/core/model_servers.py`), it says the CLI picks the model (Reception does), and it lists `lambda-pflotran` as a working backend (it is not in `RUNNABLE`).
- [Project_summary.md](Project_summary.md) holds what ARCHITECTURE.md deliberately omits: the Compy environment table, the host's gotchas, and the MCP data layer with per-server traps. Its status half is dated July 2026 and has drifted; the environment half is still the best record of the machine.
- [docs/RUN_LAYOUT.md](docs/RUN_LAYOUT.md), the run-directory contract.
- [docs/coupling/lateral_sink_design.md](docs/coupling/lateral_sink_design.md), the outlet contract, cited to source lines.
- [docs/CLAUDE_CODE_MCP_GUIDE.md](docs/CLAUDE_CODE_MCP_GUIDE.md), how to drive the models from a Claude Code terminal with no `workflow.py` in the loop. The only true how-to here, with worked examples that were actually executed.
- [docs/planning/elm_bgc_questions_2026-09-17.md](docs/planning/elm_bgc_questions_2026-09-17.md), the current direction: ELM only, coupling parked, five ranked Naches fire questions.

**Design record, still accurate on reasoning**

- [docs/ELM_MCP_PLAN.md](docs/ELM_MCP_PLAN.md) and [docs/EXP_MANAGER_ELM.md](docs/EXP_MANAGER_ELM.md), the boundary between framework and server. The former still says "not yet built" in its header; it was built.
- [docs/PFLOTRAN_PLAN.md](docs/PFLOTRAN_PLAN.md), [docs/ELM_CONCEPTUAL_PLAN.md](docs/ELM_CONCEPTUAL_PLAN.md), [docs/paper/loose_vs_tight_coupling.md](docs/paper/loose_vs_tight_coupling.md).
- [docs/reference/README.md](docs/reference/README.md) indexes self-contained HTML walkthroughs of each stage. The architecture map is **generated** by `tools/make_arch_map.py` and must be rebuilt, never hand-edited; it fails loudly when a step names a function that no longer exists.

**History; read for context, do not follow the commands**

- [docs/RUNBOOK.md](docs/RUNBOOK.md) is NERSC-era. Its setup lines and four of the tool paths it prints have moved.
- [docs/REACTION_MCP_INSTALL_COMPY.md](docs/REACTION_MCP_INSTALL_COMPY.md) carries its own "superseded" banner, though its version numbers (PFLOTRAN v7.0, PETSc 3.21.6) are still the current ones.

## Not in this repository

Cloning this repo does not give you a working system. You also need:

- **The PFLOTRAN MCP server.** It is a separate repository (`river-corridors-sfa/reaction_sandbox_mcp`, branch `compy-port`) which is **not publicly readable**, so the `PFLOTRAN` entry in `.mcp.json` is dead for anyone outside that organisation. It is pip-installed into the `ideas` environment and launched as the `pflotran-mcp` console script. [mcp/pflotran-mcp/](mcp/pflotran-mcp/) holds only the framework's driving side. The two repositories must move together: the PFLOTRAN half of the lateral sink lives over there.
- **The environment script**, `env_compy.sh`, which sits one directory above the repo and holds live credentials in plain text. It is correctly outside version control and must stay there. [env.sh.example](env.sh.example) lists every variable the framework reads, with names and no values; copy it somewhere outside the repository and fill it in.
- **An E3SM source checkout** at `$E3SM_SRC_DIR`, plus the CONUS 1 km restart and surface files and NLDAS-2 forcing on scratch. NLDAS-2 covers 1979 to 2023 and is the only forcing path that works.
- **A PFLOTRAN executable** for the coupled and walk paths.

Host constraints, stated once: Compy is CentOS 7 with glibc 2.17, so `pip` cannot build from
source. Export `PIP_ONLY_BINARY=":all:"` before any install. Intel compilers crash on a non-UTF8
locale, so CIME subprocesses are forced to `en_US.utf8`.

## House rules

These are scattered through the design documents and are collected here because breaking them is
expensive.

- **Ask before submitting any SLURM job, and announce every job id.** Account `e3sm`, partition `short` (which caps at 2 hours; pass `slurm` for longer).
- **Run from the repository root.** There is no packaging.
- **Ask for many points in one MCP call, not one call per point.** Each call spawns a process.
- **Prompts encode capabilities.** Deleting a tool or a dataset means grepping `src/agents/prompts/*.txt`; a stale line there is a capability claim the planner will design against.
- **Prefer `mcp/elm-mcp/paths.json` over the `IDEAS_*` environment variables.** A standard MCP client forwards only `HOME`, `LOGNAME`, `PATH`, `SHELL` and `USER`, so an exported variable may never arrive. The framework's own client forwards everything, which is exactly the asymmetry that makes a server work under one launcher and fail under another on the same machine.

## License

BSD 3-Clause. Copyright (c) 2026, Hoang Tran. See [LICENSE](LICENSE).
