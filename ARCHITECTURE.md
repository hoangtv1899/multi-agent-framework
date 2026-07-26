# Architecture

Four agents, in order. Each box below is one stage of the pipeline; the files
listed under it are the only files that implement it.

```
  user request
       │
       ▼
 ┌──────────────┐   brief    ┌──────────────┐   plan    ┌────────────────────┐   results   ┌──────────────┐
 │  RECEPTION   │ ─────────► │   PLANNER    │ ────────► │ EXPERIMENT MANAGER │ ──────────► │   ANALYZER   │
 │ what / where │            │  strategy +  │           │  materialize→build │             │ metrics →    │
 │  / when      │            │ feasibility  │           │  →prepare→run      │             │ validate →   │
 └──────────────┘            └──────────────┘           └────────────────────┘             │ interpret    │
        MCP tool loop           one LLM call               SLURM, no LLM                    └──────────────┘
```

Entry point: `workflow.py` (`WorkflowCoordinator`). It owns the arrows, nothing else.

```
python workflow.py --interactive          # normal use
python workflow.py --interactive --ask    # let Reception ask instead of defaulting
```

---

## 1. Reception — request → brief

| file | role |
|---|---|
| `src/agents/reception_llm.py` | `LLMReceptionAgent`: the agent. Drives MCP tools itself. |
| `src/agents/tool_loop.py` | `ToolLoopAgent`: generic LLM↔MCP tool-calling loop. |
| `src/agents/reception_adapter.py` | Coordinator-facing shim + the `ReceptionResult` contract. |
| `src/agents/prompts/reception_agentic.txt` | The system prompt. |

The LLM decides what to fetch, when to stop, and what it means. It emits
`domain: {name, huc, bbox}` — the spatial handle everything downstream needs —
plus intent, scientific framing, and `run_settings.resolved_period`.

Rule enforced by the prompt: **only state tool-returned values.** If it did not
come back from an MCP call, Reception does not claim it.

## 2. Planner — brief → strategy

| file | role |
|---|---|
| `src/agents/planner_agent.py` | `PlannerAgent`: one LLM call, then a deterministic schema check. |
| `src/agents/prompts/planner_capability_probe_v2.txt` | Production prompt. |
| `src/agents/prompts/planner_capability_probe.txt` | Frozen v0.1 — pinned by `eval/`, do not edit. |
| `src/agents/validate.py` | Deterministic brief/plan checks. |

The Planner emits a **strategy, never coordinates**: stratification rules
(elevation/forcing bands, justified N), an explicit `full`/`partial`/`infeasible`
feasibility verdict, and a `requires_capabilities` backlog. It is architecturally
forbidden from inventing a number that has to be real — turning rules into
coordinates is the next stage's job, and that stage uses data.

## 3. Experiment Manager — strategy → runs

`src/core/elm_exp_manager.py` (`ELMExpManager.execute_plan`) is the whole stage.

| step | what happens | writes |
|---|---|---|
| 0 materialize | strategy → real columns via MCP terrain/soil/WTD | `columns.json`, `run_plan.json`, `sampling_design.png` |
| 1 build | per-column domain + surface NetCDFs | `01_inputs/` |
| 2 prepare | one CIME build, then `--keepexe` clones | `02_setup_plots/column_surfaces.png` |
| 3 run | all columns as one SLURM job | `03_results/` |
| 4 analyze | history files → metrics + 5 figures | `04_analysis/` |
| 4b validate | compare against USGS / SNOTEL observations | `04_analysis/validation.json` |
| 4c interpret | LLM reads the numbers + verdicts | `04_analysis/interpretation.md` |
| 5 package | everything the Analyzer agent needs | `LLM_ANALYSIS_INPUT.json` |

Supporting modules:

| file | role |
|---|---|
| `src/core/columns_to_plan.py` | `columns.json` → executable `CONDITIONS_COUPLERS` + assumptions ledger |
| `src/core/elm_experiment_builder.py` | plan → per-column agents; owns the one-build/many-clones path |
| `src/core/elm_domain_generator.py` | per-column single-cell domain NetCDF |
| `src/core/elm_surface_generator.py` | per-column `fsurdat` from SSURGO + CONUS vegetation |
| `src/core/elm_wrapper.py` | the real CIME driver (`create_newcase`, `xmlchange`, build, `srun`) |
| `src/core/elm_input_agent.py` | adapter onto the `ModelAgentBase` contract |
| `src/core/model_agent_base.py` | the contract |

Step 0 and steps 4/4b/4c are *shared implementations*, not copies: the manager
imports `tools/expand_sampling.py`, `tools/analyze_run.py`, `tools/validate_run.py`
and `tools/interpret_run.py` through `_load_tool()`. Run a stage from the
manager or from its CLI and you get the same code.

## 4. Analyzer — runs → answer

| file | role |
|---|---|
| `src/core/elm_results_analyzer.py` | history NetCDF → metrics, spatial/soil/driver summaries |
| `src/core/limitations.py` | structural vs configuration caveats attached to every result |
| `tools/analyze_run.py` | the five science figures + `hydro_summary.json` |
| `tools/validate_run.py` | observation comparison → `validation.json` |
| `tools/interpret_run.py` | LLM interpretation grounded in numbers + verdicts |
| `src/agents/analysis_report_agent.py` | final report the coordinator returns |
| `src/agents/prompts/analyzer_system_elm.txt`, `analyzer_validation.txt` | prompts |

The five figures: `elevation_gradient`, `soil_control`, `water_budget`,
`driver_response`, `wtd_columns`. Plus `sampling_design.png` (step 0) and
`column_surfaces.png` (step 2).

---

## Shared infrastructure

| file | role |
|---|---|
| `src/agents/llm_agent.py` | LLM client + the JSON repair cascade |
| `src/core/mcp_manager.py`, `mcp_client.py` | one stdio MCP session per call (HPC-safe) |
| `mcp_config.json` | which MCP servers exist |

## Not driven by `workflow.py`

**PFLOTRAN.** Reactive transport and the ELM→PFLOTRAN coupling run through
`tools/build_pflotran_cases.py` → `tools/analyze_pflotran_run.py` /
`analyze_pflotran_coupled.py`, with `src/core/pflotran_input_agent.py` as the deck
writer. See `docs/PFLOTRAN_PLAN.md`. Re-integrating this behind the coordinator is
open work.

**The shell path.** `tools/run_watershed.sh` runs the same stages step-by-step from
the login node (`build_cases.py`, `run_cases.sh`, `submit_cases.sh`,
`plot_columns.py`, `analyze_run.py`). Useful when you want to stop between steps.
See `docs/RUNBOOK.md`.

**Warm start.** `tools/make_warmstart.py` builds the `warmstart.json` that
`columns_to_plan.py --finidat-map` consumes. Reachable from the shell path only —
the coordinator does not call it yet, so integrated runs are cold-start.

**Standalone diagnostics.** `tools/scout_watersheds.py` (pre-flight observation
coverage), `tools/mcp_conus_sweep.py` (MCP coverage), `tools/probe_planner.py`
(planner quality), `tools/replot.py` (re-plot a finished run),
`tools/make_soil_sweep.py`.

**The evaluation.** `eval/` is frozen paper provenance — pre-registered suite,
runner, scorer, raw records. See `docs/paper/REPRODUCIBILITY.md`.

---

## Testing

```
pytest                  # offline unit tests (default; fast, no network, no LLM)
pytest --runlive        # + live MCP data sources
pytest --runllm         # + real LLM round-trips (needs PNNL_API_KEY)
pytest --runcompute     # + ELM build/run (needs an salloc node)
```
