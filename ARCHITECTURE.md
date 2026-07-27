# Architecture

Four agents, in order. Each box below is one stage of the pipeline; the files
listed under it are the only files that implement it.

```
  user request
       │
       ▼
 ┌──────────────┐   brief    ┌──────────────┐   plan    ┌────────────────────┐   results   ┌──────────────┐
 │  RECEPTION   │ ─────────► │   PLANNER    │ ────────► │ EXPERIMENT MANAGER │ ──────────► │   ANALYZER   │
 │ what / where │            │  strategy +  │           │ materialize→warm   │             │ metrics →    │
 │  / when      │            │ feasibility  │           │ →build→run→couple  │             │ validate →   │
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

**Why the Planner is strict and the Analyzer is not.** A planner error is
expensive and silent: a bad sampling strategy costs a CIME build per column and
every downstream stage inherits it, and you only see the consequence once the
compute is spent. An analyzer error is cheap and visible — the data is
unchanged, regeneration takes seconds, and the failure lands in front of a
human. The two stages carry different risk, so they carry different
constraints.

## 3. Experiment Manager — strategy → runs

`src/core/elm_exp_manager.py` (`ELMExpManager.execute_plan`) is the whole stage.

| step | what happens | writes |
|---|---|---|
| 0 materialize | strategy → real columns via MCP terrain/soil/WTD | `columns.json`, `run_plan.json`, `sampling_design.png`, `assumptions.json` |
| 0b warm start | subset each column's donor gridcell out of the CONUS 1-km restart → per-column `finidat` + surfdata | `warmstart/` *(only if requested)* |
| 1 build | per-column domain + surface NetCDFs | `01_inputs/` |
| 2 prepare | one CIME build, then `--keepexe` clones | `02_setup_plots/column_surfaces.png` |
| 3 run | all columns as one SLURM job | `03_results/` |
| 4 analyze | history files → metrics + 5 figures | `04_analysis/` |
| 4b validate | compare against USGS / SNOTEL observations | `04_analysis/validation.json` |
| 4c interpret | LLM reads the numbers + verdicts | `04_analysis/interpretation.md` |
| 4d couple | each column's daily QINFL drives its own 1-D PFLOTRAN column | `05_pflotran/` *(only if the plan couples)* |
| 5 package | everything the Analyzer agent needs | `LLM_ANALYSIS_INPUT.json` |

**Step 0b — warm start** needs no prior run. `tools/make_finidat_subset.py`
subsets one gridcell — its landunits, columns and PFTs — out of the CONUS 1-km
restart into a standalone single-column `finidat`. The band is picked per column
from the MANIFEST, so a basin straddling a band edge is one pass, not one per
band.

It works because `create_crop_landunit=.false.` makes our sub-grid
`[1 natveg + 15 urban]` = 16 columns — exactly the CONUS layout (those restarts
contain zero crop columns across all 19.9M). Two consequences the code handles
explicitly, because both abort ELM at init if wrong:

- Columns are **snapped to their donor gridcell** (~250–400 m). Domain, surfdata
  and finidat must agree on coordinates; the shift goes in the ledger.
- The donor's own surfdata is subset alongside and used as the surface
  **template**, so `fsurdat` and `finidat` describe the same gridcell — ELM's
  `check_weights` gate. SSURGO soil is still overwritten on top, so the
  per-column soil science is unchanged.

`check_weights_agree()` reproduces that gate locally, so a mismatch costs a
second rather than a queue slot. Failure is non-fatal: the ensemble cold starts
and the ledger records the warm/cold split exactly.

**Step 4d — coupling** fires when the planner emits a `coupling_design` (or the
archetype is `coupling`), which both the Reception and Planner prompts already
speak. The PFLOTRAN columns are the SAME ones ELM just ran — same lat/lon, same
SSURGO profile, same Fan water table — so ELM partitions the surface water and
PFLOTRAN carries it down. 1-D Richards columns take ~1.5 s each, so this runs
serially in-process with no batch queue.

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

Every one of these steps is a *shared implementation*, not a copy: through
`_load_tool()` the manager imports `tools/expand_sampling.py` (0),
`make_warmstart.py` (0b), `plot_columns.py` (2), `analyze_run.py` (4),
`validate_run.py` (4b), `interpret_run.py` (4c), and `build_pflotran_cases.py`
+ `analyze_pflotran_coupled.py` (4d). Run a stage from the manager or from its
CLI and you get the same code.

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

Analysis figures: `partitioning` (where P goes, as fractions), `controls`
(fractions vs drivers with the orographic confound made visible), `spatial`
(the same quantities mapped over the watershed), `soil_control` (forcing held
constant) and `wtd_columns`. Validation emits one figure per observable:
`hydrograph`, `yield`, `water_table`, `swe`, and `context` (not scored). Plus
`sampling_design.png` (step 0) and `column_surfaces.png` (step 2).

**Figures illustrate; text interprets.** Figures render clean — axes, units,
legend. Every verdict lives in `figure_captions.json` beside them, which is
what the interpretation and any manuscript text draw from. `--annotate` puts
the verdicts back on the image for debugging a run rather than publishing it.

### The Analyzer is agentic, but bound to the evidence

It chooses which figures answer the question, renders them, LOOKS at them, and
revises — the same loop a person runs. Three guardrails, and only three:

1. **Numbers trace to JSON.** Any value the interpretation asserts comes from
   `hydro_summary.json` / `validation.json`. Vision judges whether a figure is
   *readable and on-point*; it never reads a measurement off an image.
2. **Provenance per figure** — registry-vetted or analyzer-authored — so a
   figure heading for a manuscript declares which it is.
3. **Verdicts are not negotiable.** It may plot anything and propose anything,
   but it cannot upgrade a `context-only` comparison to evidence, nor restate a
   number the deterministic artifacts do not contain. The honesty machinery
   (assumptions ledger, limitations catalogue, `compared` vs `context-only`,
   the domain-match and impossible-ratio refusals) binds the interpretation the
   way real coordinates bind the Planner.

Flexible about what it explores; strict about what it may claim.

---

## Shared infrastructure

| file | role |
|---|---|
| `src/agents/llm_agent.py` | LLM client + the JSON repair cascade |
| `src/core/mcp_manager.py`, `mcp_client.py` | one stdio MCP session per call (HPC-safe) |
| `mcp_config.json` | which MCP servers exist |

## Not driven by `workflow.py`

**Standalone PFLOTRAN.** The *coupled* path is step 4d above. What is still
CLI-only: recharge-scenario ensembles with no ELM behind them
(`tools/build_pflotran_cases.py --recharge-mm-yr` → `analyze_pflotran_run.py`)
and the reactive-transport demo (`tools/build_reactive_demo.py`).
`src/core/pflotran_input_agent.py` is the deck writer for all of them.
See `docs/PFLOTRAN_PLAN.md`.

**The shell path.** `tools/run_watershed.sh` runs the same stages step-by-step from
the login node (`build_cases.py`, `run_cases.sh`, `submit_cases.sh`,
`plot_columns.py`, `analyze_run.py`). Useful when you want to stop between steps.
See `docs/RUNBOOK.md`.

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
