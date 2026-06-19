# Multi-Agent Framework for Subsurface Simulations — Project Summary

**Last updated:** June 18, 2026 (added the MCP data layer + agentic reception/planner/expander pipeline, June 10–11)

**Purpose:** Hand-off document so a future Claude chat can pick up the project context without rebuilding it from scratch.

---

## 0. ⚠️ The single most important thing: there are now TWO pipelines

The framework has a **mature back-end that executes simulations** and, built on top of it (June 10–11), a **new agentic front-end that is much smarter at understanding questions and designing campaigns but is DRY (stops before execution).** They are **not yet joined.**

| | **Legacy pipeline (EXECUTES)** | **New agentic pipeline (DRY)** |
|---|---|---|
| Entry point | `workflow.py` (`PFLOTRANCoordinator`) | `tools/run_pipeline.py` |
| Reception | `reception_agent.py` — 2-pass, **scripted** MCP fetch (weather+geology only) | `reception_llm.py` + `tool_loop.py` — **agentic**, LLM drives 12 MCP tools |
| Planner | `planner_agent.py` — **closed-vocab** DSL (`CONDITIONS_COUPLERS`), executable | prompt `planner_capability_probe.txt` — **open strategy** (sampling_plan, validation_design, requires_capabilities) |
| After plan | ExpManager / ELMExpManager → **real SLURM runs** → Analyzer | Tier-2 expander → concrete columns → **STOP** |
| Scale | one site, parameter sweep (forcing×soil×substrate) | watershed-scale spatial sampling + validation |
| Status | runs simulations (last real run: May 18 Fresno ELM, 3/3, 36.8 min) | dry, verified on Naches |

**The central open task is building the bridge** between them (see §6).

---

## 1. What this project is

A multi-agent LLM-orchestrated framework for scientific simulations on Perlmutter (NERSC). Natural language requests drive the pipeline: data gathering → experiment planning → case configuration → execution → analysis. **Architecture:** Reception → Planner → Execute → Analyze. Model-agnostic — supports PFLOTRAN (steady-state subsurface) and ELM (transient land-surface, single-column).

---

## 2. Environment specifics

| Item | Value |
|---|---|
| User | `hvtran` (PNNL) · Project `m3780` · root `~/RCSFA/multi-agent/` |
| E3SM source | `/global/u2/h/hvtran/E3SM` |
| ELM case scratch | `$PSCRATCH/E3SMv3/` |
| Reference surface | `/global/homes/h/hvtran/RCSFA/1d_elm/input_files/Surfacedata_Station_2006_.nc` |
| **Runtime env for LLM/MCP** | **`module load pytorch/2.8.0`** — `openai`, `mcp`, `xarray`, `netCDF4` live here (a `--user` install under `pytorch2.8.0/`). NOT in `nersc-python` / `RCSFA` conda env / default `pytorch/2.11.0`. Prefix LLM-calling commands with `module load pytorch/2.8.0 &&` (shell state doesn't persist). `PNNL_API_KEY` is set in env. |
| Conda env (non-LLM) | `nersc-python` works for pure structure/unit tests |
| LLM gateway | `https://ai-incubator-api.pnnl.gov` (OpenAI-compatible). **Supports function/tool-calling** on gpt-5.5, gemini-2.5-flash, claude-opus-4-8, claude-sonnet-4-5 — but `tools=` must be passed on EVERY request or Bedrock-routed Claude 400s. |
| Style convention | Tabs in `src/`, spaces in `tests/` and `tools/` |
| **No packaging** | No setup.py. Uses `sys.path.insert(0, "src")` — **run everything from the project root**. |
| Git | On `main`. Remotes: `github` (hoangtv1899/multi-agent-framework), `origin` (PNNL tanuki). `mcp/usgs-water-mcp` is a **vendored nested repo** (pgiffy's), not a `.gitmodules` submodule — its changes commit in its own history. |

---

## 3. Directory layout

```
~/RCSFA/multi-agent/
├── workflow.py                     # LEGACY entry point (executes; PFLOTRAN/ELM)
├── mcp_config.json                 # 5 MCP servers (gitignored — abs paths; see .template)
├── data/fan_wtd/                   # GITIGNORED — 832 MB Fan 2013 NetCDF tiles
│
├── src/agents/
│   ├── reception_agent.py          # LEGACY 2-pass reception (scripted MCP)
│   ├── planner_agent.py            # LEGACY closed-vocab planner (executable)
│   ├── analysis_report_agent.py    # analyzer (2-call; NO MCP tools yet)
│   ├── llm_agent.py                # SimpleLLMClient (openai chat completions)
│   ├── tool_loop.py        ★NEW    # generic LLM↔MCP tool-calling runtime
│   ├── reception_llm.py    ★NEW    # AGENTIC reception (LLMReceptionAgent)
│   └── prompts/
│       ├── reception_agentic.txt        ★NEW
│       ├── planner_capability_probe.txt ★NEW  (open strategy + conceptual/site archetype)
│       └── (legacy) reception_pass*, planner_system*, analyzer_system* …
│
├── src/core/                       # execution engine (PFLOTRAN no-prefix, ELM elm_*)
│   ├── exp_manager.py / experiment_manager.py   # PFLOTRAN two-layer (both live)
│   ├── elm_exp_manager.py + elm_*.py            # ELM single layer
│   └── mcp_client.py ★MOD (added list_tools_detailed) / mcp_manager.py / mcp_context.py / mcp_gatherer.py
│
├── mcp/                            # the actual MCP servers (stdio)
│   ├── weather-mcp/  geology-mcp/  usgs-water-mcp/ (vendored, +groundwater_api.py ★NEW)
│   ├── terrain-mcp/  ★NEW   (3DEP elevation + WBD watershed boundary)
│   └── fan-wtd-mcp/  ★NEW   (Fan 2013 equilibrium WTD, static NetCDF)
│
├── tools/
│   ├── run_pipeline.py         ★NEW  agentic dry pipeline: request→reception→planner→plan
│   ├── expand_sampling.py      ★NEW  Tier-2: strategy → concrete (lat,lon) columns
│   ├── probe_planner.py        ★NEW  planner-only capability probe (hand brief)
│   ├── mcp_conus_sweep.py      ★NEW  CONUS coverage diagnostic (--assert)
│   ├── make_architecture_slides.py ★NEW  regenerates the review deck
│   ├── naches_elm_brief.json   ★NEW  hand-written domain brief (probe input)
│   └── (existing) replot.py, create_slides.py, regenerate_surface.py, …
│
├── tests/   (62 passing under pytorch/2.8.0)
│   ├── test_elm_exp_manager_structure.py (20) / test_elm_setup_plotting.py (14)
│   │   / smoke_test_elm_wrapper.py (2)        = 36 legacy ELM guardrail
│   ├── test_mcp_tools.py ★NEW (20)            = terrain/fan/groundwater/expander logic
│   └── test_agentic.py   ★NEW (6)             = tool-schema gen + brief parse
│
└── workflow_outputs/  (gitignored)
    ├── <legacy run_id>/ 01_inputs 02_setup_plots 03_results 04_analysis …
    └── pipeline_<ts>/  reception_brief.json  reception_trace.json  plan.json  columns.json
```

**Key import facts:**
- `workflow.py` → `ExpManager` wraps `ExperimentManager` (PFLOTRAN). **Both live — do not delete either.** ELM uses single `ELMExpManager`. PFLOTRAN/ELM asymmetry = Stage-2 refactor target.
- The agentic path (`run_pipeline.py`) is **additive and parallel** — it does NOT touch the legacy reception/planner/`workflow.py`.

---

## 4. The MCP data layer (5 servers)

Config-driven (`mcp_config.json`): each source is a stdio server; `MCPManager` → `MCPClient` per server (fresh session per call, HPC-safe). All tools are **read-only** data fetches.

| Server | Source | Key tools | Shape |
|---|---|---|---|
| weather | NWS / Open-Meteo | `get_climate_summary` | point |
| geology | USDA SSURGO | `get_soil_profile`, `get_pflotran_materials` | point |
| usgs_water ★ | USGS OGC API | `get_groundwater_sites`, `get_water_table_depth` (param 72019), `get_monitoring_locations` (streamflow) | bbox |
| terrain ★ | USGS 3DEP + WBD | `resolve_watershed` (HUC/name→bbox+area), `get_elevation`, `sample_elevation_grid`, `elevation_summary` | point+bbox |
| fan_wtd ★ | Fan et al. 2013 (local NetCDF) | `get_fan_wtd`, `sample_fan_wtd`, `data_status` | point+bbox |

**MCP gotchas:**
- **Observed WTD uses the OGC API** (`api.waterdata.usgs.gov`), NOT legacy `waterservices.usgs.gov` (unreachable from NERSC — SSL handshake timeout). Depth-to-water = parameter **72019** in the `field-measurements` collection.
- **Fan tiles** store WTD **negative-below-surface**; the server auto-detects sign and returns a positive `depth_to_water_m`, reduces the `time` dim, applies the land `mask`. Naches HUC8 = `17030002`, 2,860.6 km², lon −180..180.
- `terrain` uses EPQS (3DEP point/grid) + WBD ArcGIS REST. SSURGO/weather are point-only → watershed work uses `sample_elevation_grid` / `sample_fan_wtd`.
- `tools/mcp_conus_sweep.py` is a coverage diagnostic over 12 CONUS sites (surfaces sparse wells / no-data before you design a study).

---

## 5. The agentic pipeline (June 10–11) — how it works

1. **Reception** (`reception_llm.py` + `tool_loop.py`, prompt `reception_agentic.txt`): a pure-LLM agent that DRIVES the MCP tools. It classifies intent, picks **archetype** — `conceptual` (mechanism, no real site → few/no tools) vs `site` (real place → resolve domain + inventory heterogeneity & observations) — gathers *proportionally*, reasons over results, and emits a framed **brief**. Curated 12-tool allowlist. Rule: only state tool-returned values.
2. **Planner** (`planner_capability_probe.txt`): open reasoning at **strategy altitude** — designs sampling strategy + validation pinned to observations + a `requires_capabilities` backlog. N is justified from the question, not a blind grid. Branches on `design_archetype`.
3. **Tier-2 expander** (`expand_sampling.py`): **deterministic Python, no LLM** — samples the real DEM, makes elevation bands, allocates N ∝ area (≥1/band), farthest-point selection, enriches each column with Fan WTD + SSURGO texture → concrete `(lat,lon)` columns. No hallucinated coordinates.

**Three-tier principle:** LLM designs the STRATEGY (Tier 1) → Python materializes it against real data (Tier 2) → the strict per-column ELM config (Tier 3, existing builder) stays untouched.

**Verified end-to-end (Naches):** `workflow_outputs/pipeline_20260611_011237/` — reception 10 tool calls → planner 12 columns / 4 bands + validation → expander **12 real columns** (e.g. `col_01` 447 m, Fan WTD 0.70 m valley vs ridge 200 m deep). `columns.json` saved.

Run it:
```bash
module load pytorch/2.8.0
python3 tools/run_pipeline.py "your question"
python3 tools/expand_sampling.py --run-dir workflow_outputs/pipeline_<ts>
```

---

## 6. Parking lot / next steps

### Priority 1 — Join the two pipelines (the main thing)
- **Make the analyzer agentic** — `analysis_report_agent.py` is the legacy 2-call interpreter with NO MCP tools. Give it observation-retrieval tools (wells/streamflow/Fan via `tool_loop`) so it FETCHES observations and compares them to ELM output. This is where "validate against observations" becomes a result, not a plan.
- **Wire Tier-2 → Tier-3** — feed `columns.json` into the per-column ELM config + run path. (Schema seam: the capability planner emits a *strategy*, not the executable `CONDITIONS_COUPLERS`.)

### Priority 2 — Data / science
- **GSDE gridded soil** (BNU, Shangguan/Dai 2014; 30″, 8 layers to 2.3 m, NetCDF) as a fan_wtd-style MCP → makes soil a real stratification axis. Deferred.
- **FLUXNET soil-moisture** ingest (no in-domain Naches tower; Metolius US-Me is the east-Cascades analog).
- Topographic-position sampling (valley vs hillslope via TWI / height-above-drainage) — beyond elevation bands.
- ELM→PFLOTRAN weak coupling.

### Priority 3 — Legacy bugs / debt (pre-existing)
- `case_dir = "not_prepared"` not written back in `elm_experiment_builder.prepare_cases()` (replot.py works around via PSCRATCH discovery).
- Verify `--keepexe` actually parallel-clones in workflow.
- "Hanford" hallucination in `analyzer_system_elm.txt`.
- Spinup support (5–10 yr) · monthly-period planner · per-experiment timeout.
- Stage-2 refactor: unify PFLOTRAN two-layer manager; `models/<name>/` layout.

---

## 7. How to verify state in a new chat

```bash
cd ~/RCSFA/multi-agent && module load pytorch/2.8.0

# Full suite: 62 passing (36 legacy ELM + 20 MCP/expander + 6 agentic)
python3 -m pytest tests/test_elm_exp_manager_structure.py tests/test_elm_setup_plotting.py \
                  tests/smoke_test_elm_wrapper.py tests/test_mcp_tools.py tests/test_agentic.py -q

# The 5 MCP servers load + a coverage sweep
python3 tools/mcp_conus_sweep.py --max-sites 4 --assert

# The agentic pipeline end-to-end (dry)
python3 tools/run_pipeline.py "explore GW/soil-moisture partitioning in the Naches sub-watershed, validate with obs"
```

---

## 8. Most-recent runs

| | Legacy (executed) | Agentic (dry) |
|---|---|---|
| Run dir | `workflow_outputs/elm_run_20260518_174555/` | `workflow_outputs/pipeline_20260611_011237/` |
| Site | Fresno, CA | Naches sub-watershed, WA (HUC 17030002) |
| Result | 3/3 ELM cases, 36.8 min | 12 concrete columns across 4 elevation bands |
| Note | 1985 baseline · 1990 dry · 1983 wet | valley Fan WTD 0.70 m vs ridge ~200 m |

---

## 9. Working-style notes (next session)

User is **hvtran** at PNNL. Decisive; trusts technical recommendations but wants the reasoning visible; prefers concrete deliverables ("draft it and I'll revise"); iterates fast on visuals; non-native English (short clear sentences, no filler); cautious about destructive ops — **always investigate before deleting/moving** (this codebase has non-obvious deps: the two-layer PFLOTRAN manager, the vendored usgs-water-mcp nested repo). Read this summary, confirm direction, deliver focused work, don't re-explain known things.

---

## 10. What changed June 10–11 (this work)

- Built 3 MCP servers (terrain, fan_wtd) + extended usgs_water with observed-WTD groundwater tools; provisioned the 832 MB Fan NAMERICA tiles.
- Built the **agentic layer**: generic `tool_loop.py` runtime, `LLMReceptionAgent`, planner conceptual/site archetypes, deterministic Tier-2 `expand_sampling.py`.
- Verified PNNL endpoint supports tool-calling (the `tools=`-every-round rule).
- Added 26 tests (→ 62 total); a CONUS coverage sweep; a review deck (`RCSFA_agentic_pipeline_review.pptx`).
- All additive — legacy reception/planner/`workflow.py` untouched.
