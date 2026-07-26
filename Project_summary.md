# Project Summary — status and environment

**Last updated:** July 26, 2026 (Compy port; the two pipelines were joined and
the dead halves removed).

**Purpose:** hand-off document — current state, environment, and open work.
For *what the code is and where each stage lives*, see **[ARCHITECTURE.md](ARCHITECTURE.md)**;
that is the single source of truth for structure and this file does not repeat it.

---

## 1. What this project is

An LLM-orchestrated framework for scientific simulations. A natural-language
request drives: data gathering → experiment design → case configuration →
execution → analysis. Four agents: **Reception → Planner → Experiment Manager
→ Analyzer**.

The science target is watershed-scale single-column ELM ensembles on real data
(terrain, SSURGO soil, Fan water table, USGS/SNOTEL observations), with one-way
ELM→PFLOTRAN coupling for reactive transport.

**Status:** one pipeline, and it executes. `workflow.py` runs all four stages
end to end. (Until July 2026 there were two disjoint halves — `workflow.py`
executed but used a weaker reception/planner, while `tools/run_pipeline.py` was
smarter but stopped before execution. They are now the same path: the manager
imports the `tools/` implementations rather than duplicating them.)

---

## 2. Environment — Compy (PNNL)

| Item | Value |
|---|---|
| User / host | `tran289` · `compy01.pnl.gov` · CentOS 7, glibc 2.17, Slurm 18.08.6 |
| Repo root | `/qfs/people/tran289/IDEAS/multi-agent-framework` |
| **Environment** | **`source /qfs/people/tran289/IDEAS/env_compy.sh`** — modules + conda env `ideas` + all paths + `PNNL_API_KEY`. Do this first, every shell. |
| E3SM source | `/qfs/people/tran289/E3SM` |
| ELM case scratch | `/compyfs/tran289/E3SMv3/` |
| SLURM | account `e3sm`, partition `short` |
| LLM gateway | `https://ai-incubator-api.pnnl.gov` (OpenAI-compatible). Default model `claude-opus-4-8-project`. `tools=` must be passed on EVERY request or Bedrock-routed Claude 400s. |
| Style convention | Tabs in `src/core/`, spaces in `src/agents/`, `tests/`, `tools/` |
| **No packaging** | No setup.py; `sys.path.insert(0, "src")` — **run everything from the repo root**. |
| Git | branch `compy-port-agentic-pipeline`. `mcp/usgs-water-mcp` is vendored (rebuilt on Compy), not a submodule. |

**Compy-specific gotchas, all load-bearing:**

- **NLDAS-2 forcing comes from `DATM_MODE=CLMMOSARTTEST`** (`elm_wrapper.py`
  `FIXED_XML`), which reads `$DIN_LOC_ROOT/atm/datm7/NLDAS/clmforc.nldas.%ym.nc`
  (1979–2023, complete). The `atm_forcing.datm7.NLDAS2.0.125d.v1` tree that the
  NERSC path used has **empty** `Precip/` and `TPQWL/` here. The name mentions
  MOSART but it is purely a DATM stream preset — fine with the SROF stub.
- **ELM only, no MOSART**: compset uses `SROF`. Do not add a `mosart` namelist.
- Intel compilers crash on a non-UTF8 locale → every CIME subprocess is run with
  `LC_ALL=LANG=en_US.utf8` (`elm_wrapper._cime_env`).
- Slurm 18.08 has no `srun --exact` → use `--exclusive`; MPI needs `--mpi=pmi2`
  or Intel MPI hydra bootstrap hangs.
- `pip` cannot build from source on this glibc → `export PIP_ONLY_BINARY=":all:"`.

---

## 3. The MCP data layer

Config-driven (`mcp_config.json`, gitignored — absolute paths; see `.template`).
Each source is a stdio server; `MCPManager` → `MCPClient` per server, fresh
session per call (HPC-safe). All tools are **read-only** fetches.

| Server | Source | Key tools | Shape |
|---|---|---|---|
| weather | NWS / Open-Meteo | `get_climate_summary` | point |
| geology | USDA SSURGO | `get_soil_profile`, `get_pflotran_materials` | point |
| usgs_water | USGS OGC API | `get_groundwater_sites`, `get_water_table_depth` (param 72019), `get_monitoring_locations` | bbox |
| terrain | USGS 3DEP + WBD | `resolve_watershed` (HUC/name→bbox+area), `get_elevation`, `sample_elevation_grid` | point+bbox |
| fan_wtd | Fan et al. 2013 (local NetCDF) | `get_fan_wtd`, `sample_fan_wtd`, `data_status` | point+bbox |
| reaction_sandbox | PFLOTRAN reaction sandbox | reactive-transport deck helpers | — |

**Gotchas:**
- Observed WTD uses the **OGC API** (`api.waterdata.usgs.gov`), not legacy
  `waterservices.usgs.gov`. Depth-to-water = parameter **72019**, collection
  `field-measurements`.
- Fan tiles store WTD **negative-below-surface**; the server auto-detects sign,
  returns positive `depth_to_water_m`, reduces `time`, applies the land mask.
- SSURGO and weather are point-only → watershed work uses `sample_elevation_grid`
  and `sample_fan_wtd`.
- `tools/mcp_conus_sweep.py` surfaces sparse-well / no-data regions *before* you
  design a study.

---

## 4. The three-tier principle

The reason the Planner is not allowed to emit coordinates:

1. **Tier 1 — LLM designs the STRATEGY.** Stratification rules, justified N,
   feasibility verdict. No numbers that have to be real.
2. **Tier 2 — Python materializes it against real data** (`expand_sampling.py`,
   deterministic, no LLM): samples the real DEM, builds elevation bands,
   allocates N ∝ area (≥1/band), farthest-point selection, enriches each column
   with Fan WTD + SSURGO texture → concrete `(lat, lon)` columns.
3. **Tier 3 — the strict per-column ELM config** (`elm_experiment_builder.py`)
   is untouched by any LLM.

No hallucinated coordinates is a structural guarantee, not a prompt instruction.

---

## 5. Verify state in a new session

```bash
source /qfs/people/tran289/IDEAS/env_compy.sh
cd /qfs/people/tran289/IDEAS/multi-agent-framework

python -m pytest -q                       # expect: 3 failed, 155 passed, 62 skipped
python3 tools/mcp_conus_sweep.py --max-sites 4 --assert
python3 workflow.py --interactive
```

The **3 failures are known test drift**, not regressions — see §7.

---

## 6. Most-recent run

`workflow_outputs/elm_run_20260725_225034/` — Naches sub-watershed, WA
(HUC 17030002). 14 columns, 750–1868 m, NLDAS-2 1995, CONUS warm start.
14/14 succeeded. Sampling design, per-column surfaces (14 distinct SSURGO
profiles), and the five analysis figures all present.

---

## 7. Open work

**Known test drift (3 failures, pre-existing):**
- `test_validate.py::test_clean_site_brief` — `agents/validate.py` gained a
  `run_settings.resolved_period` check the fixture predates.
- `test_agentic.py::test_batch_returns_sentinel` and
  `test_tool_loop.py::test_ask_user_interactive_uses_human_answer` —
  `tool_loop._human_answer` gained a `tty` parameter and now opens `/dev/tty` at
  construction; there is no TTY under pytest.

**Wiring gaps:**
- **Warm start is not reachable from `workflow.py`.** `tools/make_warmstart.py`
  produces the `warmstart.json` that `columns_to_plan --finidat-map` consumes,
  but the manager's materialize step never calls it — integrated runs are
  cold-start. The shell path can warm-start today.
- **PFLOTRAN is not driven by the coordinator.** `tools/build_pflotran_cases.py`
  works standalone; re-integrating it behind the four-agent flow is open.
- **The refinement loop is not closed.** `conversation_context['last_analysis']`
  is stored but no agent reads it, so "now try X instead" starts from scratch.

**Validation quality (needs a look):**
- 93 gauges found in-domain but 0 with 1995 records — the validation reports a
  comparison it cannot actually make.
- SWE is off by 3–20× against SNOTEL.
- The WTD target reports `compared` on weak evidence.

**Science:**
- Only 2 of 14 Naches columns produce meaningful recharge (r²=0.018 vs
  elevation). Consistent with the shallow-soil/one-way-coupling framing, but
  worth confirming it is physics and not configuration.
- GSDE gridded soil (BNU, 30″, 8 layers to 2.3 m) as a fan_wtd-style MCP would
  make soil a real stratification axis. Deferred.
- Topographic-position sampling (TWI / height-above-drainage) beyond elevation bands.

**Debt:**
- "Hanford" hallucination in `analyzer_system_elm.txt`.
- Spin-up support (5–10 yr); monthly-period planning; per-experiment timeout.

---

## 8. Working-style notes

Decisive; trusts technical recommendations but wants the reasoning visible;
prefers concrete deliverables ("draft it and I'll revise"); iterates fast on
visuals; non-native English (short clear sentences, no filler). **Cautious about
destructive operations — investigate before deleting or moving.** Read this
summary, confirm direction, deliver focused work, don't re-explain known things.
