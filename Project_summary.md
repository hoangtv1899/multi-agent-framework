# Project Summary — status and environment

**Last updated:** July 26, 2026 (Compy port; pipelines joined, dead halves
removed, warm start and ELM→PFLOTRAN coupling wired into workflow.py).

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
| LLM gateway | `https://ai-incubator-api.pnnl.gov` (OpenAI-compatible). Default model `claude-opus-4-8-project`. `tools=` must be passed on EVERY request or Bedrock-routed Claude 400s. **`max_tokens` must be set explicitly** — the default is 4096, which truncated every planner reply mid-JSON (`SimpleLLMClient.max_tokens = 16384`, tool loop 8192). |
| Style convention | Tabs in `src/core/`, spaces in `src/agents/`, `tests/`, `tools/` |
| **No packaging** | No setup.py; `sys.path.insert(0, "src")` — **run everything from the repo root**. |
| Git | branch `compy-port-agentic-pipeline`. `mcp/usgs-water-mcp` is vendored (rebuilt on Compy), not a submodule. |

**Compy-specific gotchas, all load-bearing:**

- **NLDAS-2 forcing comes from `DATM_MODE=CLMMOSARTTEST`** (`elm_wrapper.py`
  `FIXED_XML`), which reads `$DIN_LOC_ROOT/atm/datm7/NLDAS/clmforc.nldas.%ym.nc`
  (**1979–2023**, complete; 2024 is staged but only 8/12 months). This is the
  ONLY forcing the framework can run — the Qian fallback that the prompts used
  to advertise is unreachable. `src/core/forcing_availability.py` reads the
  window off disk; nothing should restate it from memory. The `atm_forcing.datm7.NLDAS2.0.125d.v1` tree that the
  NERSC path used has **empty** `Precip/` and `TPQWL/` here. The name mentions
  MOSART but it is purely a DATM stream preset — fine with the SROF stub.
- **ELM only, no MOSART**: compset uses `SROF`. Do not add a `mosart` namelist.
- Intel compilers crash on a non-UTF8 locale → every CIME subprocess is run with
  `LC_ALL=LANG=en_US.utf8` (`elm_wrapper._cime_env`).
- Slurm 18.08 has no `srun --exact` → use `--exclusive`; MPI needs `--mpi=pmi2`
  or Intel MPI hydra bootstrap hangs.
- `pip` cannot build from source on this glibc → `export PIP_ONLY_BINARY=":all:"`.
- **cartopy** (locator inset on `sampling_design.png`) installs fine from a
  wheel: `PIP_ONLY_BINARY=":all:" pip install cartopy`. Its absence is not an
  error — the inset is skipped and the figure still renders.
- The USGS OGC `daily` collection **cancels any query over ~60 s of server
  time** with a 400 (`"Long running query has been cancelled"`). Four years of
  one bbox is under the budget, five is over, so dated queries are split per
  calendar year (`groundwater_api._year_chunks`). Cold queries take 30-60 s and
  the same query warm takes 0.2 s, which is why the timeouts are 90 s/120 s.

---

## 3. The MCP data layer

Config-driven (`mcp_config.json`, gitignored — absolute paths; see `.template`).
Each source is a stdio server; `MCPManager` → `MCPClient` per server, fresh
session per call (HPC-safe). All tools are **read-only** fetches.

| Server | Source | Key tools | Shape |
|---|---|---|---|
| weather | NWS / Open-Meteo | `get_climate_summary` | point |
| geology | USDA SSURGO | `get_soil_profile`, `get_pflotran_materials` | point |
| usgs_water | USGS OGC API | `get_streamflow`, `get_water_table` (param 72019) | bbox |
| terrain | USGS 3DEP + WBD | `resolve_watershed` (HUC/name→bbox+area), `get_elevation`, `sample_elevation_grid` | point+bbox |
| fan_wtd | Fan et al. 2013 (local NetCDF) | `get_fan_wtd`, `sample_fan_wtd`, `data_status` | point+bbox |
| reaction_sandbox | PFLOTRAN reaction sandbox | reactive-transport deck helpers | — |

**Gotchas:**
- Observed WTD uses the **OGC API** (`api.waterdata.usgs.gov`), not legacy
  `waterservices.usgs.gov`. Depth-to-water = parameter **72019**, collection
  `field-measurements`.
- **A station list is not a data list, and one tool answers both.** Each
  observation tool — `usgs_water.get_streamflow`, `usgs_water.get_water_table`,
  `snotel.get_swe` — takes the same three shapes:
  **no dates** → period of record (one cheap query: which years could this basin
  *ever* be validated in); **dates** → which stations actually reported then;
  **dates + `with_values=True`** → the series, for validators only.
  For the Naches in 1995 the counts are 93 gauges present and 1 reporting.
  Reception calls these before the design, `validate_run.py` after it, so the
  two stages cannot disagree about which station reports.
- Record **spans are an outer envelope, not truth** — the Naches outlet gauge
  reports 1899–1990 yet has nothing in 1985. A span rules a year OUT reliably;
  only a dated query rules one IN.
- Imperial→metric conversion happens **inside the servers** (mi²→km², ft→m,
  in→mm), so everything downstream is metric on arrival.
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

python -m pytest -q                       # expect: 297 passed, 62 skipped
python3 tools/mcp_conus_sweep.py --max-sites 4 --assert
python3 workflow.py --interactive
```

A green suite is the expectation. Any failure is a regression.

---

## 6. Most-recent run

`workflow_outputs/elm_run_20260726_155406/` — Naches (HUC 17030002), 13 columns,
NLDAS-2 1995, **warm-started from the CONUS 1-km restarts** (13/13, bands lat11
and lat12, snapped 227–367 m). 13/13 succeeded in 4 m 42 s of SLURM time.

Reception chose the warm start itself: the request asked about recharge, and a
cold 1-year run cannot answer that. Measured on the same column: cold gives
−0.18 mm/yr of recharge with the water table moving 1 mm in a year; warm gives
309 mm/yr.

**What it shows.** Recharge-dominated partitioning, median recharge:runoff ≈ 4,
runoff 0.4–13.5 % of P and recharge 2–70 %. Runoff and recharge both scale with
precipitation (r = 0.82, 0.62). The spatial maps show a coherent west→east
gradient — high recharge fraction along the western edge, near zero in the east,
with ET/P the exact inverse. That is the Cascade rain shadow, and its coherence
says the control is climatic rather than soil noise.

**What it does NOT show.** Magnitudes are transient: five columns drain more
water than falls on them (col_10 at 4.6× P), closure residuals reach ±1500 mm,
and the water table deepens 0.6 m across the year. That is initial CONUS storage
draining, exactly as a warm-started column with no spin-up should behave. The
*direction* is defensible; the *magnitudes* are not, until spin-up.

**Validation** (one figure per observable, each with its own verdict):
hydrograph −66 % cumulative with NSE −0.32 after the 30-day warm-up is excluded;
water yield **context-only** because the gauge drains 7.1 % of the basin and only
1 of 13 columns falls inside its NLDI catchment; the observed runoff ratio 1.74
is **refused** as physically impossible; SWE is 2–20× low with *no* elevation
gradient where SNOTEL has a steep one; the water-table comparison is
climatological (0 of 14 well measurements fall in 1995).

## 7. Open work

**Wiring gaps:**
- **Standalone PFLOTRAN and the reactive-transport demo are still CLI-only.**
  The *coupled* path is wired (step 4d); recharge-scenario ensembles are not.
- **No auto-carrier is needed any more**, but a multi-year **spin-up** still is:
  every magnitude in a 1-year run is transient (see §6).

**Validation quality — addressed, with one input still missing:**
- Metric units throughout; per-observable figures with individual verdicts;
  warm-up exclusion; area weighting; NLDI catchment restriction; runoff ratio.
- Observation coverage is now checked **before** the run, not after: Reception
  calls `get_streamflow_availability` and writes
  `run_settings.observations` + a conflict line. Verified live on the Naches —
  "only 1 of 93 stream gauges has records in 1995, draining ~7% of the basin;
  streamflow comparison will be context-only" now appears in the brief.
- **Still blocking a real partitioning verdict: precipitation over the GAUGED
  catchment.** Without it the observed ratio is computed against the basin mean
  and comes out impossible (1.74). Either an in-catchment precipitation product
  or a gauge whose catchment the sampling actually covers.

**Science:**
- The near-zero recharge across the cold ensemble was a **cold-start artifact**,
  not physics — demonstrated, not assumed. Validation run 768895 (col_02, 1995,
  same column, warm-started from CONUS):

  | term | warm | cold |
  |---|---:|---:|
  | recharge | 308.6 | −0.18 mm/yr |
  | drainage | 431.2 | 0.004 mm/yr |
  | ZWT mean → end | 4.97 → 5.33 | 8.802 → 8.803 m |

  The cold run's water table moved 1 mm in a year: it started at ELM's default
  depth with an empty aquifer and never equilibrated, pinning QCHARGE at zero.
  Re-run the ensemble warm before drawing any recharge conclusion. Only one
  column has been checked so far.
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
