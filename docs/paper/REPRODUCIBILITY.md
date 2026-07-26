# Reproducibility Record

All numbers in the manuscript trace to committed JSON records in this
repository. This file records the software and data provenance needed to
reproduce the runs.

## Compute environment
- NERSC Perlmutter, CPU nodes (pm-cpu), GNU compilers, project m3780.
- Python environment: `module load pytorch/2.8.0` (Python 3.12.11).
- Key package versions: see `requirements-lock.txt` in this directory.

## Models
- **ELM (E3SM Land Model)**: E3SM checkout at commit `3c13216be8`
  (build tag `1D_ELM.3c13216be8`). Single-column configuration:
  RES=ELMMOS_USRDAT, COMPSET=IELM, 10-layer soil column.
  All cases clone a shared executable (`create_clone --keepexe`).
  > Note (2026-07-23, Compy port): the above records the configuration used
  > for the published runs. The Compy port switched to an explicitly
  > river-free configuration — RES=ELM_USRDAT,
  > COMPSET=2000_DATM%QIA_ELM%SP_SICE_SOCN_SROF_SGLC_SWAV (stub river),
  > E3SM commit `b198763`. This is scientifically equivalent, not a change
  > in physics: the published config already set do_rtm=.false., and both
  > yield rof_present=.false. with zero flood/volr/supply/deficit into ELM.
  > DATM%QIA (DATM_MODE=CLM_QIAN) is unchanged, so forcing is identical.
- **PFLOTRAN**: user build at commit `7d24fd57c`, PETSc 3.24.
  1-D Richards mode, direct solver.

## Forcing and data products
- **NLDAS-2** met forcing, 0.125 degree, from the E3SM inputdata store:
  `/global/cfs/cdirs/e3sm/inputdata/atm/datm7/atm_forcing.datm7.NLDAS2.0.125d.v1`
  (years used: 1995; availability 1980-2018).
- **Fan et al. (2013)** equilibrium water-table depth: local tile served by
  the `fan_wtd` MCP server; used for warm-start initial conditions and as a
  validation reference.
- Live services accessed through read-only MCP servers (access window
  June-July 2026): USGS 3DEP (terrain), USGS WBD (watershed boundaries),
  USDA SSURGO (soils), USGS Water Data OGC API (wells, gauges),
  NRCS AWDB (SNOTEL). Live services are not frozen; per-study JSON records
  preserve the values actually used.

## Determinism
- Tier 2 and Tier 3 (sampling materialization, case construction, analysis)
  are deterministic given identical data-server responses. The only random
  number use is a fixed-seed jitter in one plot.
- Tier 1 (LLM planning) is controlled where the gateway allows it:
  temperature 0, seed 1995, max_tokens 8192, and the provider-reported
  model version is logged in every evaluation record.
- LLM models used: planner `claude-sonnet-4-5-20250929-v1-project`
  (production and evaluation headline); sensitivity wave
  `claude-opus-4-8-project`; served via the PNNL AI gateway.

## Per-study records
Each study directory under `workflow_outputs/` persists:
`reception_brief.json`, `plan.json`, `run_plan.json` (with the assumptions
ledger), `columns.json`, `cases.json`, `forcing.txt`, `assumptions.json`,
`04_analysis/hydro_summary.json` (with the limitations catalog),
`04_analysis/validation.json`, and `interpretation.md`.

## Evaluation records
`eval/` contains the frozen pre-registered suite, the runner and scorer,
all raw LLM outputs (main wave and Opus sensitivity wave), scores, and the
human adjudication records. The pre-registration freeze commit is
`fcd9793`; all main-wave data postdates it.

## To do before submission
- Public code + data archive with DOI (Zenodo or equivalent).
- Exact dataset version stamps for SSURGO and 3DEP retrievals.

## Planner prompt versioning
- `planner_capability_probe.txt` (v0.1) is FROZEN at eval freeze commit
  `fcd9793`. The evaluation loads it by name for the ablation surgeries, and
  the A4 framework arm pins it explicitly, so eval re-runs are byte-identical.
- Production (both ELM and PFLOTRAN) uses `planner_capability_probe_v2.txt`,
  which extends the capability inventory to PFLOTRAN. The PFLOTRAN extension
  postdates the eval and is not separately evaluated.
