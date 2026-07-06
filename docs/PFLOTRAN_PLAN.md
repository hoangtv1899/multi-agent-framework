# PFLOTRAN adaptation plan

**Verdict up front:** the adaptation is genuinely easier than ELM was — no CIME
cases, no compsets, no forcing streams, no surface NetCDF, no batch-queue
pressure (1-D RICHARDS columns run serial in seconds–minutes). Every hard
piece already exists in the repo or on disk. What deserves the careful thought
is not the plumbing but the **science plumbing**: boundary conditions, domain
depth, and what question each phase answers.

## What already exists (verified 2026-07-06)

| piece | where | state |
|---|---|---|
| PFLOTRAN executable | `~/petsc/pflotran/src/pflotran/bin/pflotran` | runs, PETSc 3.24 |
| Input-deck builder | `src/core/pflotran_input_agent.py` (531 ln) | RICHARDS mode, van Genuchten, materials/strata/regions/BCs, serial runner, plotting |
| Soil → hydraulic props | `src/core/elm_soildata.py` | pedotransfer: sand/clay% → porosity, permeability, van Genuchten α/m |
| Execution skeleton | `src/core/exp_manager.py` | plan → build → run → analyze → LLM input (legacy, ELM-era interface) |
| Per-column sites + soil | `columns.json` (manager step 1) | **reused unchanged** — materialization is model-agnostic |
| Initial water table | Fan 2013 per column (ParFlow-CONUS2 wired) | becomes PFLOTRAN's hydrostatic IC / bottom BC |
| Top-boundary forcing | completed ELM runs (cases #5/#6): QINFL per column, 3-hourly | becomes PFLOTRAN's transient infiltration flux |

## The science frame

ELM stops at 3.8 m + a bucket aquifer; its "recharge" banks into storage and
the water table barely lives. PFLOTRAN picks up exactly there: a 1-D
variably-saturated (RICHARDS) column from the surface to below the real water
table answers **"when and how does today's infiltration become recharge at
the water table?"** — travel times, lags, wetting fronts — per column across
the elevation/soil gradient. ELM does the surface partitioning; PFLOTRAN does
the deep fate. That is the coupling archetype the reception/planner have
carried since the beginning.

## Phase A — standalone PFLOTRAN ensembles behind the same manager contract

Deliverable: `columns.json → per-column 1-D PFLOTRAN runs → results package`.

1. **`tools/build_pflotran_cases.py`** (parallel to `build_cases.py`):
   - domain: 1-D vertical column; depth per column = Fan WTD + buffer
     (capped — see decision 3); fine layers near surface, coarsening down
   - materials: SSURGO layers through the existing pedotransfer (top ~2 m),
     extrapolated substrate below (same substrate concept as the ELM surfaces)
   - initial condition: hydrostatic profile pinned at the Fan/ParFlow WTD
     (the warm-start prior, reused)
   - top BC: infiltration flux; bottom BC: see decision 1
   - outputs: observation points (pressure/saturation), mass balance
2. **Runner**: serial, all N columns concurrently on a login node or one tiny
   debug job — effectively free (this is the "simpler model" dividend)
3. **`tools/analyze_pflotran_run.py`**: HDF5/observation output → per-column
   deep recharge, water-table response, saturation profiles — emitted in the
   hydro_summary schema so the driver-matrix / deck machinery is reused as-is
4. **Tests**: offline golden test on generated deck text + pedotransfer
   sanity; one-column live smoke test

## Phase B — one-way ELM→PFLOTRAN coupling (the payoff)

- extract each column's transient QINFL from the completed NLDAS runs
  (daily means from the 3-hourly history — the expensive runs keep giving)
- feed it as PFLOTRAN's transient top flux (FLOW_CONDITION ... LIST)
- run the year; report **infiltration→water-table travel time per column**,
  the recharge signal ELM's bucket cannot represent
- consistency check: PFLOTRAN deep recharge vs ELM QCHARGE; validation:
  water-table response vs USGS wells

## Phase C — framework integration

- planner capability text: PFLOTRAN + ELM→PFLOTRAN coupling AVAILABLE;
  the "coupling" archetype becomes answerable
- deck: PFLOTRAN case-study slides (surface partitioning + deep fate story)
- optional later: two-way coupling (PFLOTRAN WTD back into ELM) — real work,
  not in scope until one-way proves out

## Decision points (user input wanted)

1. **Bottom BC** — recommend: hydrostatic pressure pinned at the Fan WTD
   (uses the prior; lets us measure model-vs-prior drift like the warm-start
   study). Alternative: free drainage (simpler, but loses the water table).
2. **Phase A forcing** — recommend: skip synthetic steady forcing and go
   straight to ELM transient QINFL (Phase A+B merge); use a steady-state
   solve only to spin the initial profile.
3. **Domain depth cap** — Fan says up to ~215 m at ridge columns; recommend
   capping at ~50 m (runtime + pedotransfer credibility) and reporting the
   cap honestly, mirroring the ELM warm-start clamp.

## Honest unknowns / risks

- legacy code predates the spatial-columns pivot — expect small interface
  fixes (exp_manager consumed the old single-site plan shape)
- deep-column van Genuchten parameters come from a texture pedotransfer
  extrapolated well below SSURGO's ~2 m — a stated approximation, not a bug
- very dry ridge columns (WTD at cap) may need solver-tolerance tuning —
  classic Richards stiffness; mitigated by daily (not 3-hourly) forcing
