# Loose vs tight ELM–PFLOTRAN coupling — what is exchanged, and when each is the right tool

*One page for the report. Every claim below carries its source; where a source
could only be summarized, the caveats at the bottom say so.*

## The two designs, one sentence each

**Tight (Bisht's line of work):** PFLOTRAN's Richards solve *replaces* the land
model's own soil hydrology inside one compiled executable, exchanging
layer-resolved states and fluxes through an in-memory PETSc interface every
model timestep (Bisht et al. 2017, GMD 10:4539, Sect. 2.3; 1 h in the ELM
adaptation, Zhu et al. 2024, Hydrol. Process., 10.1002/hyp.15230).

**Loose (this framework):** both models run whole and stock, each behind its
own MCP server; complete runs exchange files — a year of daily drainage
forward, a solved water table back — iterating until the water table stops
moving between rounds (`tools/coupling_loop.sh`, `tools/coupling_delta.py`).

## What actually travels

| | tight (per timestep, per layer) | loose (per round = one simulated year) |
|---|---|---|
| land → subsurface | soil temperature, liquid and ice water per layer (`t_soisno`, `h2osoi_liq`, `h2osoi_ice`); infiltration and layer-resolved ET as mass fluxes (`mflx_infl`, `mflx_et`, kg/s) — EMI L2E list, `ExternalModelPFLOTRAN.F90` ll. 304–388 | the year's daily drainage series (QDRAI, mm/day) as the column's top boundary; ELM's solved water table as the hydrostatic anchor |
| subsurface → land | soil liquid/ice per layer, water-table depth (`zwt`), soil pressure, aquifer recharge (`qcharge`) — EMI E2L list, ll. 391–452; **drainage returned as zeros** and ELM's own drainage terms zeroed, so that water stays in PFLOTRAN's domain (ll. 1310–1315) | PFLOTRAN's solved water-table depth, stamped into each column's next warm start (`set_water_table.apply`) |
| balance guarantee | ELM's water-balance check **disabled** when PFLOTRAN is on (commit 60329910cf) | each model's own closure untouched; every exchange is a file with provenance |

## What the evidence says each buys

- **Tight coupling matters when the water table is shallow.** Bisht 2017:
  coupling and resolution "become critical" with groundwater within **6–7 m of
  the surface**; under a +5 m river stage, coarse grids err 33 % in latent heat
  vs 1.4–2.4 % otherwise (Table 4). Zhu 2024: ~**75 %** of lateral river input
  converts to added ET, concentrated within ~1 km of the channel.
- **Standalone ELM-family groundwater is wrong at field scale.** Bisht 2017:
  CLM4.5 alone put the water table 35–40 m below surface where wells show
  5–10 m (Fig. 12a) — the catchment-scale scheme "does not apply on the field
  scale". Our record agrees from the other side: ELM handed the Brandywine
  chain 46.8 m water tables that PFLOTRAN corrected by **18.5 m in one round**,
  and 10 of 16 Naches columns drain exactly zero below ELM's 3.8 m active soil.
- **Where the water table is deep, loose is enough.** Bisht 2017's own
  vertical-only run differs from full 3-D by 5.7 % in latent heat ("lateral
  flow is less important when the water table is deep") — precisely the regime
  where our round-by-round exchange of the slow variable captures what matters,
  at zero model-fork cost.

## Cost of tight, today

The published line is a fork: 14 commits atop a **2019** E3SM master (branch
`bishtgautam/lnd/elm-pflotran-coupled-cedb30ce4`), PFLOTRAN as `libpflotran.a`
(`src/clm-pflotran/pflotran_model.F90`), the upstream org dormant since
~2019–2020. The living continuation is yixiao2's port: PFLOTRAN 4.0-dev
(Dec 2023) + an E3SM 2024 snapshot, hard-requiring **PETSc ≥ 3.20** — which
Compy's PFLOTRAN v7.0 stack (PETSc 3.21.6) already satisfies. Its reported
NERSC failure is not documented in the repo (caveat below).

## The position this framework takes

The information flows **PFLOTRAN → ELM**: the subsurface model owns the
groundwater physics, the land model supplies the surface flux. Our loop
delivers that direction cheaply — stock models, measured convergence, honest
records — and is the right tool for deep-water-table regimes and
starting-state repair. Shallow-water-table questions (riparian ET, river
corridors) genuinely need the per-timestep feedback only tight coupling
provides; the clean way in, here, is the coupled executable as **one more
model behind its own MCP server** — the framework never needs to know it is
two models inside. The yixiao2 port is the closest starting point.

## Citation caveats

The GMD paper never prints its coupling-step length — do not cite a number
for it (Zhu 2024 states 1 h). Zhu 2024 was read through a summarizer; its
Results numbers rest on that reading, not verbatim text. "Ran on a desktop,
failed with PETSc errors on NERSC" for the port is colleague testimony, not
repository record. EMI variable lists were verified on the ELM side only;
the PFLOTRAN-side CPv1.0 library mirrors them but was read separately
(`pflotran-clm-trunk`: `qflx` source/sink in, `sat`/`sat_ice`/`temp` back).

**Sources:** gmd.copernicus.org/articles/10/4539/2017 ·
doi.org/10.1002/hyp.15230 · github.com/CLM-PFLOTRAN/E3SM (branch above) ·
github.com/CLM-PFLOTRAN/pflotran-clm-trunk ·
github.com/yixiao2/elm-pflotran-eh (`pflotran-dev_elm_emi`)
