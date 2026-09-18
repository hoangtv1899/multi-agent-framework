# What ELM-BGC can answer, and what that asks of the framework (2026-09-17)

Context: Xingyuan's direction of 2026-09-17, ELM only, real questions ELM
can answer, a Design Review a person could set the cases up from, and
Huilin Huang's ELM-BGC work in the Naches as the anchor. The literature
sweep below was done by a Sonnet agent (30 searches); the synthesis is
ours. Every study claim carries its URL in the sweep.

## 1. Question families ELM-BGC has answered in print

1. Carbon and water together at a site or watershed: gross primary
   production, respiration, evapotranspiration and soil moisture, and how
   drought or soil hydraulics limit them (MOFLUX, Liang et al. 2019).
   Needs a flux tower; the Naches has none, so carbon fluxes there can be
   compared only with satellite products (MODIS GPP, ET, LAI).
2. Fire and disturbance: burned area (the Li et al. process model, a
   neural surrogate, the XGBoost model coupled online in ELM2.1-XGBfire),
   and what a fire does to vegetation, litter, woody debris and soil
   carbon, and downstream to stream carbon (Huilin's Naches chain with ATS
   and PFLOTRAN after the 2021 Schneider Springs Fire).
3. Snow and mountain hydrology: SWE, snow cover and phenology, runoff, and
   the effect of sub-grid topography on the surface energy balance (Hao et
   al. 2023 and 2022). ELM's known western-US bias: SWE low by 123 mm on
   average, melt-out 27 to 36 days early. Our Naches demo found SWE low by
   44 and 86 mm and melt-out 33 and 38 days early: the same bias,
   reproduced by the framework on two pillows.
4. Vegetation representation: which plant types and traits the model
   needs (arctic PFTs, FATES trait uncertainty), and what changes when the
   land parameters go to 1 km (Li et al. 2024).
5. Nutrient limitation (nitrogen, phosphorus) on carbon uptake, mostly at
   global scale and benchmarked with ILAMB.
6. Outside our region: permafrost runoff, bioenergy crops.

## 2. What those studies say ELM-BGC needs

- A spun-up carbon state: 200 + 200 years at a point, 400 + 400 at CONUS,
  260 + 200 in Alaska (accelerated decomposition then regular). This is
  not something to run per question; it has to come as a restart file
  from an existing spin-up (Huilin's domain, or an E3SM CONUS run).
- Forcing: NLDAS-2 hourly at 1/8 degree is the standard for CONUS; the
  fire study recycles 1981-2000 for spin-up and runs 2001-2020. Our runs
  hold 1979 only; a fire question needs 2019 to 2023 at least.
- Fire or disturbance in the model: the Li et al. scheme (in ELM), the
  XGBoost model (Huilin's, needs a Python bridge), or a prescribed burn
  written into the surface data as a change of vegetation cover and
  leaf area. The prescribed burn is the one the framework can do soonest.
- Known limits to state on any review page: snow biases above; runoff
  needs routing to meet a gauge; a reported GPP underestimate near 30
  percent in ELM-BGC appears in the XGBfire paper's summary but the sweep
  could not find the figure in the text, so quote it only from the paper.

## 3. What the framework can ask now, and what needs building

Fits today (satellite phenology, one year, warm start from the CONUS
spin-up): snow and water-partitioning questions, elevation and aspect
controls, compared with SNOTEL and the gauges. That is demo 1.

Needs building for BGC and fire:
1. a BGC initial state: a restart file for the Naches from Huilin's or an
   E3SM spin-up, and the server's ability to start from it;
2. multi-year forcing (NLDAS-2 through 2023) on Compy and in the server;
3. a disturbance tool: paired burned and unburned columns at matched
   elevation and aspect, the burn severity from MTBS or MODIS, written
   into the surface data as a vegetation change;
4. BGC history variables and their semantics in the extractor (GPP, NPP,
   NEE, LAI, litter and soil carbon), and comparisons against MODIS GPP,
   ET, LAI and NDVI change;
5. the Design Review as an experiment description (section 4).

## 4. Candidate real questions for the Naches, ranked by readiness

1. How did the 2021 Schneider Springs Fire change the split of water
   between evapotranspiration, runoff and drainage on burned hillslopes
   compared with unburned ones at the same elevation, in the two years
   after the fire? (ELM-BGC or SP with a prescribed burn; MODIS ET and
   LAI change, SNOTEL, the two gauges; Huilin's baseline as the check.)
2. Did the fire change snowmelt timing in burned forest, and by how many
   days, against the unburned twin? (SP suffices; SNOTEL and MODIS snow
   cover; the known early-melt bias must be stated.)
3. How much carbon did the burned columns lose from litter, woody debris
   and soil, and how fast did leaf area recover by 2023? (BGC restart
   required; MODIS LAI and NDVI; Huilin's pools for comparison.)
4. Which elevation bands recover leaf area first after fire, and does
   recovery track snow or soil moisture? (BGC; MODIS.)
5. How does the fire's change in vegetation alter recharge to the
   aquifer, the flux demo 1 showed disappearing below the active soil?
   (SP or BGC; no observation; a model-only sensitivity, stated as such.)

The first question is the one to build toward: it reuses demo 1's basin,
columns and comparisons, needs only the disturbance tool and multi-year
forcing, and has Huilin's independent result to compare against.

## 5. Division of labour (Fable economy)

- Hoang: ask Huilin for her Naches case setup (compset, forcing, surface
  data with the burn, restart files, spin-up) and the list of questions
  her study asked; read the two papers.
- A Sonnet agent, or three shell commands: inventory NLDAS-2 forcing
  years on Compy, the compsets the ELM server can build, and whether a
  BGC restart exists for the CONUS 1 km subset.
- Fable: the Design Review v2 page design and renderer (the records are
  already on disk), the disturbance tool's design, and every comparison
  and audit that reaches a slide.

## Appendix: the literature sweep (Sonnet agent, 2026-09-17)

(15 studies with URLs, configurations, observations and findings, as
returned by the agent; see the conversation record of 2026-09-17. The
studies: Liu et al. 2025 GMD ELM2.1-XGBfire1.0; Li, Huang et al. 2026 DOE
Data Explorer Naches DOC archive; Huang et al. 2025 GMD WRF-ELM; Hao et
al. 2023 The Cryosphere western-US snow; Hao et al. 2022 JAMES sub-grid
topography; Liang et al. 2019 GMD MOFLUX; Yuan et al. 2023 JoCS 1 km
Seward Peninsula; Huang et al. 2026 GMD Alaska permafrost runoff; Murphy
et al. 2025 JGR-B arctic PFTs; Liu et al. 2024 Earth's Future ELM-FATES
traits; Zhu et al. 2022 GMD DNN fire surrogate; Sinha et al. 2023 JAMES
bioenergy crops; Li et al. 2024 ESSD 1 km parameters; Zhu et al. 2019
JAMES ELMv1-ECA; Biogeosciences 2023 phosphorus.)
