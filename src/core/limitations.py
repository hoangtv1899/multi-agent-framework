#!/usr/bin/env python3
"""
Physical-limitations catalog — the honesty that travels WITH every result.

Two kinds, kept distinct because they mean different things to a reader:
  structural      can NEVER be fixed within this model class (1-D columns);
                  reruns don't help — only a different model structure does
  configuration   an artifact of how THIS run was configured; fixable by
                  rerunning (spin-up, forcing resolution, domain depth, years)

select_limitations() picks the entries relevant to a given run so the
analyzer can attach them to hydro_summary.json, the interpreter must
acknowledge them in its Trust section, and the deck renders them beside the
results they qualify. Sourced from the same limits the planner reasons with —
this module makes them mandatory at RESULT time, not just design time.
"""

# ── structural: properties of 1-D columns, independent of configuration ─────
STRUCTURAL = [
    {"applies_to": "runoff (QOVER)",
     "caveat": "local runoff GENERATION on a 1 m² column — not routed discharge; "
               "no downslope concentration, so it is not comparable to a stream "
               "gauge without aggregation/routing."},
    {"applies_to": "runoff/infiltration partitioning",
     "caveat": "no run-on: a column never receives runoff generated uphill, which "
               "in reality re-infiltrates downslope — runoff is likely biased high "
               "on slopes and infiltration biased low in valleys."},
    {"applies_to": "water table (ZWT/WTD)",
     "caveat": "1-D vertical equilibrium only — the lateral groundwater convergence "
               "that sets real valley water tables is absent; ELM parameterizes "
               "lateral losses (QDRAI) as a local sink but routes nothing between "
               "columns."},
    {"applies_to": "streamflow validation",
     "caveat": "columns provide no native integrated streamflow; gauge comparisons "
               "use unweighted column means (specific discharge) and are labeled "
               "context-only until routing exists."},
    {"applies_to": "snow (SWE) at stations",
     "caveat": "forcing is a 12 km cell average — ridgetop SNOTEL stations sample "
               "terrain the cell cannot resolve; systematic underestimation at "
               "high-relief stations is expected."},
]

# ── configuration: artifacts of this run's setup, fixable by rerunning ──────
CONFIGURATION = [
    {"key": "no_spinup",
     "applies_to": "recharge, storage change, deep soil moisture",
     "caveat": "run starts from non-equilibrated storage (no multi-year spin-up): "
               "slow-state fluxes are transient, storage-change terms are large, "
               "and recharge magnitude/sign is initialization-dependent."},
    {"key": "warm_start_fan",
     "applies_to": "water table, recharge",
     "caveat": "initial water table prescribed from the Fan (2013) equilibrium "
               "prior; ELM's own 1-D equilibrium differs, so early-run drift "
               "reflects model-vs-prior disagreement. Note the prior is also used "
               "as a validation reference — a circularity to keep in view."},
    {"key": "warm_start_conus",
     "applies_to": "recharge, storage change, deep soil moisture",
     "caveat": "storage is inherited from the spun-up CONUS 1-km restart at the "
               "donor gridcell, and the run keeps that gridcell's soil, so the "
               "initial state is consistent with its own hydraulics. It is NOT "
               "equilibrated to THIS run's forcing period, so some adjustment "
               "remains — read the annual water-balance closure to size it "
               "rather than assuming it is either negligible or fatal."},
    {"key": "warm_start_soil_mismatch",
     "applies_to": "recharge, drainage, storage change",
     "caveat": "the run inherits CONUS storage but OVERRIDES the donor's soil, "
               "so the initial moisture is inconsistent with the hydraulics it "
               "is handed and year one is spent relaxing, not simulating. "
               "Measured on a 14-column Naches run with this configuration: "
               "five columns drained more than their annual precipitation (one "
               "at 2.98x) and closure residuals reached 2491 mm. Treat drainage, "
               "recharge and storage terms as unusable; set soil_source='conus' "
               "or add spin-up years."},
    {"key": "single_year",
     "applies_to": "all annual results",
     "caveat": "a single simulated year; interannual variability and the year's "
               "climatic percentile are not characterized."},
    {"key": "forcing_12km",
     "applies_to": "elevation-dependent results",
     "caveat": "NLDAS-2 resolves orographic gradients at ~12 km but not sub-cell "
               "terrain (no within-cell lapse rate)."},
    {"key": "forcing_coarse",
     "applies_to": "all forcing-driven results",
     "caveat": "Qian T62 (~1.9°) forcing is not elevation-resolved: columns at "
               "different elevations can share one forcing cell."},
]


def select_limitations(n_years: int = 1,
                       warm_start: bool = False,
                       forcing: str = "nldas",
                       spinup_years: int = 0,
                       warm_source: str = None,
                       soil_source: str = None) -> dict:
    """Pick the catalog entries that apply to a run configuration.

    Initialization is ONE state, not a checklist. "No spin-up" and "warm
    started from a spun-up restart" are not both true, and emitting both left
    the analyzer recommending three to five spin-up years on a run that had
    inherited an equilibrated state precisely to avoid them — a recommendation
    that is physically sensible in general and wrong for the run in hand.

    What matters is whether the inherited storage is consistent with the soil
    the run integrates:

      cold                       -> non-equilibrated, the original caveat
      warm from CONUS, CONUS soil-> consistent; residual adjustment only
      warm from CONUS, other soil-> inconsistent; the terms are unusable
      warm from Fan              -> prior-vs-model disagreement, and circular
                                    if Fan is also the validation reference
    """
    cfg = []
    pick = lambda k: next(c for c in CONFIGURATION if c["key"] == k)
    warm_conus = bool(warm_start) and (warm_source or "conus") == "conus"

    if spinup_years >= 3:
        pass                                   # genuinely spun up; no caveat
    elif warm_conus and (soil_source or "conus") == "conus":
        cfg.append(pick("warm_start_conus"))
    elif warm_conus:
        cfg.append(pick("warm_start_soil_mismatch"))
    elif not warm_start:
        cfg.append(pick("no_spinup"))

    if warm_start and (warm_source or "") == "fan":
        cfg.append(pick("warm_start_fan"))
    if n_years <= 1:
        cfg.append(next(c for c in CONFIGURATION if c["key"] == "single_year"))
    cfg.append(next(c for c in CONFIGURATION if c["key"] ==
                    ("forcing_12km" if "nldas" in forcing.lower() else "forcing_coarse")))
    return {
        "structural": [dict(x, kind="structural") for x in STRUCTURAL],
        "configuration": [
            {k: v for k, v in dict(x, kind="configuration").items() if k != "key"}
            for x in cfg],
    }
