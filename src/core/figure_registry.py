#!/usr/bin/env python3
"""The figures the Analyzer may ask for, in one place.

The Analyzer is agentic — it decides which figures answer the user's question
rather than rendering a fixed set. This registry is what it chooses FROM, and
what it emits is a plot SPECIFICATION (which figure, why), never Python.

The split is the same one the Planner lives under, for the same reason: the LLM
decides WHAT, deterministic code decides HOW. A figure rendered by
LLM-authored code that silently plotted the wrong variable, or mislabelled a
unit, would look exactly as authoritative as a correct one and nothing
downstream would catch it.

Provenance travels with every figure:

    registry     rendered by a vetted function in this table
    authored     rendered by analyzer-supplied code (marked, never silent)

so a figure heading for a manuscript declares which kind it is.

Each entry declares:
    fn        module-qualified renderer, resolved lazily (tools/ are scripts)
    needs     what must exist for it to render at all
    question  what the figure ANSWERS — this is what the LLM selects on
"""
from typing import Any, Dict, List

# What a figure can require. The Analyzer is told which of these a run has, so
# it cannot request a figure the data cannot support.
CAPABILITIES = (
    "water_budget",      # per-column budget terms (fractions of P)
    "soil",              # per-column soil predictors (clay, Ksat)
    "coordinates",       # lat/lon per column  -> anything spatial
    "boundary",          # watershed polygon in columns.json
    "observations",      # validation.json exists
    "hydrograph",        # a gauge with daily records in the simulated year
    "swe_obs",           # SNOTEL peak SWE
    "wells",             # USGS wells with records
    "timeseries",        # surviving history files (context series)
)

REGISTRY: Dict[str, Dict[str, Any]] = {
    "partitioning": {
        "tool": "analyze_run", "fn": "plot_partitioning",
        "args": ("results",),
        "needs": ("water_budget",),
        "question": "What fraction of precipitation becomes runoff, ET, drainage "
                    "and storage in each column, and which columns are not at "
                    "equilibrium?",
    },
    "controls": {
        "tool": "analyze_run", "fn": "plot_controls",
        "args": ("results",),
        "needs": ("water_budget",),
        "question": "What drives the partitioning — elevation, precipitation or "
                    "soil? Shows the orographic confound rather than asserting it.",
    },
    "spatial": {
        "tool": "analyze_run", "fn": "plot_spatial",
        "args": ("results", "run_dir"),
        "needs": ("water_budget", "coordinates"),
        "question": "How are the partitioning fractions distributed ACROSS the "
                    "watershed — is a pattern spatially coherent or scattered?",
    },
    "soil_control": {
        "tool": "analyze_run", "fn": "plot_soil",
        "args": ("soil",),
        "needs": ("soil",),
        "question": "Holding forcing constant, does soil explain the differences "
                    "between columns?",
    },
    "wtd_columns": {
        "tool": "analyze_run", "fn": "plot_wtd",
        "args": ("results", "run_dir"),
        "needs": (),
        "question": "How deep is the modelled water table in each column and how "
                    "did it evolve?",
    },
    "validation_hydrograph": {
        "tool": "validate_run", "fn": "plot_hydrograph",
        "args": ("validation",),
        "needs": ("observations", "hydrograph"),
        "question": "Does modelled streamflow match a gauge — in shape, and in "
                    "cumulative volume (the verdict a routing-free column can "
                    "fairly be held to)?",
    },
    "validation_yield": {
        "tool": "validate_run", "fn": "plot_yield",
        "args": ("validation",),
        "needs": ("observations",),
        "question": "How does modelled water yield compare with gauged specific "
                    "discharge, and is the runoff RATIO defensible when absolute "
                    "yield is not?",
    },
    "validation_water_table": {
        "tool": "validate_run", "fn": "plot_water_table",
        "args": ("validation",),
        "needs": ("observations", "wells"),
        "question": "Does the modelled water table agree with the Fan prior and "
                    "with observed wells?",
    },
    "validation_swe": {
        "tool": "validate_run", "fn": "plot_swe",
        "args": ("validation",),
        "needs": ("observations", "swe_obs"),
        "question": "Does modelled snowpack match SNOTEL, and does the model "
                    "reproduce the observed SWE-elevation gradient?",
    },
    "validation_context": {
        "tool": "validate_run", "fn": "plot_context",
        "args": ("validation",),
        "needs": ("observations", "timeseries"),
        "question": "What is the seasonal story behind the annual numbers — when "
                    "does snow melt and when do the fluxes respond? NOT scored.",
    },
}


def available(run_capabilities) -> List[str]:
    """Figure names this run can actually render, given what it has."""
    caps = set(run_capabilities or ())
    return [k for k, v in REGISTRY.items() if set(v["needs"]) <= caps]


def detect_capabilities(results: Dict[str, Any], validation: Dict[str, Any],
                        columns_meta: Dict[str, Any]) -> List[str]:
    """What this run supports, read from the artifacts rather than assumed.

    The Analyzer is handed this list so it can only request figures the data can
    support — the analysis-side equivalent of the Planner's feasibility verdict.
    """
    caps = []
    ok = [r for r in (results or {}).values() if r.get("status") == "ok"]
    if any((r["metrics"].get("water_budget") or {}) for r in ok):
        caps.append("water_budget")
    # SOIL COMES FROM soil_profile, the form the data actually arrives in.
    # This tested `r["soil"]["clay_max_pct"]` — a precomputed scalar nothing
    # ever wrote. Measured across every packaged run on disk, `soil` was null
    # on 100% of columns while `soil_profile` beside it was complete, so the
    # soil capability read as absent on every run and the figures that need it
    # were never offered. `soil` stopped being a row field on 2026-08-13.
    if any(((r.get("soil_profile") or {}).get("layers")) for r in ok):
        caps.append("soil")
    if any(r.get("lat") is not None and r.get("lon") is not None for r in ok):
        caps.append("coordinates")
    if (columns_meta or {}).get("boundary"):
        caps.append("boundary")
    v = validation or {}
    if v:
        caps.append("observations")
    if (v.get("hydrograph") or {}).get("obs_mm_day"):
        caps.append("hydrograph")
    if v.get("swe_pairs") or v.get("swe_context"):
        caps.append("swe_obs")
    if any((w.get("points") or w.get("wtd_m") is not None)
           for w in (v.get("well_series") or [])):
        caps.append("wells")
    if v.get("context_series"):
        caps.append("timeseries")
    return caps


def catalogue(caps) -> str:
    """The menu handed to the Analyzer: name -> what it answers."""
    lines = []
    for name in available(caps):
        lines.append(f"  {name}: {REGISTRY[name]['question']}")
    return "\n".join(lines) or "  (none — the run has no renderable figures)"
