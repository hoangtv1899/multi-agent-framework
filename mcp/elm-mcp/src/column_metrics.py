#!/usr/bin/env python3
"""What one column's series adds up to. DOWNSTREAM OF extract.py, ALWAYS.

    in   one column's extracted block — {dates, variables: {VAR: {values, …}}}
    out  {stats: {VAR: …}, metrics: {…}}

THE RULE THIS FILE EXISTS TO KEEP (2026-08-13, the user's): every computing
script must be downstream from the extraction. Not "may be" — must. These
numbers used to be computed inside `ELMResultsAnalyzer` during the same pass
that read the NetCDF, from the raw 3-hourly array, and that made them
**unverifiable**: checking one meant re-opening history files on `/compyfs`,
which is purgeable scratch. The run directory outlives the scratch. So a metric
computed from the open dataset stops being checkable the moment the filesystem
is cleaned, while a metric computed from `extracted.json` can be recomputed by
anyone holding the run directory, for as long as the study exists.

That is the whole argument, and it is worth more than the 0.2% it appeared to
cost. It turned out to cost nothing: the gap between the raw pass and the
published series was one partial day, `extract._drop_partial_days` removes it,
and the two now agree to six decimal places.

WHY NOT INSIDE extract.py. Because that file states, at the top, that it derives
nothing — and it is right to. `recharge_fraction` is the recharge-vs-runoff
split rather than a fraction of precipitation, and reading it the obvious way
was wrong by P/(QCHARGE+QOVER). A raw series means what its name says; a ratio
means what its denominator says. Keeping the two in separate files is what stops
the second kind quietly acquiring the authority of the first.

PURE. No xarray, no NetCDF, no I/O, no clock. Give it the same block twice and
it returns the same numbers twice, which is what makes it testable without a
model run.

WHAT BEING DOWNSTREAM COSTS, MEASURED (Brandywine col_01, against the raw
3-hourly array with the same trims applied):

  * MEANS agree to four significant figures — worst flux difference 0.0026%,
    and it is not an approximation of the physics, it is that extract publishes
    each daily value at four significant figures. Computing from what was
    published means computing from what was rounded, which is the honest
    version and is the point.
  * EXTREMA MOVE, and by more: peak SWE 67.59 -> 67.50 mm, TWS seasonal range
    379.89 -> 378.00 mm. A maximum over daily means is not a maximum over
    3-hourly values. This is a definitional change and a deliberate one — the
    comparison package reads the same daily series, so `peak_swe_modelled_mm`
    and the peak swe.py compares against a snow pillow are now the SAME
    reduction of the SAME numbers. They were not before.

UNITS. Fluxes arrive as mm/day — extract has already applied the 86400 — and
leave as mm/yr. `TWS`, `H2OSNO` and `ZWT` arrive in their own units and stay.

ANNUALISATION IS AN EXTRAPOLATION, and the record says so. A mean daily rate
times 365.25 states what a full year at this rate would deliver; the run itself
is shorter, because the start-up transient and any partial day have been
trimmed — a fortnight off a warm start, a year off a cold one, which is why the
factor has to be read off the record rather than assumed. `n_days_in_record`
travels beside every annual value so the
factor is visible rather than implied — a 352-day record is a 1.038x
extrapolation, which is larger than most of the differences anyone argues about.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from extract import FLUX_VARIABLES, VARIABLE_UNITS

# Days per year, matching the calendar ELM's forcing is built on. One constant,
# because two spellings of "a year" is how two metrics stop being comparable.
DAYS_PER_YEAR = 365.25


# ── small numerics, nan-safe and list-shaped ────────────────────────────────
# extract.py already wrote None for every non-finite value, so a gap is a None
# and never a NaN. These skip them rather than propagating, and return None from
# an empty series — "not computed" and "computed as zero" must stay distinct.
def _vals(v: Optional[List]) -> List[float]:
    return [x for x in (v or []) if x is not None]


def _mean(v: List[float]) -> Optional[float]:
    return sum(v) / len(v) if v else None


def _std(v: List[float]) -> Optional[float]:
    m = _mean(v)
    if m is None or len(v) < 2:
        return 0.0 if v else None
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def _r(x: Optional[float], n: int) -> Optional[float]:
    return None if x is None else round(x, n)


def variable_stats(var: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """One variable's summary, from its daily series.

    Shaped per variable rather than uniformly, because the useful reduction
    differs: a water table wants its range and its endpoints, a snowpack wants
    its peak, a storage term wants how much it changed. A single mean/std/min/max
    for all of them would be uniform and useless.
    """
    values = block.get("values") or []
    layered = bool(values) and isinstance(values[0], list)
    units = block.get("units") or VARIABLE_UNITS.get(var, "unknown")

    if layered:
        # Per-layer means over time, then the column total. `n_layers` and the
        # geometry stay on the extracted block; this is only the reduction.
        n_layers = block.get("n_layers") or (len(values[0]) if values else 0)
        layer_means = []
        for i in range(n_layers):
            col = _vals([row[i] if i < len(row) else None for row in values])
            layer_means.append(_mean(col))
        got = [m for m in layer_means if m is not None]
        return {
            "units": units,
            "total_column_kg_m2": _r(sum(got), 4) if got else None,
            "layer_means_kg_m2": [_r(m, 4) for m in layer_means],
            "n_layers": int(n_layers),
            "n_days": len(values),
        }

    v = _vals(values)
    if not v:
        return {"units": units, "n_days": 0}

    if var in FLUX_VARIABLES:
        # mm/day in, mm/yr out. The annual figure is a RATE stated per year,
        # not the total the run produced — see the module docstring.
        return {
            "units_raw": units,
            "units_annual": "mm/year",
            "annual_mean": _r(_mean(v) * DAYS_PER_YEAR, 4),
            "annual_std": _r((_std(v) or 0.0) * DAYS_PER_YEAR, 4),
            "annual_min": _r(min(v) * DAYS_PER_YEAR, 4),
            "annual_max": _r(max(v) * DAYS_PER_YEAR, 4),
            "total_mm_over_record": _r(sum(v), 4),
            "n_days": len(v),
        }

    if var == "TWS":
        return {
            "units": units,
            "mean_mm": _r(_mean(v), 4), "std_mm": _r(_std(v), 4),
            "min_mm": _r(min(v), 4), "max_mm": _r(max(v), 4),
            "seasonal_range": _r(max(v) - min(v), 4),
            # storage change over the run (last - first) — closes the budget
            "delta_mm": _r(v[-1] - v[0], 4),
            "n_days": len(v),
        }

    if var == "ZWT":
        return {
            "units": units,
            "mean_m": _r(_mean(v), 4), "min_m": _r(min(v), 4),
            "max_m": _r(max(v), 4),
            # initial vs final expose the cold-start problem: all columns
            # begin at ELM's default (~8.8 m) regardless of the real WTD
            "first_m": _r(v[0], 4), "last_m": _r(v[-1], 4),
            "n_days": len(v),
        }

    if var == "H2OSNO":
        return {
            "units": units,
            # MODELLED peak, and the name has to say so: `peak_swe_mm` also
            # names the OBSERVED peak at a SNOTEL station in data_gather and
            # the snotel server, and both can appear in one analysis.
            "peak_swe_modelled_mm": _r(max(v), 1),
            "mean_swe_modelled_mm": _r(_mean(v), 1),
            "n_days": len(v),
        }

    return {"units": units, "mean": _r(_mean(v), 6), "std": _r(_std(v), 6),
            "min": _r(min(v), 6), "max": _r(max(v), 6), "n_days": len(v)}


def column_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """One column's stats and derived metrics, from its extracted block.

    `data` is what extract.extract_column returns: {dates, variables}. Nothing
    else is consulted — no run directory, no case directory, no NetCDF.
    """
    variables = (data or {}).get("variables") or {}
    stats = {v: variable_stats(v, blk) for v, blk in variables.items()
             if isinstance(blk, dict)}
    n_days = len((data or {}).get("dates") or [])

    def annual(key) -> Optional[float]:
        return (stats.get(key) or {}).get("annual_mean")

    m: Dict[str, Any] = {}
    if n_days:
        # The denominator of every annualisation above, stated once.
        m["n_days_in_record"] = n_days

    # `annual_recharge_mm_yr` and `annual_runoff_mm_yr` ARE GONE (2026-08-13).
    # They duplicated water_budget.recharge_mm_yr and .runoff_mm_yr at a
    # different rounding — 140.2771 beside 140.3 for the same column, the same
    # day, in the same dict. One quantity, one name, and the budget's is the
    # one that sits beside the terms it has to balance against.
    qc, qo = annual("QCHARGE"), annual("QOVER")

    rain, snow = annual("RAIN"), annual("SNOW")
    if rain is not None or snow is not None:
        # PRECIPITATION IS RAIN + SNOW. This was RAIN alone, which in a
        # snow-dominated basin is not a rounding error: across the 2020 Naches
        # columns the two differ by 1.11x in the warm valley and 2.17x at
        # elevation — the ratio IS the snow fraction. Every runoff/P and
        # recharge/P fraction built on it was inflated by exactly that much.
        m["precip_mm_yr"] = round((rain or 0.0) + (snow or 0.0), 1)
        if rain is not None:
            m["rainfall_mm_yr"] = round(rain, 1)
        if snow is not None:
            m["snowfall_mm_yr"] = round(snow, 1)

    # THREE RATIOS DELETED 2026-08-13, and each for its own reason.
    #
    #   recharge_to_runoff_ratio  QCHARGE/QOVER, guarded only against exactly
    #                             zero. Naches col_10 — the column with no
    #                             runoff at all — carried 4,288,570; col_02
    #                             carried nan. It is also just
    #                             recharge_fraction/(1 - recharge_fraction).
    #   recharge_fraction         QCHARGE/(QCHARGE+QOVER), and the name does
    #   runoff_fraction           not say so. col_01 reported runoff_fraction
    #                             0.508 beside runoff_frac_of_P 0.133 — same
    #                             column, same run, two denominators, two
    #                             names that look interchangeable. A redundant
    #                             pair too: they sum to 1 by construction, even
    #                             when both terms are ~0.
    #
    # The water budget below answers the same questions with the denominator IN
    # THE NAME. What is lost is the recharge-vs-runoff SPLIT as a single
    # number; it is recoverable from the two `_frac_of_P` terms, and anyone who
    # recovers it will have had to name the denominator to do so.

    tws = stats.get("TWS") or {}
    if tws.get("seasonal_range") is not None:
        m["tws_seasonal_range_mm"] = tws["seasonal_range"]
    zwt = stats.get("ZWT") or {}
    if zwt.get("mean_m") is not None:
        m["water_table_depth_m"] = zwt["mean_m"]
    snow_stats = stats.get("H2OSNO") or {}
    if snow_stats.get("peak_swe_modelled_mm") is not None:
        # NOT `peak_swe_mm` (2026-08-13). That key also names the OBSERVED peak
        # at a snow pillow, in data_gather and the snotel server, and both are
        # reachable inside one analysis. Whose peak it is belongs in the name.
        m["peak_swe_modelled_mm"] = snow_stats["peak_swe_modelled_mm"]

    # ── the water budget ────────────────────────────────────────────────
    # P = RAIN + SNOW partitions into runoff and infiltration at the surface;
    # infiltration then goes to ET, recharge/drainage, or storage (dTWS).
    if rain is not None and snow is not None:
        p = rain + snow
        # `precip_total_mm_yr` DELETED — byte-identical to precip_mm_yr on
        # every column of both basins, computed six lines apart from the same
        # two variables.
        et_parts = [annual(k) for k in ("QSOIL", "QVEGE", "QVEGT")]
        et = (sum(x for x in et_parts if x is not None)
              if any(x is not None for x in et_parts) else None)
        budget: Dict[str, Any] = {}
        # RECHARGE SITS WITH THE EXPORTS but is not one, and the record says so
        # rather than relying on the reader knowing. QCHARGE moves water from
        # the soil column into the aquifer store and BOTH are inside TWS, so it
        # is not a loss from the column — it is already counted in
        # storage_change_mm. Summing this list would double-count it, which is
        # exactly what a reader does when five terms sit in a row and one of
        # them means something different.
        for label, v in (("runoff", qo), ("infiltration", annual("QINFL")),
                         ("et", et), ("recharge", qc),
                         ("drainage", annual("QDRAI"))):
            if v is not None:
                budget[f"{label}_mm_yr"] = round(v, 1)
                if p > 1e-6:
                    budget[f"{label}_frac_of_P"] = round(v / p, 3)
        if "recharge_mm_yr" in budget:
            budget["recharge_is_internal"] = (
                "QCHARGE is soil -> aquifer, both inside TWS. It is NOT an "
                "export and is not in the balance below; it is already in "
                "storage_change_mm. Do not add it to the export terms.")

        dtws = tws.get("delta_mm")
        if dtws is not None:
            budget["storage_change_mm"] = round(dtws, 1)
            # P - runoff - drainage - ET - dS, with recharge excluded for the
            # reason above.
            #
            # `unaccounted_mm_yr`, NOT `closure_residual` (2026-08-13). The
            # residual is 0 to 14% of precipitation on a real ensemble and it
            # is STRUCTURED — small where QDRAI > 0, and 120-180 mm in exactly
            # the columns where drainage is zero and storage is gaining. A term
            # called "closure residual" invites a reader to assume the budget
            # nearly closes; on this model it does not, and the name should not
            # paper over that. Whatever the gap is, it is water this extraction
            # cannot account for.
            #
            # ALWAYS WITH ITS FRACTION. "31 mm" says nothing; "31 mm, 2.9% of
            # precipitation" says whether to care, and the same 31 mm on a
            # 180 mm/yr Naches column would be 17%.
            if all(k in budget for k in
                   ("runoff_mm_yr", "drainage_mm_yr", "et_mm_yr")):
                gap = round(p - budget["runoff_mm_yr"]
                            - budget["drainage_mm_yr"]
                            - budget["et_mm_yr"] - dtws, 1)
                budget["unaccounted_mm_yr"] = gap
                if p > 1e-6:
                    budget["unaccounted_frac_of_P"] = round(gap / p, 3)
        if budget:
            m["water_budget"] = budget

    return {"stats": stats, "metrics": m}
