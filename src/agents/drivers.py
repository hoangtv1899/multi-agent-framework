#!/usr/bin/env python3
"""
What explains the spread across columns
src/agents/drivers.py

Correlations and ensemble aggregates, computed by the ANALYZER from the
package rows rather than baked into extraction.

They used to be computed in _extract and frozen into hydro_summary.json. Two
things were wrong with that. A Pearson correlation is a claim about a
relationship — that is interpretation, and extraction's job is to read the
model's output format and stop. And frozen at extract time, the driver set
could never answer a question thought of later: band, gravel, aspect.

The concrete damage was worse than the principle. The old implementation read
`row['soil'].get('clay_max_pct')`, but extraction never populates `soil` — so
every soil correlation came out null across every run, and null in a
correlation table reads as "no relationship" when it means "not computed".
Nobody noticed, because the computation lived nowhere near the figure that
displayed it.

So the rule here: a driver that cannot be computed says so by NAME. `available`
and `unavailable` are both reported, and `unavailable` carries the reason.
"""
from typing import Any, Dict, List, Optional, Sequence

_RESPONSES = {
    "runoff":            "annual_runoff_mm_yr",
    "recharge":          "annual_recharge_mm_yr",
    "recharge_fraction": "recharge_fraction",
    "runoff_fraction":   "runoff_fraction",
    "precip":            "precip_mm_yr",
}


def _num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        return None if f != f else f            # NaN
    except (TypeError, ValueError):
        return None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Plain Pearson r, or None when it is not defined.

    None when fewer than 3 pairs survive, or when either series is constant —
    a correlation against a constant is 0/0, and reporting 0 for it would
    claim independence that was never measured.
    """
    pairs = [(a, b) for a, b in zip(xs, ys)
             if a is not None and b is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return round(sxy / (sxx * syy) ** 0.5, 3)


# ── drivers ─────────────────────────────────────────────────────────────
def _clay_max_pct(row: Dict[str, Any]) -> Optional[float]:
    """Maximum clay fraction down the profile.

    Derived HERE from soil_profile's per-layer values, which is the form the
    data actually arrives in. The old code expected a precomputed
    `clay_max_pct` scalar that nothing ever wrote.
    """
    layers = ((row.get("soil_profile") or {}).get("layers")) or []
    vals = [_num(l.get("clay_pct")) for l in layers]
    vals = [v for v in vals if v is not None]
    return round(max(vals), 2) if vals else None


def _sand_max_pct(row: Dict[str, Any]) -> Optional[float]:
    layers = ((row.get("soil_profile") or {}).get("layers")) or []
    vals = [_num(l.get("sand_pct")) for l in layers]
    vals = [v for v in vals if v is not None]
    return round(max(vals), 2) if vals else None


def _organic_max(row: Dict[str, Any]) -> Optional[float]:
    layers = ((row.get("soil_profile") or {}).get("layers")) or []
    vals = [_num(l.get("organic_kg_m3")) for l in layers]
    vals = [v for v in vals if v is not None]
    return round(max(vals), 2) if vals else None


DRIVERS = {
    "elevation_m":   (lambda r: _num(r.get("elevation_m")), "m"),
    "precip_mm_yr":  (lambda r: _num((r.get("metrics") or {}).get("precip_mm_yr")),
                      "mm/yr"),
    "fan_wtd_m":     (lambda r: _num(r.get("fan_wtd_m")), "m"),
    "band":          (lambda r: _num(r.get("band")), "1"),
    "clay_max_pct":  (_clay_max_pct,   "%"),
    "sand_max_pct":  (_sand_max_pct,   "%"),
    "organic_max":   (_organic_max,    "kg/m3"),
}

# Named so it can be reported as MISSING rather than silently absent. A
# saturated hydraulic conductivity is not in CONUS-1km; deriving one needs a
# pedotransfer function, the same gap PFLOTRAN's retention curves hit.
KNOWN_UNAVAILABLE = {
    "ksat_min_ums": "not in CONUS-1km; needs a pedotransfer function from "
                    "sand/clay/organic",
}


def driver_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """response x driver correlations across the ensemble.

    Reports which drivers were AVAILABLE and which were not, because a table
    of nulls is indistinguishable from a table of measured non-relationships.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    values = {name: [fn(r) for r in rows] for name, (fn, _u) in DRIVERS.items()}
    available   = sorted(k for k, v in values.items()
                         if len([x for x in v if x is not None]) >= 3)
    unavailable = {k: "fewer than 3 columns had a value"
                   for k in values if k not in available}
    unavailable.update(KNOWN_UNAVAILABLE)

    out: Dict[str, Any] = {}
    for resp, key in _RESPONSES.items():
        ys = [_num((r.get("metrics") or {}).get(key)) for r in rows]
        if len([y for y in ys if y is not None]) < 3:
            continue
        out[resp] = {d: _pearson(values[d], ys) for d in available}

    return {
        "n_columns":   len(rows),
        "pearson_r":   out,
        "drivers": {
            "available":   {d: DRIVERS[d][1] for d in available},
            "unavailable": unavailable,
        },
        "note": "Correlations across ALL columns. Forcing and elevation "
                "covary in a mountain basin, so a high r against one is not "
                "evidence against the other. r is null where it is not "
                "defined (constant series, or fewer than 3 pairs) — that is "
                "different from a measured zero.",
    }


def spatial_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensemble shape: elevation span, forcing spread, per-band aggregates."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    elev = [_num(r.get("elevation_m")) for r in rows]
    elev = [e for e in elev if e is not None]
    precip = [_num((r.get("metrics") or {}).get("precip_mm_yr")) for r in rows]
    precip = [p for p in precip if p is not None]

    by_band: Dict[str, Any] = {}
    for r in rows:
        b = r.get("band")
        if b is None:
            continue
        by_band.setdefault(str(b), []).append(r)

    bands = {}
    for b, rs in sorted(by_band.items()):
        es = [_num(r.get("elevation_m")) for r in rs]
        es = [e for e in es if e is not None]
        bands[b] = {
            "n_columns": len(rs),
            "elevation_range_m": [round(min(es), 1), round(max(es), 1)] if es else None,
            "mean": {
                k: round(sum(v) / len(v), 2)
                for k, v in (
                    (key, [x for x in
                           (_num((r.get("metrics") or {}).get(key)) for r in rs)
                           if x is not None])
                    for key in ("precip_mm_yr", "annual_runoff_mm_yr",
                                "annual_recharge_mm_yr"))
                if v
            },
        }

    return {
        "n_columns": len(rows),
        "elevation_range_m": ([round(min(elev), 1), round(max(elev), 1)]
                              if elev else None),
        "forcing": {
            "precip_mm_yr_distinct": sorted({round(p) for p in precip}),
            "n_forcing_bins": len({round(p) for p in precip}),
            # One forcing value across every column means the gradient is not
            # resolved and no elevation claim can rest on it.
            "elevation_resolved": len({round(p) for p in precip}) > 1,
        },
        "by_band": bands,
        "n_bands": len(bands),
    }


def soil_attribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attribute the partitioning to SOIL, holding forcing constant.

    The spatial ensemble confounds soil with forcing: wetter columns are also
    higher and steeper. Fixing precipitation and letting only soil vary is the
    one comparison that separates them, which is why the figure exists.

    It never ran. The old version filtered on `row['soil']`, which extraction
    does not populate, so the candidate set was empty on every run and this
    returned {} — and soil_control.png was silently never drawn. Here the
    predictors come from soil_profile, the form the data actually arrives in.

    The bin threshold is 2, not 3, deliberately. The 12 km forcing quantises
    precipitation so heavily that a 19-column ensemble rarely puts three
    columns in one bin. With two the correlation is meaningless, but the PAIR
    is not: two columns under identical forcing that differ in recharge differ
    because of soil.
    """
    rows = [r for r in (rows or [])
            if isinstance(r, dict)
            and str(r.get("status", "ok")).lower() not in
            {"failed", "error", "timeout"}]
    usable = [r for r in rows if _clay_max_pct(r) is not None]
    if len(usable) < 2:
        return {"available": False,
                "reason": f"only {len(usable)} column(s) carry a soil profile; "
                          f"soil attribution needs at least 2"}

    bins: Dict[Any, List[Dict[str, Any]]] = {}
    for r in usable:
        p = _num((r.get("metrics") or {}).get("precip_mm_yr"))
        bins.setdefault(round(p) if p is not None else None, []).append(r)
    precip_bin, group = max(bins.items(), key=lambda kv: len(kv[1]))
    if len(group) < 2:
        return {"available": False,
                "reason": "no two columns share a forcing bin, so soil cannot "
                          "be separated from precipitation",
                "n_forcing_bins": len(bins)}

    group = sorted(group, key=lambda r: -( _num(
        (r.get("metrics") or {}).get("annual_recharge_mm_yr")) or 0))
    table = [{
        "case_name":         r.get("case_name"),
        "texture_top":       r.get("soil_top_texture"),
        "clay_max_pct":      _clay_max_pct(r),
        "sand_max_pct":      _sand_max_pct(r),
        "organic_max":       _organic_max(r),
        "recharge_mm_yr":    _num((r.get("metrics") or {}).get("annual_recharge_mm_yr")),
        "runoff_mm_yr":      _num((r.get("metrics") or {}).get("annual_runoff_mm_yr")),
        "recharge_fraction": _num((r.get("metrics") or {}).get("recharge_fraction")),
    } for r in group]

    out: Dict[str, Any] = {
        "available":     True,
        "precip_bin_mm_yr": precip_bin,
        # names plot_soil() and the CLI's print_soil() read
        "forcing_held_mm_yr": precip_bin,
        "n_columns":     len(group),
        "n_forcing_bins": len(bins),
        "columns":       table,
        # plot_soil() reads `by_recharge` and `soil_correlation`. Emitting the
        # shape it already expects keeps the figure working while the
        # computation moves; the figure itself is repointed separately.
        "by_recharge":   [dict(t, ksat_min_ums=None) for t in table],
        "note": "Forcing held constant by keeping only the most-populated "
                "precipitation bin, so differences here are attributable to "
                "soil. With 2 columns this is a PAIR, not a correlation.",
    }
    rech = [t["recharge_mm_yr"] for t in table]
    if len(group) >= 3:
        for pred in ("clay_max_pct", "sand_max_pct", "organic_max"):
            xs = [t[pred] for t in table]
            out.setdefault("pearson_r", {})[pred] = {
                "recharge": _pearson(xs, rech),
                "runoff":   _pearson(xs, [t["runoff_mm_yr"] for t in table]),
            }
    # Also the flat form plot_soil() reads. None where undefined — with two
    # columns there is no correlation, only a pair, and the figure says so.
    out["soil_correlation"] = {
        "recharge_vs_clay_max": _pearson([t["clay_max_pct"] for t in table], rech),
        "recharge_vs_sand_max": _pearson([t["sand_max_pct"] for t in table], rech),
        "recharge_vs_ksat_min": None,      # see KNOWN_UNAVAILABLE
    }
    # Whichever soil predictor actually tracks recharge best — named, so the
    # figure and the prose cannot disagree about which one they mean.
    ranked = [(k, v) for k, v in (
        ("clay_max_pct", out["soil_correlation"]["recharge_vs_clay_max"]),
        ("sand_max_pct", out["soil_correlation"]["recharge_vs_sand_max"]),
    ) if v is not None]
    out["strongest_predictor"] = (
        max(ranked, key=lambda kv: abs(kv[1]))[0] if ranked
        else "none — too few columns share a forcing bin to rank predictors")
    return out


def comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spread of each headline metric across the ensemble.

    Range and ratio, not a ranking: which column is highest matters less than
    whether the ensemble spans anything at all. A near-zero spread means the
    columns are not telling you about heterogeneity, whatever else they say.
    """
    rows = [r for r in (rows or [])
            if isinstance(r, dict)
            and str(r.get("status", "ok")).lower() not in
            {"failed", "error", "timeout"}]
    if len(rows) < 2:
        return []

    out = []
    for key in ("annual_recharge_mm_yr", "annual_runoff_mm_yr",
                "precip_mm_yr", "recharge_fraction"):
        vals = {r.get("case_name"): _num((r.get("metrics") or {}).get(key))
                for r in rows}
        vals = {k: v for k, v in vals.items() if v is not None}
        if len(vals) < 2:
            continue
        lo_k, lo = min(vals.items(), key=lambda kv: kv[1])
        hi_k, hi = max(vals.items(), key=lambda kv: kv[1])
        out.append({
            "metric":  key,
            "units":   "mm/yr" if key.endswith("mm_yr") else "1",
            "n":       len(vals),
            "min":     {"column": lo_k, "value": round(lo, 3)},
            "max":     {"column": hi_k, "value": round(hi, 3)},
            "mean":    round(sum(vals.values()) / len(vals), 3),
            "range":   round(hi - lo, 3),
            # None, not inf: a zero minimum is a real result, and inf in a
            # report reads as a bug rather than as "unbounded".
            "ratio":   round(hi / lo, 2) if lo not in (0,) and lo > 0 else None,
        })
    return out
