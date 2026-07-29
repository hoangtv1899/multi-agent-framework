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
