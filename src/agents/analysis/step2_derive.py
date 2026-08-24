#!/usr/bin/env python3
"""
Analyzer step 2 — what explains the spread across columns
src/agents/analysis/step2_derive.py

    in   ctx.data (the column rows), and the verdicts from step 1
    out  driver_matrix, spatial_summary, soil_attribution, comparisons

Every one of these is a CLAIM about the ensemble, which is why they are a
step of the Analyzer and not part of extraction.

AFTER the comparison, not before. The two do not consume each other — both need
only the context — so the order is a choice, and this is the safer one: a
correlation across columns the gauges say are wrong is a correlation of
nonsense. Validation's verdicts and the caveats it raises are available here,
so a driver table can be reported knowing whether the model it describes has
any purchase on reality.

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
    "runoff":            "runoff_mm_yr",
    "recharge":          "recharge_mm_yr",
    "recharge_fraction": "recharge_frac_of_P",
    "runoff_fraction":   "runoff_frac_of_P",
    "precip":            "precip_mm_yr",
    # DERIVED, not read: no producer writes this key into `metrics`. The value
    # comes from _penetration_depth_m over the row's own extract block — see
    # _DERIVED_RESPONSES and the definition on that function.
    "penetration_depth": "penetration_depth_m",
}


def _num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        return None if f != f else f            # NaN
    except (TypeError, ValueError):
        return None


def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
    """One metric by name, from `metrics` or from `metrics.water_budget`.

    THE BUDGET TERMS MOVED (2026-08-13). `annual_runoff_mm_yr`,
    `annual_recharge_mm_yr`, `recharge_fraction` and `runoff_fraction` were
    dropped from column_metrics because they duplicated water_budget's
    `runoff_mm_yr`, `recharge_mm_yr` and the `_frac_of_P` pair. Every function
    in this file was still asking for the old names.

    Nothing raised. A response key that is absent was skipped, so on the next
    run driver_matrix would have reported one row (precip), soil_attribution
    would have ranked every column by a recharge of 0, and comparisons would
    have covered one metric — a step 2 that runs, writes its file, and says
    almost nothing. This accessor is the same metrics-then-budget rule the
    extractor uses, so one rename cannot quietly empty the table again.
    """
    m = row.get("metrics") or {}
    if key in m:
        return _num(m[key])
    return _num((m.get("water_budget") or {}).get(key))


_WETTING_DSAT = 0.01        # a smaller saturation rise is numerical noise
_STARTS_SATURATED = 0.999   # at t0, at or above this = below the water table


def _penetration_depth_m(row: Dict[str, Any]) -> Optional[float]:
    """Wetting-front depth: how deep the run's water reached, in m.

    DEFINITION, exactly: the maximum depth_m whose saturation exceeds its own
    value in the profile at the FIRST output time by at least 0.01
    (_WETTING_DSAT) at ANY later output time, counting only depths that
    started unsaturated (first-time saturation < 0.999, _STARTS_SATURATED — a
    cell below the initial water table cannot record arrival). 0.0 when no
    such depth exists: the front reached no measured depth, which is a result,
    not a gap. None when the row carries no usable extract block — no
    `profiles`, fewer than two output times, or depth/saturation lengths that
    disagree.

    THE BASELINE IS times_y[0]. On a transient deck that snapshot is the
    initial condition BEFORE the steady spin, so spin-up adjustment counts as
    penetration; the number is "since the simulation started", not "during the
    forcing year", and a claim about the year alone cannot rest on it.

    Computed HERE from the block the extractor attached to the row
    (profiles = {depth_m: [...], times_y: [...], saturation: [[...] per
    time]}), by the rule that keeps every other claim in this file out of
    extraction: the extractor records what PFLOTRAN wrote and stops, and a
    front depth is an interpretation of it.
    """
    blk = row.get("profiles") or {}
    depths = blk.get("depth_m") or []
    sats = blk.get("saturation") or []
    if not depths or len(sats) < 2:
        return None
    init = sats[0]
    if len(init) != len(depths):
        return None
    deepest = None
    for i, d in enumerate(depths):
        d = _num(d)
        s0 = _num(init[i])
        if d is None or s0 is None or s0 >= _STARTS_SATURATED:
            continue
        for later in sats[1:]:
            s = _num(later[i]) if i < len(later) else None
            if s is not None and s - s0 >= _WETTING_DSAT:
                deepest = d if deepest is None or d > deepest else deepest
                break
    return round(deepest, 3) if deepest is not None else 0.0


# Response keys no producer writes into `metrics`, and the plain code that
# derives each from the row itself. In the analysis package on purpose: a
# derivation behind a tool call is a number nobody can audit.
_DERIVED_RESPONSES = {
    "penetration_depth_m": _penetration_depth_m,
}


def _response(row: Dict[str, Any], key: str) -> Optional[float]:
    """A response by key: the recorded metric, else the derived fallback.

    A metric a producer wrote WINS over a recomputation — the record is the
    audit trail. The fallback fires only where the record has nothing.
    """
    v = _metric(row, key)
    if v is None:
        fn = _DERIVED_RESPONSES.get(key)
        if fn is not None:
            v = fn(row)
    return v


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


def _water_table_m(row: Dict[str, Any]) -> Optional[float]:
    """Starting water-table depth — an INPUT to the column, not an output.

    Carried on the rows of runs that are BUILT at a water table: PFLOTRAN site
    rows write `water_table_m` (the depth the domain was sized from), coupled
    ELM rows write `initial_water_table_m` (the depth the warm start was
    stamped at). NOT the `wtd_prior_m` of KNOWN_UNAVAILABLE — that names the
    producer-less display prior, and its entry stays.
    """
    v = _num(row.get("water_table_m"))
    return v if v is not None else _num(row.get("initial_water_table_m"))


DRIVERS = {
    "elevation_m":   (lambda r: _num(r.get("elevation_m")), "m"),
    "precip_mm_yr":  (lambda r: _num((r.get("metrics") or {}).get("precip_mm_yr")),
                      "mm/yr"),
    "band":          (lambda r: _num(r.get("band")), "1"),
    "clay_max_pct":  (_clay_max_pct,   "%"),
    "sand_max_pct":  (_sand_max_pct,   "%"),
    "organic_max":   (_organic_max,    "kg/m3"),
    "water_table_m": (_water_table_m,  "m"),
}

# Named so it can be reported as MISSING rather than silently absent. A
# saturated hydraulic conductivity is not in CONUS-1km; deriving one needs a
# pedotransfer function, the same gap PFLOTRAN's retention curves hit.
KNOWN_UNAVAILABLE = {
    "ksat_min_ums": "not in CONUS-1km; needs a pedotransfer function from "
                    "sand/clay/organic",
    # STRUCTURALLY ABSENT, not thinly sampled. `fan_wtd_m` was a live driver
    # until 2026-08-07, when Fan left the sampler and its producer went with
    # it — the reasoning being that the water-table prior belongs to whoever
    # needs it (PFLOTRAN sizes its domain from it; ELM only displayed it), and
    # nothing fetches it for ELM. It is written by nothing in the tree today,
    # and `wtd_prior_m`, the name it was renamed to on 2026-08-12, has no
    # producer either.
    #
    # It was in DRIVERS until 2026-08-13, where it reported "fewer than 3
    # columns had a value" — true, and misleading: that phrasing says a driver
    # that is sometimes available happened to be thin here. Named as absent,
    # with the reason, which is what this block is for.
    #
    # NOT the same quantity as the `water_table_m` driver. This is the PRIOR —
    # a fetched estimate nothing produces. The starting water table IS on the
    # columns of runs built at one (PFLOTRAN site runs, coupled ELM runs), and
    # the driver reads it off the row.
    "wtd_prior_m": "no producer since 2026-08-07 (Fan left the sampler); the "
                   "water-table prior is fetched by the consumer that needs "
                   "it, and nothing fetches it for ELM",
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

    # A RESPONSE THAT CANNOT BE COMPUTED SAYS SO BY NAME, for the same reason
    # the drivers do. This loop used to `continue` past a missing key, so a
    # renamed metric shrank the table in silence and the shrunken table looked
    # like a legitimate result.
    out: Dict[str, Any] = {}
    missing: Dict[str, str] = {}
    for resp, key in _RESPONSES.items():
        ys = [_response(r, key) for r in rows]
        n = len([y for y in ys if y is not None])
        if n < 3:
            # "yield", not "carry", for a derived key: nothing carries it, and
            # saying so would send a reader hunting for a metrics key that has
            # never existed.
            verb = "yield" if key in _DERIVED_RESPONSES else "carry"
            missing[resp] = (f"{n} of {len(rows)} column(s) {verb} "
                             f"'{key}'; a correlation needs at least 3")
            continue
        out[resp] = {d: _pearson(values[d], ys) for d in available}

    return {
        "n_columns":   len(rows),
        "pearson_r":   out,
        "responses": {
            "measured":   sorted(out),
            "unmeasured": missing,
        },
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
    precip = [_metric(r, "precip_mm_yr") for r in rows]
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
                    (key, [x for x in (_metric(r, key) for r in rs)
                           if x is not None])
                    for key in ("precip_mm_yr", "runoff_mm_yr",
                                "recharge_mm_yr"))
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
        p = _metric(r, "precip_mm_yr")
        bins.setdefault(round(p) if p is not None else None, []).append(r)
    precip_bin, group = max(bins.items(), key=lambda kv: len(kv[1]))
    if len(group) < 2:
        return {"available": False,
                "reason": "no two columns share a forcing bin, so soil cannot "
                          "be separated from precipitation",
                "n_forcing_bins": len(bins)}

    group = sorted(group, key=lambda r: -(_metric(r, "recharge_mm_yr") or 0))
    table = [{
        "case_name":         r.get("case_name"),
        "texture_top":       r.get("soil_top_texture"),
        "clay_max_pct":      _clay_max_pct(r),
        "sand_max_pct":      _sand_max_pct(r),
        "organic_max":       _organic_max(r),
        "recharge_mm_yr":    _metric(r, "recharge_mm_yr"),
        "runoff_mm_yr":      _metric(r, "runoff_mm_yr"),
        "recharge_fraction": _metric(r, "recharge_frac_of_P"),
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
    for key in ("recharge_mm_yr", "runoff_mm_yr",
                "precip_mm_yr", "recharge_frac_of_P"):
        vals = {r.get("case_name"): _metric(r, key) for r in rows}
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
