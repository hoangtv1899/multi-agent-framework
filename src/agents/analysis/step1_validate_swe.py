#!/usr/bin/env python3
"""
Analyzer step 1a — snow water equivalent against SNOTEL
src/agents/analysis/step1_validate_swe.py

    in   ctx (model H2OSNO daily series + SNOTEL daily series)
    out  {metrics, comparable, caveats}  and a five-panel figure

NOT PAIRED. The previous version matched each station to its nearest column
by horizontal distance, which on the 2019 Upper Gunnison run collapsed five
stations onto two columns with elevation offsets up to 846 m — and SWE is
governed by elevation. Two stations 300 m apart in elevation were compared to
the same column 800 m above both. Any agreement that produced was arithmetic,
not skill.

So nothing is paired. Model and observed are both plotted against ELEVATION
and the question becomes whether the model reproduces the SWE-vs-elevation
GRADIENT — which uses all columns and all stations without pretending any
column is any station.

Five metrics, because they fail differently:

    peak SWE        magnitude. The most sensitive to siting and to elevation.
    peak date       when accumulation turns to melt.
    melt-out date   whether the pack persists correctly.
    duration        days above threshold; magnitude-free.
    first snow      accumulation onset.

Timing metrics matter more than they look. A pack 30% too thin that melts out
on the right day says the energetics are right and the precipitation is wrong.
Magnitude alone cannot separate those two, and they call for different fixes.

THE THRESHOLD IS A DECISION, not a detail. At 1 mm the "first snow" date
chases ephemeral dustings that a 12 km-forced column and a point sensor will
never agree on. 25 mm (~1 inch SWE) is the smallest depth at which both are
describing the same physical thing.

THE WINDOWS DIFFER. SNOTEL is reported by WATER year (Oct-Sep); the model runs
a calendar year. Only the overlap is comparable, and inside that overlap the
model's record starts mid-winter with snow already on the ground — so its
first-snow date is not observable at all. That is reported as unavailable
rather than computed, because a first-snow date taken from a warm-started
1 January is an artefact of when the run began.
"""
from typing import Any, Dict, List, Optional, Tuple

from agents.analysis.step1_geo import in_polygon    # noqa: E402

SWE_THRESHOLD_MM = 25.0      # ~1 inch SWE; see the module docstring


def _num(x) -> Optional[float]:
    try:
        f = float(x)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def swe_metrics(dates: List[str], values: List[Optional[float]],
                threshold: float = SWE_THRESHOLD_MM) -> Dict[str, Any]:
    """Five metrics from one daily SWE series.

    A series that never crosses the threshold has NO persistent snowpack, and
    its timing metrics are undefined rather than zero. Returning 0 or day 0
    would put a bare desert column on the same axis as a late-melting ridge.
    """
    pairs = [(d, _num(v)) for d, v in zip(dates or [], values or [])]
    pairs = [(d, v) for d, v in pairs if v is not None]
    if not pairs:
        return {"available": False, "reason": "no SWE series"}

    ds = [d for d, _ in pairs]
    vs = [v for _, v in pairs]
    peak = max(vs)
    out: Dict[str, Any] = {
        "available":     True,
        "threshold_mm":  threshold,
        "peak_swe_mm":   round(peak, 1),
        # Mean over the SHARED window. On its own it conflates magnitude with
        # duration — a thin pack lasting months matches a deep one melting
        # fast. Beside peak it separates them: peak agreeing while mean
        # disagrees is a duration error, both off is magnitude.
        "mean_swe_mm":   round(sum(vs) / len(vs), 1),
        "peak_date":     ds[vs.index(peak)],
        "first_date":    ds[0],
        "last_date":     ds[-1],
        "n_days":        len(vs),
    }

    above = [i for i, v in enumerate(vs) if v > threshold]
    out["days_above_threshold"] = len(above)
    if not above:
        out["has_snowpack"] = False
        out["first_snow_date"] = None
        out["melt_out_date"] = None
        out["undefined_reason"] = (
            f"SWE never exceeded {threshold:g} mm — no persistent snowpack, so "
            f"onset and melt-out are undefined rather than zero")
        return out

    out["has_snowpack"] = True
    out["first_snow_date"] = ds[above[0]]
    # Melt-out: the first day AFTER the last threshold crossing. Absent if the
    # series ends while snow is still on the ground — that is censoring, not a
    # late melt, and a date invented for it would be the record's end rather
    # than the snowpack's.
    last = above[-1]
    if last + 1 < len(ds):
        out["melt_out_date"] = ds[last + 1]
    else:
        out["melt_out_date"] = None
        out["melt_out_censored"] = (
            "the series ends with snow still above threshold; melt-out is "
            "after the record, not at its end")
    # An onset date on the FIRST day of the record is not an onset — the snow
    # was already there when the record began.
    if above[0] == 0:
        out["first_snow_date"] = None
        out["first_snow_censored"] = (
            "snow was already above threshold on the first day of the record, "
            "so accumulation onset happened before it began")
    return out


def _overlap(a: Tuple[str, str], b: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def compare(ctx, threshold: float = SWE_THRESHOLD_MM) -> Dict[str, Any]:
    """Model columns and SNOTEL stations, each against elevation.

    Both are restricted to the window they share before any metric is taken.
    Comparing a water-year peak date with a calendar-year one silently
    compares two different winters.
    """
    stations = ((ctx.data.get("observations") or {}).get("swe") or {}) \
        .get("stations") or []

    model: List[Dict[str, Any]] = []
    obs:   List[Dict[str, Any]] = []
    caveats: List[Dict[str, Any]] = []

    m_span = o_span = None
    for row in ctx.columns:
        d = ((row.get("variables") or {}).get("H2OSNO") or {}).get("daily") or {}
        if d.get("dates"):
            m_span = (d["dates"][0], d["dates"][-1])
            break
    for st in stations:
        d = st.get("daily") or {}
        if d.get("dates"):
            o_span = (d["dates"][0], d["dates"][-1])
            break

    win = _overlap(m_span, o_span) if (m_span and o_span) else None
    if win and (m_span != o_span):
        caveats.append({
            "id": "swe_window_overlap", "severity": "qualify",
            "statement": (f"SNOTEL is reported by water year ({o_span[0]} to "
                          f"{o_span[1]}) and the model runs a calendar year "
                          f"({m_span[0]} to {m_span[1]}). Metrics are taken "
                          f"over the shared window {win[0]} to {win[1]} only."),
            "applies_to": "every SWE comparison",
            "source": "step1_validate_swe"})

    def clip(dates, values):
        if not win:
            return dates, values
        keep = [i for i, dt in enumerate(dates or []) if win[0] <= dt <= win[1]]
        return ([dates[i] for i in keep], [values[i] for i in keep])

    for row in ctx.columns:
        d = ((row.get("variables") or {}).get("H2OSNO") or {}).get("daily") or {}
        dates, vals = clip(d.get("dates") or [], d.get("values") or [])
        m = swe_metrics(dates, vals, threshold)
        # the clipped series itself, not just its summary — the second figure
        # draws the accumulation and melt SHAPE, which no scalar carries
        m.update(entity=row.get("case_name"),
                 elevation_m=_num(row.get("elevation_m")), source="model",
                 series={"dates": dates, "values": vals})
        model.append(m)

    rings = ctx.data.get("boundary") or []
    outside = []
    for st in stations:
        lat, lon = _num(st.get("lat")), _num(st.get("lon"))
        if rings and lat is not None and lon is not None \
                and not in_polygon(lat, lon, rings):
            outside.append(st.get("name") or st.get("triplet"))
            continue
        d = st.get("daily") or {}
        dates, vals = clip(d.get("dates") or [], d.get("values") or [])
        m = swe_metrics(dates, vals, threshold) if dates else {
            "available": False, "reason": "station carries no daily series"}
        m.update(entity=st.get("name") or st.get("triplet"),
                 elevation_m=_num(st.get("elevation_m")), source="observed",
                 series={"dates": dates, "values": vals})
        obs.append(m)

    # SNOTEL sites are chosen for snow retention — sheltered, shaded,
    # wind-protected. A 12 km grid-cell average is not that. Observations
    # sitting above the model at matched elevation is EXPECTED and is not by
    # itself evidence of a model dry bias.
    if obs:
        caveats.append({
            "id": "snotel_siting_bias", "severity": "qualify",
            "statement": ("SNOTEL sites are deliberately placed where snow "
                          "accumulates and persists; the model column is a "
                          "12 km forced grid-cell average. Observed SWE above "
                          "modelled at the same elevation is expected from "
                          "siting alone and is not by itself a model bias."),
            "applies_to": "any SWE magnitude claim",
            "source": "step1_validate_swe"})

    censored = [m for m in model if m.get("first_snow_censored")]
    if censored:
        caveats.append({
            "id": "swe_onset_unobservable", "severity": "blocking",
            "statement": (f"{len(censored)} of {len(model)} columns already had "
                          f"snow above threshold on the first day of the record, "
                          f"so accumulation onset is not observable in this run. "
                          f"An onset date taken from a warm-started 1 January "
                          f"reports when the run began, not when snow arrived."),
            "applies_to": "first-snow / accumulation-onset claims",
            "source": "step1_validate_swe"})

    if outside:
        caveats.append({
            "id": "swe_stations_outside_basin", "severity": "context",
            "statement": (f"{len(outside)} SNOTEL station(s) were excluded for "
                          f"lying outside the watershed: {', '.join(outside)}. "
                          f"Reception fetches by bounding box, which includes "
                          f"the basin's corners."),
            "applies_to": "SWE coverage", "source": "step1_validate_swe"})

    pairs, unpaired = pair_by_elevation(model, obs)
    for u in unpaired:
        caveats.append({
            "id": f"swe_unpaired_{str(u.get('entity','?'))[:20].replace(' ','_')}",
            "severity": "context",
            "statement": f"{u.get('entity')} was not paired: {u.get('unpaired_reason')}",
            "applies_to": "SWE coverage", "source": "step1_validate_swe"})

    return {"threshold_mm": threshold, "window": win,
            "stations_excluded_outside_basin": outside,
            "model": model, "observed": obs, "caveats": caveats,
            "pairs": pairs, "unpaired": unpaired,
            "n_model_with_snowpack": sum(1 for m in model if m.get("has_snowpack")),
            "n_obs_with_series": sum(1 for m in obs if m.get("available"))}




# ── geography ───────────────────────────────────────────────────────────
MAX_PAIR_DELTA_M = 200.0     # see pair_by_elevation


def pair_by_elevation(model: List[Dict[str, Any]],
                      observed: List[Dict[str, Any]],
                      max_delta_m: float = MAX_PAIR_DELTA_M
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One station to one column, matched on ELEVATION.

    Elevation, because that is what governs SWE. The previous rule matched on
    horizontal distance and produced offsets up to 846 m, with five stations
    collapsed onto two columns; on elevation the worst offset is 162 m and
    every station finds a partner.

    A BIJECTION. Where two stations claim the same column, the closer one
    keeps it and the other is recorded as unpaired — not dropped silently.
    Allowing both would put two points at the same y on a 1:1 plot, and a
    repeated model value cannot carry the comparison it appears to make.

    Returns (pairs, unpaired). Unpaired stations carry the reason, because a
    station excluded by a rule is a different thing from a station that never
    reported.
    """
    cands = []
    for o in observed:
        if not (o.get("available") and o.get("elevation_m") is not None):
            continue
        best, delta = None, None
        for m in model:
            if not (m.get("available") and m.get("elevation_m") is not None):
                continue
            d = abs(m["elevation_m"] - o["elevation_m"])
            if delta is None or d < delta:
                best, delta = m, d
        if best is None:
            continue
        cands.append((delta, o, best))

    cands.sort(key=lambda t: t[0])          # closest claims its column first
    pairs, unpaired, taken = [], [], {}
    for delta, o, m in cands:
        if delta > max_delta_m:
            unpaired.append({**o, "unpaired_reason":
                             f"nearest column is {delta:.0f} m away in "
                             f"elevation, beyond the {max_delta_m:.0f} m limit"})
            continue
        cid = m.get("entity")
        if cid in taken:
            unpaired.append({**o, "unpaired_reason":
                             f"{cid} is already paired with {taken[cid]}, which "
                             f"is closer in elevation"})
            continue
        taken[cid] = o.get("entity")
        pairs.append({"station": o.get("entity"), "column": cid,
                      "station_elevation_m": o["elevation_m"],
                      "column_elevation_m": m["elevation_m"],
                      "delta_elevation_m": round(m["elevation_m"] - o["elevation_m"], 1),
                      "observed": o, "model": m})
    pairs.sort(key=lambda p: -p["station_elevation_m"])
    return pairs, unpaired


# ── the figure ─────────────────────────────────────────────────────────
# Title only. No interpretive subtitle: what a panel MEANS belongs in the
# caveats and the interpretation, which travel as records. A figure reports.
_PANELS = [
    ("mean_swe_mm", "mean SWE (mm)"),
    ("peak_swe_mm", "peak SWE (mm)"),
    ("peak_date",   "peak date (day of water year)"),
]


def _doy(d):
    """Day of the WATER year (Oct 1 = 1).

    Calendar day-of-year puts a late-December peak and an early-January peak
    at opposite ends of the axis while being ten days apart.
    """
    if not d:
        return None
    try:
        import datetime as _dt
        dd = _dt.date.fromisoformat(str(d)[:10])
        start = _dt.date(dd.year - (0 if dd.month >= 10 else 1), 10, 1)
        return (dd - start).days + 1
    except Exception:
        return None


def _stats(xs, ys):
    """Bias and RMSE. No fitted slope — see the docstring in plot()."""
    n = len(xs)
    if n == 0:
        return {}
    bias = sum(y - x for x, y in zip(xs, ys)) / n
    rmse = (sum((y - x) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return {"n": n, "bias": round(bias, 1), "rmse": round(rmse, 1)}


def plot_scatter(result: Dict[str, Any], out_path) -> str:
    """Three 1:1 panels: x = SNOTEL observed, y = ELM simulated.

    One point per PAIR, each pair one station and one column matched on
    elevation. The 1:1 line is the reference; distance from it is the error,
    and which side tells you the direction.

    No fitted slope. Four points, and a regression through four points — one
    of which sits 147 m off in elevation — would be noise presented as a
    trend. Bias and RMSE say what can honestly be said at this sample size.

    The systematic direction is predictable and must be read with the siting
    caveat: SNOTEL sites are chosen for snow retention, so points falling
    BELOW the line are partly a sampling artefact and not wholly model error.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = result.get("pairs") or []
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))

    for ax, (key, label) in zip(axes, _PANELS):
        is_date = key.endswith("date")
        xs, ys, ann = [], [], []
        for pr in pairs:
            o = (pr["observed"].get(key) if not is_date
                 else _doy(pr["observed"].get(key)))
            m = (pr["model"].get(key) if not is_date
                 else _doy(pr["model"].get(key)))
            o, m = (_num(o), _num(m)) if not is_date else (o, m)
            if o is None or m is None:
                continue
            xs.append(o); ys.append(m)
            ann.append(f"{pr['station'][:20]}  {pr['delta_elevation_m']:+.0f} m")

        if not xs:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "NOT COMPARABLE\nno paired values",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#b30000", fontsize=10, fontweight="bold")
            ax.set_title(label, fontweight="bold", fontsize=17)
            continue

        lo = min(xs + ys); hi = max(xs + ys)
        pad = 0.16 * (hi - lo or 1)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1.4, zorder=1)
        # Labelled ON the line rather than in a legend box: the box sat in the
        # lower-right corner where the points are, and covered them.
        ax.annotate("1:1", xy=(hi, hi), xytext=(-30, -20),
                    textcoords="offset points", fontsize=12, color="#444",
                    rotation=45, ha="center", va="center")
        ax.scatter(xs, ys, s=95, c="#2c7fb8", edgecolor="#111", linewidth=0.8,
                   zorder=3)
        # Label placement, as one search rather than two rules. The earlier
        # version staggered downward and then flipped anything that fell off
        # the axis — but the flip bypassed the collision check and put two
        # labels back at the same height. Here every candidate offset is
        # tested for BOTH constraints and the first that satisfies them wins.
        span = hi - lo
        placed: List[Tuple[float, float]] = []
        CANDIDATES = (-17, -34, -51, 13, 30, 47)

        def free(cx, cy):
            inside = lo + 0.015 * span < cy < hi - 0.015 * span
            clear = all(abs(cy - py) > 0.05 * span or abs(cx - px) > 0.40 * span
                        for px, py in placed)
            return inside and clear

        for i in sorted(range(len(xs)), key=lambda k: (-ys[k], xs[k])):
            x, y, t = xs[i], ys[i], ann[i]
            right = (x - lo) < 0.58 * span
            dy = next((d for d in CANDIDATES
                       if free(x, y + d * span / 260.0)), CANDIDATES[0])
            placed.append((x, y + dy * span / 260.0))
            ax.annotate(t, (x, y), textcoords="offset points",
                        xytext=(12 if right else -12, dy),
                        ha="left" if right else "right",
                        va="bottom" if dy > 0 else "top",
                        fontsize=10.5, color="#333", annotation_clip=False)

        st = _stats(xs, ys)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        # The metric is named in the title, so the axes only need to say
        # which side is which.
        ax.set_xlabel("SNOTEL", fontsize=19)
        ax.set_ylabel("ELM", fontsize=19)
        ax.set_title(label, fontweight="bold", fontsize=17, pad=12)
        ax.tick_params(labelsize=12)
        unit = "d" if is_date else "mm"
        ax.text(0.03, 0.97, f"n={st['n']}\nbias={st['bias']:+g} {unit}\n"
                            f"RMSE={st['rmse']:g} {unit}",
                transform=ax.transAxes, va="top", fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb",
                          alpha=0.92))
        ax.grid(alpha=0.25, zorder=0)

    # No overall title and no date strip. Each panel already names its metric,
    # and the axes name the two sides; the window, the pair count and the
    # unpaired station are in the returned record, where a caption or the
    # interpretation can quote them. A figure that repeats its own metadata
    # spends space saying what the reader already has.
    fig.tight_layout()
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_timeseries(result: Dict[str, Any], out_path) -> str:
    """One panel per pair: daily SWE, observed and simulated on one axis.

    The scatter answers "how big"; this answers "what shape". A pack that is
    30% thin but accumulates and melts on the right dates is a precipitation
    problem; one with the right peak reached a month late is an energetics
    problem. Neither is visible in a peak value, and the two call for
    different fixes.

    ONE SHARED y-AXIS across panels. Per-panel scaling would let a column
    holding 154 mm look like the station holding 925 mm — the axis would
    quietly normalise away the very deficit the figure exists to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import datetime as _dt

    pairs = result.get("pairs") or []
    if not pairs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "NOT COMPARABLE\nno station-column pairs",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="#b30000", fontweight="bold")
        fig.savefig(out_path, dpi=135, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    n = len(pairs)
    ncol = 2 if n > 2 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4 * ncol, 4.8 * nrow),
                             squeeze=False)
    flat = [a for r in axes for a in r]

    hi = 0.0
    for pr in pairs:
        for side in ("observed", "model"):
            vs = [v for v in ((pr[side].get("series") or {}).get("values") or [])
                  if v is not None]
            if vs:
                hi = max(hi, max(vs))
    hi = hi * 1.12 or 1.0

    def _dates(d):
        out = []
        for x in d:
            try:
                out.append(_dt.date.fromisoformat(str(x)[:10]))
            except Exception:
                out.append(None)
        return out

    for ax, pr in zip(flat, pairs):
        for side, colour, style, label in (
                ("observed", "#e6550d", "-",  "SNOTEL"),
                ("model",    "#2c7fb8", "-",  "ELM")):
            ser = pr[side].get("series") or {}
            xs = _dates(ser.get("dates") or [])
            ys = ser.get("values") or []
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        color=colour, lw=2.4, label=label)
        ax.set_ylim(0, hi)
        ax.set_ylabel("SWE (mm)", fontsize=17)
        ax.set_title(f"{pr['station']}  /  {pr['column']}"
                     f"   ({pr['delta_elevation_m']:+.0f} m)",
                     fontweight="bold", fontsize=17, pad=10)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=14, loc="upper right")
        for lb in ax.get_xticklabels():
            lb.set_rotation(30); lb.set_ha("right")

    for ax in flat[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def map_points(result, ctx):
    """Mean SWE as map points: (stations, columns), each (lon, lat, mm, name).

    Coordinates are NOT in the compare() record — it keys on entity name — so
    they are looked up here against reception's stations and the packaged
    columns. One definition, used by this module's own maps and by the
    combined grid, so the two figures can never disagree about where a station
    is.
    """
    stations = [m for m in (result.get("observed") or []) if m.get("available")]
    columns  = [m for m in (result.get("model") or []) if m.get("available")]
    by_col = {r.get("case_name"): r for r in ctx.columns}
    raw_st = ((ctx.data.get("observations") or {}).get("swe") or {}) \
        .get("stations") or []
    by_st = {(s.get("name") or s.get("triplet")): s for s in raw_st}

    def pts(items, lookup):
        out = []
        for m in items:
            src = lookup.get(m.get("entity")) or {}
            lat, lon = _num(src.get("lat")), _num(src.get("lon"))
            v = _num(m.get("mean_swe_mm"))
            if lat is not None and lon is not None and v is not None:
                out.append((lon, lat, v, m.get("entity")))
        return out

    return pts(stations, by_st), pts(columns, by_col)
