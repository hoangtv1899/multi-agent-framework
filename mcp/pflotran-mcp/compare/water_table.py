#!/usr/bin/env python3
"""
water_table — a PFLOTRAN column's water table against a well. MEASUREMENTS ONLY.

    model    water_table_m       the depth the column was GIVEN: ParFlow CONUS2's
                                 steady-state water table, the fixed head at the
                                 bottom face the column is anchored to
             crossing_m(t)       from the profiles: the depth where the modelled
                                 liquid pressure crosses atmospheric (101325 Pa)
                                 at each output time — where the water table IS
                                 in the model's own answer
    obs      a well's daily depth to water, m below land surface

WHAT A WELL COMPARISON MEANS HERE, said before any number. The bottom face of
every column is a FIXED HEAD — hydrostatic at the CONUS2 water-table depth
(`bottom_wt`) — and the column starts hydrostatic about that depth. The water
table itself is not held: it is the depth where the solved pressure crosses
atmospheric, and it moves with the recharge the column receives, anchored to
that pressure at the bottom face. Measured on Naches, 2026-08-18: col_09 was
given 99.9 m, mounded to 76 m over ten spin-up years at the mean recharge, and
moved 3.6 m within the forcing year; col_03 sat at 13.3 m in every snapshot.
So a well beside a column speaks to two things at once — the CONUS2 depth the
column was given (`level.given_m`) and where the model's own water table sat
through the year (`level.model_year_mean_m`) — and the record keeps them
apart. `boundary_condition` states the anchoring and how far each column's
crossing moved, so a reader can see how free it was. Context, not skill — the
rule every comparison in this framework already follows.

SNAPSHOTS, NOT DAYS. A transient column writes its profile at a few output
times (quarterly through the forcing year); a well reports daily. The pairing
is an inner join on the snapshot DATES, so a pair has as many points as the
column has snapshots in the year — four, typically — and the metrics say so in
`n_days`. The level comparison (`level`) uses the well's whole record over the
year against the depth the column was given, which needs no date at all.

DATES FROM THE ROW, NOT FROM A CONFIG. `times_y` counts from the simulation
start and INCLUDES the steady spin-up (FIELD_SEMANTICS.times_y). The transient
year is the last `n_forcing_steps/365` years of the run, so its start is
`t_last - n_forcing_steps/365`, and a snapshot at `t` falls on day
`round((t - t0) * 365)` of `forcing_start`. Both numbers are on every packaged
row; nothing here reads a run plan or a deck.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.analysis import compare_common as C

ATMOSPHERIC_PA = 101325.0
DAYS = 365                       # Daymet's calendar, and the series' step

SPEC = C.Spec(
    name="water_table", model_vars=["water_table_m", "profiles"], units="m",
    comparand=("the depth where the modelled liquid pressure crosses "
               "atmospheric at each output time — the model's water table, "
               "anchored to a fixed head at the bottom face — and the CONUS2 "
               "depth the column was given (positive down)"),
    obs_quantity="depth to water in a well (positive down)",
    colocated=True, pair_on="distance",
    # THE SAME 5 km ELM's water_table uses, for the same reason: it is a fact
    # about what a well measures — a local water table — not about the model.
    # It was set by the user after measuring the alternatives at Naches
    # (mcp/elm-mcp/src/compare/water_table.py); nothing about PFLOTRAN changes
    # how far a well can be carried.
    max_km=5.0,
    headlines=("boundary_condition", "distributions"))


# ═════════════════════════════════════════════════════════════════════════════
# 1.  THE MODEL SIDE — from the packaged rows, and nothing else
# ═════════════════════════════════════════════════════════════════════════════
def _crossing_depth(depth_m: List[float], pressure_pa: List[float]) -> Optional[float]:
    """The water table: the top of the saturated zone that reaches the bottom
    of the column, found BOTTOM-UP and linearly interpolated between the last
    saturated cell and the first unsaturated one above it.

    Bottom-up, not top-down, so a saturated pocket near the surface — rain
    applied faster than the top cells can pass it — is not read as the water
    table. None when even the bottom cell is unsaturated (the water table is
    below the domain); 0.0 when the column is saturated to the surface."""
    cells = [(z, p) for z, p in zip(depth_m, pressure_pa)
             if z is not None and p is not None]
    if not cells:
        return None
    cells.sort(key=lambda c: c[0])                  # top-down by depth
    if cells[-1][1] < ATMOSPHERIC_PA:
        return None                                 # bottom cell unsaturated
    below_z, below_p = cells[-1]
    for z, p in reversed(cells[:-1]):               # walk up from the bottom
        if p < ATMOSPHERIC_PA:
            if below_p == p:
                return round(float(below_z), 3)
            f = (ATMOSPHERIC_PA - p) / (below_p - p)
            return round(float(z + f * (below_z - z)), 3)
        below_z, below_p = z, p
    return 0.0                                      # saturated to the surface


def _snapshot_dates(times_y: List[float], forcing_start: Optional[int],
                    n_steps: Optional[int]) -> Dict[float, str]:
    """{time_y: 'YYYY-MM-DD'} for the snapshots that fall in the forcing year."""
    if not times_y or not forcing_start or not n_steps:
        return {}
    t_last = max(times_y)
    t0 = t_last - float(n_steps) / DAYS
    jan1 = _dt.date(int(forcing_start), 1, 1)
    out: Dict[float, str] = {}
    for t in times_y:
        if t < t0 - 1e-6:
            continue                                # spin-up: no calendar date
        day = int(round((t - t0) * DAYS))
        day = min(max(day, 0), DAYS - 1)            # t0 + 1 y is Dec 31, not Jan 1
        out[t] = (jan1 + _dt.timedelta(days=day)).isoformat()
    return out


def model_series(rows: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """{case_name: {dates, values, given_m, crossings, moved_m, ...}}.

    `dates`/`values` are the crossing depths at the snapshots inside the
    forcing year — the series the pairing joins on. Every column with a
    profile block appears, whether or not it has dates: a steady column has
    a given water table and crossings at its output times, but no calendar.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        name = row.get("case_name")
        prof = row.get("profiles") or {}
        if not name or not isinstance(prof, dict):
            continue
        times = prof.get("times_y") or []
        depth = prof.get("depth_m") or []
        press = prof.get("liquid_pressure_Pa") or []
        crossings: List[Tuple[float, Optional[float]]] = []
        for t, layer in zip(times, press):
            if isinstance(layer, list):
                crossings.append((float(t), _crossing_depth(depth, layer)))
        if not crossings and row.get("water_table_m") is None:
            continue
        dated = _snapshot_dates(times, row.get("forcing_start"),
                                row.get("n_forcing_steps")) \
            if row.get("transient") else {}
        dates, values = [], []
        for t, z in crossings:
            if t in dated and z is not None:
                dates.append(dated[t]); values.append(z)
        zs = [z for t, z in crossings if z is not None]
        moved = round(max(zs) - min(zs), 3) if len(zs) >= 2 else None
        # how far the crossing moved DURING the forcing year only
        zy = [z for (t, z) in crossings if t in dated and z is not None]
        moved_year = round(max(zy) - min(zy), 3) if len(zy) >= 2 else None
        out[name] = {
            "dates": dates, "values": values,
            "given_m": row.get("water_table_m"),
            "crossings": [{"time_y": t, "depth_m": z} for t, z in crossings],
            "moved_m": moved, "moved_in_year_m": moved_year,
            "transient": bool(row.get("transient")),
            "lat": row.get("lat"), "lon": row.get("lon"),
            "elevation_m": row.get("elevation_m"),
            "pinned": row.get("pinned"), "station_id": row.get("station_id"),
            "station_variable": row.get("station_variable"),
        }
    return out


def _boundary_condition(model: Dict[str, Dict]) -> Dict[str, Any]:
    """The fact that reframes every pair below: the water table is anchored at the bottom face, not held."""
    per = {c: {"given_m": m.get("given_m"), "moved_m": m.get("moved_m"),
               "moved_in_year_m": m.get("moved_in_year_m"),
               "transient": m.get("transient")} for c, m in model.items()}
    moves = [m["moved_m"] for m in model.values() if m.get("moved_m") is not None]
    moves_y = [m["moved_in_year_m"] for m in model.values()
               if m.get("moved_in_year_m") is not None]
    return {
        "note": ("the bottom face of every column is a FIXED HEAD — hydrostatic "
                 "at the CONUS2 water-table depth — and the column starts "
                 "hydrostatic about it. The water table itself is solved: it is "
                 "where the pressure crosses atmospheric, and it moves with the "
                 "recharge, anchored to that pressure at the bottom face. "
                 "moved_m is across ALL output times (spin-up included, so it "
                 "shows the mounding the spin-up produced); moved_in_year_m is "
                 "within the forcing year. A well beside a column therefore "
                 "speaks to the given CONUS2 depth and to the model's own water "
                 "table, and `level` reports both"),
        "n_columns": len(model),
        "n_transient": sum(1 for m in model.values() if m.get("transient")),
        "n_steady": sum(1 for m in model.values() if not m.get("transient")),
        "max_move_m": (max(moves) if moves else None),
        "max_move_in_forcing_year_m": (max(moves_y) if moves_y else None),
        "units": "m",
        "per_column": per,
    }


def _quantiles(values: List[float]) -> Optional[Dict[str, Any]]:
    v = sorted(x for x in values if x is not None and not math.isnan(x))
    if not v:
        return None
    def q(p):
        i = (len(v) - 1) * p
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return round(v[lo] + (v[hi] - v[lo]) * (i - lo), 3)
    return {"n": len(v), "min": v[0], "q25": q(.25), "median": q(.5),
            "q75": q(.75), "max": v[-1]}


def _distributions(model: Dict[str, Dict],
                   reception_json: Optional[str]) -> Dict[str, Any]:
    """Where the water table SITS: the columns' given depths against the
    in-basin Fan wells reception fetched (a long-term mean per site)."""
    out: Dict[str, Any] = {
        "note": ("where the water table SITS, not what it did this year. No "
                 "pairing and no score. `model` is the depth each column was "
                 "given (CONUS2); `fan_2013` is one long-term mean per in-basin "
                 "well from reception's water_table_static block. Two "
                 "distributions of the same quantity over the same basin, "
                 "drawn from different sources"),
        "units": "m below land surface",
        "model": _quantiles([m.get("given_m") for m in model.values()
                             if isinstance(m.get("given_m"), (int, float))]),
    }
    if not reception_json:
        return out
    try:
        payload = json.loads(Path(reception_json).read_text())
    except Exception:                                           # noqa: BLE001
        return out
    obs = payload.get("observations") if "observations" in payload else payload
    fan = ((obs or {}).get("water_table_static") or {}).get("wells") or []
    vals, n_out = [], 0
    for w in fan:
        v = w.get("wtd_m")
        if w.get("in_basin") is False:
            n_out += 1
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
    q = _quantiles(vals)
    if q:
        q["excluded_outside_basin"] = n_out
        out["fan_2013"] = q
    return out


def _fan_values(reception_json: Optional[str]) -> List[float]:
    """The in-basin Fan wells' long-term mean depths, for the figure."""
    if not reception_json:
        return []
    try:
        payload = json.loads(Path(reception_json).read_text())
    except Exception:                                           # noqa: BLE001
        return []
    obs = payload.get("observations") if "observations" in payload else payload
    fan = ((obs or {}).get("water_table_static") or {}).get("wells") or []
    return [float(w["wtd_m"]) for w in fan
            if w.get("in_basin") is not False
            and isinstance(w.get("wtd_m"), (int, float))]


# ═════════════════════════════════════════════════════════════════════════════
# 2.  THE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
def compare(model_columns: List[Dict], observations: Dict, station_meta: Dict,
            reception_json: Optional[str] = None, **kw) -> Dict[str, Any]:
    """One column to one well, then the numbers — plus what the water table is."""
    model = model_series(model_columns)
    rec: Dict[str, Any] = {
        "observable": SPEC.name, "units": SPEC.units,
        "model_comparand": SPEC.comparand, "obs_quantity": SPEC.obs_quantity,
        "colocated": SPEC.colocated,
        "n_columns_with_series": sum(1 for m in model.values() if m["dates"]),
        "n_columns": len(model),
        "pairs": [],
    }
    if not model:
        rec["error"] = ("no column carries a water table or a profile block — "
                        "the extraction has not run. This is an absence of "
                        "model output, not of agreement.")
        return rec

    # ── BEFORE ANY EARLY RETURN — true with or without a well ──────────────
    rec["boundary_condition"] = _boundary_condition(model)
    rec["distributions"] = _distributions(model, reception_json)
    rec["model_period"] = C.span(sorted({d for m in model.values()
                                         for d in m["dates"]}))

    # ── OUTSIDE THE DIVIDE, OUT OF THE COMPARISON — reception's tag, applied ─
    stations = C.stations_for(observations, SPEC.name)
    tags = [(station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
            for sid in stations]
    outside = sorted(sid for sid in stations
                     if (station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
                     is False)
    for sid in outside:
        stations.pop(sid, None)
    rec["stations_excluded_outside_basin"] = outside
    rec["n_stations"] = len(stations)
    synthetic = sorted(sid for sid in stations
                       if (station_meta.get((sid, SPEC.name)) or {}).get("synthetic"))
    if synthetic:
        rec["synthetic_stations"] = synthetic
        rec["synthetic_note"] = (
            f"{len(synthetic)} of the {len(stations)} well(s) here is SYNTHETIC "
            f"— written to exercise this comparison, not measured. Nothing "
            f"computed against it says anything about the basin: "
            f"{', '.join(synthetic)}")
    if sum(1 for t in tags if t is None):
        rec["in_basin_unchecked"] = {
            "n_stations": sum(1 for t in tags if t is None),
            "note": ("these wells carry no in_basin flag, so none was excluded. "
                     "Unchecked is not outside — but nor is it inside.")}
    if not stations:
        rec["error"] = (
            (f"all {len(outside)} well(s) fell outside the watershed and were "
             f"excluded, so no series was compared: {', '.join(outside)}")
            if outside else
            "no well in this domain records a daily water-table series for "
            "this period, so no series was compared. Where the water table "
            "SITS is in `distributions`, and how far it moved off its anchor is in "
            "`boundary_condition`; neither needs a well. This is a fact about "
            "the basin, not a failed comparison.")
        rec["skipped"] = "no recorder wells"
        return rec

    # ── MATCH FIRST — only columns with dated snapshots can be paired ───────
    station_rows = [{"station_id": sid,
                     **{k: (station_meta.get((sid, SPEC.name)) or {}).get(k)
                        for k in ("lat", "lon", "elevation_m")}}
                    for sid in stations]
    column_rows = [{"case_name": case,
                    **{k: m.get(k) for k in
                       ("lat", "lon", "elevation_m", "pinned", "station_id",
                        "station_variable")}}
                   for case, m in model.items() if m["dates"]]
    matched, unpaired, unmatched = C.pair_stations(
        station_rows, column_rows, on=SPEC.pair_on,
        max_delta_m=SPEC.max_delta_m, max_km=SPEC.max_km,
        observable=SPEC.name)
    rec.update({
        "matched_on": SPEC.pair_on, "max_separation_km": SPEC.max_km,
        "max_delta_elevation_m": SPEC.max_delta_m,
        "n_pinned": sum(1 for a in matched if a.get("matched_on") == "pinned"),
        "matched_in_order": ("one pass over the columns in name order; a well "
                             "leaves the pool when a column takes it"),
        "unpaired_stations": unpaired, "unmatched_columns": unmatched,
        "steady_columns_not_paired": sorted(c for c, m in model.items()
                                            if not m["dates"]),
    })

    # ── THEN COMPARE, ONLY WHAT WAS MATCHED ─────────────────────────────────
    for a in matched:
        sid, case = a.get("station_id"), a.get("case_name")
        obs, m = stations.get(sid), model.get(case)
        if not (obs and m):
            continue
        st = station_meta.get((sid, SPEC.name), {})
        entry: Dict[str, Any] = {
            "station_id": sid, "case_name": case,
            "matched_on": a.get("matched_on"),
            "separation_km": a.get("separation_km"),
            "delta_elevation_m": a.get("delta_elevation_m"),
            "obs_period": C.span(obs["dates"]), "n_obs": len(obs["dates"]),
            "in_basin": st.get("in_basin"), "source": st.get("source"),
            "licence": st.get("licence"),
            **({"synthetic": True} if st.get("synthetic") else {}),
        }
        if st.get("units") and st["units"] != SPEC.units:
            entry["units_mismatch"] = (
                f"station reports {st['units']}, this comparison is in "
                f"{SPEC.units} — NOT converted; fix the table")
        # THE LEVEL — the given water table against the well's whole record
        # in the forcing year. Needs no shared date, and is the comparison
        # that actually says something about the CONUS2 field.
        ov = [v for v in obs["values"] if v is not None]
        if ov:
            mean_o = sum(ov) / len(ov)
            lvl: Dict[str, Any] = {
                "obs_mean_m": round(mean_o, 3), "obs_min_m": round(min(ov), 3),
                "obs_max_m": round(max(ov), 3),
                "obs_range_m": round(max(ov) - min(ov), 3),
                "model_moved_in_year_m": m.get("moved_in_year_m"),
                "note": ("two model numbers against the well's mean over its "
                         "record: the CONUS2 depth the column was GIVEN, and "
                         "the mean of the model's own crossing depth over the "
                         "year's snapshots. obs_range_m against "
                         "model_moved_in_year_m is how much each moved"),
            }
            if m.get("given_m") is not None:
                lvl["given_m"] = m["given_m"]
                lvl["given_minus_obs_mean_m"] = round(m["given_m"] - mean_o, 3)
            if m["values"]:
                ym = sum(m["values"]) / len(m["values"])
                lvl["model_year_mean_m"] = round(ym, 3)
                lvl["model_year_mean_minus_obs_mean_m"] = round(ym - mean_o, 3)
            entry["level"] = lvl
        # THE SNAPSHOTS — inner join on the snapshot dates, never interpolated
        dates, mm, oo, qq = C.pair(m, obs)
        if not dates:
            entry.update({"n_days": 0, "note": "no snapshot date has a reading"})
            rec["pairs"].append(entry)
            continue
        entry.update({"n_days": len(dates), "overlap": C.span(dates),
                      "metrics": C.metrics(mm, oo),
                      "obs_quality": C.quality(qq),
                      "snapshots": [{"date": d, "model_m": a_, "obs_m": b_}
                                    for d, a_, b_ in zip(dates, mm, oo)],
                      "n_days_note": ("one point per snapshot inside the "
                                      "forcing year — a column writes a few, "
                                      "not 365; NSE and KGE over so few points "
                                      "are reported, not meaningful")})
        rec["pairs"].append(entry)
    rec["n_pairs_total"] = sum(e.get("n_days", 0) for e in rec["pairs"])
    if not rec["pairs"]:
        rec["error"] = (
            f"no well could be matched to a column within {SPEC.max_km:.0f} km. "
            f"{len(unpaired)} well(s) and {len(unmatched)} column(s) went "
            f"unmatched, each carrying the number that disqualified it. This "
            f"is a siting result, not a disagreement.")
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# 3.  THE FIGURE
# ═════════════════════════════════════════════════════════════════════════════
def plot(rec: Dict, model_columns: List[Dict], observations: Dict,
         out_path: str, **kw) -> Optional[str]:
    """Left: each paired well through the year with the column's snapshots and
    the depth it was given. Right: where the basin's water table sits — the
    columns' given depths against the in-basin Fan wells. The right panel needs
    no well and is drawn for every run; the left says so when it is empty."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt                       # noqa: F401
    except ImportError:
        return None
    model = model_series(model_columns)
    if not model:
        return None
    fig, axes = C.new_figure(2)
    ax, bx = axes[0], axes[1]

    # ── left: pairs ────────────────────────────────────────────────────────
    pairs = [p for p in (rec.get("pairs") or []) if p.get("n_days")]
    stations = C.stations_for(observations, SPEC.name)
    if not pairs:
        ax.text(.5, .5, "no well series to stand a column beside\n"
                        "(where the water table sits: right panel)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
    else:
        for i, p in enumerate(pairs[:4]):
            sid, case = p["station_id"], p["case_name"]
            obs, m = stations.get(sid), model.get(case)
            if not (obs and m):
                continue
            col = ["#2C6A5C", "#B85C38", "#4A6572", "#8A6410"][i % 4]
            od = [_dt.date.fromisoformat(d) for d in obs["dates"]]
            lab_o = f"well {sid}" + (" (SYNTHETIC)" if p.get("synthetic") else "")
            ax.plot(od, obs["values"], color=col, lw=1.2, alpha=.85, label=lab_o)
            md = [_dt.date.fromisoformat(d) for d in m["dates"]]
            ax.plot(md, m["values"], "o", color=col, ms=7, mec="black", mew=.6,
                    label=f"{case} crossing at snapshots")
            if m.get("given_m") is not None and od:
                ax.hlines(m["given_m"], min(od), max(od), colors=col,
                          linestyles="--", lw=1.0, label=f"{case} given (CONUS2)")
        ax.invert_yaxis()
        ax.set_ylabel("depth to water  (m, positive down)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_title("(a) well through the year · column snapshots", loc="left",
                     fontsize=11)
        C.legend(ax)

    # ── right: distributions ───────────────────────────────────────────────
    given = [m["given_m"] for m in model.values()
             if isinstance(m.get("given_m"), (int, float))]
    # the model's own year-mean crossing per transient column, beside the given
    year_mean = [sum(m["values"]) / len(m["values"]) for m in model.values()
                 if m.get("values")]
    fan_vals = _fan_values(kw.get("reception_json"))
    data, labels = [], []
    if given:
        data.append(given); labels.append(f"columns given\n(CONUS2, n={len(given)})")
    if year_mean:
        data.append(year_mean)
        labels.append(f"columns modelled\n(year mean, n={len(year_mean)})")
    if fan_vals:
        data.append(fan_vals)
        labels.append(f"Fan 2013 wells\nin basin (n={len(fan_vals)})")
    if data:
        bx.boxplot(data, widths=.5, showfliers=True)
        bx.set_xticks(range(1, len(labels) + 1))
        bx.set_xticklabels(labels)
        for i, d in enumerate(data, 1):
            bx.plot([i] * len(d), d, "o", ms=4, alpha=.5, color="#2C6A5C")
        bx.invert_yaxis()
        bx.set_ylabel("depth to water  (m)")
        bx.set_title("(b) where the water table sits", loc="left", fontsize=11)
    else:
        bx.set_axis_off()
    fig.suptitle("PFLOTRAN columns — water table against wells (bottom face: fixed "
                 "head at the CONUS2 depth)", fontsize=12)
    return C.save(fig, out_path)
