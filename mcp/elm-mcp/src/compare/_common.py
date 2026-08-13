#!/usr/bin/env python3
"""The half of comparison that is the same for every observable.

Pairing, metrics and quality accounting do not depend on what is being
compared, so they live once here. Everything that DOES depend on the
observable — what a peak date means, whether a water table needs a log axis,
whether a gauge is co-located at all — lives in that observable's own module.

The split is the point. Two copies of an NSE would drift; two copies of "what
snowpack phenology means" never existed to begin with.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Spec:
    """What one observable is, in the few facts every stage needs.

    Small and declarative on purpose: a new observable should be a SPEC plus a
    compare() and a plot(), not a new branch in shared code.
    """

    def __init__(self, name: str, model_vars: List[str], units: str,
                 comparand: str, obs_quantity: str,
                 colocated: bool = True, pair_on: str = "distance",
                 max_delta_m: Optional[float] = None,
                 max_km: Optional[float] = None,
                 headlines: Tuple[str, ...] = ()):
        self.name = name
        self.model_vars = model_vars
        self.units = units
        self.comparand = comparand
        self.obs_quantity = obs_quantity
        self.colocated = colocated
        self.pair_on = pair_on
        # WHICH OF THIS RECORD'S BLOCKS TRAVEL IN THE SUMMARY (2026-08-12).
        # Every observable now computes findings that need no observation —
        # where the water leaves, which water tables never move, where the ET
        # came from — and those are the whole answer in a basin with no
        # station, which is most basins. The summary the MCP returns must carry
        # them or the caller sees "error: no gauge" and nothing else.
        #
        # Declared HERE, as a tuple of top-level keys, so the dispatcher stays
        # dispatch: a fifth observable adds a file and a registry line, never a
        # branch in shared code. Blocks are compacted on the way out, so a key
        # naming a large structure costs a few scalars, not the structure.
        self.headlines = headlines
        # PER-OBSERVABLE PAIRING LIMITS (2026-08-11). These were one shared
        # constant and one shared derived distance, applied identically to all
        # four observables — which is wrong, because how close a station has to
        # be to be comparable is a fact about WHAT WAS MEASURED, not about the
        # basin. A snow pillow and a flux tower are local; a gauge integrates a
        # catchment. None means "no limit of this kind for this observable".
        self.max_delta_m = max_delta_m
        self.max_km = max_km


# ── reading what the caller wrote ────────────────────────────────────────────
# Reception's payload, one entry per observable: which block holds it, which key
# lists the stations, which field is the station's identity, and its units.
#
# THE THREE SHAPES ARE THE SERVERS' OWN and are not normalised upstream, so they
# are read as they are: usgs_water hands back a date->value dict for discharge
# and a list of records for wells, snotel hands back parallel arrays. Units are
# already what compare expects — mm/day specific discharge, metres below land
# surface, mm of water equivalent — so nothing is converted here, and a
# conversion this file does not do is a conversion it cannot get wrong.
BLOCKS = {
    "streamflow": {"block": "streamflow", "key": "stations", "id": "id",
                   "units": "mm/day", "source": "USGS NWIS daily values",
                   "licence": "public domain (US Government)"},
    # `units` is the CANONICAL symbol, matched against the Spec's; the prose
    # goes in `units_detail`. They used to be one field, so this observable
    # declared "m below land surface" against a Spec that said "m" and every
    # well in every basin came back carrying a units_mismatch warning about a
    # unit that matched perfectly — 56 of them at Naches. A guard that cries
    # wolf on every station is how a real mismatch gets scrolled past.
    #
    # KEY AND BLOCK ARE THE SAME WORD NOW (2026-08-13). This entry read
    # "wtd" -> block "water_table", and that one-word gap was a live bug: the
    # sampler records a pinned column's target as `station_variable:
    # "water_table"`, the Spec was called "wtd", and pair_stations compared the
    # two. They never matched, so every designed well pin was silently refused
    # and the column fell through to distance matching — `n_pinned: 0` on a run
    # with four pinned wells. One name for one observable, everywhere.
    "water_table": {"block": "water_table", "key": "wells", "id": "id",
                    "units": "m",
                    "units_detail": "metres below land surface",
                    "source": "USGS NWIS daily groundwater levels "
                              "(recorder wells)",
                    "licence": "public domain (US Government)"},
    "swe": {"block": "swe", "key": "stations", "id": "triplet", "units": "mm",
            "source": "USDA NRCS SNOTEL (AWDB)",
            "licence": "public domain (US Government)"},
    "et": {"block": "et", "key": "towers", "id": "id", "units": "mm/day",
           "source": "AmeriFlux",
           "licence": "see per-site AmeriFlux data policy"},
}

META_PASSTHROUGH = ("name", "lat", "lon", "elevation_m", "in_basin",
                    "drainage_area_km2", "n_days", "n_obs", "igbp")

# Value fields a record-list may use, in the order they are tried.
_VALUE_KEYS = ("value", "wtd_m", "et_mm_day", "mm_day", "swe_mm")


def _station_series(st: Dict) -> List[Tuple[str, float]]:
    """(date, value) pairs out of whichever shape this station used."""
    got: List[Tuple[Any, Any]] = []
    daily = st.get("daily")
    if isinstance(daily, dict) and daily.get("dates") is not None:
        got = list(zip(daily.get("dates") or [], daily.get("values") or []))
    else:
        for field in ("mm_day", "series", "daily"):
            blob = st.get(field)
            if isinstance(blob, dict):
                got = list(blob.items())
                break
            if isinstance(blob, list):
                for r in blob:
                    if not isinstance(r, dict):
                        continue
                    v = next((r[k] for k in _VALUE_KEYS if r.get(k) is not None),
                             None)
                    got.append((r.get("date") or r.get("time"), v))
                break
    return got


# PAIRING LIMITS ARE DECLARED PER OBSERVABLE ON THE SPEC, AND NOWHERE ELSE.
# `max_km` and `max_delta_m` are properties of what was measured. Nothing about
# the model setup or the driving data may set them. An observable with no limit
# declared gets no limit, and the record says so rather than borrowing a number.


def load_domain(reception_json: str) -> Dict[str, Any]:
    """The modelled domain, from the same file the observations came from.

    Only what a comparison needs to interpret itself. A gauge's drainage area
    means nothing alone — 10,285 km² is either the basin or fifty times it —
    so the basin's own area travels with the observations rather than being
    passed in separately and getting out of step with them.

    Nothing about the model setup belongs in this record. A pairing limit is a
    property of the observation and is declared on the Spec; a resolution
    borrowed from the driving data is not a statement about a station.
    """
    try:
        payload = json.loads(Path(reception_json).read_text())
    except Exception:                                           # noqa: BLE001
        return {}
    brief = payload.get("brief") or payload
    dom = brief.get("domain") or payload.get("domain") or {}
    return {k: dom.get(k) for k in ("name", "huc", "area_km2")
            if dom.get(k) is not None}


def load_observations(reception_json: str) -> Tuple[Dict, Dict, int]:
    """(series, meta, n_dropped) straight out of reception's own payload.

    RECEPTION IS THE SOURCE, AND THE ONLY ONE. It calls the four data servers
    once, after the period is fixed, "and never again — the validator reads this
    rather than re-querying, so the two cannot disagree" — and it persists the
    whole thing, daily series and all, under `observations` in reception.json.
    This reads that. To refresh the observations you re-run reception, which is
    the one place allowed to reach outside.

    There was briefly a second path: a framework tool that re-queried the same
    four servers and rewrote their answers as a long CSV for this function to
    parse. Measured 2026-08-10 on elm_run_20260806_162707, that round trip
    returned 14,102 values across 36 series, every one identical to what
    reception had already written — a full network fetch and a second on-disk
    copy to arrive back where it started. Deleted. One format, one source.

    series is {(station_id, variable): {dates, values, quality}}, chronological.
    A point with no parseable value is DROPPED AND COUNTED, never coerced to
    zero: a missing snow measurement is not zero snow.

    `quality` comes through as the station reported it and defaults to
    "measured", which is what these three sources return — USGS and SNOTEL
    publish measurements, not model fills. It stays in the structure because ET
    will need it: at US-NR1 in 2016 the gap-filled annual total is 464 mm
    against 89 mm from the measured half-hours alone, and a table that cannot
    tell those apart cannot be read.
    """
    payload = json.loads(Path(reception_json).read_text())
    # Accept the whole reception file or just its observations block, so this
    # also reads a payload someone has already pulled out.
    obs = payload.get("observations") if "observations" in payload else payload

    series: Dict[Tuple[str, str], Dict[str, List]] = {}
    meta: Dict[Tuple[str, str], Dict] = {}
    dropped = 0

    for var, spec in BLOCKS.items():
        block = obs.get(spec["block"]) or {}
        for st in (block.get(spec["key"]) or []):
            sid = st.get(spec["id"])
            if not sid:
                dropped += len(_station_series(st))
                continue
            sid = str(sid)
            s = {"dates": [], "values": [], "quality": []}
            for d, v in _station_series(st):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    fv = float("nan")
                if not d or math.isnan(fv):
                    dropped += 1
                    continue
                s["dates"].append(str(d)[:10])
                s["values"].append(fv)
                s["quality"].append(str(st.get("quality") or "measured").lower())
            if not s["dates"]:
                continue
            order = sorted(range(len(s["dates"])), key=lambda i: s["dates"][i])
            for k in ("dates", "values", "quality"):
                s[k] = [s[k][i] for i in order]
            series[(sid, var)] = s
            # in_basin IS READ, NEVER COMPUTED. Reception tags every station
            # against the watershed polygon (_tag_in_basin) and persists the
            # flag with the payload; the planner reads the same flag to decide
            # what it may pin. Recomputing it here would make two
            # implementations of one fact, and the second would drift.
            meta[(sid, var)] = {
                "station_id": sid, "variable": var, "units": spec["units"],
                "source": spec["source"], "licence": spec["licence"],
                **({"units_detail": spec["units_detail"]}
                   if spec.get("units_detail") else {}),
                **{k: st.get(k) for k in META_PASSTHROUGH
                   if st.get(k) is not None}}
    return series, meta, dropped


def stations_for(series: Dict, name: str) -> Dict[str, Dict]:
    """{station_id: series} for one observable."""
    return {k[0]: v for k, v in series.items()
            if isinstance(k, tuple) and k[1] == name}


def model_series(rows: List[Dict], variables: List[str]) -> Dict[str, Dict]:
    """{case_name: {dates, values, ...}}, the named ELM variables summed.

    A date survives only when EVERY component has a value there. A partial sum
    is a smaller number, not a missing one, and it would read as the model
    drying out rather than as an absent field.
    """
    out: Dict[str, Dict] = {}
    for row in rows or []:
        name = row.get("case_name")
        if not name:
            continue
        parts = []
        for var in variables:
            d = ((row.get("variables") or {}).get(var) or {}).get("daily") or {}
            if not (d.get("dates") and d.get("values")):
                parts = []
                break
            parts.append(dict(zip(d["dates"], d["values"])))
        if not parts:
            continue
        common = set(parts[0])
        for p in parts[1:]:
            common &= set(p)
        dates = sorted(d for d in common
                       if all(p.get(d) is not None for p in parts))
        if not dates:
            continue
        out[name] = {"dates": dates,
                     "values": [sum(p[d] for p in parts) for d in dates],
                     "lat": row.get("lat"), "lon": row.get("lon"),
                     "elevation_m": row.get("elevation_m"),
                     # Carried for pair_stations: a pinned column's partner was
                     # chosen by the sampler, not by geometry.
                     "pinned": row.get("pinned"),
                     "station_id": row.get("station_id"),
                     "station_variable": row.get("station_variable")}
    return out


# ── pairing ──────────────────────────────────────────────────────────────────
def pair(m: Dict, o: Dict) -> Tuple[List, List, List, List]:
    """Model and observation on their SHARED dates: (dates, model, obs, quality).

    An inner join, never interpolation. Filling a gap to make the arrays line up
    invents a measurement, and the pair count would then describe the invention.
    """
    mv = dict(zip(m["dates"], m["values"]))
    dates, mm, oo, qq = [], [], [], []
    for i, d in enumerate(o["dates"]):
        if d in mv:
            dates.append(d)
            mm.append(mv[d])
            oo.append(o["values"][i])
            qq.append(o["quality"][i] if i < len(o.get("quality") or []) else "unknown")
    return dates, mm, oo, qq


def pair_stations(stations: List[Dict], columns: List[Dict],
                  on: str = "distance",
                  max_delta_m: Optional[float] = None,
                  max_km: Optional[float] = None,
                  observable: str = "") -> Tuple[List, List, List]:
    """One column, one station. A COLUMN LOOP, and a station is used once.

    ONE PASS OVER THE COLUMNS (2026-08-11, user's rule). This used to be three
    passes — pins, then a station-major scan, then a global sort that decided
    who claimed a contested column first — and all three existed to enforce a
    single thing: the bijection. Walk the COLUMNS instead and the bijection
    falls out of the loop for free, because a claimed station simply leaves the
    pool. The pin needs no pass of its own either: `pinned` is a field ON THE
    COLUMN, so a column loop reads it where it already is.

    Nothing was protecting a score. Every comparison here is CONTEXT, not a
    skill claim, so there was no double-counting to defend against — only
    complexity defending a rule that could be expressed as one line of loop.

    THE ORDER IS THE COLUMNS' NAME ORDER, and that is load-bearing: col_01 asks
    before col_02, so when two columns want the same station the earlier name
    takes it. A global sort by distance would be order-independent; this is not.
    It is sorted explicitly rather than left to dict order so the same inputs
    always give the same pairing, and `matched_in_order` says so in the record.

    THE DESIGNED PIN WINS, AND NOTHING ELSE IS CONSULTED FOR IT. When a column
    was PINNED at a station by the sampler that pairing is a DECISION already
    made upstream, recorded in columns.json as `pinned` + `station_id` +
    `station_variable`. No distance test, no elevation test. Re-deriving it
    geometrically was guessing at an answer we were already told.

    OTHERWISE: two constraints, then a choice. A station qualifies for a column
    only if it is within `max_km` AND, for elevation matching, within
    `max_delta_m`. Among the qualifiers the CLOSEST wins — elevation FILTERS,
    distance DECIDES (2026-08-11). Nothing qualifies, and the column stands
    against nothing: there is no least-bad pair, and manufacturing one is how a
    comparison ends up describing two places with nothing to do with each other.

    Limits are PER OBSERVABLE and arrive on the Spec — a snow pillow, a well and
    a catchment gauge are not the same kind of measurement, and for a while they
    shared one pair of numbers borrowed from the forcing grid.

    Returns (pairs, unpaired_stations, unmatched_columns). BOTH leftovers are
    reported, and each carries the number that explains it: excluded by a rule
    and never considered are different findings.
    """
    def elev_delta(st, col) -> Optional[float]:
        if st.get("elevation_m") is None or col.get("elevation_m") is None:
            return None
        return abs(col["elevation_m"] - st["elevation_m"])

    def qualifies(d_km, de) -> bool:
        # AN UNKNOWN DISTANCE DOES NOT DISQUALIFY, but it cannot RANK either, so
        # a station without coordinates never wins a column. Same rule the
        # in_basin tag follows: unchecked is not failed.
        if d_km is None:
            return False
        if max_km is not None and d_km > max_km:
            return False
        if on == "elevation" and max_delta_m is not None:
            if de is None or de > max_delta_m:
                return False
        return True

    by_station = {s.get("station_id"): s for s in stations
                  if s.get("station_id")}
    pool = set(by_station)
    pairs, unmatched_columns = [], []

    for col in sorted(columns, key=lambda c: str(c.get("case_name") or "")):
        case = col.get("case_name")
        sid = col.get("station_id")

        # ── the designed pin, taken as given ─────────────────────────────
        #
        # `station_variable` IS AN OBSERVABLE NAME, and it is now the SAME
        # vocabulary this Spec is named in. It was not: the sampler wrote
        # "water_table" and the Spec was called "wtd", so this test was false
        # for every well pin ever designed and the column dropped through to
        # geometric matching — the record said `n_pinned: 0` on a run whose
        # columns.json named four pinned wells, and nothing anywhere reported a
        # refusal. A silent mismatch between two spellings of one thing.
        if (col.get("pinned") and sid in pool
                and (not observable
                     or col.get("station_variable") in (None, observable))):
            st = by_station[sid]
            p = {"station_id": sid, "case_name": case, "matched_on": "pinned",
                 "note": "the sampler placed this column at this station; the "
                         "pairing is a design decision, not a geometric match"}
            d_km, de = separation_km(st, col), elev_delta(st, col)
            if d_km is not None:
                p["separation_km"] = round(d_km, 2)
            if de is not None:
                p["delta_elevation_m"] = round(
                    (col.get("elevation_m") or 0)
                    - (st.get("elevation_m") or 0), 1)
            pairs.append(p)
            pool.discard(sid)
            continue

        # ── otherwise the nearest station still unclaimed ────────────────
        #
        # SORTED, BECAUSE A SET IS NOT ORDERED (2026-08-12). `pool` is a set of
        # station ids, and iterating it walked them in hash order — which Python
        # varies BETWEEN PROCESSES. The winner is chosen with a strict `<`, so
        # any exact tie in distance went to whichever station the seed happened
        # to visit first. Brandywine has a well nest — four wells within metres
        # of each other and three columns at the identical point, since they
        # were pinned into one gridcell — and the same run directory compared
        # three times produced three different pairings, each with different
        # NSE and bias attached to the same column. The docstring above already
        # promised "the same inputs always give the same pairing"; the column
        # loop was sorted and this one was not.
        best, nearest_km, best_de = None, None, None
        for s_id in sorted(pool):
            st = by_station[s_id]
            d_km, de = separation_km(st, col), elev_delta(st, col)
            if d_km is not None and (nearest_km is None or d_km < nearest_km):
                nearest_km = d_km
            if de is not None and (best_de is None or de < best_de):
                best_de = de
            if qualifies(d_km, de) and (best is None or d_km < best[0]):
                best = (d_km, st)

        if best is None:
            why = []
            if nearest_km is None:
                why.append("no station left in the pool has coordinates")
            elif max_km is not None and nearest_km > max_km:
                why.append(f"nearest unclaimed station is {nearest_km:.1f} km "
                           f"away, beyond the {max_km:.1f} km limit")
            if on == "elevation" and max_delta_m is not None \
                    and (best_de is None or best_de > max_delta_m):
                why.append(f"nearest in elevation is {best_de:.0f} m away, "
                           f"beyond the {max_delta_m:.0f} m limit"
                           if best_de is not None else
                           "no station has a comparable elevation")
            if not why and not pool:
                why.append("every station was already claimed by an earlier "
                           "column")
            unmatched_columns.append({
                "case_name": case,
                "reason": "; ".join(why) or "no station qualified",
                "nearest_station_km": (round(nearest_km, 2)
                                       if nearest_km is not None else None),
                "nearest_delta_elevation_m": (round(best_de, 1)
                                              if best_de is not None else None)})
            continue

        d_km, st = best
        p = {"station_id": st["station_id"], "case_name": case,
             "matched_on": "distance", "separation_km": round(d_km, 2)}
        if on == "elevation":
            # Elevation FILTERED this pair even though distance chose it, so the
            # offset it passed on is still part of what the pairing is.
            p["delta_elevation_m"] = round(
                (col.get("elevation_m") or 0) - (st.get("elevation_m") or 0), 1)
        pairs.append(p)
        pool.discard(st["station_id"])

    # ── the stations nobody took ─────────────────────────────────────────
    # NO COORDINATES IS ITS OWN FINDING, not "no column qualified". They are
    # different problems with different fixes — one is a station too far from
    # the design, the other is an observation that arrived unusable — and
    # collapsing them is what let 236 coordinate-less Naches wells read as a
    # pairing failure for a whole afternoon on 2026-08-11.
    unpaired = []
    for s_id in sorted(pool):
        st = by_station[s_id]
        nearest_km, best_de = None, None
        for col in columns:
            d_km, de = separation_km(st, col), elev_delta(st, col)
            if d_km is not None and (nearest_km is None or d_km < nearest_km):
                nearest_km = d_km
            if de is not None and (best_de is None or de < best_de):
                best_de = de
        if st.get("lat") is None or st.get("lon") is None:
            why = ("station has no coordinates, so it cannot be placed against "
                   "any column")
        elif max_km is not None and nearest_km is not None \
                and nearest_km > max_km:
            why = (f"nearest column is {nearest_km:.1f} km away, beyond the "
                   f"{max_km:.1f} km limit")
        elif on == "elevation" and max_delta_m is not None \
                and (best_de is None or best_de > max_delta_m):
            why = (f"nearest column in elevation is {best_de:.0f} m away, "
                   f"beyond the {max_delta_m:.0f} m limit"
                   if best_de is not None else
                   "no column has a comparable elevation")
        else:
            # Within every limit and still unused: one station per column, and
            # the columns that could have taken it took someone nearer first.
            why = ("within the limits, but every column that could have taken "
                   "it chose a nearer station — one column, one station")
        unpaired.append({"station_id": s_id, "reason": why,
                         "nearest_column_km": (round(nearest_km, 2)
                                               if nearest_km is not None
                                               else None),
                         "nearest_delta_elevation_m": (
                             round(best_de, 1) if best_de is not None else None)})
    return pairs, unpaired, unmatched_columns


def separation_km(a: Dict, b: Dict) -> Optional[float]:
    """Great-circle distance between two lat/lon points, or None if either lacks one."""
    if any(x.get("lat") is None or x.get("lon") is None for x in (a, b)):
        return None
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = p2 - p1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * 6371.0 * math.asin(math.sqrt(h)), 1)


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(m: List[float], o: List[float]) -> Dict[str, Any]:
    """bias · MAE · RMSE · r · NSE · KGE. No thresholds, no verdicts.

    NSE and KGE are undefined on a zero-variance observation — a well that never
    moved, a season with no snow — and returning 0.0 would be a made-up score.
    They come back None, so "no skill" and "not computable" stay distinct.
    """
    n = len(m)
    if n == 0:
        return {"n": 0}
    mean_m, mean_o = sum(m) / n, sum(o) / n
    var_m = sum((a - mean_m) ** 2 for a in m)
    var_o = sum((b - mean_o) ** 2 for b in o)
    cov = sum((a - mean_m) * (b - mean_o) for a, b in zip(m, o))
    r = cov / math.sqrt(var_m * var_o) if var_m > 0 and var_o > 0 else None
    sse = sum((a - b) ** 2 for a, b in zip(m, o))
    nse = 1.0 - sse / var_o if var_o > 0 else None
    kge = None
    if r is not None and mean_o != 0 and var_o > 0:
        kge = 1.0 - math.sqrt((r - 1) ** 2
                              + (math.sqrt(var_m / n) / math.sqrt(var_o / n) - 1) ** 2
                              + (mean_m / mean_o - 1) ** 2)
    rnd = lambda x: None if x is None else round(x, 4)          # noqa: E731
    return {"n": n, "model_mean": rnd(mean_m), "obs_mean": rnd(mean_o),
            "bias": rnd(mean_m - mean_o),
            "mae": rnd(sum(abs(a - b) for a, b in zip(m, o)) / n),
            "rmse": rnd(math.sqrt(sse / n)),
            "pearson_r": rnd(r), "nse": rnd(nse), "kge": rnd(kge)}


MEASURED = ("measured", "approved")


def quality(q: List[str]) -> Dict[str, Any]:
    """How much of the paired observation was actually measured.

    At US-NR1 in 2016 the gap-filled annual ET is 464 mm and the measured
    half-hours alone give 89 mm, because 36% of the year was observed. Both are
    true; they are not the same claim.
    """
    counts: Dict[str, int] = {}
    for x in q:
        counts[x] = counts.get(x, 0) + 1
    n = len(q) or 1
    return {"counts": counts,
            "frac_measured": round(sum(counts.get(k, 0) for k in MEASURED) / n, 4),
            "frac_gap_filled": round(counts.get("filled", 0) / n, 4)}


def span(dates: List[str]) -> Optional[List[str]]:
    return [dates[0], dates[-1]] if dates else None


# ── plotting helpers ─────────────────────────────────────────────────────────
def new_figure(ncols: int = 2, width: float = 13.0, height: float = 4.6):
    """A figure with this project's plot conventions already applied.

    Large type, and NOTHING interpretive on the canvas: axis labels name the
    quantity and its units, titles name what is drawn. What a panel MEANS
    belongs in the record, which is numbers, and in whatever reads it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                         "axes.titlesize": 13, "legend.fontsize": 9})
    return plt.subplots(1, ncols, figsize=(width, height))


def legend(ax):
    """Framed, always. A frameless legend inside the axes draws its sample
    markers on the same ground as the data, and one early version of these
    figures showed a legend swatch that read as an extra observation."""
    ax.legend(frameon=True, facecolor="white", framealpha=0.92,
              edgecolor="#CCCCCC", loc="best")


def save(fig, out_path: str) -> str:
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out_path)


def _hillshade(ax, extent: List[float]) -> bool:
    """Esri's World Hillshade, through the framework's one implementation.

    THE SERVER READING THE FRAMEWORK, the way water_table.py already reads
    core.static_wtd for the modelled water table. The alternative was a second
    copy of the tile maths inside this package, which is the thing that drifts:
    the design figure and the results figure would slowly stop showing the same
    ground. Failure is silent and returns False — the caller has a backdrop that
    needs no network to fall back on.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
        from core.basemap import paste
        return paste(ax, extent, quiet=True)
    except Exception:                                           # noqa: BLE001
        return False


def basin_backdrop(ax, reception_json: Optional[str],
                   scale_bar: bool = True, corner: str = "auto",
                   avoid: Optional[List[Tuple[float, float]]] = None) -> bool:
    """Terrain and the watershed outline under a map panel. Returns what it drew.

    THE SAME BACKDROP sampling_design.png USES — Esri's World Hillshade under
    the panel, the DEM points reception sampled as small grey dots, and the
    watershed ring over both — so a results map and the design map show the same
    ground and can be read side by side. The hillshade itself comes from
    core.basemap, which is where it lives for both figures.

    A MISSING BACKDROP IS NOT AN ERROR, and it is not a blank panel either. The
    tiles need the network and a cert store, and maps.py refused a backdrop
    outright on those grounds — a figure that silently renders without its
    relief is worse than one that never had one. So when the fetch fails this
    falls back to something with no network in it at all: the same DEM samples
    filled as terrain. Naches carries 58 of them in `grid.points` and a
    113-point ring in `grid.boundary`, so on a compute node with no route out
    the panel still has ground under it, drawn from the run's own file.

    The ground gets NO COLOURBAR either way. It is ground, not a measurement;
    the panel's own quantity owns the colour key, and two scales on one map is
    how a reader ends up reading elevation as the result.
    """
    if not reception_json:
        return False
    try:
        grid = (json.loads(Path(reception_json).read_text()).get("grid")) or {}
    except Exception:                                           # noqa: BLE001
        return False
    pts = grid.get("points") or []
    rings = grid.get("boundary") or []
    if not pts and not rings:
        return False

    lon = [p["lon"] for p in pts if p.get("lon") is not None]
    lat = [p["lat"] for p in pts if p.get("lat") is not None]
    elev = [p.get("elevation_m") for p in pts if p.get("lon") is not None]

    bx = [p[0] for r in rings for p in r] or lon
    by = [p[1] for r in rings for p in r] or lat
    shaded = False
    if bx and by:
        mx, my = 0.05 * (max(bx) - min(bx)), 0.05 * (max(by) - min(by))
        shaded = _hillshade(ax, [min(bx) - mx, max(bx) + mx,
                                 min(by) - my, max(by) + my])
    if shaded:
        # Relief underneath, so the DEM samples are marks rather than a surface.
        if lon:
            ax.scatter(lon, lat, s=5, c="0.45", zorder=1)
    elif len(lon) >= 4 and all(e is not None for e in elev):
        try:
            ax.tricontourf(lon, lat, elev, levels=12, cmap="terrain",
                           alpha=0.55, zorder=0)
        except Exception:                                       # noqa: BLE001
            # Collinear samples defeat the triangulation. The points themselves
            # still say where the ground was measured.
            ax.scatter(lon, lat, s=10, c="0.75", zorder=0)
    elif lon:
        ax.scatter(lon, lat, s=10, c="0.75", zorder=0)

    for ring in rings:
        ax.plot([p[0] for p in ring], [p[1] for p in ring],
                color="0.15" if shaded else "navy", lw=1.6, zorder=4)

    ys = lat + [p[1] for r in rings for p in r]
    xs = lon + [p[0] for r in rings for p in r]
    if not (xs and ys):
        return True
    mid = (min(ys) + max(ys)) / 2
    ax.set_aspect(1.0 / max(math.cos(math.radians(mid)), 1e-3))
    if scale_bar:
        km_per_deg = 111.32 * math.cos(math.radians(mid))
        width_km = (max(xs) - min(xs)) * km_per_deg
        bar = max([k for k in (1, 2, 5, 10, 20, 50, 100)
                   if k <= 0.3 * width_km] or [1])
        # WHICHEVER BOTTOM CORNER IS EMPTIER, measured against the box the bar
        # will actually occupy. The bar is furniture and the data is not, and at
        # Naches the two wettest columns sit exactly where a left-hand bar goes.
        # `avoid` is the caller's points; the box runs a little past the bar's
        # right end because a point there carries a label that reaches over it.
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        bar_deg = bar / km_per_deg
        side = {"left": min(xs) + 0.05 * w, "right": max(xs) - 0.05 * w - bar_deg}
        if corner not in side:
            def hits(x0: float) -> int:
                return sum(1 for px, py in (avoid or [])
                           if x0 - 0.02 * w <= px <= x0 + bar_deg + 0.10 * w
                           and py <= min(ys) + 0.14 * h)
            corner = min(("left", "right"), key=lambda k: hits(side[k]))
        x0 = side[corner]
        y0 = min(ys) + 0.04 * (max(ys) - min(ys))
        ax.plot([x0, x0 + bar / km_per_deg], [y0, y0], color="k", lw=2.5,
                solid_capstyle="butt", zorder=6)
        import matplotlib.patheffects as pe
        ax.text(x0 + bar / km_per_deg / 2, y0 + 0.015 * (max(ys) - min(ys)),
                f"{bar} km", ha="center", fontsize=8, zorder=6,
                # A haloed label survives a data label landing on it. Panel
                # furniture must stay readable without owning empty space.
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    return True


def column_map(fig, ax, model: Dict, value_of, label: str,
               reception_json: Optional[str] = None, cmap: str = "plasma",
               title: str = "") -> int:
    """Every column in its place, coloured by one number and named. Returns n.

    THE PANEL ALL THREE OBSERVABLES NOW END ON (2026-08-12, user's call): swe,
    water_table and streamflow each gave up a 1:1 scatter of matched days for
    this. The
    scatter answered "how close are the pairs" for the two or three columns that
    had a station; this answers "what did the other fourteen do, and where", and
    a basin with no station at all still gets it.

    value_of is called with (case_name, series) and returns the number to
    colour by, or None to leave that column out.

    NAMED, EVERY ONE, because the point is to take a column out of the panel
    beside this one and find it here. Seventeen entries in a legend cannot do
    that. The labels are nudged apart greedily: three Naches columns share a
    latitude to two decimal places and their labels landed on top of each other,
    reading as "col_16 ol_07 ol_04". Deterministic, and good enough at
    seventeen points, where a real label-placement solver would be more
    machinery than the problem deserves.
    """
    import matplotlib.patheffects as pe
    from matplotlib.ticker import MaxNLocator

    pts = [(c, m.get("lon"), m.get("lat"), value_of(c, m))
           for c, m in model.items()]
    pts = [p for p in pts
           if p[1] is not None and p[2] is not None and p[3] is not None]
    basin_backdrop(ax, reception_json, avoid=[(p[1], p[2]) for p in pts])
    if not pts:
        ax.text(0.5, 0.5, "no column has coordinates", ha="center", va="center",
                transform=ax.transAxes, color="#77837F", fontsize=9)
        return 0

    sc = ax.scatter([p[1] for p in pts], [p[2] for p in pts],
                    c=[p[3] for p in pts], cmap=cmap, s=90,
                    edgecolor="white", linewidth=1.0, zorder=5)
    # HOW CLOSE IS TOO CLOSE IS A FACT ABOUT THE PANEL, not about the columns.
    # Sized from the columns' own span, the thresholds shrank the moment the
    # hillshade widened the axes past the basin, and the clusters went back to
    # overlapping. dataLim carries everything drawn — tiles, ring, points.
    span_x = ax.dataLim.width or (max(p[1] for p in pts)
                                  - min(p[1] for p in pts)) or 1.0
    span_y = ax.dataLim.height or (max(p[2] for p in pts)
                                   - min(p[2] for p in pts)) or 1.0
    near_x, near_y = 0.15 * span_x, 0.042 * span_y
    taken: List[Tuple[float, float]] = []
    for c, lo, la, _ in sorted(pts, key=lambda t: (-t[2], t[1])):
        tx, ty, step = lo, la, 0
        while any(abs(tx - px) < near_x and abs(ty - py) < near_y
                  for px, py in taken) and step < 8:
            step += 1
            ty -= near_y                # stack downwards, away from the point
        taken.append((tx, ty))
        ax.annotate(c, (lo, la), xytext=(tx + 0.012 * span_x, ty),
                    fontsize=7.5, zorder=6, va="center",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")],
                    arrowprops=(dict(arrowstyle="-", lw=0.6, color="0.35",
                                     shrinkA=0, shrinkB=2) if step else None))
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(label)
    if ax.get_aspect() == "auto":
        # No backdrop, so nothing has set the aspect: a map of degrees drawn on
        # square axes is a stretched scatter with its labels off the edge.
        mid = sum(p[2] for p in pts) / len(pts)
        ax.set_aspect(1.0 / max(math.cos(math.radians(mid)), 1e-3))
        ax.margins(x=0.18, y=0.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"where those columns are ({len(pts)} of "
                          f"{len(model)} placed)")
    return len(pts)


def one_to_one(ax, xs: List[float], ys: List[float]) -> None:
    """A 1:1 line over the data's own range, with equal axes."""
    vals = [v for v in list(xs) + list(ys) if v is not None]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = 0.06 * ((hi - lo) or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", lw=1, color="#666", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")


def split_quality(qq: List[str]) -> Tuple[List[int], List[int]]:
    """(indices measured, indices gap-filled or unknown)."""
    meas = [i for i, q in enumerate(qq) if q in MEASURED]
    return meas, [i for i in range(len(qq)) if i not in set(meas)]
