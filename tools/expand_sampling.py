#!/usr/bin/env python3
"""
Tier-2 expander: planner sampling STRATEGY -> concrete column points.

Deterministic geospatial expansion (no LLM, no invented coordinates). Takes the
DEM grid reception already fetched and clipped to the basin, cuts it into
equal-interval elevation bands, gives every occupied band the planner's
per_band columns (or an even share of the budget), and picks spatially-spread
points inside each band. NOTHING is executed.

A FUNCTION OF TWO FILES (2026-08-18). reception.json carries the grid, the
basin boundary and every station; strategy.json carries the design and the
pinning rules it was made under. Nothing here fetches a DEM, a polygon or a
capability report — reception is the only component that reaches outside, and
what it fetched is what the sampler reads. The one call this module still
makes is a point-elevation query at each PINNED station, so the column is
banded by the ground it stands on rather than by a grid point kilometres away.

SELECTION IS ELEVATION-ONLY. No water table, no soil. Fan WTD used to be
attached here and never influenced a single placement; soil comes from the
warm-start donor gridcell, so a profile queried here would be a field the model
never sees. Both are the consumer's to fetch, where the decision that needs
them is made.

Operates on a pipeline run dir (reception.json + strategy.json), or standalone
via --bbox/--n/--bands, in which case it asks reception's own gather_grid for
the DEM — the same code, so there is one fetch-and-clip in the framework.

Run from the project root with the MCP runtime env:
    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/expand_sampling.py --run-dir workflow_outputs/pipeline_XXXX
    python3 tools/expand_sampling.py --bbox -121.52,46.46,-120.51,47.14 --n 12 --bands 4
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# NO MCP IMPORT AT MODULE SCOPE. The Experiment Manager imports this file for
# the sampler; the MCP client stack (mcp, anyio, ...) is needed only by the CLI
# in main(), and pulling it in here made the sampler unimportable wherever
# that stack was not installed. main() adds src/ to the path and imports it.


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC HELPERS (pure, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _make_bands(elevs, n_bands):
    """Equal-interval elevation bands as (lo, hi) pairs."""
    lo, hi = min(elevs), max(elevs)
    if hi <= lo or n_bands <= 1:
        return [(lo, hi)]
    step = (hi - lo) / n_bands
    return [(lo + i * step, hi if i == n_bands - 1 else lo + (i + 1) * step)
            for i in range(n_bands)]


def _assign_band(e, bands):
    """Index of the band containing elevation e (last band inclusive on hi).

    Out-of-range elevations clamp to the NEAREST band. The fall-through used to
    return the last band unconditionally, which is right for a value a hair
    above the maximum but files anything BELOW the minimum with the alpine
    columns. Harmless while every caller passed a grid point — the bands are
    built from those very points, so nothing could fall outside — and wrong the
    moment station pinning began passing elevations the grid never sampled: the
    2019 Gunnison well sits at 1792 m against a 2031 m sampled minimum, and
    landed in band 5 of 5.
    """
    if e <= bands[0][0]:
        return 0
    for i, (lo, hi) in enumerate(bands):
        last = (i == len(bands) - 1)
        if (lo <= e <= hi) if last else (lo <= e < hi):
            return i
    return len(bands) - 1


def _even_allocate(counts, n_total):
    """Equal columns per OCCUPIED band; any remainder to the bands holding the
    most DEM points.

    This replaced an area-proportional `_allocate` on 2026-08-07. That one gave
    each band a share of the columns proportional to its share of grid points
    and then applied a max(1, ...) floor, which is two biases pulling opposite
    ways: the floor guarantees a column to a sliver band holding three DEM
    points, while the proportional term hands most of the budget to whichever
    band happens to cover the most ground. Neither is what stratification asks
    for — the point of a stratum is equal effort inside it — and the
    proportional term in particular fought the area weighting that
    `_merge_column_metadata` applies downstream, weighting the same quantity
    twice.

    This runs only when the planner did not state `per_band`, or when a
    requested column count forces a budget `per_band` cannot fill exactly.
    """
    nonempty = [i for i, c in enumerate(counts) if c > 0]
    alloc = [0] * len(counts)
    if not nonempty or n_total <= 0:
        return alloc
    # Fewer columns than occupied bands: the most-populated bands win, so the
    # sample still spans as much of the gradient as the budget allows.
    by_size = sorted(nonempty, key=lambda i: (-counts[i], i))
    if n_total <= len(nonempty):
        for i in by_size[:n_total]:
            alloc[i] = 1
        return alloc
    base, rem = divmod(n_total, len(nonempty))
    for i in nonempty:
        alloc[i] = base
    for i in by_size[:rem]:
        alloc[i] += 1
    return alloc


def _farthest_point_select(pts, k, seeds=None):
    """Greedy farthest-point sampling for spatial spread within a band.

    Array-wise: one running array of "distance to the nearest chosen point",
    updated with a single minimum against the newest pick. The scalar form
    recomputed every candidate against every chosen point on each step
    (O(k^2 n) dict lookups); this is O(k n) in numpy and gives identical picks.

    `seeds` are points already placed in this band — the pinned station columns.
    They are not returned; they only seed the distance array, so a stratified
    column is never chosen on top of ground a station column already covers.
    With no seeds the first pick is the band centroid, exactly as before.
    """
    import numpy as np
    if k <= 0:
        return []
    if k >= len(pts):
        return list(pts)
    lat = np.fromiter((p["lat"] for p in pts), float, len(pts))
    lon = np.fromiter((p["lon"] for p in pts), float, len(pts))

    if seeds:
        d2 = np.full(len(pts), np.inf)
        for s in seeds:
            np.minimum(d2, (lat - s["lat"]) ** 2 + (lon - s["lon"]) ** 2, out=d2)
        picks = []
    else:
        first = int(np.argmin((lat - lat.mean()) ** 2 + (lon - lon.mean()) ** 2))
        picks = [first]
        d2 = (lat - lat[first]) ** 2 + (lon - lon[first]) ** 2
    while len(picks) < k:
        nxt = int(np.argmax(d2))
        picks.append(nxt)
        np.minimum(d2, (lat - lat[nxt]) ** 2 + (lon - lon[nxt]) ** 2, out=d2)
    return [pts[i] for i in picks]


# ─────────────────────────────────────────────────────────────────────────────
# STATION PINNING — the planner's validation design, obeyed
# ─────────────────────────────────────────────────────────────────────────────
#
# Until 2026-08-07 this file read two fields of the planner's sampling block,
# n_bands and n_columns, and nothing else. `per_band`, `n_validation` and the
# words "pinned to observation stations" in `approach` appeared nowhere in it,
# so a design of 15 stratified + 4 station-pinned columns became 19 stratified
# columns and nothing at any station.
#
# That dropped instruction has a scar downstream. step1_compare_swe abandoned
# station pairing because "five stations collapsed onto two columns with
# elevation offsets up to 846 m -- any agreement that produced was arithmetic,
# not skill". Pairing failed because no column was ever placed AT a station: the
# comparison rebuilt itself around elevation gradients to work around a gap
# nobody had noticed here.

STATION_SOURCES = (("streamflow", "stations"),
                   ("water_table", "wells"),
                   ("swe", "stations"),
                   ("et", "towers"))

# WHAT A COLUMN CAN BE PINNED TO IS THE MODEL SERVER'S ANSWER, NOT THIS FILE'S
# — AND IT IS READ FROM strategy.json, NOT ASKED OF THE SERVER (2026-08-18).
#
# A pinned column exists so a simulated value and an observed one describe the
# SAME place, and whether that is possible depends on what the model computes
# and where. This module used to hold the answer as a literal tuple while
# planner.txt held the same answer as prose — two statements of one rule, in two
# vocabularies, neither of which knew which model was about to run. "This model
# has no lateral transport" is true of a 1-D ELM column and false of a 3-D
# PFLOTRAN domain, so the frozen version was a bug waiting on a second backend.
#
# Then both read the server's capability report — the planner through its
# prompt, this file through a second MCP call at sampling time — which was one
# rule with two fetches, and two fetches can disagree: a server updated between
# plan and run, a resume a week later, a manager class naming a different
# server than the brief. So the block the planner was SHOWN now travels with
# the plan: workflow.py writes it into strategy.json beside the design, and this
# file reads it from there. One fetch, one record, two readers, and the sample
# is checked against exactly the rules the design was made under. This file no
# longer talks to any model server; it is a function of reception.json and
# strategy.json and nothing else.
#
# What the rules are FOR has not changed, and is worth keeping: on the 13-basin
# chain run of 2026-08-07, 14 of 40 pinned columns went to stream gauges. Eight
# were under the planner's own "basin-aggregate" label — it knew — and the rest
# were labelled "co-located", which for a gauge cannot be true. brandywine_2010
# put four gauge pins all in band 1, giving that band 7 of the basin's 13
# columns for a third of the elevation range: gauges sit on rivers, so gauge
# pins sit in valleys, so the ensemble tilts downhill.
#
# FILTERING ON THE VARIABLE, not on the planner's `comparison` string, is what
# makes that survivable — brandywine called all four "co-located", so a string
# filter would have caught none of them.
def pinning_rules(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """The `pinning` block the plan was designed against, in the filter's shape.

    Returns {model, pinnable: frozenset, why_not: {variable: reason}}.

    READ OFF strategy.json. workflow.py fetches the chosen model's block for the
    planner and writes it beside the plan (`strategy["pinning"]`), so what the
    sampler enforces is what the planner was shown — the same object, not a
    second fetch of the same question. Accepts a bare block too, for a caller
    that already holds one.

    RAISES when the block is absent. There is deliberately no built-in default:
    a fallback tuple here is exactly what this replaced, and one that engages
    silently would restate ELM's answer for whatever model actually ran. A
    strategy written before 2026-08-18 has no block; re-plan it, or add the
    server's `pinning` block by hand — do not default it.

    THE NAMES ARE THE COMPARISON'S NAMES (compare/*.py SPEC.name), which the
    server states in `vocabulary`. That agreement is not decorative: when the
    sampler wrote "water_table" and the comparison read "wtd", pair_stations
    compared the two spellings, found them unequal, and silently refused every
    well pin ever designed.
    """
    block = ((strategy or {}).get("pinning")
             if "pinning" in (strategy or {}) else strategy) or {}
    pinnable = [e.get("variable") for e in (block.get("pinnable") or [])
                if e.get("variable")]
    if not pinnable:
        raise RuntimeError(
            "strategy.json carries no `pinning.pinnable` block — the sampler "
            "cannot decide what a column may be pinned to, and guessing is what "
            "this replaced. workflow.py writes the chosen model's block beside "
            "the plan; a strategy from before 2026-08-18 does not have one.")
    return {
        "model": block.get("model"),
        "pinnable": frozenset(pinnable),
        "why_not": {e["variable"]: e.get("reason") or "no reason given"
                    for e in (block.get("not_pinnable") or [])
                    if e.get("variable")},
    }


# COLUMNS PLACED AT DOCUMENTED WATER TABLES. Not a pin and not a validation
# pairing: a Fan value is one long-term mean per site (1927-2009), so it cannot
# be paired day against day. It puts a column where the water table is known, so
# the column can be interpreted. Recorder wells are still fetched and still
# compared; they simply do not cost columns.
FAN_ANCHORS = 2                 # columns anchored at documented water tables
FAN_ANCHOR_MIN_SEPARATION_KM = 1.0      # the well-cluster radius reception uses


def _station_index(reception):
    """{station_id: record} across every observation reception fetched.

    Three fetchers, three shapes: streamflow and water_table key their records
    'id', SNOTEL keys its 'triplet', and only SNOTEL reports an elevation. The
    planner cites whichever string the fetcher used, so both are accepted.
    """
    idx = {}
    obs = (reception or {}).get("observations") or {}
    for var, key in STATION_SOURCES:
        for rec in ((obs.get(var) or {}).get(key) or []):
            sid = rec.get("id") or rec.get("triplet")
            if not sid or rec.get("lat") is None or rec.get("lon") is None:
                continue
            idx[str(sid)] = {"station_id": str(sid),
                             "station_variable": var,
                             "station_name": rec.get("name"),
                             "lat": float(rec["lat"]),
                             "lon": float(rec["lon"]),
                             "in_basin": rec.get("in_basin"),
                             "station_elevation_m": rec.get("elevation_m")}
    return idx


def _fan_anchors(reception, n=FAN_ANCHORS):
    """One or two columns placed where the long-term water table is documented.

    NOT A VALIDATION PIN, and the record says so. Fan et al. 2013 is one mean
    per site over 1927-2009 — reception labels it "long-term mean, one value per
    site" — so no simulated year can be paired against it day by day. What it
    gives is a column whose water table has a DOCUMENTED value to be read
    against: "this column sits where the long-term water table is 3.4 m" is a
    statement about representativeness, which is what the sampling is for.

    WHY FAN AND NOT THE RECORDER WELLS: coverage. Naches has 299 Fan sites and
    zero recorder wells; Brandywine has 421 against 21. A design rule that only
    works in the rare basin is not a design rule.

    WHICH SITES. Two, chosen to be interpretable rather than convenient:

      the MEDIAN in-basin depth — the typical water table for this watershed;
      the SHALLOW end (10th percentile) — because below ELM's 3.8 m active soil
          the model DIAGNOSES a water table instead of simulating one, and at
          Naches that was 16 of 16 columns. If any column is to be placed where
          the model can actually simulate the thing, this is the one.

    They are dropped to one when they land within FAN_ANCHOR_MIN_SEPARATION_KM
    of each other — the same radius reception clusters wells at, and for the
    same reason: two columns in one gridcell are one column and two ELM runs.
    """
    if n <= 0:
        return []
    obs = (reception or {}).get("observations") or {}
    sites = [w for w in ((obs.get("water_table_static") or {}).get("wells") or [])
             if w.get("in_basin") is not False
             and w.get("wtd_m") is not None
             and w.get("lat") is not None and w.get("lon") is not None]
    if not sites:
        print("   ⚠️  no in-basin Fan site to anchor a column at — the design "
              "gets no water-table anchor. This is a coverage fact, not a "
              "failure; the comparison's Fan distribution does not need one.")
        return []

    ranked = sorted(sites, key=lambda w: float(w["wtd_m"]))
    picks, seen = [], []
    for label, idx in (("median in-basin Fan depth", len(ranked) // 2),
                       ("shallow end, 10th percentile", len(ranked) // 10)):
        if len(picks) >= n:
            break
        w = ranked[idx]
        if any(_haversine_km(w["lat"], w["lon"], s["lat"], s["lon"])
               <= FAN_ANCHOR_MIN_SEPARATION_KM for s in seen):
            continue
        seen.append(w)
        # THE DEPTH ITSELF DOES NOT TRAVEL, and that is deliberate (user's
        # instruction, 2026-08-12). _place_pinned copies every field of this
        # record onto the column, columns.json carries it into the analyzer, and
        # a number labelled "the water table here is 11.66 m" sitting beside a
        # modelled ZWT is a prior handed to whatever reads it next — including
        # an LLM asked to interpret the run. Fan chose WHERE this column goes;
        # it must not also suggest what the answer should be.
        #
        # So the record is exactly the shape every other pin has: an id, the
        # variable, coordinates. The site id keeps the choice traceable — the
        # value is one lookup away for a person, and absent for a process.
        picks.append({
            "station_id": f"FAN-{w.get('id')}",
            # DELIBERATELY NOT "water_table". The comparison honours a pin only
            # for the observable named here, and a Fan anchor must never be
            # force-paired to a recorder well: they are different quantities
            # measured decades apart.
            "station_variable": "water_table_static",
            "station_name": w.get("name"),
            "lat": float(w["lat"]), "lon": float(w["lon"]),
            "in_basin": w.get("in_basin"),
            "station_elevation_m": None})
        print(f"   ⚓ Fan anchor FAN-{w.get('id')} at {w['wtd_m']} m ({label}) — "
              f"placement only; the depth is not written to the column")
    return picks


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _pinned_from_plan(plan, reception):
    """Stations the planner asked for a column at, resolved to coordinates.

    A FUNCTION OF TWO FILES. `plan` is strategy.json — the design, and the
    pinning block it was designed against; `reception` is reception.json —
    every station that was fetched, with its coordinates and its in-basin tag.
    Nothing is asked of a server here: everything this decides on was on disk
    before it ran, and the planner was shown all of it.

    Reads `plan["validation"]`, whose entries look like

        {"variable": "swe", "stations": ["538:CO:SNTL"], "comparison": "co-located — ..."}

    ONLY A CO-LOCATED VERDICT PINS. The planner speaks four verdicts —
    co-located, basin-aggregate, distance-matched, unavailable — and only the
    first means "put a column here". Across every archived strategy, stations
    appear under co-located and under nothing else, so this is the rule the
    planner already follows rather than a new one; an entry that breaks it is
    reported and its stations are not spent. That replaces a special case for
    the word "unavailable" that caught the naches_1988 slip (water_table ruled
    out AND two wells named, costing two columns and leaving two bands short)
    and would have missed the same slip under any other verdict.

    Then, per station, three checks — each one line, each with a run behind it:

      not fetched     RAISED. Every id the planner names is one it was shown,
                      so a miss means plan and observations disagree, and
                      dropping it silently is how this class of bug happened.
      not pinnable    dropped, with the server's own reason. Judged on WHICH
                      LIST RECEPTION FOUND THE STATION IN, not on the plan's
                      label for it, so a mislabelled entry is filtered on what
                      the station actually is. brandywine_2010 called four
                      gauge pins "co-located"; a string filter caught none.
      outside basin   dropped. Observations are fetched for the bbox, which
                      is larger than the divide; `in_basin` is reception's tag,
                      absent only when there was no polygon — and absent is
                      not False.

    THE FAN ANCHORS ARE APPENDED WHETHER OR NOT ANYTHING WAS PINNED. They are
    the sampler's own — the planner is never shown Fan and budgets "+ 2" for
    them blind — so a plan that pins no station still gets its two columns at
    documented water tables. An earlier version returned before reaching them
    when the pin list was empty: the count still came out right, because the
    two orphaned columns were handed to the bands, so nothing said that a
    17-column PFLOTRAN study meant to sit two columns on the water table sat
    none. They come last so a budget squeeze in expand() drops an anchor
    before a station the planner asked for.
    """
    rules = pinning_rules(plan)
    pinnable, why = rules["pinnable"], rules["why_not"]
    idx = _station_index(reception)

    keep, seen = [], set()
    for entry in (plan or {}).get("validation") or []:
        stations = [str(s) for s in entry.get("stations") or []]
        verdict = str(entry.get("comparison") or "").strip().lower()
        if not verdict.startswith("co-located"):
            if stations:
                print(f"   ⚠️  '{entry.get('variable')}' is "
                      f"'{verdict.split(' ')[0] or '(no verdict)'}' but names "
                      f"{len(stations)} station(s) {stations} — NOT pinning "
                      f"them. Only a co-located verdict places a column.")
            continue
        for sid in stations:
            if sid in seen:
                continue
            seen.add(sid)
            st = idx.get(sid)
            if st is None:
                raise ValueError(
                    f"Cannot pin columns: strategy.validation names station "
                    f"{sid!r} that is not in reception's fetched set "
                    f"({len(idx)} available: {sorted(idx)[:8]}"
                    f"{' ...' if len(idx) > 8 else ''}). A column at the "
                    f"station is the only thing that lets the analyzer pair a "
                    f"simulation with an observation, so this is not degraded "
                    f"silently — fix reception's fetch, or drop the station "
                    f"from strategy.validation.")
            var = st["station_variable"]
            if var not in pinnable:
                print(f"   ⚠️  not pinning {sid} ({var}) — this model cannot be "
                      f"co-located with it: "
                      f"{why.get(var, 'not a quantity this model produces at a point.')}")
                continue
            if st.get("in_basin") is False:
                print(f"   ⚠️  not pinning {sid} ({var}) — OUTSIDE the "
                      f"watershed. Observations are fetched for the bounding "
                      f"box, which is larger than the basin; a column out "
                      f"there is forced and soiled from ground the study "
                      f"does not model.")
                continue
            keep.append(st)

    return keep + _fan_anchors(reception)


def _place_pinned(terrain, pinned, bands, pts):
    """One column at each station's own coordinates.

    Elevation comes from a point 3DEP query AT the station, not from the nearest
    DEM grid point: at grid_n=120 over a basin this size the nearest grid point
    sits up to ~1 km away, which in this relief is worth hundreds of metres and
    would file the column under the wrong band. `station_elevation_m` is kept
    beside it wherever the station reports one, so a comparison can see how far
    the model surface sits from the instrument before trusting the pairing — the
    quantity step1_compare_swe had to give up on.
    """
    terr = terrain
    out = []
    for st in pinned:
        elev, src = None, None
        if terr is not None:
            try:
                r = terr.call_tool_json("get_elevation",
                                        {"lat": st["lat"], "lon": st["lon"]}) or {}
                elev = r.get("elevation_m")
                src = "3dep_point" if elev is not None else None
            except Exception as e:
                print(f"   ⚠️  elevation lookup failed for {st['station_id']}: {e}")
        if elev is None and st.get("station_elevation_m") is not None:
            elev, src = float(st["station_elevation_m"]), "station_reported"
        if elev is None and pts:
            near = min(pts, key=lambda p: (p["lat"] - st["lat"]) ** 2
                                          + (p["lon"] - st["lon"]) ** 2)
            elev, src = near["elevation_m"], "nearest_dem_grid_point"

        col = dict(st)
        col.update({"lat": round(st["lat"], 5), "lon": round(st["lon"], 5),
                    "elevation_m": round(float(elev), 2) if elev is not None else None,
                    "pinned": True,
                    "elevation_source": src})
        # No stored elevation-minus-station field. Both operands are on the
        # record — elevation_m and station_elevation_m — so the difference is one
        # subtraction away, and a stored copy would only drift once the warm
        # start replaces elevation_m with the donor gridcell's TOPO.
        col["_band_idx"] = (_assign_band(elev, bands) if elev is not None else 0)

        # A station outside the DEM sample's elevation range is clamped to the
        # nearest band, but say so: it means either the station sits outside the
        # basin the bands were built from, or the DEM sample is too sparse to
        # have reached that elevation. Both change what the pinned column means.
        if elev is not None and not (bands[0][0] <= elev <= bands[-1][1]):
            print(f"   ⚠️  {st['station_id']} sits at {elev:.0f} m, outside the "
                  f"sampled range {bands[0][0]:.0f}-{bands[-1][1]:.0f} m — "
                  f"clamped into band {col['_band_idx'] + 1}.")
        out.append(col)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EXPANSION — reception's grid in, columns out
# ─────────────────────────────────────────────────────────────────────────────
#
# _clip_to_polygon WAS HERE and is deleted (2026-08-18). It clipped the grid to
# the largest WBD ring with shapely — a second clip, after data_gather.gather_grid
# had already dropped every point outside the divide with its own ray-casting
# test and written the survivors to reception.json. Fifty-eight points in,
# fifty-eight out, on every run. The two predicates were not even the same
# (largest ring against all rings), so for a basin in parts they could disagree
# about a point reception had kept. One clip, reception's; this file reads it.

def expand(rec_grid, n_total, n_bands, per_band=None, pinned=None,
           terrain=None):
    """Strategy -> concrete columns.

    rec_grid  reception's `grid` block, as written: `points` already clipped
              to the basin, `boundary` (the WBD rings, kept for the record and
              the design figure), `clipped_to_watershed`. Used as given —
              nothing is fetched or clipped here.
    n_total   the column budget, after check() has reconciled the planner's
              n_columns against anything the request asked for. HARD: it is the
              number of cases that will be built, and the enforcement path
              fixed in 22cecae depends on nothing here exceeding it.
    per_band  the planner's stratified count per band, obeyed as stated.
    pinned    station records from _pinned_from_plan; one column each, at the
              station's own coordinates.
    terrain   the terrain client, for the point elevation AT each pinned
              station — the one lookup this module still makes. Optional:
              without it a pinned column takes the station's reported
              elevation, then the nearest grid point, and says which.

    WHY THE GRID IS TAKEN AND NOT FETCHED. Reception fetches this exact grid
    at the sampler's own resolution and RETRIES at higher density when too few
    points land inside the basin. This function used to fetch its own when not
    handed one, without the retry, and could design on a fraction of the
    evidence reception had already paid for: on naches_2023 (2026-08-07)
    reception escalated to n=216 and kept 101 points; a flat n=120 here got
    29, two bands held two points each against per_band=3, and the BAND EDGES
    were cut from that sparse sample. Then it re-clipped what reception had
    already clipped. Now there is one grid, and reception owns it.
    """
    rec_grid = rec_grid or {}
    pts = [p for p in (rec_grid.get("points") or [])
           if p.get("elevation_m") is not None]
    boundary = rec_grid.get("boundary") or None
    print(f"  DEM grid: {len(pts)} points from reception"
          + (", clipped to the watershed" if boundary else ", NOT clipped (no boundary)"))
    if not pts:
        return {"error": "reception's grid carries no elevation points — "
                         "gather_grid did not run, or returned nothing"}

    bands = _make_bands([p["elevation_m"] for p in pts], n_bands)
    by_band = {i: [] for i in range(len(bands))}
    for p in pts:
        by_band[_assign_band(p["elevation_m"], bands)].append(p)
    counts = [len(by_band[i]) for i in range(len(bands))]

    # ── PINNED FIRST: they are the validation design, and they consume budget ──
    pin_cols = _place_pinned(terrain, pinned or [], bands, pts)
    if len(pin_cols) > n_total:
        dropped = pin_cols[n_total:]
        pin_cols = pin_cols[:n_total]
        print(f"   ⚠️  budget {n_total} is smaller than the {len(dropped) + n_total} "
              f"stations the strategy pins. DROPPED: "
              f"{[d['station_id'] for d in dropped]}")
        print(f"   ⚠️  Those variables now have NO co-located column and cannot "
              f"be validated at a station.")

    banded_budget = max(0, n_total - len(pin_cols))

    # ── BANDED: the planner's per_band, as stated ────────────────────────────
    if per_band and per_band > 0:
        alloc = [per_band if counts[i] > 0 else 0 for i in range(len(bands))]
        if sum(alloc) == banded_budget:
            alloc_rule = f"planner per_band={per_band}"
        else:
            # The budget was rewritten (check() reconciling n_columns against a
            # request) or a band came back empty, so per_band cannot be met
            # exactly. n_total wins — it is what the run will actually build —
            # and the departure is stated rather than absorbed.
            print(f"   ⚠️  per_band={per_band} over {sum(1 for c in counts if c > 0)} "
                  f"occupied bands wants {sum(alloc)} stratified columns, but the "
                  f"budget leaves {banded_budget} after {len(pin_cols)} pinned.")
            print(f"   ⚠️  Using {banded_budget}, spread evenly across bands.")
            alloc = _even_allocate(counts, banded_budget)
            alloc_rule = f"reconciled to budget (planner asked per_band={per_band})"
    else:
        alloc = _even_allocate(counts, banded_budget)
        alloc_rule = "even per band (plan stated no per_band)"

    # ── ASSEMBLE: bands ascending, pinned before stratified within each ──────
    columns, cid = [], 1
    short = []
    pin_by_band = {i: [] for i in range(len(bands))}
    for c in pin_cols:
        pin_by_band[c.pop("_band_idx")].append(c)
    for i in range(len(bands)):
        here = pin_by_band[i]
        picks = _farthest_point_select(by_band[i], alloc[i], seeds=here)
        if len(picks) < alloc[i]:
            short.append((i + 1, alloc[i], len(picks)))
        for c in here:
            c.update({"id": f"col_{cid:02d}", "band": i + 1,
                      "band_range_m": [round(bands[i][0]), round(bands[i][1])]})
            columns.append(c)
            cid += 1
        for p in picks:
            columns.append({"id": f"col_{cid:02d}",
                            "lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                            "elevation_m": p["elevation_m"], "band": i + 1,
                            "band_range_m": [round(bands[i][0]), round(bands[i][1])],
                            "pinned": False})
            cid += 1
    for band, want, got in short:
        print(f"   ⚠️  band {band} holds only {got} DEM points, {want} asked for "
              f"— the ensemble is {want - got} column(s) short there.")

    n_pin = sum(1 for c in columns if c.get("pinned"))
    print(f"   sampled {len(columns)} columns: {n_pin} pinned at stations, "
          f"{len(columns) - n_pin} stratified ({alloc_rule})")
    for c in columns:
        if c.get("pinned"):
            se = c.get("station_elevation_m")
            d = (c["elevation_m"] - se) if (se and c.get("elevation_m")) else None
            print(f"     {c['id']}  {c['station_id']:<22} {c['station_variable']:<11} "
                  f"band {c['band']}  DEM {c['elevation_m']} m"
                  + (f"  ({d:+.1f} m vs station)" if d is not None else ""))

    # NO WATER TABLE IS GATHERED HERE ANY MORE (2026-08-07). Sampling selects on
    # ELEVATION: a DEM grid, clipped to the basin, split into bands, then
    # farthest-point selection within each. Fan's water table never influenced
    # any of that — it was queried afterwards, at coordinates already chosen,
    # which made this stage look like it depended on a dataset it did not use.
    #
    # WHO NEEDS IT AND WHERE IT GOES. `fan_wtd_m` is PFLOTRAN's initial
    # condition and sets wt_in_domain — a column whose water table is below the
    # modelled domain runs fully unsaturated, which was 10 of 19 columns on the
    # 2019 Gunnison sample. ELM only displays it. So it belongs to whoever needs
    # it, fetched where that decision is made, not pre-emptively for everyone.
    # Until PFLOTRAN's inputs are restructured, a consumer must call
    # get_fan_wtd_points itself; nothing here does it for them.
    #
    # The batching lesson still holds wherever it lands: every MCP call is a
    # fresh session, so asking per column pays a process spawn plus a dataset
    # open each time — and for Fan the open IS the cost. Ask once for all
    # coordinates.
    #
    # Also worth carrying: a value fetched HERE describes the pre-snap
    # coordinate. ELM's warm start moves each column up to ~0.4 km against a
    # ~0.93 km Fan grid, which measurably changed nothing across 19 columns, but
    # a consumer that fetches after the snap is simply correct rather than
    # correct-by-margin.

    # NO SOIL IS GATHERED HERE, deliberately. The run is warm-started from the
    # CONUS 1 km restarts, which carry the donor gridcell's own surfdata — so
    # the soil ELM runs on is decided by the donor, not by anything queried at
    # sampling time. Fetching a second profile here produced a field in
    # columns.json that the model never saw, and the only defence against
    # analysing it was that nobody happened to. `_attach_donor_soil` fills
    # soil_profile from the donor after the warm start, and that is the only
    # soil the run has.

    return {"n_requested": n_total, "n_columns": len(columns),
            "bands": [{"band": i + 1, "elev_lo_m": round(bands[i][0]),
                       "elev_hi_m": round(bands[i][1]), "grid_points": counts[i],
                       "allocated": alloc[i],
                       "pinned": sum(1 for c in columns
                                     if c.get("pinned") and c["band"] == i + 1)}
                      for i in range(len(bands))],
            # What the planner asked for, next to what was built. Whoever reads
            # columns.json can now check the design was obeyed without holding
            # plan.json open beside it — which is what nobody did for months.
            "sampling_design": {
                "n_columns_requested": n_total,
                "per_band_requested": per_band,
                "allocation_rule": alloc_rule,
                "n_pinned": n_pin,
                "n_stratified": len(columns) - n_pin,
                "pinned_stations": [{"id": c["station_id"],
                                     "variable": c["station_variable"],
                                     "column": c["id"], "band": c["band"],
                                     "station_elevation_m":
                                         c.get("station_elevation_m")}
                                    for c in columns if c.get("pinned")],
            },
            "columns": columns,
            # THE FULL DEM SAMPLE, and it IS saved: both callers write this
            # dict whole as columns.json, and the design figure reads `grid`
            # from it for the hypsometry and the map. A comment here used to
            # say the opposite.
            "grid": [{"lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                      "elevation_m": p["elevation_m"]} for p in pts],
            # THE BOUNDARY, AS RECEPTION FETCHED IT. Carried so columns.json is
            # self-contained: the design figure draws the basin outline from it
            # and a re-plot needs no MCP call. None when reception had no
            # polygon, and the caller's sampling_domain says what that means.
            # Last, where the manager used to append it, so an archived
            # columns.json keeps its key order and its hash.
            "boundary": boundary}


# ─────────────────────────────────────────────────────────────────────────────
# NOTHING MODEL-SPECIFIC BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────
#
# nldas_annual_precip AND ITS HELPERS MOVED to mcp/elm-mcp/src/forcing.py
# (2026-08-18). ~130 lines that opened NLDAS-2 monthly files at two hardcoded
# Compy paths, called by exactly one thing — the ELM server's design figure,
# which reached back into tools/ for it — under a comment claiming the forcing
# was "shared with PFLOTRAN", which stopped being true when PFLOTRAN moved to
# Daymet. forcing.py already owned the NLDAS directory and the file pattern.
# Everything ELM uses the ELM server; the sampler is not an exception.
#
# _soil_cov DELETED the same day. It summarised the donor soil for a panel of
# plot_columns, deleted 2026-08-13; the server's own figure has _soil_layers.
# Its only callers were two tests, which went with it.
#
# plot_columns() WAS HERE and is deleted (2026-08-13).
#
# It drew sampling_design.png as a 2x3 with its own rcParams — 15 pt titles on
# a 17.5-inch canvas, about 6 pt once printed — and one of its panels read
# `fan_wtd_m`, a field whose producer this module lost on 2026-08-07 when Fan
# left the sampler. The figure now belongs to the model server that knows what
# the columns became: mcp/elm-mcp/src/sampling_design.py, drawn from
# POST-warm-start values, which is the ensemble ELM integrates rather than the
# one that was sampled. See ELMExpManager._draw_design.




# ─────────────────────────────────────────────────────────────────────────────
# RUN-DIR INPUTS + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_from_brief(brief):
    b = (brief.get("domain") or {}).get("bbox") or {}
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    return {k: b[k] for k in keys} if all(k in b for k in keys) else None


def _n_from_plan(plan):
    """The planner's column count.

    THE FALLBACK KEYS ARE NOT DRIFT. `sampling.n_columns` is what planner.txt
    emits; `sampling_strategy.n_exploratory` and `experiment_summary.exploratory`
    are what planner_capability_probe.txt emitted, and that prompt is FROZEN
    for the eval (see ARCHITECTURE.md) — the plans it produced are still read.
    Same for the two readers below.
    """
    return ((plan.get("sampling") or {}).get("n_columns")
            or (plan.get("sampling_strategy") or {}).get("n_exploratory")
            or (plan.get("experiment_summary") or {}).get("exploratory"))


def _n_bands_from_plan(plan):
    """The planner's band count, read the same way n_columns is.

    It used to be ignored entirely: the manager fell back to
    len(brief.heterogeneity.elevation_bands), so a strategy asking for 5
    bands got however many elevations reception happened to list — 3, for
    the 2019 Upper Gunnison run. The planner's column count WAS honoured, so
    the ensemble ended up with the planner's N spread over reception's band
    count, a design neither box specified.

    len(elevation_bands) was the wrong quantity in any case. Those are band
    EDGES or representative elevations, not a count: three edges imply two
    bands or four, never three.
    """
    return ((plan.get("sampling") or {}).get("n_bands")
            or (plan.get("sampling_strategy") or {}).get("n_bands"))


def _per_band_from_plan(plan):
    """The planner's stratified count per band. Read the same way n_bands is.

    Ignored entirely until 2026-08-07, when the allocator computed its own
    area-proportional counts instead. See _even_allocate.
    """
    return ((plan.get("sampling") or {}).get("per_band")
            or (plan.get("sampling_strategy") or {}).get("per_band"))


def _print_table(res):
    print("\nBANDS:")
    for b in res["bands"]:
        pin = f" (+{b['pinned']} pinned)" if b.get("pinned") else ""
        print(f"  band {b['band']}: {b['elev_lo_m']:>5}-{b['elev_hi_m']:>5} m  "
              f"| {b['grid_points']:>3} grid pts -> {b['allocated']} columns{pin}")
    d = res.get("sampling_design") or {}
    if d:
        print(f"\nDESIGN: {d.get('n_pinned', 0)} pinned + "
              f"{d.get('n_stratified', 0)} stratified  [{d.get('allocation_rule')}]")
    print(f"\n{res['n_columns']} CONCRETE COLUMNS:")
    print(f"  {'id':<8}{'lat':>9}{'lon':>11}{'elev_m':>8}{'band':>5}  station")
    print("  " + "-" * 66)
    for c in res["columns"]:
        st = c.get("station_id") or "-"
        se, el = c.get("station_elevation_m"), c.get("elevation_m")
        if se and el:
            st += f"  ({el - se:+.1f} m vs station)"
        print(f"  {c['id']:<8}{c['lat']:>9}{c['lon']:>11}{c['elevation_m']:>8}"
              f"{c['band']:>5}  {st}")


def main():
    ap = argparse.ArgumentParser(description="Tier-2 expander: strategy -> concrete columns")
    ap.add_argument("--run-dir", help="pipeline output dir (reads reception.json + strategy.json)")
    ap.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat (standalone: fetches the grid via gather_grid)")
    ap.add_argument("--huc", default="", help="standalone: HUC to clip the grid to")
    ap.add_argument("--n", type=int, help="number of columns (overrides plan)")
    ap.add_argument("--bands", type=int, default=0, help="number of elevation bands")
    ap.add_argument("--per-band", type=int, default=0,
                    help="stratified columns per band (overrides plan.sampling.per_band)")
    ap.add_argument("--no-pin", action="store_true",
                    help="skip station pinning (stratified columns only)")
    args = ap.parse_args()

    # The MCP stack, only here — see the note at the top of the file.
    _root = Path(__file__).resolve().parents[1]
    for _d in (_root / "src",):
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
    from core.mcp_manager import MCPManager

    out_dir = Path(args.run_dir) if args.run_dir else Path(".")
    out = out_dir / "columns.json"

    print("=" * 72)
    print("TIER-2 EXPANDER  —  strategy -> concrete columns (deterministic; no execution)")
    print("=" * 72)

    if args.run_dir and out.exists():
        # Pure read: re-inspect an already-materialized run-dir — no fetch.
        res = json.loads(out.read_text())
        print(f"reading existing {out} ({res.get('n_columns')} columns — no MCP fetch)")
    else:
        n_total, plan, rec, rec_grid = None, {}, {}, None
        n_bands = args.bands or 4
        per_band, pinned = args.per_band, []
        clients = MCPManager(str(_root / "mcp_config.json")).get_all_clients()
        if args.run_dir:
            rd = Path(args.run_dir)
            # strategy.json is the plan AND the pinning block it was designed
            # against; plan.json is the manager's copy of the same dict.
            plan = next((json.loads((rd / f).read_text())
                         for f in ("strategy.json", "plan.json")
                         if (rd / f).exists()), {})
            if (plan.get("archetype")
            or (plan.get("model_choice") or {}).get("design_archetype")) == "conceptual":
                sys.exit("Plan is 'conceptual' archetype — no spatial expansion needed.")
            rec_f = rd / "reception.json"
            if not rec_f.exists():
                sys.exit(f"{rec_f} not found — the sampler reads reception's grid "
                         f"and stations from it, and fetches nothing itself.")
            rec = json.loads(rec_f.read_text())
            rec_grid = rec.get("grid") or {}
            n_total = _n_from_plan(plan)
            if not args.bands:
                # The planner's count, not len(brief.heterogeneity.elevation_bands)
                # — those are band edges, never a band count. See _n_bands_from_plan.
                n_bands = _n_bands_from_plan(plan) or 4
            per_band = per_band or _per_band_from_plan(plan)
            if not args.no_pin:
                pinned = _pinned_from_plan(plan, rec)
        if args.bbox:
            # STANDALONE. The grid comes from reception's OWN gather_grid, so
            # there is exactly one fetch-and-clip in the framework and this
            # path cannot drift from what a real run samples.
            from core import data_gather
            v = [float(x) for x in args.bbox.split(",")]
            bbox = {"min_lon": v[0], "min_lat": v[1], "max_lon": v[2], "max_lat": v[3]}
            rec_grid = data_gather.gather_grid(clients, bbox, huc=args.huc)
        if args.n:
            n_total = args.n
        if not rec_grid:
            sys.exit("No grid (give --run-dir with a reception.json, or --bbox).")
        if not n_total:
            sys.exit("No column count (give --run-dir with a plan, or --n).")

        print(f"grid: {len(rec_grid.get('points') or [])} pts | N={n_total} | "
              f"bands={n_bands} | per_band={per_band or '-'} | pinned={len(pinned)}")
        res = expand(rec_grid, n_total, n_bands, per_band=per_band,
                     pinned=pinned, terrain=clients.get("terrain"))
        if "error" in res:
            sys.exit(res["error"])
        out.write_text(json.dumps(res, indent=2))   # self-contained (grid + boundary)
        print(f"\nSaved {res['n_columns']} columns -> {out}")

    _print_table(res)
    # --plot IS GONE (2026-08-13). The design figure is drawn by the model
    # server, from post-warm-start values, and cannot be drawn at this point in
    # the sequence: nothing has been snapped yet and there is no finidat to
    # read. It appears after the inputs are built — see
    # ELMExpManager._draw_design.
    print()


if __name__ == "__main__":
    main()
