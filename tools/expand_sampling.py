#!/usr/bin/env python3
"""
Tier-2 expander: planner sampling STRATEGY -> concrete column points.

Deterministic geospatial expansion (no LLM, no invented coordinates). Samples
the real DEM via the terrain MCP across the domain bbox, stratifies by elevation
band, allocates the planner's N columns proportionally to occupied area, and
picks spatially-spread points per band. NOTHING is executed.

SELECTION IS ELEVATION-ONLY. No water table, no soil. Fan WTD used to be
attached here and never influenced a single placement; soil comes from the
warm-start donor gridcell, so a profile queried here would be a field the model
never sees. Both are the consumer's to fetch, where the decision that needs
them is made.

Operates on a pipeline run dir (reads reception_brief.json for the bbox and
plan.json for N / band count), or standalone via --bbox/--n/--bands.

Run from the project root with the MCP runtime env:
    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/expand_sampling.py --run-dir workflow_outputs/pipeline_XXXX
    python3 tools/expand_sampling.py --bbox -121.52,46.46,-120.51,47.14 --n 12 --bands 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
from core.mcp_manager import MCPManager


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

# WHAT A 1-D COLUMN CAN BE PINNED TO.
#
# A pinned column exists so a simulated value and an observed one describe the
# SAME place. That only works for a quantity the column actually produces at a
# point. SWE, water table and ET are vertical and local — the column computes
# them where it stands, and an instrument measures them where it stands.
#
# Streamflow is not. A gauge measures discharge integrated and ROUTED over its
# upstream area, and this framework runs 1-D columns with no lateral transport,
# so a column at the gauge's coordinates produces a point runoff flux, never the
# thing the gauge recorded. Putting a column there buys nothing the ensemble
# mean does not already give.
#
# Measured on the 13-basin chain run, 2026-08-07: 14 of 40 pinned columns went
# to gauges. Eight were under the planner's own "basin-aggregate" label — it
# knew — and the rest were labelled "co-located", which for a gauge cannot be
# true. brandywine_2010 is the clearest: four gauge pins, all in band 1, giving
# that band 7 of the basin's 13 columns for a third of the elevation range.
# Gauges sit on rivers, so gauge pins sit in valleys, so the ensemble tilts
# downhill.
#
# Filtering on the variable rather than on the planner's `comparison` string is
# deliberate: brandywine called all four "co-located", so a string filter would
# have caught none of them. The reason is structural — true of every gauge in
# every basin — which makes it the sampler's to enforce, not the planner's to
# remember.
#
# Streamflow validation is NOT dropped. It stays a basin-aggregate comparison
# against the ensemble; it simply stops costing a column.
#
# THESE ARE OBSERVABLE NAMES, and they are the SAME NAMES the comparison uses
# (compare/*.py SPEC.name): swe, water_table, streamflow, et. That was not true
# until 2026-08-13 — the comparison called the water table "wtd" while the
# sampler wrote "water_table" into every pinned column's `station_variable`, so
# pair_stations compared the two spellings, found them unequal, and silently
# refused every well pin ever designed. One vocabulary; a pin that is refused
# is refused for a reason someone can read.
PINNABLE_VARIABLES = ("swe", "et")

# WATER TABLE LEFT THIS LIST 2026-08-12, by the user's decision, and was
# replaced by the Fan anchors below. What pinning to a recorder well bought,
# measured on brandywine_2010: four wells pinned, three of them snapped onto ONE
# gridcell by the warm start, so three IDENTICAL ELM cases were built and run
# and the comparison reported three NSEs — -1.9, -88.6, -757.4 — against one
# model series. The wells were a piezometer nest: twelve of them in that cell,
# 0.42 m to 36.32 m depth to water, because nine are screened in the confined
# Potomac Formation 135-701 m down and one in the surficial Columbia aquifer.
#
# It also almost never applies. Naches has no recorder well in any year, and of
# the thirteen chain-eval basins most have none, so the scheme spent columns in
# the rare basin and did nothing in the common one.
#
# The trade is stated plainly: a Fan anchor CANNOT be paired day against day,
# because Fan is one long-term mean per site (1927-2009). It is a DESIGN
# anchor — it puts a column where the water table is documented, so the column
# can be interpreted — not a validation pairing. Recorder wells are still
# fetched and still compared; they simply stop costing columns.
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

    Reads `plan["validation"]`, whose entries look like

        {"variable": "swe", "stations": ["538:CO:SNTL", "762:CO:SNTL"], ...}

    and resolves each id against what reception actually fetched. Every station
    the planner names is one it was shown, so each should resolve; a miss means
    the plan and the observations disagree, and that is raised rather than
    dropped. Silently skipping is how this whole class of bug happened.

    A variable the planner itself ruled out gets no columns. planner.txt asks it
    to pair `comparison: "unavailable"` with `stations: []`, and the 2026-08-07
    chain run caught it emitting the verdict while keeping the list: naches_1988
    declared water_table unavailable — "observed WTDs of tens of metres lie below
    the ELM soil column" — and still named two wells. Pinning them spent two
    columns on a comparison that cannot be made AND took both out of the
    stratified budget, leaving 13 stratified columns where the plan said 15 and
    two bands one short. The verdict is the planner's real intent; the list is
    the slip, and its own n_validation=4 against 6 cited stations says so.
    """
    wanted, seen, ruled_out = [], set(), []
    for entry in (plan or {}).get("validation") or []:
        stations = [str(s) for s in entry.get("stations") or []]
        if str(entry.get("comparison") or "").strip().lower().startswith(
                "unavailable"):
            if stations:
                ruled_out.append((entry.get("variable"), stations))
            continue
        for sid in stations:
            if sid not in seen:
                seen.add(sid)
                wanted.append(sid)

    for var, sts in ruled_out:
        print(f"   ⚠️  '{var}' is marked comparison='unavailable' but names "
              f"{len(sts)} station(s) {sts} — NOT pinning them. The planner "
              f"ruled the comparison out; a column there cannot be validated.")

    # n_validation is DEFINED as the number of pinned columns, so a disagreement
    # means the plan's own arithmetic (n_bands*per_band + n_validation) does not
    # describe what it asked for.
    n_val = ((plan or {}).get("sampling") or {}).get("n_validation")
    if n_val is not None and len(wanted) != n_val:
        print(f"   ⚠️  plan says n_validation={n_val} but {len(wanted)} station(s) "
              f"remain after the 'unavailable' entries — the column budget was "
              f"computed from {n_val}.")

    if not wanted:
        return []

    idx = _station_index(reception)

    missing = [s for s in wanted if s not in idx]
    if missing:
        raise ValueError(
            f"Cannot pin columns: strategy.validation names station(s) "
            f"{missing} that are not in reception's fetched set "
            f"({len(idx)} available: {sorted(idx)[:8]}{' ...' if len(idx) > 8 else ''}). "
            f"A column at the station is the only thing that lets the analyzer "
            f"pair a simulation with an observation, so this is not degraded "
            f"silently — fix reception's fetch, or drop the station from "
            f"strategy.validation.")

    # Drop what a 1-D column cannot be co-located with. The variable comes from
    # WHICH LIST RECEPTION FOUND THE STATION IN, not from the plan's claim about
    # it, so a mislabelled entry is filtered on what the station actually is.
    keep = [idx[s] for s in wanted
            if idx[s]["station_variable"] in PINNABLE_VARIABLES]
    dropped = [idx[s] for s in wanted
               if idx[s]["station_variable"] not in PINNABLE_VARIABLES]
    if dropped:
        print(f"   ⚠️  not pinning {len(dropped)} station(s) a 1-D column cannot "
              f"be co-located with: "
              f"{[(d['station_id'], d['station_variable']) for d in dropped]}")
        # ONE REASON PER VARIABLE. This printed the gauge argument — "integrates
        # and routes an upstream area" — for whatever was dropped, which after
        # water_table left the list meant recorder wells were refused with an
        # explanation about rivers.
        why = {
            "streamflow": ("a gauge integrates and routes an upstream area and "
                           "this model has no lateral transport, so no column "
                           "produces what it measures. Streamflow stays a "
                           "basin-aggregate comparison against the ensemble."),
            "water_table": ("recorder wells stopped costing columns 2026-08-12. "
                            "They cluster — twelve in one gridcell at "
                            "Brandywine, spanning two aquifers — and most "
                            "basins have none. The design is anchored at Fan "
                            "sites instead; the wells are still fetched and "
                            "still compared, on distance."),
        }
        for var in sorted({d["station_variable"] for d in dropped}):
            print(f"   ⚠️  {var}: {why.get(var, 'not a quantity a 1-D column produces at a point.')}")

    # And drop what sits outside the watershed. Reception tags every station
    # against the WBD polygon (`in_basin`); the tag is absent only when there was
    # no polygon to test against, and absent is not the same as False.
    outside = [c for c in keep if c.get("in_basin") is False]
    if outside:
        keep = [c for c in keep if c.get("in_basin") is not False]
        print(f"   ⚠️  not pinning {len(outside)} station(s) OUTSIDE the watershed: "
              f"{[c['station_id'] for c in outside]}")
        print(f"   ⚠️  Observations are fetched for the bounding box, which is "
              f"larger than the basin. A column out there is forced and soiled "
              f"from ground the study does not model.")

    # THE FAN ANCHORS ARE THE SAMPLER'S, NOT THE PLANNER'S. The planner is never
    # shown the Fan compilation (decided 2026-08-12: "it's okay the planner
    # doesn't learn Fan exists"), and it should not be — where to put a column
    # so the ensemble is interpretable is a design question, and the design is
    # this file's job. They are appended after the planner's pins so a budget
    # squeeze in expand() drops an anchor before a station the planner asked for.
    return keep + _fan_anchors(reception)


def _place_pinned(clients, pinned, bands, pts):
    """One column at each station's own coordinates.

    Elevation comes from a point 3DEP query AT the station, not from the nearest
    DEM grid point: at grid_n=120 over a basin this size the nearest grid point
    sits up to ~1 km away, which in this relief is worth hundreds of metres and
    would file the column under the wrong band. `station_elevation_m` is kept
    beside it wherever the station reports one, so a comparison can see how far
    the model surface sits from the instrument before trusting the pairing — the
    quantity step1_compare_swe had to give up on.
    """
    terr = clients.get("terrain")
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
# EXPANSION (uses the MCP servers)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_to_polygon(pts, rings):
    """Keep only points inside the watershed polygon (largest ring). Falls back
    to the original list if shapely/polygon is unavailable or clips everything."""
    try:
        from shapely.geometry import Polygon, Point
    except Exception:
        return pts
    polys = []
    for r in rings or []:
        if len(r) >= 4:
            try:
                polys.append(Polygon(r))
            except Exception:
                pass
    if not polys:
        return pts
    poly = max(polys, key=lambda p: p.area)
    # One prepared-geometry pass over all points instead of a Python-level
    # contains() per point: shapely builds the edge index once and vectorises
    # the test. Identical predicate, same points kept.
    try:
        import numpy as np
        from shapely import points as _shp_points, contains as _shp_contains
        arr = _shp_points(np.fromiter((q["lon"] for q in pts), float, len(pts)),
                          np.fromiter((q["lat"] for q in pts), float, len(pts)))
        keep = np.asarray(_shp_contains(poly, arr), dtype=bool)
        inside = [q for q, k in zip(pts, keep) if k]
    except Exception:
        # shapely < 2 has no vectorised API; the scalar predicate is the same.
        inside = [q for q in pts if poly.contains(Point(q["lon"], q["lat"]))]
    return inside or pts          # never drop everything on a bad clip


def expand(clients, bbox, n_total, n_bands, grid_n=120, boundary=None,
           per_band=None, pinned=None, grid=None):
    """Strategy -> concrete columns.

    n_total   the column budget, after check() has reconciled the planner's
              n_columns against anything the request asked for. HARD: it is the
              number of ELM cases that will be built, and the enforcement path
              fixed in 22cecae depends on nothing here exceeding it.
    per_band  the planner's stratified count per band, obeyed as stated.
    pinned    station records from _pinned_from_plan; one column each, at the
              station's own coordinates.
    """
    terr = clients["terrain"]

    # PREFER A GRID THE CALLER ALREADY HAS. Reception fetches this exact grid --
    # data_gather.GRID_N is expand_sampling's own default, and the constant says
    # so -- and it RETRIES at higher density when fewer than MIN_IN_BASIN = 55
    # points land inside the basin. Re-fetching here repeated the query without
    # the retry, so the sampler could design on a fraction of the evidence
    # reception had already paid for and written to disk. Measured on
    # naches_2023 (2026-08-07): reception escalated to n=216 and kept 101 points
    # in basin; this call, flat at n=120, got 29. Bands 1 and 5 held two points
    # each against per_band=3, so the ensemble came out 17 columns instead of 19
    # -- and worse than the count, the BAND EDGES were computed from that sparse
    # sample, so the strata did not match the relief reception had characterised.
    if grid:
        pts = [p for p in grid if p.get("elevation_m") is not None]
        print(f"  DEM grid: {len(pts)} points from reception (no re-fetch)")
    else:
        got = terr.call_tool_json("sample_elevation_grid",
                                  {**bbox, "n": grid_n}) or {}
        pts = [p for p in got.get("points", []) if p.get("elevation_m") is not None]
    if boundary:                  # clip the rectangular bbox sample to the real basin
        n0 = len(pts)
        pts = _clip_to_polygon(pts, boundary)
        print(f"  clipped DEM grid -> {len(pts)}/{n0} points inside the watershed")
    if not pts:
        return {"error": "no elevation points returned for bbox", "bbox": bbox}

    bands = _make_bands([p["elevation_m"] for p in pts], n_bands)
    by_band = {i: [] for i in range(len(bands))}
    for p in pts:
        by_band[_assign_band(p["elevation_m"], bands)].append(p)
    counts = [len(by_band[i]) for i in range(len(bands))]

    # ── PINNED FIRST: they are the validation design, and they consume budget ──
    pin_cols = _place_pinned(clients, pinned or [], bands, pts)
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

    return {"bbox": bbox, "n_requested": n_total, "n_columns": len(columns),
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
            # full DEM sample kept for the hypsometry / map plot (not saved to columns.json)
            "grid": [{"lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                      "elevation_m": p["elevation_m"]} for p in pts]}


# ─────────────────────────────────────────────────────────────────────────────
# FORCING LOOKUP — used by the model server's design figure
# ─────────────────────────────────────────────────────────────────────────────

# NOTE (Compy): this directory exists but is currently empty — the ctsmforc
# monthly Precip files are not yet staged (/compyfs/tran289/raw_nldas holds
# only raw hourly NLDAS_FORA files, a different format).
NLDAS_PRECIP = ("/compyfs/inputdata/atm/datm7/"
                "atm_forcing.datm7.NLDAS2.0.125d.v1/Precip")
# Second NLDAS-2 layout on Compy: the single-stream CLM files that
# DATM_MODE=CLMMOSARTTEST actually reads (complete 1979-2023, whereas the
# ...NLDAS2.0.125d.v1/Precip directory above is EMPTY here). Same 12 km grid
# and the same LATIXY/LONGXY + PRECTmms structure, so the reader below works
# on either -- only the filename differs. Preferring this one means the
# preview panel shows the forcing the run is actually driven by.
NLDAS_CLM_DIR = "/compyfs/inputdata/atm/datm7/NLDAS"
# Months read in parallel; see nldas_annual_precip. Six keeps enough seeks in
# flight to hide the latency without monopolising a login node.
NLDAS_WORKERS = int(os.environ.get("IDEAS_NLDAS_WORKERS", "6"))


def _nldas_month_file(year, mm):
    """Monthly NLDAS precip file, whichever of the two layouts is staged."""
    import os
    cands = [
        f"{NLDAS_PRECIP}/ctsmforc.NLDAS2.0.125d.v1.Prec.{year}-{mm:02d}.nc",
        f"{NLDAS_CLM_DIR}/clmforc.nldas.{year}-{mm:02d}.nc",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no NLDAS precip file for {year}-{mm:02d}; looked in "
        f"{NLDAS_PRECIP} and {NLDAS_CLM_DIR}")


def _soil_cov(c):
    """(clay_max %, max organic kg/m3) for the soil-coverage panel.

    Reads the DONOR profile, which _attach_donor_soil writes onto the columns
    before this figure is drawn — the soil ELM actually runs on. The old
    version also handled SSURGO's saturated conductivity; the CONUS 1 km
    surface dataset does not carry Ksat (ELM derives it internally from sand
    and organic) but does carry organic matter, which plays the same role of
    separating soils that differ hydraulically at similar clay content.
    """
    layers = (c.get("soil_profile") or {}).get("layers") or []
    comp = layers[0].get("component") if layers else None
    hz = [l for l in layers if l.get("component") == comp]
    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    clays = [v for v in (num(l.get("clay_pct")) for l in hz) if v is not None]
    org = [v for v in (num(l.get("organic_kg_m3")) for l in hz) if v is not None]
    return (max(clays) if clays else None), (max(org) if org else None)


def _nldas_month_slab(args):
    """(slab, mm_per_step) for one month's lat/lon box, or (None, 0) if absent.

    Module-level and self-contained so it can cross a process boundary; returns
    only the small box, never the 2 GB file.
    """
    year, mm, i0, i1, j0, j1 = args
    import numpy as np
    import xarray as xr
    try:
        ds = xr.open_dataset(_nldas_month_file(year, mm))
    except FileNotFoundError:
        return None, 0.0
    try:
        var = next(v for v in ds.data_vars if "PREC" in v.upper())
        nt = ds.sizes["time"]
        slab = np.asarray(ds[var][:, i0:i1 + 1, j0:j1 + 1].values)
        per_step = 86400.0 / (24 if nt > 400 else 8)    # hourly vs 3-hourly
    finally:
        ds.close()
    return slab, per_step


def nldas_annual_precip(cols, year):
    """Per-column annual NLDAS precipitation (mm/yr) from the nearest 12 km cell.

    Reads ONE small lat/lon slab per month, then takes every column out of it in
    memory.

    The obvious form -- `ds[var][:, i, j]` once per column -- looks like a cheap
    point read and is not. Precipitation is stored (time, lat, lon) contiguously
    in a ~2 GB monthly file, so pulling one (i, j) across all timesteps strides
    the whole array, and doing it per column repeats that traversal once per
    column per month: 14 columns x 12 months was 168 passes over 24 GB, roughly
    an hour of Lustre time to fill one panel of the design figure.

    The columns of a HUC8 span a handful of 12 km cells, so their bounding box
    is tiny; reading it whole costs one pass and a few MB.
    """
    import numpy as np
    import xarray as xr
    d0 = xr.open_dataset(_nldas_month_file(year, 1))
    lats = d0["LATIXY"].values[:, 0]
    lons = d0["LONGXY"].values[0, :]
    d0.close()
    idx = {c["id"]: (int(np.abs(lats - c["lat"]).argmin()),
                     int(np.abs(lons - (c["lon"] % 360.0)).argmin())) for c in cols}
    if not idx:
        return {}
    i0 = min(i for i, _ in idx.values()); i1 = max(i for i, _ in idx.values())
    j0 = min(j for _, j in idx.values()); j1 = max(j for _, j in idx.values())

    # The twelve months are read CONCURRENTLY. Precipitation is
    # (time, lat, lon) and contiguous, so a lat/lon box across all times is one
    # strided read PER TIMESTEP -- 744 per month, 8928 for the year. That is
    # seek latency, not bandwidth or CPU (the slab is a few MB), and it was
    # 4 min 14 s of a run's step 0. Independent files, so overlapping them is
    # the same trick that fixed the warm start.
    tot = {cid: 0.0 for cid in idx}
    jobs = [(year, mm, i0, i1, j0, j1) for mm in range(1, 13)]
    parts = None
    if len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=NLDAS_WORKERS) as ex:
                parts = list(ex.map(_nldas_month_slab, jobs))
        except Exception:
            parts = None                      # fall through to serial
    if parts is None:
        parts = [_nldas_month_slab(j) for j in jobs]

    for slab, per_step in parts:
        if slab is None:
            continue
        for cid, (i, j) in idx.items():
            tot[cid] += float(slab[:, i - i0, j - j0].sum()) * per_step
    return tot


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
    ap.add_argument("--run-dir", help="pipeline output dir (reads brief + plan)")
    ap.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat (overrides brief)")
    ap.add_argument("--n", type=int, help="number of columns (overrides plan)")
    ap.add_argument("--bands", type=int, default=0, help="number of elevation bands")
    ap.add_argument("--grid-n", type=int, default=120, help="DEM sample density")
    ap.add_argument("--per-band", type=int, default=0,
                    help="stratified columns per band (overrides plan.sampling.per_band)")
    ap.add_argument("--no-pin", action="store_true",
                    help="skip station pinning (stratified columns only)")
    args = ap.parse_args()

    out_dir = Path(args.run_dir) if args.run_dir else Path(".")
    out = out_dir / "columns.json"

    print("=" * 72)
    print("TIER-2 EXPANDER  —  strategy -> concrete columns (deterministic; no execution)")
    print("=" * 72)

    if args.run_dir and out.exists():
        # Pure read: re-plot/inspect an already-materialized run-dir — no fetch.
        res = json.loads(out.read_text())
        print(f"reading existing {out} ({res.get('n_columns')} columns — no MCP fetch)")
    else:
        # Materialize once: resolve domain + N, then sample DEM/soil/Fan.
        bbox = n_total = huc = None
        n_bands = args.bands or 4
        per_band, pinned = args.per_band, []
        if args.run_dir:
            rd = Path(args.run_dir)
            brief = json.loads((rd / "reception_brief.json").read_text())
            plan = json.loads((rd / "plan.json").read_text()) if (rd / "plan.json").exists() else {}
            if (plan.get("archetype")
            or (plan.get("model_choice") or {}).get("design_archetype")) == "conceptual":
                sys.exit("Plan is 'conceptual' archetype — no spatial expansion needed.")
            bbox = _bbox_from_brief(brief)
            n_total = _n_from_plan(plan)
            huc = (brief.get("domain") or {}).get("huc")
            if not args.bands:
                # The planner's count, not len(brief.heterogeneity.elevation_bands)
                # — those are band edges, never a band count. See _n_bands_from_plan.
                n_bands = _n_bands_from_plan(plan) or 4
            per_band = per_band or _per_band_from_plan(plan)
            rec_f = rd / "reception.json"
            if rec_f.exists() and not args.no_pin:
                pinned = _pinned_from_plan(plan, json.loads(rec_f.read_text()))
            elif (plan.get("validation") or []) and not args.no_pin:
                print(f"⚠️  {rec_f} not found — the plan names validation stations "
                      f"but NO column will be pinned to one.")
        if args.bbox:
            v = [float(x) for x in args.bbox.split(",")]
            bbox = {"min_lon": v[0], "min_lat": v[1], "max_lon": v[2], "max_lat": v[3]}
        if args.n:
            n_total = args.n
        if not bbox:
            sys.exit("No bbox (give --run-dir with a site brief, or --bbox).")
        if not n_total:
            sys.exit("No column count (give --run-dir with a plan, or --n).")

        print(f"bbox: {bbox} | N={n_total} | bands={n_bands} | "
              f"per_band={per_band or '-'} | pinned={len(pinned)}")
        clients = MCPManager("mcp_config.json").get_all_clients()
        boundary = None
        if huc:   # watershed polygon — clips sampling to the basin + outlines the map
            b = clients["terrain"].call_tool_json(
                "get_watershed_boundary", {"huc": huc, "huc_level": len(huc)}) or {}
            boundary = b.get("rings")
        res = expand(clients, bbox, n_total, n_bands, grid_n=args.grid_n,
                     boundary=boundary, per_band=per_band, pinned=pinned)
        if "error" in res:
            sys.exit(res["error"])
        if boundary:
            res["boundary"] = boundary
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
