#!/usr/bin/env python3
"""
External data gathering
src/core/data_gather.py

Infrastructure, not an agent: it makes no decisions and calls no model. It sits
in core/ because that is what it is — reception CALLS it and receives a report,
rather than containing one.

Reception is the ONLY component that reaches outside the framework. Everything
downstream — the planner, the sampler, the validator, the analyzer — reads what
this module wrote. That rule exists because the alternative kept producing two
answers to one question: reception reported 25 stream gauges where validation
found 93, because the LLM was summarising a tool result truncated to 12,000
characters while the validator queried directly.

So nothing here is decided by the model. Once the LLM has fixed the DOMAIN and
the PERIOD, these fetches always happen, always the same way, and their results
are recorded verbatim with provenance. The LLM may summarise them afterwards; it
may not author them.

Two consequences worth stating:

  * The grid is fetched at the SAMPLER's resolution, not a coarser "briefing"
    one, so Tier 2 never has to fetch anything. It reads this grid, picks
    columns from it, and looks up each column's water table — no network, fully
    reproducible from reception.json alone.

  * A failed fetch is recorded as `ok: false` WITH its error. It is never an
    empty result, because "we could not look" and "there is nothing there" are
    different findings and were being conflated.

SUBSURFACE PROPERTIES, AND WHERE THEY COME FROM (settled 2026-08-17). They were
briefly a SOIL SURVEY: gather_soil asked geology-mcp for SSURGO horizons at every
grid point. A survey stops at about 1.5 m, so a model needing material properties
to tens of metres got the deepest horizon copied downward — 88 to 97% of a column
invented, and measured as such. That server is retired. gather_subsurface fetches
ParFlow CONUS2's own parameterisation instead, which covers the whole 392 m: its
top four layers are SSURGO-derived anyway, so nothing was lost above 2 m and
everything below it was gained. ELM is unaffected either way — it reads its donor
soil from the CONUS surface dataset and never read this.
"""
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

GRID_N = 120          # sampler resolution; expand_sampling's own default
MIN_IN_BASIN = 55     # below this the sampler has too little to choose from
MAX_GRID_N = 480      # ceiling on the retry, so a pathological shape cannot
                      # turn one terrain call into a very large one

# HOW COMPLETE A RECORD HAS TO BE, AS A SHARE OF THE PERIOD (2026-08-12).
# Both of these used to be fixed counts, and a fixed count changes meaning with
# the length of the run: 300 days is 82% of one year but 41% of two, so the bar
# quietly dropped as the study got longer. They are fractions now, so the same
# rule means the same thing on any period.
#
# The two numbers differ because the observations do:
#
#   streamflow 90%  a gauge either runs all year or is broken. Near-complete is
#                   the normal state, so anything much short of it is a gap that
#                   would bias an annual water balance.
#   water table 30%  a recorder well is rarer and patchier, and a season of
#                   daily levels still shows how the water table MOVED, which is
#                   the one thing a well says that a static map cannot. Holding
#                   groundwater to the streamflow bar would throw away most of
#                   the little there is: of the 10 chain-eval basins only 6 have
#                   any well covering their simulation year at all.
MIN_DAYS_FRACTION = 0.90     # streamflow — TAGS the gauge, keeps it either way
MIN_OBS_FRACTION = 0.30      # water table — DROPS the well (see gather_observations)

# HOW MANY VISITS MAKE A FAN SITE A LONG-TERM MEAN (2026-08-12).
# Fan et al. (2013) is served as a point compilation, and its `record_count`
# is 1 for most sites: 3,620 of 3,919 at Naches, 3,780 of 4,201 at Brandywine.
# A "1927-2009 mean" built from one visit is a single measurement with a long
# date range attached — the same snapshot the USGS field-measurement path was
# retired for. Two is the lowest bar that makes the word "mean" true.
#
# NOT 12. A year of monthly visits would be the honest climatology, but it
# leaves Gunnison with 0 sites of 77, and a filter that empties a basin tells
# the analyzer the basin has no data when what it has is no *good* data.
FAN_MIN_RECORDS = 2
MAX_FAN_WELLS = 500     # they arrive best-recorded first; a dense basin should
                        # not put half a megabyte of static points in every file


def _period_days(yr_start: int, yr_end: int) -> int:
    """Calendar days in the resolved period, leap years included."""
    return (date(yr_end, 12, 31) - date(yr_start, 1, 1)).days + 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _call(clients, server: str, tool: str, args: Dict[str, Any],
          budget: Optional[float] = None) -> Dict[str, Any]:
    """One MCP call, with provenance and an explicit ok/error.

    Never raises: a stage that cannot fetch still produces a valid record
    saying so, which is what lets downstream tell failure from absence.

    `budget` RAISES THE CLIENT TIMEOUT for this call only. Each server's
    configured timeout is sized for its ordinary tool, and a BATCH tool is not
    that: geology is registered at 30 s, which fits one SSURGO point and not
    the 58 a basin's grid asks for in one call. Measured at 28.5 s direct, it
    went over inside reception and the whole soil fetch came back empty — a
    silent hole in the gather, recorded honestly but empty. The pattern is
    exp_manager_base._mcp_call's, for the same reason: the ceiling belongs to
    the CALL, not to the server.
    """
    rec = {"tool": f"{server}.{tool}", "args": args, "fetched_at": _now()}
    client = (clients or {}).get(server)
    if client is None:
        return {**rec, "ok": False, "error": f"no {server} client configured",
                "result": None}
    prev = getattr(client, "timeout", None)
    try:
        if budget and prev is not None and prev < budget:
            client.timeout = budget
        out = client.call_tool_json(tool, args)
    except Exception as e:                       # noqa: BLE001 - recorded, not raised
        return {**rec, "ok": False, "error": f"{type(e).__name__}: {e}"[:200],
                "result": None}
    finally:
        if prev is not None:
            client.timeout = prev
    if not isinstance(out, dict):
        return {**rec, "ok": False, "error": "tool returned no JSON object",
                "result": None}
    if out.get("error"):
        return {**rec, "ok": False, "error": str(out["error"])[:200], "result": out}
    return {**rec, "ok": True, "error": None, "result": out}


def _rings(boundary) -> List[List]:
    """WBD boundary -> exterior rings, whatever shape the terrain server used."""
    if not boundary:
        return []
    if isinstance(boundary, dict):
        geom = boundary.get("geometry") or boundary
        t = geom.get("type")
        if t == "Polygon":
            return [geom["coordinates"][0]]
        if t == "MultiPolygon":
            return [p[0] for p in geom["coordinates"]]
        return []
    if isinstance(boundary, list) and boundary and isinstance(boundary[0], list):
        return boundary
    return []


def _inside(lat, lon, rings) -> bool:
    """Ray casting. Rings are (lon, lat), as GeoJSON stores them."""
    if lat is None or lon is None or not rings:
        return True                              # no boundary -> keep everything
    hit = False
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            if (y1 > lat) != (y2 > lat):
                xin = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < xin:
                    hit = not hit
    return hit


def gather_grid(clients, bbox: Dict[str, float], huc: str = "", boundary=None,
                n: int = GRID_N, provenance: Optional[List] = None) -> Dict[str, Any]:
    """The DEM grid at sampler resolution, clipped to the basin. TERRAIN ONLY.

    Latitude, longitude, elevation, and nothing else. The modelled water-table
    fields used to be sampled here too; they are fetched with the observations
    now, so a function named for the grid returns only the grid.

    The clipped points ARE passed on to that fetch, so the water-table lookup
    still only sees points a column could sit on — 58 at Naches, not the 121
    the bounding box produces.
    """
    prov = provenance if provenance is not None else []

    # The WBD polygon. Fetched HERE because reception is the only component
    # that reaches outside — the Experiment Manager used to make this call
    # itself, at which point the grid it clipped and the grid reception
    # described were two different things.
    if boundary is None and huc:
        b = _call(clients, "terrain", "get_watershed_boundary",
                  {"huc": str(huc), "huc_level": 8})
        prov.append({k: b[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
        boundary = (b.get("result") or {}).get("rings")

    # RETRIED, because this one call decides everything downstream. It is a few
    # hundred point queries against 3DEP and it is occasionally just slow: on
    # 2026-08-08 it timed out on 2 of 12 basins, and each time the case lost its
    # grid, then its boundary, then every station tag, then all four of its
    # pinned columns. A second attempt costs a minute; losing the case costs the
    # case.
    for attempt in range(2):
        g = _call(clients, "terrain", "sample_elevation_grid",
                  {**bbox, "n": int(n)})
        prov.append({k: g[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
        pts = [p for p in ((g.get("result") or {}).get("points") or [])
               if p.get("elevation_m") is not None]
        if pts:
            break
        if attempt == 0:
            print(f"   ⚠️  elevation grid came back empty ({g.get('error')}) "
                  f"— retrying once")

    rings = _rings(boundary)
    clipped = [p for p in pts if _inside(p.get("lat"), p.get("lon"), rings)]
    # A BAD POLYGON must not empty the grid — but an empty FETCH is not a bad
    # polygon, and conflating them threw away a boundary that had arrived
    # perfectly well. `rings = []` then propagated as boundary=None, so no
    # station could be tagged in_basin and the planner, told to pin only what it
    # could confirm was inside, pinned nothing. One slow request cost naches_1979
    # and brandywine_2010 their whole validation design. The fallback now needs
    # points to have come back at all.
    if pts and not clipped:
        clipped, rings = pts, []

    # A grid is requested over the BOUNDING BOX but used inside the BASIN, and
    # the two differ by the shape of the watershed: compact basins keep ~50% of
    # the points, an elongated coastal strip keeps 18%. Central Coastal
    # California came back with 21 usable points for 4984 km2 — 4.2 per
    # 1000 km2 against 20-37 elsewhere — which leaves farthest-point selection
    # almost nothing to choose from and barely samples the elevation bands.
    # So the request is scaled by the fill ratio and retried once.
    if pts and len(clipped) < MIN_IN_BASIN and int(n) < MAX_GRID_N:
        fill = max(len(clipped) / len(pts), 0.05)
        bigger = min(int(int(n) / fill), MAX_GRID_N)
        if bigger > int(n):
            g2 = _call(clients, "terrain", "sample_elevation_grid",
                       {**bbox, "n": bigger})
            prov.append({k: g2[k] for k in
                         ("tool", "args", "fetched_at", "ok", "error")})
            pts2 = [p for p in ((g2.get("result") or {}).get("points") or [])
                    if p.get("elevation_m") is not None]
            clip2 = [p for p in pts2 if _inside(p.get("lat"), p.get("lon"), rings)]
            if len(clip2) > len(clipped):
                pts, clipped, n = pts2, clip2, bigger

    grid = [{"lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
             "elevation_m": p["elevation_m"]} for p in clipped]

    # THE GRID IS TERRAIN AND NOTHING ELSE (2026-08-12, user's rule). The
    # water-table fields used to be sampled here, which put modelled water
    # tables inside a function whose name promises elevations; they are fetched
    # with the observations instead, where the other things a run is judged
    # against already live.
    elevs = [p["elevation_m"] for p in grid]
    return {
        "n_requested": int(n),
        "n_returned": len(pts),
        "n_in_basin": len(grid),
        "clipped_to_watershed": bool(rings),
        "boundary": rings or None,        # kept so a re-plot needs no refetch
        "elevation_min_m": min(elevs) if elevs else None,
        "elevation_max_m": max(elevs) if elevs else None,
        "relief_m": (round(max(elevs) - min(elevs), 1) if elevs else None),
        "points": grid,
    }


# A station-year of daily SWE is ~7.7 KB columnar. Five stations is fine; a
# basin with forty SNOTEL sites would put 300 KB of series into every
# reception.json. Cap the series, keep every station's SUMMARY, and record
# what was dropped — a silent first-N would quietly bias the sample toward
# whatever order the server returned.
MAX_SWE_SERIES = 12


def gather_subsurface(clients, bbox_str: str, run_dir=None,
                      provenance: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """ParFlow CONUS2's subsurface parameters over the basin, as a local file.

    THE SAME SHAPE AS THE MODELLED WATER TABLE, and for the same reasons: one
    fetch per basin, written beside reception.json, read afterwards at any point
    by core/conus2_subsurface.sample with no network and no PIN. Sampling it per
    column would be five requests per column to a university's server, and the
    answer cannot change — every field is `static`.

    WHAT IT IS FOR. A subsurface model needs material properties all the way
    down, and a soil survey stops at about 1.5 m. Until 2026-08-17 everything
    below that was the deepest surveyed horizon copied downward — 88 to 97% of
    a column invented. These five fields parameterise the whole 392 m.

    UNCONDITIONAL ON EVERY SITE RUN, never gated on which model was chosen, for
    the reason the other gathers give: gating would make two runs of the same basin
    carry different provenance depending on a decision taken moments earlier.
    The cost is controlled by the CACHE rather than by a condition — the server
    keeps a copy keyed by grid_bounds, so the second study of a basin makes no
    request at all.

    NEEDS A RUN DIRECTORY, because the product is a file rather than a number.
    Without one there is nowhere to put it and this says so instead of fetching
    something it would immediately discard.
    """
    prov = provenance if provenance is not None else []
    if not run_dir:
        return {"ok": False, "error": "no run_dir — the subsurface is a file, "
                                      "not a value, and needs somewhere to go"}
    r = _call(clients, "hydrodata", "download_conus2_subsurface",
              {"bbox": bbox_str, "out_dir": str(run_dir)}, budget=1800.0)
    prov.append({k: r[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error")}
    rr = r.get("result") or {}
    if not rr.get("ok"):
        return {"ok": False, "error": rr.get("error")}
    return {
        "ok": True,
        "source": rr.get("source"),
        # WHAT WAS PAID, kept because it is the number that decides whether an
        # unconditional fetch is defensible. `reused: true` with zero requests
        # is the normal case after a basin's first study.
        "reused": rr.get("reused"),
        "n_requests": rr.get("n_requests"),
        "grid_bounds": rr.get("grid_bounds"),
        "arrays": rr.get("arrays"),
        "meta": rr.get("meta"),
        "covers": ("the whole basin box, as a field — not a value per column. "
                   "Read it at any point with core.conus2_subsurface.sample; "
                   "it answers anywhere inside the box, including locations "
                   "chosen after this fetch."),
        "when": ("STATIC. A spun-up equilibrium parameterisation, with no "
                 "period — the same field whatever years the study runs."),
        "failed_fields": rr.get("failed") or [],
    }


def _cap_series(stations, limit, key="daily"):
    """Keep the series on the `limit` deepest-snowpack stations, summaries on all."""
    ranked = sorted(stations, key=lambda s: -(s.get("peak_swe_mm") or 0))
    dropped = 0
    for st in ranked[limit:]:
        if st.pop(key, None) is not None:
            dropped += 1
    if dropped:
        for st in ranked[:limit]:
            st.setdefault("_note", f"{dropped} further station(s) kept their "
                                   f"summary but not their daily series")
    return ranked


def _tag_in_basin(out: Dict[str, Any], boundary) -> None:
    """Mark every station inside or outside the watershed polygon.

    Observations are fetched for the BBOX, which is strictly larger than the
    basin, and until 2026-08-08 nothing tested them against the polygon the DEM
    grid was already clipped to — so stations from the corners of the rectangle
    reached the planner indistinguishable from ones inside the divide, and four
    of Gunnison's columns were pinned outside its own watershed.

    Stations are TAGGED, never dropped. A gauge just below the outlet is
    correctly outside and is still the right gauge for a basin-aggregate
    comparison. What the tag prevents is PINNING a column there: a column
    outside the divide is forced and soiled from ground the study does not model.
    """
    rings = _rings(boundary)
    if not rings:
        return
    for var, key in (("streamflow", "stations"), ("water_table", "wells"),
                     ("water_table_static", "wells"),
                     ("swe", "stations"), ("et", "towers")):
        for st in ((out.get(var) or {}).get(key) or []):
            if st.get("lat") is not None and st.get("lon") is not None:
                st["in_basin"] = _inside(st["lat"], st["lon"], rings)


def gather_observations(clients, bbox_str: str, yr_start: int, yr_end: int,
                        boundary=None, run_dir=None,
                        provenance: Optional[List] = None) -> Dict[str, Any]:
    """Everything a run will be judged against, for the RESOLVED period.

    MEASURED AND MODELLED, IN ONE PLACE. Five measured: streamflow, the water
    table as a daily series, the water table as long-term well means, snow, ET.
    One modelled, under `water_table_modelled`: the ParFlow CONUS2 steady state.
    They are kept in separate blocks and never merged — a well record and a
    simulation's equilibrium field are not the same kind of number — but they
    are fetched together because they are wanted together, and because reception
    is the only component that reaches outside.

    Called once, after the period is fixed, and never again — the validator
    reads this rather than re-querying, so the two cannot disagree.

    `with_values` is on: the series are what the analyzer compares against, and
    fetching them here is what makes the run reproducible from one file.

    run_dir: where the modelled field's GeoTIFF is written. Without it that
    fetch is skipped — there is nowhere to put the file. It replaced
    `grid_points` on 2026-08-12: the field used to be sampled at the basin grid
    points, which was 116 single-cell requests to Princeton and, worse, could
    only answer at points reception happened to choose. sample_columns SNAPS
    columns onto the CONUS grid and moves them, so those were the wrong points.
    """
    prov = provenance if provenance is not None else []
    # ONE PERIOD FOR EVERY OBSERVABLE (2026-08-12). Snow used to be fetched on
    # the WATER year, Oct 1 of the previous year to Sep 30, on the reasoning
    # that the melt feeding this calendar year began the previous October.
    # That reasoning was about the snowpack, not about the comparison: THE RUN
    # IS THE CALENDAR YEAR, so those extra autumn months had no model output to
    # sit beside and could never be compared with anything. It also meant one
    # observable in reception.json answered a different question from the other
    # four, which is a trap for whoever reads it next.
    #
    # The peak survives the change: in the western basins this framework runs,
    # peak SWE falls around April 1, well inside the calendar year. What is
    # given up is the accumulation limb from the previous October — real, but
    # outside the simulated period either way.
    start, end = f"{yr_start}-01-01", f"{yr_end}-12-31"
    swe_start, swe_end = start, end

    # Both thresholds are a share of THIS period, never a fixed count, so they
    # keep their meaning whether the run is one year or five.
    n_days = _period_days(yr_start, yr_end)
    min_days = int(round(MIN_DAYS_FRACTION * n_days))
    min_obs = int(round(MIN_OBS_FRACTION * n_days))

    out: Dict[str, Any] = {"fetched_for": {"yr_start": yr_start, "yr_end": yr_end,
                                           "bbox": bbox_str, "period_days": n_days}}

    q = _call(clients, "usgs_water", "get_streamflow",
              {"bbox": bbox_str, "start_date": start, "end_date": end,
               "with_values": True, "min_days": min_days})
    prov.append({k: q[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    qr = q.get("result") or {}
    out["streamflow"] = {
        "ok": q["ok"], "error": q["error"],
        "n_in_bbox": qr.get("n_sites"),
        "n_with_records": qr.get("n_available"),
        # WHAT THE BAR DID, not what it was — the threshold itself is already in
        # provenance, recorded as the argument it is.
        "n_meeting_min_days": qr.get("n_meeting_min_days"),
        "stations": qr.get("available") or [],
    }

    # min_obs DROPS, it does not tag — the well server filters the list rather
    # than labelling it, which is the one place reception is not purely additive.
    # The count that went is recorded below so the loss is visible rather than
    # silent, and `n_below_min_obs` from the server says how many.
    w = _call(clients, "usgs_water", "get_water_table",
              {"bbox": bbox_str, "start_date": start, "end_date": end,
               "with_values": True, "min_obs": min_obs})
    prov.append({k: w[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    wr = w.get("result") or {}
    out["water_table"] = {
        "ok": w["ok"], "error": w["error"],
        "n_with_records": wr.get("n_wells_with_records"),
        # HOW MANY WENT. min_obs drops rather than tags, so without this the
        # record cannot tell a basin with no recorder wells from one whose
        # wells were simply too patchy to keep.
        "n_below_min_obs": wr.get("n_below_min_obs"),
        # WHY THERE ARE NO WELLS, CARRIED FORWARD (2026-08-12). Since the fetch
        # returns RECORDER wells only, an empty list no longer means the basin
        # has no groundwater data — Naches has thousands of wells and not one
        # that logs a daily level. Dropping the server's explanation here left
        # the planner reading `ok: true, wells: []` and concluding the basin has
        # nothing, which is the same "could not look" / "nothing there" mix-up
        # this file already guards against for a rate-limited fetch.
        "note": wr.get("note"),
        "observation_kind": wr.get("observation_kind"),
        "statistic": wr.get("statistic"),
        "wells": wr.get("wells") or [],
    }

    # ── the water table where it typically sits ─────────────────────────────
    # A SECOND KIND OF MEASUREMENT, NOT A SECOND SOURCE OF THE SAME ONE. The
    # block above is a daily series over the run's own period; this is one mean
    # per site over 1927-2009. Kept apart because they answer different
    # questions — how the water table MOVED, and where it USUALLY IS — and a
    # basin can easily have one and not the other. Naches has no recorder well
    # in any year and 3,919 Fan sites.
    #
    # OBSERVATIONS, on hydrodata's own account: its catalogue types fan_2013 as
    # `point_observations`. It arrives from the same server as the modelled
    # fields and is nothing like them, so it sits here with the measurements
    # rather than under `water_table_modelled`.
    #
    # One request for the whole bounding box, and the only hydrodata call
    # reception makes per basin that is not a raster.
    fa = _call(clients, "hydrodata", "get_fan2013_wells",
               {"bbox": bbox_str, "min_records": FAN_MIN_RECORDS})
    prov.append({k: fa[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    far = fa.get("result") or {}
    fan_wells = far.get("wells") or []
    out["water_table_static"] = {
        "ok": fa["ok"], "error": fa["error"],
        "n_wells": far.get("n_wells"),
        # what the visit filter cost, so an empty list is never read as an
        # empty basin — the same reason n_below_min_obs is carried above
        "n_below_min_records": far.get("n_below_min_records"),
        "records_per_site": far.get("records_per_site"),
        "observation_kind": far.get("observation_kind"),
        "period": far.get("period"),
        "summary": far.get("summary"),
        "wells": fan_wells[:MAX_FAN_WELLS],
        "n_wells_kept": min(len(fan_wells), MAX_FAN_WELLS),
    }

    # with_values: the DAILY series, not just the peak. Peak alone cannot
    # answer the question SWE is actually good for — accumulation and melt
    # TIMING, which is far less sensitive to the elevation offset between a
    # SNOTEL site and a model column than magnitude is. Streamflow and water
    # table already carry their series; this was the one observable that
    # threw its away, and n_obs=365 in the old output proves the server had
    # it all along.
    s = _call(clients, "snotel", "get_swe",
              {"bbox": bbox_str, "start_date": swe_start, "end_date": swe_end,
               "with_values": True})
    prov.append({k: s[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    sr = s.get("result") or {}
    out["swe"] = {
        "ok": s["ok"], "error": s["error"],
        "period": f"{swe_start}/{swe_end}",
        "n_stations": sr.get("n_stations"),
        "n_reporting": sr.get("n_reporting"),
        "stations": _cap_series(sr.get("stations") or [], MAX_SWE_SERIES),
    }

    # ET, the observable a 1-D column is best matched to: vertical, local to the
    # tower footprint, and computed directly by ELM. SWE and water table are the
    # other two a column can produce at a point; streamflow is not, because a
    # gauge integrates and routes an upstream area that the model has no lateral
    # transport to represent.
    #
    # with_values is OFF, deliberately. AmeriFlux flux data is not on an open
    # endpoint — it needs a registered account and data-use-policy acceptance —
    # so asking for values would return ok=false and lose the tower list with it.
    # Discovery is what pinning needs anyway: coordinates, and whether the tower
    # was running. The comparison step needs the series, and that needs
    # credentials.
    #
    # Coverage is thin and that is the point of recording it rather than
    # assuming: of the 13 chain-eval basins, 2 had a tower operating in their
    # simulation year.
    e = _call(clients, "ameriflux", "get_et",
              {"bbox": bbox_str, "start_date": start, "end_date": end})
    prov.append({k: e[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    er = e.get("result") or {}
    out["et"] = {
        "ok": e["ok"], "error": e["error"],
        "n_in_bbox": er.get("n_in_bbox"),
        "n_operating": er.get("n_operating"),
        "n_with_released_data": er.get("n_with_released_data"),
        # only towers that were RUNNING in the period can validate it
        "towers": [t for t in (er.get("towers") or []) if t.get("operating")],
        "values_available": False,
        "values_note": "series need an AmeriFlux account; discovery only",
    }

    # ── the modelled water table, as a file rather than a list ──────────────
    # ONE REQUEST, AND THE ANSWER IS A RASTER. This used to sample two gridded
    # estimates at every basin grid point: 58 points x 2 variables x ma_2025
    # was 116 single-cell requests to a university's server, ~114 s, repeated
    # by anything downstream that wanted a number.
    #
    # It is a GeoTIFF now, for a reason that outlives the traffic: sample_columns
    # SNAPS each column onto the CONUS grid, which MOVES it. The points
    # reception knows about are therefore not the points anyone later asks
    # about, and a list of 58 values cannot answer a question about a location
    # that was not in the list. A raster can, at any point, with no network.
    # Read it with src/core/static_wtd.py.
    #
    # ma_2025 was dropped entirely on 2026-08-12 (user's decision): its 24 m
    # grid made the same bounding box 7.7 million cells against ParFlow's 4,650,
    # which is a 62 MB download per basin to answer questions about 58 columns.
    #
    # NOT AN OBSERVATION, and in its own block for it. This is simulation
    # output, so nothing may be scored against it as if the model were being
    # checked against a measurement.
    if run_dir:
        r = _call(clients, "hydrodata", "download_conus2_wtd",
                  {"bbox": bbox_str, "out_dir": str(run_dir)})
        prov.append({k: r[k] for k in
                     ("tool", "args", "fetched_at", "ok", "error")})
        rr = r.get("result") or {}
        out["water_table_modelled"] = {
            "ok": r["ok"], "error": r["error"],
            "parflow_conus2": {k: rr.get(k) for k in
                               ("path", "reused", "shape", "resolution_m",
                                "crs", "grid_bounds", "pad_cells", "mb",
                                "summary", "dataset", "variable",
                                # DIAGNOSTICS, and they earned their place: the
                                # second says a raster for a DIFFERENT bbox was
                                # found here and written over, which is what a
                                # run directory reused for a second basin looks
                                # like. The first is >0 on a coastal or border
                                # basin, where the box runs off CONUS2.
                                "replaced_raster_for_another_bbox",
                                "n_perimeter_points_outside_domain",
                                "bbox_written_for", "covers_bbox")},
            "note": ("the ParFlow CONUS2 steady state over this bbox, written "
                     "once as a GeoTIFF — modelled context, never an "
                     "observation. Read it with static_wtd.sample(run_dir, "
                     "lats, lons); do not re-fetch it."),
        }

    _tag_in_basin(out, boundary)
    return out


def summarise(observations: Dict[str, Any]) -> Dict[str, Any]:
    """The compact view the PLANNER gets — never the raw series.

    Feeding a planner 170 kB of well measurements repeats the mistake that
    started all this: an LLM reasoning over a payload it cannot hold. Station
    coordinates ARE included, because pinning validation columns to observation
    locations needs them.
    """
    q = observations.get("streamflow") or {}
    w = observations.get("water_table") or {}
    s = observations.get("swe") or {}
    e = observations.get("et") or {}
    m = observations.get("water_table_modelled") or {}
    # THE ERROR TRAVELS WITH ok=False. Carrying the flag alone was not enough:
    # on 2026-08-07 the Chattahoochee fetch was rate-limited (HTTP 429) and the
    # planner, seeing {"ok": false, "stations": []}, wrote "observations_summary
    # lists no streamflow gauges with records in the domain" -- reporting an
    # absence for a basin thick with USGS gauges. "We could not look" and
    # "there is nothing there" are different findings, and a bare false is too
    # easy to skim past. The reason is now in front of whoever reads it.
    return {
        "streamflow": {
            "ok": q.get("ok"), "error": q.get("error"),
            "n_in_bbox": q.get("n_in_bbox"),
            "n_with_records": q.get("n_with_records"),
            "stations": [{k: st.get(k) for k in
                          ("id", "name", "lat", "lon", "in_basin",
                           "drainage_area_km2", "n_days")}
                         for st in (q.get("stations") or [])],
        },
        "water_table": {
            "ok": w.get("ok"), "error": w.get("error"),
            "n_with_records": w.get("n_with_records"),
            # The planner decides what a study can be judged against, so it has
            # to be able to tell "no data here" from "data of the wrong kind".
            "note": w.get("note"),
            "observation_kind": w.get("observation_kind"),
            # WELLS WITH COORDINATES FIRST. The cap keeps the planner's payload
            # small, but a plain head slice let it decide the wrong thing: on
            # brandywine_2010 (2026-08-07) 116 wells were fetched, 23 of them
            # with lat/lon, and the first 25 in fetch order had none — so the
            # planner saw 25 coordinate-less wells, wrote "wells reported
            # without coordinates (lat/lon null)", and 23 pinnable wells stayed
            # invisible in a basin that otherwise has nothing co-locatable.
            # A well without coordinates cannot be pinned and cannot be paired,
            # so it is the least useful kind to spend the cap on. A well outside
            # the divide is the next least useful, for the same reason.
            "wells": [{k: st.get(k) for k in
                       ("id", "lat", "lon", "in_basin", "n_obs", "wtd_m")}
                      for st in sorted(w.get("wells") or [],
                                       key=lambda x: (x.get("lat") is None,
                                                      x.get("in_basin") is False,
                                                      -(x.get("n_obs") or 0))
                                       )[:25]],
        },
        "swe": {
            "ok": s.get("ok"), "error": s.get("error"), "n_stations": s.get("n_stations"),
            "n_reporting": s.get("n_reporting"),
            "stations": [{k: st.get(k) for k in
                          ("triplet", "name", "lat", "lon", "in_basin",
                           "elevation_m", "peak_swe_mm", "peak_date")}
                         for st in (s.get("stations") or [])],
        },
        "et": {
            "ok": e.get("ok"), "error": e.get("error"),
            "n_operating": e.get("n_operating"),
            "values_available": False,
            "towers": [{k: st.get(k) for k in
                        ("id", "name", "lat", "lon", "in_basin", "elevation_m",
                         "igbp")}
                       for st in (e.get("towers") or [])],
        },
        # THE RANGE, NOT THE FIELD. What the planner can use is roughly how deep
        # this basin's modelled water table sits — a raster path and a cell
        # count are operational detail it cannot act on. Named `modelled` in the
        # key itself so it cannot be read as an observation inventory and
        # counted as evidence the run can be scored against.
        "water_table_modelled": ({
            "ok": m.get("ok"), "error": m.get("error"),
            "parflow_conus2_depth_m": (m.get("parflow_conus2") or {}).get("summary"),
            "note": "modelled, not measured — context only",
        } if m else None),
    }
