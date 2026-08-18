"""Offline tests for the Tier-2 expander `expand()` (no terrain MCP, no network).

`test_mcp_tools.py` covers the pure helpers (_make_bands/_assign_band/_allocate);
this covers the full materialize wiring: DEM grid -> elevation bands ->
proportional allocation -> per-band spatial spread -> Fan WTD + soil enrichment
-> columns, plus the polygon clip and the empty-grid guard. Fake MCP clients
return canned grids/soil so the run is deterministic and offline.

Run under: module load pytorch/2.8.0
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp = _load("expand_sampling", "tools/expand_sampling.py")

# ── the one fixture ──────────────────────────────────────────────────────────
# _Fake, _terrain, _fan, _geo and _clients stood here and are deleted
# (2026-08-18). They stubbed MCP clients for a sampler that fetched its own
# grid and queried Fan and geology per column; it does none of that now —
# expand() takes reception's grid block and touches no server. What is left to
# test is arithmetic on a grid, so a grid is the only fixture.
def _grid(n=24, lo=100.0, hi=445.0):
    """A synthetic DEM sample: n points spread in lat/lon and elevation."""
    pts = []
    for i in range(n):
        f = i / (n - 1)
        pts.append({"lat": round(46.0 + f * 0.5, 5),
                    "lon": round(-121.5 + f * 0.5, 5),
                    "elevation_m": round(lo + f * (hi - lo), 1)})
    return {"points": pts}


# ── tests ────────────────────────────────────────────────────────────────────
def test_allocation_sums_to_n_total_and_column_shape():
    res = exp.expand(_grid(), n_total=8, n_bands=4)
    assert res["n_columns"] == 8
    assert sum(b["allocated"] for b in res["bands"]) == 8
    ids = [c["id"] for c in res["columns"]]
    assert ids == [f"col_{i:02d}" for i in range(1, 9)]   # sequential ids
    for col in res["columns"]:
        assert set(col) >= {"id", "lat", "lon", "elevation_m", "band", "band_range_m"}
        assert 1 <= col["band"] <= 4
        lo, hi = col["band_range_m"]
        assert lo <= col["elevation_m"] <= hi + 1          # within its band (rounded)


def test_no_soil_is_gathered_at_sampling_time():
    """Sampling used to query a soil profile per column.

    The run is warm-started from the CONUS 1 km restarts, which carry the donor
    gridcell's own surfdata — so the soil ELM runs on is decided by the donor,
    not by anything queried here. The fetched profile became a field in
    columns.json that the model never saw, and the only thing preventing it
    from being analysed was that nobody happened to. _attach_donor_soil fills
    soil_profile after the warm start, and that is the only soil the run has.
    """
    res = exp.expand(_grid(), n_total=4, n_bands=2)
    for c in res["columns"]:
        assert "soil_profile" not in c
        assert "soil_top_texture" not in c


def test_empty_grid_returns_error():
    res = exp.expand({"points": []}, n_total=5, n_bands=3)
    assert "error" in res and "columns" not in res


def test_the_sampler_fetches_nothing_and_clips_nothing():
    """A FUNCTION OF TWO FILES (2026-08-18). Reception fetched the DEM, fetched
    the polygon and clipped the one to the other; the sampler used to do all
    three again — a second fetch without reception's density retry, and a
    second clip with a different predicate. Asserted on the source, so a
    well-meaning re-add is a visible decision rather than a quiet regression."""
    src = (ROOT / "tools" / "expand_sampling.py").read_text()
    body = src[src.index("def expand("):src.index("def _bbox_from_brief")]
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    for word in ("sample_elevation_grid", "get_watershed_boundary",
                 "_clip_to_polygon", "shapely"):
        assert word not in code, f"expand() {word}s again; that is reception's"


def test_reception_boundary_passes_through_unclipped():
    """The polygon is carried into the result for the record and the figure —
    and NOT applied. Points outside it stay: reception already decided."""
    ring = [[-122.0, 45.9], [-120.0, 45.9], [-120.0, 46.25], [-122.0, 46.25],
            [-122.0, 45.9]]
    g = _grid(); g["boundary"] = [ring]
    res = exp.expand(g, n_total=4, n_bands=2)
    assert res["boundary"] == [ring]
    assert len(res["grid"]) == 24                           # nothing dropped here


def test_bands_metadata_present_and_consistent():
    res = exp.expand(_grid(), n_total=8, n_bands=4)
    assert len(res["bands"]) == 4
    assert sum(b["grid_points"] for b in res["bands"]) == len(res["grid"])
    for b in res["bands"]:
        assert b["elev_lo_m"] <= b["elev_hi_m"]


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment is batched — one call per source, not one per column
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# FAN ENRICHMENT WAS REMOVED FROM SAMPLING, 2026-08-07
# ─────────────────────────────────────────────────────────────────────
# test_enriches_fan_wtd and TestBatchedEnrichment lived here and are deleted
# rather than repaired: they pinned behaviour that is intentionally gone.
#
# Sampling selects on ELEVATION. Fan's water table was queried afterwards, at
# coordinates already chosen, and never influenced a placement — so the stage
# appeared to depend on a dataset it did not use. fan_wtd_m is PFLOTRAN's
# initial condition (it sets wt_in_domain; 10 of 19 columns on the 2019
# Gunnison sample had their water table below the modelled domain), so it is
# now the consumer's to fetch where that decision is made.
#
# The batching lesson those tests encoded still holds wherever it lands: one
# call for all coordinates, never one per column, because every MCP call is a
# fresh session and for Fan the dataset OPEN is the cost.
def test_sampling_no_longer_attaches_a_water_table():
    """The removal, asserted — so a well-meaning re-add is a visible decision."""
    src = (ROOT / "tools" / "expand_sampling.py").read_text()
    body = src[src.index("def expand("):src.index("def _bbox_from_brief")]
    # Comments stripped: the block above explains WHY the fetch left and names
    # the tool, so a substring match would flag its own documentation.
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "get_fan_wtd_points" not in code, (
        "sampling fetches Fan again; selection is elevation-only and the "
        "consumer fetches its own water table")
    assert "fan_wtd" not in code, "a fan_wtd client is back in expand()"
