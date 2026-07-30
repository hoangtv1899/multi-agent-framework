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

BBOX = {"min_lon": -121.5, "min_lat": 46.0, "max_lon": -121.0, "max_lat": 46.5}


# ── fakes ──────────────────────────────────────────────────────────────────
class _Fake:
    """MCP client stub: dispatches on tool name to a responder fn; records calls."""
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def call_tool_json(self, tool, args):
        self.calls.append((tool, args))
        return self.fn(tool, args)


def _grid(n=24, lo=100.0, hi=445.0):
    """A synthetic DEM sample: n points spread in lat/lon and elevation."""
    pts = []
    for i in range(n):
        f = i / (n - 1)
        pts.append({"lat": round(46.0 + f * 0.5, 5),
                    "lon": round(-121.5 + f * 0.5, 5),
                    "elevation_m": round(lo + f * (hi - lo), 1)})
    return {"points": pts}


def _terrain(grid=None):
    g = grid if grid is not None else _grid()
    return _Fake(lambda tool, args: g if tool == "sample_elevation_grid" else {})


def _fan(depth=5.0):
    """Answers the BATCHED tool: one call carries every column's point.

    Enrichment used to ask per column, which cost a fresh MCP session — and for
    Fan, a reopen of the dataset — per column. Measured live: 41.4 s for 6
    points per-call against 7.7 s batched, identical values.
    """
    def fn(tool, args):
        if tool == "get_fan_wtd_points":
            n = len(args.get("lats") or [])
            return {"n_points": n,
                    "points": [{"depth_to_water_m": depth} for _ in range(n)]}
        return {"depth_to_water_m": depth}          # legacy single-point form
    return _Fake(fn)


def _geo(layers=(("loam", 3),)):
    def _profile():
        lyrs = [{"texture_class": t} for t, _ in layers]
        return {"layers": lyrs, "num_layers": (layers[0][1] if layers else 0),
                "source": "SSURGO"}

    def fn(tool, args):
        if tool == "get_soil_profiles":
            n = len(args.get("lats") or [])
            return {"n_points": n, "profiles": [_profile() for _ in range(n)]}
        return _profile()                            # legacy single-point form
    return _Fake(fn)


def _clients(terrain=None, fan=None, geo=None):
    c = {"terrain": terrain or _terrain()}
    if fan is not None:
        c["fan_wtd"] = fan
    if geo is not None:
        c["geology"] = geo
    return c


# ── tests ────────────────────────────────────────────────────────────────────
def test_allocation_sums_to_n_total_and_column_shape():
    res = exp.expand(_clients(fan=_fan(), geo=_geo()), BBOX, n_total=8, n_bands=4)
    assert res["n_columns"] == 8
    assert sum(b["allocated"] for b in res["bands"]) == 8
    ids = [c["id"] for c in res["columns"]]
    assert ids == [f"col_{i:02d}" for i in range(1, 9)]   # sequential ids
    for col in res["columns"]:
        assert set(col) >= {"id", "lat", "lon", "elevation_m", "band", "band_range_m"}
        assert 1 <= col["band"] <= 4
        lo, hi = col["band_range_m"]
        assert lo <= col["elevation_m"] <= hi + 1          # within its band (rounded)


def test_enriches_fan_wtd():
    res = exp.expand(_clients(fan=_fan(depth=7.5)), BBOX, n_total=4, n_bands=2,
                     grid_n=24)
    assert all(c["fan_wtd_m"] == 7.5 for c in res["columns"])


def test_no_soil_is_gathered_at_sampling_time():
    """Sampling used to query a soil profile per column.

    The run is warm-started from the CONUS 1 km restarts, which carry the donor
    gridcell's own surfdata — so the soil ELM runs on is decided by the donor,
    not by anything queried here. The fetched profile became a field in
    columns.json that the model never saw, and the only thing preventing it
    from being analysed was that nobody happened to. _attach_donor_soil fills
    soil_profile after the warm start, and that is the only soil the run has.
    """
    geo = _geo()
    res = exp.expand(_clients(geo=geo), BBOX, n_total=4, n_bands=2, grid_n=24)
    assert not [c for c in geo.calls if "soil" in c[0]], \
        "sampling queried soil; the donor decides it"
    for c in res["columns"]:
        assert "soil_profile" not in c
        assert "soil_top_texture" not in c


def test_empty_grid_returns_error():
    res = exp.expand(_clients(terrain=_terrain(grid={"points": []})),
                     BBOX, n_total=5, n_bands=3)
    assert "error" in res and "columns" not in res


def test_bands_metadata_present_and_consistent():
    res = exp.expand(_clients(fan=_fan(), geo=_geo()), BBOX, n_total=8, n_bands=4)
    assert len(res["bands"]) == 4
    assert sum(b["grid_points"] for b in res["bands"]) == len(res["grid"])
    for b in res["bands"]:
        assert b["elev_lo_m"] <= b["elev_hi_m"]


def test_clips_sample_to_watershed_boundary():
    pytest.importorskip("shapely")
    # polygon covering only the lower-lat half of the grid (lat <= ~46.25)
    ring = [[-122.0, 45.9], [-120.0, 45.9], [-120.0, 46.25], [-122.0, 46.25],
            [-122.0, 45.9]]
    res = exp.expand(_clients(fan=_fan(), geo=_geo()), BBOX, n_total=4, n_bands=2,
                     boundary=[ring])
    assert len(res["grid"]) < 24                            # some points clipped out
    assert all(p["lat"] <= 46.26 for p in res["grid"])      # kept points inside poly


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment is batched — one call per source, not one per column
# ─────────────────────────────────────────────────────────────────────────────
class TestBatchedEnrichment:
    """Every MCP call opens a fresh session (HPC-safe by design), so asking per
    column paid a process spawn — and for Fan a dataset reopen — per column.
    Live: 6 points took 41.4 s per-call vs 7.7 s batched, same values."""

    def test_one_fan_call_serves_every_column(self):
        fan = _fan(depth=7.5)
        res = exp.expand(_clients(fan=fan), BBOX, n_total=6, n_bands=3,
                         grid_n=24)
        fan_calls = [c for c in fan.calls if "fan" in c[0]]
        assert len(fan_calls) == 1, f"expected 1 batched call, got {fan_calls}"
        assert fan_calls[0][0] == "get_fan_wtd_points"
        assert len(fan_calls[0][1]["lats"]) == len(res["columns"])
        assert all(c["fan_wtd_m"] == 7.5 for c in res["columns"])

    def test_points_are_sent_in_column_order(self):
        """Results are zipped back positionally — a reordering here would give
        every column its neighbour's water table, silently and plausibly."""
        fan = _fan()
        res = exp.expand(_clients(fan=fan), BBOX, n_total=5, n_bands=2,
                         grid_n=24)
        sent = [c for c in fan.calls if c[0] == "get_fan_wtd_points"][0][1]
        assert sent["lats"] == [c["lat"] for c in res["columns"]]
        assert sent["lons"] == [c["lon"] for c in res["columns"]]

    def test_a_short_reply_leaves_the_rest_none_not_shifted(self):
        """If the server returns fewer points than asked, the remainder must be
        None — never silently filled from the wrong column."""
        short = _Fake(lambda tool, args: {"n_points": 1, "points":
                                          [{"depth_to_water_m": 3.3}]})
        res = exp.expand(_clients(fan=short), BBOX, n_total=5, n_bands=2,
                         grid_n=24)
        vals = [c["fan_wtd_m"] for c in res["columns"]]
        assert vals[0] == 3.3
        assert all(v is None for v in vals[1:])
