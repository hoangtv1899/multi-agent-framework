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
    return _Fake(lambda tool, args: {"depth_to_water_m": depth})


def _geo(layers=(("loam", 3),)):
    def fn(tool, args):
        lyrs = [{"texture_class": t} for t, _ in layers]
        return {"layers": lyrs, "num_layers": (layers[0][1] if layers else 0),
                "source": "SSURGO"}
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


def test_enriches_fan_wtd_and_full_soil_profile():
    res = exp.expand(_clients(fan=_fan(depth=7.5), geo=_geo()), BBOX, n_total=6, n_bands=3)
    for col in res["columns"]:
        assert col["fan_wtd_m"] == 7.5
        assert col["soil_top_texture"] == "loam"
        assert col["soil_layers"] == 3
        assert col["soil_profile"] is not None             # full profile carried
        assert col["soil_profile"]["source"] == "SSURGO"


def test_soil_profile_none_when_no_ssurgo_layers():
    res = exp.expand(_clients(geo=_geo(layers=())), BBOX, n_total=6, n_bands=3)
    for col in res["columns"]:
        assert col["soil_profile"] is None                 # None, not {}
        assert col["soil_top_texture"] is None


def test_runs_without_optional_servers():
    res = exp.expand(_clients(), BBOX, n_total=5, n_bands=3)   # terrain only
    assert res["n_columns"] == 5
    for col in res["columns"]:
        assert "fan_wtd_m" not in col and "soil_profile" not in col


def test_do_soil_false_skips_soil_but_keeps_fan():
    res = exp.expand(_clients(fan=_fan(), geo=_geo()), BBOX, n_total=6, n_bands=3,
                     do_soil=False)
    for col in res["columns"]:
        assert col["fan_wtd_m"] is not None
        assert "soil_profile" not in col


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
