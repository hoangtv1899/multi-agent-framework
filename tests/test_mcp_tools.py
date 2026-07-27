"""
Offline unit tests for MCP tool LOGIC (no network, deterministic).

Covers the pure helpers of the spatial MCP servers:
  * terrain  — grid generation, stats, elevation bands
  * fan_wtd  — sign/mask/time handling against a synthetic Fan-format fixture
  * usgs groundwater — OGC field-measurement / site parsing + ft->m conversion

Network-dependent behaviour is exercised separately by the CONUS sweep
(tools/mcp_conus_sweep.py), not here.

Run under the MCP runtime env:
    module load pytorch/2.8.0
    python3 -m pytest tests/test_mcp_tools.py -q
(If the `mcp` / xarray stack is unavailable, the relevant tests skip.)
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.fastmcp")

ROOT = Path(__file__).resolve().parents[1]
MCP = ROOT / "mcp"


def _load(path, name, extra_syspath=None):
    """Import a module by file path (servers aren't on the normal import path)."""
    if extra_syspath and str(extra_syspath) not in sys.path:
        sys.path.insert(0, str(extra_syspath))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# terrain — pure helpers
# ─────────────────────────────────────────────────────────────────────────────
pytest.importorskip("requests")
terrain = _load(MCP / "terrain-mcp" / "main.py", "terrain_main")


class TestTerrainHelpers:
    def test_grid_points_count_and_bounds(self):
        pts = terrain._grid_points(-122.0, 46.0, -120.0, 47.0, 24)
        assert 12 <= len(pts) <= 40                      # ~n, square-ish grid
        for lat, lon in pts:
            assert 46.0 <= lat <= 47.0
            assert -122.0 <= lon <= -120.0

    def test_stats_empty_is_safe(self):
        s = terrain._stats([])
        assert s["min_m"] is None and s["max_m"] is None and s["pct"] == {}

    def test_stats_values(self):
        s = terrain._stats([0, 10, 20, 30, 40])
        assert s["min_m"] == 0 and s["max_m"] == 40
        assert s["mean_m"] == 20 and s["relief_m"] == 40
        assert s["pct"]["p50"] == 20

    def test_equal_interval_bands(self):
        elevs = [100, 200, 300, 400]
        bands = terrain._equal_interval_bands(100, 400, 3, elevs)
        assert len(bands) == 3
        assert bands[0]["elev_lo_m"] == 100
        assert bands[-1]["elev_hi_m"] == 400
        # every point lands in exactly one band
        assert sum(b["n_points"] for b in bands) == len(elevs)

    def test_equal_interval_bands_degenerate(self):
        assert terrain._equal_interval_bands(None, None, 4, []) == []
        assert terrain._equal_interval_bands(100, 100, 4, [100]) == []


# ─────────────────────────────────────────────────────────────────────────────
# fan_wtd — sign / mask / time handling against a synthetic Fan-format tile
# ─────────────────────────────────────────────────────────────────────────────
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")
fan = _load(MCP / "fan-wtd-mcp" / "main.py", "fan_main")


@pytest.fixture
def fan_fixture(tmp_path, monkeypatch):
    """A tiny NAMERICA-style tile: (time, lat, lon), WTD negative-below, mask 1/0."""
    lats = np.arange(47.5, 45.99, -0.05)        # descending, like the real tile
    lons = np.arange(-122.0, -119.99, 0.05)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    wtd = -(5.0 + (LAT - 46.0) * 5.0)           # NEGATIVE = below surface
    mask = np.ones_like(wtd)
    mask[LAT < 46.2] = 0                         # SW block = out-of-domain (ocean)

    ds = xr.Dataset(
        {"WTD": (["time", "lat", "lon"], wtd[np.newaxis, :, :]),
         "mask": (["lat", "lon"], mask)},
        coords={"time": [1], "lat": lats, "lon": lons},
    )
    f = tmp_path / "fan_synth.nc"
    ds.to_netcdf(f)
    monkeypatch.setenv("FAN_WTD_NC", str(f))
    fan._DS.update(loaded=False, da=None, error=None, sign=1.0, path=None)  # reset cache
    yield f
    fan._DS.update(loaded=False, da=None, error=None, sign=1.0, path=None)


class TestFanWtd:
    def test_status_ready_and_sign_detected(self, fan_fixture):
        st = fan._status()
        assert st["status"] == "ready"
        # land values are negative -> sign auto-corrected to positive depth
        assert "auto-corrected" in st["depth_convention"]

    def test_point_returns_positive_depth(self, fan_fixture):
        r = fan._point(47.0, -120.7)            # land cell
        assert r["wtd_m"] < 0                    # raw file value negative
        assert r["depth_to_water_m"] > 0         # reported as positive depth
        assert abs(r["depth_to_water_m"] + r["wtd_m"]) < 1e-6   # depth == -wtd

    def test_point_masked_cell_is_none(self, fan_fixture):
        r = fan._point(46.0, -121.5)            # mask==0 region (lat < 46.2)
        assert r["depth_to_water_m"] is None
        assert "domain" in r.get("note", "")

    def test_sample_summary(self, fan_fixture):
        s = fan._sample(-122.0, 46.0, -120.0, 47.5, 40)
        assert s["summary"]["n_points"] > 0
        assert s["summary"]["n_valid"] <= s["summary"]["n_points"]
        if s["summary"]["n_valid"]:
            assert s["summary"]["min_depth_m"] >= 0   # positive depths only

    def test_missing_dataset_reports_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAN_WTD_NC", str(tmp_path / "does_not_exist.nc"))
        fan._DS.update(loaded=False, da=None, error=None, sign=1.0, path=None)
        st = fan._status()
        assert st["status"] == "unavailable"
        fan._DS.update(loaded=False, da=None, error=None, sign=1.0, path=None)


# ─────────────────────────────────────────────────────────────────────────────
# usgs groundwater — OGC response parsing
# ─────────────────────────────────────────────────────────────────────────────
pytest.importorskip("httpx")
# usgs-water-mcp is vendored, not a submodule; skip its parser tests on a
# checkout where it isn't present.
_GW_API = MCP / "usgs-water-mcp" / "groundwater_api.py"
gw = (_load(_GW_API, "gw_api", extra_syspath=MCP / "usgs-water-mcp")
      if _GW_API.exists() else None)


@pytest.mark.skipif(gw is None, reason="usgs-water-mcp code not present (NERSC-only)")
class TestGroundwaterParsers:
    def test_parse_sites(self):
        data = {"features": [{
            "id": "USGS-1",
            "properties": {"monitoring_location_name": "Well A",
                           "site_type_code": "GW", "aquifer_code": "N100",
                           "altitude": "1234"},
            "geometry": {"coordinates": [-120.5, 46.5]},
        }]}
        sites = gw._parse_sites(data)
        assert len(sites) == 1
        s = sites[0]
        assert s["id"] == "USGS-1"
        assert s["lon"] == -120.5 and s["lat"] == 46.5
        assert s["aquifer_code"] == "N100"

    def test_parse_sites_empty(self):
        assert gw._parse_sites({"features": []}) == []

    def test_parse_wtd_units_sort_and_skip(self):
        data = {"features": [
            {"properties": {"time": "2000-01-01T00:00:00Z", "value": "10.0",
                            "approval_status": "Approved"}},
            {"properties": {"time": "1990-01-01T00:00:00Z", "value": "4.89",
                            "approval_status": "Approved"}},
            {"properties": {"time": "2001-01-01", "value": "not-a-number"}},
        ]}
        obs, summary = gw._parse_wtd(data)
        assert summary["n_obs"] == 2                      # bad value skipped
        assert obs[0]["time"].startswith("1990")          # sorted ascending
        ten = next(o for o in obs if o["depth_to_water_ft"] == 10.0)
        assert abs(ten["depth_to_water_m"] - 3.048) < 1e-3   # ft -> m
        assert summary["min_depth_m"] <= summary["mean_depth_m"] <= summary["max_depth_m"]

    def test_parse_wtd_empty(self):
        obs, summary = gw._parse_wtd({"features": []})
        assert obs == [] and summary["n_obs"] == 0 and summary["mean_depth_m"] is None


@pytest.mark.skipif(gw is None, reason="usgs-water-mcp code not present")
class TestStreamflowCoverage:
    """Record availability: which gauges HAVE data, not which gauges exist.

    The distinction is the whole point of the tool. In the Naches bbox, 93
    stream gauges are present and one has 1995 daily discharge — a station
    count would have reported 93 and sent the design off believing streamflow
    was well observed.
    """
    DAILY = {"features": [
        {"properties": {"monitoring_location_id": "USGS-1", "value": 12.0}},
        {"properties": {"monitoring_location_id": "USGS-1", "value": 13.0}},
        {"properties": {"monitoring_location_id": "USGS-2", "value": 5.0}},
        {"properties": {"monitoring_location_id": "USGS-1", "value": None}},
    ]}
    SITES = {"features": [
        {"id": "USGS-1", "geometry": {"coordinates": [-121.17, 46.98]},
         "properties": {"monitoring_location_name": "American R", "drainage_area": "78.9"}},
        {"id": "USGS-2", "geometry": {"coordinates": [-120.9, 46.7]},
         "properties": {"monitoring_location_name": "Dry Creek", "drainage_area": None}},
        {"id": "USGS-3", "geometry": {"coordinates": [-120.8, 46.6]},
         "properties": {"monitoring_location_name": "No Records At All"}},
    ]}

    def test_only_gauges_meeting_min_days_are_available(self):
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=2)
        assert [a["id"] for a in out["available"]] == ["USGS-1"]
        assert out["n_available"] == 1
        assert out["n_sites"] == 3
        assert out["n_without_records"] == 2

    def test_null_values_do_not_count_as_records(self):
        """A feature with value=None is a gap in the record, not a day of data;
        counting it would let an empty gauge pass the min_days gate."""
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=3)
        assert out["available"] == []           # USGS-1 has 2 real days, not 3

    def test_drainage_area_is_converted_to_km2(self):
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=2)
        assert abs(out["available"][0]["drainage_area_km2"] - 204.4) < 0.1

    def test_coordinates_travel_with_the_gauge(self):
        """The observation map and the sampling design both need them; a bare
        id would force a second round-trip to place the gauge."""
        a = gw._parse_coverage(self.DAILY, self.SITES, min_days=2)["available"][0]
        assert a["lat"] == 46.98 and a["lon"] == -121.17

    def test_missing_drainage_area_is_none_not_a_crash(self):
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=1)
        two = next(a for a in out["available"] if a["id"] == "USGS-2")
        assert two["drainage_area_km2"] is None

    def test_empty_inputs_report_nothing_available(self):
        out = gw._parse_coverage({}, {}, min_days=300)
        assert out == {"n_sites": 0, "n_available": 0, "n_without_records": 0,
                       "min_days": 300, "available": []}

    def test_sorted_by_record_length(self):
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=1)
        assert [a["n_days"] for a in out["available"]] == [2, 1]

    def test_year_chunking_splits_a_long_window(self):
        """The OGC daily collection cancels any query over ~60 s of server time
        ("Long running query has been cancelled", returned as 400). Four years
        of one bbox sits under that budget; five is over. The failure is TIME,
        not size, so paging cannot help — the first page never arrives."""
        assert gw._year_chunks("1985-01-01", "1985-12-31") == [
            ("1985-01-01", "1985-12-31")]
        chunks = gw._year_chunks("1987-01-01", "1989-12-31")
        assert len(chunks) == 3
        assert chunks[0] == ("1987-01-01", "1987-12-31")
        assert chunks[-1] == ("1989-01-01", "1989-12-31")

    def test_chunk_edges_keep_the_users_actual_dates(self):
        """A window starting mid-year must not silently widen to the whole
        year — that would report records from months never simulated."""
        chunks = gw._year_chunks("1985-06-15", "1987-03-02")
        assert chunks[0] == ("1985-06-15", "1985-12-31")
        assert chunks[-1] == ("1987-01-01", "1987-03-02")

    def test_a_reversed_window_yields_nothing(self):
        assert gw._year_chunks("1990-01-01", "1988-12-31") == []

    def test_a_truncated_fetch_announces_itself(self):
        """An undercount is indistinguishable from a real finding — "only one
        gauge reports" reads the same whether it is true or whether we stopped
        reading early. So truncation must be stated, never inferred."""
        clipped = dict(self.DAILY, truncated=True)
        out = gw._parse_coverage(clipped, self.SITES, min_days=1)
        assert out["truncated"] is True
        assert "lower bound" in out["warning"]

    def test_a_complete_fetch_carries_no_warning(self):
        out = gw._parse_coverage(self.DAILY, self.SITES, min_days=1)
        assert "truncated" not in out and "warning" not in out


# ─────────────────────────────────────────────────────────────────────────────
# Tier-2 expander — deterministic geospatial logic (bands / allocate / select)
# ─────────────────────────────────────────────────────────────────────────────
exp = _load(ROOT / "tools" / "expand_sampling.py", "expand_sampling_mod")


class TestExpander:
    def test_make_bands(self):
        bands = exp._make_bands([100, 200, 300, 400, 500], 4)
        assert len(bands) == 4
        assert bands[0][0] == 100 and bands[-1][1] == 500

    def test_assign_band(self):
        bands = exp._make_bands([0, 400], 4)          # [0,100) [100,200) [200,300) [300,400]
        assert exp._assign_band(50, bands) == 0
        assert exp._assign_band(250, bands) == 2
        assert exp._assign_band(400, bands) == 3       # max -> last band (inclusive)

    def test_allocate_sums_and_min_one(self):
        alloc = exp._allocate([10, 20, 30, 40], 10)
        assert sum(alloc) == 10
        assert all(a >= 1 for a in alloc)              # every occupied band represented

    def test_allocate_skips_empty_band(self):
        alloc = exp._allocate([10, 0, 30], 8)
        assert alloc[1] == 0 and sum(alloc) == 8

    def test_allocate_fewer_columns_than_bands(self):
        alloc = exp._allocate([5, 10, 15], 2)          # 2 columns, 3 occupied bands
        assert sum(alloc) == 2 and alloc[0] == 0       # the two largest bands win

    def test_farthest_point_select(self):
        pts = [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 0},
               {"lat": 1, "lon": 1}, {"lat": 0.5, "lon": 0.5}]
        sel = exp._farthest_point_select(pts, 2)
        assert len(sel) == 2
        assert all(s in pts for s in sel)

    def test_clip_to_polygon(self):
        pytest.importorskip("shapely")
        ring = [[[-121.0, 46.0], [-120.0, 46.0], [-120.0, 47.0],
                 [-121.0, 47.0], [-121.0, 46.0]]]               # 1°x1° box
        pts = [{"lat": 46.5, "lon": -120.5, "elevation_m": 100},   # inside
               {"lat": 46.5, "lon": -119.5, "elevation_m": 200}]   # outside
        kept = exp._clip_to_polygon(pts, ring)
        assert len(kept) == 1 and kept[0]["lon"] == -120.5

    def test_clip_to_polygon_never_drops_all(self):
        pytest.importorskip("shapely")
        ring = [[[-121.0, 46.0], [-120.9, 46.0], [-120.9, 46.1],
                 [-121.0, 46.1], [-121.0, 46.0]]]
        pts = [{"lat": 48.0, "lon": -110.0, "elevation_m": 1}]      # none inside
        assert exp._clip_to_polygon(pts, ring) == pts              # fallback

    def test_plot_columns_renders(self, tmp_path):
        pytest.importorskip("matplotlib")
        res = {
            "bbox": {"min_lon": -121.5, "min_lat": 46.5, "max_lon": -120.5, "max_lat": 47.1},
            "n_columns": 3,
            "bands": [
                {"band": 1, "elev_lo_m": 300, "elev_hi_m": 900, "grid_points": 40, "allocated": 2},
                {"band": 2, "elev_lo_m": 900, "elev_hi_m": 1500, "grid_points": 30, "allocated": 1},
            ],
            "columns": [
                {"id": "col_01", "lat": 46.7, "lon": -120.7, "elevation_m": 450,
                 "band": 1, "fan_wtd_m": 0.7, "soil_top_texture": "loam"},
                {"id": "col_02", "lat": 46.6, "lon": -121.2, "elevation_m": 800,
                 "band": 1, "fan_wtd_m": 5.0, "soil_top_texture": "sandy loam"},
                {"id": "col_03", "lat": 46.9, "lon": -120.9, "elevation_m": 1200,
                 "band": 2, "fan_wtd_m": None, "soil_top_texture": None},  # missing data path
            ],
            "grid": [{"lat": 46.55, "lon": -121.4, "elevation_m": 400},
                     {"lat": 47.05, "lon": -121.4, "elevation_m": 900},
                     {"lat": 46.55, "lon": -120.6, "elevation_m": 1300},
                     {"lat": 47.05, "lon": -120.6, "elevation_m": 1800},
                     {"lat": 46.80, "lon": -121.0, "elevation_m": 1100}],
            "boundary": [[[-121.4, 46.55], [-120.6, 46.55], [-120.6, 47.05],
                          [-121.4, 47.05], [-121.4, 46.55]]],
        }
        out = tmp_path / "p.png"
        exp.plot_columns(res, str(out))
        assert out.exists() and out.stat().st_size > 1000


# ─────────────────────────────────────────────────────────────────────────────
# SNOTEL — AWDB response parsing
# ─────────────────────────────────────────────────────────────────────────────
snotel = _load(MCP / "snotel-mcp" / "main.py", "snotel_main")


class TestSnotel:
    def test_filter_stations(self):
        rows = [
            {"networkCode": "SNTL", "stationTriplet": "375:WA:SNTL",
             "name": "Bumping Ridge", "latitude": 46.9, "longitude": -121.4,
             "elevation": 4600, "huc": "170300020106"},
            {"networkCode": "SNTL", "stationTriplet": "1:CA:SNTL",
             "name": "Far", "latitude": 39.0, "longitude": -120.0},   # outside bbox
            {"networkCode": "USGS", "stationTriplet": "x",
             "name": "NotSnotel", "latitude": 46.9, "longitude": -121.4},  # wrong net
        ]
        out = snotel._filter_stations(rows, (-121.6, 46.6, -121.2, 47.1))
        assert len(out) == 1 and out[0]["triplet"] == "375:WA:SNTL"

    def test_parse_swe(self):
        data = [{"stationTriplet": "375:WA:SNTL", "data": [{
            "stationElement": {"elementCode": "WTEQ"},
            "values": [{"date": "2010-03-02", "value": 19.1},
                       {"date": "2010-03-01", "value": 18.9},
                       {"date": "2010-03-03", "value": None}]}]}]   # None skipped
        obs, summary = snotel._parse_swe(data)
        assert summary["n_obs"] == 2
        assert obs[0]["date"] == "2010-03-01"                       # sorted
        assert abs(obs[0]["swe_mm"] - 18.9 * 25.4) < 0.1            # in -> mm
        assert summary["peak_swe_mm"] == round(19.1 * 25.4, 1)
        assert summary["peak_date"] == "2010-03-02"

    def test_parse_swe_empty(self):
        obs, summary = snotel._parse_swe([])
        assert obs == [] and summary["n_obs"] == 0 and summary["peak_swe_mm"] is None


# ─────────────────────────────────────────────────────────────────────────────
# MCP server stderr routing
# ─────────────────────────────────────────────────────────────────────────────
class TestServerLogRouting:
    """Every stdio server logs at INFO through rich and its stderr is inherited,
    so an interactive session was buried under timestamped log lines between
    prompts. The output still matters when a server fails to start, so it is
    redirected to a file rather than discarded."""

    def _fresh(self, monkeypatch, path):
        import core.mcp_client as mc
        mc._ERRLOGS.clear()
        if path is None:
            monkeypatch.delenv("IDEAS_MCP_LOG", raising=False)
        else:
            monkeypatch.setenv("IDEAS_MCP_LOG", str(path))
        return mc

    def test_output_goes_to_a_file_by_default(self, monkeypatch, tmp_path):
        mc = self._fresh(monkeypatch, tmp_path / "logs" / "mcp.log")
        h = mc.mcp_errlog()
        assert h is not sys.stderr
        h.write("hello\n")
        h.flush()
        assert (tmp_path / "logs" / "mcp.log").read_text() == "hello\n"

    def test_the_directory_is_created(self, monkeypatch, tmp_path):
        mc = self._fresh(monkeypatch, tmp_path / "deep" / "nested" / "mcp.log")
        mc.mcp_errlog()
        assert (tmp_path / "deep" / "nested").is_dir()

    def test_stderr_is_the_documented_escape_hatch(self, monkeypatch):
        """Debugging a server that will not start needs the log back inline."""
        mc = self._fresh(monkeypatch, "stderr")
        assert mc.mcp_errlog() is sys.stderr
        monkeypatch.setenv("IDEAS_MCP_LOG", "-")
        assert mc.mcp_errlog() is sys.stderr

    def test_the_handle_is_reused_across_calls(self, monkeypatch, tmp_path):
        """A fresh session opens per tool call; reopening the file hundreds of
        times per run would race between threads."""
        mc = self._fresh(monkeypatch, tmp_path / "mcp.log")
        assert mc.mcp_errlog() is mc.mcp_errlog()

    def test_an_unwritable_path_falls_back_rather_than_failing_the_call(
            self, monkeypatch, tmp_path):
        """Losing the log is better than losing the tool call."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory")
        mc = self._fresh(monkeypatch, blocker / "sub" / "mcp.log")
        assert mc.mcp_errlog() is sys.stderr

    def test_both_session_openers_are_routed(self):
        """call_tool and list_tools each open their own session — missing
        either one leaves half the noise on the terminal."""
        src = (ROOT / "src" / "core" / "mcp_client.py").read_text()
        assert src.count("stdio_client(server_params, errlog=mcp_errlog())") == 2
        assert "stdio_client(server_params)" not in src
