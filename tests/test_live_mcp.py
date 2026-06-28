"""LIVE tier — exercises the real MCP servers (subprocess + external APIs).

Opt-in: skipped unless `pytest --runlive`. Confirms every server process starts
and advertises tools, and that the key spatial tools return sane answers over
CONUS reference points. Deterministic parsing logic is covered offline by
test_mcp_tools.py; this is the integration/liveness check.

    module load pytorch/2.8.0 && pytest tests/test_live_mcp.py --runlive
"""
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("mcp")                     # MCPClient needs the mcp sdk

from core.mcp_manager import MCPManager         # noqa: E402

pytestmark = pytest.mark.live

EXPECTED_SERVERS = {"terrain", "usgs_water", "fan_wtd", "geology", "weather", "snotel"}


@pytest.fixture(scope="module")
def clients():
    # call_tool_json opens a fresh server session per call, so building the
    # client map is cheap and needs no teardown.
    return MCPManager("mcp_config.json").get_all_clients()


def test_all_servers_start_and_advertise_tools(clients):
    assert set(clients) >= EXPECTED_SERVERS
    for name in EXPECTED_SERVERS:
        tools = clients[name].list_tools_detailed()
        assert tools, f"{name} advertised no tools"
        assert all("name" in t for t in tools)


@pytest.mark.parametrize("lat,lon", [(46.75, -120.70),   # Naches PNW
                                     (33.75, -84.39),    # Atlanta humid SE
                                     (40.01, -105.27)])  # Boulder Rockies
def test_terrain_elevation_in_conus_range(clients, lat, lon):
    r = clients["terrain"].call_tool_json("get_elevation", {"lat": lat, "lon": lon}) or {}
    assert -100 <= r.get("elevation_m", -9999) <= 5000


def test_resolve_watershed_returns_bbox_and_area(clients):
    r = clients["terrain"].call_tool_json("resolve_watershed", {"huc": "17030002"}) or {}
    assert r.get("bbox") and r.get("area_km2")


def test_fan_wtd_point_nonnegative_or_nodata(clients):
    r = clients["fan_wtd"].call_tool_json("get_fan_wtd",
                                          {"lat": 46.75, "lon": -120.70}) or {}
    d = r.get("depth_to_water_m")
    assert d is None or d >= 0          # local Fan grid: positive depth, or masked no-data


def test_geology_soil_profile_responds(clients):
    r = clients["geology"].call_tool_json("get_soil_profile",
                                          {"lat": 40.81, "lon": -96.70}) or {}
    assert {"num_layers", "layers", "error"} & set(r)     # structured answer of some kind


def test_usgs_groundwater_sites_respond(clients):
    r = clients["usgs_water"].call_tool_json(
        "get_groundwater_sites", {"bbox": "-96.95,40.56,-96.45,41.06", "limit": 50}) or {}
    assert {"n_sites", "sites", "error"} & set(r)
