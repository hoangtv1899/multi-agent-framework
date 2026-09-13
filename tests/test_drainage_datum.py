#!/usr/bin/env python3
"""The drainage datum's routing, pinned on a synthetic DEM.

    pytest tests/test_drainage_datum.py -v

A tilted plane whose cross-slope drains every cell sideways into one carved
channel that runs down to the south edge, plus one pit dug into the west
slope. Every number below follows from that construction, so the fill, the
receivers, the accumulation, HAND and the gauge check are each checked
against arithmetic rather than against each other. No /compyfs, no netCDF.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import drainage_datum as dd                            # noqa: E402

NY, NX, JC = 30, 40, 20          # rows, columns, the channel's column
PIT = (10, 5)
# Rows drain sideways, and the pit gathers the west ends of rows 9 to 11
# into row 11, so a hillslope cell accumulates up to 32 cells (27.7 km^2);
# the channel head already holds a full row (34.6 km^2). A threshold between
# the two makes the channel, and only the channel, stream.
A_KM2 = 30.0
DOWN, CROSS, CUT, DIG = 0.5, 4.0, 10.0, 30.0


def synthetic_grid():
    i = np.arange(NY)[:, None]
    j = np.arange(NX)[None, :]
    z = 200.0 - DOWN * i + CROSS * np.abs(j - JC)
    z[:, JC] -= CUT                           # the carved channel
    z[PIT] -= DIG                             # one pit on the west slope
    # Near the equator dx equals dy, so every cell is CELL_KM square.
    lat = np.arange(NY) * dd.CELL_DEG
    lon = 100.0 + np.arange(NX) * dd.CELL_DEG
    return {"topo": z, "std_elev": np.full_like(z, 6.0), "lat": lat, "lon": lon}


@pytest.fixture(scope="module")
def routed():
    grid = synthetic_grid()
    return grid, dd.route(grid)


class TestFill:
    def test_the_pit_rises_to_its_spill_and_nothing_else_moves(self, routed):
        grid, r = routed
        z, filled = grid["topo"], r["filled"]
        i, j = PIT
        spill = min(z[i + di, j + dj] for di, dj in dd.NEIGHBOURS)
        assert filled[PIT] == pytest.approx(spill + dd.FILL_EPS_M, abs=1e-9)
        others = np.ones_like(z, bool)
        others[PIT] = False
        assert np.array_equal(filled[others], z[others])
        assert z[PIT] < spill                     # the raw TOPO keeps its pit

    def test_only_the_channel_outlet_has_no_receiver(self, routed):
        _grid, r = routed
        outlet = (NY - 1) * NX + JC
        assert r["rec"][outlet] == -1
        assert int((r["rec"] < 0).sum()) == 1

    def test_a_hole_makes_its_neighbours_seeds(self):
        z = synthetic_grid()["topo"].copy()
        z[15, 30] = np.nan
        filled = dd.fill_pits(z)
        assert np.isnan(filled[15, 30])
        assert np.array_equal(np.isfinite(filled), np.isfinite(z))


class TestAccumulation:
    def test_every_receiver_holds_more_than_its_donor(self, routed):
        _grid, r = routed
        rec, acc = r["rec"], r["acc"]
        k = np.nonzero(rec >= 0)[0]
        assert np.all(acc[rec[k]] > acc[k])

    def test_the_channel_grows_downstream_and_drains_everything(self, routed):
        _grid, r = routed
        acc = r["acc"].reshape(NY, NX)
        assert np.all(np.diff(acc[:, JC]) > 0)
        area = dd.CELL_KM ** 2
        assert acc[NY - 1, JC] == pytest.approx(NY * NX * area, rel=1e-4)
        assert acc[2, JC] == pytest.approx(3 * NX * area, rel=1e-4)


class TestHand:
    def test_zero_on_the_channel(self, routed):
        grid, r = routed
        for i in (0, 7, NY - 1):
            h = dd.hand_of(grid, r, A_KM2, grid["lat"][i], grid["lon"][JC])
            assert h["hand_m"] == 0.0
            assert h["hand_path_km"] == 0.0
            assert h["hand_path_end"] == "stream"

    def test_the_drop_to_the_channel_elsewhere(self, routed):
        grid, r = routed
        i, j = 7, 12
        h = dd.hand_of(grid, r, A_KM2, grid["lat"][i], grid["lon"][j])
        assert h["hand_m"] == pytest.approx(CROSS * (JC - j) + CUT)
        dx = dd.CELL_KM * np.cos(np.radians(grid["lat"][i]))
        assert h["hand_path_km"] == pytest.approx((JC - j) * dx, rel=1e-6)
        assert h["hand_stream_area_km2"] == pytest.approx(
            r["acc"][i * NX + JC])
        assert h["hand_path_end"] == "stream"

    def test_the_pit_drains_over_its_spill_from_the_raw_topo(self, routed):
        grid, r = routed
        i, j = PIT
        h = dd.hand_of(grid, r, A_KM2, grid["lat"][i], grid["lon"][j])
        # spills to (11, 6), then west to east along row 11 into the channel
        assert h["hand_m"] == pytest.approx(
            grid["topo"][PIT] - grid["topo"][i + 1, JC])
        assert h["hand_path_end"] == "stream"
        assert h["hand_path_km"] > (JC - j) * dd.CELL_KM

    def test_a_threshold_nothing_reaches_ends_at_the_edge(self, routed):
        grid, r = routed
        end, km, how = dd.walk_to_stream(7 * NX + 12, r["rec"], r["step_km"],
                                         np.zeros(NY * NX, bool))
        assert how == "edge"
        assert end == (NY - 1) * NX + JC
        assert km > 0

    def test_a_point_outside_the_window_is_refused(self, routed):
        grid, _r = routed
        with pytest.raises(ValueError, match="outside"):
            dd.cell_of(grid["lat"], grid["lon"], 5.0, 100.0)


class TestGaugeCheck:
    def gauges(self, grid):
        z = grid["topo"]

        def at(i, j, gid, alt_offset=0.0):
            return {"id": gid, "lat": grid["lat"][i], "lon": grid["lon"][j],
                    "altitude_m": float(z[i, j]) + alt_offset}
        # (5, 13): 14 cells drain through it, 12.1 km^2: a stream for A <= 10
        # (0, JC): the channel head, 34.6 km^2: a stream for A <= 25
        # (8, JC): on the channel but 100 m off its altitude: never counts
        return [at(5, 13, "G-hill"), at(0, JC, "G-head"),
                at(8, JC, "G-wrong-alt", 100.0)]

    def test_scores_fall_with_a_and_the_smallest_best_is_taken(self, routed):
        grid, r = routed
        c = dd.gauge_check(grid, r["acc"], self.gauges(grid))
        assert c["scores"] == {"5.0": 2, "10.0": 2, "15.0": 1,
                               "25.0": 1, "40.0": 0}
        assert c["threshold_km2"] == 5.0
        assert c["score"] == 2
        assert c["n_gauges"] == 3
        wrong = next(g for g in c["gauges"] if g["id"] == "G-wrong-alt")
        assert wrong["topo_matches"] is False
        assert wrong["on_stream_at_chosen"] is True

    def test_pick_largest_takes_the_sparsest_network_with_the_best_score(
            self, routed):
        grid, r = routed
        c = dd.gauge_check(grid, r["acc"], self.gauges(grid), pick="largest")
        assert c["threshold_km2"] == 10.0
        assert c["score"] == 2

    def test_no_gauge_and_a_bad_pick_are_refused(self, routed):
        grid, r = routed
        with pytest.raises(ValueError, match="no in-basin gauge"):
            dd.gauge_check(grid, r["acc"], [])
        with pytest.raises(ValueError, match="pick='middle'"):
            dd.gauge_check(grid, r["acc"], self.gauges(grid), pick="middle")


class TestDatumAndBands:
    def test_the_std_elev_floor_binds_only_when_hand_is_shallower(self):
        assert dd.sink_datum("col_01", "hand", 3.0, 10.0, None) == 5.0
        assert dd.sink_datum("col_01", "hand", 12.0, 10.0, None) == 12.0
        assert dd.sink_datum("col_01", "conus2", 12.0, 10.0, 1.5) == 1.5

    def test_conus2_without_a_value_and_an_unknown_source_are_refused(self):
        with pytest.raises(ValueError, match="col_03"):
            dd.sink_datum("col_03", "conus2", 12.0, 10.0, None)
        with pytest.raises(ValueError, match="source='fan'"):
            dd.sink_datum("col_03", "fan", 12.0, 10.0, None)

    def test_band_arithmetic(self):
        assert dd.band_index(45.5) == 11          # lat11 spans 45 to 47 N
        assert dd.band_index(47.0) == 12
        assert dd.band_index(39.5) == 8
        p = Path("/x/surfdata_conus_1k_small_lat8_with_fdrain_and_fc.updated.nc")
        assert dd._swap_band(p, 9).name == \
            "surfdata_conus_1k_small_lat9_with_fdrain_and_fc.updated.nc"

    def test_out_inside_the_run_is_refused(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        with pytest.raises(ValueError, match="inside the run"):
            dd.outside_run(run / "datum.json", run)
        assert dd.outside_run(tmp_path / "datum.json", run) == \
            (tmp_path / "datum.json").resolve()
