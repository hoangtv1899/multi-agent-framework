#!/usr/bin/env python3
"""Analyzer step 1c — water-table depth.

Three sources and usually only two available. USGS wells come in by BBOX, so
often none lie inside the watershed: on the 2019 Upper Gunnison run ALL TEN
were outside. Fan 2013 is collocated with every column by construction, which
is what makes a WTD figure possible at all when the wells fail.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step1_validate_wtd as wtd    # noqa: E402

SQUARE = [[[-108.0, 38.0], [-107.0, 38.0], [-107.0, 39.0],
           [-108.0, 39.0], [-108.0, 38.0]]]


class _Ctx:
    def __init__(self, cols, wells, boundary=SQUARE):
        self.data = {"observations": {"water_table": {"wells": wells}},
                     "boundary": boundary}
        self._c = cols

    @property
    def columns(self):
        return self._c


def _col(cid, lat, lon, fan, zwt_vals):
    return {"case_name": cid, "lat": lat, "lon": lon, "elevation_m": 3000.0,
            "fan_wtd_m": fan,
            "variables": {"ZWT": {"daily": {
                "units": "m",
                "dates": [f"2019-01-{i+1:02d}" for i in range(len(zwt_vals))],
                "values": list(zwt_vals)}}}}


def _ctx(wells=(), fan=25.0, zwt=(70.0, 70.0, 70.0)):
    return _Ctx([_col("col_01", 38.5, -107.5, fan, zwt)], list(wells))


class TestWellAvailability:

    def test_all_wells_outside_blocks(self):
        """Not 3 of 5 like SNOTEL, not 11 of 19 like the gauges — every one."""
        r = wtd.compare(_ctx(wells=[{"id": "W1", "lat": 37.0, "lon": -107.5,
                                     "wtd_m": 5.0}]))
        assert r["has_measured_wtd"] is False
        c = next(c for c in r["caveats"] if c["id"] == "wells_outside_basin")
        assert c["severity"] == "blocking"
        assert "Fan 2013 modelled prior" in c["statement"]

    def test_an_in_basin_well_downgrades_the_caveat(self):
        r = wtd.compare(_ctx(wells=[
            {"id": "in", "lat": 38.5, "lon": -107.5, "wtd_m": 5.0},
            {"id": "out", "lat": 37.0, "lon": -107.5, "wtd_m": 5.0}]))
        assert r["has_measured_wtd"] is True
        c = next(c for c in r["caveats"] if c["id"] == "wells_outside_basin")
        assert c["severity"] == "context"

    def test_no_wells_at_all_raises_no_exclusion_caveat(self):
        r = wtd.compare(_ctx())
        assert not any(c["id"] == "wells_outside_basin" for c in r["caveats"])


class TestStaticWaterTable:
    """The finding that explains the rest of the run, and the reason this is
    context rather than validation: ELM's active soil column is ~3.8 m, so a
    column whose water table sits 25 m down can never interact with it."""

    def test_a_static_column_is_counted(self):
        r = wtd.compare(_ctx(zwt=(70.0, 70.0, 70.0)))
        assert r["n_static_columns"] == 1
        assert r["model"][0]["annual_range_m"] == 0.0

    def test_a_moving_water_table_is_not(self):
        r = wtd.compare(_ctx(zwt=(3.0, 4.0, 5.5)))
        assert r["n_static_columns"] == 0
        assert r["model"][0]["annual_range_m"] == pytest.approx(2.5)

    def test_the_soil_column_caveat_blocks(self):
        c = next(c for c in wtd.compare(_ctx())["caveats"]
                 if c["id"] == "water_table_below_soil_column")
        assert c["severity"] == "blocking"
        assert "3.8 m" in c["statement"]
        assert "recharge" in c["applies_to"]


class TestMaps:
    """The panel COUNT follows the data. An empty third axis reads as
    "measured nothing" rather than "nothing to measure"."""

    def test_two_panels_without_wells(self, tmp_path):
        pytest.importorskip("matplotlib")
        from agents.analysis import step1_geo as geo
        seen = {}
        real = geo.plot_panels

        def spy(panels, *a, **k):
            seen["titles"] = [p[0] for p in panels if p[1]]
            return real(panels, *a, **k)

        geo.plot_panels = wtd.plot_panels = spy
        try:
            ctx = _ctx()
            wtd.plot_maps(wtd.compare(ctx), ctx, tmp_path / "w.png",
                          basemap=False)
        finally:
            geo.plot_panels = wtd.plot_panels = real
        assert seen["titles"] == ["Fan 2013", "ELM"]

    def test_three_panels_with_an_in_basin_well(self, tmp_path):
        pytest.importorskip("matplotlib")
        from agents.analysis import step1_geo as geo
        seen = {}
        real = geo.plot_panels

        def spy(panels, *a, **k):
            seen["titles"] = [p[0] for p in panels if p[1]]
            return real(panels, *a, **k)

        geo.plot_panels = wtd.plot_panels = spy
        try:
            ctx = _ctx(wells=[{"id": "in", "lat": 38.5, "lon": -107.5,
                               "wtd_m": 5.0, "n_obs": 11}])
            wtd.plot_maps(wtd.compare(ctx), ctx, tmp_path / "w3.png",
                          basemap=False)
        finally:
            geo.plot_panels = wtd.plot_panels = real
        assert seen["titles"] == ["USGS wells", "Fan 2013", "ELM"]

    def test_it_renders(self, tmp_path):
        pytest.importorskip("matplotlib")
        ctx = _ctx()
        out = tmp_path / "wtd.png"
        wtd.plot_maps(wtd.compare(ctx), ctx, out, basemap=False)
        assert out.exists() and out.stat().st_size > 5000

    def test_the_scale_is_log(self):
        """Depths span 0.0 to 251 m across the columns and the Fan grid reaches
        859 m in this window. Linear would collapse everything shallower than
        ~50 m into one colour, and shallow is where the behaviour is."""
        import inspect
        assert 'kw.setdefault("log", True)' in inspect.getsource(wtd.plot_maps)
