#!/usr/bin/env python3
"""Analyzer step 1b — streamflow against USGS gauges, and the shared geography.

Starting with the maps was deliberate. For SWE the map found the real problem —
three of five stations outside the basin — while the scatter and the hydrograph
both looked fine. Streamflow has the same trap and worse: on the 2019 Upper
Gunnison run 11 of 19 gauges sat in neighbouring basins.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step1_geo as geo                    # noqa: E402
from agents.analysis import step1_validate_streamflow as sf     # noqa: E402

SQUARE = [[[-108.0, 38.0], [-107.0, 38.0], [-107.0, 39.0],
           [-108.0, 39.0], [-108.0, 38.0]]]


class TestSplitByBasin:

    def test_outside_items_are_named(self):
        inside, outside = geo.split_by_basin(
            [{"id": "in", "lat": 38.5, "lon": -107.5},
             {"id": "out", "lat": 37.0, "lon": -107.5}], SQUARE)
        assert [i["id"] for i in inside] == ["in"]
        assert outside == ["out"]

    def test_an_item_with_no_coordinates_is_kept(self):
        """"Outside the basin" and "we do not know where this is" are
        different findings. The Gunnison run has one such gauge."""
        inside, outside = geo.split_by_basin([{"id": "nowhere"}], SQUARE)
        assert [i["id"] for i in inside] == ["nowhere"]
        assert outside == []

    def test_no_boundary_excludes_nothing(self):
        items = [{"id": "a", "lat": 1.0, "lon": 1.0}]
        inside, outside = geo.split_by_basin(items, [])
        assert inside == items and outside == []


class TestModelRunoff:

    @staticmethod
    def _row(qover, qdrai, dates=("2019-01-01", "2019-01-02")):
        def d(v):
            return {"daily": {"units": "mm/day", "dates": list(dates),
                              "values": list(v)}}
        return {"case_name": "col_01", "lat": 38.5, "lon": -107.5,
                "elevation_m": 3000.0,
                "variables": {"QOVER": d(qover), "QDRAI": d(qdrai)}}

    def test_it_sums_surface_and_subsurface(self):
        """A stream gauge sees both. QOVER alone understated col_19 by a
        factor of nine — 205 mm/yr against 1901 — in the direction that looks
        like a model dry bias."""
        s = sf.model_runoff(self._row([1.0, 2.0], [0.5, 0.5]))
        assert s["values"] == [1.5, 2.5]
        assert s["units"] == "mm/day"

    def test_a_date_in_only_one_variable_is_dropped(self):
        """A missing series is not a series of zeros; filling it would invent
        low-flow days."""
        row = self._row([1.0, 2.0], [0.5, 0.5])
        row["variables"]["QDRAI"]["daily"]["dates"] = ["2019-01-01", "2019-01-09"]
        s = sf.model_runoff(row)
        assert s["dates"] == ["2019-01-01"]

    def test_no_series_returns_empty(self):
        assert sf.model_runoff({"case_name": "c", "variables": {}}) == {}


class TestGaugeDaily:
    def test_the_dict_shape_is_normalised(self):
        """Reception stores gauge dailies as mm_day: {date: value} — a dict,
        unlike the model's columnar form and unlike SNOTEL's."""
        s = sf.gauge_daily({"mm_day": {"2019-01-02": 2.0, "2019-01-01": 1.0}})
        assert s["dates"] == ["2019-01-01", "2019-01-02"]
        assert s["values"] == [1.0, 2.0]

    def test_no_series_returns_empty(self):
        assert sf.gauge_daily({"id": "x"}) == {}


class _Ctx:
    def __init__(self, cols, gauges, boundary=SQUARE):
        self.data = {"observations": {"streamflow": {"stations": gauges}},
                     "boundary": boundary}
        self._c = cols

    @property
    def columns(self):
        return self._c


def _ctx(area_km2=100.0, gauge_lat=38.5):
    dates = ["2019-01-01", "2019-01-02"]
    col = TestModelRunoff._row([0.2, 0.2], [0.1, 0.1], dates)
    g = {"id": "USGS-1", "name": "G", "lat": gauge_lat, "lon": -107.5,
         "drainage_area_km2": area_km2,
         "mm_day": {d: 1.0 for d in dates}}
    return _Ctx([col], [g])


class TestCompare:

    def test_out_of_basin_gauges_are_excluded_and_named(self):
        r = sf.compare(_ctx(gauge_lat=37.0))
        assert r["gauges_excluded_outside_basin"] == ["USGS-1"]
        assert r["gauges"] == []
        c = next(c for c in r["caveats"] if c["id"] == "gauges_outside_basin")
        assert "bounding box" in c["statement"]

    def test_area_fraction_is_reported(self):
        r = sf.compare(_ctx(area_km2=100.0))
        assert r["basin_area_km2"] > 0
        assert 0 < r["gauges"][0]["area_fraction_of_basin"] < 1

    def test_a_gauge_bigger_than_the_basin_blocks(self):
        """It integrates water that was never simulated, so a bias against it
        is not attributable to the model."""
        r = sf.compare(_ctx(area_km2=99_999.0))
        c = next(c for c in r["caveats"]
                 if c["id"] == "gauge_exceeds_modelled_domain")
        assert c["severity"] == "blocking"
        assert r["gauges"][0]["area_fraction_of_basin"] > 1.0

    def test_the_routing_caveat_is_always_raised(self):
        """A gauge integrates a catchment; a column is 1 m2 with no routing."""
        c = next(c for c in sf.compare(_ctx())["caveats"]
                 if c["id"] == "unrouted_columns_vs_integrated_gauge")
        assert c["severity"] == "blocking"

    def test_means_are_taken_over_the_shared_window(self):
        r = sf.compare(_ctx())
        assert r["window"] == ("2019-01-01", "2019-01-02")
        assert r["gauges"][0]["mean_mm_day"] == 1.0
        assert r["columns"][0]["mean_mm_day"] == pytest.approx(0.3)


class TestMaps:
    def test_it_renders_without_a_basemap(self, tmp_path):
        pytest.importorskip("matplotlib")
        ctx = _ctx()
        out = tmp_path / "sf.png"
        sf.plot_maps(sf.compare(ctx), ctx, out, basemap=False)
        assert out.exists() and out.stat().st_size > 5000

    def test_the_scale_is_symlog_so_zeros_survive(self):
        """A linear scale was unreadable — one column's 5.2 mm/day flattened
        all seven gauges into one colour. But plain LogNorm cannot take a zero,
        and 15 of 19 columns produced EXACTLY zero, which is a different
        result from "very small"."""
        import inspect
        src = inspect.getsource(geo.plot_two_maps)
        assert "SymLogNorm" in src and "vmin=0.0" in src

    def test_gauges_are_sized_by_drainage_area(self):
        """They span 173 to 10,285 km2 on the Gunnison run. Drawn at one size
        a headwater gauge and a basin-integrating one look like equivalent
        evidence."""
        import inspect
        assert "drainage_area_km2" in inspect.getsource(sf.plot_maps)

    def test_the_label_says_runoff_not_discharge(self):
        """Discharge is a volume rate (m3/s); this is a depth rate over an
        area. Calling it discharge is what made the unit look wrong."""
        import inspect
        src = inspect.getsource(sf.plot_maps)
        assert 'label="mean runoff (mm/day)"' in src
