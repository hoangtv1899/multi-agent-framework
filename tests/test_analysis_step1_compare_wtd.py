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

from agents.analysis import step1_compare_wtd as wtd    # noqa: E402

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


class TestTimeseriesFigure:
    """Two panels: the depth distributions, then the depth series. Named for
    the panel that carries time, consistent with the other step-1 figures."""

    def test_it_renders(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "d.png"
        wtd.plot_timeseries(wtd.compare(_ctx()), out)
        assert out.exists() and out.stat().st_size > 5000

    def test_no_soil_column_reference_line(self):
        """It was drawn to say "below this the water table cannot reach the
        soil" — true, but it framed the deep values as a depth-scale mismatch
        when they are a diagnostic of NEGATIVE aquifer storage. A line
        implying they are water tables at an awkward depth reads as
        reassurance."""
        import inspect
        src = inspect.getsource(wtd.plot_timeseries)
        assert "axvline" not in src and "axhline" not in src

    def test_zeros_are_counted_not_plotted(self):
        """A log axis cannot take a zero, and a water table AT the surface is a
        distinct state rather than a small depth."""
        import inspect
        src = inspect.getsource(wtd.plot_timeseries)
        assert "at surface (0 m)" in src
        assert 'ax.set_xscale("log")' in src

    def test_the_wells_histogram_is_an_outline_drawn_last(self):
        """Two wells against nineteen columns lose every shared bin. At alpha
        0.6 the green bars vanished behind the models while still appearing in
        the legend — worse than omitting them."""
        import inspect
        src = inspect.getsource(wtd.plot_timeseries)
        assert 'histtype="step"' in src

    def test_both_panels_are_drawn(self):
        """It is called wtd_timeseries for consistency with the other figures,
        but it carries the distribution too — the histogram is the only place
        the Fan-vs-ELM depth compression is visible rather than inferred."""
        import inspect
        src = inspect.getsource(wtd.plot_timeseries)
        assert "subplots(1, 2" in src
        assert "ax.hist(" in src

    def test_depth_increases_downward(self):
        """A depth axis that runs upward reads as height."""
        import inspect
        assert "ax.invert_yaxis()" in inspect.getsource(wtd.plot_timeseries)

    def test_it_renders_with_no_model_series(self, tmp_path):
        pytest.importorskip("matplotlib")
        ctx = _Ctx([{"case_name": "c", "lat": 38.5, "lon": -107.5,
                     "fan_wtd_m": 20.0, "variables": {}}], [])
        out = tmp_path / "d2.png"
        wtd.plot_timeseries(wtd.compare(ctx), out)
        assert out.exists()


class TestWellSeriesNormalisation:
    """Wells carry series as [{date, wtd_m}] — a list of records, unlike the
    model's columnar {dates, values}. Three encodings for three observables is
    the recurring shape bug in this codebase, so the normalisation is pinned."""

    def test_records_become_sorted_pairs(self):
        pts = wtd._well_series({"series": [
            {"date": "2019-09-15", "wtd_m": 6.4},
            {"date": "2019-04-02", "wtd_m": 20.9}]})
        assert [p[1] for p in pts] == [20.9, 6.4]          # sorted by date
        assert str(pts[0][0]) == "2019-04-02"

    def test_unparseable_and_null_readings_are_dropped_not_zeroed(self):
        pts = wtd._well_series({"series": [
            {"date": "2019-04-02", "wtd_m": 20.9},
            {"date": "not-a-date", "wtd_m": 5.0},
            {"date": "2019-05-02", "wtd_m": None},
            "junk"]})
        assert [p[1] for p in pts] == [20.9]

    def test_no_series_is_empty_not_an_error(self):
        assert wtd._well_series({}) == []


class TestTimeseriesUsesWells:
    """The wells are the only MEASUREMENT of this quantity. A figure that drew
    Fan and ELM alone when wells existed would be showing two models and
    hiding the one measurement there is."""

    def _result(self, wells):
        return {"fan": [{"id": "c1", "wtd_m": 25.0}, {"id": "c2", "wtd_m": 4.4}],
                "model": [{"id": "c1", "wtd_m": 70.0,
                           "series": {"units": "m",
                                      "dates": ["2019-01-01", "2019-06-01"],
                                      "values": [70.0, 70.1]}},
                          {"id": "c2", "wtd_m": 2.1,
                           "series": {"units": "m",
                                      "dates": ["2019-01-01", "2019-06-01"],
                                      "values": [2.0, 2.2]}}],
                "wells": wells}

    def test_wells_are_drawn_in_both_panels(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        r = self._result([{"id": "W1", "lat": 38.4, "lon": -107.4,
                           "wtd_m": 6.2, "n_obs": 11,
                           "series": [{"date": "2019-05-15", "wtd_m": 6.0},
                                      {"date": "2019-09-15", "wtd_m": 6.4}]}])
        out = tmp_path / "d.png"
        assert Path(wtd.plot_timeseries(r, out)).exists()
        assert out.stat().st_size > 5000

    def test_degrades_to_two_fields_with_no_wells(self, tmp_path):
        """The 2019 Upper Gunnison result, not a missing feature."""
        import matplotlib
        matplotlib.use("Agg")
        out = tmp_path / "d0.png"
        assert Path(wtd.plot_timeseries(self._result([]), out)).exists()

    def test_series_panel_goes_log_only_when_scales_diverge(self):
        """Wells at 5 m beside ELM at 70 m compress to a flat line on a linear
        axis; a narrow range on a log axis exaggerates noise instead."""
        def scale(model_vals, wells=()):
            r = self._result(list(wells))
            r["model"] = [{"id": "c", "wtd_m": model_vals[0],
                           "series": {"units": "m",
                                      "dates": ["2019-01-01", "2019-06-01"],
                                      "values": list(model_vals)}}]
            return wtd._series_yscale(r)

        assert scale((2.0, 2.2)) == "linear"        # ratio 1.1
        assert scale((2.0, 200.0)) == "log"         # ratio 100

    def test_a_shallow_well_against_a_deep_column_forces_log(self):
        """The case the rule exists for: the model's own span is narrow, and
        only the well makes the panel span scales."""
        deep = [{"id": "c", "wtd_m": 70.0,
                 "series": {"units": "m",
                            "dates": ["2019-01-01", "2019-06-01"],
                            "values": [70.0, 70.1]}}]
        r = {"model": deep, "wells": []}
        assert wtd._series_yscale(r) == "linear"
        r["wells"] = [{"id": "W", "series": [{"date": "2019-05-15",
                                              "wtd_m": 1.5}]}]
        assert wtd._series_yscale(r) == "log"

    def test_zero_and_negative_depths_do_not_break_the_scale_choice(self):
        """A water table AT the surface is 0 m and cannot sit on a log axis."""
        r = {"model": [{"id": "c", "wtd_m": 0.0,
                        "series": {"units": "m", "dates": ["2019-01-01"],
                                   "values": [0.0]}}], "wells": []}
        assert wtd._series_yscale(r) == "linear"

    def test_no_data_at_all_is_linear(self):
        assert wtd._series_yscale({}) == "linear"
