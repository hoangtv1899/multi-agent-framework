#!/usr/bin/env python3
"""Analyzer step 1a — SWE against SNOTEL.

The version this replaces paired each station to its nearest column by
HORIZONTAL distance. On the 2019 Upper Gunnison run that produced elevation
offsets up to 846 m and collapsed five stations onto two columns — and SWE is
governed by elevation. Pairing on elevation instead gives a worst offset of
162 m with every station finding a partner.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step1_validate_swe as swe   # noqa: E402


def _series(vals, start="2019-01-01"):
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    return ([str(d0 + dt.timedelta(days=i)) for i in range(len(vals))], vals)


class TestMetrics:

    def test_peak_mean_and_peak_date(self):
        dates, vals = _series([0, 10, 200, 150, 0])
        m = swe.swe_metrics(dates, vals)
        assert m["peak_swe_mm"] == 200
        assert m["peak_date"] == "2019-01-03"
        assert m["mean_swe_mm"] == 72.0

    def test_no_snowpack_leaves_timing_undefined_not_zero(self):
        """A bare desert column must not sit on the same axis as a
        late-melting ridge just because both round to day 0."""
        dates, vals = _series([0, 5, 9, 3, 0])       # never above 25 mm
        m = swe.swe_metrics(dates, vals)
        assert m["has_snowpack"] is False
        assert m["first_snow_date"] is None and m["melt_out_date"] is None
        assert "no persistent snowpack" in m["undefined_reason"]

    def test_snow_present_on_day_one_is_not_an_onset(self):
        """The model record starts mid-winter after a warm start. An onset
        date taken from 1 January reports when the RUN began."""
        dates, vals = _series([200, 180, 40, 0])
        m = swe.swe_metrics(dates, vals)
        assert m["first_snow_date"] is None
        assert "before it began" in m["first_snow_censored"]

    def test_snow_still_present_at_the_end_is_censored(self):
        dates, vals = _series([0, 100, 200, 300])
        m = swe.swe_metrics(dates, vals)
        assert m["melt_out_date"] is None
        assert "after the record" in m["melt_out_censored"]

    def test_melt_out_is_the_day_after_the_last_crossing(self):
        dates, vals = _series([0, 100, 100, 0, 0])
        assert swe.swe_metrics(dates, vals)["melt_out_date"] == "2019-01-04"


def _pt(name, elev, peak=100.0):
    return {"available": True, "entity": name, "elevation_m": elev,
            "peak_swe_mm": peak}


class TestPairing:

    def test_pairs_on_elevation_not_distance(self):
        model = [_pt("col_A", 3559), _pt("col_B", 2965)]
        obs = [_pt("Slumgullion", 3524), _pt("Idarado", 2981)]
        pairs, un = swe.pair_by_elevation(model, obs)
        got = {p["station"]: p["column"] for p in pairs}
        assert got == {"Slumgullion": "col_A", "Idarado": "col_B"}
        assert not un

    def test_it_is_a_bijection_and_the_closer_station_wins(self):
        """Two points at the same y on a 1:1 plot cannot carry the comparison
        they appear to make."""
        model = [_pt("col_12", 3224)]
        obs = [_pt("Wager Gulch", 3386), _pt("Red Mountain Pass", 3371)]
        pairs, un = swe.pair_by_elevation(model, obs)
        assert len(pairs) == 1
        assert pairs[0]["station"] == "Red Mountain Pass"   # 147 m vs 162 m
        assert un[0]["entity"] == "Wager Gulch"
        assert "already paired" in un[0]["unpaired_reason"]

    def test_beyond_the_cap_a_station_is_unpaired_with_a_reason(self):
        model = [_pt("col_A", 2000)]
        obs = [_pt("High Station", 3500)]
        pairs, un = swe.pair_by_elevation(model, obs, max_delta_m=200)
        assert not pairs
        assert "beyond the" in un[0]["unpaired_reason"]

    def test_the_offset_is_recorded_and_signed(self):
        pairs, _ = swe.pair_by_elevation([_pt("col_A", 3559)],
                                         [_pt("Slumgullion", 3524)])
        assert pairs[0]["delta_elevation_m"] == 35.0


class _Ctx:
    def __init__(self, columns, stations):
        self.data = {"observations": {"swe": {"stations": stations}}}
        self._c = columns

    @property
    def columns(self):
        return self._c


class TestCompare:

    @staticmethod
    def _ctx():
        md, mv = _series([200] * 60 + [0] * 40, "2019-01-01")
        od, ov = _series([0] * 92 + [300] * 60 + [0] * 40, "2018-10-01")
        return _Ctx(
            [{"case_name": "col_A", "elevation_m": 3000,
              "variables": {"H2OSNO": {"daily": {"units": "mm", "dates": md,
                                                 "values": mv}}}}],
            [{"name": "Stn", "elevation_m": 3010,
              "daily": {"units": "mm", "dates": od, "values": ov}}])

    def test_metrics_are_taken_over_the_shared_window_only(self):
        """SNOTEL reports by water year, the model by calendar year.
        Comparing a water-year peak date to a calendar-year one silently
        compares two different winters."""
        r = swe.compare(self._ctx())
        lo, hi = r["window"]
        assert lo == "2019-01-01"
        assert any(c["id"] == "swe_window_overlap" for c in r["caveats"])

    def test_the_siting_caveat_is_always_raised(self):
        """SNOTEL sites are chosen for snow retention; a column is a 12 km
        average. Observed above modelled is expected from siting alone."""
        r = swe.compare(self._ctx())
        c = next(c for c in r["caveats"] if c["id"] == "snotel_siting_bias")
        assert c["severity"] == "qualify"

    def test_it_returns_pairs(self):
        r = swe.compare(self._ctx())
        assert len(r["pairs"]) == 1
        assert r["pairs"][0]["column"] == "col_A"

    def test_the_figure_renders(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "swe.png"
        swe.plot(swe.compare(self._ctx()), out)
        assert out.exists() and out.stat().st_size > 5000

    def test_a_run_with_no_stations_still_renders(self, tmp_path):
        """No SNOTEL in the bbox is a legitimate outcome, not a crash."""
        pytest.importorskip("matplotlib")
        ctx = self._ctx(); ctx.data["observations"]["swe"]["stations"] = []
        r = swe.compare(ctx)
        assert r["pairs"] == []
        out = tmp_path / "empty.png"
        swe.plot(r, out)
        assert out.exists()


class TestWaterYearDay:
    def test_october_is_early_and_june_is_late(self):
        """Calendar day-of-year puts a late-December peak and an early-January
        peak at opposite ends of the axis while being ten days apart."""
        assert swe._doy("2018-10-01") == 1
        assert swe._doy("2019-06-01") > swe._doy("2019-01-01")
        assert swe._doy("2018-12-28") < swe._doy("2019-01-05")
