#!/usr/bin/env python3
"""The walk's arithmetic — windows, the absolute clock, units, the slice.

    pytest tests/test_walk.py -v

Everything the walk exchanges rides on this module: a window one day short
shifts every later day of the year, a wrong unit multiplies every flux, a
leap year silently drops 31 December. Each is pinned here without a model.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import walk_lib as wl                                  # noqa: E402


class TestWindowEdges:
    def test_monthly_follows_the_calendar_and_covers_the_year(self):
        w = wl.window_edges(2010, months=1)
        assert len(w) == 12
        assert w[0] == {"i": 0, "d0": 0, "d1": 31,
                        "stop_n": 1, "stop_option": "nmonths"}
        assert w[1]["d1"] - w[1]["d0"] == 28          # February, non-leap
        assert w[-1]["d1"] == 365
        assert sum(x["d1"] - x["d0"] for x in w) == 365

    def test_weekly_gives_52_sevens_and_a_one_day_tail(self):
        w = wl.window_edges(2010, days=7)
        assert len(w) == 53
        assert all(x["stop_n"] == 7 for x in w[:52])
        assert w[-1] == {"i": 52, "d0": 364, "d1": 365,
                         "stop_n": 1, "stop_option": "ndays"}

    def test_a_leap_year_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="leap"):
            wl.window_edges(2012, days=7)

    def test_exactly_one_window_kind(self):
        with pytest.raises(ValueError, match="exactly one"):
            wl.window_edges(2010)
        with pytest.raises(ValueError, match="exactly one"):
            wl.window_edges(2010, months=1, days=7)

    def test_two_month_windows_are_refused(self):
        with pytest.raises(ValueError, match="months=1"):
            wl.window_edges(2010, months=2)


class TestFluxSeries:
    def test_absolute_clock_and_the_deck_unit(self):
        # day 31 (1 Feb) at t0=20 y; 2 mm/day = 0.73 m/y
        s = wl.flux_series([2.0, 4.0], d0=31, t0_y=20.0)
        assert s[0] == [round(20.0 + 31 / 365.0, 8), 0.73]
        assert s[1] == [round(20.0 + 32 / 365.0, 8), 1.46]

    def test_the_constant_is_the_servers_own(self):
        assert wl.MM_DAY_TO_M_Y == pytest.approx(0.001 * 365.0)


class TestWindowSlice:
    DATES = [f"2010-01-{d:02d}" for d in range(1, 32)]

    def test_cuts_exactly_the_windows_days(self):
        vals = list(range(31))
        got = wl.window_slice(self.DATES, vals, 2010, 7, 14)
        assert got == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]

    def test_a_missing_day_is_refused_not_padded(self):
        dates = self.DATES[:10]                        # record ends 10 Jan
        with pytest.raises(ValueError, match="missing"):
            wl.window_slice(dates, list(range(10)), 2010, 7, 14)

    def test_a_none_day_is_refused_by_date(self):
        vals = list(range(31))
        vals[8] = None
        with pytest.raises(ValueError, match="2010-01-09"):
            wl.window_slice(self.DATES, vals, 2010, 7, 14)

    def test_timestamped_dates_still_match(self):
        dates = [d + "T00:00:00" for d in self.DATES]
        got = wl.window_slice(dates, list(range(31)), 2010, 0, 2)
        assert got == [0.0, 1.0]


class TestRestartDates:
    def test_the_restart_name_carries_the_next_midnight(self):
        got = wl.restart_date("sliced.elm.r.2010-01-08-00000.nc")
        assert got == "2010-01-08"
        assert wl.restart_date("not_a_restart.nc") is None

    def test_window_end_matches_the_slice_proofs_naming(self):
        # 7 days from 1 Jan ended on ...2010-01-08-00000.nc (job 774960)
        assert wl.window_end_date(2010, 7) == "2010-01-08"
        assert wl.window_end_date(2010, 365) == "2011-01-01"
        assert wl.window_end_date(2010, 31) == "2010-02-01"


class TestBuiltCases:
    def test_unwraps_the_manifest_and_skips_failed_builds(self, tmp_path):
        """The manifest's rows live under 'cases' (ensemble_job.py's shape),
        and a failed build's case_dir is falsy — never handed on. Found by
        the adversarial review: the first draft iterated the top-level dict
        and crashed on its key strings."""
        import json
        d = tmp_path / "01_inputs"
        d.mkdir()
        (d / "built_cases.json").write_text(json.dumps({
            "ok": True, "n_ok": 1, "n_total": 2,
            "cases": [{"case_name": "col_01", "case_dir": "/x/col_01"},
                      {"case_name": "col_02", "case_dir": None,
                       "error": "build failed"}]}))
        assert wl.read_built_cases(tmp_path) == {"col_01": "/x/col_01"}


class TestState:
    def test_fresh_then_roundtrip(self, tmp_path):
        s = wl.load_state(tmp_path)
        assert s == {"spun": False, "windows_done": 0,
                     "checkpoints": {}, "log": []}
        s["windows_done"] = 3
        s["checkpoints"]["col_01"] = "/x/legA-restart.h5"
        wl.save_state(tmp_path, s)
        assert wl.load_state(tmp_path)["windows_done"] == 3
        assert not (tmp_path / "walk_state.json.tmp").exists()
