#!/usr/bin/env python3
"""The walk's arithmetic — windows, the absolute clock, units, the slice.

    pytest tests/test_walk.py -v

Everything the walk exchanges rides on this module: a window one day short
shifts every later day of the year, a wrong unit multiplies every flux, a
leap year silently drops 31 December. Each is pinned here without a model.
"""
import json
import sys
import types
from datetime import date, timedelta
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


# ── the forward flux and the lateral sink ───────────────────────────────
# docs/coupling/lateral_sink_design.md: the numbers below are the design's.


class TestNegativePolicy:
    def test_clip_zeroes_negative_days_and_counts_what_it_removed(self):
        vals, clipped = wl.apply_negative_policy([1.0, -0.5, 2.0, -1.5], "clip")
        assert vals == [1.0, 0.0, 2.0, 0.0]
        assert clipped == pytest.approx(2.0)          # mm over the window

    def test_pass_hands_the_series_on_unchanged(self):
        vals, clipped = wl.apply_negative_policy([1.0, -0.5], "pass")
        assert vals == [1.0, -0.5] and clipped == 0.0

    def test_an_unknown_policy_is_refused_by_name(self):
        with pytest.raises(ValueError, match="drop"):
            wl.apply_negative_policy([1.0], "drop")

    def test_a_none_day_is_refused_by_index(self):
        with pytest.raises(ValueError, match="day 1"):
            wl.apply_negative_policy([1.0, None], "clip")


class TestEmptyForwardSeries:
    def test_an_absent_or_fill_variable_is_an_empty_series(self):
        data = {"dates": ["2010-01-01"],
                "variables": {"QDRAI": {"values": [1.0]}}}
        assert wl.forward_values(data, "QDRAI") == [1.0]
        assert wl.forward_values(data, "QCHARGE") == []
        assert wl.forward_values({}, "QCHARGE") == []

    def test_the_reason_names_the_variable_and_the_finding(self):
        r = wl.no_forward_reason("QCHARGE", {"absent_variables": ["QCHARGE"]})
        assert r.startswith("ELM wrote no QCHARGE for this column")
        assert "history" in r
        r = wl.no_forward_reason("QCHARGE", {"empty_variables": ["QCHARGE"]})
        assert "fill" in r
        assert wl.no_forward_reason("QCHARGE") == \
            "ELM wrote no QCHARGE for this column"


class TestSinkConductance:
    def test_reproduces_the_designs_magnitudes(self):
        # design section 3, to three significant figures
        assert wl.sink_conductance_m(0.10, 30.0, 1.0) == \
            pytest.approx(3.95e-15, rel=5e-3)                # Naches
        assert wl.sink_conductance_m(0.05, 30.0, 1.0) == \
            pytest.approx(1.97e-15, rel=5e-3)
        assert wl.sink_conductance_m(0.20, 30.0, 1.0) == \
            pytest.approx(7.90e-15, rel=5e-3)
        assert wl.sink_conductance_m(0.10, 35.0, 1.0) == \
            pytest.approx(3.38e-15, rel=5e-3)                # Brandywine

    def test_one_metre_of_head_leaks_sy_1000_over_tau(self):
        C = wl.sink_conductance_m(0.10, 30.0, 1.0)
        assert wl.sink_outflow_mm_day(C, 1.0, 1.0) == \
            pytest.approx(3.33, rel=5e-3)
        C = wl.sink_conductance_m(0.10, 35.0, 1.0)
        assert wl.sink_outflow_mm_day(C, 1.0, 1.0) == \
            pytest.approx(2.86, rel=5e-3)

    def test_bad_inputs_are_refused_by_name(self):
        with pytest.raises(ValueError, match="sink_tau_days=0"):
            wl.sink_conductance_m(0.1, 0, 1.0)
        with pytest.raises(ValueError, match="sink_sy=1.5"):
            wl.sink_conductance_m(1.5, 30, 1.0)
        with pytest.raises(ValueError, match="sink_band_m=-1"):
            wl.sink_conductance_m(0.1, 30, -1)


class TestDatumInDomain:
    def test_a_datum_inside_the_column_is_applied_as_is(self):
        assert wl.datum_in_domain(4.0, 7.0, 1.0) == (4.0, False, False)

    def test_a_datum_below_the_column_is_clamped_and_said(self):
        # a 7 m column with a 1 m band drains at 6 m at the deepest
        assert wl.datum_in_domain(9.0, 7.0, 1.0) == (6.0, True, False)

    def test_a_datum_shallower_than_the_band_is_raised_and_said(self):
        # CONUS2 puts the steady water table at the surface in valley cells
        # (job 778434: nine Naches columns at 0 to 0.05 m). The band keeps
        # its thickness, so the datum is held at B; exactly B is not a clamp.
        assert wl.datum_in_domain(0.0, 7.0, 1.0) == (1.0, False, True)
        assert wl.datum_in_domain(0.05, 7.0, 1.0) == (1.0, False, True)
        assert wl.datum_in_domain(1.0, 7.0, 1.0) == (1.0, False, False)

    def test_a_negative_datum_is_refused(self):
        with pytest.raises(ValueError, match="-1"):
            wl.datum_in_domain(-1.0, 7.0, 1.0)


class TestLayerAtDepth:
    PROFILE = {"layers": [
        {"depth_top_m": 0.0, "depth_bot_m": 2.0, "porosity": 0.4},
        {"depth_top_m": 2.0, "depth_bot_m": 12.0, "porosity": 0.33}]}

    def test_top_inclusive_bottom_exclusive(self):
        assert wl.layer_at_depth(self.PROFILE, 2.0)["porosity"] == 0.33
        assert wl.layer_at_depth(self.PROFILE, 1.999)["porosity"] == 0.4
        assert wl.layer_at_depth(self.PROFILE, 12.0) is None
        assert wl.layer_at_depth(None, 1.0) is None


class TestImpliedTau:
    def test_the_dupuit_timescale_at_the_columns_head(self):
        # Sy 0.2, K 0.02 m/h = 0.48 m/day, L 1200 m, h 2 m
        assert wl.implied_tau_days(0.2, 0.02, 1200.0, 2.0) == \
            pytest.approx(0.2 * 1200.0 ** 2 / (0.48 * 2.0))

    def test_none_when_nothing_drains_or_an_input_is_missing(self):
        assert wl.implied_tau_days(0.2, 0.02, 1200.0, -2.0) is None
        assert wl.implied_tau_days(0.2, None, 1200.0, 2.0) is None


class TestMassBalanceReader:
    HEADER = ('"Time [y]","dt_flow [y]","Global Water Mass [kg]",'
              '"top_recharge Water Mass [kg]","top_recharge Water Mass [kg/y]",'
              '"lateral_sink Water Mass [kg]","lateral_sink Water Mass [kg/y]"')

    def _write(self, tmp_path, rows):
        p = tmp_path / "col_01-mas.dat"
        p.write_text(" " + self.HEADER + "\n" + "\n".join(rows) + "\n")
        return p

    def test_sign_flipped_once_and_kg_read_as_mm(self, tmp_path):
        p = self._write(tmp_path, [
            "  20.001  0.001  700.0  1.0  365.25  -2.5  -365.25",
            "  20.002  0.001  700.0  2.0  365.25  -12.5  -730.5"])
        b = wl.lateral_outflow_block(p)
        assert b["cumulative_kg"] == [-2.5, -12.5]    # as written: out < 0
        assert b["window_total_mm"] == 12.5            # leaving > 0; 1 kg = 1 mm
        assert b["outflow_mm_day"] == pytest.approx([1.0, 2.0])  # kg/y / 365.25
        assert b["times_y"] == [20.001, 20.002]
        assert "positive = leaving" in b["units"]["window_total_mm"]

    def test_absent_file_or_coupler_is_none_not_zero(self, tmp_path):
        assert wl.lateral_outflow_block(tmp_path / "nope-mas.dat") is None
        p = tmp_path / "col_01-mas.dat"
        p.write_text(' "Time [y]","top_recharge Water Mass [kg]"\n  1.0  2.0\n')
        assert wl.lateral_outflow_block(p) is None

    def test_a_short_row_is_refused_by_line(self, tmp_path):
        p = self._write(tmp_path, ["  20.001  0.001  700.0"])
        with pytest.raises(ValueError, match="line 2"):
            wl.read_mass_balance(p)


class TestSinkSetup:
    """walk_setup's sink dials and per-column contract, without a clone."""

    PROFILE = TestLayerAtDepth.PROFILE
    DATUM = {"col_01": {"sink_datum_source": "hand", "sink_datum_m": 4.0,
                        "hand_m": 4.0, "hand_path_km": 1.2,
                        "hand_stream_area_km2": 3.0, "hand_threshold_km2": 1.0,
                        "std_elev_m": 6.0, "conus2_wtd_m": 5.1},
             "col_02": {"sink_datum_source": "hand", "sink_datum_m": 9.0,
                        "hand_m": 9.0, "hand_path_km": 2.0,
                        "hand_stream_area_km2": 3.0, "hand_threshold_km2": 1.0,
                        "std_elev_m": 2.0, "conus2_wtd_m": 7.0}}

    def _args(self, **kw):
        base = dict(sink_datum="none", sink_tau_days=None, sink_band_m=1.0,
                    sink_sy=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def _columns(self, monkeypatch, **dials):
        import walk_setup
        for L in self.PROFILE["layers"]:
            L.setdefault("permeability_z", 0.02)
        # the module's real shape: a report with one row per column under
        # "columns" (tools/drainage_datum.py compute), read off the ELM run
        seen = {}
        monkeypatch.setattr(walk_setup, "drainage_datum", types.SimpleNamespace(
            compute=lambda run_dir, source: seen.update(run_dir=run_dir) or {
                "basin": "Fixtureville", "hand_threshold_km2": 1.0,
                "gauge_check": {"threshold_km2": 1.0, "score": 0.5},
                "sink_datum_rule": "max(hand_m, 0.5 * std_elev_m)",
                "columns": [{"id": cid, **row}
                            for cid, row in self.DATUM.items()]}))
        sd = types.SimpleNamespace(join=lambda site_dir, cols: {"columns": [
            {"id": c["id"], "subsurface_profile": self.PROFILE} for c in cols]})
        sink = walk_setup._sink_dials(self._args(sink_datum="hand",
                                                 sink_tau_days=30.0, **dials))
        pf_cols = [{"id": "col_01", "lat": 1, "lon": 2, "water_table_m": 2.0},
                   {"id": "col_02", "lat": 1, "lon": 2, "water_table_m": 8.0}]
        rows = {"col_01": {"depth_m": 7.0}, "col_02": {"depth_m": 7.0}}
        got = walk_setup._sink_columns(sink, "/elm", "/pf", pf_cols, rows, sd)
        assert seen["run_dir"] == "/elm"          # the ELM run, never the pf
        assert sink["datum_report"]["hand_threshold_km2"] == 1.0
        return got

    def test_a_datum_without_tau_is_refused(self):
        import walk_setup
        with pytest.raises(SystemExit, match="--sink-tau-days"):
            walk_setup._sink_dials(self._args(sink_datum="hand"))

    def test_a_tau_without_a_datum_is_refused(self):
        import walk_setup
        with pytest.raises(SystemExit, match="--sink-datum none"):
            walk_setup._sink_dials(self._args(sink_tau_days=30.0))

    def test_no_sink_is_the_dials_as_given(self):
        import walk_setup
        assert walk_setup._sink_dials(self._args()) == {
            "datum": "none", "tau_days": None, "band_m": 1.0, "sy": None}

    def test_every_contract_key_is_written_per_column(self, monkeypatch):
        got = self._columns(monkeypatch)
        for cid in ("col_01", "col_02"):
            assert set(wl.SINK_CONTRACT_KEYS) <= set(got[cid]), cid
        c1 = got["col_01"]
        assert (c1["sink_datum_applied_m"],
                c1["sink_datum_clipped_to_domain"],
                c1["sink_datum_raised_to_band"]) == (4.0, False, False)
        # porosity 0.33 at 4 m, capped at ELM's own 0.2
        assert c1["sink_sy"] == 0.2 and "min(0.2" in c1["sink_sy_source"]
        assert c1["sink_conductance_m"] == pytest.approx(7.90e-15, rel=5e-3)
        assert (c1["sink_tau_days"], c1["sink_tau_source"]) == \
            (30.0, "--sink-tau-days")
        assert c1["sink_implied_tau_days"] > 0
        assert (c1["hand_path_km"], c1["conus2_wtd_m"]) == (1.2, 5.1)
        c2 = got["col_02"]
        # a 9 m datum in a 7 m column: applied at 6 m, said
        assert (c2["sink_datum_applied_m"],
                c2["sink_datum_clipped_to_domain"],
                c2["sink_datum_raised_to_band"]) == (6.0, True, False)
        assert c2["sink_datum_m"] == 9.0
        # its water table (8 m) stands below the datum: nothing drains
        assert c2["sink_implied_tau_days"] is None

    def test_the_sy_dial_overrides_and_is_said(self, monkeypatch):
        c1 = self._columns(monkeypatch, sink_sy=0.10)["col_01"]
        assert (c1["sink_sy"], c1["sink_sy_source"]) == (0.10, "--sink-sy")
        assert c1["sink_conductance_m"] == pytest.approx(3.95e-15, rel=5e-3)


class TestWalkJobWindow:
    """One window of walk_job.main with every model faked: the forward
    policy on the row, the sink kwargs on the deck call, the outflow read
    off the window's -mas.dat, and an empty series as a recorded death."""

    MAS = (' "Time [y]","dt_flow [y]","Global Water Mass [kg]",'
           '"lateral_sink Water Mass [kg]","lateral_sink Water Mass [kg/y]"\n'
           '  20.5  0.001  700.0  -2.5  -365.25\n'
           '  21.0  0.001  700.0  -12.5  -730.5\n')

    def _walk(self, tmp_path):
        wd = tmp_path / "walk"
        (wd / "elm").mkdir(parents=True)
        sink = {"sink_datum_applied_m": 4.0, "sink_band_m": 1.0,
                "sink_conductance_m": 3.95e-15}
        cols = []
        for cid in ("col_01", "col_02"):
            (wd / "elm" / cid).mkdir()
            cols.append({
                "id": cid, "elm_case_dir": str(wd / "elm" / cid),
                "fsurdat": "surf.nc", "water_table_m": 3.0, "depth_m": 7.0,
                "n_cells": 14, "lat": 40.0, "lon": -75.0,
                "spin_input_file": str(wd / "pf_spin" / cid / f"{cid}.in"),
                "spin_checkpoint": str(wd / "pf_spin" / cid / "r.h5"),
                **sink})
        (wd / "walk.json").write_text(json.dumps({
            "year": 2010, "window": {"days": 365}, "n_windows": 1,
            "t0_y": 20.0, "pf_bottom": "none", "forward_var": "QCHARGE",
            "negative_forward": "clip", "site_dir": str(tmp_path),
            "sink": {"datum": "hand", "tau_days": 30.0, "band_m": 1.0,
                     "sy": None},
            "columns": cols}))
        wl.save_state(wd, {"spun": True, "windows_done": 0, "log": [],
                           "checkpoints": {c["id"]: "/x/spin-restart.h5"
                                           for c in cols}})
        return wd

    def test_one_window_end_to_end(self, tmp_path, monkeypatch):
        import walk_job
        wd = self._walk(tmp_path)
        calls = []
        dates = [str(date(2010, 1, 1) + timedelta(days=j)) for j in range(365)]
        qcharge = [1.0] * 363 + [-0.5, -1.5]

        def build_deck(**kw):
            calls.append(kw)
            case = Path(kw["out_dir"]) / kw["column"]["id"]
            case.mkdir(parents=True, exist_ok=True)
            if "lateral_sink_conductance" in kw:
                (case / f"{kw['column']['id']}-mas.dat").write_text(self.MAS)
            return {"case_dir": str(case), "input_file": str(case / "d.in"),
                    "restart_file_expected": str(case / "restart.h5")}

        def run_sim(input_file, executable, timeout):
            Path(input_file).with_name("restart.h5").write_text("ck")
            return {"exit_codes": [0]}

        class PF:
            @staticmethod
            def extract_column_series(cases, out_file):
                Path(out_file).write_text(json.dumps({"columns": {
                    cases[0]["id"]: {"depth_m": [0.25, 0.75],
                                     "saturation": [[0.5, 1.0]]}}}))

        site = types.SimpleNamespace(join=lambda d, cols: {"columns": [
            {"id": c["id"], "subsurface_profile": {}} for c in cols]})
        monkeypatch.setattr(walk_job, "_pf", lambda: (
            PF, build_deck,
            lambda prof, depth, max_cell_m: [(0.5, {})] * int(depth / 0.5),
            run_sim, site))
        monkeypatch.setattr(walk_job, "_solved_fn", lambda: (lambda b: 3.5))
        monkeypatch.setenv("PFLOTRAN_EXECUTABLE", "/x/pflotran")

        def extract_column(case_dir, variables, spinup_days):
            if variables == ["QCHARGE"] and Path(case_dir).name == "col_02":
                return ({"dates": [], "variables": {}},
                        {"absent_variables": ["QCHARGE"]})
            return ({"dates": dates, "variables": {
                v: {"units": "mm/day", "values": list(qcharge)}
                for v in variables}}, {})

        restart = "x.elm.r.2011-01-01-00000.nc"

        def latest_restart(case_dir):
            f = Path(case_dir) / "run" / restart
            if not f.is_file():
                raise ValueError("no rpointer.lnd")
            return f

        def run_built_case(case_dir):
            f = Path(case_dir) / "run" / restart
            f.parent.mkdir(exist_ok=True)
            f.write_text("r")
            return True

        monkeypatch.setattr(walk_job.ew, "latest_restart", latest_restart)
        monkeypatch.setattr(walk_job.ew, "run_built_case", run_built_case)
        monkeypatch.setattr(walk_job.ew, "configure_continuation",
                            lambda *a, **k: None)
        monkeypatch.setattr(walk_job.elm_extract, "extract_column",
                            extract_column)
        monkeypatch.setattr(walk_job.swt, "apply",
                            lambda f, wt, quiet: {"written_m": wt,
                                                  "regime": "aquifer"})
        monkeypatch.setattr(walk_job.swt, "apply_profile",
                            lambda *a, **k: {"layers_written": 10})

        assert walk_job.main(str(wd)) == 0

        # the empty series is a death on the record, before the slice
        dead = wl.load_state(wd)["dead"]["col_02"]
        assert dead["reason"] == ("ELM wrote no QCHARGE for this column "
                                  "(not in the history output)")
        assert (dead["window"], dead["window_end_day"]) == (0, 365)
        # ...and never reached the deck builder; the live column's deck
        # carried the three sink kwargs by their contract names
        assert [k["column"]["id"] for k in calls] == ["col_01"]
        kw = calls[0]
        assert (kw["lateral_sink_depth_m"], kw["lateral_sink_band_m"],
                kw["lateral_sink_conductance"]) == (4.0, 1.0, 3.95e-15)
        assert kw["bottom"] == "none"
        # the row: the policy's totals and the outflow off the -mas.dat
        (row,) = [json.loads(ln) for ln in
                  (wd / "walk_log.jsonl").read_text().splitlines()]
        r1 = row["columns"]["col_01"]
        assert r1["forward_clipped_mm"] == pytest.approx(2.0)
        assert r1["forward_mm_window"] == pytest.approx(363.0)
        assert r1["lateral_outflow_window_mm"] == 12.5
        assert row["columns"]["col_02"] == {"failed": dead["reason"]}
        # the summary carries the dials and the outflow trajectory
        summ = json.loads((wd / "walk_summary.json").read_text())
        assert (summ["forward_var"], summ["negative_forward"]) == \
            ("QCHARGE", "clip")
        assert summ["sink"]["datum"] == "hand"
        traj = summ["solved_trajectories"]["col_01"]
        assert traj["lateral_outflow_window_mm"] == [12.5]
        assert "QCHARGE" in summ["elm_daily"]["col_01"]
        assert "col_02" in summ["columns_left_the_walk"]

    def test_a_sink_free_column_gets_the_call_it_always_got(self):
        import walk_job
        assert walk_job._sink_kwargs({"id": "col_01", "water_table_m": 3.0}) == {}
        assert walk_job._sink_kwargs({"sink_conductance_m": None}) == {}



class TestReturnLegAndBalance:
    """The return-leg dial and ELM's own water balance (2026-09-13)."""

    def test_a_conserving_series_closes_and_has_no_jumps(self):
        n = 60
        P = [2.0] * n
        series = {"RAIN": [1.5] * n, "SNOW": [0.5] * n, "QOVER": [0.2] * n,
                  "QDRAI": [0.3] * n, "QCHARGE": [0.4] * n,
                  "QVEGE": [0.1] * n, "QVEGT": [0.2] * n, "QSOIL": [0.1] * n,
                  "TWS": [1000.0 + 1.1 * i for i in range(n)]}
        # P - ET - QOVER - QDRAI = 2.0 - 0.4 - 0.2 - 0.3 = 1.1 mm/day = dTWS/day
        b = wl.elm_balance(series, [(0, 30), (30, 60)])
        for w in b["windows"]:
            assert abs(w["residual_mm"] - 1.1) < 1e-6   # one day of flux: the
            # window's dTWS spans first day to last day, 29 steps for 30 fluxes
        assert abs(b["year"]["residual_mm"] - 1.1) < 0.2
        assert abs(b["stamp_jumps_mm"][0] - 1.1) < 1e-6  # the boundary step
        assert b["year"]["P_mm"] == 120.0

    def test_water_added_at_a_boundary_shows_as_a_jump_and_a_residual(self):
        n = 60
        tws = [1000.0] * 30 + [1250.0] * 30          # 250 mm stamped at day 30
        series = {"RAIN": [0.0] * n, "SNOW": [0.0] * n, "QOVER": [0.0] * n,
                  "QDRAI": [0.0] * n, "QCHARGE": [0.0] * n,
                  "QVEGE": [0.0] * n, "QVEGT": [0.0] * n, "QSOIL": [0.0] * n,
                  "TWS": tws}
        b = wl.elm_balance(series, [(0, 30), (30, 60)])
        assert b["stamp_jumps_mm"] == [250.0]
        assert b["year"]["residual_mm"] == -250.0     # water from nowhere
        assert b["windows"][0]["residual_mm"] == 0.0

    def test_missing_series_is_none_not_zero(self):
        assert wl.elm_balance({"RAIN": [1.0]}, [(0, 1)]) is None

    def test_return_dial_is_recorded_and_defaults_to_the_old_behaviour(self):
        assert wl.RETURN_LEGS == ("wt+profile", "wt")
        src = (ROOT / "tools" / "walk_setup.py").read_text()
        assert 'dest="return_leg"' in src and 'default="wt+profile"' in src
        job = (ROOT / "tools" / "walk_job.py").read_text()
        assert 'walk.get("return_leg") or "wt+profile"' in job
        assert 'if return_leg == "wt":' in job
