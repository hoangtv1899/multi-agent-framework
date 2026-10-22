"""The gridded history reader against small files in ELM's own layout.

    pytest tests/test_gridded_reader.py -v

Every check compares the reader with a formula (elm_gridded_fixtures.value),
never with the reader itself. The values encode their own date, so a record
read under the wrong label fails by a whole day, month or step, which is
exactly the error the end-of-interval `time` stamp would cause.
"""
import shutil

import numpy as np
import pytest

import elm_gridded_fixtures as fx
import gridded as G

Y0 = 2015
YEARS = [2015, 2016]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    fx.write_daily(d, YEARS)                               # h1: 2015, 2016, 2017
    fx.write_daily(d, YEARS, stream="h2", steps_per_day=8)
    fx.write_monthly(d, YEARS)
    return d


@pytest.fixture
def fresh(tmp_path):
    """A run directory a test may damage."""
    fx.write_daily(tmp_path, YEARS)
    return tmp_path


# ── finding the files ────────────────────────────────────────────────────────
class TestDiscover:

    def test_finds_the_three_streams(self, run):
        found = G.discover(run)
        assert list(found) == ["h0", "h1", "h2"]
        assert [p.name for p in found["h1"]] == [
            "synth.elm.h1.2015-01-01-00000.nc", "synth.elm.h1.2016-01-01-00000.nc",
            "synth.elm.h1.2017-01-01-00000.nc"]

    def test_ignores_restarts_and_other_files(self, fresh):
        for name in ("synth.elm.rh0.2016-01-01-00000.nc",
                     "synth.elm.r.2016-01-01-00000.nc", "notes.txt",
                     "synth.elm.h1.2016-01-01-00000.nc.tmp"):
            (fresh / name).write_text("x")
        assert list(G.discover(fresh)) == ["h1"]
        assert len(G.discover(fresh)["h1"]) == 3

    def test_two_cases_in_one_directory_are_refused(self, fresh):
        fx.write_daily(fresh, [2015], case="other")
        with pytest.raises(G.GriddedReadError, match="2 cases"):
            G.discover(fresh)

    def test_missing_stream_names_what_is_there(self, fresh):
        with pytest.raises(G.GriddedReadError, match="streams present: h1"):
            G.open_stream(fresh, "h0")


# ── cadence and labels ───────────────────────────────────────────────────────
class TestDailyLabels:

    def test_cadence(self, run):
        s = G.open_stream(run, "h1")
        assert (s.cadence, s.steps_per_day, s.calendar) == ("daily", 1, "noleap")

    def test_the_start_record_is_dropped_and_listed(self, run):
        s = G.open_stream(run, "h1")
        assert len(s.dropped) == 1
        d = s.dropped[0]
        assert d["file"] == "synth.elm.h1.2015-01-01-00000.nc" and d["record"] == 0
        assert abs(d["length_days"] - 1 / 48) < 1e-6
        assert s.extended == []

    def test_a_year_is_assembled_across_the_file_boundary(self, run):
        s = G.open_stream(run, "h1")
        labels = s.labels(2015)
        assert len(labels) == 365 and len(set(labels)) == 365
        assert labels[0] == "2015-01-01" and labels[-1] == "2015-12-31"
        assert "2015-02-29" not in labels
        have, missing = s.year_records(2015)
        assert missing == []
        # 31 December is the first record of the NEXT year's file.
        last = have[-1]
        assert s.files[last.file].name == "synth.elm.h1.2016-01-01-00000.nc"
        assert last.index == 0

    def test_values_sit_on_the_start_of_their_interval(self, run):
        """DOYCODE holds (year - 2015) * 1000 + day of year. Labelled by the
        end stamp, every value would sit one day late."""
        s = G.open_stream(run, "h1")
        for y in YEARS:
            b = s.read("DOYCODE", y)
            want = np.array([(y - Y0) * 1000 + k for k in range(1, 366)], float)
            assert b.data.shape == (365, 4, 5)
            np.testing.assert_array_equal(b.data[:, 0, 0], want)
            assert np.all(b.data != -999.0), "the start record leaked in"

    def test_the_last_year_takes_31_december_from_the_last_file(self, run):
        s = G.open_stream(run, "h1")
        b = s.read("DOYCODE", 2016)
        assert b.labels[-1] == "2016-12-31" and b.data[-1, 2, 3] == 1365.0
        assert s.complete_years() == [2015, 2016]

    def test_years_and_record_count(self, run):
        s = G.open_stream(run, "h1")
        assert s.years() == [2015, 2016]
        assert len(s.records) == 730


class TestMonthlyLabels:

    def test_january_of_the_first_year_is_kept_and_listed(self, run):
        s = G.open_stream(run, "h0")
        assert s.cadence == "monthly" and s.steps_per_day is None
        assert s.dropped == []
        assert len(s.extended) == 1
        e = s.extended[0]
        assert e["month"] == "2015-01" and abs(e["extra_days"] - 1 / 48) < 1e-6

    def test_each_month_holds_its_own_values(self, run):
        """The writer stamps each month with its mid-month day of year. Labelled
        by the end stamp, January's values would be called February."""
        s = G.open_stream(run, "h0")
        for y in YEARS:
            b = s.read("DOYCODE", y)
            assert b.labels == [f"{y}-{m:02d}" for m in range(1, 13)]
            mid = [sum(fx.MONTH_DAYS[:m - 1]) + 15 for m in range(1, 13)]
            np.testing.assert_array_equal(b.data[:, 1, 1],
                                          [(y - Y0) * 1000 + d for d in mid])

    def test_a_partial_month_is_dropped_with_a_reason(self, tmp_path):
        """A run that starts on 15 January writes a 17-day 'January'."""
        g = fx.grid()
        tb = [(14.0, 31.0), (31.0, 59.0), (59.0, 90.0), (90.0, 120.0)]
        fx._write(tmp_path / "p.elm.h0.2015-02-01-00000.nc", g, 2015, tb,
                  [(2015, 20, 0), (2015, 45, 0), (2015, 74, 0), (2015, 105, 0)],
                  ["GPP"])
        s = G.open_stream(tmp_path, "h0")
        assert s.cadence == "monthly"
        assert [d["reason"][:6] for d in s.dropped] == ["covers"]
        assert s.labels() == ["2015-02", "2015-03", "2015-04"]
        assert s.complete_years() == []


class TestSubdailyLabels:

    def test_three_hourly_steps(self, run):
        s = G.open_stream(run, "h2")
        assert (s.cadence, s.steps_per_day) == ("subdaily", 8)
        labels = s.labels(2015)
        assert len(labels) == 2920
        assert labels[0] == "2015-01-01 00:00" and labels[-1] == "2015-12-31 21:00"
        assert len(s.dropped) == 1

    @pytest.mark.parametrize("n", [24, 48])
    def test_steps_with_no_exact_binary_length(self, tmp_path, n):
        """1/24 and 1/48 of a day decode a microsecond off; labels must not."""
        tb = [(k / n, (k + 1) / n) for k in range(2 * n)]
        fx._write(tmp_path / f"x.elm.h4.2015-01-01-00000.nc", fx.grid(), 2015, tb,
                  [(2015, 1 + k // n, 0) for k in range(2 * n)], ["TSA"])
        s = G.open_stream(tmp_path, "h4")
        assert (s.cadence, s.steps_per_day, s.dropped) == ("subdaily", n, [])
        labels = s.labels()
        assert len(labels) == 2 * n and len(set(labels)) == 2 * n
        step_min = 24 * 60 // n
        want = [f"2015-01-{1 + k // n:02d} {(k % n) * step_min // 60:02d}:"
                f"{(k % n) * step_min % 60:02d}" for k in range(2 * n)]
        assert labels == want

    def test_steps_are_contiguous_across_files(self, run):
        s = G.open_stream(run, "h2")
        b = s.read("STEPCODE", 2015)
        np.testing.assert_array_equal(b.data[:, 0, 0], np.arange(1, 2921))
        b = s.read("STEPCODE", 2016)
        np.testing.assert_array_equal(b.data[:, 3, 4], np.arange(2921, 5841))


# ── what is refused, and what is said ────────────────────────────────────────
class TestRefusals:

    def test_a_missing_next_file_is_a_missing_31_december(self, fresh):
        (fresh / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(fresh, "h1")
        assert s.complete_years() == [2015]
        with pytest.raises(G.GriddedReadError) as e:
            s.read("GPP", 2016)
        msg = str(e.value)
        assert "1 of 365" in msg and "2016-12-31" in msg and "next year's file" in msg
        b = s.read("GPP", 2016, require_complete=False)
        assert b.data.shape[0] == 364 and b.missing == ["2016-12-31"]

    def test_an_overlapping_file_is_refused(self, fresh):
        shutil.copy(fresh / "synth.elm.h1.2016-01-01-00000.nc",
                    fresh / "synth.elm.h1.2016-01-01-00001.nc")
        with pytest.raises(G.GriddedReadError, match="appears twice"):
            G.open_stream(fresh, "h1")

    def test_a_single_cell_history_is_left_to_extract(self, tmp_path):
        fx.write_single_cell(tmp_path / "col.elm.h0.2015-01-01-00000.nc")
        with pytest.raises(G.GriddedReadError, match="extract.extract_column"):
            G.open_stream(tmp_path, "h0")

    def test_files_on_different_grids_are_refused(self, tmp_path):
        fx.write_daily(tmp_path, [2015])
        g2 = fx.grid()
        g2["lat"] = g2["lat"] + 0.5
        fx._write(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", g2, 2015,
                  [(364.0, 365.0)], [(2015, 365, 0)], ["GPP"])
        with pytest.raises(G.GriddedReadError, match="different grid"):
            G.open_stream(tmp_path, "h1")

    def test_an_unknown_variable_says_so(self, run):
        s = G.open_stream(run, "h1")
        with pytest.raises(G.GriddedReadError, match="not on stream h1"):
            s.read("NOPE", 2015)

    def test_a_variable_missing_from_a_later_file(self, tmp_path):
        fx.write_daily(tmp_path, [2015], variables=("GPP", "TSA"))
        (tmp_path / "synth.elm.h1.2016-01-01-00000.nc").unlink()
        fx._write(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", fx.grid(), 2015,
                  [(364.0, 365.0)], [(2015, 365, 0)], ["GPP"])
        s = G.open_stream(tmp_path, "h1")
        with pytest.raises(G.GriddedReadError, match="missing from"):
            s.read("TSA", 2015)

    def test_a_cadence_it_does_not_know(self, tmp_path):
        tb = [(5.0 * k, 5.0 * (k + 1)) for k in range(4)]
        fx._write(tmp_path / "x.elm.h3.2015-01-01-00000.nc", fx.grid(), 2015, tb,
                  [(2015, 1, 0)] * 4, ["GPP"])
        with pytest.raises(G.GriddedReadError, match="cadence"):
            G.open_stream(tmp_path, "h3")

    def test_no_bounds_no_labels(self, tmp_path):
        import netCDF4
        p = tmp_path / "nb.elm.h1.2015-01-01-00000.nc"
        fx._write(p, fx.grid(), 2015, [(0.0, 1.0)], [(2015, 1, 0)], ["GPP"])
        with netCDF4.Dataset(p, "a") as d:
            d["time"].bounds = "tbnds_absent"
        with pytest.raises(G.GriddedReadError, match="labelled by its start"):
            G.open_stream(tmp_path, "h1")


# ── values, fill, grid, units ────────────────────────────────────────────────
class TestValues:

    def test_fill_and_nan_become_nan_and_nothing_else_does(self, run):
        s = G.open_stream(run, "h1")
        b = s.read("GPP", 2015)
        assert np.all(np.isnan(b.data[:, 0, 0]))           # never covered
        assert np.isnan(b.data[99, 1, 1])                  # 1e36 on day 100
        assert np.isnan(b.data[49, 2, 2])                  # NaN on day 50
        assert int(np.isnan(b.data).sum()) == 365 + 2
        for k in (0, 98, 100, 364):
            want = fx.expected("GPP", 2015, k + 1, 1, 1, Y0, 5)
            assert b.data[k, 1, 1] == pytest.approx(want, rel=1e-7)

    def test_clean_honours_the_variables_own_fill_value(self):
        class V:
            def ncattrs(self):
                return ["_FillValue"]

            def getncattr(self, name):
                return np.int16(-9999)
        a = G.clean(np.array([1, -9999, 3], dtype=np.int16), V())
        assert np.isnan(a[1]) and a[0] == 1 and a[2] == 3
        b = G.clean(np.array([1e35, 2e30, -5e31, np.inf, 7.0]))
        assert np.isnan(b[:4]).all() and b[4] == 7.0

    def test_grid_longitudes_and_weights(self, run):
        s = G.open_stream(run, "h1")
        g = s.grid
        assert g.dims == ("lat", "lon") and g.shape == (4, 5) and g.n_cells == 20
        assert np.all((g.lon >= -180) & (g.lon < 180))
        assert g.lon[0, 0] == pytest.approx(238.40416 - 360, abs=1e-4)
        w = g.weight_km2
        assert w[-1, -2] == 0.0 and w[-1, -1] == pytest.approx(0.5 * g.area_km2[-1, -1])
        assert w[0, 0] == pytest.approx(0.58, abs=1e-6)

    def test_the_cached_grid_cannot_be_changed_by_a_caller(self, run):
        g = G.open_stream(run, "h1").grid
        with pytest.raises(ValueError):
            g.lat[0, 0] = 0.0

    def test_check_coordinates(self, run):
        g = G.open_stream(run, "h1").grid
        lat, lon = fx.grid()["lat"], fx.grid()["lon"]
        got = g.check_coordinates(lat, lon - 360.0, "same cells, other convention")
        assert got["max_abs_lon_diff_deg"] < 1e-4
        with pytest.raises(G.GriddedReadError, match="not the same cells"):
            g.check_coordinates(lat + 0.01, lon, "shifted")
        with pytest.raises(G.GriddedReadError, match="shape"):
            g.check_coordinates(lat[:3], lon, "short")

    def test_window_layer_and_pieces_agree_with_the_whole(self, run):
        s = G.open_stream(run, "h1")
        full = s.read("CELLID", 2016).data
        win = s.read("CELLID", 2016, window=(1, 3, 2, 5)).data
        np.testing.assert_array_equal(win, full[:, 1:3, 2:5])
        np.testing.assert_array_equal(full[0], np.arange(20).reshape(4, 5))
        soil = s.read("SOILLIQ", 2016)
        assert soil.data.shape == (365, fx.NLEV, 4, 5)
        lay = s.read("SOILLIQ", 2016, layer=2)
        np.testing.assert_array_equal(lay.data, soil.data[:, 2])
        assert lay.data[0, 0, 1] == pytest.approx(        # 1 January 2016
            fx.expected("SOILLIQ", 2016, 1, 0, 1, Y0, 5, layer=2), rel=1e-6)
        parts = list(s.iter_read("DOYCODE", 2016, max_records=100))
        assert [p.data.shape[0] for p in parts] == [100, 100, 100, 65]
        np.testing.assert_array_equal(np.concatenate([p.data for p in parts]),
                                      s.read("DOYCODE", 2016).data)
        assert s.levels("SOILLIQ") == ("levgrnd",) and s.levels("GPP") == ()
        with pytest.raises(G.GriddedReadError, match="no levels"):
            s.read("GPP", 2016, layer=0)

    def test_read_records_returns_the_order_asked_for(self, run):
        s = G.open_stream(run, "h1")
        have, _ = s.year_records(2016)
        pick = [have[300], have[0], have[150]]          # file 2017 first, then 2016
        got = s.read_records("DOYCODE", pick)[:, 0, 0]
        np.testing.assert_array_equal(got, [1301.0, 1001.0, 1151.0])


class TestUnits:

    @pytest.mark.parametrize("var,units,want", [
        ("GPP", "gC/m^2/s", True), ("QRUNOFF", "mm/s", True),
        ("X", "kg m-2 s-1", True), ("X", "mm s^-1", True),
        ("FSDS", "W/m^2", False), ("TSA", "K", False), ("H2OSNO", "mm", False),
        ("TLAI", "1", False), ("FAREA_BURNED", "proportion", True),
        ("X", "proportion", False), ("X", "", False)])
    def test_per_second(self, var, units, want):
        assert G.is_per_second(var, units) is want

    @pytest.mark.parametrize("var,units,want", [
        ("GPP", "gC/m^2/s", "gC/m^2/d"), ("Q", "mm/s", "mm/d"),
        ("X", "kg m-2 s-1", "kg m-2 d-1"), ("FAREA_BURNED", "proportion",
                                              "proportion/d")])
    def test_per_day_units(self, var, units, want):
        assert G.per_day_units(var, units) == want

    def test_stream_reports_units(self, run):
        s = G.open_stream(run, "h1")
        assert s.units("GPP") == "gC/m^2/s" and s.per_second("GPP")
        assert s.per_second("FAREA_BURNED") and not s.per_second("TSA")
        b = s.read("FAREA_BURNED", 2015)
        assert b.per_second and b.units == "proportion"


class TestCalendars:

    @pytest.mark.parametrize("cal,year,days", [
        ("noleap", 2016, 365), ("365_day", 2016, 365), ("standard", 2016, 366),
        ("standard", 2015, 365), ("gregorian", 1900, 365),
        ("proleptic_gregorian", 2000, 366), ("julian", 1900, 366),
        ("360_day", 2016, 360), ("all_leap", 2015, 366)])
    def test_days_in_year(self, cal, year, days):
        assert G.days_in_year(year, cal) == days
        assert len(G.expected_keys(year, "daily", cal)) == days

    def test_subdaily_and_monthly_keys(self):
        k = G.expected_keys(2015, "subdaily", "noleap", 8)
        assert len(k) == 2920 and k[0] == (2015, 1, 1, 0, 0, 0)
        assert k[-1] == (2015, 12, 31, 21, 0, 0)
        assert len(G.expected_keys(2015, "monthly", "noleap")) == 12

    def test_unknown_calendar(self):
        with pytest.raises(G.GriddedReadError, match="calendar"):
            G.days_in_month(2015, 2, "lunar")


class TestEachReaderRefusesTheOthersFiles:

    def test_extract_refuses_a_grid(self, tmp_path):
        import extract
        case = tmp_path / "case"
        fx.write_daily(case / "run", [2015], stream="h0", variables=("GPP", "QVEGT"))
        data, meta = extract.extract_column(str(case), spinup_days=0)
        assert data == {}
        assert meta["status"].startswith("multi-cell history (20 cells)")
        assert "gridded.py" in meta["status"]

    def test_extract_still_takes_a_column(self, tmp_path):
        import extract
        case = tmp_path / "col"
        (case / "run").mkdir(parents=True)
        fx.write_single_cell(case / "run" / "col.elm.h0.2015-01-01-00000.nc")
        data, meta = extract.extract_column(str(case), variables=["TSA"],
                                            spinup_days=0)
        assert not meta["status"].startswith("multi-cell")
        assert meta["status"] == "ok" and len(data["dates"]) == 5


class TestIndexCache:

    def test_a_rewritten_file_is_read_again(self, tmp_path):
        import os
        import time
        fx.write_daily(tmp_path, [2015], variables=("DOYCODE",))
        s1 = G.open_stream(tmp_path, "h1")
        assert s1.labels(2015)[-1] == "2015-12-31"
        last = tmp_path / "synth.elm.h1.2016-01-01-00000.nc"
        last.unlink()
        time.sleep(0.01)
        fx._write(last, fx.grid(), 2015, [(363.0, 364.0)], [(2015, 364, 0)],
                  ["DOYCODE"])                           # now a duplicate of 30 Dec
        os.utime(last, ns=(time.time_ns(), time.time_ns() + 10 ** 9))
        with pytest.raises(G.GriddedReadError, match="appears twice"):
            G.open_stream(tmp_path, "h1")

    def test_a_second_open_reads_no_time_bounds(self, run, monkeypatch):
        G.open_stream(run, "h1")
        import netCDF4
        calls = []
        real = netCDF4.Dataset

        def counting(*a, **k):
            calls.append(a[0])
            return real(*a, **k)
        monkeypatch.setattr(netCDF4, "Dataset", counting)
        s = G.open_stream(run, "h1")
        assert calls == [] and len(s.records) == 730


# ═════════════════════════════════════════════════════════════════════════════
# Added after the independent review of 2026-10-22: each class pins a defect
# the review demonstrated, or a branch no earlier test reached.
# ═════════════════════════════════════════════════════════════════════════════
class TestStartStep:
    """The run's start step is found by nstep 0, not by being an odd length."""

    def test_a_one_day_run(self, tmp_path):
        """mfilt = 1: the start step is a file of its own, the day another. By
        width alone the two tie and the run read as 30-minute output."""
        g = fx.grid()
        fx._write(tmp_path / "r.elm.h1.2015-01-01-00000.nc", g, 2015, [(-fx.DT, 0.0)],
                  [None], ["DOYCODE"])
        fx._write(tmp_path / "r.elm.h1.2015-01-02-00000.nc", g, 2015, [(0.0, 1.0)],
                  [(2015, 1, 0)], ["DOYCODE"])
        s = G.open_stream(tmp_path, "h1")
        assert (s.cadence, s.steps_per_day, s.labels()) == ("daily", 1, ["2015-01-01"])
        assert [d["reason"] for d in s.dropped] == ["the run's start step (nstep 0)"]
        assert s.read("DOYCODE", 2015, require_complete=False).data[0, 0, 0] == 1.0

    def test_output_every_step_drops_the_start_step(self, tmp_path):
        """hist_nhtfrq = 1: the start step is as long as every other record."""
        n = 48
        tb = [(-fx.DT, 0.0)] + [(k / n, (k + 1) / n) for k in range(2 * n)]
        labels = [None] + [(2015, 1 + k // n, k % n) for k in range(2 * n)]
        fx._write(tmp_path / "p.elm.h3.2015-01-01-00000.nc", fx.grid(), 2015, tb, labels,
                  ["TSA"])
        s = G.open_stream(tmp_path, "h3")
        assert (s.cadence, s.steps_per_day) == ("subdaily", 48)
        assert s.labels()[0] == "2015-01-01 00:00" and s.years() == [2015]
        assert [d["reason"] for d in s.dropped] == ["the run's start step (nstep 0)"]

    def test_without_nstep_the_width_rule_still_drops_it(self, tmp_path):
        g = fx.grid()
        tb = [(-fx.DT, 0.0)] + [(k, k + 1.0) for k in range(10)]
        fx._write(tmp_path / "w.elm.h1.2015-01-01-00000.nc", g, 2015, tb,
                  [None] + [(2015, k + 1, 0) for k in range(10)], ["DOYCODE"],
                  nstep=False)
        s = G.open_stream(tmp_path, "h1")
        assert s.cadence == "daily" and s.labels()[0] == "2015-01-01"
        assert s.dropped[0]["reason"].startswith("interval of 0.020833 days")

    def test_a_stream_holding_only_the_start_step(self, tmp_path):
        fx._write(tmp_path / "o.elm.h1.2015-01-01-00000.nc", fx.grid(), 2015,
                  [(-fx.DT, 0.0)], [None], ["DOYCODE"])
        with pytest.raises(G.GriddedReadError, match="only record is the run's start"):
            G.open_stream(tmp_path, "h1")


class TestFileNames:

    def test_elm_default_monthly_names(self, tmp_path):
        """hist_mfilt = 1 names each file for its month: <case>.elm.h0.YYYY-MM.nc."""
        g = fx.grid()
        for y in (2015, 2016):
            off = 365.0 * (y - 2015)
            cum = 0
            for m in range(1, 13):
                a = off + cum - (fx.DT if (y, m) == (2015, 1) else 0.0)
                cum += fx.MONTH_DAYS[m - 1]
                fx._write(tmp_path / f"m.elm.h0.{y}-{m:02d}.nc", g, 2015,
                          [(a, off + cum)], [(y, sum(fx.MONTH_DAYS[:m - 1]) + 15, 0)],
                          ["DOYCODE"])
        assert len(G.discover(tmp_path)["h0"]) == 24
        s = G.open_stream(tmp_path, "h0")
        assert s.cadence == "monthly" and s.complete_years() == [2015, 2016]
        assert [e["month"] for e in s.extended] == ["2015-01"]
        mid = [sum(fx.MONTH_DAYS[:m - 1]) + 15 for m in range(1, 13)]
        np.testing.assert_array_equal(s.read("DOYCODE", 2016).data[:, 0, 0],
                                      [1000 + d for d in mid])

    def test_a_multi_instance_run_is_refused_by_name(self, tmp_path):
        fx._write(tmp_path / "c.elm_0001.h0.2015-01-01-00000.nc", fx.grid(), 2015,
                  [(0.0, 1.0)], [(2015, 1, 0)], ["GPP"])
        fx._write(tmp_path / "c.elm_0002.h0.2015-01-01-00000.nc", fx.grid(), 2015,
                  [(0.0, 1.0)], [(2015, 1, 0)], ["GPP"])
        with pytest.raises(G.GriddedReadError, match="multi-instance.*elm_0001, elm_0002"):
            G.discover(tmp_path)


class TestUnitsInTheRealFiles:
    """Every units spelling found in the Naches h0/h1 headers (2026-10-22)."""

    @pytest.mark.parametrize("units", [
        "gC/m^2/s", "gC/m2/s", "gN/m^2/s", "mm/s", "mm H2O/s", "kg/m^2/s", "mol/m2/s",
        "umolCO2/m^2/s", "gC/m^3/s", "umol/m2s", "per sec", "proportion/sec",
        "counts/km2/sec", "kg m-2 s-1", "mm s^-1"])
    def test_rates_per_second(self, units):
        assert G.is_per_second("X", units)

    @pytest.mark.parametrize("units", [
        "W/m^2", "K", "mm", "gC/m^2", "1", "mm3/mm3", "m^2/m^2", "MJ/m2", "Pa",
        "kg/m/s^2", "kg m-1 s-2", "m/s2", "proportion", "s"])
    def test_not_rates_per_second(self, units):
        assert not G.is_per_second("X", units)

    @pytest.mark.parametrize("units,want", [
        ("umol/m2s", "umol/m2/d"), ("per sec", "per day"),
        ("proportion/sec", "proportion/d"), ("counts/km2/sec", "counts/km2/d"),
        ("mm s^-1", "mm d^-1"), ("kg/m/s^2", "kg/m/s^2")])
    def test_per_day_spellings(self, units, want):
        assert G.per_day_units("X", units) == want


class TestMetadataAndLayers:

    def test_per_record_values_are_not_listed_as_fields(self, run):
        s = G.open_stream(run, "h1")
        assert "nstep" not in s.variables and "mcdate" not in s.variables
        assert {"nstep", "mcdate"} <= set(s.metadata_variables)
        with pytest.raises(G.GriddedReadError, match="per-record value"):
            s.read("nstep", 2015)

    def test_layers_count_from_the_bottom_and_are_checked(self, run):
        s = G.open_stream(run, "h1")
        last = s.read("SOILLIQ", 2016, layer=-1).data
        np.testing.assert_array_equal(last, s.read("SOILLIQ", 2016, layer=fx.NLEV - 1).data)
        with pytest.raises(G.GriddedReadError, match="not one of them"):
            s.read("SOILLIQ", 2016, layer=fx.NLEV)
        with pytest.raises(G.GriddedReadError, match="not one of them"):
            s.read("SOILLIQ", 2016, layer=-fx.NLEV - 1)


class TestDroppedRecords:

    def test_every_record_off_the_step_grid(self, tmp_path):
        """A run started at noon (START_TOD = 43200) writes days [0.5, 1.5] ..."""
        tb = [(k + 0.5, k + 1.5) for k in range(10)]
        fx._write(tmp_path / "n.elm.h1.2015-01-01-00000.nc", fx.grid(), 2015, tb,
                  [(2015, k + 1, 0) for k in range(10)], ["GPP"])
        with pytest.raises(G.GriddedReadError,
                           match="all 10 records were dropped: 10 does not start"):
            G.open_stream(tmp_path, "h1")

    def test_width_and_step_grid_are_each_checked(self, tmp_path):
        tb = [(0, 1), (1, 2), (2, 4), (4, 5), (4.5, 5.5), (5, 6), (6, 7)]
        labels = [(2015, 1, 0), (2015, 2, 0), (2015, 3, 0), (2015, 5, 0), (2015, 5, 0),
                  (2015, 6, 0), (2015, 7, 0)]
        fx._write(tmp_path / "d.elm.h1.2015-01-01-00000.nc", fx.grid(), 2015, tb, labels,
                  ["DOYCODE"])
        # [2, 4] is on the grid but two days long; [4.5, 5.5] is a day long but
        # off the grid. Each fails one check only, so neither check hides the
        # other, and both are dropped before overlaps are looked for.
        s = G.open_stream(tmp_path, "h1")
        reasons = {d["record"]: d["reason"] for d in s.dropped}
        assert set(reasons) == {2, 4}
        assert reasons[2].startswith("interval of 2.000000 days")
        assert reasons[4] == "does not start on the stream's step grid"
        assert s.labels() == ["2015-01-01", "2015-01-02", "2015-01-05", "2015-01-06",
                              "2015-01-07"]

    def test_monthly_records_dropped_for_their_ends_and_lengths(self, tmp_path):
        tb = [(0, 31), (31, 59), (59, 90), (90, 120), (120, 151), (151, 181),
              (181, 200), (212, 273)]
        fx._write(tmp_path / "m.elm.h0.2015-02-01-00000.nc", fx.grid(), 2015, tb,
                  [(2015, 15, 0)] * len(tb), ["GPP"])
        s = G.open_stream(tmp_path, "h0")
        assert s.cadence == "monthly"
        reasons = {d["record"]: d["reason"] for d in s.dropped}
        assert reasons == {6: "does not end at the start of a month",
                           7: "is 61.0000 days long, more than a month plus one model step"}
        assert s.labels() == [f"2015-{m:02d}" for m in range(1, 7)]

    def test_a_month_far_longer_than_a_step_is_dropped(self, tmp_path):
        """April starting 0.9 days early: not a month plus a start step."""
        tb = [(0, 31), (31, 59), (59, 90), (89.1, 120), (120, 151)]
        fx._write(tmp_path / "m.elm.h0.2015-02-01-00000.nc", fx.grid(), 2015, tb,
                  [(2015, 15, 0)] * len(tb), ["GPP"])
        s = G.open_stream(tmp_path, "h0")
        assert [(d["record"], d["reason"]) for d in s.dropped] == [
            (3, "is 30.9000 days long, more than a month plus one model step")]
        assert s.labels() == ["2015-01", "2015-02", "2015-03", "2015-05"]

    def test_a_restarted_segment_folds_its_start_step_into_its_first_month(self, tmp_path):
        """MidA_fdrought: 1980-2018, then a new run from 1 January 2019 whose
        time counts from 2019-01-01 and whose January starts one step early,
        overlapping December 2018 by that step."""
        import netCDF4
        g = fx.grid()
        fx._write(tmp_path / "s.elm.h0.2018-12.nc", g, 2018, [(334.0, 365.0)],
                  [(2018, 349, 0)], ["DOYCODE"])
        fx._write(tmp_path / "s.elm.h0.2019-01.nc", g, 2019, [(-fx.DT, 31.0)],
                  [(2019, 15, 0)], ["DOYCODE"])
        fx._write(tmp_path / "s.elm.h0.2019-02.nc", g, 2019, [(31.0, 59.0)],
                  [(2019, 46, 0)], ["DOYCODE"])
        s = G.open_stream(tmp_path, "h0")
        assert s.labels() == ["2018-12", "2019-01", "2019-02"]
        assert s.dropped == []
        assert s.extended == [{"file": "s.elm.h0.2019-01.nc", "record": 0,
                               "month": "2019-01", "extra_days": round(fx.DT, 6),
                               "overlaps_previous_days": round(fx.DT, 6)}]

    def test_unwritten_time_bounds_are_dropped_and_said(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("DOYCODE",))
        p = tmp_path / "synth.elm.h1.2015-01-01-00000.nc"
        with netCDF4.Dataset(p, "a") as d:
            d["time_bounds"][200, :] = 9.969209968386869e36
        s = G.open_stream(tmp_path, "h1")
        bad = [x for x in s.dropped if x["reason"].startswith("time bounds not written")]
        assert [(x["file"], x["record"]) for x in bad] == [(p.name, 200)]
        assert s.year_records(2015)[1] == [(2015, 7, 19, 0, 0, 0)]


class TestIterRead:

    def test_a_short_year(self, tmp_path):
        fx.write_daily(tmp_path, YEARS, variables=("DOYCODE",))
        (tmp_path / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(tmp_path, "h1")
        with pytest.raises(G.GriddedReadError, match="1 of 365"):
            list(s.iter_read("DOYCODE", 2016, max_records=100))
        parts = list(s.iter_read("DOYCODE", 2016, max_records=100, require_complete=False))
        assert [p.missing for p in parts] == [["2016-12-31"], [], [], []]
        assert sum(p.data.shape[0] for p in parts) == 364

    def test_a_year_with_no_records_raises_like_read(self, run):
        s = G.open_stream(run, "h1")
        with pytest.raises(G.GriddedReadError, match="no records in 2030"):
            s.read("DOYCODE", 2030, require_complete=False)
        with pytest.raises(G.GriddedReadError, match="no records in 2030"):
            list(s.iter_read("DOYCODE", 2030, max_records=100, require_complete=False))

    def test_max_records_below_one_is_refused(self, run):
        s = G.open_stream(run, "h1")
        with pytest.raises(G.GriddedReadError, match="at least 1"):
            list(s.iter_read("DOYCODE", 2016, max_records=0))


class TestFieldKinds:

    def test_an_instantaneous_field_is_refused_on_a_daily_stream(self, tmp_path):
        fx.write_daily(tmp_path / "d", [2015], variables=("TSA",))
        for p in (tmp_path / "d").glob("*.nc"):
            import netCDF4
            with netCDF4.Dataset(p, "a") as ds:
                ds["TSA"].cell_methods = "time: point"
        s = G.open_stream(tmp_path / "d", "h1")
        with pytest.raises(G.GriddedReadError, match="instantaneous"):
            s.read("TSA", 2015)
        fx.write_monthly(tmp_path / "m", [2015], variables=("TSA",))
        for p in (tmp_path / "m").glob("*.nc"):
            import netCDF4
            with netCDF4.Dataset(p, "a") as ds:
                ds["TSA"].cell_methods = "time: point"
        assert G.open_stream(tmp_path / "m", "h0").read("TSA", 2015).data.shape[0] == 12

    def test_a_packed_variable_is_unpacked_after_its_fill_is_found(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("TSA",))
        for p in tmp_path.glob("*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                v = d.createVariable("PACKED", "i2", ("time", "lat", "lon"),
                                     fill_value=np.int16(-32767))
                v.scale_factor = 0.01
                v.add_offset = 100.0
                v.units = "K"
                v.set_auto_maskandscale(False)
                raw = np.full(v.shape, 1234, dtype=np.int16)
                raw[:, 0, 0] = -32767
                v[:] = raw
        s = G.open_stream(tmp_path, "h1")
        b = s.read("PACKED", 2015)
        assert np.all(np.isnan(b.data[:, 0, 0]))
        assert b.data[0, 1, 1] == pytest.approx(112.34)

    def test_calendar_names_that_mean_the_same_calendar(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("GPP",))
        with netCDF4.Dataset(tmp_path / "synth.elm.h1.2015-01-01-00000.nc", "a") as d:
            d["time"].calendar = "365_day"
        assert G.open_stream(tmp_path, "h1").calendar == "noleap"

    def test_a_calendar_mismatch_between_files_is_refused(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("GPP",))
        with netCDF4.Dataset(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", "a") as d:
            d["time"].calendar = "standard"
        with pytest.raises(G.GriddedReadError, match="uses calendar standard"):
            G.open_stream(tmp_path, "h1")


class TestGridChecks:

    def test_files_differing_only_in_longitude_are_refused(self, tmp_path):
        fx.write_daily(tmp_path, [2015])
        g2 = fx.grid()
        g2["lon"] = g2["lon"] + 0.5
        (tmp_path / "synth.elm.h1.2016-01-01-00000.nc").unlink()
        fx._write(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", g2, 2015,
                  [(364.0, 365.0)], [(2015, 365, 0)], ["GPP"])
        with pytest.raises(G.GriddedReadError, match="different grid"):
            G.open_stream(tmp_path, "h1")

    def test_area_not_in_km2_is_refused(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("GPP",))
        for p in tmp_path.glob("*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["area"].units = "m^2"
        with pytest.raises(G.GriddedReadError, match="area is in 'm\\^2'"):
            G.open_stream(tmp_path, "h1")

    def test_fill_in_landfrac_weighs_nothing(self, tmp_path):
        g = fx.grid()
        g["landfrac"][0, 1] = 1e36
        fx.write_daily(tmp_path, [2015], g=g, variables=("TSA",))
        s = G.open_stream(tmp_path, "h1")
        assert s.grid.weight_km2[0, 1] == 0.0 and np.isnan(s.grid.landfrac[0, 1])

    def test_the_grid_cannot_be_rebound_either(self, run):
        import dataclasses
        g = G.open_stream(run, "h1").grid
        with pytest.raises(dataclasses.FrozenInstanceError):
            g.landfrac = np.zeros(g.shape)

    def test_clean_honours_missing_value_alone(self):
        class V:
            def ncattrs(self):
                return ["missing_value"]

            def getncattr(self, name):
                return np.float32(-9999.0)
        a = G.clean(np.array([1.0, -9999.0, 3.0], dtype=np.float32), V())
        assert np.isnan(a[1]) and a[0] == 1.0 and a[2] == 3.0

    def test_describe(self, run):
        d = G.open_stream(run, "h1").describe()
        assert (d["n_files"], d["cadence"], d["steps_per_day"], d["calendar"]) == \
            (3, "daily", 1, "noleap")
        assert (d["first_label"], d["last_label"], d["n_records"]) == \
            ("2015-01-01", "2016-12-31", 730)
        assert d["complete_years"] == [2015, 2016] and len(d["dropped"]) == 1
        g = fx.grid()
        w = (np.float32(g["area"]).astype(float) * np.float32(g["landfrac"]).astype(float))
        assert d["grid"]["shape"] == [4, 5] and d["grid"]["n_cells"] == 20
        assert d["grid"]["area_km2_total"] == pytest.approx(w.sum(), rel=1e-7)


class TestTimesFarFromTheReference:

    @pytest.mark.parametrize("n", [24, 48])
    def test_steps_that_decode_a_microsecond_off(self, tmp_path, n):
        """'days since 0001-01-01' at model year 2015, as a long spin-up writes:
        here num2date really returns hh:59:59.999997, unlike near the reference."""
        import netCDF4
        off = 365.0 * 2014
        tb = [(off + k / n, off + (k + 1) / n) for k in range(2 * n)]
        fx._write(tmp_path / f"x.elm.h4.2015-01-01-00000.nc", fx.grid(), 1, tb,
                  [(2015, 1 + k // n, 0) for k in range(2 * n)], ["TSA"], nstep=False)
        raw = netCDF4.num2date(np.array(tb)[:, 0], "days since 0001-01-01 00:00:00",
                               calendar="noleap")
        assert sum(t.microsecond != 0 for t in raw) > 0, "the input must need rounding"
        s = G.open_stream(tmp_path, "h4")
        assert (s.cadence, s.steps_per_day, s.dropped) == ("subdaily", n, [])
        step_min = 24 * 60 // n
        assert s.labels() == [f"2015-01-{1 + k // n:02d} {(k % n) * step_min // 60:02d}:"
                              f"{(k % n) * step_min % 60:02d}" for k in range(2 * n)]


class TestCacheKey:

    def test_a_same_size_rewrite_with_the_old_mtime_is_read_again(self, tmp_path):
        """Lustre keeps mtime in whole seconds and full-year files share a size;
        the cache key holds the inode and ctime as well."""
        import os
        fx.write_daily(tmp_path, [2015], variables=("DOYCODE",))
        p = tmp_path / "synth.elm.h1.2016-01-01-00000.nc"
        before = p.stat()
        assert G.open_stream(tmp_path, "h1").labels()[-1] == "2015-12-31"
        p.unlink()
        fx._write(p, fx.grid(), 2015, [(363.0, 364.0)], [(2015, 364, 0)], ["DOYCODE"])
        assert p.stat().st_size == before.st_size
        os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(G.GriddedReadError, match="appears twice"):
            G.open_stream(tmp_path, "h1")


class TestUnstructured:
    """lndgrid histories: the shape of the MidAtl and MidA_fdrought runs."""

    def test_reads_labels_layers(self, tmp_path):
        fx.write_daily_lnd(tmp_path, YEARS)
        s = G.open_stream(tmp_path, "h1")
        g = s.grid
        assert g.dims == ("lndgrid",) and g.shape == (fx.NL,) and not g.regular
        assert s.complete_years() == YEARS
        np.testing.assert_allclose(g.lat, np.float32(fx.LND_LAT), atol=0)
        np.testing.assert_allclose(g.lon, np.float32(fx.LND_LON) - 360.0, atol=1e-4)
        assert s.levels("SOILLIQ") == ("levgrnd",) and s.levels("GPP") == ()
        b = s.read("GPP", 2016)
        assert b.data.shape == (365, fx.NL)
        for doy in (1, 200, 365):
            for c in range(fx.NL):
                want = fx.expected("GPP", 2016, doy, 0, c, Y0, fx.NL)
                got = b.data[doy - 1, c]
                assert (np.isnan(want) and np.isnan(got)) or got == pytest.approx(want, rel=1e-7)
        lay = s.read("SOILLIQ", 2016, layer=2)
        assert lay.data.shape == (365, fx.NL)
        assert lay.data[0, 4] == pytest.approx(
            fx.expected("SOILLIQ", 2016, 1, 0, 4, Y0, fx.NL, layer=2))
        with pytest.raises(G.GriddedReadError, match="regular"):
            s.read("GPP", 2016, window=(0, 1, 0, 1))

    def test_a_one_cell_lndgrid_is_left_to_extract(self, tmp_path):
        import netCDF4
        p = tmp_path / "one.elm.h1.2015-01-01-00000.nc"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("time", None)
            d.createDimension("lndgrid", 1)
            d.createDimension("hist_interval", 2)
            t = d.createVariable("time", "f8", ("time",))
            t.units, t.calendar, t.bounds = "days since 2015-01-01", "noleap", "time_bounds"
            d.createVariable("time_bounds", "f8", ("time", "hist_interval"))[:] = [[0, 1]]
            t[:] = [1.0]
            for n, val in (("lat", 46.5), ("lon", 238.5), ("area", 0.5), ("landfrac", 1.0)):
                v = d.createVariable(n, "f4", ("lndgrid",))
                v[:] = [val]
            d["area"].units = "km^2"
        with pytest.raises(G.GriddedReadError, match="extract.extract_column"):
            G.open_stream(tmp_path, "h1")


class TestExtractGuardShapes:

    def _case(self, tmp_path, writer):
        case = tmp_path / "case"
        (case / "run").mkdir(parents=True)
        writer(case / "run")
        return str(case)

    @pytest.mark.parametrize("shape", [(1, 5), (5, 1), (3, 4)])
    def test_extract_refuses_every_multi_cell_grid(self, tmp_path, shape):
        import extract
        g = fx.grid(*shape)
        case = self._case(tmp_path, lambda run: fx._write(
            run / "c.elm.h0.2015-01-01-00000.nc", g, 2015,
            [(k, k + 1.0) for k in range(5)], [(2015, k + 1, 0) for k in range(5)],
            ["TSA"]))
        data, meta = extract.extract_column(case, variables=["TSA"], spinup_days=0)
        n = shape[0] * shape[1]
        assert data == {} and meta["status"].startswith(f"multi-cell history ({n} cells)")

    def test_extract_refuses_an_unstructured_grid(self, tmp_path):
        import extract
        tb = [(-fx.DT, 0.0)] + [(float(k), k + 1.0) for k in range(10)]
        labels = [None] + [(2015, k + 1, 0) for k in range(10)]
        case = self._case(tmp_path, lambda run: fx.write_lnd(
            run / "u.elm.h0.2015-01-01-00000.nc", 2015, tb, labels, ["GPP", "TSA"]))
        data, meta = extract.extract_column(case, variables=["TSA"], spinup_days=0)
        assert data == {} and meta["status"].startswith(f"multi-cell history ({fx.NL} cells)")

    def test_extract_refuses_vector_history(self, tmp_path):
        import extract
        import netCDF4

        def writer(run):
            with netCDF4.Dataset(run / "v.elm.h0.2015-01-01-00000.nc", "w") as d:
                d.createDimension("time", None)
                d.createDimension("gridcell", 6)
                t = d.createVariable("time", "f8", ("time",))
                t.units, t.calendar = "days since 2015-01-01", "noleap"
                t[:] = np.arange(1.0, 6.0)
                v = d.createVariable("TSA", "f4", ("time", "gridcell"))
                v[:] = np.full((5, 6), 280.0)
        case = self._case(tmp_path, writer)
        data, meta = extract.extract_column(case, variables=["TSA"], spinup_days=0)
        assert data == {} and meta["status"].startswith("multi-cell history (6 cells)")
