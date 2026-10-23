"""The run gates on a small case in ELM's layout.

    pytest tests/test_run_gate.py -v

Year numbers are checked against sums written out here from the fixture's
formulas; every failure mode must say what failed, and every gate criterion
must fail some test when it is taken out.
"""
import json
import math
import shutil

import numpy as np
import pytest

import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import gridded as G
import regions as R
import run_gate as RG

Y0 = 2015
LIMITS = RG.YearLimits(water_closure_frac_max=10.0)   # the formulas do not close


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return cf.write_case(tmp_path_factory.mktemp("case"))


def _fresh(tmp_path, **kw):
    return cf.write_case(tmp_path, **kw)


def _edit(path, var, fn):
    import netCDF4
    with netCDF4.Dataset(path, "a") as d:
        v = d[var]
        v.set_auto_maskandscale(False)
        v[:] = fn(np.asarray(v[:]))


def _at(a, index, value):
    a = a.copy()
    a[index] = value
    return a


def _lnd_in(c):
    return c["case_dir"] / "CaseDocs" / "lnd_in"


# ── the namelist ─────────────────────────────────────────────────────────────
class TestNamelist:

    def test_values_over_lines_quotes_comments_and_types(self, tmp_path):
        p = tmp_path / "lnd_in"
        p.write_text("&a\n s = 'x, y!z'  ! a comment\n lst = 'A','B',\n   'C'\n"
                     " n = 0,-24,  -3\n f = 1.5d0\n t = .true.\n/\n&b\n k = 7\n/\n")
        nl = RG.read_namelist(p)
        assert nl == {"s": ["x, y!z"], "lst": ["A", "B", "C"], "n": [0, -24, -3],
                      "f": [1.5], "t": [True], "k": [7]}

    def test_an_indexed_entry_is_refused_not_misread(self, tmp_path):
        p = tmp_path / "lnd_in"
        p.write_text("&a\n hist_fincl2 = 'GPP','TLAI'\n hist_nhtfrq(2) = -24\n/\n")
        with pytest.raises(RG.GateError, match=r"indexed entry hist_nhtfrq\(2\)"):
            RG.read_namelist(p)

    def test_tapes_and_their_intervals(self, case):
        nl, path = RG.case_namelist(case["case_dir"], case["run_dir"])
        assert path.parts[-2:] == ("CaseDocs", "lnd_in")
        tapes = RG.requested_tapes(nl)
        assert list(tapes) == ["h0", "h1", "h2"] and tapes["h2"] == ["TSA", "CELLID"]
        assert [RG.tape_interval_days(nl, s) for s in tapes] == [30.0, 1.0, 0.125]
        assert RG.tape_interval_days({"hist_nhtfrq": [1], "dtime": [3600]}, "h0") == \
            pytest.approx(1 / 24)

    def test_a_tape_past_the_list_keeps_elms_default(self):
        """histFileMod: hist_nhtfrq = (/0, -24, -24, .../): h0 monthly, the
        others daily, where the namelist's list does not reach."""
        nl = {"hist_nhtfrq": [0]}
        assert RG.tape_interval_days(nl, "h0") == 30.0
        assert RG.tape_interval_days(nl, "h1") == 1.0
        assert RG.tape_interval_days({}, "h2") == 1.0

    def test_when_a_tape_first_writes(self):
        import cftime
        nl = {"hist_nhtfrq": [0, -24, -3]}
        jan1 = cftime.DatetimeNoLeap(2015, 1, 1)
        assert RG.first_write_days(nl, "h0", jan1, "noleap") == 31.0
        assert RG.first_write_days(nl, "h0", cftime.DatetimeNoLeap(2015, 2, 11), "noleap") == 18.0
        assert RG.first_write_days(nl, "h1", jan1, "noleap") == 1.0
        assert RG.first_write_days(nl, "h2", jan1, "noleap") == 0.125

    def test_the_run_folders_lnd_in_wins(self, tmp_path):
        c = _fresh(tmp_path)
        (c["run_dir"] / "lnd_in").write_text("&x\n fsurdat = 'other.nc'\n/\n")
        nl, path = RG.case_namelist(c["case_dir"], c["run_dir"])
        assert path == c["run_dir"] / "lnd_in" and nl["fsurdat"] == ["other.nc"]
        with pytest.raises(RG.GateError, match="no lnd_in"):
            RG.case_namelist(tmp_path / "nowhere")


# ── the job's logs ───────────────────────────────────────────────────────────
class TestScanLogs:

    def test_a_clean_job(self, case):
        lg = RG.scan_logs(case["run_dir"], case["case_dir"], case["job"])
        assert lg["pass"] and lg["reasons"] == []
        assert lg["successful_termination"] and lg["abort_lines"] == []
        assert (lg["balance_lines"], lg["nan_lines"]) == (0, 0)
        assert lg["sim_years_per_day"] == 30.18
        assert lg["hours_per_sim_year"] == pytest.approx(24 / 30.18)
        assert (lg["init_time_s"], lg["run_time_s"]) == (418.208, 2862.626)
        assert (lg["model_start"], lg["model_end"], lg["simulated_days"]) == \
            ("2015-01-01", "2017-01-01", 730.0)

    def test_only_this_jobs_logs_are_read(self, tmp_path):
        c = _fresh(tmp_path)
        cf.write_logs(c["run_dir"], c["case_dir"], c["case"], "100", lid="261014-000000",
                      ok=False, lnd_lines=["ENDRUN: an older run died"])
        assert RG.scan_logs(c["run_dir"], c["case_dir"], "111")["pass"]
        old = RG.scan_logs(c["run_dir"], c["case_dir"], "100")
        assert not old["pass"] and old["abort_lines"] == ["ENDRUN: an older run died"]
        assert old["reasons"][0] == ("the coupler log has no SUCCESSFUL TERMINATION "
                                     "(last model date 2015-01-01)")
        assert old["model_end"] is None

    def test_endrun_fails_the_scan_even_when_the_coupler_finished(self, tmp_path):
        """ENDRUN is printed alone; the cause is on the next lines."""
        c = _fresh(tmp_path, job="444")
        for p in c["run_dir"].glob("lnd.log.444.*"):
            p.unlink()
        cf.write_logs(c["run_dir"], c["case_dir"], c["case"], "444", ok=True,
                      lnd_lines=[" ENDRUN:", "    ", " ERROR in CNPBudgetMod.F90 at line 929",
                                 "", " ERROR: Unknown error submitted to shr_abort_abort."])
        lg = RG.scan_logs(c["run_dir"], c["case_dir"], "444")
        assert lg["successful_termination"] and not lg["pass"]
        assert lg["reasons"] == ["the land log stops the run: ENDRUN: ERROR in CNPBudgetMod.F90 "
                                 "at line 929 ERROR: Unknown error submitted to "
                                 "shr_abort_abort."]

    def test_an_abort_seen_only_in_the_model_log(self, tmp_path):
        c = _fresh(tmp_path, job="555")
        cf.write_logs(c["run_dir"], c["case_dir"], c["case"], "555", ok=False,
                      e3sm_lines=[" 7: ENDRUN: a task other than the land master stopped",
                                  " 0: application called MPI_Abort(MPI_COMM_WORLD, 1001)"])
        lg = RG.scan_logs(c["run_dir"], c["case_dir"], "555")
        assert lg["abort_lines"] == [] and len(lg["e3sm_error_lines"]) == 2
        assert lg["reasons"][-1] == ("the model log says: 7: ENDRUN: a task other than the "
                                     "land master stopped")

    def test_a_resubmitted_job_reads_its_newest_logs(self, tmp_path):
        c = _fresh(tmp_path)
        cf.write_logs(c["run_dir"], c["case_dir"], c["case"], "111", lid="261015-090000",
                      ok=False, lnd_lines=["ENDRUN: second attempt"], gz=False)
        lg = RG.scan_logs(c["run_dir"], c["case_dir"], "111")
        assert lg["cpl_log"].endswith("cpl.log.111.261015-090000") and not lg["pass"]

    def test_balance_lines_count_problems_not_field_names(self, tmp_path):
        c = _fresh(tmp_path, job="222")
        for p in c["run_dir"].glob("lnd.log.222.*"):
            p.unlink()
        cf.write_logs(c["run_dir"], c["case_dir"], c["case"], "222", lnd_lines=[
            "   12  CMASS_BALANCE_ERROR  carbon mass balance error",
            "skipping CN balance check for first timestep (warning)",
            "WARNING: water balance error nstep= 48 point= 7",
            "BalanceCheck: soil balance error",
            " ERROR in something",
            " value NaN at column 3"])
        lg = RG.scan_logs(c["run_dir"], c["case_dir"], "222")
        assert lg["balance_lines"] == 2 and lg["nan_lines"] == 1
        assert lg["error_lines"] == ["ERROR in something"]
        assert lg["pass"], "ERROR and balance lines are recorded, not gated"

    def test_missing_logs_and_a_missing_job_id(self, tmp_path):
        c = _fresh(tmp_path)
        lg = RG.scan_logs(c["run_dir"], c["case_dir"], "999")
        assert not lg["pass"] and lg["reasons"] == ["no cpl.log from job 999",
                                                    "no lnd.log from job 999"]
        assert lg.get("sim_years_per_day") is None and lg["timing_file"] is None
        with pytest.raises(RG.GateError, match="needs the job id"):
            RG.scan_logs(c["run_dir"], c["case_dir"], None)

    def test_the_year_must_be_this_jobs(self, case):
        lg = RG.scan_logs(case["run_dir"], case["case_dir"], case["job"])
        assert RG.job_ran_year(lg, 2015) is None and RG.job_ran_year(lg, 2016) is None
        assert RG.job_ran_year(lg, 2017) == ("this job simulated 2015-01-01 to 2017-01-01, "
                                             "not all of 2017: the year's files are not all "
                                             "its own")
        assert RG.job_ran_year(dict(lg, model_start=None), 2016).startswith(
            "the coupler log gives no model dates")
        assert RG.job_ran_year(dict(lg, successful_termination=False), 2017) is None


# ── one year's model facts ───────────────────────────────────────────────────
def _veg(g):
    f = cf.surface_fields(g)
    return f["PCT_NATVEG"] >= 100.0


def _daily_means(stream, var_parts, year, scale):
    """Vegetated-cell means per day, written out from the formulas."""
    g = stream.grid
    w = np.where(_veg(fx.grid()), g.weight_km2, 0.0)
    nlat, nlon = g.shape
    out = []
    for d in range(1, 366):
        num = den = 0.0
        for j in range(nlat):
            for i in range(nlon):
                if w[j, i] == 0:
                    continue
                v = sum(fx.expected(p, year, d, j, i, Y0, nlon) for p in var_parts)
                if math.isfinite(v):
                    num += w[j, i] * v * scale
                    den += w[j, i]
        out.append(num / den)
    return np.array(out)


class TestYearCheck:

    def test_numbers_match_the_formulas(self, case):
        y = RG.year_check(case["run_dir"], case["case_dir"], 2016, limits=LIMITS)
        assert y["pass"], y["reasons"]
        s = G.open_stream(case["run_dir"], "h1")
        gpp = _daily_means(s, ["GPP"], 2016, 86400.0)
        lai = _daily_means(s, ["TLAI"], 2016, 1.0)
        p = _daily_means(s, ["RAIN", "SNOW"], 2016, 86400.0)
        et = _daily_means(s, ["QVEGT", "QVEGE", "QSOIL"], 2016, 86400.0)
        q = _daily_means(s, ["QRUNOFF"], 2016, 86400.0)
        tws = _daily_means(s, ["TWS"], 2016, 1.0)
        snow = _daily_means(s, ["H2OSNO"], 2016, 1.0)
        eco = _daily_means(s, ["TOTECOSYSC"], 2016, 1.0)
        assert y["gpp_gC_m2_yr"] == pytest.approx(gpp.sum(), rel=1e-10)
        assert y["peak_tlai"] == pytest.approx(lai.max(), rel=1e-10)
        assert y["precip_mm"] == pytest.approx(p.sum(), rel=1e-10)
        assert y["et_mm"] == pytest.approx(et.sum(), rel=1e-10)
        assert y["runoff_mm"] == pytest.approx(q.sum(), rel=1e-10)
        assert y["tws_change_mm"] == pytest.approx(tws[-1] - tws[0], rel=1e-9)
        closure = ((p - et - q)[1:].sum() - (tws[-1] - tws[0])) / p.sum()
        assert y["water_closure_frac_of_precip"] == pytest.approx(closure, rel=1e-8)
        assert y["swe_apr1_mm"] == pytest.approx(snow[90], rel=1e-10)
        assert y["totecosysc_change_frac"] == pytest.approx((eco[-1] - eco[0]) / eco[0],
                                                            rel=1e-8)
        assert (y["days"], y["cells"], y["nan_cells"]) == (365, 18, 0)
        assert set(y["nan_cells_by_field"].values()) == {0}
        assert y["restart_next_year"]
        assert {k: (v["records"], v["missing_fields"], v["ok"]) for k, v in y["tapes"].items()} \
            == {"h0": (12, [], True), "h1": (365, [], True), "h2": (2920, [], True)}

    def test_cells_missing_values_fail_the_year(self, case):
        y = RG.year_check(case["run_dir"], case["case_dir"], 2015, limits=LIMITS)
        assert y["nan_cells"] == 3 and not y["pass"]
        assert y["reasons"] == ["3 vegetated cell-fields miss values ({'GPP': 2, 'ET': 1})"]

    def test_every_field_read_counts_its_missing_cells(self, tmp_path):
        """record 40 of the 2016 file is 2016-02-09; (1,2), (2,3), (3,4) and
        (1,3) are vegetated cells."""
        c = _fresh(tmp_path)
        p = c["run_dir"] / "synth.elm.h1.2016-01-01-00000.nc"
        _edit(p, "TLAI", lambda a: _at(a, (40, 1, 2), fx.FILL))
        _edit(p, "QRUNOFF", lambda a: _at(a, (40, 2, 3), fx.FILL))
        _edit(p, "TWS", lambda a: _at(a, (40, 3, 4), fx.FILL))
        _edit(p, "H2OSNO", lambda a: _at(a, (40, 1, 3), fx.FILL))
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert y["nan_cells"] == 4
        assert y["reasons"] == ["4 vegetated cell-fields miss values ({'TLAI': 1, 'QRUNOFF': 1, "
                                "'TWS': 1, 'H2OSNO': 1})"]

    def test_a_carbon_stock_lost_part_way_through_the_year(self, tmp_path):
        """The means are over the cells with values, so without the count a
        cell dropping out in July would only move the stock change."""
        c = _fresh(tmp_path)
        p = c["run_dir"] / "synth.elm.h1.2017-01-01-00000.nc"     # holds 31 Dec 2016
        _edit(p, "TOTECOSYSC", lambda a: _at(a, (0, 2, 3), fx.FILL))
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert y["nan_cells_by_field"]["TOTECOSYSC"] == 1 and not y["pass"]

    def test_precipitation_missing_everywhere_one_day(self, tmp_path):
        """A NaN closure must fail: abs(nan) >= limit is False."""
        c = _fresh(tmp_path)
        p = c["run_dir"] / "synth.elm.h1.2016-01-01-00000.nc"
        _edit(p, "RAIN", lambda a: _at(a, 60, fx.FILL))
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert y["precip_mm"] is None and y["water_closure_frac_of_precip"] is None
        assert "water does not close: n/a of precipitation (a field it is computed from " \
               "misses values)" in y["reasons"]
        assert y["nan_cells_by_field"]["P"] == 18 and not y["pass"]

    def test_the_default_limits_catch_water_that_does_not_close(self, case):
        y = RG.year_check(case["run_dir"], case["case_dir"], 2016)
        assert not y["pass"] and y["reasons"][0].startswith("water does not close")

    def test_bounds_on_gpp_lai_and_the_carbon_stock(self, case):
        y = RG.year_check(case["run_dir"], case["case_dir"], 2016,
                          limits=RG.YearLimits(gpp_gC_m2_yr=(1000.0, 3000.0),
                                               peak_tlai=(3.0, 15.0),
                                               water_closure_frac_max=10.0))
        assert [r.split(" ")[0] for r in y["reasons"]] == ["GPP", "peak"]
        y = RG.year_check(case["run_dir"], case["case_dir"], 2016,
                          limits=RG.YearLimits(water_closure_frac_max=10.0,
                                               carbon_stock_change_frac_max=0.005))
        cs = y["totecosysc_change_frac"]
        assert 0.005 < cs < 0.05
        assert y["reasons"] == [f"the ecosystem carbon stock changed by {cs:.4g} in a year"]

    def test_a_missing_restart_file_and_a_short_year(self, tmp_path):
        c = _fresh(tmp_path, restarts=False)
        (c["run_dir"] / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert not y["pass"] and not y["restart_next_year"]
        assert any(r.startswith("no restart for 1 January 2017") for r in y["reasons"])
        assert any("h1: stream h1, 2016: 1 of 365" in r for r in y["reasons"])
        assert any(r.startswith("year numbers:") for r in y["reasons"])

    def test_a_requested_field_the_tape_lacks(self, tmp_path):
        c = _fresh(tmp_path)
        p = _lnd_in(c)
        p.write_text(p.read_text().replace("hist_fincl3 = 'TSA','CELLID'",
                                           "hist_fincl3 = 'TSA','CELLID','NOPE'"))
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert "h2 lacks requested fields ['NOPE']" in y["reasons"]

    def test_a_requested_tape_that_was_never_written(self, tmp_path):
        c = _fresh(tmp_path)
        p = _lnd_in(c)
        p.write_text(p.read_text().replace(" hist_mfilt", " hist_fincl4 = 'TSA'\n hist_mfilt"))
        y = RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)
        assert y["tapes"]["h3"]["ok"] is False
        assert len(y["reasons"]) == 1 and y["reasons"][0].startswith("h3: no h3 history in ")

    def test_fields_are_looked_for_in_the_years_own_files(self, tmp_path):
        """A field the first file lacks but the year's files carry (added at a
        new segment) passes; one a year's file lacks is named with the file."""
        import netCDF4
        c = _fresh(tmp_path, years=(2015, 2016, 2017))
        with netCDF4.Dataset(c["run_dir"] / "synth.elm.h1.2015-01-01-00000.nc", "a") as d:
            d.renameVariable("BTRAN", "BTRAN_OLD")
        p = _lnd_in(c)
        p.write_text(p.read_text().replace("'LEAFC'\n", "'LEAFC','BTRAN'\n"))
        assert RG.year_check(c["run_dir"], c["case_dir"], 2016, limits=LIMITS)["pass"]
        y = RG.year_check(c["run_dir"], c["case_dir"], 2015, limits=LIMITS)
        assert "h1 lacks requested fields ['BTRAN'] in some of its files (first " \
               "synth.elm.h1.2015-01-01-00000.nc)" in y["reasons"]


# ── a warm start's first day ─────────────────────────────────────────────────
class TestCarryOver:

    def test_a_run_that_started_from_its_restart(self, case):
        c = RG.carry_over(case["run_dir"], case["case_dir"])
        assert c["pass"] and c["first_day"] == "2015-01-01"
        for k in ("TOTSOMC", "TOTVEGC", "LEAFC", "TLAI"):
            f = c["fields"][k]
            assert f["median_rel_diff"] < 1e-6 and f["cells"] == 18 and f["history_nan_cells"] == 0

    def test_a_run_that_did_not(self, tmp_path, case):
        other = cf.write_restart(tmp_path / "doubled.nc", scale=2.0)
        c = RG.carry_over(case["run_dir"], restart=other, surface=case["surface"])
        assert not c["pass"]
        assert [r.split(" ")[0] for r in c["reasons"]] == ["TOTSOMC", "LEAFC"]
        assert c["fields"]["TOTSOMC"]["median_rel_diff"] == pytest.approx(0.5, rel=1e-6)

    def test_a_first_day_missing_at_a_vegetated_cell(self, tmp_path):
        c = _fresh(tmp_path)
        p = c["run_dir"] / "synth.elm.h1.2015-01-01-00000.nc"
        _edit(p, "LEAFC", lambda a: _at(a, (1, 1, 2), fx.FILL))   # record 1 = 2015-01-01
        r = RG.carry_over(c["run_dir"], c["case_dir"])
        assert r["fields"]["LEAFC"]["history_nan_cells"] == 1
        assert r["fields"]["LEAFC"]["median_rel_diff"] < 1e-6
        assert r["reasons"] == ["LEAFC: 1 vegetated cells miss the first day"]

    def test_a_gated_field_the_restart_lacks(self, tmp_path, case):
        rest = cf.write_restart(tmp_path / "noleafc.nc", edit=lambda t: t.pop("leafc"))
        r = RG.carry_over(case["run_dir"], restart=rest, surface=case["surface"])
        assert r["fields"]["LEAFC"] == {"skipped": "restart has no leafc"}
        assert not r["pass"] and r["reasons"] == ["LEAFC: restart has no leafc"]

    def test_no_vegetated_cell_to_compare(self, tmp_path, case):
        surf = cf.write_surface(tmp_path / "s.nc",
                                edit=lambda f: f["PCT_NATVEG"].__setitem__(slice(None), 99.0))
        r = RG.carry_over(case["run_dir"], restart=case["restart"], surface=surf)
        assert r["fields"]["LEAFC"]["cells"] == 0
        assert r["reasons"] == [
            "TOTSOMC: no vegetated cell has both a first-day value and a saved one",
            "LEAFC: no vegetated cell has both a first-day value and a saved one"]

    def test_a_cold_start_has_nothing_to_carry_over(self, tmp_path):
        c = _fresh(tmp_path)
        p = _lnd_in(c)
        p.write_text("\n".join(l for l in p.read_text().splitlines() if "finidat" not in l))
        with pytest.raises(RG.GateError, match="no finidat"):
            RG.carry_over(c["run_dir"], c["case_dir"])


# ── a year against a reference run ───────────────────────────────────────────
def _reference(tmp_path, case, scale=None, rename=None, drop_last=False, shift=0.0,
               roll=None, setto=None, area_cell=None):
    import netCDF4
    ref = tmp_path / "ref"
    ref.mkdir()
    for p in sorted(case["run_dir"].glob("synth.elm.h1.*.nc")):
        q = ref / p.name.replace("synth", "other")
        shutil.copy(p, q)
        with netCDF4.Dataset(q, "a") as d:
            for var, k in (scale or {}).items():
                v = d[var]
                v.set_auto_maskandscale(False)
                a = np.asarray(v[:])
                v[:] = np.where(np.abs(a) < 1e30, a * k, a)
            for var, n in (roll or {}).items():
                v = d[var]
                v.set_auto_maskandscale(False)
                v[:] = np.roll(np.asarray(v[:]), n, axis=0)
            for var, x in (setto or {}).items():
                v = d[var]
                v.set_auto_maskandscale(False)
                v[:] = np.full(v.shape, x, dtype=v.dtype)
            for old, new in (rename or {}).items():
                d.renameVariable(old, new)
            if area_cell is not None:          # another source's area, off at one cell
                v = d["area"]
                v.set_auto_maskandscale(False)
                v[:] = _at(np.asarray(v[:]), area_cell, 2.0 * np.asarray(v[:])[area_cell])
            if shift:
                v = d["lat"]
                v.set_auto_maskandscale(False)
                v[:] = np.asarray(v[:], dtype=np.float64) + shift
    if drop_last:
        sorted(ref.glob("*.nc"))[-1].unlink()
    return G.open_stream(ref, "h1")


class TestReferenceGate:

    def test_a_small_bias_passes_and_is_measured(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, scale={"GPP": 1 / 1.02}),
                              2016, case["surface"])
        assert r["pass"] and r["days_compared"] == 365
        assert r["fields"]["GPP"]["bias_pct"] == pytest.approx(2.0, abs=1e-4)
        assert r["fields"]["GPP"]["daily_r"] == pytest.approx(1.0, abs=1e-9)
        assert r["fields"]["TSA"]["bias_pct"] == pytest.approx(0.0, abs=1e-9)
        assert r["fields"]["NPP"] == {"missing": "ours", "ok": None}
        a = r["annual_gpp_gC_m2"]
        assert a["ours"] == pytest.approx(1.02 * a["reference"], rel=1e-5)
        assert set(r["bands"]) == {"below 1000 m", "1000-1500 m", "1500-2000 m"}

    def test_a_large_bias_fails(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, scale={"GPP": 1 / 1.05}),
                              2016, case["surface"])
        assert not r["pass"] and r["reasons"] == [
            "GPP: bias +5.00 % (limit 3.0), r 1.000 (limit 0.99)"]

    def test_a_field_out_of_phase_fails_on_its_correlation_alone(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, roll={"GPP": 91}), 2016,
                              case["surface"])
        f = r["fields"]["GPP"]
        assert abs(f["bias_pct"]) < 3.0 and f["daily_r"] < 0.5 and f["ok"] is False
        assert len(r["reasons"]) == 1 and r["reasons"][0].startswith("GPP: bias ")
        assert r["reasons"][0].endswith(f"r {f['daily_r']:.3f} (limit 0.99)")

    def test_a_field_zero_in_both_runs_agrees(self, tmp_path, case):
        """No snow in either run: no percentage and no correlation exist,
        and none are needed."""
        import netCDF4
        for p in case["run_dir"].glob("synth.elm.h1.*.nc"):
            shutil.copy(p, tmp_path / p.name)
        for p in tmp_path.glob("synth.elm.h1.*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["H2OSNO"][:] = 0.0
        ours = G.open_stream(tmp_path, "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, setto={"H2OSNO": 0.0}), 2016,
                              case["surface"])
        h = r["fields"]["H2OSNO"]
        assert r["pass"], r["reasons"]
        assert (h["bias_pct"], h["daily_r"], h["ok"]) == (0.0, None, True)
        assert h["notes"] == ["both series constant and equal: no correlation, and none needed"]

    def test_a_reference_without_snow_against_ours_with(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, setto={"H2OSNO": 0.0}), 2016,
                              case["surface"])
        h = r["fields"]["H2OSNO"]
        assert h["bias_pct"] is None and h["daily_r"] is None and h["ok"] is False
        assert r["reasons"] == [
            "H2OSNO: bias undefined (limit 5.0), r undefined (limit 0.99); the reference mean "
            "is 0, ours is not: no percentage bias; the reference is constant: no correlation"]

    def test_a_gated_field_outside_the_usual_list_is_compared(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        limits = RG.ReferenceLimits(fields=RG.ReferenceLimits().fields + (("QRUNOFF", 1.0, 0.99),))
        r = RG.reference_gate(ours, _reference(tmp_path, case, scale={"QRUNOFF": 0.5}), 2016,
                              case["surface"], limits=limits)
        assert r["fields"]["QRUNOFF"]["bias_pct"] == pytest.approx(100.0, rel=1e-6)
        assert r["reasons"] == ["QRUNOFF: bias +100.00 % (limit 1.0), r 1.000 (limit 0.99)"]

    def test_a_gated_field_missing_from_the_reference_fails(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        r = RG.reference_gate(ours, _reference(tmp_path, case, rename={"TSA": "TSA_X"}),
                              2016, case["surface"])
        assert not r["pass"] and "TSA missing from reference" in r["reasons"]
        assert r["fields"]["TSA"] == {"missing": "reference", "ok": False}

    def test_too_few_common_days(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        ref = _reference(tmp_path, case, drop_last=True)
        r = RG.reference_gate(ours, ref, 2016, case["surface"])
        assert r["days_compared"] == 364 and r["pass"]
        r = RG.reference_gate(ours, ref, 2016, case["surface"],
                              limits=RG.ReferenceLimits(min_days=365))
        assert not r["pass"] and r["reasons"] == ["only 364 common days (at least 365)"]

    def test_the_reference_is_weighted_by_our_cells_area(self, tmp_path, case):
        """The same run with another area on file: weighted by its own area its
        means would move; weighted by ours they are ours to the last bit."""
        ours = G.open_stream(case["run_dir"], "h1")
        ref = _reference(tmp_path, case, area_cell=(1, 2))
        assert not np.array_equal(ref.grid.weight_km2, ours.grid.weight_km2)
        r = RG.reference_gate(ours, ref, 2016, case["surface"])
        for f, v in r["fields"].items():
            if "missing" not in v:
                assert v["ours"] == v["reference"] and v["bias_pct"] == 0.0, f

    def test_float32_centres_count_as_the_same_cells(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        ref = _reference(tmp_path, case, shift=5e-5)
        assert not ours.grid.same_cells(ref.grid)                 # not at 1e-9
        r = RG.reference_gate(ours, ref, 2016, case["surface"])
        assert r["pass"] and r["days_compared"] == 365

    def test_a_reference_on_other_cells_is_refused(self, tmp_path, case):
        ours = G.open_stream(case["run_dir"], "h1")
        with pytest.raises(RG.GateError, match="other cells"):
            RG.reference_gate(ours, _reference(tmp_path, case, shift=0.01), 2016,
                              case["surface"])


# ── the command line ─────────────────────────────────────────────────────────
def _smoke_run(tmp_path, days=5, h1_fields=cf.DAILY_FIELDS, job="333"):
    c = _fresh(tmp_path / "x")
    run = tmp_path / "smoke" / "run"
    run.mkdir(parents=True)
    g = fx.grid()
    fx._write(run / "s.elm.h1.2015-01-01-00000.nc", g, 2015,
              [(-fx.DT, 0.0)] + [(k, k + 1.0) for k in range(days)],
              [None] + [(2015, k + 1, 0) for k in range(days)], list(h1_fields))
    fx._write(run / "s.elm.h2.2015-01-01-00000.nc", g, 2015,
              [(-fx.DT, 0.0)] + [(k / 8, (k + 1) / 8) for k in range(8 * days)],
              [None] + [(2015, k // 8 + 1, k % 8) for k in range(8 * days)],
              ["STEPCODE", "CELLID", "TSA"], steps_per_day=8)
    cf.write_logs(run, c["case_dir"], "s", job, start=20150101, end=20150101 + days,
                  days=float(days))
    return c, run


def _main(*args):
    return RG.main([str(a) for a in args])


class TestCommandLine:

    def test_a_year_writes_its_record_and_status_line(self, tmp_path, case):
        out, status = tmp_path / "checks", tmp_path / "STATUS.log"
        code = _main("year", "--case-dir", case["case_dir"], "--run-dir", case["run_dir"],
                     "--job", "111", "--year", "2016", "--out", out, "--status", status,
                     "--stage", "y2016")
        rec = json.loads((out / "y2016_111.json").read_text())
        assert code == 1 and rec["pass"] is False      # default limits: no closure
        assert rec["logs"]["pass"] and rec["years"]["2016"]["gpp_gC_m2_yr"] > 0
        assert rec["reasons"][0].startswith("2016: water does not close")
        line = status.read_text().strip()
        assert "job=111 stage=y2016 CHECK FAIL hours_per_sim_year=0.795" in line
        assert line.endswith("(y2016_111.json)")

    def test_every_year_given_is_checked_and_tied_to_the_job(self, tmp_path, case):
        out = tmp_path / "checks"
        _main("year", "--case-dir", case["case_dir"], "--run-dir", case["run_dir"],
              "--job", "111", "--year", "2015", "--year", "2016", "--year", "2017", "--out", out)
        rec = json.loads((out / "year_111.json").read_text())
        assert sorted(rec["years"]) == ["2015", "2016", "2017"]
        assert "2015: 3 vegetated cell-fields miss values ({'GPP': 2, 'ET': 1})" in rec["reasons"]
        assert "2017: this job simulated 2015-01-01 to 2017-01-01, not all of 2017: the " \
               "year's files are not all its own" in rec["reasons"]

    def test_a_smoke_run(self, tmp_path):
        c, run = _smoke_run(tmp_path)
        out, status = tmp_path / "checks", tmp_path / "STATUS.log"
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", out, "--status", status)
        rec = json.loads((out / "smoke_333.json").read_text())
        assert code == 0 and rec["pass"], rec["reasons"]
        assert rec["daily_records"] == 5
        assert rec["missing_fields"] == {"h0": "no file expected in a 5-day run",
                                         "h1": [], "h2": []}
        assert rec["carry_over"]["pass"]
        assert "job=333 stage=smoke CHECK PASS" in status.read_text()
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", out, "--days", "6")
        assert code == 1

    def test_a_smoke_run_missing_a_requested_field(self, tmp_path):
        c, run = _smoke_run(tmp_path)
        p = _lnd_in(c)
        p.write_text(p.read_text().replace("'LEAFC'\n", "'LEAFC','NOPE'\n"))
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", tmp_path / "checks")
        rec = json.loads((tmp_path / "checks" / "smoke_333.json").read_text())
        assert code == 1 and rec["missing_fields"]["h1"] == ["NOPE"]
        assert rec["reasons"] == ["h1 lacks ['NOPE']"]

    def test_a_smoke_run_whose_namelist_lists_one_frequency(self, tmp_path):
        """hist_nhtfrq = 0 alone leaves h1 and h2 daily (ELM's default), so
        their files are expected and read."""
        c, run = _smoke_run(tmp_path)
        p = _lnd_in(c)
        p.write_text(p.read_text().replace(" hist_nhtfrq = 0,-24,-3\n", " hist_nhtfrq = 0\n")
                     .replace("'LEAFC'\n", "'LEAFC','NOPE'\n"))
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", tmp_path / "checks")
        rec = json.loads((tmp_path / "checks" / "smoke_333.json").read_text())
        assert code == 1 and rec["missing_fields"]["h1"] == ["NOPE"]

    def test_a_30_day_smoke_run_writes_no_monthly_file(self, tmp_path):
        """January has 31 days: ELM opens the first monthly file only then."""
        c, run = _smoke_run(tmp_path, days=30)
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", tmp_path / "checks", "--days", "30")
        rec = json.loads((tmp_path / "checks" / "smoke_333.json").read_text())
        assert code == 0, rec["reasons"]
        assert rec["missing_fields"]["h0"] == "no file expected in a 30-day run"

    def test_a_smoke_job_that_died_still_leaves_its_record(self, tmp_path):
        c = _fresh(tmp_path / "x")
        run = tmp_path / "dead" / "run"
        run.mkdir(parents=True)
        cf.write_logs(run, c["case_dir"], "s", "444", ok=False,
                      lnd_lines=[" ENDRUN:", " ERROR in surfrdMod.F90 at line 650",
                                 " ERROR: Unknown error submitted to shr_abort_abort."])
        out, status = tmp_path / "checks", tmp_path / "STATUS.log"
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "444",
                     "--out", out, "--status", status)
        rec = json.loads((out / "smoke_444.json").read_text())
        assert code == 1 and "job=444 stage=smoke CHECK FAIL" in status.read_text()
        assert rec["reasons"][:2] == ["the coupler log has no SUCCESSFUL TERMINATION (last "
                                      "model date 2015-01-01)", "the land log stops the run: "
                                      "ENDRUN: ERROR in surfrdMod.F90 at line 650 ERROR: "
                                      "Unknown error submitted to shr_abort_abort."]
        assert rec["reasons"][2].startswith("no daily history to check: ")

    def test_a_cold_start_smoke_says_why_there_is_no_carry_over(self, tmp_path):
        c, run = _smoke_run(tmp_path)
        p = _lnd_in(c)
        p.write_text("\n".join(l for l in p.read_text().splitlines() if "finidat" not in l))
        code = _main("smoke", "--case-dir", c["case_dir"], "--run-dir", run, "--job", "333",
                     "--out", tmp_path / "checks")
        rec = json.loads((tmp_path / "checks" / "smoke_333.json").read_text())
        assert code == 1 and len(rec["reasons"]) == 1
        assert rec["reasons"][0].startswith("carry-over: ") and "no finidat" in rec["reasons"][0]

    def test_a_folder_with_two_cases_is_a_reason_not_a_crash(self, tmp_path, case):
        c = _fresh(tmp_path)
        shutil.copy(c["run_dir"] / "synth.elm.h1.2016-01-01-00000.nc",
                    c["run_dir"] / "other.elm.h1.2016-01-01-00000.nc")
        code = _main("year", "--case-dir", c["case_dir"], "--run-dir", c["run_dir"], "--job",
                     "111", "--year", "2016", "--out", tmp_path / "checks")
        rec = json.loads((tmp_path / "checks" / "year_111.json").read_text())
        assert code == 1
        assert any(r.startswith("2016: ") and "holds history from 2 cases" in r
                   for r in rec["reasons"]), rec["reasons"]

    def test_records_are_strict_json(self, tmp_path):
        c = _fresh(tmp_path)
        p = c["run_dir"] / "synth.elm.h1.2016-01-01-00000.nc"
        _edit(p, "RAIN", lambda a: _at(a, 60, fx.FILL))
        _main("year", "--case-dir", c["case_dir"], "--run-dir", c["run_dir"], "--job", "111",
              "--year", "2016", "--out", tmp_path / "checks")
        text = (tmp_path / "checks" / "year_111.json").read_text()
        json.loads(text, parse_constant=lambda name: pytest.fail(f"{name} in the record"))

    def test_the_writer_keeps_records_strict_json(self, tmp_path):
        """The gates already store None for a number that does not exist; the
        writer guarantees it for anything that slips through."""
        rec = {"pass": False, "x": float("nan"), "y": [np.float32("inf"), 1.5],
               "z": {"n": np.int64(3), "ok": np.bool_(True)}}
        out = RG._write(rec, tmp_path, "t", "1", None, "")
        assert json.loads(out.read_text()) == {"pass": False, "x": None, "y": [None, 1.5],
                                               "z": {"n": 3, "ok": True}}

    def test_the_job_id_is_required(self, case, monkeypatch, tmp_path):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(SystemExit):
            _main("year", "--case-dir", case["case_dir"], "--run-dir", case["run_dir"],
                  "--year", "2016", "--out", tmp_path)
