"""The input-set check and the run gates against the Naches ELM-BGC runs.

    pytest tests/test_run_gate_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the runs are on disk. Paths come from the
naches_bgc_gates block of tests/local_oracle_paths.json (not tracked).

The expected values are the hand checks' own records, written while the
experiment ran (2026-10-12/14): the pairing record of the input set, the
per-job check records in checks/ and the reference-run records. A pass means
the framework gates reach the same verdicts from the same logs and output,
with the same numbers to the hand records' rounding, including two
negatives: the budget abort (job 788220) and the phenology-bug year (2016 of
the first control run). The input-set layout is also checked against a
global 0.5-degree saved state (the global_r05 block), which has masked
cells, wetland, glacier and cells without urban data.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pytest

import domain_inputs as D
import gridded as G
import run_gate as RG

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
FIX_JOBS = {2015: 789148, 2016: 789149, 2017: 789150, 2018: 789151, 2019: 789152,
            2020: 789153, 2021: 789154}
CTL_JOBS = {788874: [2015], 788875: [2016, 2017, 2018], 788876: [2019, 2020, 2021]}
CTL_STAGE = {788874: "ctl2015", 788875: "ctl1618", 788876: "ctl1921"}


def _block(name):
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}; copy {CONFIG.name}.example and fill it in")
    c = json.loads(CONFIG.read_text()).get(name) or {}
    missing = [k for k, v in c.items() if k != "reference_pattern" and not Path(v).exists()]
    if missing or not c:
        pytest.skip(f"local paths missing: {missing or name + ' block'}")
    return {k: (v if k == "reference_pattern" else Path(v)) for k, v in c.items()}


@pytest.fixture(scope="module")
def cfg():
    return _block("naches_bgc_gates")


def _record(cfg, name):
    return json.loads((cfg["checks_dir"] / name).read_text())


def _near(ours, hand, decimals):
    """ours, unrounded, rounds to the hand's value (half a unit, 1 % spare)."""
    return abs(ours - hand) <= 0.5 * 10.0 ** (-decimals) * 1.01 + 1e-12


# ── the input set ────────────────────────────────────────────────────────────
class TestInputSet:

    def test_the_input_set_the_run_used(self, cfg):
        r = D.check_pairing(cfg["surface"], cfg["domain"], cfg["restart"])
        hand = json.loads(cfg["pairing_record"].read_text())
        c = r["checks"]
        assert r["pass"] is hand["pass"] is True and r["reasons"] == []
        assert c["domain_vs_surface_coord_max_diff_deg"] == \
            hand["domain_vs_surface_coord_max_diff"] == 0.0
        assert c["pft_weight_max_abs_diff"] == hand["pft_wtlunit_max_abs_diff_vs_restart"]
        lu = hand["landunit_weight_max_abs_diff_vs_restart"]
        assert c["landunit_weight_max_abs_diff"] == {
            "natveg": lu["PCT_NATVEG"], "lake": lu["PCT_LAKE"], "urban_tbd": lu["PCT_URBAN0"],
            "urban_hd": lu["PCT_URBAN1"], "urban_md": lu["PCT_URBAN2"]}
        assert c["restart_grid_max_offset_deg"] == hand["restart_coord_offset_deg"]
        assert c["expected_dims"] == hand["expected_dims"]
        assert c["restart_dims"] == hand["restart_dims"]
        assert sum(c["soil_negative_values"].values()) == hand["soil_negative_values"]
        assert c["box_area_km2"] == pytest.approx(hand["box_area_km2"], rel=1e-12)
        # ELM's own layout, record for record, and its percentage sums
        assert c["layout_matches"] is True
        assert c["active_cells"] == c["urban_valid_cells"] == 13824 and c["lake_cells"] == 13
        assert c["pct_nat_pft_sum_max_dev"] == c["landunit_pct_sum_max_dev"] == 0.0

    def test_a_domain_cut_from_another_source_breaks_the_exact_centre_rule(self, cfg):
        """7.1e-6 deg off the surface: ELM would run it (it allows 0.001), so
        the one reason is this check's own rule, and with ELM's rule alone the
        set passes."""
        r = D.check_pairing(cfg["surface"], cfg["band_domain"], cfg["restart"])
        assert len(r["reasons"]) == 1 and r["reasons"][0].startswith(
            "the domain's centres sit 7.12e-06 deg from the surface's: this check requires")
        assert r["checks"]["domain_vs_surface_coord_max_diff_deg"] == pytest.approx(7.12e-6,
                                                                                   abs=5e-9)
        elm_only = D.PairingLimits(coord_max_diff_deg=D.ELM_COORD_MAX_DIFF_DEG)
        assert D.check_pairing(cfg["surface"], cfg["band_domain"], cfg["restart"],
                               limits=elm_only)["pass"]

    def test_elms_layout_on_a_global_grid(self):
        """95,102 active cells of 259,200, wetland and glacier land units, and
        28,084 cells without urban data: the saved state is in ELM's layout
        record for record. Its PFT weights are a transient run's (0.96 off the
        1850 surface), which ELM checks only for a run with fixed vegetation."""
        g = _block("global_r05")
        r = D.check_pairing(g["surface"], g["domain"], g["restart"])
        c = r["checks"]
        assert c["layout_matches"] is True and c["expected_dims"] == c["restart_dims"] == {
            "gridcell": 95102, "landunit": 346660, "column": 1150876, "pft": 2672508}
        assert (c["active_cells"], c["urban_valid_cells"], c["lake_cells"]) == (95102, 67018,
                                                                                17332)
        assert set(c["landunit_weight_max_abs_diff"]) == {
            "natveg", "urban_tbd", "urban_hd", "urban_md", "lake", "wetland", "glacier"}
        assert set(c["landunit_weight_max_abs_diff"].values()) == {0.0}
        assert r["reasons"] == ["ELM stops: PFT weights differ from the saved state by 0.96 "
                                "(subgridRestMod allows 0.005)"]

    def test_the_domain_written_from_the_surface_is_the_one_the_run_used(self, cfg, tmp_path):
        import netCDF4
        out = tmp_path / "domain.nc"
        D.domain_from_surface(cfg["surface"], out)
        with netCDF4.Dataset(out) as a, netCDF4.Dataset(cfg["domain"]) as b:
            for v in ("xc", "yc", "xv", "yv", "area", "mask", "frac"):
                x, y = np.asarray(a[v][:]), np.asarray(b[v][:])
                assert x.dtype == y.dtype and np.array_equal(x, y), v


# ── logs ─────────────────────────────────────────────────────────────────────
LOG_KEYS = ("successful_termination", "error_lines", "abort_lines", "balance_lines",
            "nan_lines", "sim_years_per_day", "init_time_s", "run_time_s")


def _logs_match(ours, hand):
    for k in LOG_KEYS:
        assert ours.get(k) == hand.get(k), k
    assert Path(ours["lnd_log"]).name == Path(hand["lnd_log"]).name
    assert Path(ours["cpl_log"]).name == Path(hand["cpl_log"]).name
    if hand.get("hours_per_sim_year") is not None:
        assert _near(ours["hours_per_sim_year"], hand["hours_per_sim_year"], 3)


class TestLogs:

    @pytest.mark.parametrize("year,job", sorted(FIX_JOBS.items()))
    def test_each_production_year(self, cfg, year, job):
        ours = RG.scan_logs(cfg["fix_run_dir"], cfg["fix_case_dir"], job)
        _logs_match(ours, _record(cfg, f"nbgc_y{year}_{job}.json")["logs"])
        assert ours["pass"]
        assert (ours["model_start"], ours["model_end"]) == (f"{year}-01-01", f"{year + 1}-01-01")
        assert ours["simulated_days"] == 365.0
        assert RG.job_ran_year(ours, year) is None and RG.job_ran_year(ours, year + 1)

    @pytest.mark.parametrize("job", sorted(CTL_JOBS))
    def test_each_job_of_the_first_control_run(self, cfg, job):
        ours = RG.scan_logs(cfg["ctl_run_dir"], cfg["ctl_case_dir"], job)
        _logs_match(ours, _record(cfg, f"nbgc_{CTL_STAGE[job]}_{job}.json")["logs"])

    def test_the_budget_abort_is_refused(self, cfg):
        ours = RG.scan_logs(cfg["rep_run_dir"], cfg["rep_case_dir"], 788220)
        hand = _record(cfg, "nbgc_rep2015_788220.json")["logs"]
        assert ours["abort_lines"] == hand["abort_lines"] == ["ENDRUN:"]
        assert ours["successful_termination"] is hand["successful_termination"] is False
        assert not ours["pass"] and len(ours["reasons"]) == 2
        assert ours["reasons"][1].startswith("the land log stops the run: ENDRUN: ERROR in "
                                             "CNPBudgetMod.F90 at line 929")
        assert ours["model_last"] == "2015-12-31" and ours["model_end"] is None
        # where the old checker's newest-file lookup did bite: this job wrote no
        # timing file, and its hand record carries the smoke job's throughput
        assert ours["timing_file"] is None and ours.get("sim_years_per_day") is None
        assert hand["sim_years_per_day"] == 25.74

    def test_the_smoke_job(self, cfg):
        """The record's balance_lines (6) came from the rule of the hand
        checker at the time (any line with 'balance' and 'warn' or 'error'),
        applied to this job's own log: two CMASS_BALANCE_ERROR field-name lines
        and four first-step notices. The later rule leaves those out, so the
        count is 0 here and the rest of the record must match."""
        ours = RG.scan_logs(cfg["rep_run_dir"], cfg["rep_case_dir"], 788219)
        hand = dict(_record(cfg, "nbgc_smoke_788219.json")["logs"])
        assert Path(ours["lnd_log"]).name == Path(hand["lnd_log"]).name
        lines = RG._read_text(ours["lnd_log"]).splitlines()
        assert sum(1 for l in lines if re.search("balance", l, re.I)
                   and re.search("warn|error", l, re.I)) == 6
        assert hand.pop("balance_lines") == 6 and ours["balance_lines"] == 0
        hand["balance_lines"] = 0
        _logs_match(ours, hand)
        assert ours["pass"]


# ── one year's model facts ───────────────────────────────────────────────────
ROUNDING = {"gpp_gC_m2_yr": 1, "peak_tlai": 3, "precip_mm": 1, "et_mm": 1, "runoff_mm": 1,
            "tws_change_mm": 1, "water_closure_frac_of_precip": 4, "swe_apr1_mm": 1,
            "totecosysc_change_frac": 5}


def _year_matches(ours, hand):
    for k, n in ROUNDING.items():
        assert _near(ours[k], hand[k], n), (k, ours[k], hand[k])
    for k in ("days", "cells", "nan_cells", "restart_next_year"):
        assert ours[k] == hand[k], k
    for tape in ("h0", "h1", "h2"):
        assert ours["tapes"][tape]["records"] == hand[tape]["records"], tape
        assert ours["tapes"][tape]["missing_fields"] == hand[tape]["missing_fields"], tape
    assert ours["pass"] is hand["ok"]


class TestYears:

    @pytest.mark.parametrize("year", sorted(FIX_JOBS))
    def test_each_production_year(self, cfg, year):
        hand = _record(cfg, f"nbgc_y{year}_{FIX_JOBS[year]}.json")["years"][str(year)]
        _year_matches(RG.year_check(cfg["fix_run_dir"], cfg["fix_case_dir"], year), hand)

    @pytest.mark.parametrize("job,year", [(j, y) for j, ys in sorted(CTL_JOBS.items())
                                          for y in ys])
    def test_each_year_of_the_first_control_run(self, cfg, job, year):
        hand = _record(cfg, f"nbgc_{CTL_STAGE[job]}_{job}.json")["years"][str(year)]
        _year_matches(RG.year_check(cfg["ctl_run_dir"], cfg["ctl_case_dir"], year), hand)


# ── the warm start ───────────────────────────────────────────────────────────
class TestCarryOver:

    def test_the_smoke_run_started_from_its_saved_state(self, cfg):
        ours = RG.carry_over(cfg["rep_run_dir"], cfg["rep_case_dir"])
        hand = _record(cfg, "nbgc_smoke_788219.json")["carry_over"]
        assert ours["pass"] and ours["first_day"] == "2015-01-01"
        for k, h in hand.items():
            for key in ("median_rel_diff", "p99_rel_diff", "domain_mean_history",
                        "domain_mean_restart"):
                assert ours["fields"][k][key] == pytest.approx(h[key], rel=1e-9), (k, key)
            assert ours["fields"][k]["cells"] == h["cells"]
            assert ours["fields"][k]["history_nan_cells"] == h["history_nan_cells"]

    def test_the_smoke_command_line(self, cfg, tmp_path):
        """The smoke job's own 5-day history was overwritten in the same folder
        by job 788220's year attempt, which ran to 31 December 2015 and
        aborted there: the 31 December mean would have been the first record
        of a 2016 file, never written, so 364 daily records are left (the
        first day is the same, so the carry-over above still matches). The
        command line therefore refuses the 5-day count, and passes on the 364
        days that are there, against the smoke job's logs."""
        code = RG.main(["smoke", "--case-dir", str(cfg["rep_case_dir"]), "--run-dir",
                        str(cfg["rep_run_dir"]), "--job", "788219", "--out", str(tmp_path)])
        rec = json.loads((tmp_path / "smoke_788219.json").read_text())
        assert code == 1 and rec["reasons"] == ["364 daily records, not 5"]
        code = RG.main(["smoke", "--case-dir", str(cfg["rep_case_dir"]), "--run-dir",
                        str(cfg["rep_run_dir"]), "--job", "788219", "--out", str(tmp_path),
                        "--days", "364"])
        rec = json.loads((tmp_path / "smoke_788219.json").read_text())
        assert code == 0 and rec["pass"], rec["reasons"]
        assert rec["missing_fields"] == {"h0": [], "h1": [], "h2": []}
        assert rec["carry_over"]["pass"] and rec["logs"]["pass"]


class TestCommandLine:

    def test_a_production_year_from_the_command_line(self, cfg, tmp_path):
        code = RG.main(["year", "--case-dir", str(cfg["fix_case_dir"]), "--run-dir",
                        str(cfg["fix_run_dir"]), "--job", "789149", "--year", "2016",
                        "--out", str(tmp_path)])
        rec = json.loads((tmp_path / "year_789149.json").read_text())
        assert code == 0 and rec["pass"], rec["reasons"]
        hand = _record(cfg, "nbgc_y2016_789149.json")["years"]["2016"]
        _year_matches(rec["years"]["2016"], hand)

    def test_a_year_this_job_did_not_run(self, cfg, tmp_path):
        """Job 789149 ran 2016; its logs cannot vouch for the 2017 files."""
        code = RG.main(["year", "--case-dir", str(cfg["fix_case_dir"]), "--run-dir",
                        str(cfg["fix_run_dir"]), "--job", "789149", "--year", "2017",
                        "--out", str(tmp_path)])
        rec = json.loads((tmp_path / "year_789149.json").read_text())
        assert code == 1 and rec["reasons"] == [
            "2017: this job simulated 2016-01-01 to 2017-01-01, not all of 2017: the year's "
            "files are not all its own"]


# ── against the reference run ────────────────────────────────────────────────
BAND = {"0-1000m": "below 1000 m", "1000-1500m": "1000-1500 m",
        "1500-2000m": "1500-2000 m", "2000-3000m": "2000 m and above"}


@pytest.fixture(scope="module")
def reference(cfg):
    return G.Stream("reference", sorted(glob.glob(str(cfg["reference_dir"] /
                                                      cfg["reference_pattern"]))))


class TestReference:

    @pytest.mark.parametrize("year", sorted(FIX_JOBS))
    def test_each_production_year(self, cfg, reference, year):
        ours = G.open_stream(cfg["fix_run_dir"], "h1")
        r = RG.reference_gate(ours, reference, year, cfg["surface"])
        hand = _record(cfg, f"replication_naches_bgc_fix_{year}_{FIX_JOBS[year]}.json")
        assert r["days_compared"] == hand["days_compared"]
        assert r["pass"] is hand["pass"]
        for f, h in hand["fields"].items():
            o = r["fields"][f]
            if "missing" in h:
                assert o.get("missing") in ("reference", "ours"), f
                continue
            assert _near(o["bias_pct"], h["bias_pct"], 3), (f, o["bias_pct"], h["bias_pct"])
            assert _near(o["daily_r"], h["daily_r"], 5), (f, o["daily_r"], h["daily_r"])
            assert o["ok"] == h["ok"], f
            per_s = 86400.0 if o["units"].endswith("/d") else 1.0
            assert o["ours"] / per_s == pytest.approx(h["ours"], rel=1e-9), f
            assert o["reference"] / per_s == pytest.approx(h["hers"], rel=1e-9), f
        assert _near(r["annual_gpp_gC_m2"]["ours"], hand["annual_gpp_gC_m2"]["ours"], 1)
        assert _near(r["annual_gpp_gC_m2"]["reference"], hand["annual_gpp_gC_m2"]["hers"], 1)
        for hb, ob in BAND.items():
            h, o = hand["bands"][hb], r["bands"][ob]
            assert o["cells"] == h["cells"]
            assert _near(o["gpp_ours"], h["gpp_ours"], 1) and \
                _near(o["gpp_reference"], h["gpp_hers"], 1)
            assert _near(o["swe_peak_ours"], h["swe_peak_ours"], 1) and \
                _near(o["swe_peak_reference"], h["swe_peak_hers"], 1)

    def test_the_phenology_bug_year_fails(self, cfg, reference):
        """Recorded at the time against 364 reference days (the next year's
        reference file was not yet copied), so the last digits differ."""
        ours = G.open_stream(cfg["ctl_run_dir"], "h1")
        r = RG.reference_gate(ours, reference, 2016, cfg["surface"])
        hand = _record(cfg, "replication_naches_bgc_ctl_2016_788875.json")
        assert not r["pass"] and not hand["pass"]
        assert r["fields"]["GPP"]["bias_pct"] == pytest.approx(hand["fields"]["GPP"]["bias_pct"],
                                                               abs=0.01)
        assert r["fields"]["TLAI"]["daily_r"] == pytest.approx(
            hand["fields"]["TLAI"]["daily_r"], abs=0.002)
        for f, h in hand["fields"].items():
            if "missing" not in h:
                assert r["fields"][f]["ok"] == h["ok"], f
        # TOTSOMC fails on its correlation alone (bias -0.007 %, r 0.975)
        assert [x.split(":")[0] for x in r["reasons"]] == ["GPP", "TLAI", "LEAFC", "TOTSOMC"]
        t = r["fields"]["TOTSOMC"]
        assert abs(t["bias_pct"]) < 1.0 and t["daily_r"] < 0.99
