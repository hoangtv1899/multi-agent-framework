"""The whole-domain case and its chain against the Naches runs.

    pytest tests/test_domain_case_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the runs are on disk. Paths come from the
naches_bgc_gates and naches_bgc_fw blocks of tests/local_oracle_paths.json
(not tracked).

The proof run: the Naches production case rebuilt from its own records as a
new case, through the server's tools (read_domain_recipe, then
submit_domain_chain on a run directory holding the recipe and the chain
spec), then case > smoke > 2015 > 2016 > report on the short queue, each year
gated against the production run. With the same source tree, SourceMods,
inputs and processor layout, the new run must match the production run to
the last bit: the same history values, the same year numbers, and a
reference comparison that finds no difference. The first proof (2026-10-22)
ran scripts that review 3 then corrected; the record named in naches_bgc_fw
is the re-proof with the corrected scripts (2026-10-23), and the scripts it
ran must still be what the code writes.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import domain_case as DC
import run_gate as RG

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
PHENOLOGY_FIX_SHA256 = "234f068dc3f9190fcdfbbe1aeb45d0935b5b7ab4d71c037f1e9fd50a70d75c45"


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


@pytest.fixture(scope="module")
def fw():
    """The proof chain's record. Skipped only while the chain may still run
    (its report, which waits on every job, has not finished): a chain that
    ended without a stage's record FAILS the tests below rather than
    skipping them."""
    b = _block("naches_bgc_fw")
    status = (b["record_dir"] / DC.STATUS_FILE).read_text()
    if not re.search(r"stage=report finished rc=", status):
        pytest.skip("the proof chain has not finished (its report has not run)")
    return {**b, "records": DC.chain_records(b["record_dir"]), "status": status}


def _stage(fw, name):
    return next(s for s in fw["records"]["stages"] if s["stage"] == name)


class TestTheRecipe:

    def test_the_production_cases_own_records(self, cfg):
        r = DC.recipe_from_case(cfg["fix_case_dir"], case="a_copy")
        dom = cfg["domain"]
        assert r.xml == [
            ("LND_DOMAIN_PATH", str(dom.parent)), ("ATM_DOMAIN_PATH", str(dom.parent)),
            ("LND_DOMAIN_FILE", dom.name), ("ATM_DOMAIN_FILE", dom.name),
            ("NTASKS", "40"), ("NTHRDS", "1"), ("ROOTPE", "0"),
            ("DATM_MODE", "CLMMOSARTTEST"), ("DATM_CLMNCEP_YR_ALIGN", "2013"),
            ("DATM_CLMNCEP_YR_START", "2013"), ("DATM_CLMNCEP_YR_END", "2023"),
            ("RUN_TYPE", "startup"), ("RUN_STARTDATE", "2015-01-01"), ("DOUT_S", "FALSE")]
        assert (r.compset, r.res, r.mach, r.compiler, r.project, r.start) == (
            "2000_DATM%QIA_ELM%CNPRDCTCBC_SICE_SOCN_SROF_SGLC_SWAV", "ELM_USRDAT", "compy",
            "intel", "e3sm", "2015-01-01")
        assert Path(r.srcroot).is_dir() and r.copied_from == str(cfg["fix_case_dir"])
        assert r.inputs() == {"surface": str(cfg["surface"]), "domain": str(dom),
                              "finidat": str(cfg["restart"])}
        assert r.user_nl["elm"] == (cfg["fix_case_dir"] / "user_nl_elm").read_text().splitlines()
        assert r.namelist()["do_budgets"] == [False]
        assert {k: v["sha256"] for k, v in r.sourcemods.items()} == {
            "src.elm/PhenologyMod.F90": PHENOLOGY_FIX_SHA256}
        assert r.check() == [] and r.case_dir == cfg["fix_case_dir"].parent / "a_copy"
        assert any("the case it was copied from" in p
                   for p in DC.recipe_from_case(cfg["fix_case_dir"]).check())


class TestTheProofChain:

    def test_every_stage_passed(self, fw):
        recs = fw["records"]
        assert [(s["stage"], s["check_pass"]) for s in recs["stages"]] == [
            ("case", True), ("smoke", True), ("y2015", True), ("y2016", True),
            ("report", None)]
        assert recs["years_passed"] == [2015, 2016] and recs["first_unpassed"] is None
        assert fw["status"].count("finished rc=0") == 5

    def test_the_report_names_every_job_it_was_handed(self, fw):
        """The first proof's report ran after midnight and, without -S, found
        only the two year jobs: the case and smoke jobs had ended before it."""
        handed = [s["job_id"] for s in fw["records"]["stages"] if s["stage"] != "report"]
        reported = re.findall(r"stage=report REPORT \S+ job=(\d+) state=COMPLETED ",
                              fw["status"])
        assert reported == handed

    def test_the_case_was_built_from_the_source_the_recipe_recorded(self, fw):
        built = json.loads(Path(_stage(fw, "case")["check"]).read_text())["source"]
        armed = json.loads((fw["record_dir"] / DC.RECIPE_FILE).read_text())["source"]
        assert built["commit"] == armed["commit"] and len(built["commit"]) == 40
        assert built["fingerprint"] == armed["fingerprint"]
        assert f"stage=case source {built['commit']} fingerprint {built['fingerprint']}" in \
            fw["status"]

    def test_the_proof_ran_the_scripts_the_code_writes_now(self, fw, tmp_path):
        rec = fw["record_dir"]
        r = DC.Recipe.from_json((rec / DC.RECIPE_FILE).read_text())
        spec = json.loads((rec.parent / "01_inputs" / "domain_chain.json").read_text())
        stage = (rec / DC.STAGE_SCRIPT).read_text()
        out = tmp_path / "rec"
        DC.write_case_script(r, out)
        ref = spec.get("reference")
        DC.write_stage_script(r, out, smoke_days=int(spec.get("smoke_days", 5)),
                              reference=DC.Reference(**ref) if ref else None,
                              python=re.search(r"PY=(\S+);", stage).group(1))

        def norm(p, d):
            return re.sub(r"CASE_SCRIPT_SHA256=\w+", "CASE_SCRIPT_SHA256=X",
                          p.read_text().replace(str(d), "REC"))
        for name in (DC.CASE_SCRIPT, DC.STAGE_SCRIPT):
            assert norm(rec / name, rec) == norm(out / name, out), name

    def test_the_rebuilt_case_resolves_the_production_namelist(self, cfg, fw):
        a = (cfg["fix_case_dir"] / "CaseDocs" / "lnd_in").read_text()
        b = (fw["case_dir"] / "CaseDocs" / "lnd_in").read_text()
        assert a.replace(cfg["fix_case_dir"].name, "CASE") == \
            b.replace(fw["case_dir"].name, "CASE")

    @pytest.mark.parametrize("year", [2015, 2016])
    def test_each_year_is_the_production_years(self, cfg, fw, year):
        """Computed here for both runs, the year numbers are EQUAL. The job's
        own record was computed on a compute node, where numpy sums in
        another order: the same files give a last-digit difference there
        (487.0185813046424 against 487.01858130464234 for 2015 GPP), so the
        record is held to 1e-12 instead."""
        job = json.loads(Path(_stage(fw, f"y{year}")["check"]).read_text())["years"][str(year)]
        prod = RG.year_check(cfg["fix_run_dir"], cfg["fix_case_dir"], year)
        ours = RG.year_check(fw["run_dir"], fw["case_dir"], year)
        for k in ("gpp_gC_m2_yr", "peak_tlai", "precip_mm", "et_mm", "runoff_mm",
                  "tws_change_mm", "water_closure_frac_of_precip", "swe_apr1_mm",
                  "totecosysc_change_frac", "days", "cells", "nan_cells"):
            assert ours[k] == prod[k], (k, ours[k], prod[k])
            assert job[k] == pytest.approx(prod[k], rel=1e-12, abs=1e-15), (k, job[k], prod[k])
        assert ours["pass"] is job["pass"] is prod["pass"] is True
        ref = job["reference"]
        assert ref["gated"] is True and ref["pass"] is True
        for f, v in ref["fields"].items():
            if "missing" not in v:
                # identical series: equal means, and r 1 to rounding (corrcoef of a
                # series with itself can give 0.9999999999999999)
                assert v["ours"] == v["reference"] and v["bias_pct"] == 0.0, (f, v)
                assert v["daily_r"] is None or v["daily_r"] == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("stream,file_year,names", [
        ("h1", 2016, ("GPP", "TLAI", "H2OSNO", "TOTSOMC", "QRUNOFF", "TWS")),
        ("h0", 2015, ("GPP", "TOTECOSYSC", "FAREA_BURNED"))])
    def test_the_history_is_bit_for_bit(self, cfg, fw, stream, file_year, names):
        import netCDF4
        name = {"h1": f"elm.h1.{file_year}-01-01-00000.nc",
                "h0": f"elm.h0.{file_year}-02-01-00000.nc"}
        a = cfg["fix_run_dir"] / f"{cfg['fix_case_dir'].name}.{name[stream]}"
        b = fw["run_dir"] / f"{fw['case_dir'].name}.{name[stream]}"
        with netCDF4.Dataset(a) as x, netCDF4.Dataset(b) as y:
            for v in names:
                assert np.array_equal(np.asarray(x[v][:]), np.asarray(y[v][:]), equal_nan=True), v

    def test_the_jobs_ran_their_own_years(self, fw):
        for year in (2015, 2016):
            logs = json.loads(Path(_stage(fw, f"y{year}")["check"]).read_text())["logs"]
            assert (logs["model_start"], logs["model_end"]) == (f"{year}-01-01",
                                                                f"{year + 1}-01-01")


class TestTheManagerOnTheProofChain:

    def test_the_year_rows_and_the_package(self, fw, tmp_path):
        from elm_exp_manager import ELMExpManager
        run = tmp_path / "run"
        shutil.copytree(fw["record_dir"], run / "domain_case")
        m = ELMExpManager(run_dir=str(run))
        recipe = (fw["record_dir"] / DC.RECIPE_FILE).read_text()
        (m.input_dir / m.DOMAIN_RECIPE).write_text(recipe)
        rows = [{"case_name": fw["case_dir"].name, "kind": "whole_domain"}]
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps(rows))
        assert m._outcomes_from_disk(rows) == {fw["case_dir"].name: True}
        out = m._extract_domain(rows)
        assert [(r["year"], r["status"]) for r in out["rows"]] == [(2015, "completed"),
                                                                   (2016, "completed")]
        pkg = m._package_domain({}, out, {})
        assert pkg["kind"] == "whole_domain" and pkg["columns_succeeded"] == 2
        assert pkg["case"]["sourcemods"] == {"src.elm/PhenologyMod.F90": PHENOLOGY_FIX_SHA256}
        assert pkg["case"]["source"] == json.loads(recipe)["source"]
        assert pkg["case"]["built_from"]["fingerprint"] == pkg["case"]["source"]["fingerprint"]
        assert pkg["inputs_check"]["pass"] is True
