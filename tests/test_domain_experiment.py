"""An experiment on a whole domain through the ELM server's tools: a control
run and runs that differ from it by named changes.

    pytest tests/test_domain_experiment.py -v

The server's tool functions are called in process with the real modules
behind them, a synthetic control registered in a temporary registry, and the
fake SLURM of the manager tests (no job is ever submitted from here).
"""
import json
from pathlib import Path

import pytest

import domain_case as DC
from test_domain_manager import FakeSlurm, _source_case, srv  # noqa: F401


@pytest.fixture
def lab(srv, tmp_path, monkeypatch):
    """A registry holding one control (the manager tests' source case, which
    passes its input check) and a run directory to build an experiment in."""
    slurm = FakeSlurm()
    monkeypatch.setattr(DC, "_run", slurm)
    monkeypatch.setattr(DC, "node_local", lambda path: None)
    monkeypatch.setattr(srv, "_shared_fs", lambda path: None)
    cd = _source_case(tmp_path)
    (cd / "CaseDocs").mkdir()
    (cd / "CaseDocs" / "lnd_in").write_text("&elm_inparm\n use_nofire = .false.\n/\n")
    out = tmp_path / "scratch" / "src_case" / "run"
    out.mkdir(parents=True)
    reg = tmp_path / "domain_controls.json"
    reg.write_text(json.dumps({"controls": {"synth": {
        "description": "a synthetic control", "case_dir": str(cd), "run_dir": str(out),
        "reproduced": {"years": [2015, 2016], "how": "bit for bit"}}}}))
    monkeypatch.setenv("IDEAS_DOMAIN_CONTROLS", str(reg))
    rd = tmp_path / "out" / "exp"
    (rd / "01_inputs").mkdir(parents=True)
    return {"srv": srv, "slurm": slurm, "control_dir": cd, "control_run": out, "run_dir": rd}


def _experiment(lab, runs, reuse=True, **extra):
    exp = {"control": {"name": "synth", "reuse": reuse}, "runs": runs,
           "years": [2015, 2016], "chain": {"tag": "nb", "smoke_days": 3}, **extra}
    (lab["run_dir"] / "01_inputs" / "domain_experiment.json").write_text(json.dumps(exp))
    return exp


NOFIRE = [{"name": "nofire", "changes": [{"id": "fire_off"}]}]


class TestTheMenus:

    def test_the_change_list(self, srv):
        d = json.loads(srv.describe_domain_changes())
        assert {c["id"] for c in d["changes"]} >= {"fire_off", "co2_ppmv", "burn_footprint"}

    def test_the_controls(self, lab):
        d = json.loads(lab["srv"].describe_domain_controls())
        row = d["controls"][0]
        assert row["name"] == "synth" and row["usable"] is True
        assert row["facts"] is None and "refresh_domain_control_facts" in row["facts_note"]

    def test_the_check(self, lab):
        out = json.loads(lab["srv"].check_domain_experiment(
            {"control": {"name": "synth", "reuse": True}, "years": [2015, 2016],
             "runs": [{"name": "moved", "changes": [{"id": "burn_footprint"}]}]}))
        assert out["ok"] is False and out["wont_build"][0]["requires"][0].startswith(
            "a burn tool")
        bad = json.loads(lab["srv"].check_domain_experiment(
            {"control": {"name": "nowhere", "reuse": True}, "runs": NOFIRE,
             "years": [2015]}))
        assert bad["ok"] is False and "no prepared control 'nowhere'" in bad["error"]


class TestARunAgainstAReusedControl:

    def test_its_inputs(self, lab):
        _experiment(lab, NOFIRE)
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"], out
        assert out["runs"] == [{"name": "nofire", "case": "src_case_nofire",
                                "case_dir": str(lab["control_dir"].parent / "src_case_nofire"),
                                "changes": ["fire_off"], "after": None}]
        inp = lab["run_dir"] / "01_inputs" / "runs" / "nofire"
        r = DC.Recipe.from_json((inp / "domain_recipe.json").read_text())
        assert "use_nofire = .true." in r.user_nl["elm"] and r.copied_from == str(
            lab["control_dir"])
        spec = json.loads((inp / "domain_chain.json").read_text())
        assert spec == {"smoke_days": 3, "years": [2015, 2016], "tag": "nb_nofire",
                        "reference": {"run_dir": str(lab["control_run"]), "stream": "h1",
                                      "gate_years": []}}
        e = json.loads((inp / "domain_expectations.json").read_text())
        assert e["control_case_dir"] == str(lab["control_dir"])
        assert e["settings"] == {"lnd_in": {"use_nofire": True}}
        assert out["cost"]["runs_executed"] == 1
        assert json.loads((lab["run_dir"] / "01_inputs" / "domain_runs.json").read_text()) \
            == out

    def test_its_scripts_and_its_chain(self, lab):
        _experiment(lab, NOFIRE)
        lab["srv"].build_domain_runs(str(lab["run_dir"]))
        w = json.loads(lab["srv"].write_domain_case(str(lab["run_dir"]), run="nofire"))
        assert w["ok"], w
        rec = lab["run_dir"] / "domain_case" / "nofire"
        assert Path(w["stage_script"]) == rec / DC.STAGE_SCRIPT
        text = (rec / DC.STAGE_SCRIPT).read_text()
        assert f'settings --case-dir "$CASEDIR" --control-case-dir {lab["control_dir"]}' in text
        assert '--expect "$REC/expectations.json"' in text
        assert json.loads((rec / DC.EXPECT_FILE).read_text())["run"] == "nofire"
        s = json.loads(lab["srv"].submit_domain_chain(str(lab["run_dir"]), run="nofire"))
        assert s["ok"], s
        assert [c[c.index("-J") + 1] for c in lab["slurm"].sbatch()] == [
            "nb_nofire_case", "nb_nofire_smoke", "nb_nofire_y2015", "nb_nofire_y2016",
            "nb_nofire_report"]
        st = json.loads(lab["srv"].check_domain_chain(str(lab["run_dir"]), run="nofire"))
        assert st["ok"] and st["active"]
        assert not (lab["run_dir"] / "domain_case" / DC.CHAIN_FILE).exists()

    def test_a_run_name_that_is_not_one(self, lab):
        out = json.loads(lab["srv"].write_domain_case(str(lab["run_dir"]), run="../x"))
        assert out["ok"] is False and "is not a run name" in out["error"]
        out = json.loads(lab["srv"].check_domain_chain(str(lab["run_dir"]), run="../x"))
        assert out["ok"] is False

    def test_an_experiment_that_will_not_build_writes_nothing(self, lab):
        _experiment(lab, [{"name": "moved", "changes": [{"id": "burn_footprint"}]}])
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"] is False and out["error"] == "the experiment will not build"
        assert not (lab["run_dir"] / "01_inputs" / "runs").exists()

    def test_a_reused_control_must_be_a_prepared_one(self, lab):
        exp = _experiment(lab, NOFIRE)
        exp["control"] = {"reuse": True}
        (lab["run_dir"] / "01_inputs" / "domain_experiment.json").write_text(json.dumps(exp))
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"] is False and "must be a prepared control" in out["error"]


class TestAnExperimentThatRunsItsControl:

    def test_the_other_runs_wait_for_its_case_job(self, lab):
        _experiment(lab, NOFIRE, reuse=False)
        ctl = DC.recipe_from_case(lab["control_dir"], case="ctl")
        (lab["run_dir"] / "01_inputs" / "domain_recipe.json").write_text(ctl.to_json())
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"], out
        assert [(r["name"], r["case"], r["after"]) for r in out["runs"]] == [
            ("control", "ctl", None), ("nofire", "ctl_nofire", "control")]
        assert out["cost"]["runs_executed"] == 2
        e = json.loads((lab["run_dir"] / "01_inputs" / "runs" / "nofire" /
                        "domain_expectations.json").read_text())
        assert e["control_case_dir"] == str(ctl.case_dir)
        assert not (lab["run_dir"] / "01_inputs" / "runs" / "control" /
                    "domain_expectations.json").exists()
        c = json.loads(lab["srv"].submit_domain_chain(str(lab["run_dir"]), run="control"))
        assert c["ok"], c
        case_job = c["job_ids"][0]
        n = json.loads(lab["srv"].submit_domain_chain(str(lab["run_dir"]), run="nofire",
                                                      after_job=case_job))
        assert n["ok"], n
        first = [x for x in lab["slurm"].sbatch() if "nb_nofire_case" in x][0]
        assert f"--dependency=afterok:{case_job}" in first


# ── the experiment through the manager ───────────────────────────────────────
from test_domain_manager import (_config, _passed, _stages, _year_record, _notify,  # noqa: E402
                                 mgr)  # noqa: F401,E402

CONTROL_YEARS = {"2015": {"gpp_gC_m2_yr": 487.0, "peak_tlai": 0.90, "et_mm": 333.0,
                          "runoff_mm": 494.0, "swe_apr1_mm": 1.9,
                          "totecosysc_change_frac": -0.003, "checks_pass": True},
                 "2016": {"gpp_gC_m2_yr": 536.5, "peak_tlai": 0.95, "et_mm": 355.0,
                          "runoff_mm": 627.0, "swe_apr1_mm": 22.2,
                          "totecosysc_change_frac": 0.009, "checks_pass": True}}


@pytest.fixture
def study(srv, tmp_path, mgr, monkeypatch):
    """A prepared control with cached facts, and a study config whose strategy
    is an experiment on it."""
    cd = _source_case(tmp_path)
    (cd / "CaseDocs").mkdir()
    (cd / "CaseDocs" / "lnd_in").write_text("&elm_inparm\n use_nofire = .false.\n/\n")
    out = tmp_path / "scratch" / "src_case" / "run"
    out.mkdir(parents=True)
    reg = tmp_path / "registry" / "domain_controls.json"
    reg.parent.mkdir()
    reg.write_text(json.dumps({"controls": {"synth": {
        "description": "a synthetic control", "case_dir": str(cd), "run_dir": str(out),
        "reproduced": {"years": [2015, 2016], "how": "bit for bit"}}}}))
    (reg.parent / "domain_control_facts").mkdir()
    (reg.parent / "domain_control_facts" / "synth.json").write_text(json.dumps({
        "name": "synth", "case": "src_case", "start_year": 2015, "forcing_years": [2013, 2023],
        "output_years": [2015, 2016],
        "settings": {"use_nofire": False, "ELM_CO2_TYPE": "constant", "co2_ppmv": 367.0},
        "per_year": CONTROL_YEARS,
        "burned_area_km2": {"model": {"2015": 115.0, "2016": 60.3}, "pattern": "thin",
                            "observed": {"2015": 3.7, "2016": 2.6},
                            "observed_source": "MODIS"},
        "hours_per_model_year": 0.79, "streams": {}, "field_notes": {}}))
    monkeypatch.setenv("IDEAS_DOMAIN_CONTROLS", str(reg))
    cfg = _config(srv, tmp_path, cd)
    cfg["strategy"] = {"archetype": "whole_domain", "sampling": {"n_columns": 1},
                       "domain_experiment": {"control": {"name": "synth", "reuse": True},
                                             "runs": NOFIRE, "years": [2015, 2016],
                                             "tag": "nb"}}
    # THE ANALYZER'S LLM IS OFF unless a test turns it on with a scripted
    # client, and constructing a real client fails the test: no test here may
    # reach the endpoint, with or without a key in the environment.
    cfg["analysis_llm"] = False
    import agents.llm_agent as L

    class NoLLM:
        def __init__(self, *a, **k):
            raise AssertionError("a test reached the LLM endpoint")
    monkeypatch.setattr(L, "SimpleLLMClient", NoLLM)
    return {"cfg": cfg, "control_dir": cd, "control_run": out, "mgr": mgr}


class TestAnExperimentThroughTheManager:

    def test_its_rows_and_the_rulebook_beside_them(self, study):
        m, cfg = study["mgr"], study["cfg"]
        m._materialize(dict(cfg["strategy"]), cfg)
        rows = json.loads((m.input_dir / "case_inputs.json").read_text())
        assert [(r["run"], r["case_name"], r["reused"]) for r in rows] == [
            ("control", "src_case", True), ("nofire", "src_case_nofire", False)]
        assert m._is_experiment_rows(rows)
        exp = json.loads((m.input_dir / "domain_experiment.json").read_text())
        assert exp == {"control": {"name": "synth", "reuse": True}, "runs": NOFIRE,
                       "years": [2015, 2016], "chain": {"tag": "nb"}}
        rule = json.loads((m.input_dir / "domain_control_facts.json").read_text())
        assert rule["facts"]["facts"]["per_year"] == CONTROL_YEARS
        assert any("the control's own fire burns 60-115 km2 a year" in f["what"]
                   for f in rule["check"]["facts"])
        assert cfg["mcp_clients"]["elm"].calls == [
            "check_domain_experiment", "describe_domain_controls", "build_domain_runs",
            "check_domain_inputs"]
        assert (m.input_dir / "runs" / "nofire" / "domain_expectations.json").is_file()

    def test_submit_poll_extract_and_package(self, study, capsys):
        m, cfg = study["mgr"], study["cfg"]
        first = m.execute_plan(dict(cfg["strategy"]), cfg)
        assert first.get("status") == "pending"
        assert [(j["run"], j["stage"], j["job_id"]) for j in first["chain"]] == [
            ("nofire", "case", "2001"), ("nofire", "smoke", "2002"),
            ("nofire", "y2015", "2003"), ("nofire", "y2016", "2004"),
            ("nofire", "report", "2005")]
        assert m.job_b == ["2001:2002:2003:2004:2005"]
        assert "job 2001   nofire case" in capsys.readouterr().out
        rec = m.run_dir / "domain_case" / "nofire"
        assert (rec / DC.EXPECT_FILE).is_file() and (rec / DC.CHAIN_FILE).is_file()

        m.slurm.states = {"2001": "COMPLETED", "2002": "RUNNING"}
        again = m.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert again.get("status") == "pending"

        _passed(rec, "case", "2001")
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015, gpp=480.0)
        _year_record(rec, "y2016", "2004", 2016, gpp=530.0)
        m.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2004", "2005")}
        done = m.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert done["experiments_success"] == 2
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        runs = exp["experiment"]["runs"]
        assert [(r["run"], r["reused"], r["status"]) for r in runs] == [
            ("control", True, "reused"), ("nofire", False, "completed")]
        ch = runs[1]["changes"][0]
        assert ch["id"] == "fire_off" and ch["not_meaning"].startswith(
            "the effect of an observed fire")
        rows = {(r["run"], r["year"]): r for r in exp["columns"]}
        assert rows[("control", 2015)]["status"] == "reused"
        assert rows[("control", 2015)]["metrics"]["gpp_gC_m2_yr"] == 487.0
        assert rows[("nofire", 2015)]["difference_from_control"]["gpp_gC_m2_yr"] == \
            pytest.approx(-7.0)
        assert rows[("nofire", 2016)]["difference_from_control"]["gpp_gC_m2_yr"] == \
            pytest.approx(-6.5)
        assert exp["chain"]["years_passed"] == [2015, 2016]
        assert exp["chain"]["first_unpassed"] is None and "nofire" in exp["chain"]["runs"]
        comp = exp["chain"]["compute"]
        assert comp["runs"] == {"nofire": {"jobs": [], "seconds": None}}
        assert comp["note"].startswith("a run's report stage has not written its job times")
        an = _stages(m)["analyze"]
        assert an["status"] == "skipped" and "has no regional numbers (they come from an " \
            "experiment's analysis, and its plan names none)" in an["reason"]
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 0 and body.startswith("ANALYSIS SKIPPED: ")

    def test_a_failed_year_of_a_run_leaves_the_study_incomplete(self, study):
        m, cfg = study["mgr"], study["cfg"]
        m.execute_plan(dict(cfg["strategy"]), cfg)
        rec = m.run_dir / "domain_case" / "nofire"
        _passed(rec, "case", "2001")
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        _year_record(rec, "y2016", "2004", 2016, ok=False)
        m.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2005")}
        m.slurm.states["2004"] = "FAILED"
        m.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        assert exp["chain"]["first_unpassed"] == "nofire: y2016"
        assert exp["chain"]["years_passed"] == [2015]
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 1 and "the chain stopped at nofire: y2016" in body.splitlines()[0]

    def test_a_change_the_server_lacks_stops_before_any_job(self, study):
        m, cfg = study["mgr"], study["cfg"]
        cfg["strategy"]["domain_experiment"]["runs"] = [
            {"name": "moved", "changes": [{"id": "burn_footprint"}]}]
        with pytest.raises(RuntimeError, match="the experiment will not build: run moved, "
                                               "change burn_footprint: burn_footprint is not "
                                               "available here"):
            m._materialize(dict(cfg["strategy"]), cfg)
        assert not m.slurm.sbatch()

    def test_the_base_wants_a_control_and_runs_each_its_own_case(self, study, monkeypatch):
        m, cfg = study["mgr"], study["cfg"]
        row = {"case_name": "x", "kind": "whole_domain", "run": "nofire"}
        for rows in ([], [row, dict(row)]):
            monkeypatch.setattr(type(m), "_build_domain_case", lambda self, config, r=rows: {
                "columns": r, "domain_experiment": {}})
            with pytest.raises(RuntimeError, match="a control and its runs, each its own case"):
                m._materialize(dict(cfg["strategy"]), cfg)

    @staticmethod
    def _with_history(study, tmp_path):
        """The control given a monthly history and a surface file, and the plan
        an event analysis of it: the control alone, 2016 its normal."""
        import os
        import elm_case_fixtures as cf
        import elm_gridded_fixtures as fx
        m, cfg = study["mgr"], study["cfg"]
        h = cf.write_case(tmp_path / "hist")
        fx.write_monthly(h["run_dir"], [2015, 2016], case="synth",
                         variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
        (study["control_dir"] / "CaseDocs" / "lnd_in").write_text(
            f"&elm_inparm\n use_nofire = .false.\n fsurdat = '{h['surface']}'\n/\n")
        path = Path(os.environ["IDEAS_DOMAIN_CONTROLS"])
        reg = json.loads(path.read_text())
        reg["controls"]["synth"]["run_dir"] = str(h["run_dir"])
        path.write_text(json.dumps(reg))
        # the facts were computed after the history was written, as a refresh does
        cache = path.parent / "domain_control_facts" / "synth.json"
        newest = max(q.stat().st_mtime for q in Path(h["run_dir"]).glob("*.elm.h*.nc"))
        os.utime(cache, (newest + 10, newest + 10))
        de = cfg["strategy"]["domain_experiment"]
        de["runs"] = []
        de["analysis"] = {"regions": [{"name": "vegetated cells"}], "variables": ["GPP"],
                          "normal_years": [2016]}
        return m, cfg

    def test_the_regional_numbers_reach_the_package(self, study, tmp_path):
        """A control with history: the plan's analysis becomes, per region and
        year, the control's numbers and their anomalies against normal years."""
        m, cfg = self._with_history(study, tmp_path)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        reg_t = exp["regional"]
        assert set(reg_t["runs"]) == {"control"} and reg_t["problems"] == []
        gpp = reg_t["runs"]["control"]["values"]["vegetated cells"]["GPP"]
        assert set(gpp) == {"2015", "2016"} and all(v > 0 for v in gpp.values())
        a = reg_t["event"]["anomalies"]["vegetated cells"]["GPP"]
        assert a["normal"] == pytest.approx(gpp["2016"])
        assert json.loads((m.results_dir / "regional.json").read_text()) == reg_t
        assert exp["artifacts"]["regional"] == "03_results/regional.json"

    def test_the_report_is_rendered_from_the_records(self, study, tmp_path):
        """The Analyzer switched off: the analyze stage renders the numbers
        from the regional table, and its headline's numbers are the table's."""
        from agents.analysis.step4_report import format_value
        m, cfg = self._with_history(study, tmp_path)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "done" and an["steps"] == {"rendered_report": True}
        assert an["analyzer"] == "switched off (analysis_llm: false)"
        assert an["report"] == str(m.run_dir / "04_analysis" / "DOMAIN_REPORT.md")
        assert not (m.run_dir / "04_analysis" / "analysis.json").exists()
        rec = json.loads((m.run_dir / "04_analysis" / "domain_report.json").read_text())
        a = json.loads((m.run_dir / "experiment.json").read_text())[
            "regional"]["event"]["anomalies"]["vegetated cells"]["GPP"]
        y = a["years"]["2015"]
        word = "above" if y["anomaly"] > 0 else "below"
        assert rec["headline"] == (
            f"GPP over vegetated cells was {format_value(abs(y['anomaly_pct']))} % {word} "
            f"its 2016 normal ({format_value(a['normal'])} gC/m^2/yr) in 2015.")
        assert rec["headline"] in (m.run_dir / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        assert rec["status"] == "complete"
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 0 and body.startswith("OK"), body.splitlines()[0]

    def test_the_numbers_then_the_analyzer(self, study, tmp_path, monkeypatch):
        """The Analyzer on: after the numbers, steps 0 to 4 read the same
        records through the regional frame (a scripted LLM), and both reports
        stand side by side."""
        import agents.llm_agent as L
        from test_domain_analyzer import Scripted
        m, cfg = self._with_history(study, tmp_path)
        cfg["analysis_llm"] = True
        cfg["analysis_with_images"] = False
        fig = {"id": "gpp_by_year", "question": "GPP by year", "why": "the trajectory",
               "scale": "overall", "variables": ["GPP"],
               "code": ("g = reg[(reg.variable == 'GPP') & (reg.region == 'vegetated cells')]"
                        "\nfig, ax = plt.subplots()\nax.bar(g.year, g.value)\n"
                        "fig.savefig(out_path)\n"
                        "result = {'n': 3, 'gpp': {str(int(y)): float(v) for y, v in "
                        "zip(g.year, g.value)}}\n")}

        class Script(Scripted):
            def __init__(self, model=None):
                super().__init__(model, figures=(fig,), review={
                    "claims": [], "headline": "This run cannot answer it with 2 years.",
                    "answer": "Two years are too few.", "verdict": "sufficient",
                    "feedback": ""})
        monkeypatch.setattr(L, "SimpleLLMClient", Script)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "done", an
        assert an["steps"] == {"rendered_report": True, "context": True, "compare": True,
                               "investigate": True, "interpret": True, "report": True}
        assert an["report"] == str(m.run_dir / "04_analysis" / "DOMAIN_REPORT.md")
        a = json.loads((m.run_dir / "04_analysis" / "analysis.json").read_text())
        assert a["kind"] == "whole_domain" and a["domain_report"] == "DOMAIN_REPORT.md"
        assert (m.run_dir / "04_analysis" / "REPORT.md").is_file()
        assert (m.run_dir / "04_analysis" / "gpp_by_year.png").is_file()
        # the server's whole-domain limitations reach the package and bind
        import domain_analysis as DA
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        assert exp["limitations"] == DA.LIMITATIONS
        assert {"id": "limitation_structural_1", "severity": "blocking"}.items() <= next(
            c for c in a["caveats"] if c["id"] == "limitation_structural_1").items()
        assert exp["chain"]["compute"]["control"].startswith("reused")
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 0 and body.startswith("OK"), body.splitlines()[0]

    def test_a_failed_numbers_report_fails_the_stage_whatever_the_analyzer_does(
            self, study, tmp_path, monkeypatch):
        import agents.llm_agent as L
        import agents.analysis.domain_report as DR
        from test_domain_analyzer import Scripted

        def boom(run_dir):
            raise ValueError("boom")
        monkeypatch.setattr(DR, "build", boom)
        monkeypatch.setattr(L, "SimpleLLMClient", Scripted)
        m, cfg = self._with_history(study, tmp_path)
        cfg["analysis_llm"], cfg["analysis_with_images"] = True, False
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "failed" and an["error"] == \
            "the report could not be rendered (boom)"
        assert an["steps"]["rendered_report"] is False and an["steps"]["report"] is True

    def test_stale_facts_leave_the_study_incomplete_and_say_why(self, study, tmp_path):
        """A control whose output changed after its facts were computed: its
        regional numbers are read from the output, but its year numbers are not
        taken from facts that no longer describe it."""
        import os
        m, cfg = self._with_history(study, tmp_path)
        cache = Path(os.environ["IDEAS_DOMAIN_CONTROLS"]).parent / "domain_control_facts" \
            / "synth.json"
        os.utime(cache, (1.0, 1.0))
        m.execute_plan(dict(cfg["strategy"]), cfg)
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        assert exp["chain"]["first_unpassed"] == "control: its facts hold no [2015, 2016]"
        rec = json.loads((m.run_dir / "04_analysis" / "domain_report.json").read_text())
        assert rec["status"] == ("incomplete: the chain stopped at control: its facts hold no "
                                 "[2015, 2016] (years passed [] of [2015, 2016])")
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 1 and "control: its facts hold no [2015, 2016]" in body.splitlines()[0]

    def test_regional_numbers_that_cannot_be_made_are_a_failed_analysis(self, study):
        """The plan asked for an analysis and did not get one (here the control
        has no history to read): a failure, never a deliberate skip."""
        m, cfg = study["mgr"], study["cfg"]
        de = cfg["strategy"]["domain_experiment"]
        de["runs"] = []
        de["analysis"] = {"regions": [{"name": "vegetated cells"}], "variables": ["GPP"],
                          "normal_years": [2016]}
        m.execute_plan(dict(cfg["strategy"]), cfg)
        assert "error" in json.loads((m.results_dir / "regional.json").read_text())
        an = _stages(m)["analyze"]
        assert an["status"] == "failed" and an["error"].startswith(
            "the report could not be rendered (no regional table to report: ")
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 1 and "analyze (failed)" in body.splitlines()[0]

    def test_numbers_that_cannot_be_made_spend_no_llm_call(self, study):
        """The Analyzer on, the regional numbers missing: its preflight stops it
        before any model is asked (the fixture's guard fails on a client), and
        the stage fails with the cause."""
        m, cfg = study["mgr"], study["cfg"]
        cfg["analysis_llm"] = True
        de = cfg["strategy"]["domain_experiment"]
        de["runs"] = []
        de["analysis"] = {"regions": [{"name": "vegetated cells"}], "variables": ["GPP"],
                          "normal_years": [2016]}
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "failed" and an["error"].startswith(
            "the report could not be rendered")
        assert an["steps"]["rendered_report"] is False and an["steps"]["context"] is True
        a = json.loads((m.run_dir / "04_analysis" / "analysis.json").read_text())
        assert a["status"] == "insufficient_evidence"
        assert a["answer"].startswith("This run was not investigated: this whole-domain run "
                                      "has no regional numbers")

    def test_a_report_that_cannot_be_rendered_is_a_failed_analysis(self, study, tmp_path,
                                                                    monkeypatch):
        import agents.analysis.domain_report as DR

        def boom(run_dir):
            raise ValueError("no regional table to report: boom")
        monkeypatch.setattr(DR, "build", boom)
        m, cfg = self._with_history(study, tmp_path)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "failed" and "boom" in an["error"]
        assert (m.run_dir / "experiment.json").is_file()
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 1 and "analyze (failed)" in body.splitlines()[0]

    def test_the_control_alone_spends_nothing(self, study):
        """An event question about the control itself (how did the domain behave
        in and after a dry year?) needs no run: its output is read as it is."""
        m, cfg = study["mgr"], study["cfg"]
        cfg["strategy"]["domain_experiment"]["runs"] = []
        done = m.execute_plan(dict(cfg["strategy"]), cfg)
        assert not m.slurm.sbatch() and m.job_b == []
        assert done["experiments_success"] == 1
        exp = json.loads((m.run_dir / "experiment.json").read_text())
        assert [(r["run"], r["year"], r["status"]) for r in exp["columns"]] == [
            ("control", 2015, "reused"), ("control", 2016, "reused")]
        assert exp["chain"]["years_passed"] == [2015, 2016] and \
            exp["chain"]["first_unpassed"] is None
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 0, body.splitlines()[0]


class TestTheGate:

    def _check(self, strategy):
        from core.strategy_check import check
        from test_domain_manager import _reception
        return check(_reception(), strategy)

    @pytest.mark.parametrize("de,problem", [
        ({"control": {"name": "naches", "reuse": True}, "runs": "x"}, "must be a list"),
        ({"control": {}, "runs": NOFIRE}, "names no control"),
        ({"control": {"from_case": "/x"}, "runs": NOFIRE}, "names no control"),
        ({"control": {"recipe": {"case": "c"}, "reuse": True}, "runs": NOFIRE},
         "a reused control is a prepared one"),
        ("nope", "is not an object"),
    ])
    def test_what_it_stops(self, de, problem):
        rep, _ = self._check({"archetype": "whole_domain", "domain_experiment": de})
        assert any(problem in s for s in rep["stop"]), rep["stop"]

    def test_a_well_formed_experiment_passes(self):
        rep, st = self._check({"archetype": "whole_domain", "domain_experiment": {
            "control": {"name": "naches", "reuse": True}, "runs": NOFIRE}})
        assert rep["stop"] == [] and st["sampling"]["n_columns"] == 1

    def test_an_infeasible_verdict_stops_a_whole_domain_study(self):
        rep, _ = self._check({"archetype": "whole_domain",
                              "domain_experiment": {"control": {"name": "naches",
                                                                "reuse": True},
                                                    "runs": NOFIRE},
                              "feasibility": {"verdict": "infeasible",
                                              "why": "a burn cannot be placed"},
                              "requires": ["a burn tool"]})
        assert rep["stop"] == ["the planner judged this whole-domain study infeasible: a burn "
                               "cannot be placed (requires: a burn tool)"]

    def test_the_verdict_is_not_acted_on_for_a_site_study(self):
        from test_domain_manager import _reception
        from core.strategy_check import check
        rep, _ = check(_reception(), {"archetype": "site", "sampling": {"n_columns": 5},
                                      "feasibility": {"verdict": "infeasible", "why": "x"}})
        assert not any("infeasible" in s for s in rep["stop"])
