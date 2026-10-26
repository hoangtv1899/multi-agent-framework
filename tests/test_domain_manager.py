"""The experiment manager's whole-domain mode, driving the real ELM server tools.

    pytest tests/test_domain_manager.py -v

The client routes each call to the tool function in mcp/elm-mcp/main.py, so
the manager and the server are tested together; only SLURM is a fake (no job
is ever submitted from here), and job B's submission is replaced by a record
of what it would have depended on.
"""
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "elm-mcp" / "main.py"

import domain_case as DC                 # noqa: E402  (the module the server uses)
import domain_inputs as D                # noqa: E402
import elm_case_fixtures as cf           # noqa: E402


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("elm_mcp_server_domain", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _notify():
    spec = importlib.util.spec_from_file_location("notify_study_domain",
                                                  ROOT / "tools" / "notify_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ServerClient:
    """An MCP client that calls the server's tool functions in process."""

    def __init__(self, srv):
        self.srv, self.timeout, self.calls = srv, 30.0, []

    def call_tool_json(self, tool, args):
        self.calls.append(tool)
        return json.loads(getattr(self.srv, tool)(**args))


class FakeSlurm:
    """sbatch, squeue and sacct as Compy answers them (a job that left the
    queue is an invalid id to squeue). A job not in states is still queued."""

    def __init__(self):
        self.calls, self.next, self.states = [], 2000, {}

    def __call__(self, argv, timeout=120):
        self.calls.append(list(argv))
        if argv[0] == "sbatch":
            self.next += 1
            return 0, f"{self.next}\n", ""
        if argv[0] in ("squeue", "sacct"):
            st = self.states.get(argv[argv.index("-j") + 1], "PENDING")
            if argv[0] == "squeue":
                return (0, st + "\n", "") if st in DC.ACTIVE else \
                    (1, "", "slurm_load_jobs error: Invalid job id specified")
            return (0, "", "") if st in DC.ACTIVE or st == "GONE" else (0, f" {st} \n", "")
        return 0, "", ""

    def sbatch(self):
        return [c for c in self.calls if c[0] == "sbatch"]


def _source_case(tmp_path, restart_edit=None, user_nl_extra=""):
    """A CIME case's records (env_case.xml, README.case, CaseStatus,
    user_nl_elm, SourceMods) over a synthetic input set that passes its check."""
    inp = tmp_path / "inputs"
    inp.mkdir()
    surf = cf.write_surface(inp / "surf.nc")
    rest = cf.write_restart(inp / "rest.nc", edit=restart_edit)
    D.domain_from_surface(surf, inp / "domain.nc")
    cd = tmp_path / "cases" / "src_case"
    (cd / "SourceMods" / "src.elm").mkdir(parents=True)
    (cd / "SourceMods" / "src.elm" / "PhenologyMod.F90").write_text("! the fix\n")
    (cd / "env_case.xml").write_text(f'<entry id="SRCROOT" value="{tmp_path}/E3SM">\n')
    (cd / "README.case").write_text(
        f"2026-10-14 10:12:19: ./create_newcase --case {cd} "
        f"--compset 2000_DATM%QIA_ELM%CNPRDCTCBC_SICE_SOCN_SROF_SGLC_SWAV --res ELM_USRDAT "
        f"--mach compy --compiler intel --project e3sm\n")
    (cd / "CaseStatus").write_text(
        f"t: xmlchange success <command> ./xmlchange LND_DOMAIN_PATH={inp},"
        f"LND_DOMAIN_FILE=domain.nc  </command>\n"
        f"t: xmlchange success <command> ./xmlchange RUN_TYPE=startup,"
        f"RUN_STARTDATE=2015-01-01  </command>\n"
        "t: case.setup success \n")
    (cd / "user_nl_elm").write_text(f"finidat = '{rest}'\nfsurdat = '{surf}'\n"
                                    "do_budgets = .false.\n" + user_nl_extra)
    return cd


def _config(srv, tmp_path, cd, **dc):
    # NO BOUNDING BOX: reception fetches no grid or basin for a whole-domain
    # study, so a real brief names the place and nothing else. A box here once
    # hid that materialize still demanded one (the first fire-off twin,
    # 2026-10-25, stopped on it before any job).
    reception = {"brief": {"domain": {"name": "synthetic"},
                           "run_settings": {"resolved_period": {"yr_start": 2015,
                                                                "yr_end": 2016}}},
                 "grid": {}, "observations": {}}
    strategy = {"archetype": "whole_domain", "sampling": {"n_columns": 1},
                "domain_case": {"from_case": str(cd), "case": "new_case",
                                "case_root": str(tmp_path / "cases"), "tag": "nb",
                                "reference": {"run_dir": "/ref/run", "gate_years": [2015]},
                                **dc}}
    return {"brief": reception["brief"], "reception": reception, "strategy": strategy,
            "mcp_clients": {"elm": ServerClient(srv)}, "yr_start": 2015, "yr_end": 2016}


@pytest.fixture
def mgr(srv, tmp_path, monkeypatch):
    from elm_exp_manager import ELMExpManager
    m = ELMExpManager(base_output_dir=str(tmp_path / "out"))
    slurm = FakeSlurm()
    monkeypatch.setattr(DC, "_run", slurm)
    # tmp_path is a login node's own disk, which a submission refuses (its
    # own tests are below); job B's state comes from the fake too
    monkeypatch.setattr(DC, "node_local", lambda path: None)
    monkeypatch.setattr(srv, "_shared_fs", lambda path: None)
    monkeypatch.setattr(ELMExpManager, "_slurm_state",
                        staticmethod(lambda jid: slurm.states.get(str(jid))))
    job_b = []

    def fake_job_b(self, job_a, config, email=""):
        job_b.append(job_a)
        return "2999"
    monkeypatch.setattr(ELMExpManager, "_submit_job_b", fake_job_b)
    m.slurm, m.job_b = slurm, job_b
    return m


def _year_record(rec_dir, stage, job, year, ok=True, gpp=487.0):
    (rec_dir / "checks").mkdir(parents=True, exist_ok=True)
    (rec_dir / "checks" / f"{stage}_{job}.json").write_text(json.dumps({
        "pass": ok, "years": {str(year): {
            "year": year, "pass": ok, "reasons": [] if ok else ["water does not close"],
            "gpp_gC_m2_yr": gpp, "peak_tlai": 0.9, "precip_mm": 955.1, "et_mm": 332.7,
            "runoff_mm": 492.7, "tws_change_mm": 12.0, "water_closure_frac_of_precip": 1e-4,
            "swe_apr1_mm": 1.9, "totecosysc_change_frac": -0.001, "cells": 13394, "days": 365,
            "reference": {"pass": True, "gated": year == 2015, "reference": "/ref/run",
                          "fields": {"GPP": {"bias_pct": 0.0, "daily_r": 1.0, "ok": True},
                                     "NPP": {"missing": "reference", "ok": None}},
                          "bands": {"below 1000 m": {"cells": 5807}}}}}}))


def _passed(rec_dir, stage, job, ok=True, **extra):
    (rec_dir / "checks").mkdir(parents=True, exist_ok=True)
    (rec_dir / "checks" / f"{stage}_{job}.json").write_text(json.dumps({"pass": ok, **extra}))


def _stages(mgr):
    return json.loads((mgr.run_dir / "run_state.json").read_text())["stages"]


class TestMaterialize:

    def test_one_row_its_recipe_and_its_chain(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        plan = mgr._materialize(dict(cfg["strategy"]), cfg)
        assert plan["DOMAIN_CASE"]["recipe"] == "01_inputs/domain_recipe.json"
        cols = json.loads((mgr.run_dir / "columns.json").read_text())["columns"]
        assert len(cols) == 1 and cols[0]["kind"] == "whole_domain"
        assert cols[0]["case_name"] == "new_case" and cols[0]["years"] == [2015, 2016]
        recipe = json.loads((mgr.input_dir / "domain_recipe.json").read_text())
        assert recipe["case"] == "new_case" and recipe["start"] == "2015-01-01"
        assert recipe["srcroot"] == f"{tmp_path}/E3SM" and recipe["copied_from"] == str(cd)
        assert recipe["xml"][1] == ["LND_DOMAIN_FILE", "domain.nc"]
        chain = json.loads((mgr.input_dir / "domain_chain.json").read_text())
        assert chain["years"] == [2015, 2016] and chain["tag"] == "nb"
        assert chain["reference"] == {"run_dir": "/ref/run", "gate_years": [2015]}
        assert DC.chain_spec_problems(chain) == []
        check = json.loads((mgr.run_dir / "domain_case" / "inputs_check.json").read_text())
        assert check["pass"] is True
        assert mgr._is_domain_rows(json.loads((mgr.input_dir / "case_inputs.json").read_text()))
        assert cfg["mcp_clients"]["elm"].calls == ["read_domain_recipe", "check_domain_inputs"]

    def test_a_recipe_given_whole(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        recipe = json.loads(srv.read_domain_recipe(str(cd), case="new_case"))["recipe"]
        cfg = _config(srv, tmp_path, cd)
        cfg["strategy"]["domain_case"] = {"recipe": recipe, "years": [2015]}
        mgr._materialize(dict(cfg["strategy"]), cfg)
        assert cfg["mcp_clients"]["elm"].calls == ["check_domain_inputs"]
        assert json.loads((mgr.input_dir / "domain_chain.json").read_text())["years"] == [2015]

    def test_an_input_set_that_fails_stops_before_any_job(self, srv, tmp_path, mgr):
        def bump(t):
            t["pfts1d_wtlnd"][5] += 0.01
        cd = _source_case(tmp_path, restart_edit=bump)
        cfg = _config(srv, tmp_path, cd)
        with pytest.raises(RuntimeError, match="input set fails its check, so nothing was "
                                               "submitted: ELM stops: PFT weights differ"):
            mgr._materialize(dict(cfg["strategy"]), cfg)
        assert not mgr.slurm.sbatch()

    def test_a_strategy_naming_no_case_is_stopped_by_the_gate(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        cfg["strategy"] = {"archetype": "whole_domain"}
        with pytest.raises(ValueError, match="names no case"):
            mgr._materialize(dict(cfg["strategy"]), cfg)

    def test_a_copy_without_a_new_name_is_refused(self, srv, tmp_path, mgr):
        """Without `case` the recipe names the source case, and the chain
        would build and run in it."""
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        cfg["strategy"]["domain_case"].pop("case")
        with pytest.raises(ValueError, match="names no new case"):
            mgr._materialize(dict(cfg["strategy"]), cfg)
        with pytest.raises(RuntimeError, match="names no new case"):
            mgr._build_domain_case(cfg)
        assert cfg["mcp_clients"]["elm"].calls == []
        out = json.loads(srv.read_domain_recipe(str(cd)))
        assert out["ok"] and any("the case it was copied from" in p for p in out["problems"])

    def test_a_case_directory_that_exists_is_refused_before_the_input_check(
            self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        (tmp_path / "cases" / "new_case").mkdir()
        cfg = _config(srv, tmp_path, cd)
        with pytest.raises(RuntimeError, match="new_case, which exists"):
            mgr._materialize(dict(cfg["strategy"]), cfg)
        assert cfg["mcp_clients"]["elm"].calls == ["read_domain_recipe"]

    def test_a_backend_returning_two_rows_is_refused(self, srv, tmp_path, mgr, monkeypatch):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        row = {"case_name": "x", "kind": "whole_domain"}
        monkeypatch.setattr(type(mgr), "_build_domain_case",
                            lambda self, config: {"columns": [row, dict(row)]})
        with pytest.raises(RuntimeError, match="one case, and the backend returned 2 rows"):
            mgr._materialize(dict(cfg["strategy"]), cfg)

    def test_the_chain_spec_the_manager_writes(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd, smoke_days=3, notes=["why this run"])
        cfg["queue"] = "slurm7"
        mgr._materialize(dict(cfg["strategy"]), cfg)
        chain = json.loads((mgr.input_dir / "domain_chain.json").read_text())
        recipe = json.loads((mgr.input_dir / "domain_recipe.json").read_text())
        assert chain["partition"] == "slurm7" and chain["smoke_days"] == 3
        assert "why this run" in recipe["notes"]

    def test_without_a_period_the_chain_runs_the_recipes_first_year(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        cfg.pop("yr_start")
        cfg.pop("yr_end")
        mgr._build_domain_case(cfg)
        assert json.loads((mgr.input_dir / "domain_chain.json").read_text())["years"] == [2015]


class TestTheLifecycle:

    def test_submit_poll_extract_and_package(self, srv, tmp_path, mgr, capsys):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        first = mgr.execute_plan(dict(cfg["strategy"]), cfg)
        assert first.get("status") == "pending"
        assert [(j["stage"], j["job_id"]) for j in first["chain"]] == [
            ("case", "2001"), ("smoke", "2002"), ("y2015", "2003"), ("y2016", "2004"),
            ("report", "2005")]
        printed = capsys.readouterr().out
        assert "job 2001   case" in printed and "job 2999   finalizes the study" in printed
        bc = _stages(mgr)["build_cases"]
        assert bc["status"] == "pending" and bc["scope"] == "domain"
        assert bc["job_ids"] == ["2001", "2002", "2003", "2004", "2005"]
        assert mgr.job_b == ["2001:2002:2003:2004:2005"]       # job B waits on all of them
        rec = mgr.run_dir / "domain_case"
        assert (rec / "create_case.sh").is_file() and (rec / "run_stage.sh").is_file()
        assert "ARMED case=2001 smoke=2002 y2015=2003 y2016=2004 report=2005" in \
            (rec / "STATUS.log").read_text()

        # still going: the resumed run stops again, and submits nothing
        mgr.slurm.states = {"2001": "COMPLETED", "2002": "COMPLETED", "2003": "RUNNING"}
        n_sbatch = len(mgr.slurm.sbatch())
        again = mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert again.get("status") == "pending" and len(mgr.slurm.sbatch()) == n_sbatch

        # landed: 2015 passed, 2016 failed its gate
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        _year_record(rec, "y2016", "2004", 2016, ok=False, gpp=536.0)
        mgr.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2005")}
        mgr.slurm.states["2004"] = "FAILED"
        done = mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert done["experiments_success"] == 0          # a chain that stopped is not a success
        exp = json.loads((mgr.run_dir / "experiment.json").read_text())
        assert exp["kind"] == "whole_domain" and exp["columns_total"] == 2
        assert exp["columns_succeeded"] == 1
        assert [(r["year"], r["status"], r["metrics"]["gpp_gC_m2_yr"]) for r in exp["columns"]] \
            == [(2015, "completed", 487.0), (2016, "failed", 536.0)]
        assert exp["columns"][0]["reference"]["fields"] == {
            "GPP": {"bias_pct": 0.0, "daily_r": 1.0, "ok": True}}
        assert exp["chain"]["years_passed"] == [2015]
        assert exp["chain"]["first_unpassed"] == "y2016"
        assert exp["inputs_check"]["pass"] is True
        assert exp["case"]["sourcemods"] == {"src.elm/PhenologyMod.F90": DC.sha256(
            cd / "SourceMods" / "src.elm" / "PhenologyMod.F90")}
        rows = json.loads((mgr.input_dir / "case_inputs.json").read_text())
        assert rows[0]["status"] == "incomplete"
        assert _stages(mgr)["analyze"]["reason"] == "no successful columns"
        assert "first stage without a passing check y2016" in \
            (mgr.results_dir / "execution_report.txt").read_text()
        body, rc = _notify().build(str(mgr.run_dir))
        assert rc == 1 and "the chain stopped at y2016 (years passed [2015] of [2015, 2016])" \
            in body.splitlines()[0]

    def test_a_chain_that_passed_every_year(self, srv, tmp_path, mgr):
        src = tmp_path / "E3SM"                      # the SRCROOT the case records name
        src.mkdir()
        g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        (src / "a.F90").write_text("x = 1\n")
        for argv in (["init", "-q"], ["add", "a.F90"], ["commit", "-q", "-m", "x"]):
            subprocess.run(g + argv, cwd=src, check=True)
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        mgr.execute_plan(dict(cfg["strategy"]), cfg)
        rec = mgr.run_dir / "domain_case"
        recorded = json.loads((rec / DC.RECIPE_FILE).read_text())["source"]
        assert len(recorded["commit"]) == 40
        built = dict(recorded, time="at build")
        _passed(rec, "case", "2001", source=built)
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        _year_record(rec, "y2016", "2004", 2016)
        mgr.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2004", "2005")}
        done = mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert done["experiments_success"] == 1
        exp = json.loads((mgr.run_dir / "experiment.json").read_text())
        assert exp["case"]["source"] == recorded and exp["case"]["built_from"] == built
        assert exp["case"]["copied_from"] == str(cd)
        assert exp["artifacts"]["recipe"] == "domain_case/domain_recipe.json"
        an = _stages(mgr)["analyze"]
        assert an["status"] == "skipped" and "whole-domain run has no regional numbers" in \
            an["reason"]
        body, rc = _notify().build(str(mgr.run_dir))
        assert rc == 0 and body.startswith("ANALYSIS SKIPPED: model output and experiment.json "
                                           "complete; the Analyzer reads sampled-column rows")
        assert "chain : years passed [2015, 2016] of [2015, 2016]" in body

    def test_a_chain_whose_build_failed_is_not_reported_complete(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd)
        mgr.execute_plan(dict(cfg["strategy"]), cfg)
        mgr.slurm.states = {"2001": "FAILED", "2002": "GONE", "2003": "GONE",
                            "2004": "GONE", "2005": "COMPLETED"}
        done = mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert done["experiments_success"] == 0
        rows = json.loads((mgr.input_dir / "case_inputs.json").read_text())
        assert rows[0]["status"] == "failed"
        exp = json.loads((mgr.run_dir / "experiment.json").read_text())
        assert [(r["year"], r["status"]) for r in exp["columns"]] == [(2015, "not run"),
                                                                      (2016, "not run")]
        body, rc = _notify().build(str(mgr.run_dir))
        assert rc == 1 and "the chain stopped at case" in body.splitlines()[0]

    def test_a_chain_whose_years_all_failed_is_not_complete_after_the_finalize(
            self, srv, tmp_path, mgr):
        """Job B's own path: adopt_completed_run, then the resume tail."""
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd, years=[2015])
        mgr.execute_plan(dict(cfg["strategy"]), cfg)
        rec = mgr.run_dir / "domain_case"
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015, ok=False)
        rows = json.loads((mgr.input_dir / "case_inputs.json").read_text())
        assert mgr._outcomes_from_disk(rows) == {"new_case": False}
        mgr.adopt_completed_run()
        assert json.loads((mgr.input_dir / "case_inputs.json").read_text())[0]["status"] == \
            "failed"
        mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        body, rc = _notify().build(str(mgr.run_dir))
        assert rc == 1, body.splitlines()[0]

    def test_the_finalize_adopts_the_chain_from_disk(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd, years=[2015])
        mgr.execute_plan(dict(cfg["strategy"]), cfg)
        rec = mgr.run_dir / "domain_case"
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        assert mgr.adopt_completed_run() == {"build_cases": 1, "run": 1}
        st = _stages(mgr)
        assert st["build_cases"]["status"] == "done" and st["run"]["n_ok"] == 1
        rows = json.loads((mgr.input_dir / "case_inputs.json").read_text())
        assert rows[0]["status"] == "completed" and rows[0]["chain"]["years_passed"] == [2015]

    def test_a_resume_waits_while_the_finalize_job_runs(self, srv, tmp_path, mgr):
        """Job B collects the chain itself; a resume meanwhile would write the
        same files at the same time."""
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd, years=[2015])
        mgr.execute_plan(dict(cfg["strategy"]), cfg)
        rec = mgr.run_dir / "domain_case"
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        mgr.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2004")}
        mgr.slurm.states["2999"] = "RUNNING"
        client = cfg["mcp_clients"]["elm"]
        n = len(client.calls)
        again = mgr.execute_plan(dict(cfg["strategy"]), {**cfg, "resume": True})
        assert again.get("status") == "pending" and "check_domain_chain" not in client.calls[n:]
        assert not (mgr.run_dir / "experiment.json").exists()


class TestServerTools:

    def _materialized(self, srv, tmp_path, mgr, **dc):
        cd = _source_case(tmp_path)
        cfg = _config(srv, tmp_path, cd, **dc)
        mgr._materialize(dict(cfg["strategy"]), cfg)
        return cd, cfg

    def test_submit_checks_the_input_set_again(self, srv, tmp_path, mgr):
        """The set passed at materialize; the saved state changed since."""
        self._materialized(srv, tmp_path, mgr)

        def bump(t):
            t["pfts1d_wtlnd"][5] += 0.01
        cf.write_restart(tmp_path / "inputs" / "rest.nc", edit=bump)
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert not out["ok"] and "fails its check" in out["error"]
        assert not mgr.slurm.sbatch()

    def test_the_chain_spec_reaches_the_scripts(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr, partition="debug", smoke_days=3, tag="tg",
                           walls={"year_wall": "01:30:00", "case_wall": "00:50:00"},
                           reference={"run_dir": "/ref/run", "stream": "h0",
                                      "gate_years": [2016]})
        out = json.loads(srv.write_domain_case(str(mgr.run_dir)))
        assert out["ok"], out
        jobs = {j["stage"]: j for j in out["jobs"]}
        assert {j["partition"] for j in jobs.values()} == {"debug"}
        assert jobs["y2015"]["walltime"] == "01:30:00" and jobs["case"]["walltime"] == "00:50:00"
        assert jobs["y2015"]["name"] == "tg_y2015"
        text = Path(out["stage_script"]).read_text()
        assert "STOP_N=3" in text and "--days 3" in text
        assert "--reference /ref/run --reference-stream h0" in text
        assert 'case " 2016 " in' in text
        assert not mgr.slurm.sbatch()

    def test_a_reference_that_gates_no_year_says_so(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr, reference={"run_dir": "/ref/run"})
        out = json.loads(srv.write_domain_case(str(mgr.run_dir)))
        assert out["notes"] == ["the reference gates no year: every year is compared and "
                                "reported only"]

    def test_a_resumed_chain_without_case_or_smoke(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr, years=[2016], with_case=False, smoke=False)
        out = json.loads(srv.write_domain_case(str(mgr.run_dir)))
        assert out["ok"], out
        assert [j["stage"] for j in out["jobs"]] == ["y2016", "report"]

    def test_a_misspelt_chain_key_is_refused(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr)
        p = mgr.input_dir / "domain_chain.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), yr_start=2015)))
        for tool in (srv.write_domain_case, srv.submit_domain_chain):
            out = json.loads(tool(str(mgr.run_dir)))
            assert not out["ok"] and "chain spec key 'yr_start'" in out["error"]
        assert not mgr.slurm.sbatch()

    def test_an_indexed_user_nl_entry_does_not_break_the_tools(self, srv, tmp_path, mgr):
        cd = _source_case(tmp_path, user_nl_extra="hist_nhtfrq(2) = -24\n")
        cfg = _config(srv, tmp_path, cd)
        mgr._materialize(dict(cfg["strategy"]), cfg)
        assert json.loads(srv.write_domain_case(str(mgr.run_dir)))["ok"]

    def test_a_second_submission_over_a_running_chain_is_refused(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr)
        assert json.loads(srv.submit_domain_chain(str(mgr.run_dir)))["ok"]
        rec = mgr.run_dir / "domain_case"
        before = {n: (rec / n).read_text() for n in (DC.CHAIN_FILE, DC.CASE_SCRIPT,
                                                     DC.STAGE_SCRIPT, "inputs_check.json")}
        mgr.slurm.states = {"2001": "COMPLETED", "2002": "RUNNING"}
        n = len(mgr.slurm.sbatch())
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert not out["ok"] and "still queued or running (smoke=2002 RUNNING" in out["error"]
        out = json.loads(srv.write_domain_case(str(mgr.run_dir)))
        assert not out["ok"] and "still queued or running" in out["error"]
        inp = json.loads((mgr.input_dir / "case_inputs.json").read_text())[0]["inputs"]
        chk = json.loads(srv.check_domain_inputs(inp["surface"], inp["domain"], inp["finidat"],
                                                 run_dir=str(mgr.run_dir)))
        assert chk["ok"] and chk["pass"] and "not written to" in chk["note"]
        assert len(mgr.slurm.sbatch()) == n
        assert {n: (rec / n).read_text() for n in before} == before

    def test_an_unreadable_chain_record_does_not_open_the_door(self, srv, tmp_path, mgr):
        """chain.json present but unreadable: SLURM cannot be asked about the
        jobs it names, so no second chain is armed over them."""
        self._materialized(srv, tmp_path, mgr)
        assert json.loads(srv.submit_domain_chain(str(mgr.run_dir)))["ok"]
        rec = mgr.run_dir / "domain_case"
        (rec / DC.CHAIN_FILE).write_text('{"segments": [{"jobs": [{"stage": "case", "job_id": "20')
        n = len(mgr.slurm.sbatch())
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert not out["ok"] and "cannot tell whether a chain armed here still runs" in \
            out["error"]
        assert len(mgr.slurm.sbatch()) == n

    def test_a_re_armed_chain_after_the_first_ended(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr)
        assert json.loads(srv.submit_domain_chain(str(mgr.run_dir)))["ok"]
        rec = mgr.run_dir / "domain_case"
        _passed(rec, "smoke", "2002")
        _year_record(rec, "y2015", "2003", 2015)
        _year_record(rec, "y2016", "2004", 2016, ok=False)
        mgr.slurm.states = {k: "COMPLETED" for k in ("2001", "2002", "2003", "2005")}
        mgr.slurm.states["2004"] = "FAILED"
        p = mgr.input_dir / "domain_chain.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), years=[2016], with_case=False,
                                     smoke=False)))
        (Path(json.loads((mgr.input_dir / "case_inputs.json").read_text())[0]["case_dir"])
         .mkdir(parents=True))                            # the case the first chain built
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert out["ok"] and out["job_ids"] == ["2006", "2007"]
        st = json.loads(srv.check_domain_chain(str(mgr.run_dir)))
        assert st["years_passed"] == [2015] and st["active"]

    def test_a_chain_that_would_create_an_existing_case_is_refused(self, srv, tmp_path, mgr):
        self._materialized(srv, tmp_path, mgr)
        (tmp_path / "cases" / "new_case").mkdir()
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert not out["ok"] and "which exists" in out["error"]
        assert not mgr.slurm.sbatch()

    def test_a_node_local_run_directory_is_refused(self, srv, tmp_path, mgr, monkeypatch):
        self._materialized(srv, tmp_path, mgr)
        monkeypatch.setattr(srv, "_shared_fs", lambda p: f"{p} is on /dev/sda1, node-local")
        out = json.loads(srv.submit_domain_chain(str(mgr.run_dir)))
        assert not out["ok"] and "node-local" in out["error"]
        assert not mgr.slurm.sbatch()

    def test_the_capability_report_names_the_whole_domain_tools(self, srv):
        d = json.loads(srv.describe_elm_capabilities())
        domain = [s["tool"] for s in d["workflow"] if "whole-domain only" in str(s["step"])]
        assert domain == ["read_domain_recipe", "check_domain_inputs", "write_domain_case",
                          "submit_domain_chain", "check_domain_chain"]
        assert "whole-domain" in d["model"]
        assert "say only whether ELM will run them and whether they ran" in \
            json.dumps(d["does_not"])


def test_a_forgotten_patch_cannot_reach_sbatch_from_a_node_local_dir(srv, tmp_path, monkeypatch):
    """Nothing patched but SLURM itself, which is a directory of FAKE commands
    first on PATH (the test stops unless every name resolves to the fake): a
    test under tmp_path that forgot its patches is refused before any sbatch."""
    from elm_exp_manager import ELMExpManager
    if not subprocess.check_output(["df", "-P", str(tmp_path)], text=True).splitlines()[-1] \
            .split()[0].startswith("/dev/"):
        pytest.skip("this machine's tmp is not a local disk")
    fake = tmp_path / "fakebin"
    fake.mkdir()
    log = tmp_path / "slurm_calls.log"
    for name in ("sbatch", "scancel", "squeue", "sacct"):
        p = fake / name
        p.write_text(f'#!/bin/bash\necho "$(basename $0) $*" >> {log}\n'
                     f'[ "$(basename $0)" = sbatch ] && echo 900001\nexit 0\n')
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    for name in ("sbatch", "scancel", "squeue", "sacct"):
        if shutil.which(name) != str(fake / name):
            pytest.fail(f"{name} does not resolve to the fake; stopping before any call")
    m = ELMExpManager(base_output_dir=str(tmp_path / "out"))
    cd = _source_case(tmp_path)
    cfg = _config(srv, tmp_path, cd)
    m._materialize(dict(cfg["strategy"]), cfg)
    out = json.loads(srv.submit_domain_chain(str(m.run_dir)))
    assert not out["ok"] and "node-local" in out["error"]
    calls = log.read_text().splitlines() if log.is_file() else []
    assert not [c for c in calls if c.startswith("sbatch ")]


# ── the two copies of the SLURM query ────────────────────────────────────────
def _capture_sacct(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(list(cmd))
        out = "" if cmd[0] == "squeue" else " COMPLETED \n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/fake/{name}")
    monkeypatch.setattr(subprocess, "run", run)
    return seen


def _assert_a_year_back(seen):
    sacct = next(c for c in seen if c[0] == "sacct")
    since = datetime.date.fromisoformat(sacct[sacct.index("-S") + 1])
    assert (datetime.date.today() - since).days == DC.SACCT_LOOKBACK_DAYS


def test_the_base_asks_sacct_back_past_midnight(monkeypatch):
    from core.exp_manager_base import ExperimentManagerBase
    seen = _capture_sacct(monkeypatch)
    assert ExperimentManagerBase._slurm_state("777") == "COMPLETED"
    _assert_a_year_back(seen)


def test_the_server_asks_sacct_back_past_midnight(srv, monkeypatch):
    seen = _capture_sacct(monkeypatch)
    assert srv._slurm_state("777") == "COMPLETED"
    _assert_a_year_back(seen)


# ── the base's and the gate's rules ──────────────────────────────────────────
@pytest.mark.parametrize("strategy,brief", [
    ({"archetype": "whole_domain"}, None), ({"sampling": {"approach": "whole_domain"}}, None),
    ({}, {"design_archetype": "whole_domain"}), ({"domain_case": {"from_case": "/x"}}, None)])
def test_each_source_says_whole_domain(strategy, brief):
    from core.exp_manager_base import _is_whole_domain
    assert _is_whole_domain(strategy, brief)
    assert not _is_whole_domain({"archetype": "site"}, {"design_archetype": "site"})


def _reception(**run_settings):
    return {"brief": {"domain": {"bbox": {"min_lon": -121.6, "min_lat": 46.4, "max_lon": -121.5,
                                          "max_lat": 46.5}, "name": "x"},
                      "run_settings": {"resolved_period": {"yr_start": 2015, "yr_end": 2016},
                                       **run_settings}},
            "grid": {}, "observations": {}}


class TestStrategyCheck:

    def test_one_case_whatever_the_count(self):
        from core.strategy_check import check
        rep, st = check(_reception(requested_n_columns=7),
                        {"archetype": "whole_domain", "sampling": {"n_columns": 5},
                         "domain_case": {"from_case": "/x", "case": "y"}})
        assert st["sampling"]["n_columns"] == 1 and not rep["stop"]
        assert any("one case, not 5; using 1" in c for c in rep["corrections"])
        assert not any("using the 7" in c for c in rep["corrections"])
        assert rep["checked"]["approach"] == "whole_domain"

    def test_the_planners_approach_alone_is_whole_domain(self):
        from core.strategy_check import check
        rep, _ = check(_reception(), {"sampling": {"approach": "whole_domain"}})
        assert any("names no case" in s for s in rep["stop"])

    def test_a_copy_without_a_new_name_is_stopped(self):
        from core.strategy_check import check
        rep, _ = check(_reception(), {"archetype": "whole_domain",
                                      "domain_case": {"from_case": "/x"}})
        assert any("names no new case" in s for s in rep["stop"])
