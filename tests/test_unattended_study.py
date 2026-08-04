"""The whole study as ONE job — and the ledger it leaves behind.

The split flow costs three invocations: submit the build, come back and submit
the run, come back and analyze. Unattended mode collapses that to one job that
does all three and mails you when the analysis is written.

The stage that this file exists to protect is the LAST one. A job that
finalizes its own output cannot use the ordinary resume path, because resume
polls the scheduler and the job doing the polling IS the job being polled:
squeue says RUNNING, _poll returns None, and the run stops one line short of
the analysis it was submitted to produce. adopt_completed_run() takes the
evidence off the filesystem instead.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from core.elm_exp_manager import ELMExpManager
from core.exp_manager_base import Pending

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# WHICH MODE
# ─────────────────────────────────────────────────────────────────────
class TestChoosingUnattended:

    def test_off_unless_asked(self, monkeypatch):
        """Unattended gives up the chance to inspect the cases between
        building and running — which is exactly what you want while something
        is wrong. So it is opt-in."""
        monkeypatch.delenv("IDEAS_UNATTENDED", raising=False)
        assert ELMExpManager._unattended({}) is False

    def test_env_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("IDEAS_UNATTENDED", "1")
        assert ELMExpManager._unattended({}) is True

    def test_config_overrides_the_environment(self, monkeypatch):
        """One run must be able to opt out of a machine-wide default."""
        monkeypatch.setenv("IDEAS_UNATTENDED", "1")
        assert ELMExpManager._unattended({"unattended": False}) is False


class TestTheNotificationAddress:

    def test_never_guessed(self, monkeypatch):
        """A wrong address sends the one signal the unattended flow depends on
        silently nowhere. Absent is better than invented."""
        monkeypatch.delenv("IDEAS_NOTIFY_EMAIL", raising=False)
        assert ELMExpManager._notify_email({}) == ""

    def test_config_beats_environment(self, monkeypatch):
        monkeypatch.setenv("IDEAS_NOTIFY_EMAIL", "env@x.gov")
        assert ELMExpManager._notify_email({"notify_email": "cfg@x.gov"}) == "cfg@x.gov"


# ─────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────
def _mgr(tmp_path):
    m = ELMExpManager.__new__(ELMExpManager)
    m.run_dir = tmp_path
    m.input_dir = tmp_path / "01_inputs"
    m.input_dir.mkdir(parents=True, exist_ok=True)
    return m


class TestRouting:

    def test_unattended_calls_run_elm_study_not_build(self, tmp_path, monkeypatch):
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps([{"case_name": "c1"}]))
        called = {}
        def _stub(c, tool, args, budget=None):
            called["tool"] = tool
            return {"job_id": "9001"}
        monkeypatch.setattr(ELMExpManager, "_mcp_call", staticmethod(_stub))
        out = m._build_cases_via_mcp([{"case_name": "c1"}],
                                     {"unattended": True,
                                      "notify_email": "a@b.gov"}, object())
        assert called["tool"] == "run_elm_study"
        assert isinstance(out, Pending) and out.job_id == "9001"
        assert out.detail.get("scope") == "study"

    def test_split_mode_still_calls_build_elm_cases(self, tmp_path, monkeypatch):
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps([{"case_name": "c1"}]))
        called = {}
        def _stub(c, tool, args, budget=None):
            called["tool"] = tool
            return {"job_id": "9002"}
        monkeypatch.setattr(ELMExpManager, "_mcp_call", staticmethod(_stub))
        out = m._build_cases_via_mcp([{"case_name": "c1"}],
                                     {"unattended": False}, object())
        assert called["tool"] == "build_elm_cases"
        assert out.detail.get("scope") is None


# ─────────────────────────────────────────────────────────────────────
# ADOPTING THE FINISHED STAGES
# ─────────────────────────────────────────────────────────────────────
class TestAdoptCompletedRun:
    """The in-job finalize: evidence from the filesystem, not the scheduler."""

    def _seed(self, tmp_path, with_built=True):
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps(
            [{"case_name": "col_01"}, {"case_name": "col_02"}]))
        if with_built:
            cd1 = tmp_path / "case_col_01"; (cd1 / "run").mkdir(parents=True)
            cd2 = tmp_path / "case_col_02"; (cd2 / "run").mkdir(parents=True)
            (cd1 / "run" / "x.elm.h0.2020-01-01-00000.nc").write_text("")
            (m.input_dir / m.BUILT_CASES).write_text(json.dumps(
                {"ok": True, "cases": [
                    {"case_name": "col_01", "case_dir": str(cd1)},
                    {"case_name": "col_02", "case_dir": str(cd2)}]}))
        return m

    def test_it_reattaches_where_each_case_landed(self, tmp_path):
        """Case dirs carry a creation timestamp, so they cannot be derived
        from the case names. Without re-attaching them _extract globs against
        a case_dir of None and a finished ensemble reads as a total failure.
        """
        m = self._seed(tmp_path)
        adopted = m.adopt_completed_run()
        assert adopted["build_cases"] == 2
        rehydrated = m._rehydrate_case_inputs()
        assert all(e.get("case_dir") for e in rehydrated)

    def test_it_marks_both_job_shaped_stages_done(self, tmp_path):
        m = self._seed(tmp_path)
        m.adopt_completed_run()
        stages = m._load_state()["stages"]
        assert stages["build_cases"]["status"] == "done"
        assert stages["run"]["status"] == "done"
        assert stages["run"]["adopted_from"] == "disk"

    def test_outcomes_come_from_history_files(self, tmp_path):
        """col_01 wrote a history file, col_02 did not — a 1/2 ensemble, and
        the ledger has to say so rather than claim both."""
        m = self._seed(tmp_path)
        m.adopt_completed_run()
        assert m._load_state()["stages"]["run"]["n_ok"] == 1

    def test_missing_built_cases_is_not_fatal(self, tmp_path, capsys):
        """The model output may still be on disk; refusing to finalize over a
        missing manifest would throw away a finished ensemble."""
        m = self._seed(tmp_path, with_built=False)
        adopted = m.adopt_completed_run()
        assert "build_cases" not in adopted


# ─────────────────────────────────────────────────────────────────────
# THE JOB SCRIPT
# ─────────────────────────────────────────────────────────────────────
class TestTheGeneratedJob:
    """Structural checks on run_study.sh's output — cheap, and they catch the
    two mistakes that would silently break the unattended promise."""

    def _generate(self, tmp_path, email="who@example.gov", tag="a"):
        rd = tmp_path / f"run_{tag}"
        (rd / "01_inputs").mkdir(parents=True)
        (rd / "01_inputs" / "case_inputs.json").write_text(
            json.dumps([{"case_name": "c1"}]))
        cmd = ["bash", str(ROOT / "tools" / "run_study.sh"), str(rd), "--dry"]
        if email:
            cmd += ["-m", email]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr
        return (rd / "run_study.sbatch").read_text()

    def test_it_finalizes_rather_than_resumes(self, tmp_path):
        """--resume here would poll the job that is running it, see itself
        ACTIVE, and stop before the analysis."""
        sb = self._generate(tmp_path)
        assert "--finalize" in sb
        assert "--resume" not in sb.split("|| echo")[0]

    def test_the_interpreter_is_named_not_looked_up(self, tmp_path):
        """The job gets a login shell with no conda env, where `python3` is a
        system interpreter that cannot import the framework."""
        sb = self._generate(tmp_path)
        assert "/.conda/envs/ideas/bin/python3" in sb

    def test_mail_lines_only_when_an_address_was_given(self, tmp_path):
        assert "--mail-user=who@example.gov" in self._generate(tmp_path)
        assert "--mail-user" not in self._generate(tmp_path, email=None, tag="b")

    def test_a_failed_column_does_not_abort_before_the_analysis(self, tmp_path):
        """set -e would lose a 17/19 result over the two that failed."""
        sb = self._generate(tmp_path)
        lines = [ln.strip() for ln in sb.splitlines()]
        assert "set -e" not in lines and "set -euo pipefail" not in lines
        assert "set -u" in lines


# ─────────────────────────────────────────────────────────────────────
# THE ANALYZER'S VERDICT REACHES THE LEDGER
# ─────────────────────────────────────────────────────────────────────
class TestAFailedAnalysisIsNotRecordedAsDone:
    """Analyzer.run() reports a failed step by RETURNING {"error": ...} rather
    than raising, so the manager's try/except never saw it and marked the
    stage done regardless.

    Found live on 2026-08-03 (job 770816): step 0 printed "context failed",
    04_analysis held one partial file, and the ledger said analyze: done.
    Harmless while a human watched the console. Not harmless once the
    unattended flow mails "your analysis is ready" off that same entry.
    """

    def _run_analyze_block(self, tmp_path, monkeypatch, verdict):
        from core import exp_manager_base as B

        class M(B.ExperimentManagerBase):
            MODEL = "elm"
            NEEDS_CASE_BUILD = False
            # the plan already carries its payload, so materialize is a no-op
            def _already_executable(self, plan):        return True
            def _build_case_inputs(self, plan, config): return [{"case_name": "c1"}]
            def _run(self, experiments, config):        return {"c1": True}
            def _extract(self, experiments, **kw):      return type("R", (), {"results": [1]})()
            def _package(self, plan, analyzer, config): pass
            def _save_llm_input(self, plan, analyzer):  pass

        fake = type("A", (), {"__init__": lambda s, rd: None,
                              "run": lambda s, **kw: verdict})
        mod = type(sys)("agents.analyzer"); mod.Analyzer = fake
        monkeypatch.setitem(sys.modules, "agents.analyzer", mod)

        m = M(base_output_dir=str(tmp_path), run_dir=str(tmp_path / "r"))
        m.execute_plan({"x": 1}, {})
        return m._load_state()["stages"]["analyze"]

    def test_an_error_verdict_is_recorded_as_failed(self, tmp_path, monkeypatch):
        st = self._run_analyze_block(
            tmp_path, monkeypatch,
            {"error": "no experiment.json", "steps": {"context": False}})
        assert st["status"] == "failed"
        assert "no experiment.json" in st["error"]

    def test_a_clean_verdict_is_still_done(self, tmp_path, monkeypatch):
        st = self._run_analyze_block(
            tmp_path, monkeypatch, {"steps": {"context": True, "report": True}})
        assert st["status"] == "done"

    def test_a_backend_returning_nothing_does_not_crash(self, tmp_path, monkeypatch):
        """Older Analyzers returned None; that must read as success, not as a
        TypeError at the last stage of a finished run."""
        st = self._run_analyze_block(tmp_path, monkeypatch, None)
        assert st["status"] == "done"
