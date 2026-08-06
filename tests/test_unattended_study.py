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
# ROUTING
# ─────────────────────────────────────────────────────────────────────
def _mgr(tmp_path):
    m = ELMExpManager.__new__(ELMExpManager)
    m.run_dir = tmp_path
    m.input_dir = tmp_path / "01_inputs"
    m.analysis_dir = tmp_path / "04_analysis"
    for d in (m.input_dir, m.analysis_dir):
        d.mkdir(parents=True, exist_ok=True)
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
                                     {"notify_email": "a@b.gov"}, object())
        assert called["tool"] == "run_elm_study"
        assert isinstance(out, Pending) and out.job_id == "9001"
        assert out.detail.get("scope") == "study"

    def test_there_is_no_split_flow_to_fall_back_to(self, tmp_path, monkeypatch):
        """Even asked for explicitly. Splitting build from run made the user
        the scheduler for a 25-40 minute study; the debugging value of
        stopping between stages did not pay for three round trips."""
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps([{"case_name": "c1"}]))
        called = {}
        def _stub(c, tool, args, budget=None):
            called["tool"] = tool
            return {"job_id": "9002"}
        monkeypatch.setattr(ELMExpManager, "_mcp_call", staticmethod(_stub))
        for cfg in ({}, {"unattended": False}, {"split": True}):
            called.clear()
            m._build_cases_via_mcp([{"case_name": "c1"}], cfg, object())
            assert called["tool"] == "run_elm_study", f"config {cfg} found a split"

    def test_the_mcp_still_publishes_the_split_tools(self):
        """Removed from the FRAMEWORK's path, not from the server: an agent
        driving ELM without this framework may still want them, and
        _submit_via_mcp uses submit_elm_ensemble to recover a study whose job
        died after building but before running."""
        src = (ROOT / "mcp" / "elm-mcp" / "main.py").read_text()
        assert "def build_elm_cases(" in src
        assert "def submit_elm_ensemble(" in src


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
        system interpreter that cannot import the framework.

        EVERY python invocation, not just one. The first version of this
        asserted the conda path merely appeared somewhere in the script — and
        passed while the finalize and notify lines were calling a literal
        `$PY`, a variable that exists only in the generating shell and is
        empty inside the job.
        """
        sb = self._generate(tmp_path)
        assert "/.conda/envs/ideas/bin/python3" in sb
        offenders = [ln.strip() for ln in sb.splitlines()
                     if ".py" in ln and not ln.strip().startswith("#")
                     and "/.conda/envs/" not in ln]
        assert not offenders, (
            "these run a .py with an interpreter that is not resolved at "
            "generation time: " + "; ".join(offenders))

    def test_mail_lines_only_when_an_address_was_given(self, tmp_path):
        assert "--mail-user=who@example.gov" in self._generate(tmp_path)
        assert "--mail-user" not in self._generate(tmp_path, email=None, tag="b")

    def test_delivery_is_slurms_job_not_the_nodes(self, tmp_path):
        """The compute node cannot mail. It has no /bin/mail (job 770819) and
        its local Postfix ACCEPTS messages it cannot relay — smtplib returned
        success and nothing was ever delivered (770821, 770905). Slurm's
        MailProg runs on the controller and works, so delivery is its job and
        the exit code is what makes its subject meaningful."""
        sb = self._generate(tmp_path, email="who@example.gov")
        assert "--mail-type=END,FAIL" in sb
        assert "--mail-user=who@example.gov" in sb
        notify = [l for l in sb.splitlines() if "notify_study.py" in l][0]
        assert "--mail" not in notify, "the node must not try to send mail"

    def test_the_job_is_named_after_the_study(self, tmp_path):
        """Slurm gives you a subject line and nothing else, so it has to say
        WHICH run finished."""
        sb = self._generate(tmp_path)
        jline = [l for l in sb.splitlines() if l.startswith("#SBATCH -J")][0]
        assert jline.strip() != "#SBATCH -J elm_study", "the run is not identified"
        assert "elm_study." in jline

    def test_the_job_exits_with_the_studys_status_not_the_scripts(self, tmp_path):
        """Slurm reported job 770816 as COMPLETED/ExitCode 0 for a study whose
        package stage had failed. The subject line is the only signal in the
        unattended flow, so the exit code has to mean something."""
        sb = self._generate(tmp_path)
        assert "exit $STUDY_RC" in sb

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


# ─────────────────────────────────────────────────────────────────────
# TELLING THE USER WHAT IS ABOUT TO HAPPEN
# ─────────────────────────────────────────────────────────────────────
class TestTheAnnouncement:
    """ELM has no split flow, so the user must be told the run is unattended,
    where the results will land, and be offered a way to be notified."""

    def _announce(self, tmp_path, monkeypatch, cfg=None, tty=False,
                  reply="", env=None, prefs=None):
        monkeypatch.setenv("IDEAS_PREFS_DIR", str(tmp_path / "prefs"))
        monkeypatch.delenv("IDEAS_NOTIFY_EMAIL", raising=False)
        if env:
            monkeypatch.setenv("IDEAS_NOTIFY_EMAIL", env)
        if prefs:
            import core.notify_prefs as P
            monkeypatch.setattr(P, "PREFS_DIR", tmp_path / "prefs")
            monkeypatch.setattr(P, "PREFS_FILE", tmp_path / "prefs" / "notify.json")
            P.remember_email(prefs)
        m = _mgr(tmp_path)
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda s: tty})())
        monkeypatch.setattr("builtins.input", lambda _p="": reply)
        return m._announce([1, 2], cfg or {})

    def test_it_names_where_the_results_will_be(self, tmp_path, monkeypatch, capsys):
        self._announce(tmp_path, monkeypatch)
        out = capsys.readouterr().out
        assert "unattended" in out
        assert "04_analysis" in out

    def test_no_tty_never_blocks(self, tmp_path, monkeypatch):
        """This path also runs from scripts and from --resume. A blocking
        input() there would hang a job nobody is watching."""
        called = []
        monkeypatch.setattr("builtins.input",
                            lambda _p="": called.append(1) or "")
        got = self._announce(tmp_path, monkeypatch, tty=False,
                             env="env@x.gov")
        assert got == "env@x.gov"
        assert not called, "prompted without a terminal"

    def test_at_a_tty_a_typed_address_wins_and_is_remembered(self, tmp_path, monkeypatch):
        got = self._announce(tmp_path, monkeypatch, tty=True,
                             reply="typed@x.gov", prefs="old@x.gov")
        assert got == "typed@x.gov"
        import core.notify_prefs as P
        assert P.remembered_email() == "typed@x.gov"

    def test_enter_keeps_the_remembered_address(self, tmp_path, monkeypatch):
        got = self._announce(tmp_path, monkeypatch, tty=True, reply="",
                             prefs="kept@x.gov")
        assert got == "kept@x.gov"

    def test_dash_means_no_email(self, tmp_path, monkeypatch):
        got = self._announce(tmp_path, monkeypatch, tty=True, reply="-",
                             prefs="kept@x.gov")
        assert got == ""
        import core.notify_prefs as P
        assert P.remembered_email() is None

    def test_an_explicit_config_address_is_not_second_guessed(self, tmp_path, monkeypatch):
        """A caller that passed an address programmatically has already
        decided; prompting over the top would hang an automated run."""
        called = []
        monkeypatch.setattr("builtins.input", lambda _p="": called.append(1) or "x")
        got = self._announce(tmp_path, monkeypatch, tty=True,
                             cfg={"notify_email": "cfg@x.gov"})
        assert got == "cfg@x.gov" and not called
