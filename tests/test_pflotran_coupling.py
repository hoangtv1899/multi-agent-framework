"""Offline tests for the ELM -> PFLOTRAN coupling step (no PFLOTRAN binary).

Reception and the planner have described this coupling in their prompts for
months — design_archetype "coupling", a coupling_design block naming the driver
— but nothing executed it. These tests cover the decision layer that turns that
plan into runs: WHEN step 4d fires, when it refuses, and whether the prose it
writes about the driving ELM run is true. Actually solving Richards needs the
binary and is exercised by tools/build_pflotran_cases.py --run.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elm_exp_manager import ELMExpManager  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


apc = _load("apc_mod", "tools/analyze_pflotran_coupled.py")
bpc = _load("bpc_mod", "tools/build_pflotran_cases.py")


@pytest.fixture
def mgr(tmp_path):
    """A manager whose run_dir has NO columns.json, so step 4d always bails
    right after deciding it wants to run — which is exactly the decision under
    test, with no PFLOTRAN binary involved."""
    m = ELMExpManager.__new__(ELMExpManager)
    m.run_dir = tmp_path / "run"
    m.analysis_dir = m.run_dir / "04_analysis"
    m.analysis_dir.mkdir(parents=True)
    return m


HEADER = "STEP 4d"


class TestCouplingTrigger:
    def test_plain_elm_plan_does_not_couple(self, mgr, capsys):
        assert mgr._couple_pflotran({"model_choice": {"design_archetype": "site"}},
                                    {}) is False
        assert HEADER not in capsys.readouterr().out

    def test_empty_plan_does_not_couple(self, mgr, capsys):
        assert mgr._couple_pflotran({}, {}) is False
        assert HEADER not in capsys.readouterr().out

    def test_coupling_design_block_triggers_it(self, mgr, capsys):
        """The planner's own output is the trigger — no extra config needed."""
        plan = {"coupling_design": {"from_model": "ELM", "to_model": "PFLOTRAN",
                                    "driver": "ELM QINFL -> top recharge BC"}}
        mgr._couple_pflotran(plan, {})
        out = capsys.readouterr().out
        assert HEADER in out
        assert "ELM QINFL -> top recharge BC" in out      # driver echoed

    def test_coupling_archetype_triggers_it(self, mgr, capsys):
        mgr._couple_pflotran({"model_choice": {"design_archetype": "coupling"}}, {})
        assert HEADER in capsys.readouterr().out

    def test_config_can_force_it(self, mgr, capsys):
        mgr._couple_pflotran({}, {"pflotran": {"run": True}})
        assert HEADER in capsys.readouterr().out

    def test_missing_columns_is_non_fatal(self, mgr, capsys):
        """The ELM study must stand even if coupling cannot start."""
        assert mgr._couple_pflotran({"coupling_design": {"driver": "x"}}, {}) is False
        assert "no columns.json" in capsys.readouterr().out

    def test_empty_columns_list_is_non_fatal(self, mgr, capsys):
        mgr.run_dir.mkdir(exist_ok=True)
        (mgr.run_dir / "columns.json").write_text(json.dumps({"columns": []}))
        assert mgr._couple_pflotran({"coupling_design": {"driver": "x"}}, {}) is False
        assert "no columns.json" in capsys.readouterr().out


class TestElmContextHonesty:
    """The coupled figure and study.json describe the ELM run that forced them.

    That prose used to be hardcoded as "Naches" and "warm-started NLDAS run",
    which is wrong for any other basin and wrong for a cold run — the same class
    of bug as a hardcoded forcing label on an axis.
    """
    def _run(self, tmp_path, couplers, basin="Testwater"):
        rd = tmp_path / "elm"
        rd.mkdir()
        (rd / "reception_brief.json").write_text(
            json.dumps({"domain": {"name": basin}}))
        (rd / "run_plan.json").write_text(
            json.dumps({"CONDITIONS_COUPLERS": couplers}))
        return {"flux_from": str(rd)}

    def test_cold_run_is_reported_cold(self, tmp_path):
        sc = self._run(tmp_path, [{"EXPERIMENT": "col_01"}, {"EXPERIMENT": "col_02"}])
        basin, init = apc._elm_context(sc)
        assert basin == "Testwater"
        assert init == "cold-started"

    def test_warm_run_reports_the_column_count(self, tmp_path):
        sc = self._run(tmp_path, [{"EXPERIMENT": "col_01", "FINIDAT": "/w/a.nc"},
                                  {"EXPERIMENT": "col_02", "FINIDAT": "/w/b.nc"}])
        _, init = apc._elm_context(sc)
        assert init == "warm-started (2/2 columns)"

    def test_partial_warm_run_is_not_rounded_up(self, tmp_path):
        sc = self._run(tmp_path, [{"EXPERIMENT": "col_01", "FINIDAT": "/w/a.nc"},
                                  {"EXPERIMENT": "col_02"}])
        _, init = apc._elm_context(sc)
        assert init == "warm-started (1/2 columns)"

    def test_no_driving_run_claims_nothing(self):
        basin, init = apc._elm_context({})
        assert basin is None
        assert "not recorded" in init

    def test_unreadable_driving_run_claims_nothing(self, tmp_path):
        basin, init = apc._elm_context({"flux_from": str(tmp_path / "gone")})
        assert basin is None
        assert "not recorded" in init


class TestBuildEnsembleWiring:
    def test_limit_truncates_and_manifest_is_written(self, tmp_path, monkeypatch):
        """build_ensemble writes pflotran_cases.json even without running."""
        cols = [{"id": f"col_{i:02d}", "lat": 46.0, "lon": -121.0,
                 "fan_wtd_m": 2.0, "elevation_m": 900,
                 "soil_profile": {"layers": [
                     {"component": "c1", "depth_top_cm": 0, "depth_bot_cm": 100,
                      "van_genuchten": {"theta_s": 0.45, "theta_r": 0.05,
                                        "ksat_ms": 1e-6, "alpha_per_m": 1e-4,
                                        "m": 0.4}}]}}
                for i in range(1, 6)]
        res = bpc.build_ensemble(cols, tmp_path / "out", run=False, limit=2,
                                 quiet=True)
        assert len(res["cases"]) == 2
        assert (tmp_path / "out" / "pflotran_cases.json").exists()
        # a standalone (non-coupled) ensemble records its scenario recharge
        assert res["scenario"]["recharge_mm_yr"] == 100.0
        assert res["scenario"]["flux_from"] is None

    def test_each_column_gets_its_own_case_dir(self, tmp_path):
        cols = [{"id": f"col_{i:02d}", "lat": 46.0, "lon": -121.0,
                 "fan_wtd_m": 2.0, "elevation_m": 900, "soil_profile": None}
                for i in range(1, 4)]
        res = bpc.build_ensemble(cols, tmp_path / "out", run=False, quiet=True)
        dirs = [m["case_dir"] for m in res["cases"]]
        assert len(set(dirs)) == 3


# ─────────────────────────────────────────────────────────────────────
# Phase 5 — the ensemble as a scheduler job
# ─────────────────────────────────────────────────────────────────────
class _RxClient:
    def __init__(self, **replies):
        self.timeout = 300.0
        self.calls = []
        self.replies = replies

    def call_tool_json(self, tool, args):
        self.calls.append((tool, args))
        return self.replies.get(tool)


class TestSubmittingIsOptInForPFLOTRAN:
    """Unlike ELM (D5), the submitted path is NOT the default here. A framework
    column solves in ~0.3 s — it is why NEEDS_SCHEDULER is False — so a queue
    slot costs more than the solve for the ordinary ensemble."""

    def _mgr(self, tmp_path):
        from pflotran_exp_manager import PFLOTRANExpManager
        return PFLOTRANExpManager(base_output_dir=str(tmp_path))

    def test_the_inline_path_is_still_the_default(self, tmp_path):
        src = (ROOT / "mcp" / "pflotran-mcp" / "pflotran_exp_manager.py").read_text()
        i = src.index("def _run(")
        body = src[i:i + 2500]
        assert 'config.get("submit")' in body, \
            "submitting must be asked for, not assumed"

    def test_submitting_returns_a_pending(self, tmp_path):
        from core.exp_manager_base import Pending
        m = self._mgr(tmp_path)
        case = tmp_path / "col_01"
        case.mkdir()
        (case / "col_01.in").write_text("SIMULATION\nEND\n")
        c = _RxClient(submit_pflotran_ensemble={"job_id": "770699",
                                                "n_decks": 1})
        out = m._submit_via_mcp([{"id": "col_01", "case_dir": str(case)}],
                                c, 300.0, 4, {})
        assert isinstance(out, Pending) and out.job_id == "770699"

    def test_a_submission_with_no_job_id_raises(self, tmp_path):
        """A Pending nobody can poll is worse than a failure."""
        m = self._mgr(tmp_path)
        case = tmp_path / "col_01"
        case.mkdir()
        (case / "col_01.in").write_text("x")
        c = _RxClient(submit_pflotran_ensemble={"error": "sbatch missing"})
        with pytest.raises(RuntimeError, match="could not submit"):
            m._submit_via_mcp([{"id": "col_01", "case_dir": str(case)}],
                              c, 300.0, 4, {})

    def test_poll_waits_while_the_job_is_active(self, tmp_path):
        m = self._mgr(tmp_path)
        c = _RxClient(check_pflotran_job={"active": True, "state": "RUNNING"})
        assert m._poll({"job_id": "770699", "stage": "run"}, [],
                       {"mcp_clients": {"pflotran": c}}) is None
        assert "collect_pflotran_results" not in [t for t, _ in c.calls]

    def test_poll_collects_and_attributes(self, tmp_path):
        m = self._mgr(tmp_path)
        case = tmp_path / "col_01"
        (case).mkdir()
        (case / "col_01.in").write_text("x")
        (case / "col_01.tec").write_text("x")
        c = _RxClient(
            check_pflotran_job={"active": False, "state": "COMPLETED"},
            collect_pflotran_results={
                "status": "completed",
                "results_by_input": {
                    str(case / "col_01.in"): {"validation_status": "success",
                                              "exit_codes": [0],
                                              "execution_time": 0.31}}})
        rows = m._poll({"job_id": "770699", "stage": "run"},
                       [{"id": "col_01", "case_dir": str(case)}],
                       {"mcp_clients": {"pflotran": c}})
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert rows[0]["runtime_seconds"] == 0.31
        assert rows[0]["run_via"] == "mcp"

    def test_a_finished_job_with_nothing_attributable_raises(self, tmp_path):
        """The scheduler being done is not a reason to keep waiting, and it is
        not a reason to invent rows either."""
        m = self._mgr(tmp_path)
        c = _RxClient(
            check_pflotran_job={"active": False, "state": "COMPLETED"},
            collect_pflotran_results={"status": "incomplete",
                                      "error": "no result file"})
        with pytest.raises(RuntimeError, match="no attributable results"):
            m._poll({"job_id": "770699", "stage": "run"}, [],
                    {"mcp_clients": {"pflotran": c}})

    def test_polling_without_a_client_raises(self, tmp_path):
        """The job was submitted through the server; only the server can
        collect it."""
        m = self._mgr(tmp_path)
        with pytest.raises(RuntimeError, match="no pflotran client"):
            m._poll({"job_id": "770699", "stage": "run"}, [], {})

    def test_both_paths_share_one_attribution(self):
        """The inline and submitted paths return the same shape by design; a
        second copy of the mapping is a second place for it to drift."""
        src = (ROOT / "mcp" / "pflotran-mcp" / "pflotran_exp_manager.py").read_text()
        assert src.count("def _rows_from_mcp") == 1
        assert src.count("self._rows_from_mcp(") == 2
