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
