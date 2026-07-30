#!/usr/bin/env python3
"""
Model dispatch — tests/test_backends.py

The workflow can run more than one model, and which one it runs is a choice
with no safe default behaviour when it goes wrong: every file in a run
directory agrees with whatever actually ran, so a misdispatch is not visible
by inspecting the output.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core import backends                                    # noqa: E402


class TestTheTableIsTheOneSourceOfNames:

    def test_both_backends_resolve_to_their_manager(self):
        assert backends.get("elm").__name__ == "ELMExpManager"
        assert backends.get("pflotran").__name__ == "PFLOTRANExpManager"

    def test_names_lists_what_get_accepts(self):
        """argparse choices come from names(); a name it advertises and
        cannot resolve would fail after reception and planning have run."""
        for n in backends.names():
            assert backends.get(n) is not None

    def test_the_default_is_a_real_backend(self):
        assert backends.DEFAULT in backends.names()

    def test_case_and_whitespace_do_not_change_the_model(self):
        assert backends.get("  PFLOTRAN ") is backends.get("pflotran")

    def test_an_unknown_name_raises_rather_than_falling_back(self):
        """A typo'd --model that quietly ran the default would report a
        completed ELM study to someone who asked for PFLOTRAN."""
        with pytest.raises(ValueError, match="unknown model"):
            backends.get("pflotan")
        with pytest.raises(ValueError):
            backends.get("")


class TestEachModelGetsOnlyWhatItUnderstands:

    BASE = {"brief": {}, "reception": {}, "strategy": {}, "mcp_clients": {}}
    PERIOD = {"yr_start": 2019, "yr_end": 2019}

    def test_the_shared_keys_are_untouched(self):
        for name in backends.names():
            cfg = backends.config_for(name, self.BASE, period=self.PERIOD)
            for k in self.BASE:
                assert k in cfg

    def test_base_is_not_modified_in_place(self):
        before = dict(self.BASE)
        backends.config_for("elm", self.BASE, period=self.PERIOD)
        assert self.BASE == before

    def test_warm_start_does_not_reach_pflotran(self):
        """PFLOTRAN has no restart file and no donor gridcell. A run recording
        a warm start it never performed is a false statement about its own
        initial condition, which here is the Fan 2013 water table."""
        cfg = backends.config_for("pflotran", self.BASE, period=self.PERIOD)
        assert "warm_start" not in cfg

    def test_elm_warm_starts_by_default_and_cold_is_an_opt_out(self):
        warm = backends.config_for("elm", self.BASE, period=self.PERIOD)
        assert warm["warm_start"]["source"] == "conus"
        cold = backends.config_for("elm", self.BASE, period=self.PERIOD,
                                   initialization={"mode": "cold"})
        assert "warm_start" not in cold

    def test_pflotran_defaults_match_the_standalone_tool(self):
        """A deck built through the workflow and one built from the command
        line must be the same deck."""
        cfg = backends.config_for("pflotran", self.BASE, period=self.PERIOD)
        assert cfg["bottom"] == "fan"
        assert cfg["years"] == 20.0
        assert cfg["depth_cap"] == 50.0

    def test_an_explicit_setting_wins_over_the_default(self):
        cfg = backends.config_for("pflotran", {**self.BASE, "years": 5.0},
                                  period=self.PERIOD)
        assert cfg["years"] == 5.0

    def test_every_backend_records_the_requested_period(self):
        for name in backends.names():
            cfg = backends.config_for(name, self.BASE, period=self.PERIOD)
            assert cfg["yr_start"] == 2019 and cfg["yr_end"] == 2019


class TestTheCoordinatorActuallyDispatches:
    """The dispatch itself, driven through _workflow_design_and_run.

    backends.get() being right is not the same as the coordinator USING it —
    that link is three lines in workflow.py and nothing else covers it.
    """

    @staticmethod
    def _reception():
        return {
            "route": {"action": "design"},
            "user_request": "how does the water table respond to elevation?",
            "brief": {
                "domain": {"name": "Naches", "huc": "17030002",
                           "bbox": {"min_lon": -121.6, "min_lat": 46.5,
                                    "max_lon": -120.3, "max_lat": 47.1}},
                "run_settings": {"resolved_period": {"yr_start": 2019,
                                                     "yr_end": 2019}},
            },
            "grid": {"n_in_basin": 68},
            "observations": {},
        }

    STRATEGY = {
        "archetype": "elevation_gradient",
        "goals": ["how does the water table respond to elevation"],
        "feasibility": {"verdict": "feasible"},
        "sampling": {"n_bands": 3, "per_band": 2, "n_columns": 6,
                     "approach": "stratified by elevation"},
        "validation": [], "requires": [], "coupling": None,
    }

    def _drive(self, tmp_path, model, monkeypatch):
        """Run the coordinator with every PFLOTRAN stage stubbed, and report
        which manager class it built and what config that manager received."""
        import workflow as wf
        from core.pflotran_exp_manager import PFLOTRANExpManager
        from agents.analysis import step3_interpret as _s3

        # The Analyzer's LLM steps are a live external call.
        monkeypatch.setattr(_s3, "investigate_and_interpret",
                            lambda ctx, out_dir, **kw: {
                                "investigation": {"n_succeeded": 0,
                                                  "n_proposed": 0,
                                                  "findings": [],
                                                  "caveats": []},
                                "interpretation": {"verdict": "insufficient",
                                                   "audit": {"n_claims": 0,
                                                             "n_struck": 0},
                                                   "claims": [], "struck": []},
                                "rounds": [], "stopped_because": "stubbed",
                                "n_rounds": 0})

        seen = {}
        cols = [{"id": f"col_{i:02d}", "lat": 46.5 + i * 0.02,
                 "lon": -121.0 + i * 0.02, "elevation_m": 600.0 + i * 100,
                 "band": i % 3 + 1, "fan_wtd_m": 2.0 + i}
                for i in range(1, 7)]

        def fake_materialize(self, plan, config):
            seen["cls"] = type(self).__name__
            seen["config"] = dict(config)
            config = self.check(plan, config)      # the real gate, it is cheap
            (self.run_dir / "columns.json").write_text(
                json.dumps({"columns": cols}, indent=2))
            return {**plan, **self._to_run_plan(plan, cols, config, {})}

        def fake_build(self, plan, config):
            return [{"id": c["id"], "case_dir": str(tmp_path / c["id"])}
                    for c in cols]

        def fake_run(self, experiments, config):
            return experiments

        def fake_extract(self, experiments, plan=None, config=None):
            import types
            rows = [{"case_name": c["id"], "status": "ok",
                     "metrics": {"final_water_table_depth_m": 3.0}}
                    for c in cols]
            return types.SimpleNamespace(results=rows, units={}, summary={})

        monkeypatch.setattr(PFLOTRANExpManager, "_materialize", fake_materialize)
        monkeypatch.setattr(PFLOTRANExpManager, "_build", fake_build)
        monkeypatch.setattr(PFLOTRANExpManager, "_run", fake_run)
        monkeypatch.setattr(PFLOTRANExpManager, "_extract", fake_extract)

        co = object.__new__(wf.WorkflowCoordinator)
        co.model = model
        co.mcp_clients = {"terrain": object()}
        co.conversation_context = {}
        co.planner = type("P", (), {
            "plan": lambda s, r: dict(TestTheCoordinatorActuallyDispatches
                                      .STRATEGY)})()
        co.analyzer = type("A", (), {
            "generate_analysis_report": lambda s, **kw: {"summary": "ok"}})()
        out = co._workflow_design_and_run(self._reception(), str(tmp_path))
        return out, seen

    def test_model_pflotran_builds_the_pflotran_manager(self, tmp_path,
                                                        monkeypatch):
        out, seen = self._drive(tmp_path, "pflotran", monkeypatch)
        assert "❌" not in out, out
        assert seen["cls"] == "PFLOTRANExpManager", \
            "--model pflotran must not run ELM"

    def test_the_run_directory_names_the_model_that_ran(self, tmp_path,
                                                        monkeypatch):
        """Every run directory used to be elm_run_*, which becomes a mislabel
        the moment ELM is not the only backend — the directory name is the
        first thing anyone reads, and archived PFLOTRAN studies would all
        claim to be ELM."""
        self._drive(tmp_path, "pflotran", monkeypatch)
        assert list(Path(tmp_path).glob("pflotran_run_*"))
        assert not list(Path(tmp_path).glob("elm_run_*"))

    def test_the_manager_is_not_handed_a_warm_start_it_cannot_do(
            self, tmp_path, monkeypatch):
        _, seen = self._drive(tmp_path, "pflotran", monkeypatch)
        assert "warm_start" not in seen["config"]
        # but it IS told the period that was asked about
        assert seen["config"]["yr_start"] == 2019
