#!/usr/bin/env python3
"""The coordinator, driven end to end with the expensive stages stubbed.

Every runtime break this framework has had came from verifying that imports
resolve rather than that calls connect: a method reaching for a caller's local,
a renamed keyword, an alias orphaned when its producer was deleted. None of
those are import errors, and none of them are visible until a real run reaches
that line — which costs a queue slot and an hour to find out.

So this drives the REAL coordinator, the REAL execute_plan skeleton, the REAL
packaging, with only the four stages that need MCP, CIME or SLURM replaced.
Fast enough to run every time; catches exactly the class of bug that has
actually happened here.

The reception package is a genuine artifact from the evaluation suite when one
is present, so the shape under test is the shape reception really emits.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ── the reception package under test ────────────────────────────────────
def _reception():
    """A real reception artifact if the eval suite has run, else a minimal
    one with the same shape."""
    for name in ("naches_1988.json", "brandywine_2010.json"):
        f = ROOT / "workflow_outputs" / "reception_eval" / name
        if f.exists():
            try:
                pkg = json.loads(f.read_text())
                if (pkg.get("brief") or {}).get("domain"):
                    return pkg
            except Exception:
                pass
    return {
        "route": {"action": "design"},
        "brief": {
            "domain": {"name": "Naches", "huc": "17030002",
                       "bbox": "-121.6,46.5,-120.3,47.1"},
            "run_settings": {
                "resolved_period": {"yr_start": 1988, "yr_end": 1988,
                                    "source": "user"},
                "initialization": {"mode": "warm", "source": "conus"},
            },
            "heterogeneity": {"relief_m": 1800},
            "scientific_framing": {"goals": ["partition P"]},
        },
        "observations": {"streamflow": {"ok": True, "n_in_bbox": 3,
                                        "stations": [{"id": "12494000"}]}},
        "grid": {"n_in_basin": 96},
        "provenance": [],
    }


STRATEGY = {
    "archetype": "elevation_gradient",
    "goals": ["partition precipitation into runoff and recharge"],
    "feasibility": {"verdict": "feasible"},
    "sampling": {"n_bands": 4, "per_band": 3, "n_columns": 12,
                 "approach": "stratified by elevation"},
    "validation": [{"variable": "streamflow", "stations": ["12494000"],
                    "comparison": "annual yield"}],
    "requires": [], "coupling": None,
}


def _columns(n=12):
    return [{"id": f"col_{i:02d}", "lat": 46.5 + i * 0.02,
             "lon": -121.0 + i * 0.02, "elevation_m": 600.0 + i * 100,
             "band": i % 4 + 1,
             "soil": {"sand_pct": 40, "clay_pct": 20, "organic": 57.4}}
            for i in range(1, n + 1)]


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Patch out MCP, CIME, SLURM and the Analyzer's two LLM steps; leave every
    seam between them real.

    The LLM steps are stubbed for the same reason as SLURM: they are a live
    external call. Before this the coordinator test reached the real step 2/3
    loop and hung on the gateway — an end-to-end test that spends API calls is
    not a test anyone will run.
    """
    from elm_exp_manager import ELMExpManager
    from agents.analysis import step3_interpret as _s3

    monkeypatch.setattr(_s3, "investigate_and_interpret",
                        lambda ctx, out_dir, **kw: {
                            "investigation": {"n_succeeded": 0, "n_proposed": 0,
                                              "findings": [], "caveats": []},
                            "interpretation": {"verdict": "insufficient",
                                               "audit": {"n_claims": 0,
                                                         "n_struck": 0},
                                               "claims": [], "struck": []},
                            "rounds": [], "stopped_because": "stubbed",
                            "n_rounds": 0})

    cols  = _columns()
    calls = []

    def fake_materialize(self, plan, config):
        calls.append("materialize")
        # the gate is cheap and real — run it
        config = self.check(plan, config)
        (self.run_dir / "columns.json").write_text(
            json.dumps({"columns": cols}, indent=2))
        merged = {**plan, **self._to_run_plan(plan, cols, config, {})}
        (self.run_dir / "run_plan.json").write_text(json.dumps(merged, indent=2))
        return merged

    def fake_build(self, plan, config):
        calls.append("build_case_inputs")
        # the same keys mcp/elm-mcp/src/elm_experiment_builder.py really returns
        return [{"case_name": c["id"], "scenario_name": c["id"],
                 "case_dir": f"/scratch/{c['id']}",
                 "forcing_period": "1988-1988",
                 "forcing_start": 1988, "forcing_end": 1988} for c in cols]

    def fake_prepare(self, experiments, config=None):
        calls.append("build_cases")

    def fake_run(self, experiments, config):
        calls.append("run")
        return {e["case_name"]: True for e in experiments}

    def fake_extract(self, experiments, plan=None, config=None):
        calls.append("extract")
        import types
        rows = [{"case_name": c["id"], "status": "ok", "lat": c["lat"],
                 "lon": c["lon"], "elevation_m": c["elevation_m"],
                 "metrics": {"precip_mm_yr": 900.0 + c["elevation_m"],
                             "annual_runoff_mm_yr": 300.0,
                             "annual_recharge_mm_yr": 120.0,
                             "runoff_fraction": 0.33}}
                for c in cols]
        # the shape the REAL _extract now returns: per-variable stats with a
        # daily series attached, plus the honesty payload
        for r in rows:
            r["variables"] = {"QOVER": {
                "annual_mean": 300.0, "n_timesteps": 2921,
                "daily": {"units": "mm/day",
                          "values": [0.5] * 365,
                          "dates": ["2019-01-01"] * 365}}}
        ns = types.SimpleNamespace(results=rows, units={"QOVER": "mm/s"},
                                   extra_summary={
                                       "limitations": [{"l": 1}],
                                       "assumptions_ledger": [{"a": 1}]})
        # ELMResultsAnalyzer exposes this; the base uses it for the alias file
        ns.get_llm_analysis_input = lambda: {"experiments": rows}
        return ns

    with patch.object(ELMExpManager, "_materialize", fake_materialize), \
         patch.object(ELMExpManager, "_build_case_inputs",   fake_build),   \
         patch.object(ELMExpManager, "_build_cases", fake_prepare), \
         patch.object(ELMExpManager, "_run",     fake_run),     \
         patch.object(ELMExpManager, "_extract", fake_extract):
        yield calls


class TestCoordinatorEndToEnd:

    def _coordinator(self, tmp_path):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.mcp_clients = {"terrain": object()}
        co.conversation_context = {}
        co.planner = type("P", (), {"plan": lambda self, r: dict(STRATEGY)})()
        # the WRITTEN-REPORT agent (an LLM call), distinct from the Analyzer box
        co.analyzer = type("A", (), {
            "generate_analysis_report": lambda self, **kw: {"summary": "ok"}})()
        return co

    def test_design_and_run_completes(self, tmp_path, stubbed):
        co  = self._coordinator(tmp_path)
        out = co._workflow_design_and_run(_reception(), str(tmp_path))
        assert "❌" not in out, out
        # every stage was actually reached, in order
        assert stubbed == ["materialize", "build_case_inputs", "build_cases", "run",
                           "extract", "couple"]

    def test_the_four_files_land_in_one_run_dir(self, tmp_path, stubbed):
        """reception.json, strategy.json, experiment.json — each written by
        whatever produced it, all in the directory the coordinator owns."""
        co = self._coordinator(tmp_path)
        co._workflow_design_and_run(_reception(), str(tmp_path))
        runs = list(Path(tmp_path).glob("elm_run_*"))
        assert len(runs) == 1, f"expected ONE run dir, got {runs}"
        rd = runs[0]
        for f in ("reception.json", "strategy.json", "experiment.json",
                  "RUN_SUMMARY.json"):
            assert (rd / f).exists(), f"{f} missing from {rd.name}"

    def test_experiment_json_is_readable_and_populated(self, tmp_path, stubbed):
        co = self._coordinator(tmp_path)
        co._workflow_design_and_run(_reception(), str(tmp_path))
        rd  = next(Path(tmp_path).glob("elm_run_*"))
        exp = json.loads((rd / "experiment.json").read_text())
        assert exp["columns_total"] == 12
        assert exp["columns_succeeded"] == 12
        assert exp["period"]["yr_start"]                # reception's period survived
        assert exp["field_semantics"]["precip_mm_yr"]["from"] == ["RAIN", "SNOW"]

    def test_reception_period_reaches_the_couplers(self, tmp_path, stubbed):
        """The period the user gave must not be silently replaced by the
        manager's 1995 default — that is the whole point of resolving it."""
        pkg = _reception()
        period = ((pkg["brief"].get("run_settings") or {})
                  .get("resolved_period") or {})
        co = self._coordinator(tmp_path)
        co._workflow_design_and_run(pkg, str(tmp_path))
        rd   = next(Path(tmp_path).glob("elm_run_*"))
        plan = json.loads((rd / "run_plan.json").read_text())
        cc   = plan["CONDITIONS_COUPLERS"][0]
        assert str(cc["DATM_CLMNCEP_YR_START"]) == str(period["yr_start"])
        assert str(cc["DATM_CLMNCEP_YR_END"])   == str(period["yr_end"])

    def test_analyzer_is_invoked_and_cannot_sink_the_run(self, tmp_path,
                                                         stubbed):
        """The Analyzer runs after extraction, and a failure inside it must
        not cost the compute that produced the numbers."""
        from agents.analyzer import Analyzer
        seen = {}
        real = Analyzer.run

        def boom(self, results=None, config=None):
            seen["called"] = True
            raise RuntimeError("analyzer exploded")

        with patch.object(Analyzer, "run", boom):
            co  = self._coordinator(tmp_path)
            out = co._workflow_design_and_run(_reception(), str(tmp_path))
        assert seen.get("called"), "the Analyzer was never invoked"
        assert "❌" not in out, (
            "an Analyzer failure sank the whole run:\n" + out)
        rd = next(Path(tmp_path).glob("elm_run_*"))
        assert (rd / "experiment.json").exists(), (
            "the results package was lost when the Analyzer failed")


class TestRouteDispatch:
    """Reception decides the route; the coordinator must honour all three and
    not crash on a malformed one."""

    def _co(self):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.mcp_clients = {}
        co.conversation_context = {}
        co.planner  = type("P", (), {"plan": lambda self, r: dict(STRATEGY)})()
        co.analyzer = type("A", (), {
            "generate_analysis_report": lambda self, **kw: {"summary": "ok"}})()
        return co

    def test_clarify_asks_rather_than_running(self, tmp_path):
        co  = self._co()
        out = co._workflow_clarification(
            {"route": {"action": "clarify",
                       "questions": ["Which years should I simulate?"]}})
        assert "Which years" in out
        assert not list(Path(tmp_path).glob("elm_run_*")), \
            "a clarify route must not create a run directory"

    def test_unknown_route_is_reported_not_raised(self):
        co = self._co()
        co.reception = type("R", (), {
            "process": lambda self, *a, **k: {"route": {"action": "nonsense"}}})()
        out = co.process_request("do something", output_dir="/tmp")
        assert "nonsense" in out or "Unknown route" in out

    def test_a_missing_route_defaults_to_clarify_not_to_compute(self):
        """A reception package with no route must not fall through into a
        design-and-run — that spends a queue slot on an unresolved request."""
        co = self._co()
        co.reception = type("R", (), {
            "process": lambda self, *a, **k: {"brief": {}}})()
        out = co.process_request("vague", output_dir="/tmp")
        assert "elm_run_" not in out


class TestPackageIsSelfSufficient:
    """experiment.json must stand alone — that is the whole contract.

    Checked through the REAL execute_plan rather than by calling _package
    directly, because the ordering is what makes it true: _extract writes
    hydro_summary.json, _package reads it back, and both run before the
    Analyzer touches anything.
    """

    def test_the_package_carries_series_and_caveats(self, tmp_path, stubbed):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.mcp_clients, co.conversation_context = {"terrain": object()}, {}
        co.planner  = type("P", (), {"plan": lambda self, r: dict(STRATEGY)})()
        co.analyzer = type("A", (), {
            "generate_analysis_report": lambda self, **kw: {"s": "ok"}})()
        co._workflow_design_and_run(_reception(), str(tmp_path))

        rd  = next(Path(tmp_path).glob("elm_run_*"))
        exp = json.loads((rd / "experiment.json").read_text())

        # the honesty payload — an Analyzer reading only this file would
        # otherwise state conclusions with none of the caveats attached
        assert exp["limitations"] and exp["assumptions_ledger"]

        # the daily series, so the Analyzer stops needing $PSCRATCH
        daily = exp["columns"][0]["variables"]["QOVER"]["daily"]
        assert len(daily["values"]) == 365
        assert daily["units"] == "mm/day", \
            "a hydrograph is plotted in mm/day; ELM stores mm/s and the " \
            "annual metrics use mm/yr, so the series must say which it is"

    def test_package_is_written_before_the_analyzer_runs(self):
        import inspect
        from elm_exp_manager import ELMExpManager
        src = inspect.getsource(ELMExpManager.execute_plan)
        assert src.index("self._package") < src.index("Analyzer("), \
            "the manager's product must be on disk before anything " \
            "interpretive can fail and take it down"


class TestNothingPostComputeDiscardsTheRun:
    """Once 19 columns have run, no downstream stage may throw the run away.

    _run and _extract stay fatal on purpose — without them there are no
    numbers, so there is nothing to preserve. Everything after them is
    packaging, coupling or prose, and all of that is recoverable.
    """

    def _co(self, tmp_path):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.mcp_clients, co.conversation_context = {"terrain": object()}, {}
        co.planner  = type("P", (), {"plan": lambda self, r: dict(STRATEGY)})()
        co.analyzer = type("A", (), {
            "generate_analysis_report": lambda self, **kw: {"s": "ok"}})()
        return co

    @pytest.mark.parametrize("stage", ["_package"])
    def test_a_post_compute_failure_is_survived(self, tmp_path, stubbed, stage):
        from elm_exp_manager import ELMExpManager

        def boom(self, *a, **k):
            raise RuntimeError(f"{stage} exploded")

        with patch.object(ELMExpManager, stage, boom):
            out = self._co(tmp_path)._workflow_design_and_run(
                _reception(), str(tmp_path))
        assert "❌" not in out, f"a {stage} failure sank the whole run:\n{out}"
        rd = next(Path(tmp_path).glob("elm_run_*"))
        assert (rd / "RUN_SUMMARY.json").exists(), \
            "the run summary must survive a downstream failure"

    def test_extraction_failure_is_still_fatal(self, tmp_path, stubbed):
        """Deliberately NOT guarded: no extraction means no numbers, and
        reporting success for a run with no results is worse than failing."""
        from elm_exp_manager import ELMExpManager

        def boom(self, *a, **k):
            raise RuntimeError("extract exploded")

        with patch.object(ELMExpManager, "_extract", boom):
            out = self._co(tmp_path)._workflow_design_and_run(
                _reception(), str(tmp_path))
        assert "❌" in out
