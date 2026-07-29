#!/usr/bin/env python3
"""
Phase A regression tests for ELMExpManager.
~/RCSFA/multi-agent/tests/test_elm_exp_manager_structure.py

Verifies:
    1. ELMExpManager.__init__ creates four numbered subdirs
       (01_inputs, 02_setup_plots, 03_results, 04_analysis)
       matching PFLOTRAN's layout exactly.
    2. Each pipeline step writes its outputs to the correct subdir:
         _build()    → 01_inputs/experiment_summary.json
         _run()      → 03_results/execution_report.txt
                     → 03_results/results_summary.csv
         _extract()  → 04_analysis/ (via ELMResultsAnalyzer)
    3. Top-level files (LLM_ANALYSIS_INPUT.json, RUN_SUMMARY.json)
       stay at run_dir top-level, not under any subdir.
    4. Analysis figures never create a "plots/" subdir — they save
       directly into 04_analysis/.

These tests don't run real ELM — they mock the builder, adapter,
and analyzer so verification takes seconds and works on a login node.

USAGE
─────
    cd ~/RCSFA/multi-agent
    python3 -m pytest tests/test_elm_exp_manager_structure.py -v

Expected: all tests pass.
"""
import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from core.elm_exp_manager       import ELMExpManager
from core.elm_results_analyzer  import ELMResultsAnalyzer


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_experiments():
    """Two fake experiments with mocked elm_agent for run/summary calls."""
    def _make(case_name: str, period: str):
        agent = MagicMock()
        agent.run_simulation.return_value = True
        agent.get_run_summary.return_value = {
            'case_name':     case_name,
            'case_dir':      f'/fake/pscratch/{case_name}',
            'history_files': [f'/fake/h0_{i:02d}.nc' for i in range(5)],
            'status':        'ok',
        }
        return {
            'case_name':      case_name,
            'scenario_name':  f'scn_{case_name}',
            'forcing_period': period,
            'forcing_start':  1980,
            'forcing_end':    1984,
            'soil_config':    'native',
            'substrate':      'extrapolate',
            'case_dir':       f'/fake/pscratch/{case_name}',
            'elm_agent':      agent,
        }
    return [
        _make('elm_baseline', 'baseline'),
        _make('elm_dry',      'dry'),
    ]


@pytest.fixture
def fake_builder_summary():
    return {
        'model_type':        'elm',
        'total_experiments': 2,
        'experiments': [
            {'case_name': 'elm_baseline', 'forcing_period': 'baseline'},
            {'case_name': 'elm_dry',      'forcing_period': 'dry'},
        ],
    }


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 1 — SUBDIR STRUCTURE
# ═════════════════════════════════════════════════════════════════════

class TestSubdirStructure:
    """Four numbered subdirs are created on __init__."""

    def test_run_dir_created_with_correct_prefix(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        assert mgr.run_dir.exists()
        assert mgr.run_dir.is_dir()
        assert mgr.run_dir.name.startswith("elm_run_")

    def test_all_four_subdirs_exist_on_disk(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        assert (mgr.run_dir / "01_inputs").is_dir()
        assert (mgr.run_dir / "02_setup_plots").is_dir()
        assert (mgr.run_dir / "03_results").is_dir()
        assert (mgr.run_dir / "04_analysis").is_dir()

    def test_named_attributes_point_to_correct_paths(self, tmp_path):
        """Each subdir has a named Python attribute on the manager."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        assert mgr.input_dir       == mgr.run_dir / "01_inputs"
        assert mgr.setup_plots_dir == mgr.run_dir / "02_setup_plots"
        assert mgr.results_dir     == mgr.run_dir / "03_results"
        assert mgr.analysis_dir    == mgr.run_dir / "04_analysis"

    def test_subdir_names_match_pflotran_exactly(self, tmp_path):
        """Stable 0N_ names: run dirs are consumed by tools/ CLIs and by eye."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        assert mgr.input_dir.name       == "01_inputs"
        assert mgr.setup_plots_dir.name == "02_setup_plots"
        assert mgr.results_dir.name     == "03_results"
        assert mgr.analysis_dir.name    == "04_analysis"

    def test_subdirs_are_empty_after_init(self, tmp_path):
        """Subdirs exist but no files yet — pipeline hasn't run."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        for d in [mgr.input_dir, mgr.setup_plots_dir,
                  mgr.results_dir, mgr.analysis_dir]:
            assert list(d.iterdir()) == []


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 2 — STEP 1: BUILD → 01_inputs/
# ═════════════════════════════════════════════════════════════════════

class TestBuildStepOutputs:
    """_build() writes experiment_summary.json to 01_inputs/."""

    def test_experiment_summary_lands_in_01_inputs(
            self, tmp_path, fake_builder_summary):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_builder = MagicMock()
        mock_builder.build_experiments.return_value = []
        mock_builder.get_experiment_summary.return_value = (
            fake_builder_summary
        )
        mock_class = MagicMock(return_value=mock_builder)

        with patch("core.elm_exp_manager.ELMExperimentBuilder",
                   mock_class):
            mgr._build({}, {})

        assert (mgr.input_dir / "experiment_summary.json").exists()

    def test_experiment_summary_NOT_at_top_level(
            self, tmp_path, fake_builder_summary):
        """Regression: must not be at the old top-level location."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_builder = MagicMock()
        mock_builder.build_experiments.return_value = []
        mock_builder.get_experiment_summary.return_value = (
            fake_builder_summary
        )
        with patch("core.elm_exp_manager.ELMExperimentBuilder",
                   MagicMock(return_value=mock_builder)):
            mgr._build({}, {})

        # Old location should NOT be present
        assert not (mgr.run_dir / "experiment_summary.json").exists()

    def test_experiment_summary_content_matches_builder(
            self, tmp_path, fake_builder_summary):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_builder = MagicMock()
        mock_builder.build_experiments.return_value = []
        mock_builder.get_experiment_summary.return_value = (
            fake_builder_summary
        )
        with patch("core.elm_exp_manager.ELMExperimentBuilder",
                   MagicMock(return_value=mock_builder)):
            mgr._build({}, {})

        with open(mgr.input_dir / "experiment_summary.json") as f:
            data = json.load(f)
        assert data == fake_builder_summary


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 3 — STEP 3: RUN → 03_results/
# ═════════════════════════════════════════════════════════════════════

class TestRunStepOutputs:
    """_run() writes execution_report.txt and results_summary.csv to 03_results/."""

    def test_execution_report_lands_in_03_results(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._run(fake_experiments, {})

        assert (mgr.results_dir / "execution_report.txt").exists()

    def test_execution_report_has_expected_content(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._run(fake_experiments, {})

        text = (mgr.results_dir / "execution_report.txt").read_text()
        assert "ELM EXECUTION REPORT" in text
        assert "Total experiments: 2" in text
        assert "Successful:        2" in text
        assert "elm_baseline" in text
        assert "elm_dry"      in text
        # Forcing-period info shows up
        assert "baseline" in text.lower()
        assert "dry"      in text.lower()

    def test_results_csv_lands_in_03_results(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._run(fake_experiments, {})

        csv_file = mgr.results_dir / "results_summary.csv"
        assert csv_file.exists()

    def test_results_csv_has_correct_columns_and_rows(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._run(fake_experiments, {})

        csv_file = mgr.results_dir / "results_summary.csv"
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        expected_cols = {'case_name', 'scenario_name', 'forcing_period',
                         'forcing_start', 'forcing_end', 'soil_config',
                         'substrate', 'status', 'history_file_count',
                         'case_dir'}
        assert expected_cols.issubset(set(rows[0].keys()))

        names = {r['case_name'] for r in rows}
        assert names == {'elm_baseline', 'elm_dry'}

        for row in rows:
            assert row['status'] == 'completed'
            assert row['history_file_count'] == '5'

    def test_results_csv_reports_failure_correctly(
            self, tmp_path, fake_experiments):
        """If run_simulation returns False, status should be 'failed'."""
        # Flip the first experiment to fail
        fake_experiments[0]['elm_agent'].run_simulation.return_value = False

        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._run(fake_experiments, {})

        csv_file = mgr.results_dir / "results_summary.csv"
        with open(csv_file) as f:
            rows = {r['case_name']: r for r in csv.DictReader(f)}

        assert rows['elm_baseline']['status'] == 'failed'
        assert rows['elm_dry']['status']      == 'completed'


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 4 — STEP 4: EXTRACT → 04_analysis/
# ═════════════════════════════════════════════════════════════════════

class TestAnalyzeStepOutputs:
    """_extract() points ELMResultsAnalyzer at 04_analysis/."""

    def test_analyzer_receives_04_analysis_as_dir(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_analyzer = MagicMock()
        mock_class    = MagicMock(return_value=mock_analyzer)

        with patch("core.elm_exp_manager.ELMResultsAnalyzer",
                   mock_class):
            mgr._extract(fake_experiments)

        # Verify the analyzer was constructed with analysis_dir =
        # str(self.analysis_dir), which is 04_analysis/
        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs['analysis_dir'] == str(mgr.analysis_dir)
        assert call_kwargs['analysis_dir'].endswith("04_analysis")


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 5 — TOP-LEVEL FILES STAY AT TOP
# ═════════════════════════════════════════════════════════════════════

class TestTopLevelFiles:
    """LLM_ANALYSIS_INPUT.json and RUN_SUMMARY.json stay at run_dir top."""

    def test_llm_input_at_top_level(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_analyzer = MagicMock()
        mock_analyzer.get_llm_analysis_input.return_value = {
            'model_type':  'elm',
            'experiments': [],
        }

        mgr._save_llm_input(plan={'fake': 'plan'},
                            analyzer=mock_analyzer)

        assert (mgr.run_dir / "LLM_ANALYSIS_INPUT.json").exists()
        # Regression: not in 04_analysis
        assert not (mgr.analysis_dir / "LLM_ANALYSIS_INPUT.json").exists()

    def test_llm_input_includes_run_directory_and_plan(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_analyzer = MagicMock()
        mock_analyzer.get_llm_analysis_input.return_value = {
            'model_type':  'elm',
            'experiments': [],
        }
        fake_plan = {'CONDITIONS_COUPLERS': [{'EXPERIMENT': 'e1'}]}

        mgr._save_llm_input(plan=fake_plan, analyzer=mock_analyzer)

        with open(mgr.run_dir / "LLM_ANALYSIS_INPUT.json") as f:
            data = json.load(f)
        assert data['run_directory']   == str(mgr.run_dir)
        assert data['experiment_plan'] == fake_plan

    def test_run_summary_at_top_level(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        summary = {
            'experiments_total':     2,
            'experiments_success':   2,
            'experiments_failed':    0,
            'total_runtime_seconds': 123.4,
        }
        mgr._save_run_summary(summary)

        assert (mgr.run_dir / "RUN_SUMMARY.json").exists()
        assert not (mgr.results_dir / "RUN_SUMMARY.json").exists()


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 6 — RUN SUMMARY OUTPUT_FILES POINTS AT SUBDIRS
# ═════════════════════════════════════════════════════════════════════

class TestRunSummaryOutputFiles:
    """_create_run_summary's output_files reflects the 4-subdir layout."""

    def test_output_files_includes_all_subdirs(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        from datetime import datetime
        s = datetime.now()
        e = datetime.now()

        results = {exp['case_name']: True for exp in fake_experiments}
        run_summary = mgr._create_run_summary(
            plan={}, experiments=fake_experiments,
            results=results, start_time=s, end_time=e
        )

        of = run_summary['output_files']
        assert of['inputs']      == str(mgr.input_dir)
        assert of['setup_plots'] == str(mgr.setup_plots_dir)
        assert of['results']     == str(mgr.results_dir)
        assert of['analysis']    == str(mgr.analysis_dir)

    def test_output_files_includes_key_artifacts(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        from datetime import datetime
        s = datetime.now()
        e = datetime.now()

        results = {exp['case_name']: True for exp in fake_experiments}
        run_summary = mgr._create_run_summary(
            plan={}, experiments=fake_experiments,
            results=results, start_time=s, end_time=e
        )

        of = run_summary['output_files']
        assert of['experiment_summary'].endswith(
            "01_inputs/experiment_summary.json")
        assert of['execution_report'].endswith(
            "03_results/execution_report.txt")
        assert of['results_csv'].endswith(
            "03_results/results_summary.csv")
        assert of['hydro_summary'].endswith(
            "04_analysis/hydro_summary.json")
        assert of['llm_input'].endswith(
            "LLM_ANALYSIS_INPUT.json")


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 7 — ANALYSIS FIGURES LAND IN 04_analysis/, NOT A plots/ SUBDIR
# ═════════════════════════════════════════════════════════════════════

class TestAnalyzerNoPlotsSubdir:
    """Analysis figures go directly into analysis_dir."""

    def test_plotting_does_not_create_plots_subdir(self, tmp_path):
        """
        Analyzer.figures() on an empty analyzer must not produce a plots/
        subdir under 04_analysis/, and must not raise. (Guards the invariant
        that used to be tested against the removed
        ELMResultsAnalyzer.plot_all.)
        """
        from agents.analyzer import Analyzer
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        analyzer = ELMResultsAnalyzer(
            experiments  = [],
            analysis_dir = str(mgr.analysis_dir),
        )
        Analyzer(str(mgr.run_dir)).figures(analyzer)   # non-fatal by contract
        assert not (mgr.analysis_dir / "plots").exists()


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# ═════════════════════════════════════════════════════════════════════
# CATEGORY 4c — THE ANALYZER IS A SEPARATE BOX
# ═════════════════════════════════════════════════════════════════════

class TestAnalyzerBoundary:
    """Figures, validation and interpretation left the Experiment Manager.

    The manager runs the model; the Analyzer says what the numbers mean. The
    two used to be one class, which made 'the run finished' and 'the run
    showed something' the same judgement. Guards the split.
    """

    def test_manager_no_longer_owns_analysis_stages(self):
        for gone in ("_analyze", "_plot_analysis", "_validate", "_interpret"):
            assert not hasattr(ELMExpManager, gone), (
                f"{gone} is back on the manager — analysis belongs to "
                f"src/agents/analyzer.py")
        assert hasattr(ELMExpManager, "_extract"), (
            "the manager must keep extraction: reading ELM history NetCDFs "
            "is a property of the backend, not of the analysis")

    def test_every_analyzer_stage_is_non_fatal(self, tmp_path):
        """A run stands without figures, observations or prose.

        Each stage returns False rather than raising, and run() survives all
        three failing at once — losing the interpretation must never cost the
        compute that produced it.
        """
        from agents.analyzer import Analyzer
        az = Analyzer(str(tmp_path / "run"), verbose=False)

        # no MCP clients, no results, nothing on disk to interpret
        assert az.validate({}) is False
        assert az.figures(object()) is False           # no .results attribute
        out = az.run(results=None, config={"agentic_analyzer": False})
        assert set(out) == {"figures", "validation", "interpretation"}
        assert out["figures"] is False                 # skipped without results
        assert all(v is False for v in out.values())

    def test_analyzer_creates_its_own_analysis_dir(self, tmp_path):
        """It must work against a run directory it did not create itself —
        that is how it is invoked against an older run."""
        from agents.analyzer import Analyzer
        az = Analyzer(str(tmp_path / "old_run"), verbose=False)
        assert az.analysis_dir.exists()
        assert az.analysis_dir.name == "04_analysis"


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 0 — BASE / BACKEND SPLIT
# ═════════════════════════════════════════════════════════════════════

class TestManagerSplit:
    """Sampling and packaging are model-agnostic; warm start and case
    building are not. PFLOTRAN will subclass the same base, and coupling is
    only meaningful if both backends sampled from one materialisation."""

    def test_elm_manager_is_a_backend(self):
        from core.exp_manager_base import ExperimentManagerBase
        assert issubclass(ELMExpManager, ExperimentManagerBase)
        assert ELMExpManager.MODEL == "elm"

    def test_base_refuses_to_guess_a_backend_plan(self, tmp_path):
        """A backend that forgets _to_run_plan must fail loudly, not produce
        an empty experiment."""
        from core.exp_manager_base import ExperimentManagerBase
        base = ExperimentManagerBase(base_output_dir=str(tmp_path))
        with pytest.raises(NotImplementedError):
            base._to_run_plan({}, [], {}, {})

    def test_base_refinement_hook_is_a_no_op(self, tmp_path):
        """A backend with nothing to add to the columns still materializes."""
        from core.exp_manager_base import ExperimentManagerBase
        base = ExperimentManagerBase(base_output_dir=str(tmp_path))
        assert base._refine_columns([], {}) == {}

    def test_elm_hook_turns_columns_into_conditions_couplers(self, tmp_path):
        mgr  = ELMExpManager(base_output_dir=str(tmp_path))
        cols = [{"id": "col_01", "lat": 46.8, "lon": -121.0,
                 "elevation_m": 900.0, "band": 1,
                 "soil": {"sand_pct": 40, "clay_pct": 20, "organic": 57.4}}]
        plan = mgr._to_run_plan({}, cols,
                                {"yr_start": 1988, "yr_end": 1988}, {})
        cc = plan.get("CONDITIONS_COUPLERS")
        assert cc and len(cc) == 1
        # xmlchange values are strings — every reader int()s them, and
        # pinning the string here is what keeps that true.
        assert str(cc[0]["DATM_CLMNCEP_YR_START"]) == "1988"
        assert str(cc[0]["DATM_CLMNCEP_YR_END"])   == "1988"

    def test_warm_start_finidat_reaches_the_coupler(self, tmp_path):
        """FINIDAT is a per-coupler key the builder reads at build time, which
        is the whole reason the warm start runs in _refine_columns rather than
        in _to_run_plan."""
        mgr  = ELMExpManager(base_output_dir=str(tmp_path))
        cols = [{"id": "col_01", "lat": 46.8, "lon": -121.0,
                 "elevation_m": 900.0, "band": 1,
                 "soil": {"sand_pct": 40, "clay_pct": 20, "organic": 57.4}}]
        plan = mgr._to_run_plan(
            {}, cols, {"yr_start": 1988, "yr_end": 1988},
            {"finidat_map": {"col_01": {"finidat": "/x/Warmstart_col_01.nc"}}})
        assert plan["CONDITIONS_COUPLERS"][0]["FINIDAT"] == \
            "/x/Warmstart_col_01.nc"


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 5 — PACKAGE → experiment.json
# ═════════════════════════════════════════════════════════════════════

class TestPackage:
    """experiment.json is the manager's product and the Analyzer's input."""

    @staticmethod
    def _rows():
        return [{"case_name": "col_01", "status": "ok", "lat": 46.7,
                 "lon": -120.8, "elevation_m": 700.0,
                 "metrics": {"precip_mm_yr": 507.0,
                             "annual_runoff_mm_yr": 3.0}},
                {"case_name": "col_02", "status": "ok", "lat": 46.9,
                 "lon": -121.0, "elevation_m": 1400.0,
                 "metrics": {"precip_mm_yr": 1200.0,
                             "annual_runoff_mm_yr": 640.0}}]

    def _pkg(self, tmp_path, rows=None, config=None):
        import types
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        res = types.SimpleNamespace(results=rows if rows is not None
                                    else self._rows(), units={"QOVER": "mm/s"})
        return mgr, mgr._package({}, res, config or {})

    def test_it_writes_experiment_json(self, tmp_path):
        mgr, pkg = self._pkg(tmp_path)
        on_disk = json.loads((mgr.run_dir / "experiment.json").read_text())
        assert on_disk["columns_total"] == 2
        assert on_disk["model"] == "elm"

    def test_success_is_counted_by_excluding_failure(self, tmp_path):
        """Backends spell success differently ('ok', 'success', 'completed').
        Enumerating success strings reported a fully good 14-column run as
        0/14; enumerating FAILURE cannot fail that way."""
        rows = self._rows() + [
            {"case_name": "col_03", "status": "failed", "metrics": None},
            {"case_name": "col_04", "status": "success",
             "metrics": {"precip_mm_yr": 900.0}},
            {"case_name": "col_05", "status": "completed",
             "metrics": {"precip_mm_yr": 800.0}}]
        _, pkg = self._pkg(tmp_path, rows)
        assert pkg["columns_total"] == 5
        assert pkg["columns_succeeded"] == 4

    def test_a_column_without_metrics_did_not_produce_numbers(self, tmp_path):
        _, pkg = self._pkg(tmp_path, [{"case_name": "col_01", "status": "ok",
                                       "metrics": {}}])
        assert pkg["columns_succeeded"] == 0

    def test_every_reported_field_says_what_it_means(self, tmp_path):
        """precip_mm_yr once held RAIN alone, so every fraction against it was
        inflated and a run reported draining more water than fell on it. A
        semantic bug no schema check catches — naming the source variables
        beside the number is what makes the next one visible."""
        _, pkg = self._pkg(tmp_path)
        sem = pkg["field_semantics"]
        reported = {k for r in pkg["columns"] for k in (r.get("metrics") or {})}
        undocumented = reported - set(sem)
        assert not undocumented, f"fields with no stated meaning: {undocumented}"
        assert sem["precip_mm_yr"]["from"] == ["RAIN", "SNOW"]
        assert all("units" in v for v in sem.values())

    def test_the_fractions_are_declared_as_a_split_not_of_precipitation(
            self, tmp_path):
        """runoff_fraction is QOVER/(QCHARGE+QOVER), NOT runoff/P.

        They sum to 1 by construction even when both terms are ~0, which on
        the 2019 Gunnison run made columns draining ~0 mm/yr report
        recharge_fraction = 1.00. Declaring them against precip_mm_yr — as
        the first version of FIELD_SEMANTICS did — tells a reader to multiply
        by P and get a number too large by P/(QCHARGE+QOVER)."""
        _, pkg = self._pkg(tmp_path)
        sem = pkg["field_semantics"]
        for f in ("runoff_fraction", "recharge_fraction"):
            assert set(sem[f]["from"]) == {"QOVER", "QCHARGE"}, \
                f"{f} must be declared as the recharge/runoff split"
            assert "precip" not in " ".join(sem[f]["from"]).lower()
            assert "not a fraction of precipitation" in sem[f].get("note", "")

    def test_runoff_is_declared_as_surface_only(self, tmp_path):
        """annual_runoff_mm_yr is QOVER alone. QDRAI is excluded, so it is
        NOT the streamflow-comparable total — and a validation that compares
        it to a gauge without saying so is comparing two different things."""
        _, pkg = self._pkg(tmp_path)
        sem = pkg["field_semantics"]["annual_runoff_mm_yr"]
        assert sem["from"] == ["QOVER"]
        assert "QDRAI" in sem["note"] and "NOT" in sem["note"]

    def test_domain_and_period_come_from_the_brief(self, tmp_path):
        _, pkg = self._pkg(tmp_path, config={"brief": {
            "domain": {"name": "Naches", "huc": "17030002"},
            "run_settings": {"resolved_period": {"yr_start": 1988,
                                                 "yr_end": 1988,
                                                 "source": "user"}}}})
        assert pkg["domain"]["huc"] == "17030002"
        assert pkg["period"] == {"yr_start": 1988, "yr_end": 1988,
                                 "source": "user"}


    def test_it_accepts_both_results_shapes(self, tmp_path):
        """ELMResultsAnalyzer.results is a DICT keyed by case name;
        hydro_summary.json['experiments'] is the LIST form of the same thing.
        Both reach this stage depending on how it was invoked, and a dict
        silently packaged as zero columns would report a finished ensemble as
        empty."""
        import types
        mgr  = ELMExpManager(base_output_dir=str(tmp_path))
        rows = self._rows()
        as_list = mgr._package({}, types.SimpleNamespace(results=rows), {})
        as_dict = mgr._package(
            {}, types.SimpleNamespace(
                results={r["case_name"]: r for r in rows}), {})
        assert as_list["columns_total"] == as_dict["columns_total"] == 2
        assert as_list["columns_succeeded"] == as_dict["columns_succeeded"] == 2


class TestInputPreBuild:
    """_build_inputs() runs before any CIME work.

    It changes nothing about WHAT ELM receives — the generators key on
    coordinates plus a content hash, so the builder's own calls become cache
    hits on these files (verified equivalent across the 19 columns of the
    2019 Upper Gunnison run). It changes WHEN a failure is visible: buried in
    _build_one it surfaces partway through case creation, after CIME work has
    started.
    """

    def test_it_runs_before_the_builder(self):
        import inspect
        src = inspect.getsource(ELMExpManager._build)
        assert src.index("_build_inputs") < src.index("ELMExperimentBuilder"), \
            "inputs must be built BEFORE the builder, or the early warning " \
            "is worth nothing"

    def test_it_is_non_fatal(self, tmp_path):
        """The builder keeps its own generation path, so a failure here costs
        the early warning and the manifest — not the run."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        assert mgr._build_inputs({}) == {}      # no columns.json, no raise

    def test_it_forwards_the_soil_settings(self, tmp_path, monkeypatch):
        """soil_config and substrate are part of the surface file's identity;
        dropping them would generate a different file than the builder asks
        for and defeat the cache-hit equivalence."""
        import core.elm_exp_manager as M
        seen = {}
        fake = type("m", (), {"build_all": staticmethod(
            lambda rd, **kw: seen.update(kw) or {"built": {}, "failed": {}})})
        monkeypatch.setattr(M, "_load_tool", lambda name: fake)
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._build_inputs({"soil_config": "sandy", "substrate": "template"})
        assert seen["soil_config"] == "sandy"
        assert seen["substrate"] == "template"

    def test_defaults_match_the_builders(self, tmp_path, monkeypatch):
        import core.elm_exp_manager as M
        seen = {}
        fake = type("m", (), {"build_all": staticmethod(
            lambda rd, **kw: seen.update(kw) or {"built": {}, "failed": {}})})
        monkeypatch.setattr(M, "_load_tool", lambda name: fake)
        ELMExpManager(base_output_dir=str(tmp_path))._build_inputs({})
        assert seen["soil_config"] == "native"
        assert seen["substrate"] == "extrapolate"


class TestPackageCarriesTheEnsemble:
    """experiment.json must be sufficient on its own.

    The first cut took the per-column rows and left the ensemble-level
    products behind, so hydro_summary.json carried MORE than the package
    meant to replace it — and the Analyzer went on reading five files.
    """

    def _run_dir(self, tmp_path, hydro=None, columns=None, extra=None):
        import types
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        if hydro is not None:
            (mgr.analysis_dir / "hydro_summary.json").write_text(
                json.dumps(hydro))
        if columns is not None:
            (mgr.run_dir / "columns.json").write_text(json.dumps(columns))
        res = types.SimpleNamespace(
            results=[{"case_name": "col_01", "status": "ok",
                      "metrics": {"precip_mm_yr": 900.0}}],
            extra_summary=extra or {})
        return mgr, mgr._package({}, res, {})

    def test_ensemble_products_are_carried(self, tmp_path):
        _, pkg = self._run_dir(tmp_path, hydro={
            "soil_attribution": {"a": 1}, "driver_matrix": {"b": 2},
            "comparisons": [{"c": 3}], "spatial_summary": {"d": 4}})
        for k in ("soil_attribution", "driver_matrix", "comparisons",
                  "spatial_summary"):
            assert pkg.get(k), f"{k} was dropped — a figure depends on it"

    def test_absent_is_omitted_not_nulled(self, tmp_path):
        """An empty block means 'not computed for this run' — the 2019
        Gunnison run had soil_attribution={} and drew no soil_control figure.
        Emitting it as null would say 'computed as nothing', which is a
        different claim."""
        _, pkg = self._run_dir(tmp_path, hydro={
            "soil_attribution": {}, "comparisons": [],
            "driver_matrix": {"b": 2}})
        assert "soil_attribution" not in pkg
        assert "comparisons" not in pkg
        assert pkg["driver_matrix"] == {"b": 2}

    def test_the_honesty_payload_reaches_the_package(self, tmp_path):
        """limitations and the assumptions ledger reached the written report
        but not the package. An Analyzer reading only this file would have
        stated conclusions with none of the caveats attached."""
        _, pkg = self._run_dir(
            tmp_path, hydro={},
            extra={"limitations": [{"l": 1}], "assumptions_ledger": [{"a": 1}]})
        assert pkg["limitations"] == [{"l": 1}]
        assert pkg["assumptions_ledger"] == [{"a": 1}]

    def test_assumptions_fall_back_to_the_ledger_file(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        (mgr.run_dir / "assumptions.json").write_text(json.dumps([{"a": 9}]))
        import types
        pkg = mgr._package({}, types.SimpleNamespace(results=[]), {})
        assert pkg["assumptions_ledger"] == [{"a": 9}]

    def test_sampling_provenance_is_carried(self, tmp_path):
        """Whether the sample was clipped to the watershed or fell back to the
        raw bbox changes what the figures are entitled to claim."""
        _, pkg = self._run_dir(tmp_path, hydro={}, columns={
            "boundary": [[[1, 2]]],
            "sampling_domain": {"clipped_to_watershed": False,
                                "caveat": "bbox sample"},
            "grid": {"n_in_basin": 67}})
        assert pkg["sampling_domain"]["clipped_to_watershed"] is False
        assert pkg["boundary"] and pkg["grid"]["n_in_basin"] == 67

    def test_columns_json_is_found_in_either_location(self, tmp_path):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        (mgr.input_dir / "columns.json").write_text(
            json.dumps({"sampling_domain": {"clipped_to_watershed": True}}))
        import types
        pkg = mgr._package({}, types.SimpleNamespace(results=[]), {})
        assert pkg["sampling_domain"]["clipped_to_watershed"] is True


class TestDailySeries:
    """_extract captures a daily series, so the Analyzer stops needing scratch.

    Nothing used to extract one: validate_run rebuilt hydrographs by
    re-reading history NetCDFs off $PSCRATCH. That left the Analyzer coupled
    to scratch rather than to the manager's package, and meant the series
    died with the purge while the run directory survived.
    """

    def _summarize(self, values, var, with_time=True):
        pytest.importorskip("xarray")
        import numpy as np, xarray as xr
        from core.elm_results_analyzer import ELMResultsAnalyzer
        n = len(values)
        coords, dims = {}, ("time",)
        if with_time:
            coords["time"] = np.array(
                [np.datetime64("2019-01-01") + np.timedelta64(3 * i, "h")
                 for i in range(n)])
        da = xr.DataArray(np.array(values, dtype=float), dims=dims,
                          coords=coords)
        ra = ELMResultsAnalyzer(experiments=[], analysis_dir="/tmp")
        return ra._summarize(da, var)

    def test_three_hourly_is_aggregated_to_daily(self):
        """ELM writes 3-hourly. 13 variables x 2921 steps x 19 columns is
        ~5.8 MB of JSON that nothing reads at that resolution — USGS
        observations are daily, so a comparison resamples anyway."""
        out = self._summarize([1.0] * 80, "QOVER")     # 80 steps = 10 days
        assert len(out["daily"]["values"]) == 10

    def test_fluxes_are_converted_to_mm_per_day(self):
        """ELM stores mm/s; the annual metrics use mm/yr. A daily hydrograph
        is plotted in mm/day, and saying so is what stops the next reader
        guessing which of the three this is."""
        out = self._summarize([1.0] * 8, "QOVER")
        assert out["daily"]["units"] == "mm/day"
        assert out["daily"]["values"][0] == pytest.approx(86400.0)

    def test_states_keep_their_native_units(self):
        out = self._summarize([100.0] * 8, "TWS")
        assert out["daily"]["units"] == "mm"
        assert out["daily"]["values"][0] == pytest.approx(100.0)

    def test_the_series_carries_dates(self):
        out = self._summarize([1.0] * 16, "QOVER")
        assert out["daily"]["dates"][0] == "2019-01-01"
        assert len(out["daily"]["dates"]) == len(out["daily"]["values"])

    def test_stats_still_produced_alongside(self):
        """The series must be additive — the annual metrics are built from
        the stats and must not change."""
        out = self._summarize([1.0] * 8, "QOVER")
        assert "annual_mean" in out and "n_timesteps" in out

    def test_no_time_coordinate_is_not_fatal(self):
        out = self._summarize([1.0] * 8, "QOVER", with_time=False)
        assert "annual_mean" in out          # stats survive
