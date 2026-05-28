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
         _analyze()  → 04_analysis/ (via ELMResultsAnalyzer)
    3. Top-level files (LLM_ANALYSIS_INPUT.json, RUN_SUMMARY.json)
       stay at run_dir top-level, not under any subdir.
    4. ELMResultsAnalyzer.plot_all() no longer creates a "plots/"
       subdir (it saves directly to analysis_dir).

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
        """Names must match PFLOTRAN's exp_manager for create_slides.py."""
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
# CATEGORY 4 — STEP 4: ANALYZE → 04_analysis/
# ═════════════════════════════════════════════════════════════════════

class TestAnalyzeStepOutputs:
    """_analyze() points ELMResultsAnalyzer at 04_analysis/."""

    def test_analyzer_receives_04_analysis_as_dir(
            self, tmp_path, fake_experiments):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))

        mock_analyzer = MagicMock()
        mock_class    = MagicMock(return_value=mock_analyzer)

        with patch("core.elm_exp_manager.ELMResultsAnalyzer",
                   mock_class):
            mgr._analyze(fake_experiments)

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
# CATEGORY 7 — ELMResultsAnalyzer DOES NOT CREATE plots/ SUBDIR
# ═════════════════════════════════════════════════════════════════════

class TestAnalyzerNoPlotsSubdir:
    """ELMResultsAnalyzer.plot_all() saves directly to analysis_dir."""

    def test_plot_all_does_not_create_plots_subdir(self, tmp_path):
        """
        Empty analyzer + plot_all() should not produce a plots/ subdir
        under analysis_dir. (This was the old behavior we removed in
        Phase A.)
        """
        analysis_dir = tmp_path / "04_analysis"
        analyzer = ELMResultsAnalyzer(
            experiments  = [],
            analysis_dir = str(analysis_dir),
        )
        analyzer.plot_all()
        # The "plots" subdir must NOT exist
        assert not (analysis_dir / "plots").exists()


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))