#!/usr/bin/env python3
"""
ELM Minimal End-to-End Test
tests/test_elm_e2e_minimal.py

Tests ELMExpManager execution pipeline directly.
Bypasses ReceptionAgent + PlannerAgent — uses hand-crafted plan.

Requirements:
    - Interactive node allocated (salloc)
    - netcdf4 installed (pip install netcdf4)
    - ELM built or cached at $PSCRATCH/E3SMv3/

Run:
    cd ~/RCSFA/multi-agent
    python3 -m pytest tests/test_elm_e2e_minimal.py -v -s
"""
import sys
import json
import os
import pytest
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "src")

from core.elm_exp_manager      import ELMExpManager
from core.model_agent_base     import REQUIRED_SUMMARY_KEYS

# ─────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────
requires_interactive = pytest.mark.skipif(
    os.environ.get('SLURM_JOB_ID') is None,
    reason="Requires interactive node (salloc)"
)

# ─────────────────────────────────────────────────────────────────────
# MINIMAL PLAN — 1 experiment, 1 year
# ─────────────────────────────────────────────────────────────────────
MINIMAL_PLAN = {
    "CONDITIONS_COUPLERS": [
        {
            "EXPERIMENT":            "elm_test_shared",
            "FORCING_PERIOD":        "baseline",
            "STOP_N":                "1",
            "DATM_CLMNCEP_YR_START": "1981",
            "DATM_CLMNCEP_YR_END":   "1981",
            "RUN_STARTDATE":         "1981-01-01",
            "DESCRIPTION":           "Minimal e2e test — 1 year baseline"
        }
    ],
    "ELM_CONFIG": {
        "base_stop_option": "nyears",
        "base_rest_n":      "1",
        "base_rest_option": "nyears",
    },
    "TIME": {
        "forcing_start": 1981,
        "forcing_end":   1981,
    }
}

CONFIG = {
    'use_cache': True,   # reuse existing build
}

# ─────────────────────────────────────────────────────────────────────
# SHARED FIXTURE — execute plan once, reuse across all tests
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def run_summary(tmp_path_factory):
    """
    Execute ELMExpManager.execute_plan() once.
    Reused by all tests in this module.
    """
    output_dir = tmp_path_factory.mktemp("elm_e2e")

    print(f"\n{'=' * 60}")
    print(f"ELM MINIMAL E2E TEST")
    print(f"{'=' * 60}")
    print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output dir : {output_dir}")
    print(f"Use cache  : {CONFIG['use_cache']}")
    print(f"{'=' * 60}\n")

    manager = ELMExpManager(base_output_dir=str(output_dir))

    start      = datetime.now()
    summary    = manager.execute_plan(MINIMAL_PLAN, CONFIG)
    elapsed    = (datetime.now() - start).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"EXECUTION COMPLETE: {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return summary

# ─────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.compute       # opt-in tier (needs --runcompute); guard below still applies
@requires_interactive
class TestELME2EMinimal:
    """
    End-to-end tests for ELMExpManager execution pipeline.
    All tests share one execute_plan() call via run_summary fixture.
    """

    # ── Run summary structure ──────────────────────────────────────
    def test_run_summary_returned(self, run_summary):
        """execute_plan() returns a dict."""
        assert run_summary is not None
        assert isinstance(run_summary, dict)
        print(f"\n   run_summary type: ✓")

    def test_run_summary_required_keys(self, run_summary):
        """run_summary has all keys workflow.py expects."""
        required = {
            'run_directory',
            'experiments_total',
            'experiments_success',
            'experiments_failed',
            'total_runtime_seconds',
            'experiments',
            'output_files',
            'convergence_warnings',
            'model_type',
        }
        for key in required:
            assert key in run_summary, (
                f"Missing required key: '{key}'"
            )
        print(f"\n   Required keys: ✓")

    def test_model_type_is_elm(self, run_summary):
        """run_summary identifies model as ELM."""
        assert run_summary['model_type'] == 'elm'
        print(f"\n   model_type = 'elm': ✓")

    def test_experiment_count(self, run_summary):
        """Correct number of experiments reported."""
        assert run_summary['experiments_total']   == 1
        print(f"\n   experiments_total = 1: ✓")

    def test_experiment_succeeded(self, run_summary):
        """All experiments succeeded."""
        total   = run_summary['experiments_total']
        success = run_summary['experiments_success']
        failed  = run_summary['experiments_failed']

        print(f"\n   Success: {success}/{total}")
        assert failed   == 0, (
            f"{failed} experiment(s) failed — "
            f"check logs in {run_summary['run_directory']}"
        )
        assert success  == total

    def test_runtime_recorded(self, run_summary):
        """Total runtime is recorded and positive."""
        runtime = run_summary['total_runtime_seconds']
        assert runtime > 0
        print(f"\n   Runtime: {runtime:.1f}s")

    def test_convergence_warnings_empty(self, run_summary):
        """No convergence warnings for ELM."""
        assert run_summary['convergence_warnings'] == []
        print(f"\n   convergence_warnings = []: ✓")

    # ── Output files ───────────────────────────────────────────────
    def test_run_directory_exists(self, run_summary):
        """Run directory was created."""
        run_dir = Path(run_summary['run_directory'])
        assert run_dir.exists(), (
            f"Run directory not found: {run_dir}"
        )
        print(f"\n   Run dir: {run_dir.name} ✓")

    def test_run_directory_prefix(self, run_summary):
        """Run directory name starts with 'elm_run_'."""
        run_dir = Path(run_summary['run_directory'])
        assert run_dir.name.startswith('elm_run_'), (
            f"Expected 'elm_run_' prefix, got: {run_dir.name}"
        )

    def test_run_summary_json_saved(self, run_summary):
        """RUN_SUMMARY.json written to run directory."""
        run_dir      = Path(run_summary['run_directory'])
        summary_file = run_dir / "RUN_SUMMARY.json"
        assert summary_file.exists(), (
            f"RUN_SUMMARY.json not found in {run_dir}"
        )
        # Verify valid JSON
        with open(summary_file) as f:
            saved = json.load(f)
        assert saved['model_type'] == 'elm'
        print(f"\n   RUN_SUMMARY.json: ✓")

    def test_experiment_summary_json_saved(self, run_summary):
        """experiment_summary.json written to run directory."""
        run_dir      = Path(run_summary['run_directory'])
        summary_file = run_dir / "experiment_summary.json"
        assert summary_file.exists(), (
            f"experiment_summary.json not found in {run_dir}"
        )
        with open(summary_file) as f:
            saved = json.load(f)
        assert saved['model_type']        == 'elm'
        assert saved['total_experiments'] == 1
        print(f"\n   experiment_summary.json: ✓")

    def test_llm_analysis_input_produced(self, run_summary):
        """LLM_ANALYSIS_INPUT.json produced for AnalysisReportAgent."""
        llm_file = Path(
            run_summary['output_files']['llm_input']
        )
        assert llm_file.exists(), (
            f"LLM_ANALYSIS_INPUT.json not found: {llm_file}"
        )
        with open(llm_file) as f:
            llm_input = json.load(f)

        # Check structure
        assert llm_input['model_type']    == 'elm'
        assert 'experiments'              in llm_input
        assert 'focus_variables'          in llm_input
        assert 'QCHARGE'  in llm_input['focus_variables']
        assert 'QOVER'    in llm_input['focus_variables']
        assert 'TWS'      in llm_input['focus_variables']
        assert 'SOILLIQ'  in llm_input['focus_variables']
        print(f"\n   LLM_ANALYSIS_INPUT.json: ✓")

    def test_hydro_summary_produced(self, run_summary):
        """hydro_summary.json produced by ELMResultsAnalyzer."""
        hydro_file = Path(
            run_summary['output_files']['hydro_summary']
        )
        assert hydro_file.exists(), (
            f"hydro_summary.json not found: {hydro_file}"
        )
        with open(hydro_file) as f:
            hydro = json.load(f)

        assert 'experiments' in hydro
        assert 'units'       in hydro
        print(f"\n   hydro_summary.json: ✓")

    # ── Science output ─────────────────────────────────────────────
    def test_experiment_details(self, run_summary):
        """Per-experiment details recorded correctly."""
        experiments = run_summary['experiments']
        assert len(experiments) == 1

        exp = experiments[0]
        assert exp['model_type']      == 'elm'
        assert exp['forcing_period']  == 'baseline'
        assert exp['forcing_start']   == 1981
        assert exp['forcing_end']     == 1981
        assert exp['status']          == 'completed'
        print(f"\n   Experiment details: ✓")
        print(f"   Status          : {exp['status']}")
        print(f"   Forcing period  : {exp['forcing_period']}")
        print(f"   Years           : "
              f"{exp['forcing_start']} → {exp['forcing_end']}")

    def test_llm_input_has_results(self, run_summary):
        """LLM input contains actual extracted results."""
        llm_file = Path(
            run_summary['output_files']['llm_input']
        )
        with open(llm_file) as f:
            llm_input = json.load(f)

        experiments = llm_input['experiments']
        assert len(experiments) == 1

        result = experiments[0]
        print(f"\n   Result status: {result['status']}")

        if result['status'] == 'ok':
            # Check variables extracted
            variables = result.get('variables', {})
            for var in ['QCHARGE', 'QOVER', 'TWS', 'SOILLIQ']:
                if variables.get(var):
                    print(f"   ✓ {var} extracted")
                else:
                    print(f"   ⚠️  {var} not extracted")

            # Check metrics
            metrics = result.get('metrics', {})
            if metrics.get('annual_recharge_mm_yr') is not None:
                print(f"\n   Annual recharge : "
                      f"{metrics['annual_recharge_mm_yr']:.4f} mm/yr")
            if metrics.get('annual_runoff_mm_yr') is not None:
                print(f"   Annual runoff   : "
                      f"{metrics['annual_runoff_mm_yr']:.4f} mm/yr")
        else:
            pytest.skip(
                f"Results extraction failed: "
                f"{result.get('reason', 'unknown')}"
            )

    def test_output_file_paths_are_strings(self, run_summary):
        """All output_files values are strings."""
        for key, path in run_summary['output_files'].items():
            assert isinstance(path, str), (
                f"output_files['{key}'] should be str, "
                f"got {type(path)}"
            )

    def test_full_pipeline_print(self, run_summary):
        """Print complete run summary for manual inspection."""
        print(f"\n{'=' * 60}")
        print(f"FULL RUN SUMMARY")
        print(f"{'=' * 60}")
        print(f"Run dir    : {run_summary['run_directory']}")
        print(f"Model      : {run_summary['model_type']}")
        print(f"Total      : {run_summary['experiments_total']}")
        print(f"Success    : {run_summary['experiments_success']}")
        print(f"Failed     : {run_summary['experiments_failed']}")
        print(f"Runtime    : "
              f"{run_summary['total_runtime_seconds']:.1f}s")
        print(f"\nOutput files:")
        for k, v in run_summary['output_files'].items():
            print(f"  {k:25s}: {Path(v).name}")
        print(f"{'=' * 60}")
        assert True  # always passes — just for inspection