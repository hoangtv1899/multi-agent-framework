#!/usr/bin/env python3
"""
ELM Integration Test
tests/test_elm_integration.py

Tests real ELM case build and simulation run.
Mirrors exactly what 1d_elm.sh does.

Requirements:
    - Interactive node allocated (salloc)
    - E3SM source at /global/u2/h/hvtran/E3SM
    - Input files at /global/homes/h/hvtran/RCSFA/1d_elm/input_files/
    - PSCRATCH defined

Run with:
    cd ~/RCSFA/multi-agent
    python3 -m pytest tests/test_elm_integration.py -v -s

Skip build if cached:
    python3 -m pytest tests/test_elm_integration.py -v -s --use-cache

Estimated time:
    First run  : 5-10 min (build + run)
    Cached run : 1-2  min (run only)
"""
import sys
import os
import json
import pytest
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "src")

from core.elm_input_agent        import ELMAgentAdapter, ELM_AVAILABLE
from core.elm_experiment_builder import ELMExperimentBuilder
from core.elm_results_analyzer   import (
    ELMResultsAnalyzer,
    TARGET_VARIABLES,
)
from core.elm_exp_manager        import ELMExpManager
from core.model_agent_base       import REQUIRED_SUMMARY_KEYS

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS — mirror your bash script exactly
# ─────────────────────────────────────────────────────────────────────
INPUT_FILES_DIR = Path(
    "/global/homes/h/hvtran/RCSFA/1d_elm/input_files"
)
DOMAIN_FILE  = "Domainfile_station_2006_.nc"
SURFACE_FILE = "Surfacedata_Station_2006_.nc"
E3SM_SRC     = Path("/global/u2/h/hvtran/E3SM")
PSCRATCH     = Path(os.environ.get("PSCRATCH", "/tmp"))
ELM_CASE_DIR = PSCRATCH / "E3SMv3"

# Minimal config — mirrors bash script
BASELINE_CONFIG = {
    'STOP_N':                '3',      # short run for testing
    'STOP_OPTION':           'nyears',
    'DATM_CLMNCEP_YR_START': '1981',
    'DATM_CLMNCEP_YR_END':   '1983',  # 3 years only
    'RUN_STARTDATE':         '1981-01-01',
    'REST_N':                '1',
    'REST_OPTION':           'nyears',
}

# ─────────────────────────────────────────────────────────────────────
# GUARDS — skip if environment not ready
# ─────────────────────────────────────────────────────────────────────
def is_interactive_node() -> bool:
    """Check if running on an allocated interactive node."""
    return os.environ.get('SLURM_JOB_ID') is not None

def pscratch_available() -> bool:
    """Check if PSCRATCH is defined and accessible."""
    return (
        'PSCRATCH' in os.environ and
        Path(os.environ['PSCRATCH']).exists()
    )

def input_files_exist() -> bool:
    """Check if ELM input files are present."""
    return (
        (INPUT_FILES_DIR / DOMAIN_FILE).exists()  and
        (INPUT_FILES_DIR / SURFACE_FILE).exists()
    )

def e3sm_source_exists() -> bool:
    """Check if E3SM source is present."""
    return (E3SM_SRC / "cime" / "scripts" / "create_newcase").exists()

# Decorator — skip entire test if environment not ready
requires_perlmutter = pytest.mark.skipif(
    not all([
        is_interactive_node(),
        pscratch_available(),
        input_files_exist(),
        e3sm_source_exists(),
        ELM_AVAILABLE,
    ]),
    reason=(
        "Requires: interactive node + PSCRATCH + "
        "input files + E3SM source + GeneratedELMAgent"
    )
)

# ONE shared suffix for all tests — prevents rebuilds
SHARED_CASE_SUFFIX = "elm_test_shared"

# All fixtures use SHARED_CASE_SUFFIX
@pytest.fixture(scope="session")
def built_adapter():
    adapter = ELMAgentAdapter(
        case_name        = SHARED_CASE_SUFFIX,
        config_overrides = BASELINE_CONFIG,
        use_cache        = True,
    )
    case_dir = adapter.prepare_case()
    return adapter

@pytest.fixture(scope="session")
def completed_adapter(built_adapter):
    """
    ELMAgentAdapter that has been prepared AND run.
    Session-scoped — run once, reused by all tests.
    """
    print(f"\n⏳ Running ELM simulation...")
    start   = datetime.now()
    success = built_adapter.run_simulation()
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'✓' if success else '✗'} "
          f"Simulation {'completed' if success else 'failed'} "
          f"in {elapsed:.1f}s")

    return built_adapter

@pytest.fixture(scope="session")
def completed_experiments(completed_adapter):
    """
    Single-experiment list built from completed_adapter.
    Used by ELMResultsAnalyzer tests.
    """
    return [{
        'scenario_index': 0,
        'scenario_name':  'elm_integration_test',
        'case_name':      'integration_test',
        'forcing_period': 'baseline',
        'forcing_start':  1981,
        'forcing_end':    1983,
        'stop_n':         3,
        'start_date':     '1981-01-01',
        'description':    'Integration test — 3 year baseline',
        'elm_agent':      completed_adapter,
        'case_dir':       completed_adapter._case_dir,
    }]

# ─────────────────────────────────────────────────────────────────────
# TEST CLASS 1 — Case Build
# ─────────────────────────────────────────────────────────────────────
@requires_perlmutter
class TestELMCaseBuild:
    """Verify ELM case is correctly created and built."""

    def test_case_directory_exists(self, built_adapter):
        """Case directory created under $PSCRATCH/E3SMv3/."""
        assert built_adapter._case_dir is not None
        case_dir = Path(built_adapter._case_dir)
        assert case_dir.exists(), (
            f"Case directory not found: {case_dir}"
        )
        print(f"\n   Case dir: {case_dir.name}")

    def test_case_dir_under_pscratch(self, built_adapter):
        """Case directory is under $PSCRATCH/E3SMv3/."""
        case_dir = Path(built_adapter._case_dir)
        assert str(case_dir).startswith(str(ELM_CASE_DIR)), (
            f"Expected case under {ELM_CASE_DIR}, "
            f"got {case_dir}"
        )

    def test_case_name_contains_suffix(self, built_adapter):
        """Case name contains our experiment suffix."""
        case_dir = Path(built_adapter._case_dir)
        assert 'integration_test' in case_dir.name, (
            f"Expected 'integration_test' in {case_dir.name}"
        )

    def test_namelist_elm_exists(self, built_adapter):
        """user_nl_elm namelist file was written."""
        case_dir  = Path(built_adapter._case_dir)
        nl_file   = case_dir / "user_nl_elm"
        assert nl_file.exists(), (
            f"user_nl_elm not found in {case_dir}"
        )

    def test_namelist_elm_has_fsurdat(self, built_adapter):
        """user_nl_elm contains fsurdat path."""
        case_dir = Path(built_adapter._case_dir)
        nl_file  = case_dir / "user_nl_elm"
        content  = nl_file.read_text()
        assert 'fsurdat' in content, (
            "fsurdat not found in user_nl_elm"
        )
        assert SURFACE_FILE in content, (
            f"{SURFACE_FILE} not found in user_nl_elm"
        )
        print(f"\n   fsurdat: ✓")

    def test_namelist_elm_has_history_vars(self, built_adapter):
        """user_nl_elm contains required history variables."""
        case_dir  = Path(built_adapter._case_dir)
        nl_file   = case_dir / "user_nl_elm"
        content   = nl_file.read_text()

        required_vars = ['QCHARGE', 'QOVER', 'TWS', 'SOILLIQ']
        for var in required_vars:
            assert var in content, (
                f"History variable {var} not found in user_nl_elm"
            )
            print(f"   {var}: ✓")

    def test_namelist_mosart_exists(self, built_adapter):
        """user_nl_mosart namelist file was written."""
        case_dir = Path(built_adapter._case_dir)
        nl_file  = case_dir / "user_nl_mosart"
        assert nl_file.exists(), (
            f"user_nl_mosart not found in {case_dir}"
        )

    def test_namelist_mosart_rtm_disabled(self, built_adapter):
        """user_nl_mosart has do_rtm = .false."""
        case_dir = Path(built_adapter._case_dir)
        nl_file  = case_dir / "user_nl_mosart"
        content  = nl_file.read_text()
        assert 'do_rtm = .false.' in content, (
            "do_rtm = .false. not found in user_nl_mosart"
        )
        print(f"\n   do_rtm = .false.: ✓")

    def test_executable_exists(self, built_adapter):
        """e3sm.exe was built successfully."""
        case_dir = Path(built_adapter._case_dir)
        exe_path = case_dir / "build" / "e3sm.exe"
        assert exe_path.exists(), (
            f"e3sm.exe not found at {exe_path}\n"
            f"Build may have failed — check logs in {case_dir}"
        )
        print(f"\n   e3sm.exe: ✓ ({exe_path.stat().st_size // 1024}KB)")

    def test_config_hash_saved(self, built_adapter):
        """Config hash file saved for caching."""
        case_dir    = Path(built_adapter._case_dir)
        hash_file   = case_dir / ".agent_config_hash"
        assert hash_file.exists(), (
            f".agent_config_hash not found in {case_dir}"
        )
        stored_hash = hash_file.read_text().strip()
        assert len(stored_hash) == 32, (
            f"Expected MD5 hash (32 chars), got: {stored_hash}"
        )
        print(f"\n   Config hash: {stored_hash[:8]}...")

    def test_adapter_is_ready_after_build(self, built_adapter):
        """is_ready() returns True after prepare_case()."""
        assert built_adapter.is_ready() is True

    def test_cache_reuse_on_second_call(self, built_adapter):
        """
        Second prepare_case() call reuses cached build.
        Should complete in seconds, not minutes.
        """
        start    = datetime.now()
        case_dir = built_adapter.prepare_case()
        elapsed  = (datetime.now() - start).total_seconds()

        assert case_dir is not None
        # Cache reuse should be very fast (< 30 seconds)
        assert elapsed < 30, (
            f"Cache reuse took {elapsed:.1f}s — "
            f"expected < 30s. Cache may not be working."
        )
        print(f"\n   Cache reuse: {elapsed:.1f}s ✓")

# ─────────────────────────────────────────────────────────────────────
# TEST CLASS 2 — Simulation Run
# ─────────────────────────────────────────────────────────────────────
@requires_perlmutter
class TestELMSimulationRun:
    """Verify ELM simulation runs and produces output."""

    def test_simulation_returns_true(self, completed_adapter):
        """run_simulation() returns True on success."""
        assert completed_adapter._status == 'completed', (
            f"Simulation status: {completed_adapter._status}\n"
            f"Check logs in: {completed_adapter._case_dir}"
        )

    def test_run_directory_exists(self, completed_adapter):
        """Run directory created under case directory."""
        case_dir = Path(completed_adapter._case_dir)
        run_dir  = case_dir / "run"
        assert run_dir.exists(), (
            f"Run directory not found: {run_dir}"
        )

    def test_history_files_created(self, completed_adapter):
        """ELM history files (*.elm.h0.*.nc) produced."""
        case_dir   = Path(completed_adapter._case_dir)
        run_dir    = case_dir / "run"
        hist_files = sorted(run_dir.glob("*.elm.h0.*.nc"))

        assert len(hist_files) > 0, (
            f"No ELM history files found in {run_dir}\n"
            "Simulation may have failed — check logs"
        )
        print(f"\n   History files: {len(hist_files)} found")
        for f in hist_files[:3]:
            print(f"   → {f.name}")

    def test_history_files_are_readable(self, completed_adapter):
        """History files can be opened with xarray."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not available")

        case_dir   = Path(completed_adapter._case_dir)
        run_dir    = case_dir / "run"
        hist_files = sorted(run_dir.glob("*.elm.h0.*.nc"))

        assert len(hist_files) > 0, "No history files to read"

        # Open first file
        ds = xr.open_dataset(hist_files[0])
        assert ds is not None
        print(f"\n   Variables in history file:")
        for var in list(ds.data_vars)[:10]:
            print(f"   → {var}")
        ds.close()

    def test_target_variables_in_history(self, completed_adapter):
        """All 4 target variables present in history files."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not available")

        case_dir   = Path(completed_adapter._case_dir)
        run_dir    = case_dir / "run"
        hist_files = sorted(run_dir.glob("*.elm.h0.*.nc"))

        assert len(hist_files) > 0

        ds = xr.open_dataset(hist_files[0])
        for var in TARGET_VARIABLES:
            assert var in ds, (
                f"Target variable {var} not found in history file\n"
                f"Check hist_fincl1 in user_nl_elm"
            )
            print(f"   {var}: ✓")
        ds.close()

    def test_restart_files_created(self, completed_adapter):
        """ELM restart files produced (confirms run completed)."""
        case_dir      = Path(completed_adapter._case_dir)
        run_dir       = case_dir / "run"
        restart_files = sorted(run_dir.glob("*.elm.r.*.nc"))

        assert len(restart_files) > 0, (
            f"No restart files found in {run_dir}\n"
            "Run may not have completed cleanly"
        )
        print(f"\n   Restart files: {len(restart_files)} found")

    def test_coupling_variables_returned(self, completed_adapter):
        """get_coupling_variables() returns non-None dict."""
        coupling = completed_adapter._coupling
        assert coupling is not None, (
            "coupling_variables is None — "
            "run may not have completed"
        )
        assert isinstance(coupling, dict)
        print(f"\n   Coupling variable groups:")
        for group in coupling:
            print(f"   → {group}")

    def test_run_summary_valid(self, completed_adapter):
        """get_run_summary() passes ModelAgentBase contract."""
        summary = completed_adapter.get_run_summary()

        # Check all required keys
        for key in REQUIRED_SUMMARY_KEYS:
            assert key in summary, f"Missing required key: {key}"

        # Check values
        assert summary['model_type'] == 'elm'
        assert summary['status']     == 'completed'
        assert summary['case_dir']   is not None

        print(f"\n   Run summary keys: ✓")
        print(f"   Status          : {summary['status']}")
        print(f"   Model type      : {summary['model_type']}")

    def test_history_files_in_run_summary(self, completed_adapter):
        """get_run_summary() includes history file paths."""
        summary      = completed_adapter.get_run_summary()
        hist_files   = summary.get('history_files', [])

        assert len(hist_files) > 0, (
            "No history files in run summary"
        )
        # All paths should exist
        for f in hist_files:
            assert Path(f).exists(), (
                f"History file in summary not found: {f}"
            )
        print(f"\n   History files in summary: {len(hist_files)}")

# ─────────────────────────────────────────────────────────────────────
# TEST CLASS 3 — Results Analyzer (real NetCDF)
# ─────────────────────────────────────────────────────────────────────
@requires_perlmutter
class TestELMResultsAnalyzerReal:
    """Test ELMResultsAnalyzer with real ELM history files."""

    def test_extract_all_succeeds(self,
                                   completed_experiments,
                                   tmp_path):
        """extract_all() processes real history files."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results = analyzer.extract_all()

        assert len(results) == 1
        result = results['integration_test']
        assert result['status'] == 'ok', (
            f"Extraction failed: {result.get('reason', 'unknown')}"
        )
        print(f"\n   Extraction status: {result['status']}")

    def test_qcharge_extracted(self,
                                completed_experiments,
                                tmp_path):
        """QCHARGE (aquifer recharge) extracted with valid values."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results  = analyzer.extract_all()
        result   = results['integration_test']

        qcharge = result['variables'].get('QCHARGE')
        assert qcharge is not None, "QCHARGE not extracted"
        assert 'annual_mean'    in qcharge
        assert 'annual_std'     in qcharge
        assert 'units_annual'   in qcharge
        assert qcharge['units_annual'] == 'mm/year'

        print(f"\n   QCHARGE annual mean : "
              f"{qcharge['annual_mean']:.2f} mm/yr")
        print(f"   QCHARGE annual std  : "
              f"{qcharge['annual_std']:.2f} mm/yr")

    def test_qover_extracted(self,
                              completed_experiments,
                              tmp_path):
        """QOVER (surface runoff) extracted with valid values."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results = analyzer.extract_all()
        result  = results['integration_test']

        qover = result['variables'].get('QOVER')
        assert qover is not None, "QOVER not extracted"
        assert 'annual_mean' in qover

        print(f"\n   QOVER annual mean: "
              f"{qover['annual_mean']:.2f} mm/yr")

    def test_tws_extracted(self,
                            completed_experiments,
                            tmp_path):
        """TWS (total water storage) extracted with valid values."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results = analyzer.extract_all()
        result  = results['integration_test']

        tws = result['variables'].get('TWS')
        assert tws is not None, "TWS not extracted"
        assert 'mean_mm'         in tws
        assert 'seasonal_range'  in tws

        print(f"\n   TWS mean          : {tws['mean_mm']:.2f} mm")
        print(f"   TWS seasonal range: "
              f"{tws['seasonal_range']:.2f} mm")

    def test_soilliq_extracted(self,
                                completed_experiments,
                                tmp_path):
        """SOILLIQ (soil moisture) extracted with layer info."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results = analyzer.extract_all()
        result  = results['integration_test']

        soilliq = result['variables'].get('SOILLIQ')
        assert soilliq is not None, "SOILLIQ not extracted"
        assert 'total_column_kg_m2' in soilliq
        assert 'layer_means_kg_m2'  in soilliq
        assert 'n_layers'           in soilliq
        assert soilliq['n_layers']  > 0

        print(f"\n   SOILLIQ layers    : {soilliq['n_layers']}")
        print(f"   SOILLIQ total col : "
              f"{soilliq['total_column_kg_m2']:.2f} kg/m2")

    def test_metrics_computed(self,
                               completed_experiments,
                               tmp_path):
        """Derived metrics computed from extracted variables."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        results = analyzer.extract_all()
        result  = results['integration_test']

        metrics = result['metrics']
        assert 'annual_recharge_mm_yr' in metrics
        assert 'annual_runoff_mm_yr'   in metrics

        print(f"\n   Annual recharge : "
              f"{metrics['annual_recharge_mm_yr']:.2f} mm/yr")
        print(f"   Annual runoff   : "
              f"{metrics['annual_runoff_mm_yr']:.2f} mm/yr")

        ratio = metrics.get('recharge_to_runoff_ratio')
        if ratio is not None:
            print(f"   Recharge/Runoff : {ratio:.3f}")

    def test_hydro_summary_saved(self,
                                  completed_experiments,
                                  tmp_path):
        """hydro_summary.json written to analysis directory."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        analyzer.extract_all()

        hydro_file = tmp_path / "hydro_summary.json"
        assert hydro_file.exists(), (
            f"hydro_summary.json not found in {tmp_path}"
        )

        # Verify it's valid JSON
        with open(hydro_file) as f:
            data = json.load(f)

        assert 'experiments' in data
        assert 'units'       in data
        print(f"\n   hydro_summary.json: ✓")

    def test_llm_analysis_input_structure(self,
                                           completed_experiments,
                                           tmp_path):
        """get_llm_analysis_input() has correct structure."""
        analyzer = ELMResultsAnalyzer(
            experiments  = completed_experiments,
            analysis_dir = str(tmp_path),
        )
        analyzer.extract_all()

        llm_input = analyzer.get_llm_analysis_input()

        assert llm_input['model_type']      == 'elm'
        assert len(llm_input['experiments']) == 1
        assert 'QCHARGE' in llm_input['focus_variables']
        print(f"\n   LLM input structure: ✓")

# ─────────────────────────────────────────────────────────────────────
# TEST CLASS 4 — Full ELMExpManager Pipeline
# ─────────────────────────────────────────────────────────────────────
@requires_perlmutter
class TestELMExpManagerPipeline:
    """
    Test complete ELMExpManager.execute_plan() pipeline.
    Uses a minimal 1-experiment plan to keep runtime short.
    """

    @pytest.fixture
    def minimal_elm_plan(self):
        """Single-experiment plan — fastest possible integration test."""
        return {
            "CONDITIONS_COUPLERS": [
                {
                    "EXPERIMENT":            "elm_pipeline_test",
                    "FORCING_PERIOD":        "baseline",
                    "STOP_N":                "3",
                    "DATM_CLMNCEP_YR_START": "1981",
                    "DATM_CLMNCEP_YR_END":   "1983",
                    "RUN_STARTDATE":         "1981-01-01",
                    "DESCRIPTION":           "Pipeline integration test",
                }
            ],
            "ELM_CONFIG": {
                "base_stop_option":  "nyears",
                "base_rest_n":       "1",
                "base_rest_option":  "nyears",
            },
            "TIME": {
                "forcing_start": 1981,
                "forcing_end":   1983,
            }
        }

    def test_execute_plan_returns_run_summary(self,
                                               minimal_elm_plan,
                                               tmp_path):
        """execute_plan() returns valid run summary dict."""
        manager = ELMExpManager(
            base_output_dir = str(tmp_path)
        )
        run_summary = manager.execute_plan(
            experiment_plan = minimal_elm_plan,
            config          = {'use_cache': True},
        )

        assert run_summary is not None
        assert isinstance(run_summary, dict)

    def test_execute_plan_run_summary_keys(self,
                                            minimal_elm_plan,
                                            tmp_path):
        """execute_plan() run summary has all keys workflow.py needs."""
        manager = ELMExpManager(
            base_output_dir = str(tmp_path)
        )
        run_summary = manager.execute_plan(
            experiment_plan = minimal_elm_plan,
            config          = {'use_cache': True},
        )

        required = {
            'run_directory',
            'experiments_total',
            'experiments_success',
            'experiments_failed',
            'total_runtime_seconds',
            'experiments',
            'output_files',
            'convergence_warnings',
        }
        for key in required:
            assert key in run_summary, f"Missing key: {key}"

    def test_execute_plan_llm_input_produced(self,
                                              minimal_elm_plan,
                                              tmp_path):
        """execute_plan() produces LLM_ANALYSIS_INPUT.json."""
        manager = ELMExpManager(
            base_output_dir = str(tmp_path)
        )
        run_summary = manager.execute_plan(
            experiment_plan = minimal_elm_plan,
            config          = {'use_cache': True},
        )

        llm_input_file = Path(
            run_summary['output_files']['llm_input']
        )
        assert llm_input_file.exists(), (
            f"LLM_ANALYSIS_INPUT.json not found at {llm_input_file}"
        )

        with open(llm_input_file) as f:
            llm_input = json.load(f)

        assert llm_input['model_type'] == 'elm'
        print(f"\n   LLM_ANALYSIS_INPUT.json: ✓")

    def test_execute_plan_run_summary_saved(self,
                                             minimal_elm_plan,
                                             tmp_path):
        """execute_plan() saves RUN_SUMMARY.json."""
        manager = ELMExpManager(
            base_output_dir = str(tmp_path)
        )
        run_summary = manager.execute_plan(
            experiment_plan = minimal_elm_plan,
            config          = {'use_cache': True},
        )

        run_dir      = Path(run_summary['run_directory'])
        summary_file = run_dir / "RUN_SUMMARY.json"

        assert summary_file.exists(), (
            f"RUN_SUMMARY.json not found at {summary_file}"
        )
        print(f"\n   RUN_SUMMARY.json: ✓")

    def test_execute_plan_model_type_elm(self,
                                          minimal_elm_plan,
                                          tmp_path):
        """execute_plan() run summary identifies model as ELM."""
        manager = ELMExpManager(
            base_output_dir = str(tmp_path)
        )
        run_summary = manager.execute_plan(
            experiment_plan = minimal_elm_plan,
            config          = {'use_cache': True},
        )
        assert run_summary['model_type'] == 'elm'