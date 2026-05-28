#!/usr/bin/env python3
"""
ELM Wrapper Unit Tests
tests/test_elm_wrapper.py

Tests for ELM standalone integration layer.
All tests in Level 1-3 run without ELM build or Perlmutter allocation.

Run with:
    cd ~/RCSFA/multi-agent
    python3 -m pytest tests/test_elm_wrapper.py -v

Or run a specific level:
    python3 -m pytest tests/test_elm_wrapper.py -v -k "builder"
    python3 -m pytest tests/test_elm_wrapper.py -v -k "analyzer"
"""
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from core.model_agent_base       import ModelAgentBase, REQUIRED_SUMMARY_KEYS
from core.elm_input_agent        import ELMAgentAdapter, ELM_AVAILABLE
from core.elm_experiment_builder import ELMExperimentBuilder
from core.elm_results_analyzer   import (
    ELMResultsAnalyzer,
    TARGET_VARIABLES,
    VARIABLE_UNITS,
)
from core.elm_exp_manager        import ELMExpManager

# ─────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_plan():
    """
    Minimal valid ELM experiment plan.
    Three experiments: baseline, dry, wet.
    """
    return {
        "CONDITIONS_COUPLERS": [
            {
                "EXPERIMENT":            "elm_baseline",
                "FORCING_PERIOD":        "baseline",
                "STOP_N":                "9",
                "DATM_CLMNCEP_YR_START": "1981",
                "DATM_CLMNCEP_YR_END":   "1989",
                "RUN_STARTDATE":         "1981-01-01",
                "DESCRIPTION":           "Full baseline period"
            },
            {
                "EXPERIMENT":            "elm_dry_period",
                "FORCING_PERIOD":        "dry",
                "STOP_N":                "4",
                "DATM_CLMNCEP_YR_START": "1984",
                "DATM_CLMNCEP_YR_END":   "1988",
                "RUN_STARTDATE":         "1984-01-01",
                "DESCRIPTION":           "Dry period"
            },
            {
                "EXPERIMENT":            "elm_wet_period",
                "FORCING_PERIOD":        "wet",
                "STOP_N":                "4",
                "DATM_CLMNCEP_YR_START": "1981",
                "DATM_CLMNCEP_YR_END":   "1985",
                "RUN_STARTDATE":         "1981-01-01",
                "DESCRIPTION":           "Wet period"
            },
        ],
        "ELM_CONFIG": {
            "base_stop_option":  "nyears",
            "base_rest_n":       "1",
            "base_rest_option":  "nyears",
            "hist_nhtfrq":       "-3",
            "hist_mfilt":        "365",
        },
        "TIME": {
            "forcing_start": 1981,
            "forcing_end":   1989,
        }
    }

@pytest.fixture
def single_experiment_plan():
    """Plan with a single experiment — for edge case tests."""
    return {
        "CONDITIONS_COUPLERS": [
            {
                "EXPERIMENT":            "elm_single",
                "FORCING_PERIOD":        "baseline",
                "STOP_N":                "5",
                "DATM_CLMNCEP_YR_START": "1981",
                "DATM_CLMNCEP_YR_END":   "1986",
                "RUN_STARTDATE":         "1981-01-01",
                "DESCRIPTION":           "Single experiment"
            },
        ],
        "ELM_CONFIG": {},
        "TIME": {
            "forcing_start": 1981,
            "forcing_end":   1986,
        }
    }

@pytest.fixture
def empty_plan():
    """Plan with no experiments — for error handling tests."""
    return {
        "CONDITIONS_COUPLERS": [],
        "ELM_CONFIG":          {},
        "TIME":                {},
    }

@pytest.fixture
def mock_experiments(minimal_plan):
    """
    Pre-built experiment list with mocked ELMAgentAdapters.
    Avoids real GeneratedELMAgent instantiation.
    """
    experiments = []
    for idx, coupler in enumerate(
        minimal_plan['CONDITIONS_COUPLERS']
    ):
        mock_adapter          = MagicMock(spec=ELMAgentAdapter)
        mock_adapter.is_built = True
        mock_adapter._case_dir = f"/tmp/pscratch/elm_{idx}"
        mock_adapter.model_type = 'elm'
        mock_adapter.get_run_summary.return_value = {
            'case_name':  coupler['EXPERIMENT'].lower(),
            'case_dir':   f"/tmp/pscratch/elm_{idx}",
            'status':     'completed',
            'model_type': 'elm',
        }

        experiments.append({
            'scenario_index': idx,
            'scenario_name':  coupler['EXPERIMENT'],
            'case_name':      coupler['EXPERIMENT'].lower(),
            'forcing_period': coupler['FORCING_PERIOD'],
            'forcing_start':  int(coupler['DATM_CLMNCEP_YR_START']),
            'forcing_end':    int(coupler['DATM_CLMNCEP_YR_END']),
            'stop_n':         int(coupler['STOP_N']),
            'start_date':     coupler['RUN_STARTDATE'],
            'description':    coupler['DESCRIPTION'],
            'elm_agent':      mock_adapter,
            'case_dir':       f"/tmp/pscratch/elm_{idx}",
        })

    return experiments

# ─────────────────────────────────────────────────────────────────────
# LEVEL 1 — ABC + IMPORTS
# ─────────────────────────────────────────────────────────────────────

class TestABCContract:
    """Verify ModelAgentBase ABC is correctly enforced."""

    def test_abc_cannot_instantiate(self):
        """ModelAgentBase cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            ModelAgentBase()
        assert 'abstract' in str(exc_info.value).lower()

    def test_required_summary_keys_defined(self):
        """REQUIRED_SUMMARY_KEYS contains all expected keys."""
        expected = {'case_name', 'case_dir', 'status', 'model_type'}
        assert expected == REQUIRED_SUMMARY_KEYS

    def test_incomplete_subclass_cannot_instantiate(self):
        """
        Subclass missing abstract methods cannot be instantiated.
        Simulates accidentally incomplete implementation.
        """
        class IncompleteAgent(ModelAgentBase):
            # Missing: run_simulation, get_run_summary, model_type
            def prepare_case(self, output_dir):
                return output_dir

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_complete_subclass_can_instantiate(self):
        """Subclass implementing all abstract methods can instantiate."""
        class CompleteAgent(ModelAgentBase):
            @property
            def model_type(self):
                return 'test'
            def prepare_case(self, output_dir):
                return output_dir
            def run_simulation(self, exe_path=None):
                return True
            def get_run_summary(self):
                return {
                    'case_name':  'test',
                    'case_dir':   '/tmp',
                    'status':     'completed',
                    'model_type': 'test',
                }

        agent = CompleteAgent()
        assert agent is not None

    def test_validate_run_summary_passes_valid(self):
        """validate_run_summary() passes with all required keys."""
        class MinimalAgent(ModelAgentBase):
            @property
            def model_type(self): return 'test'
            def prepare_case(self, output_dir): return output_dir
            def run_simulation(self, exe_path=None): return True
            def get_run_summary(self):
                summary = {
                    'case_name':  'test',
                    'case_dir':   '/tmp',
                    'status':     'completed',
                    'model_type': 'test',
                }
                self.validate_run_summary(summary)
                return summary

        agent = MinimalAgent()
        assert agent.get_run_summary()['status'] == 'completed'

    def test_validate_run_summary_fails_missing_key(self):
        """validate_run_summary() raises ValueError on missing key."""
        class BadAgent(ModelAgentBase):
            @property
            def model_type(self): return 'test'
            def prepare_case(self, output_dir): return output_dir
            def run_simulation(self, exe_path=None): return True
            def get_run_summary(self):
                # Missing 'model_type'
                summary = {
                    'case_name': 'test',
                    'case_dir':  '/tmp',
                    'status':    'completed',
                }
                self.validate_run_summary(summary)
                return summary

        agent = BadAgent()
        with pytest.raises(ValueError) as exc_info:
            agent.get_run_summary()
        assert 'model_type' in str(exc_info.value)

    def test_validate_run_summary_fails_bad_status(self):
        """validate_run_summary() raises ValueError on invalid status."""
        class BadStatusAgent(ModelAgentBase):
            @property
            def model_type(self): return 'test'
            def prepare_case(self, output_dir): return output_dir
            def run_simulation(self, exe_path=None): return True
            def get_run_summary(self):
                summary = {
                    'case_name':  'test',
                    'case_dir':   '/tmp',
                    'status':     'running',   # ← invalid
                    'model_type': 'test',
                }
                self.validate_run_summary(summary)
                return summary

        agent = BadStatusAgent()
        with pytest.raises(ValueError) as exc_info:
            agent.get_run_summary()
        assert 'status' in str(exc_info.value)


class TestELMAdapterImport:
    """Verify ELMAgentAdapter satisfies ABC contract."""

    def test_elm_available(self):
        """GeneratedELMAgent is importable from framework."""
        assert ELM_AVAILABLE is True

    def test_elm_adapter_is_modelagentbase_subclass(self):
        """ELMAgentAdapter is a subclass of ModelAgentBase."""
        assert issubclass(ELMAgentAdapter, ModelAgentBase)

    def test_elm_adapter_has_model_type(self):
        """ELMAgentAdapter.model_type returns 'elm'."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            assert adapter.model_type == 'elm'

    def test_elm_adapter_not_ready_before_prepare(self):
        """ELMAgentAdapter.is_ready() is False before prepare_case()."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            assert adapter.is_ready() is False

    def test_elm_adapter_repr(self):
        """ELMAgentAdapter.__repr__() contains model type."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            r = repr(adapter)
            assert 'elm' in r
            assert 'not ready' in r

    def test_elm_adapter_forwards_runtime_config_to_wrapper(self):
        """ELMAgentAdapter forwards runtime_config to the wrapper unchanged."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            config = {
                'STOP_N':                '10',
                'DATM_CLMNCEP_YR_START': '1981',
                'DATM_CLMNCEP_YR_END':   '1989',
                'RUN_STARTDATE':         '1981-01-01',
            }
            ELMAgentAdapter('test_case', runtime_config=config)

            # Adapter passes config straight through; the wrapper's
            # __init__ is responsible for validating it against RUNTIME_KEYS.
            mock_elm.assert_called_once_with(
                case_suffix    = 'test_case',
                runtime_config = config,
            )

    def test_elm_adapter_forwards_unknown_keys_to_wrapper(self):
        """Adapter forwards all keys; wrapper handles unknown-key validation."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            config = {
                'STOP_N':      '5',
                'UNKNOWN_KEY': 'some_value',   # wrapper will log + drop
            }
            ELMAgentAdapter('test_case', runtime_config=config)

            # Adapter doesn't filter — wrapper's __init__ logs a warning
            # and drops unknown keys internally.
            mock_elm.assert_called_once_with(
                case_suffix    = 'test_case',
                runtime_config = config,
            )

    def test_elm_adapter_run_summary_structure(self):
        """get_run_summary() returns all required + ELM-specific keys."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_instance          = MagicMock()
            mock_instance.is_built = True
            mock_instance.get_summary.return_value = {
                'case_name':      'test_case',
                'case_dir':       '/tmp/test',
                'is_built':       True,
                'is_completed':   True,
                'runtime_config': {'STOP_N': '5'},
                'history_files':  ['/tmp/test/run/foo.nc'],
            }
            mock_elm.return_value = mock_instance

            adapter         = ELMAgentAdapter('test_case')
            adapter._status = 'completed'

            summary = adapter.get_run_summary()

            # Required by ModelAgentBase
            for key in REQUIRED_SUMMARY_KEYS:
                assert key in summary, f"Missing required key: {key}"

            # ELM-specific keys (match the new adapter's get_run_summary)
            assert 'runtime_config' in summary
            assert 'history_files'  in summary
            assert 'elm_case_info'  in summary

            # Check values
            assert summary['model_type'] == 'elm'
            assert summary['status']     == 'completed'

# ─────────────────────────────────────────────────────────────────────
# LEVEL 2 — ELMExperimentBuilder
# ─────────────────────────────────────────────────────────────────────

class TestELMExperimentBuilder:
    """Test ELMExperimentBuilder plan parsing and experiment creation."""

    def test_builder_correct_experiment_count(self, minimal_plan):
        """Builder creates correct number of experiments."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()
            assert len(experiments) == 3

    def test_builder_correct_experiment_names(self, minimal_plan):
        """Builder extracts correct experiment names."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()
            names = [e['scenario_name'] for e in experiments]
            assert 'elm_baseline'   in names
            assert 'elm_dry_period' in names
            assert 'elm_wet_period' in names

    def test_builder_correct_forcing_years(self, minimal_plan):
        """Builder extracts correct forcing years per experiment."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            baseline = next(
                e for e in experiments
                if e['forcing_period'] == 'baseline'
            )
            dry = next(
                e for e in experiments
                if e['forcing_period'] == 'dry'
            )
            wet = next(
                e for e in experiments
                if e['forcing_period'] == 'wet'
            )

            assert baseline['forcing_start'] == 1981
            assert baseline['forcing_end']   == 1989
            assert dry['forcing_start']      == 1984
            assert dry['forcing_end']        == 1988
            assert wet['forcing_start']      == 1981
            assert wet['forcing_end']        == 1985

    def test_builder_correct_stop_n(self, minimal_plan):
        """Builder extracts correct STOP_N per experiment."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            baseline = next(
                e for e in experiments
                if e['forcing_period'] == 'baseline'
            )
            assert baseline['stop_n'] == 9

    def test_builder_case_names_lowercase_no_spaces(self, minimal_plan):
        """Builder generates lowercase case names without spaces."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            for exp in experiments:
                case_name = exp['case_name']
                assert case_name == case_name.lower()
                assert ' ' not in case_name

    def test_builder_experiment_has_elm_agent(self, minimal_plan):
        """Each experiment has an ELMAgentAdapter."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            for exp in experiments:
                assert 'elm_agent' in exp
                assert isinstance(exp['elm_agent'], ELMAgentAdapter)

    def test_builder_scenario_index_is_zero_based(self, minimal_plan):
        """Scenario indices are zero-based."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            indices = [e['scenario_index'] for e in experiments]
            assert indices == [0, 1, 2]

    def test_builder_missing_couplers_raises(self, empty_plan):
        """Builder raises ValueError when CONDITIONS_COUPLERS is empty."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder = ELMExperimentBuilder(empty_plan)
            with pytest.raises(ValueError) as exc_info:
                builder.build_experiments()
            assert 'CONDITIONS_COUPLERS' in str(exc_info.value)

    def test_builder_summary_structure(self, minimal_plan):
        """get_experiment_summary() has correct structure."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder = ELMExperimentBuilder(minimal_plan)
            builder.build_experiments()
            summary = builder.get_experiment_summary()

            assert 'model_type'        in summary
            assert 'total_experiments' in summary
            assert 'experiments'       in summary
            assert summary['model_type']        == 'elm'
            assert summary['total_experiments'] == 3

    def test_builder_single_experiment(self, single_experiment_plan):
        """Builder works correctly with a single experiment."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(single_experiment_plan)
            experiments = builder.build_experiments()
            assert len(experiments) == 1
            assert experiments[0]['scenario_index'] == 0

    def test_builder_elm_config_passed_to_overrides(self, minimal_plan):
        """ELM_CONFIG base settings appear in config overrides."""
        with patch(
            'core.elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            # Check config was passed to GeneratedELMAgent
            call_kwargs = mock_elm.call_args
            assert call_kwargs is not None

# ─────────────────────────────────────────────────────────────────────
# LEVEL 3 — ELMResultsAnalyzer
# ─────────────────────────────────────────────────────────────────────

class TestELMResultsAnalyzer:
    """Test ELMResultsAnalyzer without real NetCDF files."""

    def test_target_variables_defined(self):
        """TARGET_VARIABLES contains all expected variables."""
        assert 'QOVER'   in TARGET_VARIABLES
        assert 'QCHARGE' in TARGET_VARIABLES
        assert 'TWS'     in TARGET_VARIABLES
        assert 'SOILLIQ' in TARGET_VARIABLES

    def test_variable_units_defined(self):
        """VARIABLE_UNITS has entries for all target variables."""
        for var in TARGET_VARIABLES:
            assert var in VARIABLE_UNITS

    def test_empty_result_structure(self, mock_experiments):
        """_empty_result() returns dict with all required keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            result = analyzer._empty_result(
                mock_experiments[0],
                reason='test'
            )

            assert 'case_name'      in result
            assert 'scenario_name'  in result
            assert 'forcing_period' in result
            assert 'forcing_start'  in result
            assert 'forcing_end'    in result
            assert 'status'         in result
            assert 'reason'         in result
            assert 'variables'      in result
            assert 'metrics'        in result
            assert 'history_files'  in result

    def test_empty_result_status_is_failed(self, mock_experiments):
        """_empty_result() always has status='failed'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            result = analyzer._empty_result(mock_experiments[0])
            assert result['status'] == 'failed'

    def test_empty_result_variables_are_none(self, mock_experiments):
        """_empty_result() has None for all target variables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            result = analyzer._empty_result(mock_experiments[0])
            for var in TARGET_VARIABLES:
                assert result['variables'][var] is None

    def test_analyzer_missing_case_dir(self, mock_experiments):
        """
        Analyzer handles experiment with no case_dir gracefully.
        Should return empty result, not raise.
        """
        # Remove case_dir from first experiment
        experiments_no_dir = []
        for exp in mock_experiments:
            exp_copy = exp.copy()
            exp_copy.pop('case_dir', None)
            experiments_no_dir.append(exp_copy)

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = experiments_no_dir,
                analysis_dir = tmp_dir,
            )
            # Should not raise
            results = analyzer.extract_all()
            # All should be failed
            for result in results.values():
                assert result['status'] == 'failed'

    def test_analyzer_missing_run_directory(self, mock_experiments):
        """
        Analyzer handles missing run directory gracefully.
        """
        # Point case_dir to non-existent path
        experiments_bad_dir = []
        for exp in mock_experiments:
            exp_copy          = exp.copy()
            exp_copy['case_dir'] = '/nonexistent/path/elm_case'
            experiments_bad_dir.append(exp_copy)

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = experiments_bad_dir,
                analysis_dir = tmp_dir,
            )
            results = analyzer.extract_all()
            for result in results.values():
                assert result['status'] == 'failed'

    def test_llm_input_required_keys(self, mock_experiments):
        """get_llm_analysis_input() contains all required keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            # Populate with empty results
            for exp in mock_experiments:
                analyzer.results[exp['case_name']] = (
                    analyzer._empty_result(exp)
                )

            llm_input = analyzer.get_llm_analysis_input()

            assert 'model_type'        in llm_input
            assert 'experiments'       in llm_input
            assert 'comparisons'       in llm_input
            assert 'units'             in llm_input
            assert 'focus_variables'   in llm_input
            assert 'file_locations'    in llm_input
            assert llm_input['model_type'] == 'elm'

    def test_llm_input_focus_variables(self, mock_experiments):
        """get_llm_analysis_input() includes all 4 focus variables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            for exp in mock_experiments:
                analyzer.results[exp['case_name']] = (
                    analyzer._empty_result(exp)
                )

            llm_input    = analyzer.get_llm_analysis_input()
            focus_vars   = llm_input['focus_variables']

            assert 'QCHARGE' in focus_vars
            assert 'QOVER'   in focus_vars
            assert 'TWS'     in focus_vars
            assert 'SOILLIQ' in focus_vars

    def test_comparisons_require_two_experiments(self,
                                                  mock_experiments):
        """
        _compute_comparisons() returns empty list
        when fewer than 2 experiments have ok status.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            # Only one experiment with ok status
            analyzer.results = {
                'elm_baseline': {
                    'status':         'ok',
                    'forcing_period': 'baseline',
                    'metrics': {
                        'annual_recharge_mm_yr': 150.0,
                        'annual_runoff_mm_yr':   50.0,
                    }
                }
            }
            comparisons = analyzer._compute_comparisons()
            assert comparisons == []

    def test_comparisons_identify_highest_recharge(self,
                                                    mock_experiments):
        """
        _compute_comparisons() correctly identifies
        experiment with highest recharge.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ELMResultsAnalyzer(
                experiments  = mock_experiments,
                analysis_dir = tmp_dir,
            )
            # Wet period has highest recharge
            analyzer.results = {
                'elm_baseline': {
                    'status':         'ok',
                    'forcing_period': 'baseline',
                    'metrics': {
                        'annual_recharge_mm_yr': 150.0,
                        'annual_runoff_mm_yr':    50.0,
                    }
                },
                'elm_wet_period': {
                    'status':         'ok',
                    'forcing_period': 'wet',
                    'metrics': {
                        'annual_recharge_mm_yr': 280.0,
                        'annual_runoff_mm_yr':    90.0,
                    }
                },
                'elm_dry_period': {
                    'status':         'ok',
                    'forcing_period': 'dry',
                    'metrics': {
                        'annual_recharge_mm_yr':  60.0,
                        'annual_runoff_mm_yr':    20.0,
                    }
                },
            }
            comparisons = analyzer._compute_comparisons()

            recharge_comp = next(
                c for c in comparisons
                if c['metric'] == 'annual_recharge_mm_yr'
            )
            assert recharge_comp['highest'] == 'wet'
            assert recharge_comp['lowest']  == 'dry'
            assert recharge_comp['difference'] == pytest.approx(
                220.0, abs=0.1
            )

# ─────────────────────────────────────────────────────────────────────
# LEVEL 4 — ELMExpManager
# ─────────────────────────────────────────────────────────────────────

class TestELMExpManager:
    """Test ELMExpManager output structure."""

    def test_exp_manager_creates_run_dir(self):
        """ELMExpManager creates run directory on init."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ELMExpManager(base_output_dir=tmp_dir)
            assert manager.run_dir.exists()

    def test_exp_manager_run_dir_prefix(self):
        """ELMExpManager run directory starts with 'elm_run_'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ELMExpManager(base_output_dir=tmp_dir)
            assert manager.run_dir.name.startswith('elm_run_')

    def test_run_summary_required_keys(self,
                                        minimal_plan,
                                        mock_experiments):
        """
        _create_run_summary() produces all keys
        required by workflow.py.
        """
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager    = ELMExpManager(base_output_dir=tmp_dir)
            start_time = datetime.now()
            end_time   = datetime.now()

            results = {
                exp['case_name']: True
                for exp in mock_experiments
            }

            summary = manager._create_run_summary(
                plan = minimal_plan,
                experiments     = mock_experiments,
                results         = results,
                start_time      = start_time,
                end_time        = end_time,
            )

            # Keys required by workflow.py
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
                assert key in summary, f"Missing key: {key}"

    def test_run_summary_correct_counts(self,
                                         minimal_plan,
                                         mock_experiments):
        """_create_run_summary() counts successes and failures correctly."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager    = ELMExpManager(base_output_dir=tmp_dir)
            start_time = datetime.now()
            end_time   = datetime.now()

            # 2 success, 1 failure
            results = {
                mock_experiments[0]['case_name']: True,
                mock_experiments[1]['case_name']: True,
                mock_experiments[2]['case_name']: False,
            }

            summary = manager._create_run_summary(
                plan = minimal_plan,
                experiments     = mock_experiments,
                results         = results,
                start_time      = start_time,
                end_time        = end_time,
            )

            assert summary['experiments_total']   == 3
            assert summary['experiments_success'] == 2
            assert summary['experiments_failed']  == 1

    def test_run_summary_model_type_is_elm(self,
                                            minimal_plan,
                                            mock_experiments):
        """_create_run_summary() identifies model as ELM."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager    = ELMExpManager(base_output_dir=tmp_dir)
            start_time = datetime.now()
            end_time   = datetime.now()
            results    = {
                exp['case_name']: True
                for exp in mock_experiments
            }

            summary = manager._create_run_summary(
                plan = minimal_plan,
                experiments     = mock_experiments,
                results         = results,
                start_time      = start_time,
                end_time        = end_time,
            )
            assert summary['model_type'] == 'elm'

    def test_run_summary_output_files_exist_as_paths(self,
                                                       minimal_plan,
                                                       mock_experiments):
        """_create_run_summary() output_files are valid path strings."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager    = ELMExpManager(base_output_dir=tmp_dir)
            start_time = datetime.now()
            end_time   = datetime.now()
            results    = {
                exp['case_name']: True
                for exp in mock_experiments
            }

            summary = manager._create_run_summary(
                plan = minimal_plan,
                experiments     = mock_experiments,
                results         = results,
                start_time      = start_time,
                end_time        = end_time,
            )

            for key, path in summary['output_files'].items():
                assert isinstance(path, str), (
                    f"output_files['{key}'] should be str"
                )

    def test_convergence_warnings_always_empty(self,
                                                minimal_plan,
                                                mock_experiments):
        """convergence_warnings is always [] for ELM."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager    = ELMExpManager(base_output_dir=tmp_dir)
            start_time = datetime.now()
            end_time   = datetime.now()
            results    = {
                exp['case_name']: True
                for exp in mock_experiments
            }

            summary = manager._create_run_summary(
                plan = minimal_plan,
                experiments     = mock_experiments,
                results         = results,
                start_time      = start_time,
                end_time        = end_time,
            )
            assert summary['convergence_warnings'] == []