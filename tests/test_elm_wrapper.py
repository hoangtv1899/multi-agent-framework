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
from elm_input_agent        import ELMAgentAdapter, ELM_AVAILABLE
from elm_experiment_builder import ELMExperimentBuilder
# ELMResultsAnalyzer was DELETED 2026-08-13 — it was the second writer of
# 03_results/extracted.json, and two producers of one artifact is the shape of
# bug that cost a day. Its tests (the old LEVEL 3 block) went with it. Rows are
# built by mcp/elm-mcp/src/column_rows.py::build_rows now, from the artifact
# that one producer writes; TARGET_VARIABLES and VARIABLE_UNITS live in
# mcp/elm-mcp/src/extract.py.
#
# THE FILE STOPPED COLLECTING when the module went, so pytest failed at import
# and every test below — the adapter, the builder, the manager — was silently
# absent from the suite for a day. A red file is visible; a file that does not
# collect is not.
from elm_exp_manager        import ELMExpManager

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
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            assert adapter.model_type == 'elm'

    def test_elm_adapter_not_ready_before_prepare(self):
        """ELMAgentAdapter.is_ready() is False before prepare_case()."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            assert adapter.is_ready() is False

    def test_elm_adapter_repr(self):
        """ELMAgentAdapter.__repr__() contains model type."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            adapter = ELMAgentAdapter('test_case')
            r = repr(adapter)
            assert 'elm' in r
            assert 'not ready' in r

    def test_elm_adapter_forwards_runtime_config_to_wrapper(self):
        """ELMAgentAdapter forwards runtime_config to the wrapper unchanged."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
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
            #
            # prescribed_weather=None is the SITE PATH, and is asserted rather
            # than ignored: a design that never mentioned written weather must
            # reach the wrapper saying so, not saying nothing. A default that
            # drifted to something else would otherwise be invisible here.
            mock_elm.assert_called_once_with(
                case_suffix    = 'test_case',
                runtime_config = config,
                prescribed_weather = None,
            )

    def test_elm_adapter_forwards_unknown_keys_to_wrapper(self):
        """Adapter forwards all keys; wrapper handles unknown-key validation."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            config = {
                'STOP_N':      '5',
                'UNKNOWN_KEY': 'some_value',   # the wrapper will REFUSE it
            }
            ELMAgentAdapter('test_case', runtime_config=config)

            # The adapter does not filter; the wrapper's __init__ validates.
            # It RAISES on an unknown key as of 2026-08-14 — see
            # TestAnUnknownRuntimeKeyIsRefused below. Until then it logged a
            # warning and used the DEFAULT in the key's place, so a misspelt
            # STOP_OPTION built a case that ran the wrong length with the only
            # evidence a line in a server log.
            mock_elm.assert_called_once_with(
                case_suffix    = 'test_case',
                runtime_config = config,
                prescribed_weather = None,
            )

    def test_elm_adapter_run_summary_structure(self):
        """get_run_summary() returns all required + ELM-specific keys."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()
            assert len(experiments) == 3

    def test_builder_correct_experiment_names(self, minimal_plan):
        """Builder extracts correct experiment names."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            indices = [e['scenario_index'] for e in experiments]
            assert indices == [0, 1, 2]

    def test_builder_missing_couplers_raises(self, empty_plan):
        """Builder raises ValueError when CONDITIONS_COUPLERS is empty."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder = ELMExperimentBuilder(empty_plan)
            with pytest.raises(ValueError) as exc_info:
                builder.build_experiments()
            assert 'CONDITIONS_COUPLERS' in str(exc_info.value)

    def test_builder_summary_structure(self, minimal_plan):
        """get_experiment_summary() has correct structure."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
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
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(single_experiment_plan)
            experiments = builder.build_experiments()
            assert len(experiments) == 1
            assert experiments[0]['scenario_index'] == 0

    def test_builder_elm_config_passed_to_overrides(self, minimal_plan):
        """ELM_CONFIG base settings appear in config overrides."""
        with patch(
            'elm_input_agent.GeneratedELMAgent'
        ) as mock_elm:
            mock_elm.return_value = MagicMock(is_built=False)
            builder     = ELMExperimentBuilder(minimal_plan)
            experiments = builder.build_experiments()

            # Check config was passed to GeneratedELMAgent
            call_kwargs = mock_elm.call_args
            assert call_kwargs is not None

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

# ─────────────────────────────────────────────────────────────────────
# AN UNKNOWN RUNTIME KEY IS REFUSED, NOT DROPPED
# ─────────────────────────────────────────────────────────────────────
class TestAnUnknownRuntimeKeyIsRefused:
    """RUNTIME_KEYS is a whitelist over what the planner may override on a
    case. It used to warn and carry on, which meant the DEFAULT was silently
    substituted: a case that builds, runs, and simulates the wrong period.
    """

    def test_an_unknown_key_raises(self):
        from elm_wrapper import GeneratedELMAgent
        with pytest.raises(ValueError, match="Unknown runtime key"):
            GeneratedELMAgent(runtime_config={'NOT_A_REAL_KEY': '1'})

    def test_a_misspelt_key_raises_rather_than_using_the_default(self):
        """The failure this exists for: STOP_OPTION mistyped once."""
        from elm_wrapper import GeneratedELMAgent, DEFAULT_RUNTIME
        with pytest.raises(ValueError) as e:
            GeneratedELMAgent(runtime_config={'STOP_OPTIN': 'nyears'})
        assert 'STOP_OPTIN' in str(e.value)
        assert 'STOP_OPTION' in str(e.value), "say what the allowed keys are"
        assert DEFAULT_RUNTIME['STOP_OPTION'], "the default it would have used"

    def test_the_error_names_every_bad_key(self):
        from elm_wrapper import GeneratedELMAgent
        with pytest.raises(ValueError) as e:
            GeneratedELMAgent(runtime_config={'A_BAD_KEY': '1', 'B_BAD_KEY': '2'})
        assert 'A_BAD_KEY' in str(e.value) and 'B_BAD_KEY' in str(e.value)

    def test_a_good_key_still_applies(self):
        from elm_wrapper import GeneratedELMAgent
        a = GeneratedELMAgent(runtime_config={'STOP_N': '7'})
        assert a.runtime_config['STOP_N'] == '7'

    def test_values_are_stringified(self):
        """xmlchange takes strings; an int here used to reach it as an int."""
        from elm_wrapper import GeneratedELMAgent
        a = GeneratedELMAgent(runtime_config={'STOP_N': 7})
        assert a.runtime_config['STOP_N'] == '7'

    def test_no_config_leaves_the_defaults_intact(self):
        from elm_wrapper import GeneratedELMAgent, DEFAULT_RUNTIME
        a = GeneratedELMAgent()
        assert a.runtime_config == DEFAULT_RUNTIME
        assert a.runtime_config is not DEFAULT_RUNTIME, "must be a copy"
