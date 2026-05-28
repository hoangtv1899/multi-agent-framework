#!/usr/bin/env python3
"""
Phase B regression tests for ELM setup plotting.
~/RCSFA/multi-agent/tests/test_elm_setup_plotting.py

Verifies:
    1. The new src/core/elm_setup_plotting.py module imports
       and exposes three public functions.
    2. plot_domain_configuration() produces a non-empty PNG file
       for both native (with layers) and synthetic-soil experiments,
       and degrades gracefully when soil data is missing.
    3. compare_experiments() and compare_forcing_conditions() produce
       non-empty PNG files for a 3-experiment suite.
    4. ELMExpManager._plot_setups() creates the expected directory
       structure under 02_setup_plots/:
           exp_NNN_<case_name>/domain_configuration.png  (per-experiment)
           comparison_experiments.png                     (top-level)
           comparison_forcing_conditions.png              (top-level)

USAGE
─────
    cd ~/RCSFA/multi-agent
    python3 -m pytest tests/test_elm_setup_plotting.py -v

These tests need matplotlib. They run in seconds, no SLURM needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

# Skip the whole module if matplotlib isn't available — these tests
# can't do anything useful without it.
matplotlib = pytest.importorskip("matplotlib")

from core.elm_setup_plotting import (
    plot_domain_configuration,
    compare_experiments,
    compare_forcing_conditions,
)
from core.elm_exp_manager import ELMExpManager


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def native_experiment():
    """A native-soil experiment with SUBSTRATE."""
    return {
        'case_name':      'elm_baseline',
        'scenario_name':  'baseline_5yr',
        'forcing_period': 'baseline',
        'forcing_start':  1980,
        'forcing_end':    1984,
        'soil_config':    'native',
        'substrate':      'extrapolate',
        'stop_n':         5,
    }


@pytest.fixture
def synthetic_experiment():
    """A synthetic-soil experiment (no SUBSTRATE)."""
    return {
        'case_name':      'elm_sandy_baseline',
        'scenario_name':  'sandy_5yr',
        'forcing_period': 'baseline',
        'forcing_start':  1985,
        'forcing_end':    1989,
        'soil_config':    'sandy',
        'substrate':      None,
        'stop_n':         5,
    }


@pytest.fixture
def elm_config_with_soil():
    """ELM_CONFIG containing lat/lon and a 4-layer SSURGO profile."""
    return {
        'lat':              46.50,
        'lon':              -119.30,
        'base_stop_option': 'nyears',
        'base_rest_n':      '1',
        'base_rest_option': 'nyears',
        'soil_profile': {
            'num_layers':       4,
            'source':           'SSURGO',
            'depth_coverage_m': 1.8,
            'layers': [
                {'depth_top_m': 0.00, 'depth_bot_m': 0.30,
                 'sand_pct': 65, 'clay_pct': 12,
                 'organic_pct': 2.0, 'gravel_pct':  5},
                {'depth_top_m': 0.30, 'depth_bot_m': 0.80,
                 'sand_pct': 60, 'clay_pct': 18,
                 'organic_pct': 1.0, 'gravel_pct':  8},
                {'depth_top_m': 0.80, 'depth_bot_m': 1.30,
                 'sand_pct': 55, 'clay_pct': 22,
                 'organic_pct': 0.5, 'gravel_pct': 12},
                {'depth_top_m': 1.30, 'depth_bot_m': 1.80,
                 'sand_pct': 50, 'clay_pct': 25,
                 'organic_pct': 0.2, 'gravel_pct': 15},
            ],
        },
    }


@pytest.fixture
def three_experiments():
    """A typical 3-experiment forcing-only suite."""
    return [
        {'case_name': 'elm_baseline', 'scenario_name': 'baseline',
         'forcing_period': 'baseline',
         'forcing_start': 1980, 'forcing_end': 1984,
         'soil_config': 'native', 'substrate': 'extrapolate',
         'stop_n': 5},
        {'case_name': 'elm_dry', 'scenario_name': 'dry',
         'forcing_period': 'dry',
         'forcing_start': 1976, 'forcing_end': 1980,
         'soil_config': 'native', 'substrate': 'extrapolate',
         'stop_n': 5},
        {'case_name': 'elm_wet', 'scenario_name': 'wet',
         'forcing_period': 'wet',
         'forcing_start': 1996, 'forcing_end': 2000,
         'soil_config': 'native', 'substrate': 'extrapolate',
         'stop_n': 5},
    ]


@pytest.fixture
def three_experiments_plan(elm_config_with_soil):
    """Minimal plan dict the way ELMExpManager._build receives it."""
    return {
        'CONDITIONS_COUPLERS': [],  # not used by _plot_setups
        'ELM_CONFIG':          elm_config_with_soil,
        'TIME': {'forcing_start': 1948, 'forcing_end': 2004},
    }


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 1 — DOMAIN CONFIGURATION FIGURE
# ═════════════════════════════════════════════════════════════════════

class TestPlotDomainConfiguration:
    """plot_domain_configuration() creates files for various inputs."""

    def test_creates_file_with_native_soil(
            self, tmp_path, native_experiment, elm_config_with_soil):
        out = tmp_path / "domain.png"
        ok  = plot_domain_configuration(
            native_experiment, elm_config_with_soil, str(out))
        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 5000   # PNG header + content

    def test_creates_file_with_synthetic_soil(
            self, tmp_path, synthetic_experiment, elm_config_with_soil):
        out = tmp_path / "domain.png"
        ok  = plot_domain_configuration(
            synthetic_experiment, elm_config_with_soil, str(out))
        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 5000

    def test_handles_missing_soil_profile(
            self, tmp_path, native_experiment):
        """Empty elm_config (no soil_profile) should still produce a figure."""
        out = tmp_path / "domain.png"
        ok  = plot_domain_configuration(
            native_experiment, {}, str(out))
        assert ok is True
        assert out.exists()

    def test_handles_missing_lat_lon(
            self, tmp_path, native_experiment):
        """elm_config without lat/lon should still produce a figure."""
        out = tmp_path / "domain.png"
        ok  = plot_domain_configuration(
            native_experiment, {'soil_profile': {}}, str(out))
        assert ok is True
        assert out.exists()

    def test_handles_empty_experiment(self, tmp_path):
        """Sparsely-populated experiment dict shouldn't crash."""
        out = tmp_path / "domain.png"
        ok  = plot_domain_configuration({}, {}, str(out))
        # May return True (drawing nothing) or False — but must not crash
        assert isinstance(ok, bool)


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 2 — COMPARE EXPERIMENTS TABLE
# ═════════════════════════════════════════════════════════════════════

class TestCompareExperiments:
    """compare_experiments() produces a parameter-matrix figure."""

    def test_creates_file_with_three_experiments(
            self, tmp_path, three_experiments):
        out = tmp_path / "comp.png"
        ok  = compare_experiments(three_experiments, str(out))
        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 5000

    def test_returns_false_for_empty_list(self, tmp_path):
        out = tmp_path / "comp.png"
        ok  = compare_experiments([], str(out))
        assert ok is False
        assert not out.exists()


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 3 — COMPARE FORCING CONDITIONS TIMELINE
# ═════════════════════════════════════════════════════════════════════

class TestCompareForcingConditions:
    """compare_forcing_conditions() produces a timeline figure."""

    def test_creates_file_with_three_experiments(
            self, tmp_path, three_experiments):
        out = tmp_path / "forcing.png"
        ok  = compare_forcing_conditions(three_experiments, str(out))
        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 5000

    def test_returns_false_for_empty_list(self, tmp_path):
        out = tmp_path / "forcing.png"
        ok  = compare_forcing_conditions([], str(out))
        assert ok is False
        assert not out.exists()


# ═════════════════════════════════════════════════════════════════════
# CATEGORY 4 — ELMExpManager._plot_setups INTEGRATION
# ═════════════════════════════════════════════════════════════════════

class TestPlotSetupsHook:
    """ELMExpManager._plot_setups routes files into 02_setup_plots/."""

    def test_per_experiment_subdirs_created(
            self, tmp_path, three_experiments, three_experiments_plan):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._plot_setups(three_experiments, three_experiments_plan)

        for i, exp in enumerate(three_experiments, 1):
            exp_dir = mgr.setup_plots_dir / f"exp_{i:03d}_{exp['case_name']}"
            assert exp_dir.is_dir(), f"missing {exp_dir}"
            domain_file = exp_dir / "domain_configuration.png"
            assert domain_file.exists(), f"missing {domain_file}"
            assert domain_file.stat().st_size > 5000

    def test_top_level_comparisons_created(
            self, tmp_path, three_experiments, three_experiments_plan):
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._plot_setups(three_experiments, three_experiments_plan)

        comp_exp = mgr.setup_plots_dir / "comparison_experiments.png"
        comp_fc  = mgr.setup_plots_dir / "comparison_forcing_conditions.png"
        assert comp_exp.exists()
        assert comp_fc.exists()
        assert comp_exp.stat().st_size > 5000
        assert comp_fc.stat().st_size > 5000

    def test_no_comparisons_with_single_experiment(
            self, tmp_path, three_experiments, three_experiments_plan):
        """Single-experiment plans skip the cross-experiment comparisons."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._plot_setups(three_experiments[:1], three_experiments_plan)

        # Per-experiment plot should exist
        exp_dir = mgr.setup_plots_dir / f"exp_001_{three_experiments[0]['case_name']}"
        assert (exp_dir / "domain_configuration.png").exists()

        # But comparisons should NOT — single experiment is degenerate
        assert not (mgr.setup_plots_dir /
                    "comparison_experiments.png").exists()
        assert not (mgr.setup_plots_dir /
                    "comparison_forcing_conditions.png").exists()

    def test_empty_experiment_list_is_noop(self, tmp_path):
        """Empty experiments → no files written, no error raised."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        mgr._plot_setups([], {})

        # Subdir exists (created by __init__), but is empty
        assert mgr.setup_plots_dir.is_dir()
        assert list(mgr.setup_plots_dir.iterdir()) == []

    def test_missing_elm_config_does_not_crash(
            self, tmp_path, three_experiments):
        """Plan without ELM_CONFIG should not crash _plot_setups."""
        mgr = ELMExpManager(base_output_dir=str(tmp_path))
        # Pass a plan with no ELM_CONFIG block
        mgr._plot_setups(three_experiments, {})

        # Per-experiment plots should still be produced
        for i, exp in enumerate(three_experiments, 1):
            exp_dir = mgr.setup_plots_dir / f"exp_{i:03d}_{exp['case_name']}"
            assert (exp_dir / "domain_configuration.png").exists()


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))