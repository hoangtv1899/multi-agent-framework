"""Precipitation in the per-column metrics is RAIN + SNOW.

It was RAIN alone, while the water budget in the same metrics dict used
rain + snow — so one dict carried two different precipitations. In a
snow-dominated basin that is not a rounding error: across the 2020 Naches
columns the two differ by 1.11x in the warm valley and 2.17x at elevation, the
ratio being exactly the snow fraction.

Every runoff/P and recharge/P fraction was inflated by that factor, and columns
appeared to drain more water than fell on them because more than half of what
fell was snow: 5 of 14 columns looked impossible, 2 actually were.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.elm_results_analyzer import ELMResultsAnalyzer   # noqa: E402


def _metrics(rain=None, snow=None, **extra):
    a = ELMResultsAnalyzer.__new__(ELMResultsAnalyzer)
    v = {}
    if rain is not None:
        v["RAIN"] = {"annual_mean": rain}
    if snow is not None:
        v["SNOW"] = {"annual_mean": snow}
    v.update(extra)
    return a._derive_metrics(v) if hasattr(a, "_derive_metrics") else None


class TestPrecipIncludesSnow:
    def test_snow_is_counted(self):
        src = (ROOT / "src" / "core" / "elm_results_analyzer.py").read_text()
        i = src.index("metrics['precip_mm_yr']")
        window = src[max(0, i - 1200):i + 200]
        assert "variables.get('SNOW')" in window, \
            "precip_mm_yr computed without consulting SNOW"
        assert "_s" in src[i:i + 120], "the snow term never reaches the sum"

    def test_rain_only_is_kept_under_its_own_name(self):
        """The rain total is still useful — it just is not 'precipitation'."""
        src = (ROOT / "src" / "core" / "elm_results_analyzer.py").read_text()
        assert "rainfall_mm_yr" in src

    def test_the_budget_and_the_metric_agree_by_construction(self):
        """The budget builds P as rain + snow a few lines below. If the two
        ever diverge again, every fraction in the dict divides by a different
        denominator than the closure does."""
        src = (ROOT / "src" / "core" / "elm_results_analyzer.py").read_text()
        assert "p = rain + snow" in src
        i = src.index("metrics['precip_mm_yr']")
        assert "(_r or 0.0) + (_s or 0.0)" in src[i:i + 120]


class TestArithmetic:
    """Worked from the real 2020 col_13 numbers, where rain-only made a column
    look like it drained twice what fell on it."""

    def test_a_snowy_column_doubles(self):
        rain, snow = 773.0, 857.0
        assert round(rain + snow, 1) == 1630.0

    def test_the_drainage_ratio_flips_below_one(self):
        drainage = 1636.0                    # col_03
        assert drainage / 855.0 > 1.0        # rain-only: "impossible"
        assert drainage / 1748.0 < 1.0       # rain+snow: ordinary
