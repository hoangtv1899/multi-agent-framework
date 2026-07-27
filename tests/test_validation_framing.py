"""Offline tests for how validation FRAMES a comparison (no MCP, no network).

The numbers were never the weak part — the framing was. A 204 km2 headwater
gauge scored against a 2861 km2 basin ensemble, imperial units leaking out of
USGS, and 1951-2001 well measurements presented against a 1995 run all produced
verdicts that looked like model failures and were not. These tests pin the
guards that stop that.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vr = _load("vr_mod", "tools/validate_run.py")


# ── units ────────────────────────────────────────────────────────────────────
class TestMetricUnits:
    def test_square_mile_conversion_is_right(self):
        assert vr.MI2_TO_KM2 == pytest.approx(2.58999, rel=1e-6)
        # the real gauge: 78.9 mi2 is 204.3 km2
        assert 78.9 * vr.MI2_TO_KM2 == pytest.approx(204.4, abs=0.2)

    def test_foot_conversion_is_right(self):
        assert vr.FT_TO_M == pytest.approx(0.3048, rel=1e-9)

    def test_area_conversions_are_consistent(self):
        """MI2_TO_M2 is used for the discharge maths, MI2_TO_KM2 for reporting;
        if they ever disagree the yield and the label describe different areas."""
        assert vr.MI2_TO_M2 / 1e6 == pytest.approx(vr.MI2_TO_KM2, rel=1e-9)

    def test_no_imperial_units_in_emitted_field_names(self):
        """Guards the regression: USGS reports mi2 and SNOTEL reports ft, and
        both used to flow straight into validation.json and the figure labels.

        Reading them is fine — conversion happens at the boundary — so this
        looks for dict-KEY form ("name":), not any mention.
        """
        src = (ROOT / "tools" / "validate_run.py").read_text()
        emitted = [ln.strip() for ln in src.splitlines()
                   if '"drainage_mi2":' in ln or '"elevation_ft":' in ln]
        assert emitted == [], f"imperial keys still emitted: {emitted}"

    def test_imperial_is_still_read_at_the_boundary(self):
        """The conversion must have a source: USGS/SNOTEL only offer imperial."""
        src = (ROOT / "tools" / "validate_run.py").read_text()
        assert 'get("elevation_ft")' in src
        assert "MI2_TO_KM2" in src


# ── figures ──────────────────────────────────────────────────────────────────
class TestFigureSplit:
    def test_dispatcher_names_one_figure_per_observable(self):
        src = (ROOT / "tools" / "validate_run.py").read_text()
        for fn in ("plot_hydrograph", "plot_yield", "plot_water_table",
                   "plot_swe", "plot_context"):
            assert f"def {fn}(" in src

    def test_each_figure_degrades_to_none_without_data(self, tmp_path):
        """A missing observation must skip its figure, never crash the run."""
        for fn in (vr.plot_hydrograph, vr.plot_yield, vr.plot_water_table,
                   vr.plot_swe, vr.plot_context):
            assert fn({}, tmp_path / "x.png") is None

    def test_dispatcher_survives_an_empty_validation(self, tmp_path):
        made = vr.plot_validation({}, tmp_path / "validation.png")
        assert made == {}
        assert not list(tmp_path.glob("*.png"))

    def test_context_figure_exists_but_is_never_scored(self):
        """P is model INPUT and there is no in-basin flux tower for ET, so the
        context figure must not carry a verdict."""
        import inspect
        doc = inspect.getdoc(vr.plot_context) or ""
        assert "NOT validation" in doc or "not validation" in doc.lower()
