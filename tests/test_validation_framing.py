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


# ── warm-up exclusion (item 3a) ──────────────────────────────────────────────
class TestWarmupExclusion:
    def _series(self, n=200, spike=None):
        obs = {f"1995-{1 + d // 30:02d}-{1 + d % 30:02d}": 5.0 + (d % 7)
               for d in range(n)}
        mod = dict.fromkeys(obs, 5.0)
        if spike is not None:
            mod[list(obs)[0]] = spike
        return obs, mod

    def test_first_days_are_dropped(self):
        obs, mod = self._series()
        m = vr.flow_metrics(obs, mod, warmup_days=30)
        assert m["warmup_days_excluded"] == 30
        assert m["n_days"] == m["n_days_before_warmup_cut"] - 30

    def test_an_initialisation_spike_stops_dominating(self):
        """The regression this exists for: one 405 mm/day day-1 flush drove
        alpha to 6.1 and NSE to -36.9 on the real Naches run."""
        obs, mod = self._series(spike=400.0)
        scored = vr.flow_metrics(obs, mod, warmup_days=0)
        clean = vr.flow_metrics(obs, mod, warmup_days=30)
        assert scored["alpha_var_ratio"] > 5      # spike inflates variability
        assert clean["alpha_var_ratio"] < 1       # and is gone once discarded
        assert clean["NSE"] > scored["NSE"]

    def test_zero_warmup_keeps_everything(self):
        obs, mod = self._series()
        assert (vr.flow_metrics(obs, mod, warmup_days=0)["n_days"]
                == len(set(obs) & set(mod)))

    def test_too_short_after_the_cut_returns_none(self):
        obs, mod = self._series(n=110)
        assert vr.flow_metrics(obs, mod, warmup_days=30) is None


# ── area weighting (item 3b) ─────────────────────────────────────────────────
class TestAreaWeighting:
    COLS = [{"id": "a", "band": 1}, {"id": "b", "band": 2}, {"id": "c", "band": 2}]
    BANDS = [{"band": 1, "grid_points": 90}, {"band": 2, "grid_points": 10}]

    def test_weights_sum_to_one(self):
        w = vr.band_weights(self.COLS, self.BANDS)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_band_area_beats_column_count(self):
        """Band 1 is 90% of the area with ONE column; band 2 is 10% with two.
        An unweighted mean would give band 2 twice band 1's influence."""
        w = vr.band_weights(self.COLS, self.BANDS)
        assert w["a"] == pytest.approx(0.9)
        assert w["b"] == pytest.approx(0.05)
        assert w["c"] == pytest.approx(0.05)

    def test_missing_band_metadata_falls_back_to_equal(self):
        w = vr.band_weights(self.COLS, None)
        assert len(w) == 3
        assert all(v == pytest.approx(1 / 3) for v in w.values())

    def test_weighted_mean_uses_the_weights(self):
        vals = {"a": 100.0, "b": 0.0, "c": 0.0}
        w = vr.band_weights(self.COLS, self.BANDS)
        assert vr.weighted_mean(vals, w) == pytest.approx(90.0)
        assert vr.weighted_mean({}, w) is None


# ── catchment restriction (item 4a) ──────────────────────────────────────────
class TestCatchmentGeometry:
    SQUARE = [[(-121.0, 46.0), (-120.0, 46.0), (-120.0, 47.0),
               (-121.0, 47.0), (-121.0, 46.0)]]

    def test_point_inside_and_outside(self):
        assert vr._in_polygon(46.5, -120.5, self.SQUARE) is True
        assert vr._in_polygon(48.0, -120.5, self.SQUARE) is False
        assert vr._in_polygon(46.5, -119.0, self.SQUARE) is False

    def test_missing_coordinates_are_not_inside(self):
        assert vr._in_polygon(None, -120.5, self.SQUARE) is False
        assert vr._in_polygon(46.5, None, self.SQUARE) is False
        assert vr._in_polygon(46.5, -120.5, []) is False

    def test_rings_handles_polygon_and_multipolygon(self):
        poly = {"type": "Polygon", "coordinates": [self.SQUARE[0]]}
        multi = {"type": "MultiPolygon", "coordinates": [[self.SQUARE[0]]]}
        assert len(vr._rings(poly)) == 1
        assert len(vr._rings(multi)) == 1
        assert vr._rings(None) == []
        assert vr._rings({"type": "Point", "coordinates": [0, 0]}) == []


# ── runoff ratio (item 4b) ───────────────────────────────────────────────────
class TestRunoffRatioGuard:
    def test_an_impossible_observed_ratio_is_rejected_in_source(self):
        """yield > precipitation cannot happen over a closed catchment in a
        year; it means the P used is not the catchment's P. On the real Naches
        run the observed ratio came out 1.74 and must not read as a verdict."""
        src = (ROOT / "tools" / "validate_run.py").read_text()
        assert "observed_ratio_valid" in src
        assert "> 1.0" in src or "> 1:" in src

    def test_ratio_uses_shared_precipitation_and_says_so(self):
        src = (ROOT / "tools" / "validate_run.py").read_text()
        assert "tests \"\n                \"partitioning, not forcing" in src \
            or "partitioning, not forcing" in src
