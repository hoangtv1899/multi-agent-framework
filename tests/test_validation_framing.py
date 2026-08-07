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

    def test_imperial_is_still_converted_at_the_boundary(self):
        """The conversion must have a source: USGS/SNOTEL only offer imperial.

        The boundary MOVED when the observation tools were consolidated —
        drainage area and station elevation are now converted inside the MCP
        servers, so everything downstream is metric on arrival. This asserts the
        conversion still exists SOMEWHERE rather than having been dropped along
        with the code that used to do it, which is exactly how a unit bug gets
        reintroduced during a refactor.
        """
        gw = (ROOT / "mcp" / "usgs-water-mcp" / "groundwater_api.py").read_text()
        assert "MI2_TO_KM2" in gw and "drainage_area_km2" in gw
        assert 'FT_TO_M' in gw                       # well depths, feet -> metres

        sn = (ROOT / "mcp" / "snotel-mcp" / "main.py").read_text()
        assert 'elevation_ft' in sn and 'elevation_m' in sn
        assert "IN_TO_MM" in sn                      # SWE, inches -> mm


# ── how strong is the well comparison, really ───────────────────────────────
class TestTemporalNote:
    """The note used to assert a mismatch unconditionally. That was correct
    while the well query was capped at 10 sites and found nothing in-year; once
    the bulk query found 86 wells all measured in 1995, the fixed sentence
    contradicted the count printed next to it. Strength is a property of the
    data, so these pin that it is read off rather than assumed."""

    def test_all_in_year_is_year_matched(self):
        n = vr._temporal_note(71, 71, 1995, "1995", "1995")
        assert "YEAR-MATCHED" in n
        assert "MISMATCH" not in n and "CLIMATOLOGICAL" not in n

    def test_none_in_year_is_a_mismatch(self):
        n = vr._temporal_note(0, 14, 1995, "1951", "2001")
        assert "TEMPORAL MISMATCH" in n and "CLIMATOLOGICAL" in n
        assert "1951-2001" in n

    def test_some_in_year_says_partially(self):
        n = vr._temporal_note(3, 20, 1995, "1990", "2001")
        assert "PARTIALLY" in n and "3 of 20" in n

    def test_no_records_says_nothing_rather_than_guessing(self):
        assert vr._temporal_note(0, 0, 1995, None, None) == ""


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


# ── analysis figures: partitioning + controls ────────────────────────────────
ar = _load("ar_mod", "mcp/elm-mcp/scripts/analyze_run.py")


def _col(name, P, runoff, et, drain, ds, elev, clay=None, rech=0.0):
    return {"case_name": name, "status": "ok", "elevation_m": elev,
            "soil": {"clay_max_pct": clay},
            "metrics": {"precip_total_mm_yr": P, "precip_mm_yr": P,
                        "annual_runoff_mm_yr": runoff,
                        "annual_recharge_mm_yr": rech,
                        "water_budget": {"runoff_mm_yr": runoff, "et_mm_yr": et,
                                         "drainage_mm_yr": drain,
                                         "recharge_mm_yr": rech,
                                         "storage_change_mm": ds}}}


class TestPartitioningFigure:
    def test_storage_release_is_not_clamped_away(self):
        """The defect this replaces: the old plot_budget did max(v, 0) on
        Δstorage, so a column DRAINING storage showed no storage term and its
        stack silently exceeded P (col_12: 3750 mm exported against 1870 mm)."""
        src = (ROOT / "mcp" / "elm-mcp" / "scripts" / "analyze_run.py").read_text()
        assert "def plot_partitioning(" in src
        assert "def plot_budget(" not in src
        body = src[src.index("def plot_partitioning("):src.index("def plot_controls(")]
        assert "max(b.get(key) or 0, 0)" not in body
        assert "released from storage" in body

    def test_it_renders_a_column_that_drains_storage(self, tmp_path):
        results = {"a": _col("col_a", 1000, 100, 300, 1800, -1200, 900),
                   "b": _col("col_b", 800, 80, 400, 200, 120, 1500)}
        out = tmp_path / "partitioning.png"
        assert ar.plot_partitioning(results, out) is True
        assert out.exists() and out.stat().st_size > 0

    def test_no_budget_terms_means_no_figure(self, tmp_path):
        assert ar.plot_partitioning({}, tmp_path / "x.png") is False


class TestControlsFigure:
    def test_it_replaces_the_absolute_flux_matrix(self):
        src = (ROOT / "mcp" / "elm-mcp" / "scripts" / "analyze_run.py").read_text()
        assert "def plot_controls(" in src
        assert "def plot_relations(" not in src
        assert "def plot_gradient(" not in src

    def test_it_renders_and_needs_at_least_three_columns(self, tmp_path):
        results = {f"c{i}": _col(f"col_{i}", 800 + 200 * i, 50 + i, 300, 200,
                                 10, 800 + 100 * i, clay=5 + 3 * i,
                                 rech=100 + 20 * i)
                   for i in range(4)}
        out = tmp_path / "controls.png"
        assert ar.plot_controls(results, out) is True
        assert out.exists()
        assert ar.plot_controls({k: results[k] for k in list(results)[:2]},
                                tmp_path / "y.png") is False

    def test_shared_forcing_bin_panel_survives_no_shared_bins(self, tmp_path):
        """Every column in its own precipitation bin — the attribution panel
        must say so rather than crash."""
        results = {f"c{i}": _col(f"col_{i}", 500 + 400 * i, 20, 300, 100, 5,
                                 700 + 200 * i, clay=10)
                   for i in range(3)}
        assert ar.plot_controls(results, tmp_path / "z.png") is True


# ─────────────────────────────────────────────────────────────────────────────
# A failed query is not a finding
# ─────────────────────────────────────────────────────────────────────────────
class TestFailureIsNotAbsence:
    """Every fetcher recorded out["error"] and nothing read it, so a 400 from
    USGS and a genuinely ungauged basin produced the same verdict. The 2020 run
    reported "no in-domain gauge had 2020 daily records" for a basin whose gauge
    has reported every year since 1979."""

    def test_a_failed_fetch_is_labelled_unavailable(self):
        status, text = vr._unavailable({"error": "400 Bad Request"}, "gauge", 2020)
        assert status == "unavailable"
        assert "FAILED" in text
        assert "not evidence of absence" in text

    def test_a_clean_empty_fetch_stays_context_only(self):
        status, text = vr._unavailable({"gauges": []}, "gauge", 2020)
        assert status == "context-only"
        assert "FAILED" not in text

    def test_none_is_treated_as_a_clean_empty(self):
        assert vr._unavailable(None, "gauge", 2020)[0] == "context-only"

    def test_the_verdicts_actually_consult_it(self):
        """It is only worth having if the targets use it."""
        src = (ROOT / "tools" / "validate_run.py").read_text()
        i = src.index('"variable": "streamflow (water yield)"')
        assert "_unavailable(" in src[i:i + 900]
        j = src.index('"variable": "water-table depth"')
        assert "_unavailable(" in src[j:j + 700], \
            "water-table status was hardcoded 'compared'"


class TestYieldMatchesWhatAGaugeMeasures:
    """A stream gauge integrates surface runoff plus baseflow. The annual yield
    summed runoff + QCHARGE — the soil-to-aquifer flux — while the daily
    hydrograph two panels away already used QOVER + QDRAI. The same run was
    compared against the gauge two different ways: 538.5 mm/yr one way, 874.6
    the other, against 964.2 observed."""

    def test_yield_uses_drainage_not_recharge(self):
        src = (ROOT / "tools" / "validate_run.py").read_text()
        i = src.index("yields = [")
        seg = src[i:i + 400]
        assert "drainage_mm_yr" in seg
        assert "annual_recharge_mm_yr" not in seg

    def test_the_ratio_uses_the_same_definition(self):
        """If the two drift apart, the ratio and the yield describe different
        quantities under the same heading."""
        src = (ROOT / "tools" / "validate_run.py").read_text()
        i = src.index("y_by_id = {")
        seg = src[i:i + 400]
        assert "drainage_mm_yr" in seg
        assert "annual_recharge_mm_yr" not in seg

    def test_it_agrees_with_the_hydrograph(self):
        src = (ROOT / "tools" / "validate_run.py").read_text()
        assert 'ds["QOVER"] + ds["QDRAI"]' in src
