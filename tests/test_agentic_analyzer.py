"""Offline tests for the agentic Analyzer (no LLM, no network, no rendering).

The Analyzer is allowed to be flexible where the Planner is not, because its
errors are cheap and visible rather than expensive and silent. What it is NOT
allowed to do is invent numbers, hide where a figure came from, or argue past a
validation refusal. These tests pin those three, plus the capability detection
that stops it requesting a figure the data cannot support.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.figure_registry import (CAPABILITIES, REGISTRY, available,  # noqa: E402
                                  catalogue, detect_capabilities)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load("ag_mod", "tools/analyze_agentic.py")


# ── the registry is the menu; nothing outside it exists ─────────────────────
class TestRegistry:
    def test_every_entry_is_complete(self):
        for name, e in REGISTRY.items():
            assert e["tool"] in ("analyze_run", "validate_run"), name
            assert e["fn"].startswith("plot_"), name
            assert isinstance(e["args"], tuple) and e["args"], name
            assert e["question"], name

    def test_every_declared_need_is_a_known_capability(self):
        """A typo'd requirement would make a figure permanently unavailable and
        silently so — the analyzer would just never see it in the menu."""
        for name, e in REGISTRY.items():
            for need in e["needs"]:
                assert need in CAPABILITIES, f"{name} needs unknown '{need}'"

    def test_availability_is_gated_by_capability(self):
        assert "spatial" in available(["water_budget", "coordinates"])
        assert "spatial" not in available(["water_budget"])   # no coordinates
        assert "validation_swe" not in available(["observations"])  # no swe_obs
        assert available([]) == [name for name, e in REGISTRY.items()
                                 if not e["needs"]]

    def test_catalogue_never_offers_what_cannot_render(self):
        caps = ["water_budget"]
        menu = catalogue(caps)
        for name in REGISTRY:
            if name in available(caps):
                assert name in menu
            else:
                assert f"  {name}:" not in menu

    def test_empty_run_says_so_rather_than_offering_nothing(self):
        assert "none" in catalogue([]).lower() or catalogue([]).strip()


# ── capabilities are DETECTED from artifacts, never assumed ─────────────────
class TestCapabilityDetection:
    def _results(self, **kw):
        r = {"case_name": "col_01", "status": "ok",
             "metrics": {"precip_mm_yr": 900, "water_budget": {}}}
        r.update(kw)
        return {"col_01": r}

    def test_no_artifacts_means_no_capabilities(self):
        assert detect_capabilities({}, {}, {}) == []

    def test_coordinates_gate_the_spatial_figure(self):
        without = detect_capabilities(self._results(), {}, {})
        withxy = detect_capabilities(self._results(lat=46.7, lon=-121.0), {}, {})
        assert "coordinates" not in without
        assert "coordinates" in withxy
        assert "spatial" not in available(without)

    def test_water_budget_needs_actual_terms(self):
        empty = self._results()
        empty["col_01"]["metrics"]["water_budget"] = {}
        assert "water_budget" not in detect_capabilities(empty, {}, {})
        full = self._results()
        full["col_01"]["metrics"]["water_budget"] = {"et_mm_yr": 300}
        assert "water_budget" in detect_capabilities(full, {}, {})

    def test_hydrograph_needs_a_daily_series_not_just_a_gauge(self):
        val = {"hydrograph": {"gauge": "somewhere", "NSE": -1}}
        assert "hydrograph" not in detect_capabilities({}, val, {})
        val["hydrograph"]["obs_mm_day"] = {"1995-01-01": 1.0}
        assert "hydrograph" in detect_capabilities({}, val, {})


# ── the evidence payload is the ONLY thing the interpreter may use ──────────
class TestEvidencePayload:
    HS = {"experiments": [
        {"case_name": "col_01", "status": "ok", "elevation_m": 750,
         "metrics": {"precip_mm_yr": 472.1, "annual_runoff_mm_yr": 2.3,
                     "annual_recharge_mm_yr": 10.2,
                     "water_budget": {"et_mm_yr": 289.6}}},
        {"case_name": "col_bad", "status": "failed", "metrics": {}}]}
    VAL = {"targets": [{"variable": "streamflow (water yield)",
                        "status": "context-only", "result": "r", "note": "n"}],
           "domain_match": {"gauge_fraction_of_basin": 0.071},
           "runoff_ratio": {"observed_ratio_valid": False}}

    def test_failed_columns_are_excluded(self):
        p = ag.evidence_payload(self.HS, self.VAL, {}, {}, [])
        assert [c["column"] for c in p["columns"]] == ["col_01"]

    def test_verdicts_are_carried_verbatim(self):
        """If the status did not travel, the interpreter could not be bound
        by it — this is the mechanism that makes 'context-only' stick."""
        p = ag.evidence_payload(self.HS, self.VAL, {}, {}, [])
        assert p["validation_verdicts"][0]["status"] == "context-only"
        assert p["domain_match"]["gauge_fraction_of_basin"] == 0.071
        assert p["runoff_ratio"]["observed_ratio_valid"] is False

    def test_raw_timeseries_are_not_shipped(self):
        """Sending day-by-day arrays would blow the context and invites the
        model to compute its own statistics instead of using the JSON."""
        val = dict(self.VAL, hydrograph={"NSE": -0.32, "obs_mm_day": {"d": 1},
                                         "mod_mm_day": {"d": 2}, "days": [1],
                                         "obs": [1], "mod": [2]})
        p = ag.evidence_payload(self.HS, val, {}, {}, [])
        hm = p["hydrograph_metrics"]
        assert hm["NSE"] == -0.32
        for k in ("obs_mm_day", "mod_mm_day", "days", "obs", "mod"):
            assert k not in hm


# ── grounding: catch inventions, do not cry wolf ───────────────────────────
class TestGrounding:
    PAYLOAD = {"catchment": {"gauge_id": "USGS-12488500"},
               "columns": [{"recharge_mm_yr": -55.8, "precip_mm_yr": 822.4},
                           {"recharge_mm_yr": 1315.4}]}

    def test_an_invented_number_is_flagged(self):
        assert "9999.9" in ag.check_grounding("recharge was 9999.9 mm/yr",
                                              self.PAYLOAD)

    def test_a_unicode_minus_is_not_an_invention(self):
        """The model writes −55.8; the JSON holds -55.8. Flagging that would
        make the checker noise, and a noisy checker gets ignored."""
        assert ag.check_grounding("recharge fell to −55.8 mm/yr",
                                  self.PAYLOAD) == []

    def test_a_bare_identifier_is_not_an_invention(self):
        assert ag.check_grounding("gauge 12488500 drains it", self.PAYLOAD) == []

    def test_rounding_is_allowed(self):
        assert ag.check_grounding("about 822 mm/yr of precipitation",
                                  self.PAYLOAD) == []

    def test_years_and_small_counts_pass(self):
        assert ag.check_grounding("in 1995 across 13 columns", self.PAYLOAD) == []

    def test_empty_interpretation_is_clean(self):
        assert ag.check_grounding("", self.PAYLOAD) == []
        assert ag.check_grounding(None, self.PAYLOAD) == []


# ── provenance ─────────────────────────────────────────────────────────────
class TestProvenance:
    def test_a_failed_render_is_recorded_not_swallowed(self, tmp_path):
        path, prov = ag.render("partitioning", {"results": {}}, tmp_path)
        assert path is None
        assert prov["figure"] == "partitioning"
        assert prov["provenance"] == "registry"
        assert prov["rendered"] is False

    def test_provenance_names_the_actual_renderer(self, tmp_path):
        _, prov = ag.render("spatial", {"results": {}, "run_dir": tmp_path},
                            tmp_path)
        assert prov["renderer"] == "analyze_run.plot_spatial"
