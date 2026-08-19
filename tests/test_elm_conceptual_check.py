#!/usr/bin/env python3
"""The ELM server's own design check owns ELM's reasons.

The framework's strategy gate used to refuse a sweep with no coordinates,
explaining that ELM reads its weather from a grid cell — a fact about one
model, judged by the framework, that stopped a PFLOTRAN sweep with no place by
design (2026-08-18). The rule lives in the ELM server's check now, and the
gate says nothing about coordinates.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import conceptual as elm_conceptual            # noqa: E402
from core import strategy_check                # noqa: E402


def _design(**held):
    return {"factors": [{"name": "soil_texture", "levels": [5, 55]}],
            "held_fixed": {"years": [1995, 1995], **held}}


def test_the_elm_check_refuses_a_sweep_with_no_point():
    v = elm_conceptual.check(_design())
    assert v["buildable"] is False
    why = " | ".join(w["why"] for w in v["wont_build"])
    assert "names no coordinates" in why and "reads its weather from a grid cell" in why


def test_written_weather_changes_the_reason_not_the_verdict():
    v = elm_conceptual.check(_design(weather={"fill": "uniform", "values": {"PRECTmms": 1e-4}}))
    assert v["buildable"] is False
    why = " | ".join(w["why"] for w in v["wont_build"])
    assert "bookkeeping" in why and "reads its weather" not in why


def test_coordinates_or_a_site_factor_satisfy_it():
    assert elm_conceptual.check(_design(lat=47.1, lon=-121.4))["buildable"] is True
    d = {"factors": [{"name": "forcing_site", "levels": [[47.1, -121.4], [46.0, -119.0]]}],
         "held_fixed": {"years": [1995, 1995]}}
    assert not any("names no coordinates" in w["why"]
                   for w in elm_conceptual.check(d)["wont_build"])


def test_the_gate_says_nothing_about_coordinates():
    """A sweep with no lat/lon is the gate's business only as arithmetic."""
    stops = strategy_check._sweep_stops(
        {"approach": "factor_sweep",
         "factors": [{"name": "water_table_m", "levels": [2, 10]}],
         "held_fixed": {"soil": "loam", "rain": {"fill": "seasonal", "mm_yr": 500}}})
    assert stops == []
    stops = strategy_check._sweep_stops(
        {"approach": "factor_sweep", "factors": [{"name": "x", "levels": [1]}]})
    assert len(stops) == 1 and "at least 2" in stops[0]
