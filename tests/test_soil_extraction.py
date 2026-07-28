"""Offline tests for how SSURGO soil reaches ELM (no network, no MCP).

Every column printed seven lines of "MCP layer N missing ['organic','gravel']
— using loam fallback values". Organic matter was not missing: SSURGO reports
it and the geology server returned it as `organic_matter_pct`, while the
extractor looked only for `organic_pct`/`organic`. So a real measurement was
discarded and a constant substituted, silently, for every layer of every
column. Gravel genuinely was not being fetched, and now is.

The stakes are not cosmetic: at one Naches column the real profile runs
organic 5.0 -> 0.5 % and gravel 10 -> 60 % with depth, against the loam
constants 3.0 % and 2.0 %. A 60 %-gravel horizon modelled as 2 % has quite
different water retention and drainage.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.elm_surface_generator import (ELMSurfaceGenerator,  # noqa: E402
                                        _FALLBACK_LAYER, _PCT_KEYS)


@pytest.fixture
def gen():
    return ELMSurfaceGenerator.__new__(ELMSurfaceGenerator)


def _layer(**kw):
    base = {"depth_top_cm": 0.0, "depth_bot_cm": 18.0,
            "sand_pct": 80.9, "clay_pct": 2.0}
    base.update(kw)
    return base


class TestKeyAliases:
    def test_the_name_the_mcp_actually_emits_is_recognised(self, gen):
        v = gen._extract_pct(_layer(organic_matter_pct=5.0), "organic")
        assert v == 5.0

    def test_the_short_form_still_works(self, gen):
        assert gen._extract_pct(_layer(organic_pct=2.5), "organic") == 2.5

    def test_gravel_is_recognised(self, gen):
        assert gen._extract_pct(_layer(gravel_pct=60.0), "gravel") == 60.0

    def test_genuinely_absent_returns_none_not_a_guess(self, gen):
        assert gen._extract_pct(_layer(), "organic") is None
        assert gen._extract_pct(_layer(), "gravel") is None

    def test_every_quantity_has_the_mcp_spelling_first_or_present(self):
        """A quantity missing from the table falls back to <what>_pct/<what>,
        which is exactly the bug this table exists to prevent."""
        for what in ("sand", "clay", "organic", "gravel"):
            assert what in _PCT_KEYS
            assert f"{what}_pct" in _PCT_KEYS[what]


class TestParsedLayers:
    def test_measured_values_reach_elm_unchanged(self, gen):
        prof = {"layers": [
            _layer(organic_matter_pct=5.0, gravel_pct=10.0),
            _layer(depth_top_cm=84.0, depth_bot_cm=152.0, sand_pct=47.1,
                   clay_pct=8.0, organic_matter_pct=0.5, gravel_pct=60.0)]}
        rows = gen._parse_mcp_layers(prof)
        assert [r["ORGANIC"] for r in rows] == [5.0, 0.5]
        assert [r["PCT_GRVL"] for r in rows] == [10.0, 60.0]
        # and they are NOT the constants that used to be substituted
        assert rows[1]["ORGANIC"] != _FALLBACK_LAYER["ORGANIC"]
        assert rows[1]["PCT_GRVL"] != _FALLBACK_LAYER["PCT_GRVL"]

    def test_fallback_is_per_field_not_per_layer(self, gen):
        """A layer missing gravel must keep its measured sand/clay/organic."""
        rows = gen._parse_mcp_layers({"layers": [
            _layer(organic_matter_pct=4.0)]})
        assert rows[0]["PCT_SAND"] == 80.9
        assert rows[0]["ORGANIC"] == 4.0
        assert rows[0]["PCT_GRVL"] == _FALLBACK_LAYER["PCT_GRVL"]

    def test_missing_fields_are_summarised_once_not_per_layer(self, gen, caplog):
        """Eight layers x fourteen columns printed ~100 identical lines, which
        buries anything that matters."""
        import logging
        caplog.set_level(logging.WARNING)
        gen._parse_mcp_layers({"layers": [_layer() for _ in range(8)]})
        soil_warnings = [r for r in caplog.records if "soil:" in r.getMessage()]
        assert len(soil_warnings) == 1
        assert "8 of 8 layers" in soil_warnings[0].getMessage()

    def test_a_complete_profile_warns_about_nothing(self, gen, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        gen._parse_mcp_layers({"layers": [
            _layer(organic_matter_pct=5.0, gravel_pct=10.0)]})
        assert [r for r in caplog.records if "soil:" in r.getMessage()] == []
