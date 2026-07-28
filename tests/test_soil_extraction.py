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
    def test_measured_values_reach_elm(self, gen):
        """Percentages stay percentages (sand/clay/gravel); organic becomes a
        density, because that is the unit ELM reads it in."""
        from core.elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
        prof = {"layers": [
            _layer(organic_matter_pct=5.0, gravel_pct=10.0),
            _layer(depth_top_cm=84.0, depth_bot_cm=152.0, sand_pct=47.1,
                   clay_pct=8.0, organic_matter_pct=0.5, gravel_pct=60.0)]}
        rows = gen._parse_mcp_layers(prof)
        assert [r["PCT_GRVL"] for r in rows] == [10.0, 60.0]
        assert rows[0]["ORGANIC"] == pytest.approx(5.0 / 100 * BD * 1000)
        assert rows[1]["ORGANIC"] == pytest.approx(0.5 / 100 * BD * 1000)
        # and neither is the constant that used to be substituted
        assert rows[1]["ORGANIC"] != _FALLBACK_LAYER["ORGANIC"]
        assert rows[1]["PCT_GRVL"] != _FALLBACK_LAYER["PCT_GRVL"]

    def test_fallback_is_per_field_not_per_layer(self, gen):
        """A layer missing gravel must keep its measured sand/clay/organic."""
        from core.elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
        rows = gen._parse_mcp_layers({"layers": [
            _layer(organic_matter_pct=4.0)]})
        assert rows[0]["PCT_SAND"] == 80.9
        assert rows[0]["ORGANIC"] == pytest.approx(4.0 / 100 * BD * 1000)
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


# ─────────────────────────────────────────────────────────────────────────────
# ORGANIC is a DENSITY, not a percentage
# ─────────────────────────────────────────────────────────────────────────────
class TestOrganicUnits:
    """ELM divides this field by organic_max = 130 kg/m3 to get om_frac, which
    sets porosity, conductivity and retention (SoilStateType.F90). SSURGO
    reports a percent, and it was written straight in — so a soil the CONUS
    donor describes as 57 kg/m3 (om_frac 0.44) reached ELM as 3.0 (om_frac
    0.023): organic soil told to behave as mineral, in the field that decides
    how much water it holds."""

    def test_percent_is_converted_to_density(self, gen):
        v = gen._organic_kg_m3({"organic_matter_pct": 3.0,
                                "bulk_density_gcc": 1.2})
        assert v == pytest.approx(36.0)          # 3 % x 1.2 g/cc -> 36 kg/m3

    def test_the_result_lands_in_elms_range_not_a_percentage(self, gen):
        """Sanity on the whole point: organic_max is 130, so a real soil sits
        in the tens. A value under ~10 means percentages leaked through."""
        v = gen._organic_kg_m3({"organic_matter_pct": 5.0,
                                "bulk_density_gcc": 1.2})
        assert 10.0 < v < 130.0

    def test_missing_bulk_density_uses_a_nominal_one(self, gen):
        from core.elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
        v = gen._organic_kg_m3({"organic_matter_pct": 2.0})
        assert v == pytest.approx(2.0 / 100 * BD * 1000)

    def test_absent_organic_stays_none(self, gen):
        assert gen._organic_kg_m3({"bulk_density_gcc": 1.2}) is None

    def test_the_fallback_constant_is_also_a_density(self):
        """It was 3.0 — the same percentage mistake, in the value used whenever
        SSURGO reports nothing."""
        assert _FALLBACK_LAYER["ORGANIC"] > 10.0

    def test_parsed_layers_carry_density_not_percent(self, gen):
        rows = gen._parse_mcp_layers({"layers": [
            _layer(organic_matter_pct=3.0, bulk_density_gcc=1.2,
                   gravel_pct=10.0)]})
        assert rows[0]["ORGANIC"] == pytest.approx(36.0)


class TestSoilSource:
    """A warm start hands ELM moisture equilibrated against the CONUS
    gridcell's soil. Overwriting that soil with SSURGO leaves the inherited
    water inconsistent with its own hydraulics: five of fourteen Naches columns
    drained more than their annual precipitation, one at 2.98x."""

    def test_both_values_are_accepted_and_others_refused(self, gen):
        import inspect
        sig = inspect.signature(gen.generate_from_mcp)
        assert sig.parameters["soil_source"].default == "ssurgo"

    def test_the_builder_keeps_donor_soil_when_warm_started(self):
        """Same condition as veg_source: a CONUS-subset template IS the warm
        start, so the two decisions cannot drift apart."""
        src = (ROOT / "src" / "core" / "elm_experiment_builder.py").read_text()
        assert "soil_source = 'conus' if surface_template else 'ssurgo'" in src
        assert "soil_source = soil_source" in src

    def test_the_cache_key_separates_the_two_soils(self):
        """Without this a column's ssurgo-soil and conus-soil surfaces would
        share a filename, and the second run would silently reuse the first."""
        src = (ROOT / "src" / "core" / "elm_surface_generator.py").read_text()
        assert "soil_tag" in src and "_soil-" in src
