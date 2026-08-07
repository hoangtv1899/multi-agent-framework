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

from elm_surface_generator import (ELMSurfaceGenerator,  # noqa: E402
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
        from elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
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
        from elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
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
        from elm_surface_generator import _NOMINAL_BULK_DENSITY_GCC as BD
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

    def test_the_write_a_profile_path_is_not_named_after_one_dataset(self, gen):
        """It was called 'ssurgo' because a survey query was its usual input.

        The capability is "write the supplied horizons into surfdata", which is
        what mcp/elm-mcp/scripts/make_soil_sweep.py needs to vary clay and sand at one site.
        Naming a general capability after one dataset made it look like that
        dataset's plumbing, and therefore deletable along with it.
        """
        import inspect
        sig = inspect.signature(gen.generate_from_mcp)
        assert sig.parameters["soil_source"].default == "profile"

    def test_unknown_soil_sources_are_refused(self, gen):
        src = (ROOT / "mcp" / "elm-mcp" / "src" / "elm_surface_generator.py").read_text()
        assert "if soil_source not in ('profile', 'conus')" in src

    def test_the_builder_always_keeps_donor_soil(self):
        """Warm start is required, so a CONUS-subset template is always present
        and there is no second branch for the two decisions to drift between."""
        src = (ROOT / "mcp" / "elm-mcp" / "src" / "elm_experiment_builder.py").read_text()
        assert "veg_source, soil_source = 'template', 'conus'" in src
        assert "else 'ssurgo'" not in src

    def test_the_cache_key_separates_the_two_soils(self):
        """Without this a column's swept-soil and donor-soil surfaces would
        share a filename, and the second run would silently reuse the first.

        'profile' keeps the empty tag its old name had, so every surface file
        cached under the previous naming stays valid.
        """
        src = (ROOT / "mcp" / "elm-mcp" / "src" / "elm_surface_generator.py").read_text()
        assert "soil_tag" in src and "_soil-" in src


# ─────────────────────────────────────────────────────────────────────────────
# The design figure must describe the run, not the gathering
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util  # noqa: E402


def _tool(name):
    _TOOL_DIRS = (ROOT / "mcp" / "elm-mcp" / "src", ROOT / "tools")
    path = next((d / f"{name}.py" for d in _TOOL_DIRS
                 if (d / f"{name}.py").is_file()), None)
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class TestWarmStartIsTheDefault:
    """A cold single-column year starts from ELM's generic state and spends
    itself relaxing: -0.18 mm/yr recharge cold against 309 warm on the SAME
    column. The subset costs ~2 s per column, so cold is now an opt-out."""

    def test_no_initialization_still_warm_starts(self):
        """Asserts the BEHAVIOUR, not the text of one file.

        This used to grep workflow.py for `get('mode') != 'cold'`, which
        passed for the right reason until the per-model config moved to
        core/backends.py — then it failed while the behaviour it names was
        still correct. A test that breaks when code moves rather than when it
        changes is a test of where the code lives.
        """
        sys.path.insert(0, str(ROOT / "src"))
        from core import backends
        base = {"brief": {}, "reception": {}, "strategy": {}, "mcp_clients": {}}

        assert backends.config_for("elm", base)["warm_start"]["source"] == "conus"
        assert backends.config_for("elm", base, initialization={})[
            "warm_start"]["source"] == "conus"
        assert "warm_start" not in backends.config_for(
            "elm", base, initialization={"mode": "cold"})

    def test_the_prompt_agrees_with_the_code(self):
        """If the prompt still said 'otherwise cold', reception would report a
        cold start while the manager warm-started — the run record would be
        wrong about what it did."""
        p = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
        assert "DEFAULT IS WARM" in p


class TestDonorSoilProfile:
    def test_shape_matches_what_the_figure_reads(self, tmp_path):
        """Same keys the geology MCP emits, so the figure needs no special
        case for which dataset it was handed."""
        import numpy as np
        import netCDF4
        f = tmp_path / "surfdata.nc"
        with netCDF4.Dataset(f, "w") as d:
            d.createDimension("nlevsoi", 4)
            d.createDimension("lsmlat", 1)
            d.createDimension("lsmlon", 1)
            for name, vals in (("PCT_SAND", [40.0, 41.0, 42.0, 43.0]),
                               ("PCT_CLAY", [17.0, 17.0, 18.0, 18.0]),
                               ("ORGANIC", [57.4, 57.4, 36.7, 27.3]),
                               ("PCT_GRVL", [13.9, 14.0, 14.1, 14.3])):
                v = d.createVariable(name, "f8", ("nlevsoi", "lsmlat", "lsmlon"))
                v[:] = np.array(vals).reshape(4, 1, 1)
        fs = _tool("make_finidat_subset")
        prof = fs.donor_soil_profile(str(f))
        assert prof["num_layers"] == 4
        L = prof["layers"][0]
        for k in ("component", "depth_top_cm", "depth_bot_cm", "sand_pct",
                  "clay_pct", "texture_class", "organic_kg_m3", "gravel_pct"):
            assert k in L
        assert L["sand_pct"] == 40.0 and L["organic_kg_m3"] == 57.4
        assert L["depth_top_cm"] == 0.0 and L["depth_bot_cm"] > 0

    def test_depths_are_monotonic_and_start_at_the_surface(self, tmp_path):
        import numpy as np, netCDF4
        f = tmp_path / "s.nc"
        with netCDF4.Dataset(f, "w") as d:
            d.createDimension("nlevsoi", 5)
            for name in ("PCT_SAND", "PCT_CLAY"):
                v = d.createVariable(name, "f8", ("nlevsoi",))
                v[:] = np.full(5, 30.0)
        fs = _tool("make_finidat_subset")
        L = fs.donor_soil_profile(str(f))["layers"]
        tops = [l["depth_top_cm"] for l in L]
        assert tops[0] == 0.0
        assert all(b > a for a, b in zip(tops, tops[1:]))


class TestTheSamplingFigureDrawsTheDonorSoil:
    """The panel shows the soil the RUN uses, which it can now guarantee.

    _refine_columns runs before the figure is drawn, so soil_profile holds the
    warm-start donor's own soil by then. Previously a second profile was also
    fetched at sampling time and the panel had to guess which one it was
    holding — its own comment named the hazard: "a panel captioned SSURGO would
    describe a profile the model never saw."
    """

    def test_it_reads_clay_and_organic_from_the_donor_profile(self):
        exp = _tool("expand_sampling")
        col = {"soil_profile": {"layers": [
            {"component": "CONUS 1km", "clay_pct": 21.0, "organic_kg_m3": 57.4},
            {"component": "CONUS 1km", "clay_pct": 24.0, "organic_kg_m3": 36.7}]}}
        assert exp._soil_cov(col) == (24.0, 57.4)

    def test_a_column_with_no_profile_reports_nothing_plottable(self):
        exp = _tool("expand_sampling")
        assert exp._soil_cov({}) == (None, None)

    def test_ksat_is_not_expected_of_the_donor(self):
        """CONUS 1 km carries no saturated conductivity — ELM derives it from
        sand and organic. The old helper's Ksat branch only ever served the
        survey profile that sampling no longer fetches."""
        src = (ROOT / "tools" / "expand_sampling.py").read_text()
        assert "ksat_ums" not in src

class TestGridDensityAdaptsToShape:
    """A grid is requested over the BOUNDING BOX and used inside the BASIN, so
    the yield depends on the watershed's shape: compact basins keep ~50% of the
    points, an elongated coastal strip keeps 18%. Central Coastal California
    returned 21 usable points for 4984 km2 — 4.2 per 1000 km2 against 20-37
    elsewhere — leaving farthest-point selection almost nothing to choose from.
    """

    def test_thresholds_are_defined_and_bounded(self):
        from core import data_gather as g
        assert g.MIN_IN_BASIN > 0
        assert g.MAX_GRID_N > g.GRID_N, "the retry must be able to ask for more"
        assert g.MAX_GRID_N <= 1000, "and must not be unbounded"

    def test_the_retry_is_scaled_by_the_fill_ratio(self):
        """Not a fixed bump: a basin keeping 18% needs ~5x, one keeping 45%
        needs ~2x, and asking 5x for the second wastes a large terrain call."""
        src = (ROOT / "src" / "core" / "data_gather.py").read_text()
        i = src.index("MIN_IN_BASIN")
        body = src[src.index("def gather_grid"):]
        assert "fill" in body and "/ fill" in body

    def test_it_only_fires_when_the_grid_is_thin(self):
        src = (ROOT / "src" / "core" / "data_gather.py").read_text()
        body = src[src.index("def gather_grid"):]
        assert "len(clipped) < MIN_IN_BASIN" in body

    def test_a_worse_retry_is_discarded(self):
        """If the bigger request somehow yields fewer in-basin points, keep the
        first result rather than degrading the design."""
        src = (ROOT / "src" / "core" / "data_gather.py").read_text()
        body = src[src.index("def gather_grid"):]
        assert "if len(clip2) > len(clipped)" in body
