# tests/test_plan_validator.py
"""
Unit tests for PlanValidator.
Pure Python — no LLM mocking, no MCP, no file I/O.
Tests each structural check with deliberately broken plans.
"""
import pytest
from src.core.plan_validator import PlanValidator


# ─────────────────────────────────────────────────────────────────────
# HELPERS — build test plans
# ─────────────────────────────────────────────────────────────────────

def make_valid_plan(n_soil_layers: int = 4) -> dict:
    """
    Build a structurally valid plan matching the demo [1]:
    5 deep foundation layers + n SSURGO soil layers.
    Total depth: 7.946 m (9 layers for n_soil=4) [1]
    """
    deep_layers = [
        {"name": "bedrock",           "z_min": 0.000, "z_max": 2.000},
        {"name": "weathered_bedrock", "z_min": 2.000, "z_max": 3.000},
        {"name": "deep_layer3",       "z_min": 3.000, "z_max": 4.426},
        {"name": "deep_layer2",       "z_min": 4.426, "z_max": 5.426},
        {"name": "deep_layer1",       "z_min": 5.426, "z_max": 6.426},
    ]

    # SSURGO layers stacked on top — Richland WA profile [1]
    soil_tops = [6.426, 6.946, 7.386, 7.866]
    soil_bots = [6.946, 7.386, 7.866, 7.946]
    soil_layers = [
        {"name": f"ssurgo_layer{i+1}",
         "z_min": soil_tops[i],
         "z_max": soil_bots[i]}
        for i in range(n_soil_layers)
    ]

    all_layers = deep_layers + soil_layers

    return {
        "LAYER_REGIONS_AND_COORDINATES": [
            {
                "layer_name": l["name"],
                "coordinates": {
                    "x_min": "0.0d0", "y_min": "0.0d0",
                    "x_max": "1.0d0", "y_max": "1.0d0",
                    "z_min": f"{l['z_min']}d0",
                    "z_max": f"{l['z_max']}d0",
                }
            }
            for l in all_layers
        ],
        "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES": [
            {"ID": i + 1, "POROSITY": "0.40d0",
             "PERMEABILITY": {"PERM_ISO": "1.0d-12"}}
            for i in range(len(all_layers))
        ],
        "FLOW_CONDITIONS": {
            "near_surface_initial": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 7.4d0",   # high z = near surface [1]
            },
            "shallow_initial": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 6.5d0",
            },
            "mid_depth_initial": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 5.0d0",   # critical transition [1]
            },
            "deep_initial": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 3.5d0",
            },
            "very_deep_initial": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 2.0d0",   # low z = deep underground [1]
            },
            "recharge_standard": {
                "TYPE": "FLUX",
                "FLUX_LIST": [
                    {"TIME": "0.0d0",  "FLUX": "7.16"},  # 2.268e-9 m/s → cm/y [1]
                    {"TIME": "10.0d0", "FLUX": "7.16"},
                ],
                "TIME_UNITS": "y",
                "DATA_UNITS": "cm/y",
            },
            "bottom_boundary": {
                "TYPE":  "HYDROSTATIC",
                "DATUM": "0.0d0 0.0d0 0.5d0",
            },
        },
        "CONDITIONS_COUPLERS": [
            # 5 simulation cases [1]
            {"EXPERIMENT": "near_surface_water_table",
             "INITIAL_CONDITION": "near_surface_initial",
             "BOUNDARY_CONDITION_SURFACE": "recharge_standard",
             "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
            {"EXPERIMENT": "shallow_water_table",
             "INITIAL_CONDITION": "shallow_initial",
             "BOUNDARY_CONDITION_SURFACE": "recharge_standard",
             "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
            {"EXPERIMENT": "mid_depth_water_table",
             "INITIAL_CONDITION": "mid_depth_initial",
             "BOUNDARY_CONDITION_SURFACE": "recharge_standard",
             "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
            {"EXPERIMENT": "deep_water_table",
             "INITIAL_CONDITION": "deep_initial",
             "BOUNDARY_CONDITION_SURFACE": "recharge_standard",
             "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
            {"EXPERIMENT": "very_deep_water_table",
             "INITIAL_CONDITION": "very_deep_initial",
             "BOUNDARY_CONDITION_SURFACE": "recharge_standard",
             "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
        ]
    }


def invert_layers(plan: dict) -> dict:
    """Reverse layer order to simulate LLM top-down mistake."""
    import copy
    p = copy.deepcopy(plan)
    p["LAYER_REGIONS_AND_COORDINATES"] = list(
        reversed(p["LAYER_REGIONS_AND_COORDINATES"])
    )
    p["MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"] = list(
        reversed(p["MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"])
    )
    return p


def add_gap(plan: dict, after_layer: int, gap: float = 0.1) -> dict:
    """Introduce a z gap after a specific layer index."""
    import copy
    p = copy.deepcopy(plan)
    layers = p["LAYER_REGIONS_AND_COORDINATES"]
    # Shift all layers above the gap upward
    for i in range(after_layer, len(layers)):
        coords = layers[i]["coordinates"]
        z_min  = float(coords["z_min"].replace("d", "e")) + gap
        z_max  = float(coords["z_max"].replace("d", "e")) + gap
        coords["z_min"] = f"{z_min}d0"
        coords["z_max"] = f"{z_max}d0"
    return p


# ─────────────────────────────────────────────────────────────────────
# FIXTURE
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def validator():
    return PlanValidator()


@pytest.fixture
def valid_plan():
    return make_valid_plan()


# ─────────────────────────────────────────────────────────────────────
# TESTS: Valid plan passes all checks
# ─────────────────────────────────────────────────────────────────────

class TestValidPlan:

    def test_valid_plan_passes_all_checks(self, validator, valid_plan):
        """A correctly structured plan passes without issues."""
        _, issues = validator.check(valid_plan)
        assert issues == ["✓ All structural checks passed"]

    def test_valid_plan_unchanged(self, validator, valid_plan):
        """A valid plan is not modified by the validator."""
        import copy
        original = copy.deepcopy(valid_plan)
        fixed, _ = validator.check(valid_plan)
        assert (fixed["LAYER_REGIONS_AND_COORDINATES"] ==
                original["LAYER_REGIONS_AND_COORDINATES"])

    def test_five_experiments_preserved(self, validator, valid_plan):
        """All 5 simulation cases survive validation [1]."""
        fixed, _ = validator.check(valid_plan)
        assert len(fixed["CONDITIONS_COUPLERS"]) == 5


# ─────────────────────────────────────────────────────────────────────
# TESTS: Check 1 — Layer ordering
# ─────────────────────────────────────────────────────────────────────

class TestLayerOrdering:

    def test_detects_inverted_layers(self, validator, valid_plan):
        """Inverted layer order is detected and flagged."""
        bad_plan = invert_layers(valid_plan)
        _, issues = validator.check(bad_plan)
        assert any("inverted" in i or "auto-fixed" in i
                   for i in issues)

    def test_auto_fixes_inverted_layers(self, validator, valid_plan):
        """Inverted layers are automatically sorted ascending."""
        bad_plan  = invert_layers(valid_plan)
        fixed, _  = validator.check(bad_plan)
        layers    = fixed["LAYER_REGIONS_AND_COORDINATES"]
        z_mins    = [
            validator._parse_z(l["coordinates"]["z_min"])
            for l in layers
        ]
        assert z_mins == sorted(z_mins)

    def test_bedrock_is_first_after_fix(self, validator, valid_plan):
        """After fixing, bedrock (lowest z) is layer 1 [1]."""
        bad_plan = invert_layers(valid_plan)
        fixed, _ = validator.check(bad_plan)
        first_layer = fixed["LAYER_REGIONS_AND_COORDINATES"][0]
        assert first_layer["layer_name"] == "bedrock"

    def test_surface_soil_is_last_after_fix(self, validator, valid_plan):
        """After fixing, surface soil (highest z) is last layer [1]."""
        bad_plan = invert_layers(valid_plan)
        fixed, _ = validator.check(bad_plan)
        last_layer = fixed["LAYER_REGIONS_AND_COORDINATES"][-1]
        assert "ssurgo_layer1" in last_layer["layer_name"]

    def test_material_ids_reassigned_after_fix(self, validator,
                                               valid_plan):
        """Material IDs are reassigned 1..N after layer sort."""
        bad_plan = invert_layers(valid_plan)
        fixed, _ = validator.check(bad_plan)
        mats     = fixed[
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"
        ]
        ids = [m["ID"] for m in mats]
        assert ids == list(range(1, len(mats) + 1))

    def test_already_sorted_plan_not_modified(self, validator,
                                              valid_plan):
        """A correctly ordered plan is not touched."""
        import copy
        original_layers = copy.deepcopy(
            valid_plan["LAYER_REGIONS_AND_COORDINATES"]
        )
        fixed, issues = validator.check(valid_plan)
        assert (fixed["LAYER_REGIONS_AND_COORDINATES"] ==
                original_layers)
        assert not any("auto-fixed" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────
# TESTS: Check 2 — Z continuity
# ─────────────────────────────────────────────────────────────────────

class TestZContinuity:

    def test_detects_gap_between_layers(self, validator, valid_plan):
        """A gap between layers is detected."""
        bad_plan = add_gap(valid_plan, after_layer=2, gap=0.1)
        _, issues = validator.check(bad_plan)
        assert any("Gap" in i for i in issues)

    def test_gap_cites_correct_layer_numbers(self, validator,
                                             valid_plan):
        """Gap error message identifies the correct layer pair."""
        bad_plan = add_gap(valid_plan, after_layer=2, gap=0.1)
        _, issues = validator.check(bad_plan)
        gap_issues = [i for i in issues if "Gap" in i]
        assert any("layer 3" in i for i in gap_issues)

    def test_no_false_positive_on_valid_plan(self, validator,
                                             valid_plan):
        """No gap errors on a correctly continuous plan."""
        _, issues = validator.check(valid_plan)
        assert not any("Gap" in i for i in issues)

    def test_small_float_tolerance(self, validator, valid_plan):
        """Gaps smaller than 1e-4 m are not flagged (float precision)."""
        import copy
        p = copy.deepcopy(valid_plan)
        # Introduce a tiny floating point difference
        coords = p["LAYER_REGIONS_AND_COORDINATES"][1]["coordinates"]
        coords["z_min"] = "2.000001d0"   # 0.000001 m gap — below tolerance
        _, issues = validator.check(p)
        assert not any("Gap" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────
# TESTS: Check 3 — Material count
# ─────────────────────────────────────────────────────────────────────

class TestMaterialCount:

    def test_detects_material_mismatch(self, validator, valid_plan):
        """Mismatch between layer count and material count is flagged."""
        import copy
        bad_plan = copy.deepcopy(valid_plan)
        # Remove one material
        bad_plan[
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"
        ].pop()
        _, issues = validator.check(bad_plan)
        assert any("material count" in i for i in issues)

    def test_correct_counts_pass(self, validator, valid_plan):
        """Equal layer and material counts pass without issue."""
        layers = valid_plan["LAYER_REGIONS_AND_COORDINATES"]
        mats   = valid_plan[
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"
        ]
        assert len(layers) == len(mats)
        _, issues = validator.check(valid_plan)
        assert not any("material count" in i for i in issues)

    @pytest.mark.parametrize("n_soil", [3, 4, 5])
    def test_various_soil_layer_counts(self, validator, n_soil):
        """Material count check works for different soil layer counts."""
        plan = make_valid_plan(n_soil_layers=n_soil)
        _, issues = validator.check(plan)
        assert not any("material count" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────
# TESTS: Check 4 — Datum bounds
# ─────────────────────────────────────────────────────────────────────

class TestDatumBounds:

    def test_datum_above_domain_flagged(self, validator, valid_plan):
        """Datum above domain total depth is flagged."""
        import copy
        bad_plan = copy.deepcopy(valid_plan)
        bad_plan["FLOW_CONDITIONS"]["near_surface_initial"][
            "DATUM"
        ] = "0.0d0 0.0d0 9.0d0"   # above 7.946 m total depth [1]
        _, issues = validator.check(bad_plan)
        assert any("out of bounds" in i for i in issues)

    def test_datum_below_zero_flagged(self, validator, valid_plan):
        """Datum below 0 is flagged."""
        import copy
        bad_plan = copy.deepcopy(valid_plan)
        bad_plan["FLOW_CONDITIONS"]["deep_initial"][
            "DATUM"
        ] = "0.0d0 0.0d0 -0.5d0"
        _, issues = validator.check(bad_plan)
        assert any("out of bounds" in i for i in issues)

    def test_valid_datums_pass(self, validator, valid_plan):
        """All 5 demo datums are within bounds [1]."""
        _, issues = validator.check(valid_plan)
        assert not any("out of bounds" in i for i in issues)

    @pytest.mark.parametrize("datum_z,expected_pass", [
        (7.4, True),    # near-surface [1]
        (6.5, True),    # shallow [1]
        (5.0, True),    # mid-depth, critical transition [1]
        (3.5, True),    # deep [1]
        (2.0, True),    # very deep [1]
        (0.0, False),   # at boundary — invalid
        (7.946, False), # at top boundary — invalid
        (8.5, False),   # above domain
        (-1.0, False),  # below domain
    ])
    def test_datum_boundary_values(self, validator, valid_plan,
                                   datum_z, expected_pass):
        """Parametric test for datum boundary conditions."""
        import copy
        p = copy.deepcopy(valid_plan)
        p["FLOW_CONDITIONS"]["mid_depth_initial"][
            "DATUM"
        ] = f"0.0d0 0.0d0 {datum_z}d0"

        _, issues = validator.check(p)
        has_bounds_error = any("out of bounds" in i for i in issues)

        if expected_pass:
            assert not has_bounds_error
        else:
            assert has_bounds_error


# ─────────────────────────────────────────────────────────────────────
# TESTS: Check 5 — Recharge positive
# ─────────────────────────────────────────────────────────────────────

class TestRechargePositive:

    def test_negative_recharge_flagged(self, validator, valid_plan):
        """Negative recharge flux is flagged."""
        import copy
        bad_plan = copy.deepcopy(valid_plan)
        bad_plan["FLOW_CONDITIONS"]["recharge_standard"][
            "FLUX_LIST"
        ] = [
            {"TIME": "0.0d0",  "FLUX": "-7.16"},
            {"TIME": "10.0d0", "FLUX": "-7.16"},
        ]
        _, issues = validator.check(bad_plan)
        assert any("Negative recharge" in i for i in issues)

    def test_positive_recharge_passes(self, validator, valid_plan):
        """Positive recharge flux (2.268e-9 m/s → 7.16 cm/y) passes [1]."""
        _, issues = validator.check(valid_plan)
        assert not any("Negative recharge" in i for i in issues)

    def test_zero_recharge_passes(self, validator, valid_plan):
        """Zero recharge is technically valid (no rainfall scenario)."""
        import copy
        p = copy.deepcopy(valid_plan)
        p["FLOW_CONDITIONS"]["recharge_standard"]["FLUX_LIST"] = [
            {"TIME": "0.0d0",  "FLUX": "0.0"},
            {"TIME": "10.0d0", "FLUX": "0.0"},
        ]
        _, issues = validator.check(p)
        assert not any("Negative recharge" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────
# TESTS: Utilities
# ─────────────────────────────────────────────────────────────────────

class TestUtilities:

    @pytest.mark.parametrize("val,expected", [
        ("2.0d0",    2.0),
        ("1.5d-12",  1.5e-12),
        ("0.45d0",   0.45),
        ("7.946d0",  7.946),
        ("101325.0d0", 101325.0),
        (2.0,        2.0),
        ("bad",      0.0),
        (None,       0.0),
    ])
    def test_parse_z_fortran_notation(self, validator, val, expected):
        """_parse_z correctly handles Fortran and standard notation."""
        assert abs(validator._parse_z(val) - expected) < 1e-10

    @pytest.mark.parametrize("datum_str,expected", [
        ("0.0d0 0.0d0 7.4d0",  7.4),   # near-surface [1]
        ("0.0d0 0.0d0 2.0d0",  2.0),   # very deep [1]
        ("0.0d0 0.0d0 0.5d0",  0.5),   # bottom boundary
        ("",                   None),
        ("bad string",         None),
    ])
    def test_extract_datum(self, validator, datum_str, expected):
        """_extract_datum correctly parses DATUM strings."""
        condition = {"DATUM": datum_str} if datum_str else {}
        result    = validator._extract_datum(condition)
        if expected is None:
            assert result is None
        else:
            assert abs(result - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────
# TESTS: Edge cases
# ─────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_plan_does_not_crash(self, validator):
        """Empty plan returns issues without crashing."""
        _, issues = validator.check({})
        assert isinstance(issues, list)

    def test_plan_with_no_couplers(self, validator, valid_plan):
        """Plan with no CONDITIONS_COUPLERS passes datum check."""
        import copy
        p = copy.deepcopy(valid_plan)
        p["CONDITIONS_COUPLERS"] = []
        _, issues = validator.check(p)
        assert not any("out of bounds" in i for i in issues)

    def test_single_layer_plan(self, validator):
        """Single layer plan skips ordering and continuity checks."""
        plan = {
            "LAYER_REGIONS_AND_COORDINATES": [
                {"layer_name": "bedrock",
                 "coordinates": {"z_min": "0.0d0", "z_max": "2.0d0",
                                 "x_min": "0.0d0", "x_max": "1.0d0",
                                 "y_min": "0.0d0", "y_max": "1.0d0"}}
            ],
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES": [
                {"ID": 1, "POROSITY": "0.05d0"}
            ],
            "FLOW_CONDITIONS":    {},
            "CONDITIONS_COUPLERS": []
        }
        _, issues = validator.check(plan)
        assert isinstance(issues, list)
