# tests/test_planner_agent.py
"""
Unit tests for PlannerAgent.
LLM calls are mocked — tests verify orchestration logic only.
Structural correctness is covered by test_plan_validator.py.
"""
import json
import copy
import pytest
from unittest.mock import patch, MagicMock

from src.agents.planner_agent  import PlannerAgent
from src.core.plan_validator   import PlanValidator


# ─────────────────────────────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────

# Richland, WA brief from ReceptionAgent [1]
SAMPLE_BRIEF = {
    "user_request":     "How do water table depths affect saturation near Richland, WA?",
    "intent":           "design_and_run",
    "experiment_focus": "water table depth effects on saturation",
    "region": {
        "location": "Richland, WA",
        "lat":       46.2856,
        "lon":      -119.2844,
    },
    "climate": {
        "precip_mm_yr": 238.5,
        "recharge_m_s": 2.268e-09,   # [1]
        "temp_min_c":   7.8,
        "temp_max_c":   19.4,
        "period":       "2006-2025",
    },
    "soil_profile": {
        "num_layers": 4,
        "source":     "SSURGO",
        "layers": [
            {"depth_top_cm":  0, "depth_bot_cm":   8,
             "texture": "Loam",       "sand_pct": 65.7, "clay_pct": 7.0},
            {"depth_top_cm":  8, "depth_bot_cm":  33,
             "texture": "Loam",       "sand_pct": 65.7, "clay_pct": 7.0},
            {"depth_top_cm": 33, "depth_bot_cm":  71,
             "texture": "Loam",       "sand_pct": 47.6, "clay_pct": 7.0},
            {"depth_top_cm": 71, "depth_bot_cm": 152,
             "texture": "Sandy Loam", "sand_pct": 96.5, "clay_pct": 2.0},
        ]
    },
    "pflotran_materials": [
        {"layer": i+1, "porosity": "0.45d0",
         "perm_iso": "5.0d-13", "alpha": "1.5d-4",
         "m": "0.56d0", "lrs": "0.08d0"}
        for i in range(4)
    ]
}

# Valid plan matching demo output [1]
VALID_PLAN = {
    "DOMAIN_CONFIGURATION": {
        "grid_type": "structured",
        "nxyz": {"x": 1, "y": 1, "z": 9},       # 1×1×9 [1]
        "dxyz": {"x": "1.0d0", "y": "1.0d0",
                 "z": ["2.0d0", "1.0d0", "1.426d0",
                       "1.0d0", "1.0d0", "0.52d0",
                       "0.44d0", "0.48d0", "0.08d0"]}
    },
    "LAYER_REGIONS_AND_COORDINATES": [
        {"layer_name": "bedrock",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "0.0d0",  "z_max": "2.0d0"}},
        {"layer_name": "weathered_bedrock",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "2.0d0",  "z_max": "3.0d0"}},
        {"layer_name": "deep_layer3",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "3.0d0",  "z_max": "4.426d0"}},
        {"layer_name": "deep_layer2",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "4.426d0", "z_max": "5.426d0"}},
        {"layer_name": "deep_layer1",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "5.426d0", "z_max": "6.426d0"}},
        {"layer_name": "ssurgo_layer4",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "6.426d0", "z_max": "6.946d0"}},
        {"layer_name": "ssurgo_layer3",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "6.946d0", "z_max": "7.386d0"}},
        {"layer_name": "ssurgo_layer2",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "7.386d0", "z_max": "7.866d0"}},
        {"layer_name": "ssurgo_layer1",
         "coordinates": {"x_min": "0.0d0", "y_min": "0.0d0",
                         "x_max": "1.0d0", "y_max": "1.0d0",
                         "z_min": "7.866d0", "z_max": "7.946d0"}},
    ],
    "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES": [
        {"ID": i+1, "POROSITY": "0.40d0",
         "PERMEABILITY": {"PERM_ISO": "1.0d-12"},
         "CHARACTERISTIC_CURVE": {
             "SATURATION_FUNCTION":        "VAN_GENUCHTEN",
             "ALPHA":                      "1.0d-4",
             "M":                          "0.5d0",
             "LIQUID_RESIDUAL_SATURATION": "0.08d0"
         },
         "PERMEABILITY_FUNCTION": {
             "TYPE":                       "MUALEM_VG_LIQ",
             "M":                          "0.5d0",
             "LIQUID_RESIDUAL_SATURATION": "0.08d0"
         }}
        for i in range(9)
    ],
    "TIME": {
        "FINAL_TIME":                  "10.0d0",
        "FINAL_TIME_UNITS":            "y",
        "INITIAL_TIMESTEP_SIZE":       "1.0d0",
        "INITIAL_TIMESTEP_SIZE_UNITS": "h",
        "MAXIMUM_TIMESTEP_SIZE":       "0.1d0",
        "MAXIMUM_TIMESTEP_SIZE_UNITS": "y",
    },
    "OUTPUT": {
        "TIMES":      ["0.5d0", "1.0d0", "2.0d0", "5.0d0", "10.0d0"],
        "TIME_UNITS": "y",
        "FORMAT":     "TECPLOT POINT",
    },
    "FLOW_CONDITIONS": {
        "near_surface_initial": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 7.4d0",          # near-surface [1]
        },
        "shallow_initial": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 6.5d0",
        },
        "mid_depth_initial": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 5.0d0",           # critical [1]
        },
        "deep_initial": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 3.5d0",
        },
        "very_deep_initial": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 2.0d0",           # very deep [1]
        },
        "recharge_standard": {
            "TYPE": "FLUX",
            "FLUX_LIST": [
                {"TIME": "0.0d0",  "FLUX": "7.16"},
                {"TIME": "10.0d0", "FLUX": "7.16"},
            ],
            "TIME_UNITS": "y",
            "DATA_UNITS": "cm/y",
        },
        "bottom_boundary": {
            "TYPE":  "HYDROSTATIC",
            "DATUM": "0.0d0 0.0d0 0.5d0",
        }
    },
    "CONDITIONS_COUPLERS": [
        # 5 simulation cases [1]
        {"EXPERIMENT":                       "near_surface_water_table",
         "INITIAL_CONDITION":                "near_surface_initial",
         "BOUNDARY_CONDITION_SURFACE":       "recharge_standard",
         "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
        {"EXPERIMENT":                       "shallow_water_table",
         "INITIAL_CONDITION":                "shallow_initial",
         "BOUNDARY_CONDITION_SURFACE":       "recharge_standard",
         "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
        {"EXPERIMENT":                       "mid_depth_water_table",
         "INITIAL_CONDITION":                "mid_depth_initial",
         "BOUNDARY_CONDITION_SURFACE":       "recharge_standard",
         "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
        {"EXPERIMENT":                       "deep_water_table",
         "INITIAL_CONDITION":                "deep_initial",
         "BOUNDARY_CONDITION_SURFACE":       "recharge_standard",
         "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
        {"EXPERIMENT":                       "very_deep_water_table",
         "INITIAL_CONDITION":                "very_deep_initial",
         "BOUNDARY_CONDITION_SURFACE":       "recharge_standard",
         "BOUNDARY_CONDITION_DEEP_BOUNDARY": "bottom_boundary"},
    ],
    "parameter_space": {
        "primary_variable":  "initial water table depth",
        "sampling_strategy": "transition-focused",
        "range":             "2.0 to 7.4 m",
        "transition_zones":  ["2.0 m capillary transition"]   # [1]
    }
}

VALID_VALIDATION = json.dumps({
    "valid": True,
    "questions": {
        "datum_direction":     "pass",
        "parameter_coverage":  "pass",
        "transition_sampling": "pass",
        "experiment_count":    "pass",
        "scientific_names":    "pass",
    },
    "issues":     [],
    "suggestion": None
})

FAILED_VALIDATION = json.dumps({
    "valid": False,
    "questions": {
        "datum_direction":     "fail",
        "parameter_coverage":  "pass",
        "transition_sampling": "pass",
        "experiment_count":    "pass",
        "scientific_names":    "pass",
    },
    "issues": [
        {
            "experiment": "shallow_water_table",
            "question":   "datum_direction",
            "issue":      "datum_z=2.0 is too low for a shallow table",
            "severity":   "critical"
        }
    ],
    "suggestion": "Increase datum_z for shallow experiments"
})


# ─────────────────────────────────────────────────────────────────────
# FIXTURE
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """PlannerAgent with mocked prompt loading."""
    with patch("src.agents.planner_agent.load_prompt",
               return_value="mock system prompt"):
        return PlannerAgent(model="test-model")


# ─────────────────────────────────────────────────────────────────────
# TESTS: create_plan() orchestration
# ─────────────────────────────────────────────────────────────────────

class TestCreatePlan:

    def test_full_pipeline_returns_valid_plan(self, agent):
        """Full pipeline returns a complete plan dict."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),  # design
                              VALID_VALIDATION,         # scientific check
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        assert isinstance(plan, dict)
        assert "CONDITIONS_COUPLERS"   in plan
        assert "LAYER_REGIONS_AND_COORDINATES" in plan
        assert "FLOW_CONDITIONS"       in plan

    def test_returns_five_experiments(self, agent):
        """Planner produces 5 simulation cases [1]."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        assert len(plan["CONDITIONS_COUPLERS"]) == 5

    def test_returns_nine_layers(self, agent):
        """Plan has 9 layers for 4 SSURGO + 5 deep layers [1]."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        assert len(plan["LAYER_REGIONS_AND_COORDINATES"]) == 9

    def test_design_call_receives_brief(self, agent):
        """Design LLM call receives the full brief in the prompt."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]) as mock_ask:
            agent.create_plan(SAMPLE_BRIEF)

        first_call = mock_ask.call_args_list[0]
        prompt     = first_call[1]["user_message"]
        assert "Richland, WA"    in prompt
        assert "design_and_run"  in prompt

    def test_design_uses_planner_system_prompt(self, agent):
        """Design call uses planner_system prompt as system message."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]) as mock_ask:
            agent.create_plan(SAMPLE_BRIEF)

        first_call    = mock_ask.call_args_list[0]
        system_msg    = first_call[1]["system_message"]
        assert system_msg == agent.prompt_design

    def test_validator_called_on_plan(self, agent):
        """PlanValidator.check() is called with the designed plan."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]), \
             patch.object(agent.validator, "check",
                          return_value=(VALID_PLAN, [])) as mock_check:
            agent.create_plan(SAMPLE_BRIEF)

        mock_check.assert_called_once()

    def test_scientific_check_receives_conditions_only(self, agent):
        """Scientific validation receives conditions only, not full plan."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]) as mock_ask:
            agent.create_plan(SAMPLE_BRIEF)

        second_call = mock_ask.call_args_list[1]
        prompt      = second_call[1]["user_message"]

        # Should contain experiment conditions
        assert "experiments"     in prompt
        # Should NOT contain full material properties
        assert "MUALEM_VG_LIQ"  not in prompt

    def test_scientific_check_uses_validation_prompt(self, agent):
        """Scientific check uses planner_validation prompt."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]) as mock_ask:
            agent.create_plan(SAMPLE_BRIEF)

        second_call = mock_ask.call_args_list[1]
        system_msg  = second_call[1]["system_message"]
        assert system_msg == agent.prompt_validation


# ─────────────────────────────────────────────────────────────────────
# TESTS: Structural auto-fix integration
# ─────────────────────────────────────────────────────────────────────

class TestStructuralFix:

    def test_inverted_layers_auto_fixed(self, agent):
        """Planner auto-fixes inverted layer order before returning."""
        inverted = copy.deepcopy(VALID_PLAN)
        inverted["LAYER_REGIONS_AND_COORDINATES"] = list(
            reversed(inverted["LAYER_REGIONS_AND_COORDINATES"])
        )
        inverted["MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"] = list(
            reversed(inverted[
                "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"
            ])
        )

        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(inverted),
                              VALID_VALIDATION,
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        # Output should be sorted ascending
        validator = PlanValidator()
        z_mins    = [
            validator._parse_z(l["coordinates"]["z_min"])
            for l in plan["LAYER_REGIONS_AND_COORDINATES"]
        ]
        assert z_mins == sorted(z_mins)

    def test_bedrock_is_first_in_output(self, agent):
        """Output always has bedrock as first layer [1]."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        first = plan["LAYER_REGIONS_AND_COORDINATES"][0]
        assert first["layer_name"] == "bedrock"


# ─────────────────────────────────────────────────────────────────────
# TESTS: Scientific validation reporting
# ─────────────────────────────────────────────────────────────────────

class TestScientificValidation:

    def test_critical_issue_logged(self, agent, capsys):
        """Critical scientific issues are printed with ❌ icon."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              FAILED_VALIDATION,
                          ]):
            agent.create_plan(SAMPLE_BRIEF)

        captured = capsys.readouterr()
        assert "❌"                  in captured.out
        assert "datum_direction"     in captured.out

    def test_all_pass_shows_success(self, agent, capsys):
        """All passing validation shows success message."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              VALID_VALIDATION,
                          ]):
            agent.create_plan(SAMPLE_BRIEF)

        captured = capsys.readouterr()
        assert "All scientific checks passed" in captured.out

    def test_failed_validation_does_not_block(self, agent):
        """Scientific validation failure does not raise — it only warns."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              FAILED_VALIDATION,
                          ]):
            # Should not raise even with failed validation
            plan = agent.create_plan(SAMPLE_BRIEF)

        assert plan is not None

    def test_validation_parse_failure_returns_plan(self, agent):
        """If scientific validation LLM returns garbage, plan still returned."""
        with patch.object(agent, "ask_with_system",
                          side_effect=[
                              json.dumps(VALID_PLAN),
                              "not valid json",
                          ]):
            plan = agent.create_plan(SAMPLE_BRIEF)

        assert plan is not None


# ─────────────────────────────────────────────────────────────────────
# TESTS: Design failure handling
# ─────────────────────────────────────────────────────────────────────

class TestDesignFailure:

    def test_design_failure_raises_runtime_error(self, agent):
        """If design LLM returns invalid JSON, RuntimeError is raised."""
        with patch.object(agent, "ask_with_system",
                          return_value="not valid json"):
            with pytest.raises(RuntimeError,
                               match="Experiment design failed"):
                agent.create_plan(SAMPLE_BRIEF)

    def test_design_llm_exception_raises_runtime_error(self, agent):
        """If LLM call throws, RuntimeError is raised with context."""
        with patch.object(agent, "ask_with_system",
                          side_effect=Exception("API timeout")):
            with pytest.raises(RuntimeError,
                               match="Experiment design failed"):
                agent.create_plan(SAMPLE_BRIEF)
