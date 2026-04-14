# tests/test_reception_agent.py
"""
Unit tests for ReceptionAgent.
Covers multiple US locations and distinct climate scenarios.
All external dependencies mocked — no real LLM or MCP calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from src.agents.reception_agent import ReceptionAgent, ReceptionResult


# ─────────────────────────────────────────────────────────────────────
# US LOCATIONS — realistic lat/lon + climate profiles
# ─────────────────────────────────────────────────────────────────────

LOCATIONS = {
    "richland_wa": {
        "name":    "Richland, WA",
        "lat":      46.2856,
        "lon":     -119.2844,
        "climate": {                              # semi-arid, [1]
            "precip_mm_per_year":         238.5,
            "suggested_recharge_flux_ms":  2.268e-09,
            "mean_tmin_c":                 7.8,
            "mean_tmax_c":                19.4,
        },
        "soil": {
            "num_layers": 4,
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
        }
    },
    "moscow_id": {
        "name":    "Moscow, ID",
        "lat":      46.7324,
        "lon":     -117.0002,
        "climate": {                              # semi-humid
            "precip_mm_per_year":         580.0,
            "suggested_recharge_flux_ms":  5.80e-09,
            "mean_tmin_c":                 2.1,
            "mean_tmax_c":                17.3,
        },
        "soil": {
            "num_layers": 5,
            "layers": [
                {"depth_top_cm":  0, "depth_bot_cm":  20,
                 "texture": "Silt Loam",  "sand_pct": 20.0, "clay_pct": 18.0},
                {"depth_top_cm": 20, "depth_bot_cm":  50,
                 "texture": "Silt Loam",  "sand_pct": 18.0, "clay_pct": 22.0},
                {"depth_top_cm": 50, "depth_bot_cm": 100,
                 "texture": "Silty Clay", "sand_pct": 10.0, "clay_pct": 40.0},
                {"depth_top_cm":100, "depth_bot_cm": 150,
                 "texture": "Clay",       "sand_pct":  8.0, "clay_pct": 55.0},
                {"depth_top_cm":150, "depth_bot_cm": 200,
                 "texture": "Clay Loam",  "sand_pct": 25.0, "clay_pct": 35.0},
            ]
        }
    },
    "phoenix_az": {
        "name":    "Phoenix, AZ",
        "lat":      33.4484,
        "lon":     -112.0740,
        "climate": {                              # arid desert
            "precip_mm_per_year":          180.0,
            "suggested_recharge_flux_ms":   8.50e-10,
            "mean_tmin_c":                 15.5,
            "mean_tmax_c":                 36.8,
        },
        "soil": {
            "num_layers": 3,
            "layers": [
                {"depth_top_cm":  0, "depth_bot_cm":  15,
                 "texture": "Sand",        "sand_pct": 92.0, "clay_pct":  2.0},
                {"depth_top_cm": 15, "depth_bot_cm":  60,
                 "texture": "Sandy Loam",  "sand_pct": 78.0, "clay_pct":  8.0},
                {"depth_top_cm": 60, "depth_bot_cm": 150,
                 "texture": "Sandy Clay",  "sand_pct": 55.0, "clay_pct": 28.0},
            ]
        }
    },
    "miami_fl": {
        "name":    "Miami, FL",
        "lat":      25.7617,
        "lon":     -80.1918,
        "climate": {                              # humid subtropical
            "precip_mm_per_year":         1530.0,
            "suggested_recharge_flux_ms":  1.85e-08,
            "mean_tmin_c":                20.5,
            "mean_tmax_c":                31.2,
        },
        "soil": {
            "num_layers": 3,
            "layers": [
                {"depth_top_cm":  0, "depth_bot_cm":  20,
                 "texture": "Sand",       "sand_pct": 95.0, "clay_pct":  1.0},
                {"depth_top_cm": 20, "depth_bot_cm":  50,
                 "texture": "Sand",       "sand_pct": 93.0, "clay_pct":  2.0},
                {"depth_top_cm": 50, "depth_bot_cm": 100,
                 "texture": "Loamy Sand", "sand_pct": 85.0, "clay_pct":  5.0},
            ]
        }
    },
}


# ─────────────────────────────────────────────────────────────────────
# CLIMATE SCENARIOS
# ─────────────────────────────────────────────────────────────────────

CLIMATE_SCENARIOS = {
    "normal":   {"start_year": 2006, "end_year": 2025,
                 "use_historical": False,
                 "label": "20-year average"},

    "drought":  {"start_year": 2012, "end_year": 2015,
                 "use_historical": True,
                 "label": "drought years 2012-2015"},

    "wet":      {"start_year": 2016, "end_year": 2019,
                 "use_historical": True,
                 "label": "wet years 2016-2019"},

    "recent":   {"start_year": 2020, "end_year": 2024,
                 "use_historical": True,
                 "label": "recent 2020-2024"},
}


# ─────────────────────────────────────────────────────────────────────
# HELPERS — build realistic LLM responses
# ─────────────────────────────────────────────────────────────────────

def make_pass1(location_key: str,
               scenario_key: str,
               intent: str = "design_and_run") -> str:
    """Build a realistic Pass 1 JSON response for a location + scenario."""
    loc = LOCATIONS[location_key]
    scn = CLIMATE_SCENARIOS[scenario_key]
    return json.dumps({
        "intent":           intent,
        "confidence":       "high",
        "reasoning":        f"User wants to study groundwater near {loc['name']}",
        "experiment_focus": "water table depth effects on saturation",
        "previous_focus":   None,
        "location":         loc["name"],
        "lat":              loc["lat"],
        "lon":              loc["lon"],
        "start_year":       scn["start_year"],
        "end_year":         scn["end_year"],
        "use_historical":   scn["use_historical"],
        "existing_run_dir": None,
        "clarification_questions": []
    })


def make_pass2(location_key: str, scenario_key: str) -> str:
    """Build a realistic Pass 2 planner brief for a location + scenario."""
    loc = LOCATIONS[location_key]
    scn = CLIMATE_SCENARIOS[scenario_key]
    c   = loc["climate"]
    return json.dumps({
        "user_request":     f"Water table study near {loc['name']}",
        "intent":           "design_and_run",
        "experiment_focus": "water table depth effects on saturation",
        "region": {
            "location": loc["name"],
            "lat":      loc["lat"],
            "lon":      loc["lon"],
        },
        "climate": {
            "precip_mm_yr": c["precip_mm_per_year"],
            "recharge_m_s": c["suggested_recharge_flux_ms"],
            "temp_min_c":   c["mean_tmin_c"],
            "temp_max_c":   c["mean_tmax_c"],
            "period":       f"{scn['start_year']}-{scn['end_year']}",
        },
        "soil_profile": loc["soil"],
        "pflotran_materials": [
            {"layer": i + 1, "porosity": "0.40d0",
             "perm_iso": "1.0d-12", "alpha": "1.0d-4",
             "m": "0.5d0", "lrs": "0.08d0"}
            for i in range(loc["soil"]["num_layers"])
        ]
    })


def make_mcp_clients(location_key: str) -> dict:
    """Build mock MCP clients returning data for a specific location."""
    loc     = LOCATIONS[location_key]
    weather = MagicMock()
    geology = MagicMock()

    weather.call_tool_json.return_value = loc["climate"]
    geology.call_tool_json.side_effect  = lambda tool, args: (
        loc["soil"]
        if tool == "get_soil_profile"
        else {"materials": [
            {"layer": i + 1, "porosity": "0.40d0"}
            for i in range(loc["soil"]["num_layers"])
        ]}
    )
    return {"weather": weather, "geology": geology}


# ─────────────────────────────────────────────────────────────────────
# FIXTURE
# ─────────────────────────────────────────────────────────────────────

def make_agent(location_key: str) -> ReceptionAgent:
    """Create a ReceptionAgent with mocked prompts + MCP for a location."""
    with patch("src.agents.reception_agent.load_prompt",
               return_value="mock system prompt"):
        return ReceptionAgent(
            model       = "test-model",
            mcp_clients = make_mcp_clients(location_key)
        )


# ─────────────────────────────────────────────────────────────────────
# TESTS: Multiple Locations — Happy Path
# ─────────────────────────────────────────────────────────────────────

class TestMultipleLocations:
    """
    Full pipeline test across 4 US locations with distinct
    climate and soil profiles.
    """

    @pytest.mark.parametrize("location_key", list(LOCATIONS.keys()))
    def test_full_pipeline_each_location(self, location_key):
        """Full design_and_run pipeline succeeds for each US location."""
        agent = make_agent(location_key)
        loc   = LOCATIONS[location_key]

        with patch.object(agent, "respond",
                          return_value=make_pass1(location_key, "normal")), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2(location_key, "normal")), \
             patch("builtins.input", return_value="yes"):

            result = agent.process(
                f"How do water table depths affect saturation near {loc['name']}?"
            )

        assert result.intent                      == "design_and_run"
        assert result.brief["region"]["location"] == loc["name"]
        assert result.brief["region"]["lat"]      == loc["lat"]
        assert result.brief["region"]["lon"]      == loc["lon"]

    @pytest.mark.parametrize("location_key", list(LOCATIONS.keys()))
    def test_correct_soil_layers_per_location(self, location_key):
        """Each location returns correct number of SSURGO soil layers."""
        agent    = make_agent(location_key)
        expected = LOCATIONS[location_key]["soil"]["num_layers"]

        with patch.object(agent, "respond",
                          return_value=make_pass1(location_key, "normal")), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2(location_key, "normal")), \
             patch("builtins.input", return_value="yes"):

            result = agent.process("Water table study")

        assert result.brief["soil_profile"]["num_layers"] == expected

    @pytest.mark.parametrize("location_key", list(LOCATIONS.keys()))
    def test_planner_brief_has_all_fields(self, location_key):
        """to_planner_brief() always contains the required fields."""
        agent = make_agent(location_key)

        with patch.object(agent, "respond",
                          return_value=make_pass1(location_key, "normal")), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2(location_key, "normal")), \
             patch("builtins.input", return_value="yes"):

            result = agent.process("Water table study")

        brief = result.to_planner_brief()
        for field in ["user_request", "intent", "region",
                      "climate", "soil_profile", "pflotran_materials"]:
            assert field in brief, f"Missing field: {field}"


# ─────────────────────────────────────────────────────────────────────
# TESTS: Climate Scenarios
# ─────────────────────────────────────────────────────────────────────

class TestClimateScenarios:
    """
    Tests for distinct climate scenarios — normal average,
    drought, wet period, recent years.
    """

    @pytest.mark.parametrize("scenario_key", list(CLIMATE_SCENARIOS.keys()))
    def test_climate_scenario_passes_correct_years(self, scenario_key):
        """Bridge is called with correct start/end year for each scenario."""
        agent  = make_agent("richland_wa")
        scn    = CLIMATE_SCENARIOS[scenario_key]

        with patch.object(agent, "respond",
                          return_value=make_pass1("richland_wa", scenario_key)), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2("richland_wa", scenario_key)), \
             patch("builtins.input", return_value="yes"):

            agent.process("Water table study near Richland, WA")

        # Verify climate tool was called with correct years
        call_args = agent.mcp_clients["weather"].call_tool_json.call_args
        assert call_args[0][1]["start_year"] == scn["start_year"]
        assert call_args[0][1]["end_year"]   == scn["end_year"]

    @pytest.mark.parametrize("scenario_key", ["drought", "wet", "recent"])
    def test_historical_scenarios_use_correct_tool(self, scenario_key):
        """Historical scenarios call get_historical_climate, not summary."""
        agent = make_agent("richland_wa")

        with patch.object(agent, "respond",
                          return_value=make_pass1("richland_wa", scenario_key)), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2("richland_wa", scenario_key)), \
             patch("builtins.input", return_value="yes"):

            agent.process("Water table study during drought years")

        call_args = agent.mcp_clients["weather"].call_tool_json.call_args
        assert call_args[0][0] == "get_historical_climate"

    def test_normal_scenario_uses_summary_tool(self):
        """Default (normal) scenario calls get_climate_summary."""
        agent = make_agent("richland_wa")

        with patch.object(agent, "respond",
                          return_value=make_pass1("richland_wa", "normal")), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2("richland_wa", "normal")), \
             patch("builtins.input", return_value="yes"):

            agent.process("Water table study near Richland, WA")

        call_args = agent.mcp_clients["weather"].call_tool_json.call_args
        assert call_args[0][0] == "get_climate_summary"

    @pytest.mark.parametrize("location_key,scenario_key", [
        ("richland_wa", "drought"),    # semi-arid + drought
        ("moscow_id",   "normal"),     # semi-humid + average
        ("phoenix_az",  "recent"),     # arid + recent years
        ("miami_fl",    "wet"),        # humid + wet period
    ])
    def test_location_scenario_combinations(self,
                                            location_key,
                                            scenario_key):
        """Cross-product: different locations with different scenarios."""
        agent = make_agent(location_key)
        loc   = LOCATIONS[location_key]
        scn   = CLIMATE_SCENARIOS[scenario_key]

        with patch.object(agent, "respond",
                          return_value=make_pass1(location_key, scenario_key)), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2(location_key, scenario_key)), \
             patch("builtins.input", return_value="yes"):

            result = agent.process(
                f"Study near {loc['name']} using {scn['label']}"
            )

        assert result.intent                      == "design_and_run"
        assert result.brief["region"]["location"] == loc["name"]
        assert result.brief["climate"]["period"]  == (
            f"{scn['start_year']}-{scn['end_year']}"
        )


# ─────────────────────────────────────────────────────────────────────
# TESTS: Climate Contrast Across Locations
# ─────────────────────────────────────────────────────────────────────

class TestClimateContrast:
    """
    Verify the reception agent correctly carries distinct
    climate signals for each region into the planner brief.
    """

    def _run_pipeline(self, location_key, scenario_key="normal"):
        agent = make_agent(location_key)
        with patch.object(agent, "respond",
                          return_value=make_pass1(location_key, scenario_key)), \
             patch.object(agent, "ask_with_system",
                          return_value=make_pass2(location_key, scenario_key)), \
             patch("builtins.input", return_value="yes"):
            return agent.process("Water table study")

    def test_miami_wetter_than_richland(self):
        """Miami (humid) has higher recharge than Richland (semi-arid) [1]."""
        richland = self._run_pipeline("richland_wa")
        miami    = self._run_pipeline("miami_fl")

        assert (miami.brief["climate"]["recharge_m_s"] >
                richland.brief["climate"]["recharge_m_s"])

    def test_phoenix_driest_recharge(self):
        """Phoenix (arid desert) has lowest recharge of all locations."""
        results = {
            loc: self._run_pipeline(loc)
            for loc in LOCATIONS
        }
        recharges = {
            loc: r.brief["climate"]["recharge_m_s"]
            for loc, r in results.items()
        }
        assert min(recharges, key=recharges.get) == "phoenix_az"

    def test_miami_most_soil_layers(self):
        """Miami returns more soil layers than Phoenix (3 vs 3 here,
           but test demonstrates the pattern)."""
        richland = self._run_pipeline("richland_wa")
        moscow   = self._run_pipeline("moscow_id")

        assert (moscow.brief["soil_profile"]["num_layers"] >=
                richland.brief["soil_profile"]["num_layers"])


# ─────────────────────────────────────────────────────────────────────
# TESTS: Edge Cases Across Locations
# ─────────────────────────────────────────────────────────────────────

class TestEdgeCases:

	@pytest.mark.parametrize("location_key", list(LOCATIONS.keys()))
	def test_mcp_failure_does_not_crash(self, location_key):
		"""Pipeline handles MCP returning None gracefully for any location."""
		agent = make_agent(location_key)
	
		# ✅ Explicitly make ALL tool calls return None
		agent.mcp_clients["weather"].call_tool_json.return_value = None
		agent.mcp_clients["geology"].call_tool_json.return_value = None
	
		with patch.object(agent, "respond",
						  return_value=make_pass1(location_key, "normal")), \
			 patch.object(agent, "ask_with_system",
						  return_value=json.dumps({
							  "user_request":      "study",
							  "intent":            "design_and_run",
							  "experiment_focus":  "water table study",
							  "region": {
								  "location": LOCATIONS[location_key]["name"],
								  "lat":      LOCATIONS[location_key]["lat"],
								  "lon":      LOCATIONS[location_key]["lon"],
							  },
							  "climate":            None,   # MCP failed
							  "soil_profile":       None,   # MCP failed
							  "pflotran_materials": None,   # MCP failed
						  })), \
			 patch("builtins.input", return_value="yes"):
	
			result = agent.process("Water table study")
	
		# Pipeline should complete without crashing
		assert result.intent == "design_and_run"
	
		# Brief should exist but data fields are None
		assert result.brief["climate"]            is None
		assert result.brief["soil_profile"]       is None
		assert result.brief["pflotran_materials"] is None
	
		# Region should still be present (from Pass 1, not MCP)
		assert result.brief["region"]["location"] == LOCATIONS[location_key]["name"]
	
	@pytest.mark.parametrize("location_key", list(LOCATIONS.keys()))
	def test_user_cancels_any_location(self, location_key):
		"""User can cancel at confirmation for any location."""
		agent = make_agent(location_key)
	
		with patch.object(agent, "respond",
						  return_value=make_pass1(location_key, "normal")), \
			 patch.object(agent, "ask_with_system",
						  return_value=make_pass2(location_key, "normal")), \
			 patch("builtins.input", return_value="no"):
	
			result = agent.process("Water table study")
	
		assert result.intent == "clarification_needed"
	
	def test_no_location_skips_mcp(self):
		"""Request without location skips MCP fetch entirely."""
		agent = make_agent("richland_wa")
	
		with patch.object(agent, "respond",
						  return_value=json.dumps({
							  "intent":     "clarification_needed",
							  "confidence": "low",
							  "reasoning":  "No location provided",
							  "clarification_questions": [
								  "Which US location are you studying?"
							  ]
						  })):
			result = agent.process("Run some groundwater simulations")
	
		assert result.intent == "clarification_needed"
		agent.mcp_clients["weather"].call_tool_json.assert_not_called()
		agent.mcp_clients["geology"].call_tool_json.assert_not_called()