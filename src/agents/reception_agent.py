#!/usr/bin/env python3
"""
Reception Agent
Two-pass pipeline:
    Pass 1 → LLM extracts location + intent + temporal context
    Bridge → Python fetches MCP data (climate baseline/dry/wet + soil)
    Pass 2 → LLM synthesizes into clean planner brief
    Confirm → User confirms with one-line summary (Option B)
"""
import json
from dataclasses import dataclass, field
import sys
from typing      import Dict, Any, List, Optional

from agents.llm_agent import LLMAgent
from agents.prompts   import load_prompt


# ─────────────────────────────────────────────────────────────────────
# OUTPUT CONTRACT
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ReceptionResult:
    """Structured output of the reception phase."""
    user_request:            str
    intent:                  str
    confidence:              str
    parameters:              Dict[str, Any]
    brief:                   Dict[str, Any]
    clarification_questions: List[str] = field(default_factory=list)

    def to_planner_brief(self) -> Dict[str, Any]:
        """Clean JSON dict for the Planner Agent."""
        return self.brief


# ─────────────────────────────────────────────────────────────────────
# RECEPTION AGENT
# ─────────────────────────────────────────────────────────────────────

class ReceptionAgent(LLMAgent):
	"""
	Two-pass reception pipeline.
	
	Pass 1 : LLM classifies intent + extracts location + temporal context
	Bridge : Python calls MCP tools directly (no LLM)
	Pass 2 : LLM synthesizes everything into planner brief
	Confirm: One-line summary → user yes/no (Option B)
	"""
	
	# CHANGE 1: Add model_type parameter
	def __init__(self,
				 model:       str  = "gemini-2.5-flash-project",
				 mcp_clients: Dict = None,
				 model_type:  str  = "pflotran"):   # ← ADD THIS
	
		self.mcp_clients  = mcp_clients or {}
		self.model_type   = model_type              # ← ADD THIS
		mcp_section       = self._build_mcp_section(self.mcp_clients)
		self.prompt_pass1 = load_prompt(
			"reception_pass1",
			mcp_sources_section = mcp_section
		)
	
		# CHANGE 2: Load correct Pass 2 prompt
		if model_type == "elm":
			self.prompt_pass2 = load_prompt("reception_pass2_elm")
		else:
			self.prompt_pass2 = load_prompt("reception_pass2")
	
		super().__init__("reception", self.prompt_pass1, model)
	
	# ─────────────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────────────
	
	def process(self,
				user_request:         str,
				conversation_context: Dict[str, Any] = None
				) -> ReceptionResult:
		"""Full reception pipeline."""
		conversation_context = conversation_context or {}
	
		# ── Pass 1 ────────────────────────────────────────────────
		print("\n📥 Pass 1: Classifying intent...")
		pass1 = self._pass1_extract(user_request, conversation_context)
		print(f"   Intent:   {pass1.get('intent')} "
			  f"({pass1.get('confidence')})")
		print(f"   Location: {pass1.get('location', 'none')}")
		print(f"   Period:   {pass1.get('start_year')}–"
			  f"{pass1.get('end_year')} "
			  f"({'historical' if pass1.get('use_historical') else 'summary'})")
	
		# ── Early exit ────────────────────────────────────────────
		if pass1.get("intent") == "clarification_needed":
			return ReceptionResult(
				user_request            = user_request,
				intent                  = "clarification_needed",
				confidence              = pass1.get("confidence", "low"),
				parameters              = pass1,
				brief                   = {},
				clarification_questions = pass1.get(
					"clarification_questions", []
				),
			)
	
		# ── Bridge ────────────────────────────────────────────────
		mcp_data = {}
		if pass1.get("lat") and pass1.get("lon") and self.mcp_clients:
			print(f"\n🌍 Fetching regional data for: "
				  f"{pass1.get('location')}")
			mcp_data = self._fetch_mcp_data(
				lat            = pass1["lat"],
				lon            = pass1["lon"],
				start_year     = pass1.get("start_year",    2006),
				end_year       = pass1.get("end_year",      2025),
				use_historical = pass1.get("use_historical", False),
				dry_start      = pass1.get("dry_start_year"),
				dry_end        = pass1.get("dry_end_year"),
				wet_start      = pass1.get("wet_start_year"),
				wet_end        = pass1.get("wet_end_year"),
			)
	
		# ── Pass 2 ────────────────────────────────────────────────
		print("\n📤 Pass 2: Synthesizing planner brief...")
		brief = self._pass2_synthesize(user_request, pass1, mcp_data)
	
		# ── Confirm ───────────────────────────────────────────────
		if not self._confirm(brief):
			print("  Resubmit with a different location.\n")
			return ReceptionResult(
				user_request            = user_request,
				intent                  = "clarification_needed",
				confidence              = "low",
				parameters              = pass1,
				brief                   = {},
				clarification_questions = [
					"Please resubmit with a different location or request."
				],
			)
	
		return ReceptionResult(
			user_request = user_request,
			intent       = pass1["intent"],
			confidence   = pass1.get("confidence", "medium"),
			parameters   = pass1,
			brief        = brief,
		)
	
	# ─────────────────────────────────────────────────────────────────
	# PASS 1
	# ─────────────────────────────────────────────────────────────────
	
	def _pass1_extract(self,
					   user_request:         str,
					   conversation_context: Dict) -> Dict:
		"""One LLM call — returns intent, location, lat/lon, temporal."""
		has_run    = conversation_context.get("last_run_dir")
		has_plan   = conversation_context.get("last_plan")
		last_focus = conversation_context.get("last_focus")
	
		prompt = (
			f'User request: "{user_request}"\n\n'
			f'Context:\n'
			f'- Previous run:   '
			f'{"Yes (" + has_run + ")" if has_run else "No"}\n'
			f'- Previous plan:  {"Yes" if has_plan else "No"}\n'
			f'- Previous focus: {last_focus or "None"}\n\n'
			f'Classify and extract all relevant fields. '
			f'Output JSON only.'
		)
	
		try:
			response = self.respond(prompt)
			return self.parse_json(response)
		except Exception as e:
			print(f"⚠️  Pass 1 failed: {e}")
			return self._default_clarification()
		finally:
			self.reset()
	
	# ─────────────────────────────────────────────────────────────────
	# BRIDGE — Python fetches MCP data directly
	# ─────────────────────────────────────────────────────────────────
	
	def _fetch_mcp_data(self,
						lat:            float,
						lon:            float,
						start_year:     int   = 2006,
						end_year:       int   = 2025,
						use_historical: bool  = False,
						dry_start:      Optional[int] = None,
						dry_end:        Optional[int] = None,
						wet_start:      Optional[int] = None,
						wet_end:        Optional[int] = None
						) -> Dict[str, Any]:
		"""
		Fetch MCP data. Always fetches baseline climate + soil.
		If use_historical=True, also fetches dry + wet periods.
		"""
		data    = {}
		weather = self.mcp_clients.get("weather")
		geology = self.mcp_clients.get("geology")
		args    = {"lat": lat, "lon": lon}
	
		if weather:
			# Always fetch baseline summary
			data["climate_baseline"] = weather.call_tool_json(
				"get_climate_summary",
				{**args, "start_year": start_year, "end_year": end_year}
			)
			if data["climate_baseline"]:
				precip   = (
					data["climate_baseline"].get("precip_mm_per_year") or
					data["climate_baseline"].get("annual", {})
										   .get("precip_mm_per_year")
				)
				recharge = (
					data["climate_baseline"].get("suggested_recharge_flux_ms") or
					data["climate_baseline"].get("annual", {})
										   .get("recharge_flux_ms")
				)
				print(f"  ✓ Climate baseline: "
					  f"{precip} mm/yr | {recharge} m/s")
	
			# Fetch dry period if requested
			if use_historical and dry_start and dry_end:
				data["climate_dry"] = weather.call_tool_json(
					"get_historical_climate",
					{**args,
					 "start_year": dry_start,
					 "end_year":   dry_end}
				)
				if data["climate_dry"]:
					dry_r = (data["climate_dry"].get("annual", {})
											   .get("recharge_flux_ms"))
					dry_p = (data["climate_dry"].get("annual", {})
											   .get("precip_mm_per_year"))
					print(f"  ✓ Climate dry "
						  f"({dry_start}–{dry_end}): "
						  f"{dry_p} mm/yr | {dry_r} m/s")
	
			# Fetch wet period if requested
			if use_historical and wet_start and wet_end:
				data["climate_wet"] = weather.call_tool_json(
					"get_historical_climate",
					{**args,
					 "start_year": wet_start,
					 "end_year":   wet_end}
				)
				if data["climate_wet"]:
					wet_r = (data["climate_wet"].get("annual", {})
											   .get("recharge_flux_ms"))
					wet_p = (data["climate_wet"].get("annual", {})
											   .get("precip_mm_per_year"))
					print(f"  ✓ Climate wet "
						  f"({wet_start}–{wet_end}): "
						  f"{wet_p} mm/yr | {wet_r} m/s")
	
		if geology:
			data["soil_profile"] = geology.call_tool_json(
				"get_soil_profile", args
			)
			if data["soil_profile"]:
				n = data["soil_profile"].get("num_layers", 0)
				print(f"  ✓ Soil: {n} SSURGO layers")
				if n < 3:
					print(f"  ⚠️  Only {n} soil layer(s) — "
						  f"verify coordinates or try nearby location")
	
			data["pflotran_materials"] = geology.call_tool_json(
				"get_pflotran_materials", args
			)
			if data["pflotran_materials"]:
				print(f"  ✓ PFLOTRAN materials ready")
	
		return data
	
	# ─────────────────────────────────────────────────────────────────
	# PASS 2
	# ─────────────────────────────────────────────────────────────────
	
	def _pass2_synthesize(self,
						  user_request: str,
						  pass1:        Dict,
						  mcp_data:     Dict) -> Dict:
		"""One LLM call — synthesizes into planner brief."""
		prompt = (
			f"User request: {user_request}\n\n"
			f"Intent classification:\n"
			f"{json.dumps(pass1,    indent=2)}\n\n"
			f"Regional MCP data:\n"
			f"{json.dumps(mcp_data, indent=2)}\n\n"
			f"Synthesize into a planner brief. Output JSON only."
		)
	
		try:
			response = self.ask_with_system(
				user_message   = prompt,
				system_message = self.prompt_pass2
			)
			return self.parse_json(response)
		except Exception as e:
			print(f"⚠️  Pass 2 failed: {e} — returning raw data")
			return {"user_request": user_request, **pass1, **mcp_data}
	
	# ─────────────────────────────────────────────────────────────────
	# OPTION B CONFIRMATION
	# ─────────────────────────────────────────────────────────────────

	def _confirm(self, brief: Dict) -> bool:
		"""One-line summary → user yes/no."""
		region  = brief.get("region")       or {}
		climate = brief.get("climate_baseline") or {}
		soil    = brief.get("soil_profile") or {}

		# Try both field name conventions
		precip   = (climate.get("precip_mm_yr") or
					climate.get("precip_mm_per_year") or "N/A")
		recharge = (climate.get("recharge_m_s") or
					climate.get("suggested_recharge_flux_ms") or "N/A")

		# Note wet/dry if available
		has_dry = "climate_dry" in brief
		has_wet = "climate_wet" in brief
		climate_note = ""
		if has_dry and has_wet:
			climate_note = " | dry + wet periods fetched"
		elif has_dry:
			climate_note = " | dry period fetched"
		elif has_wet:
			climate_note = " | wet period fetched"

		print(
			f"\n📍 {region.get('location', 'Unknown')}  |  "
			f"Precip: {precip} mm/yr  |  "
			f"Recharge: {recharge} m/s  |  "
			f"{soil.get('num_layers', '?')} soil layers"
			f"{climate_note}"
		)
		if not sys.stdin.isatty():
			print("Proceed? (yes / no): yes  [non-interactive — auto-confirming]")
			answer = "yes"
		else:
			answer = input("Proceed? (yes / no): ").strip().lower()
		return answer in ("yes", "y", "")
	
	# ─────────────────────────────────────────────────────────────────
	# UTILITIES
	# ─────────────────────────────────────────────────────────────────
	
	@staticmethod
	def _build_mcp_section(mcp_clients: Dict) -> str:
		if not mcp_clients:
			return "No regional data sources currently available."
		SOURCE_MAP = {
			"weather": "- Weather: climate summary, historical dry/wet periods",
			"geology": "- Geology: SSURGO soil profile, PFLOTRAN materials",
		}
		lines = ["AVAILABLE DATA SOURCES:"]
		for key, desc in SOURCE_MAP.items():
			if key in mcp_clients:
				lines.append(desc)
		lines.append(
			"\nWhen a US location is mentioned, "
			"these sources will be queried automatically."
		)
		return "\n".join(lines)
	
	@staticmethod
	def _default_clarification() -> Dict:
		return {
			"intent":     "clarification_needed",
			"confidence": "low",
			"reasoning":  "Could not parse intent",
			"clarification_questions": [
				"Could you rephrase your request?",
				"Are you (1) designing new experiments or "
				"(2) analyzing existing results?",
			],
		}
	
	def __repr__(self):
		return (f"ReceptionAgent(model={self.llm.model}, "
				f"mcp={list(self.mcp_clients.keys())})")