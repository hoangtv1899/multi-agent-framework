#!/usr/bin/env python3
"""
Planner Agent
Two LLM calls:
    1. Design     → designs complete experiment plan (PFLOTRAN or ELM)
    2. Validation → checks and fixes the plan
"""
import json
from typing import Dict, Any

from agents.llm_agent import LLMAgent
from agents.prompts   import load_prompt


class PlannerAgent(LLMAgent):
	"""
	Designs and validates experiment plans for PFLOTRAN or ELM.
	Receives clean brief from ReceptionAgent.
	Two LLM calls only — no Python validator.
	"""

	def __init__(self,
				 model:       str  = "claude-sonnet-4-5-20250929-v1-project",
				 model_type:  str  = "pflotran",
				 mcp_clients: Dict = None):

		# ── Store model type + prompts ────────────────────────
		self.model_type = model_type.lower()

		if self.model_type == "elm":
			self.prompt_design     = load_prompt("planner_system_elm")
			self.prompt_validation = load_prompt("planner_validation_elm")
			self._model_label      = "ELM land-surface"
		else:
			self.prompt_design     = load_prompt("planner_system")
			self.prompt_validation = load_prompt("planner_validation")
			self._model_label      = "PFLOTRAN groundwater"
		# ──────────────────────────────────────────────────────

		super().__init__("planner", self.prompt_design, model)

	# ─────────────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────────────

	def create_plan(self, brief: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Create a validated experiment plan.

		Args:
			brief: Clean dict from ReceptionResult.to_planner_brief()

		Returns:
			Validated plan ready for inspection and the matching ExpManager
		"""
		# ── Call 1: Design ────────────────────────────────────────
		print("\n📋 Designing experiments...")
		plan = self._design(brief)
		print(f"   ✓ {len(plan.get('CONDITIONS_COUPLERS', []))} "
			  f"experiments designed")

		# ── Call 2: Validate + fix ────────────────────────────────
		print("\n🔬 Validating plan...")
		plan = self._validate_and_fix(plan, brief)

		return plan

	# ─────────────────────────────────────────────────────────────────
	# CALL 1: DESIGN
	# ─────────────────────────────────────────────────────────────────

	def _design(self, brief: Dict) -> Dict:
		"""One heavyweight LLM call — designs the complete plan."""
		prompt = (
			f"Design {self._model_label} experiments "
			f"based on this brief:\n\n"
			f"{json.dumps(brief, indent=2)}\n\n"
			f"Think through the parameter space first, "
			f"then output the complete plan as JSON."
		)
		try:
			response = self.ask_with_system(
				user_message   = prompt,
				system_message = self.prompt_design
			)
			return self.parse_json(response)
		except Exception as e:
			raise RuntimeError(
				f"Experiment design failed: {e}"
			) from e

	# ─────────────────────────────────────────────────────────────────
	# CALL 2: VALIDATE + FIX
	# ─────────────────────────────────────────────────────────────────

	def _validate_and_fix(self,
						   plan:  Dict,
						   brief: Dict) -> Dict:
		"""
		One LLM call — checks AND fixes the plan.
		Returns corrected plan if issues found.

		The validator receives both the brief (for bounds checks like
		year ranges and dry/wet period consistency) and the plan.
		"""
		prompt = (
			f"Review and fix this {self._model_label} "
			f"experiment plan.\n\n"
			f"Original brief:\n"
			f"{json.dumps(brief, indent=2)}\n\n"
			f"Generated plan:\n"
			f"{json.dumps(plan, indent=2)}"
		)
		try:
			response = self.ask_with_system(
				user_message   = prompt,
				system_message = self.prompt_validation
			)
			result = self.parse_json(response)

			# Report issues
			for issue in result.get("issues", []):
				icon = ("❌" if issue.get("severity") == "critical"
						else "⚠️")
				print(f"   {icon} [{issue.get('check')}] "
					  f"{issue.get('issue')}")

			# Use corrected plan if provided
			corrected = result.get("corrected_plan")
			if corrected:
				print("   ✓ Plan corrected by validator")
				return corrected

			print("   ✓ All checks passed")
			return plan

		except Exception as e:
			print(f"   ⚠️  Validation failed: {e} — using original plan")
			return plan

	def __repr__(self):
		return (f"PlannerAgent(model={self.llm.model}, "
				f"type={self.model_type})")