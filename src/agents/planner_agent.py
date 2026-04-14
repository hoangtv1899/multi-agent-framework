#!/usr/bin/env python3
"""
Planner Agent
Two LLM calls:
    1. Design     → designs complete PFLOTRAN experiment plan
    2. Validation → checks and fixes the plan
"""
import json
from typing import Dict, Any

from agents.llm_agent import LLMAgent
from agents.prompts   import load_prompt


class PlannerAgent(LLMAgent):
    """
    Designs and validates PFLOTRAN experiment plans.
    Receives clean brief from ReceptionAgent.
    Two LLM calls only — no Python validator.
    """

    def __init__(self,
                 model:       str  = "claude-sonnet-4-6-project",
                 mcp_clients: Dict = None):

        self.prompt_design     = load_prompt("planner_system")
        self.prompt_validation = load_prompt("planner_validation")

        super().__init__("planner", self.prompt_design, model)

    # ─────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────

    def create_plan(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a validated PFLOTRAN experiment plan.

        Args:
            brief: Clean dict from ReceptionResult.to_planner_brief()

        Returns:
            Validated plan ready for inspection and ExpManager
        """
        # ── Call 1: Design ────────────────────────────────────────
        print("\n📋 Designing experiments...")
        plan = self._design(brief)
        print(f"   ✓ {len(plan.get('CONDITIONS_COUPLERS', []))} "
              f"experiments designed")

        # ── Call 2: Validate + fix ────────────────────────────────
        print("\n🔬 Validating plan...")
        plan = self._validate_and_fix(plan)

        return plan

    # ─────────────────────────────────────────────────────────────────
    # CALL 1: DESIGN
    # ─────────────────────────────────────────────────────────────────

    def _design(self, brief: Dict) -> Dict:
        """One heavyweight LLM call — designs the complete plan."""
        prompt = (
            f"Design PFLOTRAN groundwater experiments "
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

    def _validate_and_fix(self, plan: Dict) -> Dict:
        """
        One LLM call — checks AND fixes the plan.
        Returns corrected plan if issues found.
        """
        prompt = (
            f"Review and fix this PFLOTRAN experiment plan:\n\n"
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
        return f"PlannerAgent(model={self.llm.model})"