#!/usr/bin/env python3
"""
Planner Agent

Two modes:

  capability-aware (DEFAULT for ELM / multi-model) — ONE LLM call with the
  capability-probe system prompt. The LLM emits a STRATEGY only: scientific
  decomposition, an explicit full/partial/infeasible feasibility verdict
  (answerable vs not-answerable), a stratified sampling_strategy expressed as
  RULES (elevation/forcing strata, justified N) — never coordinates or per-
  column configs — plus a requires_capabilities backlog and recorded
  assumptions. Materialization of coordinates/soil/WTD is the deterministic
  Tier-2 expander's job (expand_sampling.py); this planner is architecturally
  forbidden from inventing a number that must be real. A light deterministic
  schema check replaces the old LLM "fix" call (reproducible, no second call).

  legacy treatment-suite (capability_aware=False, or PFLOTRAN) — the original
  two-call design+validate path that emits CONDITIONS_COUPLERS treatment suites
  (baseline/wet/dry x soil). Kept for the single-site PFLOTRAN workflow and the
  retired ELM tests.
"""
import json
from typing import Dict, Any

from agents.llm_agent import LLMAgent
from agents.prompts   import load_prompt

# strategy fields the capability-aware planner must emit (deterministic check)
STRATEGY_REQUIRED = ("feasibility", "sampling_strategy", "requires_capabilities")


class PlannerAgent(LLMAgent):
    """Designs experiment plans for ELM / PFLOTRAN. Capability-aware by default."""

    def __init__(self,
                 model:            str  = "claude-sonnet-4-5-20250929-v1-project",
                 model_type:       str  = "pflotran",
                 capability_aware: bool = True,
                 mcp_clients:      Dict = None):

        self.model_type = model_type.lower()
        # PFLOTRAN currently only has the legacy treatment prompt; ELM defaults
        # to the capability-aware strategy planner.
        self.capability_aware = bool(capability_aware) and self.model_type == "elm"

        if self.capability_aware:
            self.prompt_design     = load_prompt("planner_capability_probe")
            self.prompt_validation = None
            self._model_label      = "multi-model (ELM/PFLOTRAN) strategy"
        elif self.model_type == "elm":
            self.prompt_design     = load_prompt("planner_system_elm")
            self.prompt_validation = load_prompt("planner_validation_elm")
            self._model_label      = "ELM land-surface"
        else:
            self.prompt_design     = load_prompt("planner_system")
            self.prompt_validation = load_prompt("planner_validation")
            self._model_label      = "PFLOTRAN groundwater"

        super().__init__("planner", self.prompt_design, model)

    # ─────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def create_plan(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Design (and validate) an experiment plan from a reception brief."""
        if self.capability_aware:
            print("\n📋 Designing strategy (capability-aware)...")
            plan = self._design_strategy(brief)
            self._report_strategy(plan)
            return plan

        # ── legacy treatment-suite path ──
        print("\n📋 Designing experiments...")
        plan = self._design(brief)
        print(f"   ✓ {len(plan.get('CONDITIONS_COUPLERS', []))} experiments designed")
        print("\n🔬 Validating plan...")
        plan = self._validate_and_fix(plan, brief)
        return plan

    # ─────────────────────────────────────────────────────────────────
    # CAPABILITY-AWARE STRATEGY (one call + deterministic schema check)
    # ─────────────────────────────────────────────────────────────────
    def _design_strategy(self, brief: Dict) -> Dict:
        prompt = (
            f"Design a simulation STRATEGY for this brief. Emit a feasibility "
            f"verdict (answerable vs not-answerable), a stratified sampling "
            f"strategy as RULES (bands + justified N, NO coordinates), a "
            f"requires_capabilities backlog, and recorded assumptions. "
            f"Return JSON only.\n\n{json.dumps(brief, indent=2)}"
        )
        try:
            response = self.ask_with_system(
                user_message=prompt, system_message=self.prompt_design)
            plan = self.parse_json(response)
        except Exception as e:
            raise RuntimeError(f"Strategy design failed: {e}") from e

        # deterministic, reproducible schema check (replaces the LLM fix call)
        missing = [k for k in STRATEGY_REQUIRED if not plan.get(k)]
        if missing:
            plan.setdefault("_schema_warnings", []).extend(
                f"missing required strategy field: {k}" for k in missing)
        # anti-hallucination invariant: the planner must NOT have emitted
        # coordinates or an enumerated column list — that is the expander's job.
        leaked = self._coordinate_leak(plan)
        if leaked:
            plan.setdefault("_schema_warnings", []).append(
                f"anti-hallucination: planner emitted coordinate-like fields {leaked} "
                "— these are ignored downstream (materialized by the Tier-2 expander)")
        return plan

    @staticmethod
    def _coordinate_leak(plan: Dict) -> list:
        """Report any lat/lon/coordinate keys the LLM emitted despite the rule
        (the deterministic guard behind the anti-hallucination boundary)."""
        hits = []
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    kl = str(k).lower()
                    if kl in ("lat", "lon", "latitude", "longitude", "coordinates",
                              "lat_lon", "latlon") and v not in (None, "", [], {}):
                        hits.append(path + "/" + str(k))
                    walk(v, path + "/" + str(k))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
        walk(plan)
        return hits[:8]

    def _report_strategy(self, plan: Dict) -> None:
        fe = plan.get("feasibility") or {}
        verdict = fe.get("verdict", "(no verdict)")
        ss = plan.get("sampling_strategy") or {}
        n = (plan.get("sampling_plan") or {}).get("n_exploratory") or ss.get("n_exploratory")
        print(f"   ✓ feasibility: {str(verdict)[:88]}")
        if n:
            print(f"   ✓ sampling: N={n} (strategy rules only — no coordinates)")
        reqs = plan.get("requires_capabilities") or []
        if reqs:
            print(f"   ✓ requires_capabilities: {len(reqs)} item(s) flagged")
        for w in plan.get("_schema_warnings", []):
            print(f"   ⚠️  {w}")

    # ─────────────────────────────────────────────────────────────────
    # LEGACY TREATMENT-SUITE PATH (design + validate)
    # ─────────────────────────────────────────────────────────────────
    def _design(self, brief: Dict) -> Dict:
        prompt = (
            f"Design {self._model_label} experiments based on this brief:\n\n"
            f"{json.dumps(brief, indent=2)}\n\n"
            f"Think through the parameter space first, then output the complete "
            f"plan as JSON."
        )
        try:
            response = self.ask_with_system(
                user_message=prompt, system_message=self.prompt_design)
            return self.parse_json(response)
        except Exception as e:
            raise RuntimeError(f"Experiment design failed: {e}") from e

    def _validate_and_fix(self, plan: Dict, brief: Dict) -> Dict:
        prompt = (
            f"Review and fix this {self._model_label} experiment plan.\n\n"
            f"Original brief:\n{json.dumps(brief, indent=2)}\n\n"
            f"Generated plan:\n{json.dumps(plan, indent=2)}"
        )
        try:
            response = self.ask_with_system(
                user_message=prompt, system_message=self.prompt_validation)
            result = self.parse_json(response)
            for issue in result.get("issues", []):
                icon = "❌" if issue.get("severity") == "critical" else "⚠️"
                print(f"   {icon} [{issue.get('check')}] {issue.get('issue')}")
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
        mode = "capability-aware" if self.capability_aware else "treatment-suite"
        return f"PlannerAgent(model={self.llm.model}, type={self.model_type}, mode={mode})"
