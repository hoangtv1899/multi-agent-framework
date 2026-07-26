#!/usr/bin/env python3
"""
Planner Agent

ONE LLM call with the capability-probe system prompt. The LLM emits a
STRATEGY only: scientific
  decomposition, an explicit full/partial/infeasible feasibility verdict
  (answerable vs not-answerable), a stratified sampling_strategy expressed as
  RULES (elevation/forcing strata, justified N) — never coordinates or per-
  column configs — plus a requires_capabilities backlog and recorded
  assumptions. Materialization of coordinates/soil/WTD is the deterministic
  Tier-2 expander's job (expand_sampling.py); this planner is architecturally
  forbidden from inventing a number that must be real. A light deterministic
  schema check replaces the old LLM "fix" call (reproducible, no second call).

The older two-call design+validate path (capability_aware=False) was removed:
no caller ever set the flag, so it and its four prompts were unreachable.
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
                 model:            str  = "claude-opus-4-8-project",
                 model_type:       str  = "elm",
                 mcp_clients:      Dict = None,
                 capability_prompt: str = "planner_capability_probe_v2"):

        self.model_type = model_type.lower()
        # The frozen v1 prompt (planner_capability_probe) is pinned by the
        # pre-registered eval via capability_prompt — do not change v1.
        self.prompt_design = load_prompt(capability_prompt)

        super().__init__("planner", self.prompt_design, model)

    # ─────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def create_plan(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Design an experiment strategy from a reception brief."""
        print("\n📋 Designing strategy (capability-aware)...")
        plan = self._design_strategy(brief)
        self._report_strategy(plan)
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
        response = None
        try:
            response = self.ask_with_system(
                user_message=prompt, system_message=self.prompt_design)
            # The strategy JSON is long (~90 lines) and structural slips such
            # as a missing comma are the single most common way an otherwise
            # good plan is lost, so allow one self-repair round.
            plan = self.parse_json_resilient(response)
        except Exception as e:
            err = RuntimeError(f"Strategy design failed: {e}")
            err.raw_response = response
            raise err from e

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
        verdict = fe.get("verdict", "(no verdict)") if isinstance(fe, dict) else str(fe)
        ss = plan.get("sampling_strategy") or {}
        sp = plan.get("sampling_plan")
        n = None
        for src in (sp, ss):
            if isinstance(src, dict):
                n = n or src.get("n_exploratory") or src.get("n_columns")
            elif isinstance(src, list):
                n = n or len(src)
        print(f"   ✓ feasibility: {str(verdict)[:88]}")
        if n:
            print(f"   ✓ sampling: N={n} (strategy rules only — no coordinates)")
        reqs = plan.get("requires_capabilities") or []
        if reqs:
            print(f"   ✓ requires_capabilities: {len(reqs)} item(s) flagged")
        for w in plan.get("_schema_warnings", []):
            print(f"   ⚠️  {w}")

    def __repr__(self):
        return f"PlannerAgent(model={self.llm.model}, type={self.model_type})"
