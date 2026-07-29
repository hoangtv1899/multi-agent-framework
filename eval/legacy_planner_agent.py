#!/usr/bin/env python3
"""
FROZEN. This is the planner as it stood when eval/ was run, kept beside the
record it produced so the paper stays reproducible. The live pipeline uses
src/agents/planner.py; do not import this from there.

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
            # A malformed-JSON repair can recover the document but drop a field
            # (observed: requires_capabilities lost at char ~6000). Ask for just
            # the missing keys rather than accepting the gap or re-running the
            # whole design — a second full call would also re-roll the strategy.
            plan = self._fill_missing(plan, missing, brief)
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

    def _fill_missing(self, plan: Dict, missing: list, brief: Dict) -> Dict:
        """Ask for ONLY the missing strategy keys and merge them in.

        Keeps the rest of the accepted plan byte-identical, so a dropped field
        cannot silently re-roll the sampling design. Failure is non-fatal — the
        caller records a schema warning instead.
        """
        try:
            ask = (
                f"Your previous strategy response was missing these REQUIRED "
                f"top-level field(s): {', '.join(missing)}.\n\n"
                f"Emit JSON containing ONLY those field(s), consistent with the "
                f"strategy you already gave. Do not restate anything else.\n\n"
                f"Strategy so far:\n"
                f"{json.dumps({k: v for k, v in plan.items() if not k.startswith('_')}, indent=1)[:4000]}\n\n"
                f"Brief:\n{json.dumps(brief, indent=1)[:2000]}"
            )
            patch = self.parse_json(
                self.ask_with_system(user_message=ask,
                                     system_message=self.prompt_design))
            filled = [k for k in missing if patch.get(k)]
            for k in filled:
                plan[k] = patch[k]
            if filled:
                print(f"   ✓ recovered missing field(s): {', '.join(filled)}")
        except Exception as e:
            print(f"   ⚠️  could not recover {missing} ({e})")
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
        # Report the count the EXPERIMENT MANAGER will build, which is
        # expand_sampling._n_from_plan: sampling_strategy.n_exploratory, then
        # experiment_summary.exploratory. Preferring `sampling_plan` here (as
        # this did) printed a different N two lines above the coordinator's,
        # from a field nothing downstream reads — a plan showed "N=4" and then
        # "N=11", and 11 was what ran.
        ss = plan.get("sampling_strategy") or {}
        n = ss.get("n_exploratory") if isinstance(ss, dict) else None
        if n is None:
            n = (plan.get("experiment_summary") or {}).get("exploratory")
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
