#!/usr/bin/env python3
"""
Reception -> coordinator adapter
src/agents/reception_adapter.py

LLMReceptionAgent speaks the agentic vocabulary; workflow.py dispatches on the
coordinator's. This shim owns the translation, and the ReceptionResult contract
the coordinator routes on. Two mismatches, both handled here:

  1. Return type   LLMReceptionAgent.process() -> plain dict
                   {brief, trace, rounds, raw}
                   workflow.py needs the ReceptionResult dataclass API
                   (.intent, .confidence, .parameters, .user_request,
                    .clarification_questions, .to_planner_brief()).

  2. Intent words  agentic emits "design"; workflow.py dispatches on
                   "design_and_run". An unmapped intent falls straight through
                   to "Unknown intent" and the run silently does nothing.

"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agents.reception_llm import LLMReceptionAgent, DEFAULT_ALLOWLIST


@dataclass
class ReceptionResult:
    """Structured output of the reception phase — what the coordinator routes on."""
    user_request:            str
    intent:                  str
    confidence:              str
    parameters:              Dict[str, Any]
    brief:                   Dict[str, Any]
    clarification_questions: List[str] = field(default_factory=list)

    def to_planner_brief(self) -> Dict[str, Any]:
        """Clean JSON dict for the Planner Agent."""
        return self.brief


# agentic vocabulary -> workflow.py vocabulary
_INTENT_MAP = {
    "design":              "design_and_run",
    "clarification_needed": "clarification_needed",
    "analyze_existing":     "analyze_existing",
    "parse_error":          "clarification_needed",
}


def _distil_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compact the coordinator's conversation state into something a prompt can
    actually use.

    The raw state carries `last_plan` and `last_analysis` — whole nested
    documents. Passing them verbatim floods the context window and buries the
    two things a follow-up actually needs: what was asked last time, and what
    came back. Without this, "now try X instead" started from scratch even
    though the state was sitting right there.
    """
    ctx = ctx or {}
    out: Dict[str, Any] = {}
    if ctx.get("last_run_dir"):
        out["prior_run_dir"] = str(ctx["last_run_dir"])
    if ctx.get("last_focus"):
        out["prior_focus"] = str(ctx["last_focus"])[:300]

    plan = ctx.get("last_plan") or {}
    if isinstance(plan, dict):
        strat = plan.get("sampling_strategy") or {}
        if strat:
            out["prior_design"] = {
                "approach": strat.get("approach"),
                "n_exploratory": strat.get("n_exploratory")}
        verdict = (plan.get("feasibility") or {}).get("verdict")
        if verdict:
            out["prior_feasibility"] = verdict

    ana = ctx.get("last_analysis")
    if isinstance(ana, dict):
        answer = (ana.get("answer_to_user_question")
                  or ana.get("answer") or "")
        if answer:
            out["prior_answer"] = str(answer)[:600]
        finds = ana.get("key_findings") or []
        if finds:
            out["prior_findings"] = [str(f)[:200] for f in finds[:3]]
    elif isinstance(ana, str) and ana.strip():
        out["prior_answer"] = ana[:600]
    return out


class AgenticReceptionAdapter:
    """LLMReceptionAgent behind the coordinator's ReceptionResult interface."""

    def __init__(self,
                 model:       str,
                 mcp_clients: Dict[str, Any] = None,
                 model_type:  str = "elm",
                 allowlist:   set = None,
                 max_rounds:  int = 10,
                 verbose:     bool = True,
                 interactive: bool = False):
        self.model_type = model_type
        self.agent = LLMReceptionAgent(
            model       = model,
            mcp_clients = mcp_clients or {},
            allowlist   = allowlist if allowlist is not None else DEFAULT_ALLOWLIST,
            max_rounds  = max_rounds,
            verbose     = verbose,
            interactive = interactive,
        )
        # Populated after process(); workflow.py reads it to drive the
        # Experiment Manager with the period reception actually resolved.
        self.last_run_settings: Dict[str, Any] = {}

    @property
    def exposed_tools(self):
        return self.agent.exposed_tools

    def process(self,
                user_request:         str,
                conversation_context: Dict[str, Any] = None) -> ReceptionResult:
        """Run the agentic loop and return a legacy-shaped ReceptionResult."""
        # The agentic agent takes a prior-experiment dict for cross-model
        # follow-ups; workflow.py's context carries last_plan/last_run_dir/etc.
        ctx = _distil_context(conversation_context)
        out = self.agent.process(user_request, context=ctx or None)
        brief = out.get("brief") or {}

        raw_intent = brief.get("intent", "parse_error")
        intent     = _INTENT_MAP.get(raw_intent, "clarification_needed")

        questions = list(brief.get("questions") or [])
        if raw_intent == "parse_error" and not questions:
            questions = ["Reception could not produce a valid brief. "
                         "Please restate the request, naming the region "
                         "(ideally a HUC code) and what you want to learn."]

        self.last_run_settings = brief.get("run_settings") or {}

        # workflow.py reads only these two parameter keys.
        framing = brief.get("scientific_framing") or {}
        goals   = framing.get("goals") or []
        params = {
            "experiment_focus":  (goals[0] if goals else brief.get("notes")),
            "existing_run_dir":  brief.get("run_dir"),
            "resolved_period":   (self.last_run_settings.get("resolved_period")),
            "initialization":    (self.last_run_settings.get("initialization")),
            "design_archetype":  brief.get("design_archetype"),
        }

        return ReceptionResult(
            user_request            = user_request,
            intent                  = intent,
            # The agentic schema has no confidence field. Report the honest
            # thing rather than inventing a number: a parsed brief is "high",
            # anything needing clarification is "low".
            confidence              = "low" if intent == "clarification_needed" else "high",
            parameters              = params,
            brief                   = brief,
            clarification_questions = questions,
        )
