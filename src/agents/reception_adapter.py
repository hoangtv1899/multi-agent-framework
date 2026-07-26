#!/usr/bin/env python3
"""
Agentic reception -> workflow.py adapter
src/agents/reception_adapter.py

workflow.py was written against the legacy two-pass ReceptionAgent, which is
structurally SINGLE-POINT: its brief carries `region: {location, lat, lon}` and
has no notion of a watershed (grep bbox|huc|domain in reception_agent.py finds
nothing). The modern spatial pipeline needs `domain: {name, huc, bbox}`, which
only LLMReceptionAgent (the agentic tool-loop agent) produces.

The two are not drop-in compatible in two ways, both handled here:

  1. Return type   LLMReceptionAgent.process() -> plain dict
                   {brief, trace, rounds, raw}
                   workflow.py needs the ReceptionResult dataclass API
                   (.intent, .confidence, .parameters, .user_request,
                    .clarification_questions, .to_planner_brief()).

  2. Intent words  agentic emits "design"; workflow.py dispatches on
                   "design_and_run". An unmapped intent falls straight through
                   to "Unknown intent" and the run silently does nothing.

Everything else in workflow.py is left alone.
"""
from typing import Any, Dict

from agents.reception_agent import ReceptionResult
from agents.reception_llm   import LLMReceptionAgent, DEFAULT_ALLOWLIST


# agentic vocabulary -> workflow.py vocabulary
_INTENT_MAP = {
    "design":              "design_and_run",
    "clarification_needed": "clarification_needed",
    "analyze_existing":     "analyze_existing",
    "parse_error":          "clarification_needed",
}


class AgenticReceptionAdapter:
    """LLMReceptionAgent behind the legacy ReceptionAgent interface."""

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
        ctx = {k: v for k, v in (conversation_context or {}).items() if v}
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
