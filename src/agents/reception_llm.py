#!/usr/bin/env python3
"""
Agentic reception agent.

A pure-LLM reception that DRIVES the MCP tools itself (via ToolLoopAgent) to
turn a natural-language request into a framed brief for the planner — or a
clarification / analysis route. All "what to fetch / when to stop / what it
means" decisions live in the LLM; this class only runs the loop and parses the
final JSON.

This is additive: the legacy two-pass ReceptionAgent (reception_agent.py) and
the production workflow are untouched.
"""
import json
import re
from typing import Any, Dict

from agents.prompts import load_prompt
from agents.tool_loop import ToolLoopAgent

# Curated tool subset reception may use (keeps the schema focused vs all ~30).
DEFAULT_ALLOWLIST = {
    "terrain__resolve_watershed",
    "terrain__elevation_summary",
    "terrain__get_elevation",
    "terrain__sample_elevation_grid",
    "usgs_water__get_groundwater_sites",
    "usgs_water__get_water_table_depth",
    "usgs_water__get_monitoring_locations",
    "fan_wtd__get_fan_wtd",
    "fan_wtd__sample_fan_wtd",
    "fan_wtd__data_status",
    "geology__get_soil_profile",
    "weather__get_climate_summary",
    "snotel__get_snotel_stations",
    "snotel__get_snotel_swe",
}


class LLMReceptionAgent:
    """Tool-using reception: request -> framed brief (+ tool trace)."""

    def __init__(self,
                 model: str,
                 mcp_clients: Dict[str, Any],
                 allowlist: set = None,
                 max_rounds: int = 10,
                 verbose: bool = True,
                 interactive: bool = False):
        self.system = load_prompt("reception_agentic")
        self.loop = ToolLoopAgent(
            model=model,
            mcp_clients=mcp_clients,
            allowlist=allowlist if allowlist is not None else DEFAULT_ALLOWLIST,
            max_rounds=max_rounds,
            verbose=verbose,
            interactive=interactive,
        )

    @property
    def exposed_tools(self):
        return [t["function"]["name"] for t in self.loop.tools]

    def process(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the agentic loop and parse the brief.

        `context` (optional) is a prior experiment in this session — for
        cross-model follow-ups ("run PFLOTRAN with that ELM output").

        Returns {brief, trace, rounds, raw}. `brief` is the parsed JSON dict
        (intent = design | clarification_needed | analyze_existing), or
        {"intent": "parse_error", ...} if the final message had no JSON.
        """
        msg = f"User request: {user_request}\n\n"
        if context:
            msg += ("CONVERSATION CONTEXT (a prior experiment this session):\n"
                    + json.dumps(context, indent=2) + "\n\n")
        msg += "Gather what you need with the tools, then emit ONLY your final JSON."
        out = self.loop.run(self.system, msg)
        brief = self._parse(out["content"])
        return {"brief": brief, "trace": out["trace"],
                "rounds": out["rounds"], "raw": out["content"]}

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as e:
                return {"intent": "parse_error", "error": str(e),
                        "raw": text[:400]}
        return {"intent": "parse_error", "error": "no JSON found",
                "raw": text[:400]}
