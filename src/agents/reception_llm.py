#!/usr/bin/env python3
"""
Agentic reception agent.

A pure-LLM reception that DRIVES the MCP tools itself (via ToolLoopAgent) to
turn a natural-language request into a framed brief for the planner — or a
clarification / analysis route. All "what to fetch / when to stop / what it
means" decisions live in the LLM; this class only runs the loop and parses the
final JSON.

This is the Reception box of the framework. workflow.py reaches it through
workflow.py directly; tools/run_pipeline.py uses it for the dry path.
"""
import json
import re
from typing import Any, Dict

from agents.prompts import load_prompt
from core.forcing_availability import render_forcing_facts
from agents.tool_loop import ToolLoopAgent
from core import data_gather as gather

# What the LLM may call. Deliberately tiny.
#
# Everything else reception fetches — the DEM grid, the water table, the three
# observation sets — is fetched by CODE once the domain and period are fixed
# (see reception_gather). Those are not decisions, so they should not cost an
# LLM round each: with twelve tools the loop spent 100-250 s of a 167-315 s
# reception choosing tools, and the model then summarised results truncated to
# 12,000 characters, which is how it came to report 25 stream gauges where the
# validator found 93.
#
# The model's job is the part that needs judgement: what is being asked, where,
# and when. Soil is absent entirely — it comes from the CONUS 1 km surface
# dataset at the donor gridcell, which the warm start already subsets.
DEFAULT_ALLOWLIST = {
    "terrain__resolve_watershed",
    "terrain__get_elevation",        # fallback: a point or town, no HUC given
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
        # Forcing years are read off disk at construction, not asserted in the
        # prompt text: the window is the one hard constraint on a request, and
        # a hardcoded sentence had already drifted 5 years from the filesystem.
        self.system = load_prompt("reception_agentic",
                                  forcing_facts=render_forcing_facts())
        self._clients = mcp_clients or {}
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
        """LLM loop, then the deterministic gather. Returns the reception package.

        Two phases, and the split is the point:

          LLM     what is being asked, WHERE (domain) and WHEN (period).
                  Judgement, and the only part that can need a question.
          CODE    the DEM grid, the water table, and the three observation
                  sets for that period. Not decisions, so not the model's.

        Returns {route, brief, observations, grid, provenance, trace, rounds,
        raw} — `route` carries the framework's dispatch (design / clarify /
        analyze_existing) so the brief stays science.
        """
        msg = f"User request: {user_request}\n\n"
        if context:
            msg += ("CONVERSATION CONTEXT (a prior experiment this session):\n"
                    + json.dumps(context, indent=2) + "\n\n")
        msg += ("Resolve the domain and the period, then emit ONLY your final "
                "JSON. Do NOT attempt to fetch observations — they are "
                "collected for you once the period is fixed.")
        out = self.loop.run(self.system, msg)
        brief = self._parse(out["content"])

        pkg = {"route": self._route(brief), "brief": brief,
               "observations": {}, "grid": {}, "provenance": [],
               "trace": out["trace"], "rounds": out["rounds"],
               "raw": out["content"]}
        if pkg["route"]["action"] != "design":
            return pkg                       # nothing to gather for yet

        prov: list = []
        dom = (brief.get("domain") or {})
        bbox = dom.get("bbox") or {}
        period = ((brief.get("run_settings") or {}).get("resolved_period") or {})
        y0, y1 = period.get("yr_start"), period.get("yr_end")

        if bbox:
            pkg["grid"] = gather.gather_grid(
                self._clients, bbox, huc=str(dom.get("huc") or ""),
                boundary=dom.get("boundary"), provenance=prov)
            # heterogeneity is DERIVED from the grid, not written by the model:
            # the planner stratifies on it, so it must be the same numbers the
            # sampler will see.
            brief.setdefault("heterogeneity", {})
            brief["heterogeneity"]["relief_m"] = pkg["grid"].get("relief_m")
            brief["heterogeneity"]["elevation_min_m"] = pkg["grid"].get("elevation_min_m")
            brief["heterogeneity"]["elevation_max_m"] = pkg["grid"].get("elevation_max_m")
            brief["heterogeneity"]["n_grid_points"] = pkg["grid"].get("n_in_basin")

            if isinstance(y0, int) and isinstance(y1, int):
                bs = (f'{bbox["min_lon"]},{bbox["min_lat"]},'
                      f'{bbox["max_lon"]},{bbox["max_lat"]}')
                pkg["observations"] = gather.gather_observations(
                    self._clients, bs, y0, y1, provenance=prov)
                brief["observations_summary"] = gather.summarise(pkg["observations"])
        pkg["provenance"] = prov
        return pkg

    @staticmethod
    def _route(brief: Dict[str, Any]) -> Dict[str, Any]:
        """Framework dispatch, kept OUT of the science brief.

        `intent` used to live inside the brief, where it was the only key the
        planner prompt never mentions — it is routing, not science.
        """
        raw = brief.get("intent", "parse_error")
        action = {"design": "design",
                  "analyze_existing": "analyze_existing"}.get(raw, "clarify")
        return {"action": action,
                "questions": list(brief.get("questions") or []),
                "prior_run_dir": brief.get("run_dir"),
                "llm_intent": raw}

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
