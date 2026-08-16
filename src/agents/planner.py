#!/usr/bin/env python3
"""
The planner
src/agents/planner.py

One LLM call, one JSON, seven keys — every one of which something reads.

PURELY LLM. It proposes; nothing here corrects it. An earlier version
recomputed the column count and dropped station ids the planner had named,
which made this stage half deterministic and half not — and across fourteen
evaluation cases it corrected NOTHING: no invented station id, no column count
changed. A guard that never fires is complexity without evidence. The column
rule lives in the prompt, the planner applies it, and the justification makes a
wrong one visible. Anything worth checking is checked once, by the Experiment
Manager, at the point where compute starts being spent.

The old planner emitted ten top-level keys of which `assumptions`,
`open_questions` and `scientific_rationale` were read by nothing, and
`sampling_plan` was read only by a status line that reported a DIFFERENT column
count from the one the sampler used (a plan printed "N=4" and then "N=11", and
11 was what ran). Keys nobody consumes cannot be wrong in any way that gets
caught, so they drift.

What the pipeline actually needs from this stage:

    n_columns          expand_sampling — how many columns to materialise
    archetype          expand_sampling, elm_exp_manager — conceptual vs site
    coupling           elm_exp_manager — only for cross-model runs
    feasibility/goals  the analyzer's framing

plus `validation`, which is new and is the point: reception now carries station
COORDINATES, so the planner can name which stations to pin columns to, and the
sampler can place them there. Previously the planner wrote a validation design
that nothing read, and 0 of 14 Naches columns landed in the gauged catchment —
not bad luck, but because nothing ever tried.

The column count is arithmetic here, not judgement. Asked three times for the
same brief the old planner answered 10, 12 and 14, because the prompt requested
a justified integer and justification is not a constraint.
"""
import json
import re
from typing import Any, Dict, Optional

from agents.llm_agent import SimpleLLMClient
from agents.prompts import load_prompt

def _summary_for_prompt(reception: Dict[str, Any]) -> Dict[str, Any]:
    """What the planner is shown: the brief, plus the observation SUMMARY.

    Never the raw series. The summary carries station ids and coordinates,
    which is all the planner needs to pin validation columns — the measurements
    themselves are the analyzer's business and would be ~170 kB here.
    """
    brief = dict(reception.get("brief") or {})
    brief.pop("observations_summary", None)
    return {"brief": brief,
            "observations_summary": (reception.get("brief") or {}).get(
                "observations_summary") or {}}


class Planner:
    """reception.json -> planner.json."""

    def __init__(self, model: str = "claude-opus-4-8-project",
                 pinning: Optional[Dict[str, Any]] = None):
        """`pinning` is the model server's own `pinning` block.

        WHICH STATIONS MAY BE PINNED IS THE MODEL'S ANSWER, and it used to be
        frozen into planner.txt as prose about 1-D columns and lateral
        transport. That was true of ELM and would have been false of the next
        backend, which is a wrong plan rather than a missing one — the study
        still runs, and spends columns on comparisons that cannot be made.

        Passed in rather than fetched here because the caller already holds the
        MCP clients and knows which backend it is about to drive; the planner
        does not, and giving it a client to guess with would put the same
        knowledge back one layer down.
        """
        self.llm = SimpleLLMClient(model=model)
        self.system = load_prompt("planner")
        self.pinning = pinning

    def plan(self, reception: Dict[str, Any]) -> Dict[str, Any]:
        payload = _summary_for_prompt(reception)
        system = self.system
        if self.pinning:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nPINNING RULES — from the model server that "
                       "will run this study\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━━━━━━━━━━━━━\n"
                       + json.dumps(self.pinning, indent=1, default=str))
        raw = self.llm.ask(
            [{"role": "user",
              "content": json.dumps(payload, indent=1, default=str)
                         + "\n\nEmit ONLY the JSON."}],
            system_message=system)
        return self._parse(raw)

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json\s*", "", text or "")
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        a, b = cleaned.find("{"), cleaned.rfind("}")
        if a == -1 or b <= a:
            return {}
        try:
            return json.loads(cleaned[a:b + 1])
        except json.JSONDecodeError:
            return {}
