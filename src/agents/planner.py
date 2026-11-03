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

    NEVER THE SOIL OR THE RAIN EITHER. reception.soil holds a horizon profile
    for every candidate column location (~2.5 kB a point, 150 kB for a basin's
    grid) and reception.precipitation holds a daily series for every one of
    them (121 kB for a single year, ~2.4 MB for twenty). The planner
    stratifies on ELEVATION and reads neither, so every byte of them here
    would be paid for and unused. They are excluded twice on purpose: both are
    siblings of `brief`, so building this dict by naming keys already leaves
    them out, and the pops below stop them reaching the prompt if either is
    ever written into the brief by mistake. The second guard is the one that
    matters — the first is a property of today's shape, not a rule anyone
    stated.
    """
    brief = dict(reception.get("brief") or {})
    brief.pop("observations_summary", None)
    brief.pop("soil", None)
    brief.pop("precipitation", None)
    brief.pop("subsurface", None)
    return {"brief": brief,
            "observations_summary": (reception.get("brief") or {}).get(
                "observations_summary") or {}}


# OBSERVED NUMBERS STAY AWAY FROM THE PLANNER (2026-11-03). The rulebook the
# server returns carries how each control compares with observations (its
# evaluation, its observation list, the not-in-the-run items with their
# observed numbers, the observed burned area). The design page shows the
# scientist all of it, so strategy.json keeps the whole rulebook; but the
# planner's caveats go on to the Analyzer, which reads the model's output
# only, and a caveat copied from the text the planner was shown would carry
# an observed number there. So the planner's copy has none of it, and keeps
# `not_in_it_words`, the same items without their numbers.
_CONTROL_KEYS_NOT_FOR_PLANNER = ("evaluation", "observations", "not_in_it")
_FIRE_KEYS_NOT_FOR_PLANNER = ("observed", "observed_source")


def _rulebook_for_prompt(rulebook: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The rulebook as the planner is shown it: every control without its
    comparison with observations. A copy; the rulebook given is not changed."""
    if not rulebook:
        return rulebook
    out = dict(rulebook)
    ctl = rulebook.get("controls")
    if isinstance(ctl, dict) and isinstance(ctl.get("controls"), list):
        rows = []
        for c in ctl["controls"]:
            if not isinstance(c, dict):
                rows.append(c)
                continue
            c = {k: v for k, v in c.items() if k not in _CONTROL_KEYS_NOT_FOR_PLANNER}
            facts = c.get("facts")
            if isinstance(facts, dict) and isinstance(facts.get("burned_area_km2"), dict):
                facts = dict(facts)
                facts["burned_area_km2"] = {
                    k: v for k, v in facts["burned_area_km2"].items()
                    if k not in _FIRE_KEYS_NOT_FOR_PLANNER}
                c["facts"] = facts
            rows.append(c)
        out["controls"] = dict(ctl, controls=rows)
    return out


class Planner:
    """reception.json -> planner.json."""

    def __init__(self, model: str = "claude-opus-4-8-project",
                 pinning: Optional[Dict[str, Any]] = None,
                 constraints: Optional[Dict[str, Any]] = None):
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
        # THE OTHER HALF OF THE SAME QUESTION, set by the caller after
        # reception has named a model — `pinning` is what a column may be
        # COMPARED against, `constraints` is what a plan may CHOOSE. Both come
        # from the chosen model's own server; neither is stated in planner.txt
        # any more, because both were ELM's answers written as the framework's.
        self.constraints = constraints
        # A WHOLE-DOMAIN STUDY'S RULEBOOK, set by the caller for a whole_domain
        # brief: the model server's prepared controls (with facts measured
        # from their output) and its change list. Like the two above, the
        # server's answers, never stated in planner.txt.
        self.domain_rulebook: Optional[Dict[str, Any]] = None

    # THE SERVER'S FINDINGS COME BACK HERE (2026-10-28). The model server
    # checks a design before the design review now, and a design it refuses
    # (a run named after an earlier study's case, a field with soil levels, a
    # region reaching outside the one a change applies in) is handed back with
    # its own words, once or twice, before anyone reads the page. Only what
    # each finding needs may change: the design the person agreed with
    # Reception stays.
    FINDINGS_NOTE = (
        "THE MODEL SERVER CHECKED THE DESIGN YOU EMITTED AND REFUSED IT. Its "
        "findings, in its own words, are below, after the design it refused. "
        "Emit the whole design again, changed only as each finding requires, "
        "and keep everything else as it was: the question, the region, the "
        "years, the runs' changes and the analysis the person agreed to. Never "
        "drop a run or a region to make a finding go away when a finding names "
        "the change that fixes it.")

    def plan(self, reception: Dict[str, Any],
             findings: Optional[list] = None,
             previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = _summary_for_prompt(reception)
        system = self.system
        if self.constraints:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nDESIGN LIMITS — what this model's studies "
                       "cannot vary, from the\nserver that will run it. This "
                       "is the fence referred to above.\n━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       + json.dumps(self.constraints, indent=1, default=str))
        if self.domain_rulebook:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nWHOLE-DOMAIN RULEBOOK: the prepared controls and "
                       "the change list,\nfrom the server that will run this "
                       "study\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━\n"
                       + json.dumps(_rulebook_for_prompt(self.domain_rulebook),
                                    indent=1, default=str))
        if self.pinning:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nPINNING RULES — from the model server that "
                       "will run this study\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━━━━━━━━━━━━━\n"
                       + json.dumps(self.pinning, indent=1, default=str))
        content = json.dumps(payload, indent=1, default=str)
        if findings:
            # the refused design without the blocks this code adds after the
            # planner (they come back from the server's own answers)
            prior = {k: v for k, v in (previous or {}).items()
                     if k not in ("pinning", "domain_rulebook")}
            content += ("\n\n" + self.FINDINGS_NOTE
                        + "\n\nTHE DESIGN IT REFUSED:\n"
                        + json.dumps(prior, indent=1, default=str)
                        + "\n\nTHE SERVER'S FINDINGS:\n"
                        + json.dumps(list(findings), indent=1, default=str))
        raw = self.llm.ask(
            [{"role": "user", "content": content + "\n\nEmit ONLY the JSON."}],
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
