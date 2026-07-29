#!/usr/bin/env python3
"""
The lean planner
src/agents/planner_lean.py

One LLM call, one JSON, seven keys — every one of which something reads.

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
from typing import Any, Dict, List

from agents.llm_agent import SimpleLLMClient
from agents.prompts import load_prompt

MIN_COLUMNS, MAX_COLUMNS = 6, 24
MAX_PINNED = 4


def bands_for_relief(relief_m) -> int:
    """Elevation bands from relief. Deterministic, so two runs agree."""
    try:
        r = float(relief_m)
    except (TypeError, ValueError):
        return 4
    return 3 if r < 500 else (4 if r < 1200 else 5)


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


class LeanPlanner:
    """reception.json -> planner.json."""

    def __init__(self, model: str = "claude-opus-4-8-project"):
        self.llm = SimpleLLMClient(model=model)
        self.system = load_prompt("planner_lean")

    def plan(self, reception: Dict[str, Any]) -> Dict[str, Any]:
        payload = _summary_for_prompt(reception)
        raw = self.llm.ask(
            [{"role": "user",
              "content": json.dumps(payload, indent=1, default=str)
                         + "\n\nEmit ONLY the JSON."}],
            system_message=self.system)
        plan = self._parse(raw)
        return self._enforce(plan, reception)

    # ── the deterministic half ──────────────────────────────────────────
    @staticmethod
    def _enforce(plan: Dict[str, Any], reception: Dict[str, Any]) -> Dict[str, Any]:
        """Recompute what must not be a judgement, and refuse invented stations.

        The LLM proposes; this decides. Two things are corrected rather than
        trusted:

          * the column arithmetic, so the same brief gives the same N twice;
          * the pinned station ids, checked against what was actually fetched.
            A station the planner names but reception never saw would send the
            sampler to a coordinate that does not exist.
        """
        brief = reception.get("brief") or {}
        het = brief.get("heterogeneity") or {}
        summ = brief.get("observations_summary") or {}
        notes: List[str] = []

        known = set()
        for kind, key in (("streamflow", "stations"), ("swe", "stations"),
                          ("water_table", "wells")):
            for st in ((summ.get(kind) or {}).get(key) or []):
                sid = st.get("id") or st.get("triplet")
                if sid:
                    known.add(str(sid))

        val = []
        for v in (plan.get("validation") or []):
            ids = [str(s) for s in (v.get("stations") or [])]
            keep = [s for s in ids if s in known]
            if len(keep) != len(ids):
                notes.append(f"dropped {len(ids) - len(keep)} station id(s) for "
                             f"{v.get('variable')} not present in the fetched "
                             f"observations")
            v = dict(v, stations=keep)
            if not keep:
                v["comparison"] = "unavailable"
            val.append(v)
        plan["validation"] = val

        n_pinned = min(sum(len(v.get("stations") or []) for v in val), MAX_PINNED)
        s = dict(plan.get("sampling") or {})
        n_bands = bands_for_relief(het.get("relief_m"))
        per_band = 3
        n_cols = max(MIN_COLUMNS, min(n_bands * per_band + n_pinned, MAX_COLUMNS))
        if s.get("n_columns") != n_cols:
            notes.append(f"column count recomputed from relief: "
                         f"{s.get('n_columns')} -> {n_cols}")
        s.update(n_bands=n_bands, per_band=per_band, n_validation=n_pinned,
                 n_columns=n_cols)
        s.setdefault("approach", "stratified by elevation band")
        s["justification"] = (f"relief {het.get('relief_m')} m -> {n_bands} bands "
                              f"x {per_band} = {n_bands * per_band}"
                              + (f", +{n_pinned} pinned to stations" if n_pinned else "")
                              + f" = {n_cols} columns")
        plan["sampling"] = s

        plan.setdefault("archetype", brief.get("design_archetype") or "site")
        plan.setdefault("feasibility", {"verdict": "partial", "why": "not stated"})
        plan.setdefault("goals", [])
        plan.setdefault("requires", [])
        plan.setdefault("coupling", None)
        if notes:
            plan["enforcement_notes"] = notes

        # Two aliases the Experiment Manager still reads. They are DERIVED, so
        # they cannot disagree with the lean keys above; they go when the
        # manager is rewired to read planner.json directly.
        plan["sampling_strategy"] = {"n_exploratory": n_cols,
                                     "approach": s["approach"]}
        plan["model_choice"] = {"design_archetype": plan["archetype"]}
        return plan

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
