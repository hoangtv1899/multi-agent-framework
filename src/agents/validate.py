#!/usr/bin/env python3
"""
Lightweight deterministic checks on reception briefs + planner plans.

A safety net for the agentic (dry) pipeline: it REPORTS issues — it never
raises and never "fixes". Pure functions, no LLM, no network. The heavier
executability validation (date arithmetic, namelist vocabulary) belongs at the
strategy -> executable translation when execution is wired.
"""
from typing import Any, Dict, List

ARCHETYPES = {"conceptual", "site", "coupling"}
INTENTS = {"design", "clarification_needed", "analyze_existing"}


def check_brief(brief: Any) -> List[str]:
    """Return a list of issue strings for a reception brief ([] = clean)."""
    if not isinstance(brief, dict):
        return ["brief is not a JSON object"]
    intent = brief.get("intent")
    if intent == "parse_error":
        return [f"brief did not parse ({brief.get('error', '?')})"]
    issues: List[str] = []
    if intent not in INTENTS:
        issues.append(f"unknown intent '{intent}'")
    if intent != "design":
        return issues

    arche = brief.get("design_archetype")
    if arche not in ARCHETYPES:
        issues.append(f"unknown design_archetype '{arche}'")
    if arche == "site":
        bbox = (brief.get("domain") or {}).get("bbox") or {}
        if not all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
            issues.append("site brief missing a complete domain.bbox (expander needs it)")
    if arche == "coupling" and not brief.get("coupling"):
        issues.append("coupling brief missing the 'coupling' block")
    for k in ("observations_available", "observations_missing"):
        if k in brief and not isinstance(brief[k], list):
            issues.append(f"{k} should be a list")
    return issues


def check_plan(plan: Any, brief: Dict = None) -> List[str]:
    """Return a list of issue strings for a planner plan ([] = clean)."""
    if not isinstance(plan, dict):
        return ["plan did not parse to a JSON object"]
    issues: List[str] = []

    arche = (plan.get("model_choice") or {}).get("design_archetype")
    if arche not in ARCHETYPES:
        issues.append(f"model_choice.design_archetype invalid: '{arche}'")
    if plan.get("requires_capabilities") is not None \
            and not isinstance(plan["requires_capabilities"], list):
        issues.append("requires_capabilities should be a list")

    if arche == "coupling":
        if not plan.get("coupling_design"):
            issues.append("coupling plan missing 'coupling_design'")
    else:
        if not isinstance((plan.get("sampling_strategy") or {}).get("n_exploratory"), int):
            issues.append("sampling_strategy.n_exploratory is not an integer")
        rows = plan.get("sampling_plan")
        if not isinstance(rows, list) or not rows:
            issues.append("sampling_plan is missing or empty")
        else:
            for i, r in enumerate(rows):
                if not isinstance(r, dict) or "n" not in r or "group" not in r:
                    issues.append(f"sampling_plan[{i}] missing n/group")

    if isinstance(brief, dict):
        b_arche = brief.get("design_archetype")
        if b_arche and arche and b_arche != arche:
            issues.append(f"plan archetype '{arche}' != brief archetype '{b_arche}'")
    return issues
