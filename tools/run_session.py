#!/usr/bin/env python3
"""
Stateful multi-turn design session (DRY / design-only).

Designs experiments step by step, carrying each turn's design forward as
context so a follow-up can build on "the previous experiment" — including a
CROSS-MODEL follow-up like "now design a PFLOTRAN run using that ELM recharge".
Reception + planner reason; nothing executes.

Run from the project root (module load pytorch/2.8.0):
    python3 tools/run_session.py \
      "Design an ELM experiment for the Naches sub-watershed (HUC8 17030002) to study recharge partitioning" \
      --then "Now design a PFLOTRAN run that uses the recharge / sub-surface drainage from that ELM experiment, and say what comparing the two would reveal"
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
from core.mcp_manager import MCPManager
from agents.reception_llm import LLMReceptionAgent
from agents.llm_agent import SimpleLLMClient
from agents.prompts import load_prompt

DEFAULT_MODEL = "claude-opus-4-8-project"


def _parse_json(text):
    t = re.sub(r"```json\s*", "", text)
    t = re.sub(r"```\s*$", "", t).strip()
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            return None
    return None


def run_planner(brief, request, model, max_tokens=8000):
    system = load_prompt("planner_capability_probe")
    user = (f"SCIENTIFIC QUESTION:\n{request}\n\nDOMAIN BRIEF (from reception):\n"
            f"{json.dumps(brief, indent=2)}\n\nThink step by step in prose first, "
            "then output the JSON plan as specified.")
    llm = SimpleLLMClient(model=model)
    r = llm.client.chat.completions.create(
        model=llm.model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def context_from(request, brief, plan):
    """Compact summary of a completed turn, to seed the next turn."""
    d = brief.get("domain") or {}
    mc = (plan or {}).get("model_choice") or {}
    return {
        "prior_request": request,
        "prior_model": mc.get("primary_model", "ELM"),
        "prior_domain": {"name": d.get("name"), "huc": d.get("huc"),
                         "area_km2": d.get("area_km2")},
        "prior_outputs": ["sub-surface drainage / recharge flux (ELM QDRAI)",
                          "per-layer soil moisture", "water-table depth",
                          "ET, surface runoff"],
        "prior_design": (plan or {}).get("sampling_strategy", {}).get("approach"),
    }


def show(plan):
    if not plan:
        print("  (no parseable plan — see the .md transcript)")
        return
    print(f"  archetype: {(plan.get('model_choice') or {}).get('design_archetype', '?')}")
    cd = plan.get("coupling_design")
    if cd:
        print("  COUPLING DESIGN:")
        for k in ("from_model", "to_model", "driver", "temporal_mapping",
                  "receiving_setup", "compare"):
            if cd.get(k):
                print(f"    {k}: {cd[k]}")
    samp = plan.get("sampling_strategy") or {}
    if samp.get("n_exploratory"):
        print(f"  sampling: N={samp.get('n_exploratory')} ({(samp.get('approach') or '')[:60]})")
    reqs = plan.get("requires_capabilities") or []
    if reqs:
        print("  needs:", [r.get("capability") if isinstance(r, dict) else r
                           for r in reqs])


def main():
    ap = argparse.ArgumentParser(description="Stateful multi-turn design session (dry)")
    ap.add_argument("request", nargs="+", help="turn 1 request")
    ap.add_argument("--then", action="append", default=[],
                    help="a follow-up turn (repeatable)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quiet", action="store_true", help="hide the live tool trace")
    args = ap.parse_args()
    turns = [" ".join(args.request)] + args.then

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("workflow_outputs") / f"session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STATEFUL SESSION  —  step-by-step design (dry; nothing executed)")
    print(f"out: {out_dir}")
    print("=" * 72)

    clients = MCPManager("mcp_config.json").get_all_clients()
    rx = LLMReceptionAgent(args.model, clients, verbose=not args.quiet)

    context = None
    for i, req in enumerate(turns, 1):
        print("\n" + "=" * 72)
        print(f"TURN {i}: {req}")
        print("=" * 72)
        rec = rx.process(req, context=context)
        brief = rec["brief"]
        (out_dir / f"turn{i}_brief.json").write_text(json.dumps(brief, indent=2))
        print(f"reception: intent={brief.get('intent')} "
              f"archetype={brief.get('design_archetype', '-')} "
              f"({rec['rounds']} rounds, {len(rec['trace'])} tool calls)")
        if brief.get("coupling"):
            print("  coupling:", json.dumps(brief["coupling"])[:220])
        if brief.get("intent") != "design":
            print("  (not a design turn — stopping carry-forward)")
            continue
        raw = run_planner(brief, req, args.model)
        (out_dir / f"turn{i}_plan.md").write_text(raw)
        plan = _parse_json(raw)
        if plan:
            (out_dir / f"turn{i}_plan.json").write_text(json.dumps(plan, indent=2))
        show(plan)
        context = context_from(req, brief, plan)        # carry forward

    print(f"\nSession complete. Transcripts in: {out_dir}\n")


if __name__ == "__main__":
    main()
