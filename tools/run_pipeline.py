#!/usr/bin/env python3
"""
Full DRY pipeline:  user request -> agentic reception (MCP tools) -> planner -> plan.

Reception is a pure-LLM agent that drives the MCP tools itself to frame a brief;
the planner designs the experiment (conceptual or site archetype). NOTHING is
executed — no experiment manager, no SLURM. Transcripts (reception trace + brief
+ plan) are saved under workflow_outputs/pipeline_<ts>/.

Run from the project root with the MCP runtime env:
    module load pytorch/2.8.0
    python3 tools/run_pipeline.py "explore GW / soil-moisture partitioning in the \
        Naches sub-watershed using ELM, validate with observations"
    python3 tools/run_pipeline.py --reception-model gemini-2.5-flash-project "..."
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
from agents.validate import check_brief, check_plan

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
    user = (f"SCIENTIFIC QUESTION:\n{request}\n\n"
            f"DOMAIN BRIEF (from reception):\n{json.dumps(brief, indent=2)}\n\n"
            "Think step by step in prose first, then output the JSON plan as "
            "specified. Strategy altitude only — do not enumerate per-column "
            "configs and do not invent coordinates.")
    llm = SimpleLLMClient(model=model)
    resp = llm.client.chat.completions.create(
        model=llm.model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content or ""


def print_plan(plan):
    print("\n" + "=" * 72)
    print("PLAN SUMMARY   (full reasoning saved to transcript)")
    print("=" * 72)
    arche = (plan.get("model_choice") or {}).get("design_archetype", "?")
    print(f"archetype: {arche}")
    for g in (plan.get("scientific_decomposition") or {}).get("goals", []):
        print(f"  goal: {g}")

    samp = plan.get("sampling_strategy", {}) or {}
    rows = plan.get("sampling_plan") or []
    if rows:
        label = "treatments" if arche == "conceptual" else "exploratory columns"
        print(f"\nSAMPLING — {samp.get('n_exploratory', len(rows))} {label} "
              f"({samp.get('approach', '')}):")
        print(f"  {'n':>2}  {'group':<32}  why")
        print(f"  {'--':>2}  {'-' * 32}  {'-' * 24}")
        for r in rows:
            if not isinstance(r, dict):
                print(f"      - {r}"); continue
            n = str(r.get("n", r.get("count", "?")))
            grp = str(r.get("group", r.get("label", "")))[:32]
            print(f"  {n:>2}  {grp:<32}  {r.get('reason', r.get('why', ''))}")

    vd = plan.get("validation_design", []) or []
    if vd:
        print("\nVALIDATION:")
        for v in vd:
            here = "in-domain" if v.get("in_domain_available") else "NOT in-domain"
            print(f"  - {v.get('target_variable')}: {v.get('observation_source')} "
                  f"[{here}] — {v.get('comparison', '')}")

    summ = plan.get("experiment_summary", {}) or {}
    if summ:
        print(f"\nTOTAL: {summ.get('total_columns')} columns "
              f"({summ.get('exploratory')} exploratory + "
              f"{summ.get('validation')} validation)")

    reqs = plan.get("requires_capabilities", []) or []
    if reqs:
        print(f"\nNEEDS TO RUN ({len(reqs)}):")
        for r in reqs:
            print(f"  - {r.get('capability') if isinstance(r, dict) else r}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Dry pipeline: request -> reception -> planner")
    ap.add_argument("request", nargs="+", help="the user request (natural language)")
    ap.add_argument("--reception-model", default=DEFAULT_MODEL)
    ap.add_argument("--planner-model", default=DEFAULT_MODEL)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--quiet", action="store_true", help="hide the live tool trace")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="let reception ask clarifying questions on ambiguity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    request = " ".join(args.request)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("workflow_outputs") / f"pipeline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DRY PIPELINE  —  request -> reception (MCP tools) -> planner  (no execution)")
    print("=" * 72)
    print(f"request: {request}")
    print(f"out    : {out_dir}\n")

    clients = MCPManager("mcp_config.json").get_all_clients()

    # ── Reception (agentic) ──────────────────────────────────────────────
    print("-" * 72)
    print("RECEPTION  (agentic — driving MCP tools)")
    print("-" * 72)
    rx = LLMReceptionAgent(args.reception_model, clients, verbose=not args.quiet,
                           max_rounds=args.max_rounds, interactive=args.interactive)
    rec = rx.process(request)
    brief = rec["brief"]
    (out_dir / "reception_brief.json").write_text(json.dumps(brief, indent=2))
    (out_dir / "reception_trace.json").write_text(json.dumps(rec["trace"], indent=2))
    print(f"\nreception: intent={brief.get('intent')} "
          f"archetype={brief.get('design_archetype', '-')} "
          f"({rec['rounds']} rounds, {len(rec['trace'])} tool calls)")
    for w in check_brief(brief):
        print(f"  ⚠️  brief check: {w}")

    intent = brief.get("intent")
    if intent == "clarification_needed":
        print("\nNEEDS CLARIFICATION:")
        for q in brief.get("questions", []):
            print(f"  - {q}")
        print(f"\nSaved to {out_dir}. Nothing executed.")
        return
    if intent != "design":
        print(f"\nintent '{intent}' — not a design request; stopping. "
              f"Brief saved to {out_dir}.")
        return

    # ── Planner ──────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("PLANNER  (designing experiment)")
    print("-" * 72)
    raw = run_planner(brief, request, args.planner_model)
    (out_dir / "plan_raw.md").write_text(raw)
    plan = _parse_json(raw)
    if plan is None:
        print("[pipeline] planner produced no parseable JSON — see plan_raw.md")
    else:
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        for w in check_plan(plan, brief):
            print(f"  ⚠️  plan check: {w}")
        print_plan(plan)

    print(f"\nDry pipeline complete. Transcripts in: {out_dir}\n")


if __name__ == "__main__":
    main()
