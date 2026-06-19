#!/usr/bin/env python3
"""
Planner capability probe — DRY RUN, planner only.

Tests whether the reception->planner reasoning can produce a SENSIBLE plan
for complex, multi-clause scientific questions (spatial sampling + observation
validation) BEFORE any downstream model development.

Pipeline:
    hand-written domain brief  ->  capability-aware planner (1 open LLM call)
    ->  prints REASONING + structured plan + requires_capabilities  ->  STOP.

What it NEVER does:
    It never imports or calls any experiment manager, builder, wrapper, or
    SLURM. Nothing is executed. The existing pipeline and its 36 guardrail
    tests are untouched. Safe and cheap.

USAGE (run from project root):
    python3 tools/probe_planner.py
    python3 tools/probe_planner.py --brief tools/naches_elm_brief.json
    python3 tools/probe_planner.py --once          # single shot, no loop
    python3 tools/probe_planner.py --model claude-opus-4-7-project

Iterate: after each plan, type a critique to refine it, or 'done' to stop.
Transcripts are saved under workflow_outputs/probe_<timestamp>/.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
from agents.llm_agent import LLMAgent
from agents.prompts import load_prompt

CAPABILITY_PROMPT = "planner_capability_probe"
DEFAULT_BRIEF = "tools/naches_elm_brief.json"
DEFAULT_MODEL = "claude-opus-4-8-project"


def build_initial_message(brief: dict) -> str:
    question = (brief.get("user_request")
                or brief.get("request")
                or "(no user_request field in brief)")
    return (
        f"SCIENTIFIC QUESTION:\n{question}\n\n"
        f"DOMAIN BRIEF (from reception):\n"
        f"{json.dumps(brief, indent=2)}\n\n"
        "Design the sampling + validation strategy for this question. "
        "Think step by step in prose first, then output the JSON plan as "
        "specified. Strategy altitude only — do not enumerate per-column "
        "configs and do not invent coordinates."
    )


def show_response(raw: str, agent: LLMAgent) -> None:
    """Print a tight, scannable SUMMARY (not the reasoning).

    The full reasoning + plan is saved to the transcript by the caller;
    the console shows only the decision: what gets sampled and why.
    """
    try:
        plan = agent.parse_json(raw)
    except Exception:
        # No clean JSON — fall back to the raw text so nothing is lost.
        print("\n[probe] (no parseable JSON block — showing raw response)\n")
        print(raw)
        return

    print("\n" + "=" * 72)
    print("PLAN SUMMARY   (full reasoning saved to transcript)")
    print("=" * 72)

    for goal in plan.get("scientific_decomposition", {}).get("goals", []):
        print(f"  goal: {goal}")

    samp = plan.get("sampling_strategy", {})
    rows = plan.get("sampling_plan") or []
    if rows:
        print(f"\nSAMPLING — {samp.get('n_exploratory', '?')} exploratory "
              f"columns ({samp.get('approach', '')}):")
        print(f"  {'n':>2}  {'group':<32}  why")
        print(f"  {'--':>2}  {'-' * 32}  {'-' * 24}")
        for r in rows:
            if not isinstance(r, dict):
                print(f"      - {r}")
                continue
            n = str(r.get("n", r.get("count", "?")))
            grp = str(r.get("group", r.get("label", "")))[:32]
            why = str(r.get("reason", r.get("why", "")))
            print(f"  {n:>2}  {grp:<32}  {why}")
    elif samp:
        print(f"\nSAMPLING — N={samp.get('n_exploratory')} "
              f"({samp.get('approach')})")

    vd = plan.get("validation_design", [])
    if vd:
        print("\nVALIDATION:")
        for v in vd:
            here = ("in-domain" if v.get("in_domain_available")
                    else "NOT in-domain")
            print(f"  - {v.get('target_variable')}: "
                  f"{v.get('observation_source')} [{here}] — "
                  f"{v.get('comparison', '')}")

    summ = plan.get("experiment_summary", {})
    if summ:
        print(f"\nTOTAL: {summ.get('total_columns')} columns "
              f"({summ.get('exploratory')} exploratory + "
              f"{summ.get('validation')} validation)  |  "
              f"treatment: {summ.get('primary_treatment', 'n/a')}")

    reqs = plan.get("requires_capabilities", [])
    if reqs:
        print(f"\nNEEDS TO RUN ({len(reqs)}):")
        for r in reqs:
            print(f"  - {r.get('capability') if isinstance(r, dict) else r}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Planner capability probe (dry run — nothing executed)")
    parser.add_argument("--brief", default=DEFAULT_BRIEF,
                        help="Path to domain brief JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model id")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max completion tokens (avoids JSON truncation)")
    parser.add_argument("--once", action="store_true",
                        help="Single shot, no iterate loop")
    parser.add_argument("--out", default=None,
                        help="Output dir (default workflow_outputs/probe_<ts>)")
    args = parser.parse_args()

    brief_path = Path(args.brief)
    if not brief_path.exists():
        sys.exit(f"Brief not found: {brief_path} (run from project root)")
    brief = json.loads(brief_path.read_text())
    system_prompt = load_prompt(CAPABILITY_PROMPT)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out) if args.out
               else Path("workflow_outputs") / f"probe_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PLANNER CAPABILITY PROBE  —  DRY RUN (no execution, no SLURM)")
    print("=" * 72)
    print(f"  brief : {brief_path}")
    print(f"  model : {args.model}")
    print(f"  out   : {out_dir}")
    print("=" * 72)

    try:
        agent = LLMAgent("planner_probe", system_prompt, args.model)
    except Exception as e:
        # Most likely PNNL_API_KEY not set. Still dump the assembled request
        # so the brief + prompt can be inspected offline.
        req_file = out_dir / "request_preview.txt"
        req_file.write_text(
            "SYSTEM PROMPT:\n" + system_prompt + "\n\n"
            + "USER MESSAGE:\n" + build_initial_message(brief)
        )
        sys.exit(
            f"Could not init LLM client ({e}).\n"
            f"  Set PNNL_API_KEY to actually run the probe.\n"
            f"  The assembled request was written to {req_file} for "
            f"offline inspection (brief + prompt both load OK)."
        )

    # Drive the chat directly so we can set max_tokens (the shared
    # LLMAgent.respond does not expose it) while still keeping multi-turn
    # history for the iterate loop. llm_agent.py is left untouched.
    client = agent.llm.client
    history = []

    def call(user_msg: str) -> str:
        history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": system_prompt}] + history
        resp = client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=args.max_tokens)
        out = resp.choices[0].message.content
        history.append({"role": "assistant", "content": out})
        return out

    msg = build_initial_message(brief)
    it = 0
    while True:
        it += 1
        print(f"\n[probe] calling planner (iteration {it})...")
        try:
            raw = call(msg)
        except Exception as e:
            print(f"LLM call failed: {e}")
            break

        show_response(raw, agent)
        transcript = out_dir / f"iter_{it:02d}.md"
        transcript.write_text(
            f"# Probe iteration {it}\n\n"
            f"## Input\n\n{msg}\n\n## Response\n\n{raw}\n"
        )
        print(f"\n[probe] saved -> {transcript}")

        if args.once or not sys.stdin.isatty():
            print("\n[probe] single-shot mode — stopping. Nothing executed.")
            break

        print("\n" + "=" * 72)
        crit = input("Critique to refine (or 'done' to stop): ").strip()
        if crit.lower() in ("done", "quit", "exit", "q", ""):
            print("[probe] done. Nothing was executed.")
            break
        msg = (
            f"Revise the plan based on this feedback:\n{crit}\n\n"
            "Keep the same output format (prose reasoning, then JSON)."
        )

    print(f"\nProbe complete. Transcripts in: {out_dir}\n")


if __name__ == "__main__":
    main()
