#!/usr/bin/env python3
"""
Pre-registered agent evaluation — RUNNER.

Tests whether the framework's architecture prevents failures a capable
general LLM commits, on a frozen prompt suite (eval/prompt_suite.json).
All arms use the SAME model, temperature=0, seed=1995; only the
*constraints* differ:

  A0_naive        question only -> asked for a complete runnable plan
                  (columns with coordinates + ELM config) from parametric
                  knowledge. No data context.
  A1_informed     same ask, but WITH the same brief the framework gets
                  (domain bbox, obs availability, forcing years) — the
                  "well-equipped generalist" proxy. Isolates: does data
                  context alone prevent grounding failures?
  A2_no_boundary  ablation: the capability prompt with the anti-
                  hallucination boundary REMOVED (told to enumerate concrete
                  columns itself). Isolates the boundary's contribution.
  A3_no_limits    ablation: the capability prompt with the KNOWN EXECUTION
                  LIMITS block REMOVED. Isolates whether feasibility honesty
                  comes from the limits inventory or from world knowledge.
  A4_framework    the production capability-aware PlannerAgent (strategy
                  only; coordinates are the deterministic expander's job).

Raw outputs -> eval/results/raw/{prompt}__{arm}__rep{k}.json. Scoring is a
separate offline step (score_eval.py). Never edit the suite after main-wave
data exists.

    python3 eval/run_eval.py --pilot            # plumbing check (excluded prompts)
    python3 eval/run_eval.py --main             # the pre-registered wave
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.llm_agent import LLMAgent                      # noqa: E402
from agents.planner_agent import PlannerAgent              # noqa: E402
from agents.prompts import load_prompt                     # noqa: E402

MODEL = "claude-sonnet-4-5-20250929-v1-project"            # same for ALL arms
TEMPERATURE, SEED = 0.0, 1995
DET_REPS = 3                                               # reps on determinism prompts
DET_ARMS = ("A1_informed", "A4_framework")

RUNNABLE_ASK = """
Output STRICT JSON with exactly these keys:
{
  "feasibility": {"verdict": "...", "answerable": [...], "not_answerable": [...]},
  "n_columns": <int>, "n_justification": "...",
  "columns": [{"id": "...", "lat": <float>, "lon": <float>, "soil_texture": "..."}, ...],
  "elm_config": {"DATM_CLMNCEP_YR_START": "...", "DATM_CLMNCEP_YR_END": "...",
                  "STOP_N": "...", "RUN_STARTDATE": "YYYY-MM-DD",
                  "hist_variables": ["...", ...]},
  "requires_capabilities": [...], "assumptions": [...]
}
Choose real, appropriate column locations and a correct, runnable ELM
configuration yourself. JSON only — no prose outside the JSON."""

NAIVE_SYSTEM = (
    "You are a senior land-surface hydrologist configuring single-column ELM "
    "(E3SM Land Model) simulation campaigns, with PFLOTRAN available for "
    "subsurface questions. Design a complete, runnable experiment for the "
    "user's question." + RUNNABLE_ASK)


def _capability_prompt() -> str:
    return load_prompt("planner_capability_probe")


def a2_prompt() -> str:
    """Capability prompt with the anti-hallucination boundary removed."""
    s = _capability_prompt()
    m1 = ("strategy into concrete points by sampling real DEM / soil rasters.\n"
          "    So you DO NOT invent coordinates and you DO NOT enumerate columns.")
    assert m1 in s, "A2 surgery marker missing — prompt drifted"
    s = s.replace(m1, "strategy into concrete points.\n"
                      "    You SHOULD choose and enumerate the concrete columns "
                      "yourself: include sampling_plan.columns = a list of "
                      "{id, lat, lon} you select, plus elm_config "
                      "{DATM_CLMNCEP_YR_START, DATM_CLMNCEP_YR_END, STOP_N, "
                      "RUN_STARTDATE, hist_variables}.")
    m2 = ("PLACEMENT RULES. A downstream deterministic expander turns your\n"
          "    strategy into concrete points")
    if m2 in s:
        s = s.replace(m2, "PLACEMENT RULES, and you also place the points\n"
                          "    yourself, turning your strategy into concrete points")
    return s


def a3_prompt() -> str:
    """Capability prompt with the KNOWN EXECUTION LIMITS block removed."""
    s = _capability_prompt()
    start = s.find("WHAT THE FRAMEWORK CANNOT")
    end = s.find("DESIGN ARCHETYPE — READ FIRST")
    assert start > 0 and end > start, "A3 surgery markers missing — prompt drifted"
    # cut back to the ━ header line that precedes each marker
    start = s.rfind("━", 0, start)
    start = s.rfind("\n", 0, start) + 1
    end = s.rfind("━", 0, end)
    end = s.rfind("\n", 0, end) + 1
    return s[:start] + s[end:]


def make_brief(p: dict) -> dict:
    b = dict(p.get("brief") or {})
    b["user_request"] = p["question"]
    return b


def call_arm(arm: str, p: dict) -> dict:
    """Run one (arm, prompt) -> {raw, parsed|None, model_reported}."""
    brief = make_brief(p)
    if arm == "A4_framework":
        agent = PlannerAgent(model=MODEL, model_type="elm", capability_aware=True)
        agent.llm.temperature, agent.llm.seed = TEMPERATURE, SEED
        plan = agent.create_plan(brief)
        return {"raw": json.dumps(plan), "parsed": plan,
                "model_reported": agent.llm.last_response_model}

    if arm == "A0_naive":
        system, user = NAIVE_SYSTEM, p["question"]
    elif arm == "A1_informed":
        system = NAIVE_SYSTEM
        user = (f"{p['question']}\n\nContext brief (real data availability):\n"
                f"{json.dumps(brief, indent=2)}")
    elif arm == "A2_no_boundary":
        system = a2_prompt()
        user = (f"Design the experiment for this brief. Return JSON only.\n\n"
                f"{json.dumps(brief, indent=2)}")
    elif arm == "A3_no_limits":
        system = a3_prompt()
        user = (f"Design a simulation STRATEGY for this brief. Emit a feasibility "
                f"verdict (answerable vs not-answerable), a stratified sampling "
                f"strategy as RULES (bands + justified N, NO coordinates), a "
                f"requires_capabilities backlog, and recorded assumptions. "
                f"Return JSON only.\n\n{json.dumps(brief, indent=2)}")
    else:
        raise ValueError(arm)

    agent = LLMAgent(f"eval_{arm}", system, MODEL)
    agent.llm.temperature, agent.llm.seed = TEMPERATURE, SEED
    raw = agent.ask_with_system(user_message=user, system_message=system)
    try:
        parsed = agent.parse_json(raw)
    except Exception:
        parsed = None
    return {"raw": raw, "parsed": parsed,
            "model_reported": agent.llm.last_response_model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="pilot prompts only")
    ap.add_argument("--main", action="store_true", help="main (non-pilot) prompts")
    ap.add_argument("--arms", default="A0_naive,A1_informed,A2_no_boundary,"
                                      "A3_no_limits,A4_framework")
    ap.add_argument("--prompts", default=None, help="comma list of prompt ids")
    args = ap.parse_args()

    suite = json.loads((ROOT / "eval" / "prompt_suite.json").read_text())
    suite_sha = hashlib.sha256(
        (ROOT / "eval" / "prompt_suite.json").read_bytes()).hexdigest()[:16]
    prompts = suite["prompts"]
    if args.pilot:
        prompts = [p for p in prompts if p.get("pilot")]
    elif args.main:
        prompts = [p for p in prompts if not p.get("pilot")]
    if args.prompts:
        want = set(args.prompts.split(","))
        prompts = [p for p in prompts if p["id"] in want]
    arms = args.arms.split(",")

    out = ROOT / "eval" / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    n_done = n_err = 0
    for p in prompts:
        reps = DET_REPS if p.get("determinism") else 1
        for arm in arms:
            for rep in range(1, (reps if arm in DET_ARMS else 1) + 1):
                dst = out / f"{p['id']}__{arm}__rep{rep}.json"
                if dst.exists():
                    print(f"  skip (exists): {dst.name}")
                    continue
                rec = {"prompt_id": p["id"], "arm": arm, "rep": rep,
                       "suite_sha": suite_sha, "model_requested": MODEL,
                       "temperature": TEMPERATURE, "seed": SEED}
                try:
                    rec.update(call_arm(arm, p))
                    ok = rec["parsed"] is not None
                    print(f"  ✓ {p['id']} {arm} rep{rep}"
                          f"{'' if ok else '  (JSON parse failed)'}")
                    n_done += 1
                except Exception as e:
                    rec["error"] = f"{e}\n{traceback.format_exc()[-800:]}"
                    print(f"  ✗ {p['id']} {arm} rep{rep}: {str(e)[:90]}")
                    n_err += 1
                dst.write_text(json.dumps(rec, indent=2))
    print(f"\n{n_done} calls completed, {n_err} errors -> {out}")


if __name__ == "__main__":
    main()
