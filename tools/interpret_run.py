#!/usr/bin/env python3
"""
LLM interpretation of a completed study — the agentic analyzer step.

The deterministic analyzer produces numbers (hydro_summary.json) and the
validator produces observation comparisons (validation.json); this step hands
BOTH, plus the original question and the planner's design, to the LLM and asks
for a scientist's interpretation: what the partitioning story is, what drives
it, whether the water balance makes sense, what the validation verdicts mean,
and what to run next. Writes 04_analysis/interpretation.md (picked up by the
story deck).

    module load pytorch/2.8.0            # needs PNNL_API_KEY
    python3 tools/interpret_run.py --run-dir <dir> [--model claude-opus-4-8-project]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

SYSTEM = """You are the analysis agent of a hydrologic simulation framework
(single-column ELM ensembles on real watershed data). You are given the study's
question, the planner's design + feasibility verdict, the per-column results
(including water-budget terms where present), and the observation-validation
verdicts. Write the scientific interpretation a careful hydrologist would.

Rules:
- Reason from the NUMBERS given; cite them. Never invent values.
- Separate what the experiment DEMONSTRATES from what it cannot (respect the
  feasibility verdict and validation caveats: coarse uniform forcing, 1-yr
  no-spin-up, no routing, single-column physics).
- If water_budget terms exist, tell the budget story per regime: where does the
  precipitation go (runoff / infiltration / ET / recharge / storage), and does
  the closure residual look acceptable?
- Interpret correlations mechanistically (WHY clay impedes recharge), not just
  numerically. Call out leverage points / replicates when N is small.
- End with 3-5 concrete next experiments, each tied to a limitation you found.

Output MARKDOWN with sections: ## Key findings (numbered, one line each),
## Water-balance story, ## What the validation says, ## Caveats,
## Recommended next experiments. Keep it under ~500 words, dense and specific."""


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def compact_results(hs):
    """Boil hydro_summary down to what the LLM needs (drop timeseries bulk)."""
    cols = []
    for r in hs.get("experiments", []):
        if r.get("status") != "ok":
            continue
        m = r["metrics"]
        cols.append({"column": r["case_name"], "elevation_m": r.get("elevation_m"),
                     "soil": r.get("soil"), **m})
    return {"columns": cols,
            "spatial_summary": hs.get("spatial_summary"),
            "soil_attribution": {k: v for k, v in (hs.get("soil_attribution") or {}).items()
                                 if k != "by_recharge"}}


def main():
    ap = argparse.ArgumentParser(description="LLM interpretation of a completed study")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="claude-opus-4-8-project")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    hs = load(rd / "04_analysis" / "hydro_summary.json")
    if not hs:
        sys.exit("run analyze_run.py first (no hydro_summary.json)")

    brief = load(rd / "reception_brief.json")
    plan = load(rd / "plan.json")
    val = load(rd / "04_analysis" / "validation.json")
    payload = {
        "question": brief.get("user_request") or "(controlled experiment — see design)",
        "domain": brief.get("domain"),
        "design": {"goals": (plan.get("scientific_decomposition") or {}).get("goals"),
                   "sampling": plan.get("sampling_strategy"),
                   "feasibility": plan.get("feasibility")},
        "results": compact_results(hs),
        "validation": {"targets": val.get("targets"),
                       "hydrograph_metrics": {k: v for k, v in (val.get("hydrograph") or {}).items()
                                              if k not in ("days", "obs", "mod")}},
    }

    from agents.llm_agent import SimpleLLMClient
    llm = SimpleLLMClient(model=args.model)
    print(f"interpreting {rd.name} with {args.model} …")
    resp = llm.client.chat.completions.create(
        model=llm.model, max_tokens=2000,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(payload, indent=1, default=str)}])
    md = resp.choices[0].message.content or ""

    out = rd / "04_analysis" / "interpretation.md"
    out.write_text(md)
    print("\n" + md)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
