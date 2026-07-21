#!/usr/bin/env python3
"""
Capability-growth demonstration: does the planner's feasibility verdict move
when a real new capability (PFLOTRAN reactive transport) enters its inventory?

Runs the SAME two questions against the production planner before and after
the reactive-transport entry is added to planner_capability_probe_v2:

  I01  frozen-suite prompt, expected infeasible in the pre-registered eval:
       nitrate from fertilized FIELDS to drinking-water WELLS. Needs LATERAL
       transport, which reactive chemistry does NOT provide, so a
       well-calibrated planner should NOT fully clear this one.
  R01  a 1-D column question reactive transport genuinely does enable:
       organic-matter respiration and N cycling within a soil column.

The pairing is the point. A capability inventory that is merely permissive
would clear both; a calibrated one clears only what the new tool actually
covers. The evaluation's frozen v0.1 prompt is untouched, so Section 4
results remain reproducible.

    python3 tools/reactive_capability_demo.py --tag before
    python3 tools/reactive_capability_demo.py --tag after
    python3 tools/reactive_capability_demo.py --compare
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "workflow_outputs" / "reactive_capability_demo"
MODEL = "claude-opus-4-8-project"

R01_BRIEF = {
    "intent": "design", "design_archetype": "conceptual",
    "user_request": (
        "How does organic-matter respiration control dissolved oxygen and "
        "inorganic nitrogen within a soil column under different recharge "
        "rates? Use a single vertical column, no lateral transport needed."),
    "domain": None,
    "heterogeneity": None,
    "observations_available": [],
    "observations_missing": [
        {"variable": "pore-water chemistry profiles",
         "reason": "no in-domain solute observations"}],
    "notes": ("Single-column vadose-zone biogeochemistry question. Treatments "
              "are recharge scenarios; no lateral or field-to-well transport "
              "is required."),
}


def briefs():
    suite = json.loads((ROOT / "eval" / "prompt_suite.json").read_text())
    ps = suite.get("prompts", suite) if isinstance(suite, dict) else suite
    i01 = next(p for p in ps if p["id"] == "I01")
    b = dict(i01.get("brief") or {})
    b["user_request"] = i01["question"]
    b["intent"] = "design"
    return {"I01_fields_to_wells": b, "R01_column_biogeochem": R01_BRIEF}


def run(tag):
    from agents.planner_agent import PlannerAgent
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, brief in briefs().items():
        a = PlannerAgent(model=MODEL, model_type="pflotran", capability_aware=True)
        a.llm.temperature, a.llm.max_tokens = 0.0, 8192
        try:
            plan = a.create_plan(brief)
        except Exception as e:
            print(f"  ! {name}: planner failed ({e})")
            continue
        fe = plan.get("feasibility") or {}
        reqs = [r.get("capability") if isinstance(r, dict) else r
                for r in (plan.get("requires_capabilities") or [])]
        (OUT / f"{tag}__{name}.json").write_text(json.dumps(plan, indent=2))
        rows.append({"prompt": name, "verdict": fe.get("verdict"),
                     "n_answerable": len(fe.get("answerable") or []),
                     "n_not_answerable": len(fe.get("not_answerable") or []),
                     "requires": reqs})
        print(f"  {name}: verdict={fe.get('verdict')}  "
              f"answerable={len(fe.get('answerable') or [])}  "
              f"blocked={len(fe.get('not_answerable') or [])}")
        for r in reqs[:4]:
            print(f"      needs: {r}")
    (OUT / f"{tag}__summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {OUT}/{tag}__summary.json")


def compare():
    b = {r["prompt"]: r for r in json.loads((OUT / "before__summary.json").read_text())}
    a = {r["prompt"]: r for r in json.loads((OUT / "after__summary.json").read_text())}
    print(f"{'prompt':28s} {'before':>12s} {'after':>12s}   change")
    for k in b:
        vb, va = b[k]["verdict"], a.get(k, {}).get("verdict")
        chg = "UNCHANGED" if vb == va else f"{vb} -> {va}"
        print(f"{k:28s} {str(vb):>12s} {str(va):>12s}   {chg}")
    print("\nrequires_capabilities, after:")
    for k, r in a.items():
        print(f"  {k}: {r['requires'][:4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=("before", "after"))
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    elif args.tag:
        run(args.tag)
    else:
        ap.error("need --tag or --compare")


if __name__ == "__main__":
    main()
