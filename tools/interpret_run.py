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

    source /qfs/people/tran289/IDEAS/env_compy.sh            # needs PNNL_API_KEY
    python3 tools/interpret_run.py --run-dir <dir> [--model claude-opus-4-8-project]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

SYSTEM = """You are the analysis agent of a hydrologic simulation framework
(single-column ELM ensembles on real watershed data). Given the question, the
planner's design + feasibility, the per-column results (with water-budget terms
where present), and the observation-validation verdicts, write a SHORT, direct
interpretation — an abstract, not a discussion section.

Rules:
- LEAD with a 1-2 sentence answer to the question, using the key numbers.
- Reason only from the numbers given; cite them; never invent values.
- Be mechanistic but brief (why clay impedes recharge in <=1 line) — no lecturing.
- State the SINGLE most important caveat, not every limitation.
- The payload includes an assumptions_ledger (source-tagged) and a limitations
  catalog (structural vs configuration). In **Trust** you MUST name the most
  consequential configuration limitation, and flag any DEFAULT-sourced
  assumption that could plausibly change the answer (e.g. simulation period,
  initialization). Do not list them all — pick what matters.
- No throat-clearing, no restating the setup.

Output MARKDOWN, UNDER 180 WORDS, exactly these four sections:
**Answer** — 1-2 sentences.
**Why** — 3-4 bullets; each is  driver -> response (with the number) + a <=6-word mechanism.
**Trust** — one line: what the observations confirm or contradict + the top caveat.
**Next** — 1-2 experiments, each tied to a limitation found."""


from agents import drivers as _drv

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
            "spatial_summary": _drv.spatial_summary(
                hs.get("experiments") or []),
            "soil_attribution": {k: v for k, v in (hs.get("soil_attribution") or {}).items()
                                 if k != "by_recharge"},
            "driver_matrix": _drv.driver_matrix(
                hs.get("experiments") or [])}


def interpret(run_dir, model: str = "claude-opus-4-8-project",
              quiet: bool = False) -> str:
    """Interpret a completed study; returns the markdown and writes it to
    <run_dir>/04_analysis/interpretation.md.

    Importable so the integrated pipeline (ELMExpManager) and the CLI share
    ONE implementation — this is the only place that sees the computed
    numbers, the plan's feasibility verdict and the observation validation
    together, which is what makes the interpretation honest.

    Raises FileNotFoundError if the deterministic analysis has not run.
    """
    rd = Path(run_dir)
    hs = load(rd / "04_analysis" / "hydro_summary.json")
    if not hs:
        raise FileNotFoundError(
            f"no {rd}/04_analysis/hydro_summary.json — run the analysis first")

    brief = load(rd / "reception_brief.json")
    plan = load(rd / "plan.json")
    val = load(rd / "04_analysis" / "validation.json")
    payload = {
        "assumptions_ledger": hs.get("assumptions_ledger"),
        "limitations": hs.get("limitations"),
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
    llm = SimpleLLMClient(model=model)
    if not quiet:
        print(f"interpreting {rd.name} with {model} …")
    resp = llm.client.chat.completions.create(
        model=llm.model, max_tokens=1500,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(payload, indent=1, default=str)}])
    md = resp.choices[0].message.content or ""

    out = rd / "04_analysis" / "interpretation.md"
    out.write_text(md)
    if not quiet:
        print("\n" + md)
        print(f"\nwritten to {out}")
    return md


def main():
    ap = argparse.ArgumentParser(description="LLM interpretation of a completed study")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="claude-opus-4-8-project")
    args = ap.parse_args()
    try:
        interpret(args.run_dir, model=args.model)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
