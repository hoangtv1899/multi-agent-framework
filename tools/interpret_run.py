#!/usr/bin/env python3
"""
LLM interpretation of a completed study — the agentic analyzer step.

The deterministic analyzer produces numbers (experiment.json) and the
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


from agents.analysis import step2_derive as _drv

def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def _forcing_groups(rows):
    """Columns sharing an NLDAS-2 forcing cell — the soil control, as facts.

    Same weather, different soil, so a difference between them is soil or
    terrain. Groups of one, and groups where every column sits at the same
    place, are not controls and are left out. No correlation is computed: the
    block this replaced computed one over two points.
    """
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("status") != "ok" or r.get("forcing_cell") is None:
            continue
        g[tuple(r["forcing_cell"])].append(r)
    out = []
    for cell, members in sorted(g.items()):
        if len(members) < 2:
            continue
        if len({(m.get("lat"), m.get("lon")) for m in members}) < 2:
            continue
        out.append({"forcing_cell": list(cell), "columns": [
            {"column": m["case_name"],
             "soil": m.get("soil_summary"),
             "water_budget": (m.get("metrics") or {}).get("water_budget")}
            for m in members]})
    return out


def compact_results(hs):
    """Boil the package down to what the LLM needs (drop timeseries bulk).

    `columns`, not `experiments` — the rows come from experiment.json now.
    """
    cols = []
    for r in hs.get("columns", []):
        if r.get("status") != "ok":
            continue
        m = r["metrics"]
        cols.append({"column": r["case_name"], "elevation_m": r.get("elevation_m"),
                     "soil": r.get("soil"), **m})
    return {"columns": cols,
            "spatial_summary": _drv.spatial_summary(
                hs.get("columns") or []),
            # soil_attribution is gone (2026-08-13): columns sharing a
            # forcing_cell got the same weather, so grouping on that field is
            # the soil control, and it needs no precomputed block.
            "forcing_groups": _forcing_groups(hs.get("columns") or []),
            "driver_matrix": _drv.driver_matrix(
                hs.get("columns") or [])}


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
    # experiment.json holds the same rows plus the honesty payload; the 8 MB
    # hydro_summary.json copy beside it went on 2026-08-13.
    hs = load(rd / "experiment.json")
    if not hs:
        raise FileNotFoundError(
            f"no {rd}/experiment.json — the Experiment Manager has not "
            f"packaged this run")

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
