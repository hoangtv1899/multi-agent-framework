#!/usr/bin/env python3
"""
Analyzer step 4 — the report
src/agents/analysis/step4_report.py

    in   ctx + comparison.json + investigation.json + interpretation.json
    out  analysis.json — the Analyzer's boundary file, and the framework's

This is the LAST step, and the only one whose output crosses a box boundary.
run_layout names four boundary files, one per box: reception, strategy,
experiment, analysis. Everything in 04_analysis/ is intermediate; this is the
one another box — or a person, or a paper — is meant to read.

IT COMPUTES NOTHING AND CONCLUDES NOTHING. Step 2 computed, step 3 concluded,
and step 3's audit already decided which claims survived. Step 4 assembles.
Anything it recomputed would be a number with no step behind it, which is the
failure the whole chain is built to prevent.

WHAT IT REFUSES TO PRINT. Only claims step 3's audit KEPT. The struck ones
travel too, under `withheld`, with the reason each was struck — a claim that
was made and rejected is evidence about the run, and deleting it would leave
the report looking like the reviewer never disagreed with anything.

COST IS PART OF THE RECORD, not a footnote. Three things a reader needs and no
current artifact carries:

    LLM     tokens and wall time per STEP, not just in total. Attributed via
            the label each step sets on its client, so "step 2 cost 40k tokens
            across 2 rounds" is answerable rather than inferred.
    compute the ELM ensemble's runtime from RUN_SUMMARY, per column and total.
    rounds  how many step2->step3 passes ran, and why it stopped. A result that
            took two rounds and was still judged insufficient is a different
            result from one that passed first time, and the answer alone does
            not show that.

Without this a reader cannot tell an answer that cost one LLM call from one
that cost six, or a 19-column ensemble from a 200-column one.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

FILENAME = "analysis.json"


def _llm_accounting() -> Dict[str, Any]:
    """Tokens and wall time per pipeline step, from the client's usage log.

    Reads the module-level registry rather than any client instance, because
    the steps construct their clients internally — step 4 must account for
    spend by objects it never sees.
    """
    try:
        from agents.llm_agent import USAGE_LOG, usage_totals
    except Exception:
        return {}
    labels = sorted({r.get("label") for r in USAGE_LOG if r.get("label")})
    return {"total": usage_totals(),
            "by_step": {lab: usage_totals(lab) for lab in labels},
            "unattributed": usage_totals(None) if any(
                not r.get("label") for r in USAGE_LOG) else None}


def _compute_accounting(run_dir) -> Dict[str, Any]:
    """The ensemble's runtime, from the manager's own records.

    Not recomputed here: the manager timed the runs and wrote the numbers down.
    Reading them back is the whole contract.

    TWO SOURCES, AND THE ORDER MATTERS. RUN_SUMMARY.json is the fuller record
    but it DOES NOT EXIST YET during a live run: execute_plan writes it last,
    after the Analyzer it is meant to describe. Reading only that file made
    every live run report `compute: null` — including ELM's — and the numbers
    appeared only when the Analyzer was re-run by hand against a finished
    directory. The report claimed to account for compute and silently did not.

    experiment.json is written at stage 4b, which the base guarantees runs
    BEFORE the Analyzer, so the fallback is always available when the primary
    is not.
    """
    for name in ("RUN_SUMMARY.json", "run_summary.json"):
        p = Path(run_dir) / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # A PENDING summary describes a SUBMISSION, not a run. Since Phase 3 a
        # detached run writes one of these and returns, so on resume this file
        # is still the old one — execute_plan overwrites it with the real
        # summary only at the very end, after the Analyzer.
        #
        # Reading it accounted for the wrong thing entirely: a 2/2 ensemble
        # that took 9 minutes was reported as "2.9 s over 0/2 columns", which
        # is the three seconds the submitting call took. experiment.json is
        # written at stage 4b and describes the actual run, so fall through.
        if d.get("status") == "pending":
            continue
        exps = d.get("experiments") or []
        per = [e.get("runtime_seconds") for e in exps
               if isinstance(e, dict) and isinstance(e.get("runtime_seconds"),
                                                     (int, float))]
        return {"total_runtime_seconds": d.get("total_runtime_seconds"),
                "start_time": d.get("start_time"), "end_time": d.get("end_time"),
                "columns_total": d.get("experiments_total"),
                "columns_succeeded": d.get("experiments_success"),
                "columns_failed": d.get("experiments_failed"),
                "per_column_runtime_seconds": {
                    "n": len(per),
                    "total": round(sum(per), 1) if per else None,
                    "median": (round(sorted(per)[len(per) // 2], 1)
                               if per else None),
                    "max": round(max(per), 1) if per else None} if per else None}

    # Fallback: the packaged results, written before this step by construction.
    p = Path(run_dir) / "experiment.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
        except Exception:
            return {}
        per = [c.get("runtime_seconds") for c in (d.get("columns") or [])
               if isinstance(c, dict)
               and isinstance(c.get("runtime_seconds"), (int, float))]
        return {"source": "experiment.json (run summary not yet written)",
                "total_runtime_seconds": round(sum(per), 1) if per else None,
                "columns_total": d.get("columns_total"),
                "columns_succeeded": d.get("columns_succeeded"),
                "columns_failed": (
                    (d.get("columns_total") - d.get("columns_succeeded"))
                    if isinstance(d.get("columns_total"), int)
                    and isinstance(d.get("columns_succeeded"), int) else None),
                "per_column_runtime_seconds": {
                    "n": len(per),
                    "total": round(sum(per), 1),
                    "median": round(sorted(per)[len(per) // 2], 1),
                    "max": round(max(per), 1)} if per else None}
    return {}


def build(ctx, comparison: Dict[str, Any], investigation: Dict[str, Any],
          interpretation: Dict[str, Any], run_dir,
          rounds: Optional[List[Dict[str, Any]]] = None,
          stopped_because: str = None) -> Dict[str, Any]:
    """Assemble analysis.json. Pure — no API call, no recomputation."""
    plan = ctx.plan or {}
    caveats = list(ctx.caveats or []) + list((comparison or {}).get("caveats") or [])
    caveats += list((investigation or {}).get("caveats") or [])

    # BOTH STEPS' FINDINGS, because a claim may cite either. Step 3 audits
    # against step 1's comparison as well as step 2's figures, so a report
    # built from step 2 alone would strip the evidence off exactly the claims
    # that rest on an observation — leaving them in the report with no figure
    # and no n, which reads as a claim from nowhere.
    inv = {f["id"]: f for f in ((investigation or {}).get("findings") or [])}
    findings = {**inv,
                **{f["id"]: f for f in ((comparison or {}).get("findings") or [])}}

    # Each surviving claim carries its evidence with it: the figure a reader
    # can look at and the script that produced the numbers. A claim whose
    # provenance has to be looked up elsewhere is a claim that will not be.
    claims = []
    for c in ((interpretation or {}).get("claims") or []):
        f = findings.get(c.get("finding_id")) or {}
        claims.append({**c,
                       "n": f.get("n"),
                       "figure": f.get("figure"),
                       "script": f.get("script"),
                       "variables": f.get("variables")})

    return {
        "schema": "analysis/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),

        "question": plan.get("question"),
        "answer": (interpretation or {}).get("answer"),
        "verdict": (interpretation or {}).get("verdict"),

        "claims": claims,
        # Rejected claims travel with their reason. Deleting them would make
        # the report look like the reviewer never disagreed with anything.
        "withheld": [{"claim": c.get("claim"),
                      "finding_id": c.get("finding_id"),
                      "struck_because": c.get("struck_because")}
                     for c in ((interpretation or {}).get("struck") or [])],

        "caveats": caveats,
        "figures": {
            "comparison": (comparison or {}).get("figures") or {},
            "investigation": {f["id"]: f.get("figure")
                              for f in inv.values() if f.get("figure")},
        },

        "provenance": {
            "rounds": rounds or [],
            "stopped_because": stopped_because,
            "n_findings": len(findings),
            "audit": (interpretation or {}).get("audit"),
            "spinup_dropped": ((ctx.data or {}).get("spinup_dropped")
                               if isinstance(ctx.data, dict) else None),
        },

        "cost": {
            "llm": _llm_accounting(),
            "compute": _compute_accounting(run_dir),
        },
    }


def write(report: Dict[str, Any], out_dir) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / FILENAME
    p.write_text(json.dumps(report, indent=2, default=str))
    return str(p)


def summary(report: Dict[str, Any]) -> str:
    """A few lines for a terminal. The JSON is the artifact; this is a receipt."""
    cost = report.get("cost") or {}
    llm = (cost.get("llm") or {}).get("total") or {}
    comp = cost.get("compute") or {}
    lines = [
        f"question : {str(report.get('question'))[:96]}",
        f"verdict  : {report.get('verdict')}  "
        f"({len(report.get('claims') or [])} claims, "
        f"{len(report.get('withheld') or [])} withheld)",
        f"rounds   : {len(((report.get('provenance') or {}).get('rounds')) or [])}"
        f"  ({(report.get('provenance') or {}).get('stopped_because')})",
        f"llm      : {llm.get('calls')} calls, "
        f"{(llm.get('prompt_tokens') or 0) + (llm.get('completion_tokens') or 0)} "
        f"tokens, {llm.get('seconds')} s",
        f"compute  : {comp.get('total_runtime_seconds')} s over "
        f"{comp.get('columns_succeeded')}/{comp.get('columns_total')} columns",
    ]
    for step, u in sorted(((cost.get("llm") or {}).get("by_step") or {}).items()):
        lines.append(f"    {step:20s} {u['calls']} calls  "
                     f"{u['prompt_tokens'] + u['completion_tokens']:>7} tok  "
                     f"{u['seconds']:>6} s")
    return "\n".join(lines)
