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
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

FILENAME = "analysis.json"


# The labels the Analyzer's own steps set on their clients. Named here so the
# report can say what THIS box spent, separately from what the process spent:
# Reception and the Planner run in the same process on an end-to-end run and
# do not label their clients, so `total` is pipeline-wide while `by_step` is
# Analyzer-only. Those were being read as the same quantity.
# EVERY ANALYZER LABEL (fix 6 of G part 2, 2026-11-08): the plan's step 2a,
# its review, the one rewrite and its review were missing, so question 1's
# printed line counted 3 of its 5 calls while analysis.json's total held 5.
ANALYZER_LABELS = ("step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                   "step3_plan_review_rewrite", "step2_investigate", "step3_interpret")


def _llm_accounting(expect_calls: bool = False) -> Dict[str, Any]:
    """Tokens and wall time per pipeline step, from the client's usage log.

    Reads the module-level registry rather than any client instance, because
    the steps construct their clients internally — step 4 must account for
    spend by objects it never sees.

    WHICH IS WHY IT CAN BE WRONG, AND NOW SAYS SO. USAGE_LOG is a module-level
    list in THIS process. Assemble the report in a different process from the
    calls — a step-4 rebuild, a resumed run — and every figure reads zero. The
    job-B verification run reported `calls: 0` for an analysis that made four,
    and nothing distinguished that from a run which genuinely called nothing.
    Worse, zero calls is already the signature of a gateway failure, so the two
    were indistinguishable in the direction that matters.

    `expect_calls` is the caller saying "an interpretation exists, so the model
    must have been asked". When that disagrees with an empty log, the numbers
    are not zero — they are missing, and `measured` says which.
    """
    try:
        from agents.llm_agent import USAGE_LOG, usage_totals
    except Exception:
        return {"measured": False,
                "note": "agents.llm_agent could not be imported"}
    labels = sorted({r.get("label") for r in USAGE_LOG if r.get("label")})
    by_step = {lab: usage_totals(lab) for lab in labels}

    analyzer = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "seconds": 0.0, "models": []}
    for lab in ANALYZER_LABELS:
        t = by_step.get(lab)
        if not t:
            continue
        analyzer["calls"] += t.get("calls") or 0
        analyzer["prompt_tokens"] += t.get("prompt_tokens") or 0
        analyzer["completion_tokens"] += t.get("completion_tokens") or 0
        analyzer["seconds"] += t.get("seconds") or 0.0
        analyzer["models"] = sorted(set(analyzer["models"])
                                    | set(t.get("models") or []))
    analyzer["seconds"] = round(analyzer["seconds"], 2)

    # SUMMED HERE, NOT ASKED OF usage_totals. Its `label=None` means "no
    # filter", so usage_totals(None) is the whole log — which made this field a
    # second copy of `total` rather than the unlabelled remainder it is named
    # for. On an end-to-end run that is the Planner's spend, and it read as the
    # Analyzer's.
    bare = [r for r in USAGE_LOG if not r.get("label")]
    unattributed = ({"calls": len(bare),
                     "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in bare),
                     "completion_tokens": sum(r.get("completion_tokens") or 0
                                              for r in bare),
                     "seconds": round(sum(r.get("seconds") or 0.0 for r in bare), 2),
                     "models": sorted({r.get("model") for r in bare if r.get("model")}),
                     "note": "spend by steps outside the Analyzer that ran in "
                             "this process — Reception and the Planner do not "
                             "label their clients"}
                    if bare else None)

    measured = bool(USAGE_LOG) or not expect_calls
    out = {"total": usage_totals(),          # the whole PROCESS, not this box
           "analyzer": analyzer,             # this box only
           "by_step": by_step,
           "unattributed": unattributed,
           "measured": measured}
    if not measured:
        out["note"] = ("this report was assembled in a different process from "
                       "the model calls, so the spend was not recorded here. "
                       "The zeros mean UNMEASURED, not free.")
    return out


def _slurm_elapsed(run_dir) -> Dict[str, Any]:
    """The ensemble job's wall time, asked of the scheduler that ran it.

    NEITHER FILE IN THE RUN DIRECTORY HOLDS THIS. RUN_SUMMARY.json's
    `total_runtime_seconds` times the FRAMEWORK PROCESS, not the model: since
    the ensemble was detached into SLURM job A, the submitting call returns in
    about three seconds and a later --finalize measures its own tail. On
    2026-08-13 that field read 593.8 s for a study whose ELM compute was
    18 m 23 s, and nothing said the two were different quantities.

    The run state records job A's id at the build_cases stage, so the honest
    source is sacct. Absent scheduler, absent id, or a job the accounting has
    already rolled off — all return {} and the caller reports what it has.
    """
    import subprocess
    try:
        st = json.loads((Path(run_dir) / "run_state.json").read_text())
    except Exception:                                           # noqa: BLE001
        return {}
    bc = (st.get("stages") or {}).get("build_cases") or {}
    # job_id_a is the job that BUILT AND RAN. `job_id` is whatever the
    # framework polls, which is job B when one was submitted — B is the
    # reporter and its 40 seconds are not the ensemble's runtime. Runs from
    # before 2026-08-13 carry only `job_id`, and for those this reports
    # nothing rather than reporting B's clock as the model's.
    jid = bc.get("job_id_a")
    if not jid:
        # TWO REASONS FOR NO JOB A, and they are not the same fact. An
        # ensemble run INSIDE an existing allocation never submits one — it
        # runs in place, synchronously — so there is no scheduler record to
        # ask for and nothing is missing. An older run submitted job A but
        # recorded only `job_id`, which is the REPORTING job's. Saying
        # "predates job_id_a" about an in-allocation run blames a vintage for
        # a design.
        if not bc.get("job_id"):
            return {"note": "no ensemble job — this run's columns ran inside "
                            "an existing allocation, in place, so the "
                            "scheduler has no record of them to account for"}
        return {"note": "no ensemble job id recorded — this run predates "
                        "job_id_a (2026-08-13), and the id it does carry is "
                        "the reporting job's"}
    try:
        # -S IS NOT OPTIONAL. sacct's default start time is MIDNIGHT TODAY, so
        # a job that ran last night is invisible to a plain `sacct -j <id>` —
        # it returns nothing and says nothing. That is exactly the case for a
        # study submitted in the evening whose job B lands after midnight,
        # which is how this was found: 773420 reported COMPLETED at 23:47 and
        # was gone from an unqualified query eleven minutes later.
        #
        # 30 days back, which is longer than any run here and still inside
        # what the accounting database keeps.
        since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        out = subprocess.run(
            ["sacct", "-j", str(jid), "-S", since, "-X", "-n", "-P",
             "--format=JobID,JobName,Elapsed,State"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:                                           # noqa: BLE001
        return {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job, name, elapsed, state = parts[0], parts[1], parts[2], parts[3]
        # job A is the one that built and ran; job B reports on it
        if not name.endswith("_A"):
            continue
        try:
            h, m, s = elapsed.split(":")
            d, _, h = h.rpartition("-")
            secs = int(h) * 3600 + int(m) * 60 + int(s) + int(d or 0) * 86400
        except ValueError:
            return {}
        return {"job_id": job, "job_name": name, "elapsed": elapsed,
                "seconds": secs, "state": state, "source": "sacct"}
    return {}


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
        return {"framework_seconds": d.get("total_runtime_seconds"),
                "start_time": d.get("start_time"), "end_time": d.get("end_time"),
                "columns_total": d.get("experiments_total"),
                "columns_succeeded": d.get("experiments_success"),
                "columns_failed": d.get("experiments_failed"),
                "per_column_runtime_seconds": {
                    "n": len(per),
                    "total": round(sum(per), 1) if per else None,
                    "median": (round(statistics.median(per), 1)
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
                "framework_seconds": None,
                "model_seconds": round(sum(per), 1) if per else None,
                "columns_total": d.get("columns_total"),
                "columns_succeeded": d.get("columns_succeeded"),
                "columns_failed": (
                    (d.get("columns_total") - d.get("columns_succeeded"))
                    if isinstance(d.get("columns_total"), int)
                    and isinstance(d.get("columns_succeeded"), int) else None),
                "per_column_runtime_seconds": {
                    "n": len(per),
                    "total": round(sum(per), 1),
                    "median": round(statistics.median(per), 1),
                    "max": round(max(per), 1)} if per else None}
    return {}


def _domain_compute(ctx) -> Dict[str, Any]:
    """A whole-domain experiment's compute, as its chains recorded it.

    Not RUN_SUMMARY and not job A: a whole-domain experiment is one chain of
    one-year jobs per executed run, and each chain's report stage read every
    job's elapsed time back from the scheduler into its STATUS.log, which the
    manager packaged as chain.compute. A reused control spent nothing, and
    the record says so rather than leaving a zero to be read as a fast run.
    """
    ch = (ctx.data or {}).get("chain") or {}
    c = ch.get("compute") or {}
    return {"source": "experiment.json chain.compute (each chain's report stage, "
                      "read from the scheduler)",
            "runs": {r: (v or {}).get("seconds") for r, v in (c.get("runs") or {}).items()},
            "model_seconds": c.get("seconds"),
            "control": c.get("control"),
            "note": c.get("note"),
            "years_planned": ch.get("years_planned"),
            "years_passed": ch.get("years_passed"),
            "first_unpassed": ch.get("first_unpassed"),
            "ensemble": None}


def scope_line(report: Dict[str, Any]) -> str:
    """What the run covered, in one line: columns for a sampled study; years,
    runs and model time for a whole-domain experiment."""
    comp = ((report.get("cost") or {}).get("compute") or {})
    if report.get("kind") == "whole_domain":
        runs = comp.get("runs") or {}
        return (f"years passed {comp.get('years_passed')} of {comp.get('years_planned')}; "
                f"{len(runs)} run(s) executed"
                + (f", {comp.get('model_seconds')} s of model time" if runs else "")
                + ("; the control reused" if comp.get("control") else ""))
    return (f"{comp.get('columns_succeeded')} of {comp.get('columns_total')} "
            f"columns succeeded")


# ─────────────────────────────────────────────────────────────────────
# WHAT KIND OF RESULT THIS IS
# ─────────────────────────────────────────────────────────────────────
# Four outcomes, and the point of naming them is that three of them used to
# render identically. A run whose steps 2-3 raised produced verdict=None and
# zero claims; so did a run that finished and found nothing to say. Both
# reached the reader as a thin report, and the only record of the difference
# lived in the status dict Analyzer.run() RETURNS and never persists.
SUPPORTED    = "supported"              # it ran, concluded, and claims survived
NO_CLAIMS    = "no_supported_claims"    # it concluded; the audit kept nothing
INSUFFICIENT = "insufficient_evidence"  # it ran and judged the evidence too thin
FAILED       = "analysis_failed"        # a step did not run to completion
# A REPORT WHOSE GATES PASS WITH NO CLAIM ELIGIBLE TO HEAD IT (the user's
# decision of 2026-11-08): published, its headline saying so plainly, and
# never an acceptance success
INCONCLUSIVE = "inconclusive"


def _status(steps: Optional[Dict[str, Any]],
            interpretation: Dict[str, Any],
            n_claims: int,
            preflight: Optional[Dict[str, Any]] = None) -> str:
    """Which of the four this run is.

    `steps` is Analyzer.run()'s own record of which stages completed. It is
    normally checked FIRST: a run whose interpreter never executed has no
    verdict to read, and reading the missing verdict as "insufficient" would
    report a crash as a considered scientific judgement.

    THE ONE EXCEPTION IS A DELIBERATE SKIP. When preflight found no frame to
    analyse, steps 1-3 are marked False because they did not run — but they
    did not run because the run was inspected and found to hold nothing, which
    is a finding about the evidence and not a malfunction. Checking that first
    is what stops "we looked, and there is nothing here" from being filed as
    "we broke".
    """
    if (preflight or {}).get("blocked"):
        return INSUFFICIENT
    # SKIPPED IS NOT FAILED HERE EITHER (2026-11-08): a study whose analysis
    # plan did not pass its review made no lookup, and step 3 was not asked;
    # that is a finding about the plan, not a step that broke
    if (interpretation or {}).get("skipped"):
        return INSUFFICIENT
    failed = [k for k, v in (steps or {}).items() if v is False]
    if failed:
        return FAILED
    verdict = (interpretation or {}).get("verdict")
    if verdict is None:
        # No steps were reported failed, yet nothing interpreted. The caller
        # did not tell us what happened, so this is still not a conclusion.
        return FAILED
    if verdict != "sufficient":
        return INSUFFICIENT
    return SUPPORTED if n_claims else NO_CLAIMS


def _model_setup(run_dir) -> Dict[str, Any]:
    """The model-setup record of a whole-domain run, or why it is missing."""
    from agents.analysis import model_setup as MS
    try:
        return MS.collect(run_dir)
    except Exception as e:                                          # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _numbers_check(run_dir) -> Optional[Dict[str, Any]]:
    try:
        from agents.analysis.numbers_check import summary
        return summary(run_dir)
    except Exception:                                               # noqa: BLE001
        return None


NUMBERS_LEAD = ("This analysis reads its numbers from 03_results/regional.json (each claim "
                "below shows how its values trace to it)")


def _domain_report_name(run_dir) -> Optional[str]:
    from agents.analysis.domain_report import REPORT_MD as DOMAIN_MD
    return DOMAIN_MD if (Path(run_dir) / "04_analysis" / DOMAIN_MD).is_file() else None


EXPLORATORY_PREFIX = "Exploratory: "


def exploratory_headline(headline: str) -> str:
    """An exploratory question's headline with its label: "Exploratory: "
    before the model's words, unless they already begin with it."""
    h = str(headline or "").strip()
    return h if h.lower().startswith("exploratory") else EXPLORATORY_PREFIX + h


# THE INCONCLUSIVE HEADLINE, by question type: only when every other gate
# of the report passes and no claim may head it; it meets no acceptance
# criterion the claims left unmet
INCONCLUSIVE_HEADLINE = {
    "hypothesis": "Inconclusive: no tested expectation of this run traced, so it does not "
                  "answer the question as asked.",
    "exploratory": EXPLORATORY_PREFIX + "inconclusive: no claim of this run traced, so it "
                                        "does not answer the question as asked."}
INCONCLUSIVE_WHY = {
    "hypothesis": "the report is inconclusive: no claim may head it, since none is a traced "
                  "tested expectation resting on the planned analyses",
    "exploratory": "the report is inconclusive: no claim traced to head it"}


def build(ctx, comparison: Dict[str, Any], investigation: Dict[str, Any],
          interpretation: Dict[str, Any], run_dir,
          rounds: Optional[List[Dict[str, Any]]] = None,
          stopped_because: str = None,
          steps: Optional[Dict[str, Any]] = None,
          preflight: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble analysis.json. Pure — no API call, no recomputation.

    `steps` is the coordinator's per-stage completion record. Passing it is
    what lets the report distinguish a failure from a finding; omitting it
    means any run with no verdict is reported as `analysis_failed`, which is
    the safe direction to be wrong in.
    """
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

    status = _status(steps, interpretation, len(claims), preflight)
    failed_steps = [k for k, v in (steps or {}).items() if v is False]

    report = {
        "schema": "analysis/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),

        "question": plan.get("question"),
        # WHICH MODEL RAN, from the package: the deck's title and any reader
        # branching on it read this rather than assume (2026-08-18).
        "model": (ctx.data or {}).get("model"),
        # WHICH KIND OF RUN: "whole_domain" for one case over a whole domain,
        # whose scope is years and runs rather than columns. None otherwise.
        "kind": (ctx.data or {}).get("kind"),
        # ITS NUMBERS BY REGION AND YEAR, written from the records before the
        # Analyzer ran (agents.analysis.domain_report). None otherwise.
        "domain_report": (_domain_report_name(run_dir)
                          if getattr(ctx, "is_domain", False) else None),
        # THE MODEL SETUP (2026-11-03): the control, grid, years, configuration
        # and runs, read from the experiment's records (agents.analysis.
        # model_setup) for the block near the top of REPORT.md. None otherwise.
        "model_setup": (_model_setup(run_dir)
                        if getattr(ctx, "is_domain", False) else None),
        # THE NUMBERS CHECK (2026-11-03): the independent remaking of the
        # regional table this analysis reads, for the line near the top of
        # REPORT.md (numbers_check.summary). None otherwise.
        "numbers_check": (_numbers_check(run_dir)
                          if getattr(ctx, "is_domain", False) else None),
        # WHAT KIND OF RESULT THIS IS, before what it says. A reader — or a
        # mailer, or a dashboard — that branches on `verdict` alone cannot tell
        # a crash from a conclusion, because a crash has no verdict at all.
        "status": status,
        # A BLOCKED RUN STILL ANSWERS. Leaving `answer` null made the one
        # case a reader most needs a sentence for — nothing was investigated —
        # the one case with no sentence at all.
        "answer": ((interpretation or {}).get("answer")
                   or ((preflight or {}).get("blocked") and
                       f"This run was not investigated: "
                       f"{preflight['blocked']}.")
                   or None),
        "verdict": (interpretation or {}).get("verdict"),
        # THE HEADLINE AND THE WORDING CHECK travel with the answer: the
        # one-sentence form a slide title carries, and what the readability
        # gate still objects to, so a reader knows before pasting.
        "headline": (interpretation or {}).get("headline"),
        "readability": (interpretation or {}).get("readability"),

        "claims": claims,
        # Rejected claims travel with their reason. Deleting them would make
        # the report look like the reviewer never disagreed with anything.
        "withheld": [{"claim": c.get("claim"),
                      "finding_id": c.get("finding_id"),
                      # WHICH RULE, beside why. `struck_because` is prose for a
                      # reader; `struck_by` is one of AUDIT_RULES, so a run-to-
                      # run comparison can count strikes per rule without
                      # parsing English.
                      "struck_by": c.get("struck_by"),
                      "struck_because": c.get("struck_because")}
                     for c in ((interpretation or {}).get("struck") or [])],

        "caveats": caveats,
        "figures": {
            "comparison": (comparison or {}).get("figures") or {},
            "investigation": {f["id"]: f.get("figure")
                              for f in inv.values() if f.get("figure")},
        },

        "provenance": {
            # WHICH STAGES COMPLETED. The coordinator has always known this and
            # has always thrown it away at the boundary: it lived in the dict
            # Analyzer.run() returns, which nothing writes down. A reader
            # opening this file six months later gets the same account the
            # terminal gave on the night.
            "steps": dict(steps) if steps else None,
            "failed_steps": failed_steps or None,
            # WHAT THE RUN CONTAINED, decided before any model was asked. On a
            # blocked run this is the entire account of why nothing was
            # investigated; on a normal run it is the record of what the figure
            # planner was working from.
            "preflight": preflight,
            # A STEP SKIPPED WITH ITS REASON (2026-11-08): step 3 when the
            # analysis plan did not pass its review
            "skipped_steps": ({"interpret": (interpretation or {}).get("skipped_because")}
                              if (interpretation or {}).get("skipped") else None),
            "rounds": rounds or [],
            "stopped_because": stopped_because,
            "n_findings": len(findings),
            # WHAT STEP 2 SET OUT TO DO, and the document step 3 reviewed it
            # against. Recorded as a path because it is the same file both
            # read; a reader following a claim back can see the plan the
            # figure came from, not only the figure.
            "investigation_plan": (investigation or {}).get("plan_md"),
            "audit": (interpretation or {}).get("audit"),
            "spinup_dropped": ((ctx.data or {}).get("spinup_dropped")
                               if isinstance(ctx.data, dict) else None),
        },

        "cost": {
            # An interpretation exists only if the model was asked, so an empty
            # usage log alongside one means the accounting was lost, not that
            # the run was free.
            "llm": _llm_accounting(
                expect_calls=bool((interpretation or {}).get("verdict"))),
            # THREE DIFFERENT QUANTITIES, NAMED APART. `ensemble` is the model
            # actually running, from the scheduler; `framework_seconds` is this
            # process's own wall time, which for a detached run is the tail and
            # not the science. They were one field called
            # `total_runtime_seconds` and it reported the second while reading
            # as the first.
            "compute": (_domain_compute(ctx) if getattr(ctx, "is_domain", False) else
                        {**_compute_accounting(run_dir),
                         "ensemble": _slurm_elapsed(run_dir) or None}),
        },
    }
    # CLAIMS TRACED TO THE NUMBERS (2026-11-03). The audit of step 3 checks a
    # claim's values against the figure script's own result, so a value the
    # script computed wrongly passed it. On a whole-domain run each value is
    # traced to the regional table the numbers check reproduced from the model
    # output, by a small set of derivations (claim_trace), and the derivation
    # travels with the claim; a headline whose values cannot be traced is
    # withheld (render_markdown).
    #
    # A STUDY WITH A SUITE (package F1, 2026-11-05) reaches its values only
    # through the logged suite calls, so its claims are traced to their
    # evidence records instead (claim_trace.trace_evidence): numbers, periods,
    # regions, evidence class, wording, warnings, the headline's rule and the
    # proposed runs. Any claim that fails withholds the report.
    from agents.analysis import analysis_plan as _plan
    if getattr(ctx, "is_domain", False) and _plan.suite_bearing(ctx):
        # THE PLAN, ITS REVIEW AND THE PLANNED PHASE (the user's decisions on
        # G part 2, 2026-11-08): each version of the plan with what its
        # review found, the close of the planned analyses with each one's
        # status, and the acceptance blockers (a required analysis skipped or
        # failed). A plan that did not pass made no lookup: the page says so
        # and is withheld, and nothing is traced.
        adir = _plan.analysis_dir(ctx)
        ok, why = _plan.plan_passed(adir)
        report["analysis_plan"] = {"passed": ok, "why": why,
                                   "versions": _plan.plan_versions(adir)}
        phase = _plan.load_planned_phase(adir) or None
        report["planned_phase"] = phase
        report["acceptance"] = _plan.acceptance(phase, plan_ok=ok)
        if not ok:
            report["withheld_because"] = (
                f"{why}; no lookup was made, so there is nothing to report")
            return report
        # THE HEADLINE IS A VERIFIED CLAIM'S OWN TEXT (the user's decision of
        # 2026-11-08): no model writes it. It is chosen after the trace, from
        # step 3's ranking, by the headline rule (claim_trace.select_headline);
        # a headline a recorded reply wrote is kept beside, never used
        qt = _plan.load_plan(adir).get("question_type")
        if report.get("headline"):
            report["headline_as_written"] = report["headline"]
        report["headline"] = None
        try:
            from agents.analysis import claim_trace
            report["open_causes"] = list((interpretation or {}).get("open_causes") or [])
            report["evidence_trace"] = claim_trace.trace_evidence(
                report["claims"], None, _plan.analysis_dir(ctx),
                study_dir=ctx.run_dir, regions=suite_regions(ctx),
                open_causes=report["open_causes"])
        except Exception as e:                                  # noqa: BLE001
            report["evidence_trace"] = {
                "error": f"{type(e).__name__}: {e}",
                "withheld_because": f"the claims could not be traced to their evidence "
                                    f"records ({type(e).__name__}: {e})"}
        # THE NUMBERS THE CLAIMS CITE, MADE AGAIN (package F2, 2026-11-05). The
        # trace holds a claim to its records; it cannot see a record that is
        # itself wrong. Every number of each cited record is made again from
        # the model output by code written apart from the suite and the
        # methods (numbers_check.check_evidence), and any mismatch withholds
        # the report, naming the record, the value logged and the value made
        # again.
        from agents.analysis import numbers_check as _nc
        try:
            report["evidence_numbers"] = _nc.check_evidence(ctx.run_dir, report["claims"])
        except Exception as e:                                  # noqa: BLE001
            report["evidence_numbers"] = {"pass": False, "n_compared": 0, "mismatches": [],
                                          "error": f"{type(e).__name__}: {e}"}
        if not report["evidence_numbers"].get("pass"):
            et = report["evidence_trace"]
            et["withheld_because"] = "; ".join(
                x for x in (et.get("withheld_because"),
                            _nc.evidence_reason(report["evidence_numbers"])) if x)
        if report["evidence_trace"].get("withheld_because"):
            report["withheld_because"] = report["evidence_trace"]["withheld_because"]
        # THE SUMMARY HOLDS TO THE KEPT CLAIMS WHEN THE AUDIT STRUCK ONE
        # (2026-11-10). Step 3 writes its summary before the audit, so a
        # sentence of it may rest on a claim the audit then struck. Such a
        # sentence is left out of what the page prints when it gives a number
        # no kept claim's record gives (claim_trace.summary_support); the
        # summary as written stays in the record ("answer"), with what was
        # left out and why ("answer_left_out").
        if (interpretation or {}).get("struck") and report.get("answer") \
                and "records" in report["evidence_trace"]:
            try:
                rows = claim_trace.summary_support(
                    report["answer"], report["claims"], report["evidence_trace"]["records"],
                    report["evidence_trace"].get("runs") or {}, _plan.load_plan(adir))
                left = [r for r in rows if r["unsupported"]]
                if left:
                    report["answer_kept"] = " ".join(r["sentence"] for r in rows
                                                     if not r["unsupported"])
                    report["answer_left_out"] = [{"sentence": r["sentence"],
                                                  "unsupported": r["unsupported"]} for r in left]
            except Exception as e:                              # noqa: BLE001
                report["answer_left_out_error"] = f"{type(e).__name__}: {e}"
        # THE HEADLINE: the first claim of step 3's ranking that traced and
        # that the question type allows; with none, and every other gate
        # passed, a headline that says the run is inconclusive, which is no
        # acceptance success
        if "records" in report["evidence_trace"]:
            sel = claim_trace.select_headline(report["claims"], report["evidence_trace"],
                                              (interpretation or {}).get("headline_rank"))
        else:
            sel = {"chosen": None, "text": None, "passed_over": [],
                   "why": "the claims could not be traced"}
        report["headline_selection"] = sel
        if sel["chosen"]:
            report["headline"] = (exploratory_headline(sel["text"]) if qt == "exploratory"
                                  else sel["text"])
            # LINKED TO ITS CLAIM, WITH THE CLAIM'S CAVEATS (2026-11-08): its
            # number, records and warnings, and the warnings in the registry's
            # words, shown beside the headline wherever it is shown
            c = report["claims"][sel["chosen"] - 1]
            report["headline_caveats"] = claim_trace.headline_caveats(c)
            sel["source"] = {"claim": sel["chosen"], "evidence": list(c.get("evidence") or []),
                             "warnings": [x["code"] for x in report["headline_caveats"]]}
        elif not report.get("withheld_because"):
            kind = qt if qt in INCONCLUSIVE_HEADLINE else "exploratory"
            report["headline"] = INCONCLUSIVE_HEADLINE[kind]
            sel["inconclusive"] = True
            if report["status"] in (SUPPORTED, NO_CLAIMS):
                report["status"] = INCONCLUSIVE
            report["acceptance"] = dict(report.get("acceptance") or {}, ok=False,
                                        why=INCONCLUSIVE_WHY[kind],
                                        inconclusive=INCONCLUSIVE_WHY[kind])
        # THE HEADLINE AS PLAIN TEXT, its caveats beside it, for any place that
        # shows it without the page (the study record, a notice, a summary)
        report["headline_text"] = claim_trace.headline_text(report)
    elif getattr(ctx, "is_domain", False):
        try:
            from agents.analysis import claim_trace
            report["numbers_trace"] = claim_trace.trace(
                report["claims"], report.get("headline"),
                (ctx.data or {}).get("regional") or {})
        except Exception as e:                                  # noqa: BLE001
            report["numbers_trace"] = {
                "error": f"{type(e).__name__}: {e}",
                "withheld_because": f"the claims could not be traced to the numbers "
                                    f"({type(e).__name__}: {e})"}
        if report["numbers_trace"].get("withheld_because"):
            report["withheld_because"] = report["numbers_trace"]["withheld_because"]
    return report


def suite_regions(ctx) -> List[str]:
    """The region names of a study's control suite (its manifest), for the
    claim trace's check of the regions a claim names."""
    from agents.analysis import analysis_plan as _plan
    d = _plan.suite_dirs(ctx).get("control")
    man = _plan._read_json(Path(d) / _plan.SUITE_MANIFEST) if d else {}
    return [str(r.get("name")) for r in man.get("regions") or []
            if isinstance(r, dict) and r.get("name")]


# ─────────────────────────────────────────────────────────────────────
# THE READER'S RENDERING (2026-09-12)
# ─────────────────────────────────────────────────────────────────────
# analysis.json is the record and stays exact. REPORT.md is the same content
# laid out for a person who will read it aloud or paste it onto a slide: the
# headline first, then the answer, then each surviving claim with its numbers
# shown to three significant figures and the finding it rests on, then the
# caveats that bind it, then what the audit withheld. Nothing is computed or
# re-worded here; display precision is not recomputation, and the exact
# values stay one file away.
REPORT_MD = "REPORT.md"


def format_value(v) -> str:
    """A measured value as a reader sees it: 3 significant figures, no exponent
    in the ranges this work lives in, integers left whole."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:                                      # NaN
        return "n/a"
    if x == 0:
        return "0"
    if float(x).is_integer() and abs(x) < 1e7:
        return f"{int(x):,}" if abs(x) >= 10000 else str(int(x))
    mag = abs(x)
    if mag >= 1e6 or mag < 1e-3:
        return f"{x:.3g}"
    decimals = max(0, 2 - int(f"{mag:e}".split("e")[1]))
    # A thousands separator only from 10,000 up: "1,913 m" reads as a
    # typeset number, "1913 m" as an elevation.
    out = f"{x:,.{decimals}f}" if mag >= 10000 else f"{x:.{decimals}f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def _plain(text) -> str:
    """Display form of recorded prose: dashes used as punctuation become
    commas (the record itself is untouched; this is the reader's copy)."""
    from agents.analysis.dashes import strip_dashes
    return strip_dashes(str(text if text is not None else ""))


def render_markdown(report: Dict[str, Any]) -> str:
    """REPORT.md from the assembled report dict. Pure."""
    # WITHHELD WHEN THE HEADLINE CANNOT BE TRACED (2026-11-03): the page then
    # says which values could not be traced and lists every claim's derivations
    # in place of the answer; analysis.json keeps the whole record.
    if report.get("withheld_because"):
        if (report.get("analysis_plan") or {}).get("passed") is False:
            return plan_withheld_markdown(report)
        if (report.get("evidence_trace") or {}).get("withheld_because"):
            from agents.analysis.claim_trace import evidence_withheld_markdown
            return with_planned_phase(evidence_withheld_markdown(report), report)
        from agents.analysis.claim_trace import withheld_markdown
        return withheld_markdown(report)
    q = _plain(report.get("question") or "(no question recorded)")
    model = report.get("model") or "model"
    status = report.get("status") or "unknown"
    lines = [f"# {q}", "",
             f"*{model} run, status **{status}**, generated "
             f"{report.get('generated_at')}, "
             f"{Path(str(report.get('run_dir') or '')).name}*", ""]
    if report.get("kind") == "whole_domain" and "numbers_check" in report:
        from agents.analysis.numbers_check import page_line
        lines += [page_line(report.get("numbers_check"), NUMBERS_LEAD), ""]
    if "evidence_numbers" in report:
        from agents.analysis.numbers_check import evidence_page_line
        lines += [evidence_page_line(report.get("evidence_numbers")), ""]
    lines += ["## The answer", ""]
    if report.get("headline"):
        # ITS CAVEATS BESIDE IT, in the same paragraph (2026-11-08), so a
        # reader of the headline alone, or a page quoting this paragraph,
        # carries them too
        from agents.analysis.claim_trace import headline_caveat_line
        cl = headline_caveat_line(report)
        lines += ([f"**{_plain(report['headline'])}**  ", f"*{_plain(cl)}*"] if cl else
                  [f"**{_plain(report['headline'])}**"]) + [""]
    if "answer_kept" in report:
        n = len(report.get("answer_left_out") or [])
        lines += [_plain(report["answer_kept"] or "(no sentence of the summary rests on the "
                                                  "kept claims alone)"), "",
                  f"*{n} sentence{'s' if n != 1 else ''} of the Analyzer's summary "
                  f"{'are' if n != 1 else 'is'} left out here: each gives a number that no "
                  f"kept claim's record gives, since the audit withheld the claim it rested "
                  f"on. The summary as written is in analysis.json.*", ""]
    else:
        lines += [_plain(report.get("answer") or "(no answer)"), ""]
    rd = report.get("readability") or {}
    if rd.get("problems_after"):
        lines += ["> Wording the readability check still flags: "
                  + "; ".join(str(x) for x in rd["problems_after"][:6]), ""]

    claims = report.get("claims") or []
    lines += [f"## What the run found ({len(claims)} audited claims)", ""]
    for i, c in enumerate(claims, 1):
        lines.append(f"{i}. {_plain(c.get('claim'))}")
        vals = c.get("values") or []
        tail = []
        if vals:
            tail.append("values " + ", ".join(format_value(v) for v in vals[:8]))
        if c.get("n") is not None:
            tail.append(f"n = {c['n']}")
        tail.append(f"finding `{c.get('finding_id')}`")
        for cid in (c.get("caveats") or []):
            tail.append(f"caveat `{cid}`")
        lines.append("   " + "; ".join(tail))
        # Each value's derivation from the regional table (claim_trace).
        for d in (c.get("derivations") or []):
            if d.get("text"):
                lines.append(f"   - {d['text']}")
        # A study with a suite: the claim's evidence class, its records and
        # every warning they carry, beside the claim (package F1).
        if c.get("evidence_trace"):
            from agents.analysis.domain_report import claim_evidence_lines
            lines += claim_evidence_lines(c, (report.get("evidence_trace") or {})
                                         .get("records") or {})
        if c.get("trace"):
            lines.append(f"   - **{c['trace']}**")
        lines.append("")

    caveats = report.get("caveats") or []
    blocking = [c for c in caveats if c.get("severity") == "blocking"]
    others = [c for c in caveats if c.get("severity") != "blocking"]
    if blocking:
        lines += ["## Caveats that bind this answer", ""]
        for c in blocking:
            lines.append(f"- **{c.get('id')}**: {_plain(c.get('statement'))}")
        lines.append("")
    if others:
        lines += ["## Other caveats on record", ""]
        for c in others:
            lines.append(f"- {c.get('id')} ({c.get('severity')}): "
                         f"{_plain(c.get('statement'))}")
        lines.append("")

    if report.get("evidence_trace") and not report.get("withheld_because"):
        from agents.analysis.domain_report import suite_sections
        lines += suite_sections(report)
    lines += planned_phase_lines(report)

    withheld = report.get("withheld") or []
    if withheld:
        lines += [f"## Claims the audit withheld ({len(withheld)})", ""]
        for w in withheld:
            lines.append(f"- {_plain(w.get('claim'))}")
            lines.append(f"  struck by `{w.get('struck_by')}`: "
                         f"{_plain(w.get('struck_because'))}")
        lines.append("")

    lines += verification_limits(report)

    prov = report.get("provenance") or {}
    cost = report.get("cost") or {}
    comp = cost.get("compute") or {}
    llm = (cost.get("llm") or {}).get("analyzer") or {}
    lines += ["## Provenance", "",
              f"- rounds: {len(prov.get('rounds') or [])}, stopped because "
              f"{prov.get('stopped_because')}",
              f"- audit: {(prov.get('audit') or {}).get('n_claims')} claims made, "
              f"{(prov.get('audit') or {}).get('n_struck')} struck",
              (f"- {scope_line(report)}" if report.get("kind") == "whole_domain" else
               f"- columns: {comp.get('columns_succeeded')} of "
               f"{comp.get('columns_total')} succeeded"),
              f"- model calls in the Analyzer: {llm.get('calls', 'not measured')}",
              "", "*Values are shown to 3 significant figures; the exact "
              "measured values are in analysis.json beside this file.*", ""]
    if report.get("domain_report"):
        lines[-3:-3] = [f"- the numbers by region, year, month and season, written from "
                        f"the records "
                        f"before this analysis: {report['domain_report']}"]
    if report.get("model_setup"):
        from agents.analysis import model_setup as MS
        lines = MS.insert_after_answer(lines, [_plain(x) for x in MS.render(
            report["model_setup"])])
    return with_planned_phase("\n".join(lines), report, section=False)


# WHAT CODE CHECKED, AND WHAT IT DID NOT, ON THE LIVE PAGE (2026-11-10). Two
# checks are not built: the numbers check does not cover what a figure draws,
# a plotted series among it, and the claim trace does not hold a ranking to
# its whole comparison set. Only the preview pages said so
# (report_views.coverage_lines); a report a live run wrote carried claims with
# ranking words and figures with no note that code had not checked them. The
# wording is fixed here, by code; the counts are the report's own.
RANKING = re.compile(r"\b(?:most|least|largest|smallest|highest|lowest|earliest|latest|"
                     r"greatest|warmest|coldest|every|all)\b", re.I)
LIMITS_HEADING = "## What code checked, and what it did not"


def ranking_claims(report: Dict[str, Any]) -> List[int]:
    """The kept claims that use a ranking word or speak of every member of a
    set: the claim trace holds their numbers to the records they cite, not
    the ranking to records they leave out."""
    return [i for i, c in enumerate(report.get("claims") or [], 1)
            if RANKING.search(str(c.get("claim") or ""))]


def verification_limits(report: Dict[str, Any]) -> List[str]:
    """The section of a whole-domain report that says which kinds of
    statement code checked and which it did not. [] for another kind of
    report, or one with no claims to speak of."""
    if report.get("kind") != "whole_domain":
        return []
    kept, held = len(report.get("claims") or []), len(report.get("withheld") or [])
    en = report.get("evidence_numbers") or {}
    L = [LIMITS_HEADING, "", "Checked by code:", ""]
    if en.get("n_compared"):
        L.append(f"- the numbers of the records the claims cite: {en['n_compared']:,} "
                 f"numbers made again from the model output by an independent reading"
                 + (", with no mismatch" if en.get("pass") else
                    ", WITH MISMATCHES, listed above"))
    L.append(f"- each claim against the records it cites: the audit kept {kept} and "
             f"withheld {held}")
    ranked = ranking_claims(report)
    L += ["", "Not checked by code:", "",
          "- **Ranking words.** A claim's numbers are held to the records it cites. A "
          "ranking, or a statement about every member of a set (\"most\", \"least\", "
          "\"largest\", \"every band\"), is not held to the records the claim leaves out. "
          + (f"{'Claim' if len(ranked) == 1 else 'Claims'} "
             f"{', '.join(map(str, ranked))} above "
             f"{'uses' if len(ranked) == 1 else 'use'} such a word."
             if ranked else "No kept claim above uses such a word."),
          "- **Figures and plotted series.** What a figure draws beyond the records the "
          "claims cite, a plotted series among it, was not made again by the independent "
          "reading.",
          "- **Interpretation.** Whether the answer follows from its claims is for a "
          "reader to judge; no code judges it.", ""]
    return L


# ─────────────────────────────────────────────────────────────────────
# THE PLAN AND THE PLANNED PHASE ON THE PAGE (the user's decisions on G
# part 2, 2026-11-08)
# ─────────────────────────────────────────────────────────────────────
def not_fully_tested(report: Dict[str, Any]) -> Optional[str]:
    """The first line of a hypothesis study's page when a required analysis
    was skipped or failed: in plain words, that the hypothesis was not fully
    tested and why. None otherwise."""
    acc = report.get("acceptance") or {}
    blockers = acc.get("blockers") or []
    if not blockers:
        return None
    parts = "; ".join(f"{b['analysis']} was {b['status']} ({_plain(b['reason'])})"
                      for b in blockers)
    many = len(blockers) > 1
    return (f"**The hypothesis was not fully tested.** {len(blockers)} required "
            f"{'analyses' if many else 'analysis'} of the plan (named by its expectations) "
            f"{'were' if many else 'was'} not completed: {parts}. This does not count as "
            f"acceptance success.")


def untestable_line(report: Dict[str, Any]) -> Optional[str]:
    """A line under the title when a required output was found not
    estimable: which expectation it leaves untested, and that its task is
    complete. Kept apart from the not-fully-tested line: not estimable is an
    answer, not a gap (the user's decision of 2026-11-08). None otherwise."""
    un = (report.get("acceptance") or {}).get("untestable") or []
    if not un:
        return None
    parts = "; ".join(
        f"{u['expectation']} (analysis {u['analysis']}): "
        + ", ".join(f"its required output {o['output']} is not estimable ({o['record']}: "
                    f"{_plain(o['why'])})" for o in u["outputs"])
        + (", so nothing in this run tests it" if u.get("none_estimable") else "")
        for u in un)
    return (f"**Part of the hypothesis cannot be tested from this run.** {parts}. Each task is "
            f"complete; a value that cannot be estimated supports no claim either way.")


def planned_phase_lines(report: Dict[str, Any]) -> List[str]:
    """The "Planned analyses" section: each analysis of the reviewed plan
    with its status, whether it is required, its required outputs produced,
    not estimable and unmet (each list apart), and its reason, from the
    close of the planned phase (planned_phase.json); then each expectation
    with its analysis's completion beside the tested claims that name it,
    kept apart: completion is not support. [] for a study with none."""
    from agents.analysis import plan_contract as PC
    ph = report.get("planned_phase") or {}
    rows = ph.get("analyses") or []
    if not rows:
        return []
    c = ph.get("counts") or {}
    counts = ", ".join(f"{c.get(s, 0)} {s}" for s in PC.STATUSES)

    def outs(r, outcome, why=False):
        got = [o for o in r.get("outputs") or [] if o.get("outcome") == outcome]
        return "; ".join(f"{o.get('id')}" + (f" ({o.get('record')}" + (f": {_plain(o.get('why'))}"
                                                                       if why else "") + ")"
                                             if o.get("record") else "")
                         for o in got).replace("|", "/") or "none"
    L = ["## Planned analyses", "",
         f"The planned phase was closed {ph.get('closed')} by {_plain(ph.get('by'))} "
         f"(04_analysis/planned_phase.json, after {ph.get('lookups_before')} lookup lines): "
         f"{counts}. Each analysis is judged by its required outputs, each produced, not "
         f"estimable or unmet; its scope (what it may look up) decides nothing.", "",
         "| analysis | status | required | produced | not estimable | unmet | reason |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        req = ("yes (" + ", ".join(r.get("required_by") or []) + ")") if r.get("required") \
            else "no"
        L.append(f"| {r.get('id')} | {r.get('status')} | {req} | {outs(r, PC.PRODUCED)} | "
                 f"{outs(r, PC.NOT_ESTIMABLE, why=True)} | "
                 f"{', '.join(r.get('unmet') or []) or 'none'} | "
                 f"{_plain(r.get('reason')).replace('|', '/')} |")
    exps = ph.get("expectations") or []
    if exps:
        by = {r.get("id"): r for r in rows}
        claims = report.get("claims") or []
        un = {u["expectation"]: u for u in (report.get("acceptance") or {}).get("untestable")
              or []}
        L += ["", f"{_plain(PC.SUPPORT_WORDS)}.", "",
              "| expectation | its analysis | the analysis's completion | its outputs not "
              "estimable | the tested claims that name it |", "|---|---|---|---|---|"]
        for e in exps:
            tested = [i for i, cl in enumerate(claims, 1)
                      if cl.get("evidence_class") == "tested expectation"
                      and cl.get("expectation") == e.get("id")]
            said = ", ".join(f"claim {i}" + (" (fails the evidence trace)"
                                              if claims[i - 1].get("trace") else "")
                             for i in tested) or "none"
            r = by.get(e.get("analysis")) or {}
            ne = ", ".join(o["output"] for o in (un.get(e.get("id")) or {}).get("outputs") or [])
            L.append(f"| {e.get('id')} | {e.get('analysis')} | {r.get('status') or 'no such '}"
                     f"{'' if r else 'analysis'} | {ne or 'none'} | {said} |")
    acc = report.get("acceptance") or {}
    if acc.get("blockers"):
        L += ["", f"Acceptance: not met. {_plain(acc.get('rule'))}."]
    if acc.get("inconclusive"):
        L += ["", f"Acceptance: not met: {_plain(acc['inconclusive'])}."]
    if acc.get("not_estimable"):
        L += ["", "Required outputs found not estimable (their task done, their expectation "
                  "without an estimate): " + "; ".join(
                      f"{x.get('analysis')} {x.get('output')} ({x.get('record')}: "
                      f"{_plain(x.get('why'))})" for x in acc["not_estimable"]) + "."]
    L.append("")
    return L


def with_planned_phase(page: str, report: Dict[str, Any], section: bool = True) -> str:
    """The page with the not-fully-tested line first under its title, and
    (for a page that has none yet) the Planned analyses section at its end."""
    lines = page.split("\n")
    first = [x for x in (not_fully_tested(report), untestable_line(report)) if x]
    for x in reversed(first):
        lines[1:1] = ["", x]
    if section and report.get("planned_phase"):
        lines += [""] + planned_phase_lines(report)
    return "\n".join(lines)


def plan_withheld_markdown(report: Dict[str, Any]) -> str:
    """REPORT.md when the analysis plan did not pass its review: that it did
    not, each version with what its review found (the code check first, then
    the model review), and that no lookup was made. A renderer of the
    record."""
    ap = report.get("analysis_plan") or {}
    vs = ap.get("versions") or []
    q = _plain(report.get("question") or "(no question recorded)")
    lines = ["# WITHHELD: the analysis plan did not pass its review", "",
             f"*Question: {q}*", "",
             "The analysis plan (step 2a) is checked by code and reviewed by a model before "
             "any result is looked up. "
             + (f"It was written {len(vs)} times (the first failed its review and was "
                f"rewritten once, before any lookup), and the last version did not pass "
                f"either. " if len(vs) > 1 else "It did not pass. ")
             + "No lookup was made and step 3 was not asked, so there is nothing to report. "
               "The record, analysis.json, is kept whole beside this page.", ""]
    for v in vs:
        lines += [f"## Version {v.get('attempt')} ({v.get('plan_file')})", ""]
        cp = v.get("code_problems") or []
        lines.append("- the code check: " + ("; ".join(_plain(x) for x in cp)
                                             if cp else "no problem"))
        verdict = {True: "the plan follows from the question",
                   False: "the plan does not follow from the question"}.get(
            v.get("follows"), "no verdict on record")
        mp = v.get("model_problems") or []
        lines.append(f"- the model review: {verdict}"
                     + (f" ({_plain(v.get('model_why'))})"
                        if v.get("follows") is False and v.get("model_why") else "")
                     + ("; its problems: " + "; ".join(_plain(x) for x in mp) if mp else
                        "; it listed no problem"))
        lines.append(f"- passed: {'yes' if v.get('passed') else 'no'}")
        lines.append("")
    lines += ["## Lookups", "", "None: 04_analysis/lookups.jsonl holds no lookup, because "
              "every lookup waits for a plan that passed its review.", ""]
    return "\n".join(lines)


def write(report: Dict[str, Any], out_dir, slides: bool = True) -> str:
    """Write analysis.json, and beside it a deck rendered from the same dict.

    THE JSON IS WRITTEN FIRST AND ALWAYS. The deck is a second rendering of an
    artifact that is already on disk, so a failure in the presentation layer
    cannot cost the run its result — and the returned path is the JSON's,
    because that is the boundary artifact every caller means by "the report".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / FILENAME
    p.write_text(json.dumps(report, indent=2, default=str))
    # THE READER'S COPY, beside the record. Written second, never instead.
    try:
        (out_dir / REPORT_MD).write_text(render_markdown(report))
    except Exception as e:                                      # noqa: BLE001
        print(f"   ⚠️  REPORT.md not written: {e}")

    # A WITHHELD REPORT HAS NO DECK (2026-11-03): slides would present the
    # headline the trace could not support.
    if slides and not report.get("withheld_because"):
        try:
            from agents.analysis import step4_slides
            deck = step4_slides.build(report, out_dir)
            if deck:
                print(f"   ✓ slides → {Path(deck).name}")
        except Exception as e:                                  # noqa: BLE001
            print(f"   ⚠️  slide deck failed ({e}) — analysis.json stands")
    # A DECK FROM AN EARLIER PASS IS SET ASIDE WHEN THIS ONE IS WITHHELD
    # (2026-11-03): left in place it would present the headline that no longer
    # traces. Moved into superseded/, never deleted.
    if report.get("withheld_because"):
        try:
            from agents.analysis.numbers_check import set_aside
            for name in set_aside(out_dir, ("analysis.pptx",), figures=False):
                print(f"   moved the earlier deck aside: {name}")
        except Exception as e:                                  # noqa: BLE001
            print(f"   the earlier deck could not be set aside ({e})")
    return str(p)


def summary(report: Dict[str, Any]) -> str:
    """A few lines for a terminal. The JSON is the artifact; this is a receipt."""
    cost = report.get("cost") or {}
    llm_all = (cost.get("llm") or {})
    llm = llm_all.get("analyzer") or llm_all.get("total") or {}
    comp = cost.get("compute") or {}
    prov = report.get("provenance") or {}
    # THE RECEIPT LEADS WITH WHAT KIND OF RESULT THIS IS. The terminal line is
    # what a person actually reads after a run, and it used to print
    # "verdict: None" for a crash — which looks like a quiet result rather than
    # two steps that never executed.
    failed = prov.get("failed_steps")
    # SKIPPED IS NOT FAILED. When preflight blocked the run, steps 1-3 are
    # marked False because they did not execute — but they did not execute
    # because the run was inspected first. Calling that "did not complete"
    # reads as a malfunction, which is the distinction this whole change
    # exists to keep.
    blocked = (prov.get("preflight") or {}).get("blocked")
    skipped = prov.get("skipped_steps") or {}
    if skipped and not failed:
        why = "  (skipped: " + "; ".join(f"{k}: {v}" for k, v in skipped.items()) + ")"
    elif failed and blocked:
        why = f"  — skipped ({', '.join(failed)}): {blocked}"
    elif failed:
        why = f"  — these steps did not complete: {', '.join(failed)}"
    else:
        why = ""
    lines = [
        f"question : {str(report.get('question'))[:96]}",
        f"status   : {report.get('status')}" + why,
        f"verdict  : {report.get('verdict')}  "
        f"({len(report.get('claims') or [])} claims, "
        f"{len(report.get('withheld') or [])} withheld)",
        f"rounds   : {len(prov.get('rounds') or [])}"
        f"  ({prov.get('stopped_because')})",
        (f"llm      : NOT MEASURED — assembled in a different process from "
         f"the model calls" if llm_all.get("measured") is False else
         f"llm      : {llm.get('calls')} calls, "
         f"{(llm.get('prompt_tokens') or 0) + (llm.get('completion_tokens') or 0)} "
         f"tokens, {llm.get('seconds')} s"),
        (f"compute  : {scope_line(report)}" if report.get("kind") == "whole_domain" else
         f"compute  : {(comp.get('ensemble') or {}).get('elapsed') or '?'} "
         f"({(comp.get('ensemble') or {}).get('job_id') or 'no job id'}) over "
         f"{comp.get('columns_succeeded')}/{comp.get('columns_total')} columns"),
    ]
    for step, u in sorted(((cost.get("llm") or {}).get("by_step") or {}).items()):
        lines.append(f"    {step:20s} {u['calls']} calls  "
                     f"{u['prompt_tokens'] + u['completion_tokens']:>7} tok  "
                     f"{u['seconds']:>6} s")
    return "\n".join(lines)
