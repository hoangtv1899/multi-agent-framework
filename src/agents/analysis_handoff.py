#!/usr/bin/env python3
"""
The plan that was shown on the design page, handed to the Analyzer
src/agents/analysis_handoff.py

Off unless its switch is set (2026-11-12). With the analysis preview on
(agents/analysis_preview.py) a person reads an analysis plan before the yes,
and after the runs the Analyzer writes another one, which may differ. This
module closes that gap for one case and no other: a plan that passed its
check and its review, shown for the design that was then approved, is the
plan the Analyzer carries out. It writes no second one.

THE YES IS OF THE PLAN TOO (stage 4, 2026-11-12). A page that promises the
plan says that a yes approves it together with the runs and the years. The
approval then names the plan by its hash (design_review.approve), apart from
the design's own fingerprint and apart from the cost the yes allows. After
the runs the plan is handed over only if the yes on record names it, and a
design whose approved plan has changed is not run at all
(design_review.approval_problem).

A DESIGN WITH NO PLAN TO APPROVE CANNOT BE APPROVED. With this switch on, a
design whose plan did not pass its check and its review, or could not be
written, has no yes: there is no analysis a yes could cover, and the runs of
a study that cannot be analysed as approved are not spent (unapprovable). A
study with no results store has no analysis plan at all and is as before.

WHAT IT DOES NOT DO.

  - It changes no plan. The files are copied byte for byte, and the plan's
    hash is checked against the hash recorded when the page was written and
    against the hash the yes names.
  - A planned year that a run did not produce is closed as failed by the
    Analyzer's own close of the planned phase, as before.
  - A record made before stage 4 promises its plan without the yes naming
    it. Such a study is handed its plan as it was promised then.

THE PROMISE IS MADE AT THE PAGE AND KEPT FROM THE RECORD. Whether a plan
will be carried out is decided when the page is written, with the switch read
then, and written into the preview's record (`handoff`). After the runs the
Analyzer reads that record and not the environment: the analysis is the last
line of a batch job on a compute node, where a switch set in a login shell
does not exist. So the page and the Analyzer cannot disagree.

HOW THE ANALYZER IS HANDED THE PLAN. Its planning step keeps what is on
disk: write_plan returns a plan already in 04_analysis, and the review of a
plan is recorded once. The handoff copies the shown plan, its review, the
first version when it was rewritten and the model's exchanges into the
study's 04_analysis before that step runs. The step then finds them, makes
no model call, and its gate (plan_passed) holds the plan to the hash its
review recorded.

WHEN THE PROMISE CANNOT BE KEPT, THE ANALYSIS STOPS (the user's decision of
2026-11-12, evening). The design approved is not the design the plan was
written for, the plan or its review changed or is gone, or the study's
04_analysis already holds another plan: the Analyzer is not handed the plan,
and it does not write another in its place. A person read one plan and was
told it would be carried out, so a second plan written out of sight is the
thing this stage exists to end. 04_analysis/plan_handoff.json records the
handoff as blocked, with the reason. The planning step is not called, the
gate every lookup passes (analysis_plan.plan_passed) refuses, and the report
is a page that says the handoff is blocked and what is kept. Nothing is
deleted: the runs, their numbers and the plan shown stay as they were.

A blocked handoff is checked again on a resume, and goes through only if the
plan shown can then be handed over unchanged. To analyse such a study with
another plan is a new decision, which this stage does not take.

A STUDY WHOSE PAGE MADE NO PROMISE gets no record and is exactly as before:
its Analyzer writes its own plan after the runs.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SWITCH = "IDEAS_CARRY_OUT_SHOWN_PLAN"
RECORD_FILE = "plan_handoff.json"

# The words of the page, in the place of "the Analyzer writes its plan again
# after the runs", when the plan shown will be carried out.
CARRIED_OUT = ("After the runs the Analyzer carries out this plan as it stands and writes no "
               "second one. If it cannot, the analysis stops and its report says why. "
               "Approval of the analysis is not built yet, so the record of your "
               "yes names the runs and the years and not this plan.")
# a design that makes no new run: there are no runs to wait for or to name
CARRIED_OUT_NO_RUN = ("After your yes the Analyzer carries out this plan as it stands and "
                      "writes no second one. If it cannot, the analysis stops and its report "
                      "says why. Approval of the analysis is not built yet, so "
                      "the record of your yes names the years and not this plan.")
# The words of a page whose yes approves the plan (stage 4): the two above
# stay for a record made before, which carries its own words.
APPROVED = ("A yes approves this plan together with the runs and the years. After the runs "
            "the Analyzer carries the plan out as it stands and writes no second one. If it "
            "cannot, the analysis stops and its report says why.")
APPROVED_NO_RUN = ("A yes approves this plan together with the years. After your yes the "
                   "Analyzer carries the plan out as it stands and writes no second one. If "
                   "it cannot, the analysis stops and its report says why.")
USED = ("the plan shown on the design page before the yes was carried out as it stood: "
        "the Analyzer wrote no second plan")
NOT_NAMED = "the yes on record does not name this plan"
NO_PLAN_TO_APPROVE = "so there is no analysis plan a yes could approve"
NOT_PASSED = "the plan shown did not pass its check and its review"
# why nothing passes the Analyzer's gate when the promise could not be kept
BLOCKED = ("the plan shown on the design page was promised to the Analyzer and could not be "
           "handed to it ({why}), and the Analyzer writes no other plan in its place")


def enabled() -> bool:
    """Whether a plan shown on the page is promised to the Analyzer. Read
    when the page is written, never after the runs."""
    return str(os.environ.get(SWITCH) or "").strip().lower() in ("1", "true", "yes", "on")


def promise(passed: bool, new_runs: bool = True) -> Optional[Dict[str, Any]]:
    """What the preview's record says of the handoff, or None with the switch
    off (the record is then as it was). Only a plan that passed is promised.
    `new_runs` is whether the design makes a run: the page's words differ."""
    if not enabled():
        return None
    if passed:
        # bound: a yes given on this page approves the plan (stage 4)
        return {"promised": True, "bound": True,
                "words": APPROVED if new_runs else APPROVED_NO_RUN}
    return {"promised": False, "why": NOT_PASSED}


def page_words(record: Optional[Dict[str, Any]]) -> str:
    """The page's sentence for a plan the record promises to the Analyzer."""
    return str(((record or {}).get("handoff") or {}).get("words")
               or (APPROVED if bound(record) else CARRIED_OUT))


def promised(record: Optional[Dict[str, Any]]) -> bool:
    h = (record or {}).get("handoff")
    return isinstance(h, dict) and h.get("promised") is True


def bound(record: Optional[Dict[str, Any]]) -> bool:
    """Whether a yes given on this record's page approves the plan it shows."""
    return promised(record) and (record or {}).get("handoff", {}).get("bound") is True


def unapprovable(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a design cannot be approved for want of an analysis plan, or None.
    Only where the page was written with this switch on (the record then
    says whether it promises a plan) and a plan was needed: one that did not
    pass, or could not be written. A study that has no analysis plan at all
    (no results store, no whole-domain experiment) is not held to one."""
    if not isinstance(record, dict) or not isinstance(record.get("handoff"), dict) \
            or promised(record):
        return None
    status = record.get("status")
    if status == "not passed":
        return (f"its analysis plan did not pass its check and its review, "
                f"{NO_PLAN_TO_APPROVE}")
    if status == "failed":
        return f"its analysis plan could not be written, {NO_PLAN_TO_APPROVE}"
    return None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha(p: Path) -> Optional[str]:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _read(p: Path) -> Dict[str, Any]:
    try:
        got = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def load(analysis_dir) -> Dict[str, Any]:
    """The handoff's record in a study's 04_analysis, {} when there is none."""
    return _read(Path(analysis_dir) / RECORD_FILE)


def blocked(record: Optional[Dict[str, Any]]) -> bool:
    """Whether a handoff's record is of a promise that could not be kept. A
    record is written only for a study whose page promised its plan, so one
    that does not say the plan was carried out is blocked."""
    return bool(record) and record.get("used") is not True


def why_blocked(analysis_dir) -> Optional[str]:
    """Why nothing may be analysed in this folder, when its handoff is
    blocked; None for a handoff that went through and for a study with no
    promise. Read by the Analyzer's gate (analysis_plan.plan_passed)."""
    rec = load(analysis_dir)
    if not blocked(rec):
        return None
    return BLOCKED.format(why=str(rec.get("why") or "no reason on record"))


def _why_not(rd: Path, rec: Dict[str, Any], src: Optional[Path]) -> Optional[str]:
    """Why the plan shown may not be carried out, or None."""
    from agents import analysis_preview as AV
    from agents import design_review as DR
    from agents.analysis import analysis_plan as AP
    if rec.get("status") != AV.PLANNED:
        return NOT_PASSED
    approval = _read(rd / DR.APPROVAL_FILE)
    if not approval:
        return "no approval of the design is on record"
    if not rec.get("design_sha256") or approval.get("design_sha256") != rec["design_sha256"]:
        return ("the design that was approved is not the design this plan was written for "
                "(its control, runs or years differ)")
    # THE YES NAMES THE PLAN (stage 4). A page that said a yes approves the
    # plan is kept to it: the plan handed over is the one the approval names.
    if bound(rec) and approval.get("analysis_plan_sha256") != rec.get("plan_sha256"):
        return NOT_NAMED
    if src is None or not (src / AP.PLAN_FILE).is_file():
        return "the plan shown is no longer on disk"
    if not rec.get("plan_sha256") or _sha(src / AP.PLAN_FILE) != rec["plan_sha256"]:
        return "the plan on disk is not the plan that was shown (its file changed)"
    ok, why = AP.plan_file_passed(src)
    if not ok:
        return f"the plan shown no longer passes: {why}"
    return None


def _recheck(ctx, plan: Dict[str, Any]) -> Dict[str, Any]:
    """The plan's own check, made again against the runs as they finished. It
    was first made before any run, against the runs as planned. What it finds
    now is recorded and changes nothing: an analysis whose inputs a run did
    not produce is closed as failed by the Analyzer's close."""
    from agents.analysis import analysis_plan as AP
    out: Dict[str, Any] = {"against": "the runs as they finished"}
    try:
        fields, regions = AP._catalog_names(ctx)
        out["problems"] = AP.plan_problems(plan, fields, regions, AP.planner_normal_years(ctx),
                                           plan.get("question_type"), runs=AP.study_runs(ctx))
    except Exception as e:                                          # noqa: BLE001
        out["not_made"] = f"{type(e).__name__}: {e}"[:300]
    return out


def _write(out: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORD_FILE).write_text(json.dumps(record, indent=1, default=str))
    return record


def _hand_over(ctx, out: Path, copied: List[str]) -> Optional[Dict[str, Any]]:
    from agents import analysis_preview as AV
    from agents import design_review as DR
    from agents.analysis import analysis_plan as AP
    rd = Path(str(getattr(ctx, "run_dir", "") or "."))
    rec = AV.load(rd)
    if not promised(rec):
        return None                     # no promise on the page: nothing changes
    done = load(out)
    if done.get("used") is True:
        return done                     # carried out: a resume reads the decision
    # A BLOCKED HANDOFF IS CHECKED AGAIN, by the same checks: its cause may be
    # gone (a file put back, a disk that was full). The first stop stays on
    # record.
    src = (rd / rec["plan_file"]).parent if rec.get("plan_file") else None
    base = {"at": _now(), "shown_at": rec.get("made"),
            "design_sha256": rec.get("design_sha256"), "plan_sha256": rec.get("plan_sha256"),
            "from": str(src.relative_to(rd)) if src is not None else None}
    if done:
        base["first_blocked_at"] = done.get("first_blocked_at") or done.get("at")
    why = _why_not(rd, rec, src)
    if why:
        return _write(out, _stopped(base, why))

    # WHAT THE STUDY'S OWN FOLDER ALREADY HOLDS. Nothing there is overwritten.
    # The shown plan already in place (an interrupted handoff) is finished; any
    # other planning file means the Analyzer planned here before, and its plan
    # stands.
    names = sorted(p.name for p in src.iterdir() if p.is_file())
    clash = [n for n in names if (out / n).is_file() and _sha(out / n) != _sha(src / n)]
    if clash:
        return _write(out, _stopped(base, (
            "the study's 04_analysis already holds planning files that differ from the "
            "plan shown: " + ", ".join(clash))))
    out.mkdir(parents=True, exist_ok=True)
    # the plan last: a copy cut short leaves no plan, and is made again
    for n in [n for n in names if n != AP.PLAN_FILE] + [AP.PLAN_FILE]:
        if not (out / n).is_file():
            shutil.copyfile(src / n, out / n)
            copied.append(n)
    ok, passed_why = AP.plan_file_passed(out)
    if AP.plan_sha256(out) != rec["plan_sha256"] or not ok:
        _remove(out, copied)
        return _write(out, _stopped(base, (
            "the copy of the plan shown did not match it"
            + ("" if ok else f" ({passed_why})"))))
    approval = _read(rd / DR.APPROVAL_FILE)
    return _write(out, dict(
        base, used=True, why=USED, files=names,
        # whether the yes was of this plan (stage 4), or only of its design
        approved=bool(bound(rec)
                      and approval.get("analysis_plan_sha256") == rec["plan_sha256"]),
        approval={k: approval.get(k) for k in ("decision", "when")},
        recheck=_recheck(ctx, AP.load_plan(out))))


def _stopped(base: Dict[str, Any], why: str) -> Dict[str, Any]:
    """The record of a promise that could not be kept: nothing handed over,
    and nothing written in its place."""
    return dict(base, used=False, blocked=True, why=why, files=[])


def _remove(out: Path, names: List[str]) -> None:
    """Take back what this call copied, so that a handoff that did not
    complete leaves no plan for the Analyzer to mistake for its own."""
    while names:
        try:
            (out / names.pop()).unlink()
        except OSError:
            pass


def hand_over(ctx, analysis_dir) -> Optional[Dict[str, Any]]:
    """Before the Analyzer plans: put the plan shown on the design page in
    the study's 04_analysis when the page promised it and it is still the plan
    that was shown, for the design that was approved. Returns the handoff's
    record, or None when the page made no promise.

    NEVER RAISES. Whatever goes wrong under a promise, the handoff is
    blocked and the record says why when one can be written. The caller asks
    blocked(record) and, when it is, does not plan."""
    out = Path(analysis_dir)
    copied: List[str] = []
    try:
        return _hand_over(ctx, out, copied)
    except Exception as e:                                          # noqa: BLE001
        _remove(out, copied)
        record = {"at": _now(), "used": False, "blocked": True, "files": [],
                  "why": f"the handoff could not be made: {type(e).__name__}: {e}"[:400]}
        try:
            return _write(out, record)
        except Exception:                                           # noqa: BLE001
            return record


def report_lines(record: Optional[Dict[str, Any]]) -> List[str]:
    """What the report says of the handoff, [] when the page made no promise."""
    if not record:
        return []
    if record.get("used"):
        L = ["This plan was shown on the design page and approved with the yes, and the "
             "Analyzer carried it out as it stood. The plan file is the one that was shown, "
             "byte for byte, and the record of the yes names it by its hash."
             if record.get("approved") else
             "This plan was shown on the design page before the yes, and the Analyzer "
             "carried it out as it stood. The plan file is the one that was shown, byte for "
             "byte. The yes approved the design. It was not tied to this plan, since "
             "approval of the analysis was not built when this study was approved."]
        problems = (record.get("recheck") or {}).get("problems") or []
        if problems:
            L.append("The plan was written before any run. Checked again against the runs "
                     "as they finished, it has " + str(len(problems)) + " problem"
                     + ("" if len(problems) == 1 else "s") + ": " + "; ".join(
                         str(p) for p in problems) + ".")
        return L
    return blocked_lines(record)


def blocked_lines(record: Optional[Dict[str, Any]]) -> List[str]:
    """What is said of a promise that could not be kept. The reason may
    quote a model's review, so its dashes are taken out for the reader."""
    from agents.analysis.dashes import strip_dashes
    why = strip_dashes(str((record or {}).get("why") or "no reason on record"))
    return ["A plan was shown on the design page before the yes, and the page said the "
            "Analyzer would carry it out as it stood. The Analyzer could not be handed that "
            "plan: " + why + ". It wrote no other plan in its place, so no result was looked "
            "up and no claim was made."]


def blocked_page(report: Dict[str, Any]) -> str:
    """REPORT.md of a study whose handoff is blocked: what was promised, why
    it could not be kept, what is kept and what a resume does. A renderer of
    the record."""
    rec = (report.get("analysis_plan") or {}).get("handoff") or {}
    from agents.analysis.dashes import strip_dashes
    q = strip_dashes(str(report.get("question") or "(no question recorded)"))
    kept = ["- The runs, their output, the numbers by region and the check of those numbers "
            "are as they were. Nothing was deleted or written over.",
            "- The plan that was shown and its review are in `" + str(rec.get("from")) + "`."
            if rec.get("from") else
            "- The record of the plan that was shown is `01_inputs/analysis_preview.json`.",
            f"- The record of this stop is `04_analysis/{RECORD_FILE}`."]
    return "\n".join(
        ["# BLOCKED: the plan shown on the design page could not be carried out", "",
         f"*Question: {q}*", ""] + blocked_lines(rec) + ["", "## What is kept", ""] + kept
        + ["", "## What a resume does", "",
           "A resume of this study checks the handoff again. The analysis goes ahead only if "
           "the plan that was shown can then be handed over unchanged. To analyse the study "
           "with another plan is a new decision, and this version does not take it.", ""])
