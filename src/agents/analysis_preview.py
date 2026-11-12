#!/usr/bin/env python3
"""
The analysis plan, written before the design page and shown on it
src/agents/analysis_preview.py

TRANSITIONAL (2026-11-12), and off unless its switch is set. A person
approved a design from a page that said little about the analysis: the
Analyzer's planning step (analysis_plan.write_plan, step 2a) runs after the
runs, and its plan was first seen in the report. This module runs that same
step, unchanged, before the page, from the design alone, and the page shows
what it wrote.

WHAT IT DOES NOT DO, said on the page as well:

  - The yes is not bound to this plan. It approves what it approved before:
    the control, the runs and the years (design_review.design_fingerprint).
  - The Analyzer does not carry this plan out. After the runs it writes its
    own plan, from the same inputs, in the study's 04_analysis, as it always
    did. That plan may differ from the one shown here.
  - A plan that fails its check, or that could not be written, does not stop
    a design: it is shown as it is.

So this stage is a look before the decision and nothing more. Binding the
yes to the plan, and handing the approved plan to the Analyzer, are later
stages, and until they are built and tested this page must not be read as
the approval of an analysis.

WHERE THE PLAN IS WRITTEN. Under the study's 01_inputs/analysis_preview/, in
a folder of its own that has the shape of a small study: the two records the
planning step's catalog reads (01_inputs/domain_experiment.json with each
change in the server's own words, 01_inputs/domain_control_facts.json with
the control's location) and a 04_analysis holding the plan, its review and
the model's exchanges. Nothing is written into the study's own 04_analysis,
and no record of the Experiment Manager is touched.

THE SAME PLANNING, NOT A COPY OF IT. The context is the one the planning step
reads after a run, built from the design: the question, the Planner's
analysis and goals, the control from the registry, each run's changes as the
server's own resolve() words them. What does not exist before a run is left
out, never guessed: whether each planned year passes its checks, and a run's
own results store. The plan, its check, its review and its one rewrite are
step3_interpret.plan_and_review, the call the Analyzer makes.
"""
import json
import os
import re
import time
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SWITCH = "IDEAS_PLAN_BEFORE_APPROVAL"
RECORD_FILE = "01_inputs/analysis_preview.json"
FOLDER = "01_inputs/analysis_preview"
EXPERIMENT_FILE = "01_inputs/domain_experiment.json"
CONTROL_FILE = "01_inputs/domain_control_facts.json"
CHANGE_KEYS = ("id", "params", "summary", "meaning", "not_meaning")

PLANNED, NOT_PASSED, NOT_PLANNED, FAILED = "planned", "not passed", "not planned", "failed"
HEADING = "## The analysis, as planned before the runs (a preview)"
NOT_BOUND = ("Your yes approves the runs and the years, as before. It does not approve this "
             "analysis: after the runs the Analyzer writes its plan again, from the same "
             "inputs, and that plan may differ from the one shown here.")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SEASONS = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
STEP = {"day": "daily", "month": "monthly", "season": "seasonal", "year": "yearly",
        "wateryear": "water-year"}


def enabled() -> bool:
    """Whether the analysis is planned before the page. Off unless the switch
    is set: the planning is a paid model call, made for every design that
    reaches the page, and in this stage the Analyzer plans again after the
    runs."""
    return str(os.environ.get(SWITCH) or "").strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTEXT THE PLANNING STEP READS, WITH NO RUN BEHIND IT
# ─────────────────────────────────────────────────────────────────────────────
def resolved_changes(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A run's changes in the server's own words: resolve() fills the
    defaults and writes what the difference means and does not mean."""
    from agents.analysis import analysis_plan as AP
    AP._elm()
    import domain_changes as DC                                     # noqa: N812
    out = []
    for c in run.get("changes") or []:
        rc = DC.resolve({"id": c.get("id"), "params": dict(c.get("params") or {})})
        out.append({k: rc[k] for k in CHANGE_KEYS if rc.get(k) not in (None, "", {}, [])})
    return out


def design_runs(de: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The control's row, then each run with its resolved changes, as the
    packaged experiment lists them after a run."""
    ctl = de.get("control") or {}
    rows = [{"run": "control", "reused": bool(ctl.get("reuse")), "changes": []}]
    for r in de.get("runs") or []:
        rows.append({"run": r.get("name") or r.get("run"), "reused": False,
                     "changes": resolved_changes(r)})
    return rows


class PreRunContext:
    """What analysis_plan.plan_inputs reads, built from a design."""
    is_domain = True

    def __init__(self, question: Any, strategy: Dict[str, Any], folder,
                 domain: Optional[Dict[str, Any]] = None):
        from agents.analysis import analysis_plan as AP
        de = strategy.get("domain_experiment") or {}
        ctl = de.get("control") or {}
        entry = AP._registry_entry(ctl.get("name"))
        control = {"name": ctl.get("name"), "reuse": bool(ctl.get("reuse")),
                   "case_dir": str(entry.get("case_dir")), "run_dir": str(entry.get("run_dir"))}
        runs = design_runs(de)
        d = Path(folder)
        (d / "01_inputs").mkdir(parents=True, exist_ok=True)
        (d / EXPERIMENT_FILE).write_text(json.dumps(
            {"control": {"name": control["name"], "reuse": control["reuse"]},
             "runs": [{"name": r["run"], "changes": r["changes"]} for r in runs
                      if r["run"] != "control"],
             "years": de.get("years"), "analysis": de.get("analysis") or {}}, indent=1))
        (d / CONTROL_FILE).write_text(json.dumps({"control": control}, indent=1))
        self.run_dir = str(d)
        self.plan = {"question": question, "domain": domain, "goals": strategy.get("goals"),
                     "experiment": {"control": control, "runs": runs,
                                    "planned_runs": de.get("runs") or [],
                                    "years": de.get("years"),
                                    "analysis": de.get("analysis") or {},
                                    "caveats": de.get("caveats") or [], "check": None}}

    # no result exists before a run: asked for one, the context says so
    def regional(self):
        raise RuntimeError("no result exists before a run")
    regional_monthly = regional_seasonal = regional


def context(reception: Dict[str, Any], strategy: Dict[str, Any], run_dir) -> PreRunContext:
    """The planning context of a design, with its two records written under
    the study's analysis-preview folder (named for the study, so the catalog
    names the study as it will after the run)."""
    rd = Path(run_dir)
    brief = reception.get("brief") or {}
    dom = brief.get("domain") or {}
    return PreRunContext(reception.get("user_request") or brief.get("user_request"),
                         strategy, rd / FOLDER / rd.name,
                         domain={k: dom.get(k) for k in ("name", "huc", "bbox")}
                         if isinstance(dom, dict) else None)


# ─────────────────────────────────────────────────────────────────────────────
# THE PLANNING, BEFORE THE PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def plan_before_approval(reception: Dict[str, Any], strategy: Dict[str, Any], run_dir,
                         clients: Optional[Dict[str, Any]] = None,
                         model: Optional[str] = None) -> Dict[str, Any]:
    """Write the analysis plan of a design before its page, and return the
    record of it (also written to 01_inputs/analysis_preview.json).

    NEVER RAISES. A design must reach its page whatever happens here, so an
    error is the record's status and reason. `clients` maps a planning label
    to the client that step uses, as in step3_interpret; a step with none
    makes its own, labelled for the spend."""
    from agents import design_review as DR
    rd = Path(run_dir)
    rec: Dict[str, Any] = {"made": _now(), "transitional": NOT_BOUND,
                           "design_sha256": DR.design_fingerprint(strategy),
                           "status": NOT_PLANNED, "why": None, "plan_file": None,
                           "review_file": None, "attempts": 0, "elapsed_s": 0.0}
    t0 = time.time()
    try:
        if not isinstance((strategy or {}).get("domain_experiment"), dict):
            rec["why"] = "the design is not a whole-domain experiment"
            return _written(rd, rec)
        from agents.analysis import analysis_plan as AP
        from agents.analysis import step2_investigate as S2
        from agents.analysis import step3_interpret as S3
        ctx = context(reception, strategy, rd)
        if not AP.suite_bearing(ctx):
            rec["why"] = ("the prepared run has no results store, and the planning step "
                          "plans from one")
            return _written(rd, rec)
        adir = Path(ctx.run_dir) / "04_analysis"
        review, ok = S3.plan_and_review(ctx, adir, model=S2.analyzer_model(model),
                                        clients=clients)
        plan = AP.load_plan(adir)
        rel = adir.relative_to(rd)
        rec.update(status=PLANNED if ok else NOT_PASSED,
                   why=None if ok else AP.plan_passed(adir)[1],
                   plan_file=str(rel / AP.PLAN_FILE), review_file=str(rel / AP.REVIEW_FILE),
                   attempts=int(plan.get("attempt") or 1),
                   plan_sha256=AP.plan_sha256(adir),
                   review_passed=bool((review or {}).get("passed")),
                   model=S2.analyzer_model(model))
    except Exception as e:                                          # noqa: BLE001
        rec.update(status=FAILED, why=f"{type(e).__name__}: {e}"[:400])
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return _written(rd, rec)


def _written(rd: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
    try:
        p = rd / RECORD_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, default=str))
    except OSError as e:
        rec["record_not_written"] = str(e)
    return rec


def load(run_dir) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((Path(run_dir) / RECORD_FILE).read_text())
    except (OSError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGE'S SECTION: RENDERED FROM THE PLAN, NEVER WRITTEN
# ─────────────────────────────────────────────────────────────────────────────
def _join(xs: Sequence[str]) -> str:
    xs = [x for x in xs if x]
    return ("" if not xs else xs[0] if len(xs) == 1 else
            ", ".join(xs[:-1]) + " and " + xs[-1])


def _span(years: Any) -> str:
    ys = sorted(int(y) for y in years or [] if str(y).lstrip("-").isdigit())
    if not ys:
        return "no year"
    return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]} to {ys[-1]}"


def _cell(text: Any) -> str:
    return " ".join(str(text if text is not None else "").split()).replace("|", "/")


def periods_words(periods: Sequence[Any]) -> str:
    """A plan's period keys in words: 2031-06 a month, 2031-JJA a season,
    2031 a year, WY2031 a water year; anything else as it is written."""
    months, rest = OrderedDict(), []
    for p in periods or []:
        s = str(p)
        m = re.fullmatch(r"(\d{4})-(\d{2})", s)
        q = re.fullmatch(r"(\d{4})-([A-Z]{3})", s)
        if m:
            months.setdefault(m.group(1), []).append(int(m.group(2)))
        elif q:
            rest.append(f"{SEASONS.get(q.group(2), q.group(2))} {q.group(1)}")
        elif re.fullmatch(r"\d{4}", s):
            rest.append(f"the year {s}")
        elif re.fullmatch(r"WY\d{4}", s):
            rest.append(f"the water year {s[2:]}")
        else:
            rest.append(s)
    words = []
    for y, ms in months.items():
        ms = sorted(set(ms))
        if len(ms) == 12:
            words.append(f"each month of {y}")
        elif len(ms) > 1 and ms == list(range(ms[0], ms[-1] + 1)):
            words.append(f"{MONTHS[ms[0] - 1]} to {MONTHS[ms[-1] - 1]} {y}, by month")
        else:
            words.append(_join([MONTHS[m - 1] for m in ms if 1 <= m <= 12]) + f" {y}")
    out: List[str] = []
    for w in words + rest:
        if w not in out:
            out.append(w)
    return _join(out)


OUTPUT_WORDS = {"number": ("number", "numbers"), "map": ("map", "maps"),
                "series": ("series", "series"),
                "elevation_split": ("summary by elevation band", "summaries by elevation band"),
                "overlap": ("comparison of two maps", "comparisons of two maps")}


def outputs_words(outputs: Sequence[Dict[str, Any]]) -> str:
    """A plan's required outputs, counted by what each one is."""
    c: Counter = Counter()
    for o in outputs or []:
        if not isinstance(o, dict):
            continue
        k = o.get("kind")
        c[("map" if o.get("region") is None else "number")
          if k in ("difference", "anomaly", "map") else str(k)] += 1
    return _join([f"{n} " + OUTPUT_WORDS.get(k, (k, k))[0 if n == 1 else 1]
                  for k, n in c.items()]) or "none"


def _compared(c: Dict[str, Any], normal_years: Sequence[int]) -> str:
    who = "the prepared run" if c.get("run") in (None, "control") else f"`{c.get('run')}`"
    other = ("the prepared run" if c.get("against") == "control" else
             f"its normal years ({_span(normal_years)})" if c.get("against") == "normal years"
             else str(c.get("against")))
    return f"{c.get('id')}: {who} against {other}, {_span(c.get('years'))}"


def page_lines(record: Optional[Dict[str, Any]], run_dir,
               design_sha256: Optional[str] = None) -> List[str]:
    """The design page's section on the analysis preview, from its record and
    the plan the record names. [] with no record. A record of another design
    is not shown: one line says so."""
    if not isinstance(record, dict):
        return []
    L = [HEADING, ""]
    if design_sha256 and record.get("design_sha256") and \
            record["design_sha256"] != design_sha256:
        return L + ["An analysis plan on record here was written for another design, so it "
                    "is not shown.", ""]
    L += ["The Analyzer's planning step wrote this from the design above, before any run, "
          "so that you can read it before you decide. " + NOT_BOUND, ""]
    status = record.get("status")
    if status == NOT_PLANNED:
        return L + [f"No plan was written: {_cell(record.get('why'))}.", ""]
    if status == FAILED:
        return L + [f"The plan could not be written: {_cell(record.get('why'))}. This does "
                    f"not stop the design: the Analyzer plans after the runs, as before.", ""]
    rd = Path(run_dir) if run_dir is not None else None
    plan: Dict[str, Any] = {}
    review: Dict[str, Any] = {}
    try:
        plan = json.loads((rd / record["plan_file"]).read_text())
        review = json.loads((rd / record["review_file"]).read_text())
    except (OSError, ValueError, TypeError, KeyError):
        pass
    if not plan:
        return L + ["The plan's file could not be read, so it is not shown.", ""]
    analyses = [a for a in plan.get("analyses") or [] if isinstance(a, dict)]
    b = plan.get("baseline") or {}
    ny = b.get("normal_years") or []
    again = " on its second writing" if int(record.get("attempts") or 1) > 1 else ""
    if status == PLANNED:
        L.append(f"- Code's check and the model review: **passed**{again}.")
    else:
        L.append(f"- Code's check and the model review: **not passed**. "
                 f"{_cell(record.get('why'))}. This does not stop the design in this "
                 f"version: the Analyzer plans again after the runs.")
    for c in plan.get("comparisons") or []:
        if isinstance(c, dict):
            L.append(f"- Compared: {_compared(c, ny)}.")
    if ny:
        L.append(f"- Normal years: {_span(ny)}"
                 + (f"; {_cell(b['normal_years_from'])}" if b.get("normal_years_from") else "")
                 + ".")
    if b.get("inclusive"):
        L.append(f"- On the normal years: {_cell(b['inclusive'])}.")
    fields: List[str] = []
    for a in analyses:
        fields += [f for f in a.get("fields") or [] if f not in fields]
    n_out = sum(len(a.get("outputs") or []) for a in analyses)
    L.append(f"- Read: {len(analyses)} analys{'is' if len(analyses) == 1 else 'es'}, "
             f"{n_out} required output{'' if n_out == 1 else 's'}, "
             f"{len(fields)} quantit{'y' if len(fields) == 1 else 'ies'}"
             + (f" ({', '.join(fields)})" if fields else "") + ".")
    if analyses:
        L += ["", "| | The question this analysis answers | Where | When | Required outputs |",
              "|---|---|---|---|---|"]
        for a in analyses:
            when = periods_words(a.get("periods") or [])
            if a.get("steps"):
                when += ("; " if when else "") + \
                    _join([STEP.get(s, str(s)) for s in a["steps"]]) + " series"
            L.append(f"| {_cell(a.get('id'))} | {_cell(a.get('question'))} | "
                     f"{_cell('; '.join(str(r) for r in a.get('regions') or []) or 'each cell')}"
                     f" | {_cell(when or 'none named')} | "
                     f"{_cell(outputs_words(a.get('outputs') or []))} |")
        L.append("")
    comp = [c for c in plan.get("competing") or [] if isinstance(c, dict)]
    if comp:
        L.append(f"- Other explanations it will check ({len(comp)}):")
        L += [f"  - {_cell(c.get('text'))}" for c in comp]
    sc = plan.get("screen") or {}
    if sc:
        L.append(f"- A screen of every quantity runs {sc.get('when')}, over "
                 f"{sc.get('region')} for {sc.get('year')}; what only the screen turns up "
                 f"is reported as found by screening, not as tested.")
    remarks = [x for x in (review.get("model_problems") or []) if str(x).strip()]
    if remarks:
        L.append(f"- The model review's remark{'s' if len(remarks) != 1 else ''}:")
        L += [f"  - {_cell(x)}" for x in remarks]
    problems = [x for x in (plan.get("problems") or []) if str(x).strip()]
    if problems:
        L.append(f"- What code's check found ({len(problems)}):")
        L += [f"  - {_cell(x)}" for x in problems[:8]]
        if len(problems) > 8:
            L.append(f"  - and {len(problems) - 8} more, in the plan's file")
    L += [f"- The plan in full: `{record.get('plan_file')}`.", ""]
    return L
