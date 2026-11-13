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

So with this switch alone the page is a look before the decision and nothing
more, and it must not be read as the approval of an analysis.

WITH THE HANDOFF'S SWITCH AS WELL (agents/analysis_handoff.py), a plan that
passed is promised to the Analyzer, and since stage 4 a yes given on its page
approves it. The record made here then says so (`handoff`), the page's words
come from that record, and a design whose plan did not pass or could not be
written cannot be approved at all. This module gives each analysis of a plan
it shows a status as well, by code (analysis_support).

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
            # in words the design page may print (2026-11-13): it names no
            # step of the framework
            rec["why"] = "the control run has no results store to plan the comparisons from"
            return _written(rd, rec)
        # the places the planning step may name, with their sizes and rules,
        # from the control's results store: the page states the mask an
        # analysis reads, by its cells and its area
        man = AP._read_json(Path(AP.suite_dirs(ctx).get("control", "")) / AP.SUITE_MANIFEST)
        rec["regions"] = {str(r["name"]): {k: r.get(k) for k in ("cells", "area_km2", "rule")}
                          for r in man.get("regions") or []
                          if isinstance(r, dict) and r.get("name")}
        adir = Path(ctx.run_dir) / "04_analysis"
        review, ok = S3.plan_and_review(ctx, adir, model=S2.analyzer_model(model),
                                        clients=clients)
        plan = AP.load_plan(adir)
        rel = adir.relative_to(rd)
        from agents import analysis_handoff as AH
        # WHETHER THIS PLAN WILL BE CARRIED OUT is decided here, with the page,
        # and kept in the record: after the runs the Analyzer reads the record
        # and not a switch (agents/analysis_handoff.py). Absent with that
        # switch off, and the record is then as it was.
        kept = AH.promise(bool(ok), new_runs=bool(
            (strategy.get("domain_experiment") or {}).get("runs")))
        if kept is not None:
            rec["handoff"] = kept
            if kept.get("promised"):
                rec["transitional"] = kept["words"]
        rec.update(status=PLANNED if ok else NOT_PASSED,
                   why=None if ok else AP.plan_passed(adir)[1],
                   plan_file=str(rel / AP.PLAN_FILE), review_file=str(rel / AP.REVIEW_FILE),
                   attempts=int(plan.get("attempt") or 1),
                   plan_sha256=AP.plan_sha256(adir),
                   review_passed=bool((review or {}).get("passed")),
                   model=S2.analyzer_model(model))
    except Exception as e:                                          # noqa: BLE001
        rec.update(status=FAILED, why=f"{type(e).__name__}: {e}"[:400])
        try:
            # no plan, so nothing is promised; the record still says whether a
            # plan was needed for a yes (analysis_handoff.unapprovable)
            from agents import analysis_handoff as AH
            none = AH.promise(False)
            if none is not None:
                rec["handoff"] = none
            rec["transitional"] = NOT_BOUND
        except Exception:                                           # noqa: BLE001
            pass
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


def _years_words(years: Sequence[int]) -> str:
    """Years in words: three or more in a row as a span, else each one."""
    ys = sorted({int(y) for y in years})
    if len(ys) >= 3 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]} to {ys[-1]}"
    return _join([str(y) for y in ys])


def _grouped(periods: Sequence[Any]) -> str:
    """The short page's words for a plan's period keys (2026-11-13). Years
    that share their months, or a season, are said once ("June to August of
    2033 and 2034", "summer of 2033 and 2034"), in the words the table of a
    new run uses for its months. Groups are set apart with "; ", so a list of
    years inside one is never read as two groups."""
    by_year: "OrderedDict[int, List[int]]" = OrderedDict()
    seasons: "OrderedDict[str, List[int]]" = OrderedDict()
    years: Dict[str, List[int]] = {"year": [], "wy": []}
    rest: List[str] = []
    for p in periods or []:
        s = str(p)
        m = re.fullmatch(r"(\d{4})-(\d{2})", s)
        q = re.fullmatch(r"(\d{4})-([A-Z]{3})", s)
        if m and 1 <= int(m.group(2)) <= 12:
            by_year.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        elif q:
            seasons.setdefault(q.group(2), []).append(int(q.group(1)))
        elif re.fullmatch(r"\d{4}", s) or re.fullmatch(r"WY\d{4}", s):
            years["wy" if s.startswith("WY") else "year"].append(int(s[-4:]))
        else:
            rest.append(s)
    same: "OrderedDict[tuple, List[int]]" = OrderedDict()
    for y, ms in by_year.items():
        same.setdefault(tuple(sorted(set(ms))), []).append(y)
    words = []
    for ms, ys in same.items():
        of = _years_words(ys)
        if len(ms) == 12:
            words.append(f"each month of {of}")
        elif len(ms) > 1 and list(ms) == list(range(ms[0], ms[-1] + 1)):
            words.append(f"{MONTHS[ms[0] - 1]} to {MONTHS[ms[-1] - 1]} of {of}")
        else:
            words.append(f"{_join([MONTHS[m - 1] for m in ms])} of {of}")
    for q, ys in seasons.items():
        words.append(f"{SEASONS.get(q, q)} of {_years_words(ys)}")
    if years["year"]:
        words.append(_years_words(years["year"]))
    if years["wy"]:
        n = len(set(years["wy"]))
        words.append(f"the water year{'' if n == 1 else 's'} {_years_words(years['wy'])}")
    out: List[str] = []
    for w in words + rest:
        if w not in out:
            out.append(w)
    return "; ".join(out)


def periods_words(periods: Sequence[Any], grouped: bool = False) -> str:
    """A plan's period keys in words: 2031-06 a month, 2031-JJA a season,
    2031 a year, WY2031 a water year; anything else as it is written. Three
    or more years in a row, or water years, are one span ("the years 2031 to
    2035"), placed where the first of them stood. `grouped`: the short
    page's words (_grouped)."""
    if grouped:
        return _grouped(periods)
    months, rest = OrderedDict(), []
    years: Dict[str, List[int]] = {"year": [], "wy": []}
    for p in periods or []:
        s = str(p)
        m = re.fullmatch(r"(\d{4})-(\d{2})", s)
        q = re.fullmatch(r"(\d{4})-([A-Z]{3})", s)
        if m:
            months.setdefault(m.group(1), []).append(int(m.group(2)))
        elif q:
            rest.append(f"{SEASONS.get(q.group(2), q.group(2))} {q.group(1)}")
        elif re.fullmatch(r"\d{4}", s) or re.fullmatch(r"WY\d{4}", s):
            kind = "wy" if s.startswith("WY") else "year"
            if not years[kind]:
                rest.append((kind,))                # where the first of its kind stood
            years[kind].append(int(s[-4:]))
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
    for w in rest:
        if not isinstance(w, tuple):
            words.append(w)
            continue
        name = "water year" if w[0] == "wy" else "year"
        ys = sorted(set(years[w[0]]))
        if len(ys) >= 3 and ys == list(range(ys[0], ys[-1] + 1)):
            words.append(f"the {name}s {ys[0]} to {ys[-1]}")
        else:
            words += [f"the {name} {y}" for y in ys]
    out: List[str] = []
    for w in words:
        if w not in out:
            out.append(w)
    return _join(out)


# THE STATUS OF EACH ANALYSIS, BEFORE THE YES (stage 4, 2026-11-12; the user's
# decision that a calculation the outputs cannot support gets an explicit
# status, not a warning in prose). Computed by code from the plan and its
# check, never written by a model:
#
#   blocked      code's check of the plan names a problem of this analysis.
#                Such a plan did not pass, and where a yes approves the plan
#                the design cannot be approved.
#   conditional  it can be carried out, under a condition a reader must know
#                before the yes. One is known to code today: every output of
#                the analysis is a map or a daily series, whose records keep
#                no value, so it gives figures and no number a sentence of the
#                report may state (domain_access.series_values).
#   supported    neither.
#
# What code cannot tell yet is a calculation the outputs only approximate,
# such as a monthly water budget from stored water kept as a monthly mean:
# the plan has no place to say what a calculation needs of a field.
SUPPORTED, CONDITIONAL, BLOCKED = "supported", "conditional", "blocked"
NO_NUMBER = ("it gives figures and no number the report may state, since each of its "
             "outputs is a map or a daily series")


def _no_number(o: Dict[str, Any]) -> bool:
    k = o.get("kind")
    return (k == "map" or (k == "series" and o.get("step") == "day")
            or (k in ("anomaly", "difference") and o.get("region") is None))


def analysis_support(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{analysis id: {"status", "why": [..]}} of a plan, in the plan's order
    (see above). A problem of code's check belongs to the analysis whose id
    it opens with."""
    out: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    analyses = [a for a in (plan or {}).get("analyses") or [] if isinstance(a, dict)]
    problems = [str(x) for x in (plan or {}).get("problems") or [] if str(x).strip()]
    for a in analyses:
        aid = str(a.get("id"))
        mine = [x for x in problems if re.match(rf"{re.escape(aid)}(?:[.:\s]|$)", x)]
        outs = [o for o in a.get("outputs") or [] if isinstance(o, dict)]
        if mine:
            out[aid] = {"status": BLOCKED, "why": mine}
        elif outs and all(_no_number(o) for o in outs):
            out[aid] = {"status": CONDITIONAL, "why": [NO_NUMBER]}
        else:
            out[aid] = {"status": SUPPORTED, "why": []}
    return out


def support_lines(plan: Dict[str, Any]) -> List[str]:
    """The reasons under the page's table: one line for each analysis that is
    not supported, [] when all are."""
    L = []
    for aid, s in analysis_support(plan).items():
        if s["status"] != SUPPORTED:
            L.append(f"- {aid} is {s['status']}: {_cell('; '.join(s['why']))}.")
    return L


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
    who = "the control run" if c.get("run") in (None, "control") else f"`{c.get('run')}`"
    other = ("the control run" if c.get("against") == "control" else
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
          "so that you can read it before you decide. "
          + after_the_runs(record, NOT_BOUND), ""]
    status = record.get("status")
    if status == NOT_PLANNED:
        return L + [f"No plan was written: {_cell(record.get('why'))}.", ""]
    from agents import analysis_handoff as AH
    no_plan = AH.unapprovable(record)
    if status == FAILED:
        return L + [f"The plan could not be written: {_cell(record.get('why'))}. "
                    + ("A design is approved together with its analysis plan, so this "
                       "design cannot be approved as it is." if no_plan else
                       "This does not stop the design: the Analyzer plans after the runs, "
                       "as before."), ""]
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
                 f"{_cell(record.get('why'))}. "
                 + ("A design is approved together with its analysis plan, so this design "
                    "cannot be approved as it is." if no_plan else
                    "This does not stop the design in this version: the Analyzer plans "
                    "again after the runs."))
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
        sup = analysis_support(plan)
        # THE SHORT TITLE AND WHAT IT TELLS (2026-11-13): the short page lists
        # an analysis by these and no id, so the details give them beside the
        # id, and a reader finds here each analysis the short page names.
        L += ["", "| | The question this analysis answers | Where | When | Required outputs "
                  "| Status | Short title | What it tells us |",
              "|---|---|---|---|---|---|---|---|"]
        for a in analyses:
            when = periods_words(a.get("periods") or [])
            if a.get("steps"):
                when += ("; " if when else "") + \
                    _join([STEP.get(s, str(s)) for s in a["steps"]]) + " series"
            L.append(f"| {_cell(a.get('id'))} | {_cell(a.get('question'))} | "
                     f"{_cell('; '.join(str(r) for r in a.get('regions') or []) or 'each cell')}"
                     f" | {_cell(when or 'none named')} | "
                     f"{_cell(outputs_words(a.get('outputs') or []))} | "
                     f"{(sup.get(str(a.get('id'))) or {}).get('status', SUPPORTED)} | "
                     f"{_cell(a.get('title') or 'none')} | {_cell(a.get('tells') or 'none')} |")
        L.append("")
        L += support_lines(plan)
        L += places_lines(analyses, record)
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
    # EVERY PROBLEM, WHOLE (2026-11-13): the first eight were printed and the
    # rest counted, while the short page printed each one. The short page now
    # says only what each problem stops, and the details keep them all.
    problems = [x for x in (plan.get("problems") or []) if str(x).strip()]
    if problems:
        L.append(f"- What code's check found ({len(problems)}):")
        L += [f"  - {_cell(x)}" for x in problems]
    later = not_made_yet(plan)
    if later:
        L.append(later)
    L += [f"- The plan in full: `{record.get('plan_file')}`.", ""]
    return L


def places_lines(analyses: Sequence[Dict[str, Any]], record: Dict[str, Any]) -> List[str]:
    """The places the analyses read, each with its cells, its area and its
    rule, as the control's results store gives them (the record's
    `regions`). In the details since 2026-11-13: the short page has no table
    of places, and the details keep what it left."""
    used: List[str] = []
    for a in analyses:
        used += [str(r) for r in a.get("regions") or [] if str(r) not in used]
    if not used:
        return []
    sizes = record.get("regions") if isinstance(record.get("regions"), dict) else {}
    L = ["", "The analyses read these places and no others.", "",
         "| Place | Cells | Area (km2) | Rule |", "|---|---|---|---|"]
    for r in used:
        g = sizes.get(r) if isinstance(sizes.get(r), dict) else {}
        cells = f"{g['cells']:,}" if isinstance(g.get("cells"), int) else "not on record"
        area = (f"{g['area_km2']:,.0f}" if isinstance(g.get("area_km2"), (int, float))
                else "not on record")
        L.append(f"| {_cell(r)} | {cells} | {area} | {_cell(g.get('rule') or 'not on record')} |")
    return L + [""]


def not_made_yet(plan: Dict[str, Any]) -> Optional[str]:
    """The details' line on a run whose outputs do not exist yet, or None.
    The short page said it among its points to weigh until 2026-11-13."""
    todo = list(dict.fromkeys(
        str(c.get("run")) for c in plan.get("comparisons") or []
        if isinstance(c, dict) and c.get("run") not in (None, "control")))
    if not todo:
        return None
    one = len(todo) == 1
    return (f"- {_join([f'`{t}`' for t in todo])} {'is' if one else 'are'} not made yet, so "
            f"the plan takes {'its' if one else 'their'} outputs to be the control run's "
            f"quantities over the planned years. Each year is checked after it runs.")


# ─────────────────────────────────────────────────────────────────────────────
# THE SHORT PAGE'S SECTION (agents/design_page.py)
# ─────────────────────────────────────────────────────────────────────────────
# The same plan record as page_lines above, which is the details file's
# section. REBUILT ON 2026-11-13 as one table in the reader's terms: what is
# compared, when, how, and what it tells us. No id, no count of quantities,
# no status column and no table of places, which the details keep. A
# comparison code finds conditional or blocked gets one sentence under the
# table that names it by its title and gives the reason. The page says "the
# comparisons" and never names a step of the framework.
MAIN_SHOWN = 4
# HOW A COMPARISON IS MADE, in plain words: one entry for each kind of output
# the plan contract knows (agents/analysis/plan_contract.KINDS), and "timing"
# for a timing measure, which is any other kind the plan's check lets
# through. A series is said with the runs it draws and its step (series_how).
HOW_WORDS = {
    "difference": "{run} minus the control run",
    "anomaly": "the departure from the mean of the normal years",
    "series": "{who} as {step} lines",
    "map": "a map over the grid",
    "elevation_split": "a split by elevation band",
    "overlap": "one map set against another",
    "timing": "the date it happens in each year",
}
PER_CELL = ", cell by cell"
STEP_UNIT = {"day": "day", "month": "month", "season": "season", "year": "year",
             "wateryear": "water year"}
CONDITIONAL_WORDS = ("gives figures and no number the report may state, since each of its "
                     "outputs is a map or a daily series")


def _sentence(text: Any) -> str:
    """A record's words as one sentence: a capital first, a full stop last."""
    s = " ".join(str(text or "").split())
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + "."


def after_the_runs(record: Optional[Dict[str, Any]], words: str) -> str:
    """What the details say becomes of the plan after the runs: `words` (the
    Analyzer plans again, and the yes approves no plan), or, for a plan the
    record promises to the Analyzer, the record's own words: that a yes
    approves it and that it is carried out as it stands."""
    from agents import analysis_handoff as AH
    return AH.page_words(record) if AH.promised(record) else words


def _read_plan(record: Dict[str, Any], run_dir):
    rd = Path(run_dir) if run_dir is not None else Path(".")
    try:
        return (json.loads((rd / record["plan_file"]).read_text()),
                json.loads((rd / record["review_file"]).read_text()))
    except (OSError, ValueError, TypeError, KeyError):
        return {}, {}


def _other_design(record, design_sha256) -> bool:
    return bool(design_sha256 and record.get("design_sha256")
                and record["design_sha256"] != design_sha256)


def normal_comparisons(record: Optional[Dict[str, Any]], run_dir,
                       design_sha256: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """The plan's comparisons of years with the normal years, each {"run",
    "years", "normal", "held"} (held: the years compared that the normal
    years contain). None where no plan of this design is shown, so the page
    says it from the design instead."""
    if not isinstance(record, dict) or _other_design(record, design_sha256) or \
            record.get("status") not in (PLANNED, NOT_PASSED):
        return None
    plan, _ = _read_plan(record, run_dir)
    if not isinstance(plan, dict) or not plan:
        return None
    b = plan.get("baseline") if isinstance(plan.get("baseline"), dict) else {}
    ny = [int(y) for y in b.get("normal_years") or [] if str(y).isdigit()]
    held = [int(y) for y in b.get("includes_compared_years") or [] if str(y).isdigit()]
    out = []
    for c in plan.get("comparisons") or []:
        if isinstance(c, dict) and c.get("against") == "normal years":
            ys = sorted({int(y) for y in c.get("years") or [] if str(y).isdigit()})
            out.append({"run": c.get("run") or "control", "years": ys, "normal": ny,
                        "held": [y for y in held if y in ys]})
    return out


def compare_words(a: Dict[str, Any]) -> str:
    """What an analysis compares, for the page: its title; without one, its
    fields joined in words; without those, its question."""
    return str(a.get("title") or "").strip() or \
        _join([str(f) for f in a.get("fields") or []]) or str(a.get("question") or "")


def _series_when(a: Dict[str, Any], de: Dict[str, Any], row: Dict[str, Any]) -> str:
    """The years of an analysis's series. A series names no period of its
    own: it covers the new runs' years, or, for a series of the control run
    alone, the control run's whole output."""
    outs = [o for o in a.get("outputs") or [] if isinstance(o, dict)
            and o.get("kind") == "series"]
    steps = [s for s in dict.fromkeys(o.get("step") for o in outs) if s] or \
        list(a.get("steps") or [])
    if not steps:
        return ""
    oy = (row.get("facts") or {}).get("output_years") or []
    runs = {str(o.get("run") or "control") for o in outs} or {"control"}
    if runs == {"control"} and oy:
        ys = list(range(int(oy[0]), int(oy[-1]) + 1))
    else:
        ys = [int(y) for y in de.get("years") or [] if str(y).isdigit()]
    of = f" of {_years_words(ys)}" if ys else ""
    return _join([f"each {STEP_UNIT.get(str(s), str(s))}{of}" for s in steps])


def when_words(a: Dict[str, Any], de: Dict[str, Any], row: Dict[str, Any]) -> str:
    """When an analysis looks: its periods, as periods_words groups them, and
    the years of its series. Without either, the periods its outputs name."""
    parts = [w for w in (periods_words(a.get("periods") or [], grouped=True),
                         _series_when(a, de, row)) if w]
    if not parts:
        own = [o.get("period") for o in a.get("outputs") or []
               if isinstance(o, dict) and o.get("period") is not None]
        parts = [w for w in (periods_words(own, grouped=True),) if w]
    return "; ".join(parts) or "not named"


def series_how(outs: Sequence[Dict[str, Any]], n_runs: int = 1) -> str:
    """A series in plain words: the runs it draws and its step."""
    runs = list(dict.fromkeys(str(o.get("run") or "control") for o in outs))
    steps = [s for s in dict.fromkeys(str(o.get("step") or "") for o in outs) if s]
    step = _join([STEP.get(s, s) for s in steps]) or "its"
    if len(runs) == 1:
        who = ("the control run" if runs[0] == "control" else
               "the new run" if n_runs == 1 else f"`{runs[0]}`")
        if len(outs) == 1:
            return f"{who} as a {step} line"
    else:
        who = "the two runs" if len(runs) == 2 and "control" in runs else "the runs"
    return HOW_WORDS["series"].format(who=who, step=step)


def how_words(a: Dict[str, Any], n_runs: int = 1) -> str:
    """How an analysis compares, in plain words from HOW_WORDS, one phrase
    for each kind of its outputs, joined with "and"."""
    from agents.analysis import plan_contract as PC
    outs = [o for o in a.get("outputs") or [] if isinstance(o, dict)]
    said: List[str] = []
    for o in outs:
        k = str(o.get("kind"))
        if k == "series":
            w = series_how([x for x in outs if x.get("kind") == "series"], n_runs)
        elif k == "difference":
            w = HOW_WORDS[k].format(run="the new run" if n_runs == 1 else f"`{o.get('run')}`")
        else:
            w = HOW_WORDS[k if k in PC.KINDS else "timing"]
        if k in PC.NULL_REGION and o.get("region") is None:
            w += PER_CELL
        if w not in said:
            said.append(w)
    return _join(said) or "not named"


def _without_id(aid: str, problem: Any) -> str:
    """A problem of code's check without the analysis id it opens with."""
    return re.sub(rf"^{re.escape(aid)}(?:\.\w+)?\s*[:.]?\s*", "", str(problem)).strip()


def support_sentences(plan: Dict[str, Any]) -> List[str]:
    """One sentence for each comparison code finds conditional or blocked,
    naming it by its title and giving the reason. [] when all are
    supported."""
    by_id = {str(a.get("id")): a for a in plan.get("analyses") or [] if isinstance(a, dict)}
    out = []
    for aid, s in analysis_support(plan).items():
        name = _cell(compare_words(by_id.get(aid) or {}))
        if s["status"] == CONDITIONAL:
            out.append(f'The comparison "{name}" {CONDITIONAL_WORDS}.')
        elif s["status"] == BLOCKED:
            out.append(f'The comparison "{name}" cannot be made as planned. '
                       + " ".join(_sentence(_cell(_without_id(aid, w))) for w in s["why"]))
    return out


def _after(n_runs: Optional[int]) -> str:
    """When the comparisons are chosen, where no plan is shown."""
    return "your yes" if not n_runs else "the run" if n_runs == 1 else "the runs"


def short_lines(record: Optional[Dict[str, Any]], run_dir, design_sha256: Optional[str],
                de: Dict[str, Any], row: Dict[str, Any], n_runs: Optional[int] = None
                ) -> List[str]:
    """The short page's section "How your question will be answered", after
    the sentence that says what is compared with what: the table of the
    comparisons, a sentence for each that is conditional or blocked, the
    other explanations, and what becomes of the comparisons after the yes.

    AT MOST FOUR ROWS (2026-11-13): those the plan marks `main`, or its first
    four where it marks none. The rest are counted, and the details list
    every one."""
    if n_runs is None:
        n_runs = len([r for r in de.get("runs") or [] if isinstance(r, dict)])
    after = _after(n_runs)
    if not isinstance(record, dict):
        return [f"The comparisons are chosen after {after}, so none is listed here.", ""]
    if _other_design(record, design_sha256):
        return ["The comparisons on record here were planned for another design, so they are "
                "not shown.", ""]
    from agents import analysis_handoff as AH
    no_plan = AH.unapprovable(record)
    status = record.get("status")
    if status == NOT_PLANNED:
        return [f"The comparisons are chosen after {after}, so none is listed here. None "
                f"could be planned before this page, since {_cell(record.get('why'))}.", ""]
    if status == FAILED:
        return [f"The comparisons could not be planned before this page. The error was "
                f"{_cell(record.get('why'))}."
                + ("" if no_plan else f" This does not stop the design, since they are then "
                                      f"chosen after {after}."), ""]
    plan, _ = _read_plan(record, run_dir)
    if not isinstance(plan, dict) or not plan:
        return ["The list of comparisons could not be read, so none is shown here.", ""]
    analyses = [a for a in plan.get("analyses") or [] if isinstance(a, dict)]
    L: List[str] = []
    if analyses:
        main = [a for a in analyses if a.get("main") is True]
        shown = (main or analyses)[:MAIN_SHOWN]
        L += ["| What we compare | When | How | What it tells us |", "|---|---|---|---|"]
        for a in shown:
            L.append(f"| {_cell(compare_words(a))} | {_cell(when_words(a, de, row))} | "
                     f"{_cell(how_words(a, n_runs))} | "
                     f"{_cell(a.get('tells') or a.get('question'))} |")
        L.append("")
        more = len(analyses) - len(shown)
        if more:
            L += [f"{more} more comparison{' is' if more == 1 else 's are'} listed in "
                  f"`DESIGN_DETAILS.md`.", ""]
        why = support_sentences(plan)
        if why:
            L += [" ".join(why), ""]
    comp = [c for c in plan.get("competing") or [] if isinstance(c, dict)]
    if comp:
        titles = [str(c.get("title") or "").strip() for c in comp]
        n = len(comp)
        L += [f"We will also check {'one other explanation' if n == 1 else f'{n} other explanations'}"
              f", which {'is' if n == 1 else 'are'} "
              + (_join(titles) if all(titles) else "listed in `DESIGN_DETAILS.md`") + ".", ""]
    if status == NOT_PASSED:
        ids = [str(a.get("id")) for a in analyses]
        loose = [str(x) for x in plan.get("problems") or [] if str(x).strip() and not any(
            re.match(rf"{re.escape(i)}(?:[.:\s]|$)", str(x)) for i in ids)]
        L.append("These comparisons did not pass their check."
                 + ("" if no_plan else f" This does not stop the design, since they are "
                                       f"chosen again after {after}."))
        L += [f"- {_cell(x)}" for x in loose] + [""]
    elif AH.bound(record):
        pass                            # the yes says so, in "Your decision"
    elif AH.promised(record):
        L += [f"These comparisons are carried out as listed after {after}.", ""]
    else:
        # A PLAN SHOWN THAT A YES DOES NOT APPROVE (the preview's switch
        # alone): said where it is shown, since nothing else on the page does
        L += [f"These comparisons are a preview. Your yes does not approve them, and they are "
              f"chosen again after {after}.", ""]
    return L
