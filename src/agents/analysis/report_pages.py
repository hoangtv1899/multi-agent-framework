#!/usr/bin/env python3
"""
The four pages of a whole-domain report
src/agents/analysis/report_pages.py

    write_pages(report, out_dir) -> {page name: path}

A finished whole-domain study used to hand its reader one REPORT.md of
fourteen sections, with codes such as "O1 (L2)" and the same caveat printed
four times. The person's verdict (2026-11-13): "too convoluted without any
clear structure of a scientific report". The same record now gives four
pages, written by code into 04_analysis/, where REPORT.md always was:

    REPORT.md             the scientific report, in the order of a paper:
                          Summary, Methods, Results, Limitations, Next steps,
                          Checks and records. Its figures lead and its text
                          supports them
    SCREEN.md             the first look at every quantity, when a screen ran
    PLANNED_ANALYSES.md   each approved comparison in words, what became of
                          it and where the report shows it
    REPORT_DETAILS.md     everything else: every finding with its records,
                          the findings left out and why, every caveat whole,
                          the framework's wording rules, what code checked,
                          the model set-up, provenance

WHAT IT MAY NOT DO. It computes nothing and words no finding. A sentence in
the Summary or under a figure is an audited sentence of the Analyzer, word
for word. Every other sentence is fixed here, and every number on a page is
a record's value, placed by code. No plan code, record name or identifier
appears on the three short pages; the details page may show them. Each item
of the old page appears on one of the four pages, once.

IT NEVER STOPS A REPORT. A page whose source is missing says so in one line.
The record (analysis.json) is written before these pages, and step4_report
writes the old page if this module raises.

THE FIGURES AND THE CHECK'S SENTENCE COME FROM THEIR OWN MODULES. The
figures and tables are report_figures' record, the names in words are
field_words', and the sentence on the second reading of the numbers is
numbers_check.evidence_sentence. This module places them and words nothing
of theirs.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agents.analysis.dashes import strip_dashes

REPORT_MD = "REPORT.md"
DETAILS_MD = "REPORT_DETAILS.md"
SCREEN_MD = "SCREEN.md"
PLANNED_MD = "PLANNED_ANALYSES.md"
DESIGN_MD = "DESIGN_REVIEW.md"
PAGES = (REPORT_MD, DETAILS_MD, PLANNED_MD, SCREEN_MD)

SECTIONS = ("1. Summary", "2. Methods", "3. Results", "4. Limitations", "5. Next steps",
            "6. Checks and records")
MAX_LIMITS = 6                  # bullets under Limitations
MAX_PER_FIGURE = 2              # findings under a figure
MAX_NEXT = 3                    # lines under Next steps

QUESTION_KIND = {
    "exploratory": ("This was an exploratory question. Its findings come from looking at the "
                    "results, and none of them was set out as an expectation before the results "
                    "were seen."),
    "hypothesis": ("This was a question with a hypothesis. The plan set out its expectations "
                   "before any result was looked up.")}
SCORE_WORDS = ("A score is the change divided by how much that period usually varies from year "
               "to year, the standard deviation of the control run's normal years.")
LOOKING_WORDS = ("The rows were found by looking at every quantity and period, not by testing "
                 "an expectation, so a large score shows where to look and is not a finding.")
JUDGEMENT_WORDS = ("Whether the answer follows from the findings is for the reader to judge, "
                   "since no code judges it.")
SUPPORT_WORDS = ("A comparison that is complete has produced its numbers. Whether they support "
                 "an expectation is said only by the findings that test it.")
NO_CAUSE = ("No run in this study isolates a cause. These results are what happened together "
            "within the model, not what caused what.")
FIXED_RULES = ("*These are the framework's fixed rules for which words a finding may use, the "
               "same on every report.*")
NO_DESIGN = "This study holds no design page."
LATER_START = ("The new run starts from the control run's saved state of {date}, and no run that "
               "starts this way had been completed before, so its start is untested.")
LATER_START_NO_DATE = ("The new run starts from a later saved state of the control run, and no "
                       "run that starts this way had been completed before, so its start is "
                       "untested.")

# THE SCREEN'S RULES IN PLAIN WORDS (2026-11-13). The server's rule text is
# one long string with clauses, codes and brackets ("ddof 1", "the
# catalog's", "above_normal_maximum"), and cut at its semicolons it read as
# fragments. These sentences are fixed here, the same on every report, and a
# test holds their key facts to the server's own rule text. The server's text
# stays whole on the details page.
MARK_WORDS = {"above_normal_maximum": "above the largest value of the normal years",
              "differs_from_control": "differs from the control run"}
STATUS_WORDS = {"near-constant normal": "the normal years hardly vary",
                "insufficient support": "too few normal years",
                "period absent": "the run lacks the period",
                "no type in the field notes": "no type on record"}
SCORING_SENTENCES = (
    "A score is given only to a flow, such as rain or plant growth, or to a state, such as "
    "topsoil water.",
    "The score is the change divided by the standard deviation of the normal years for the same "
    "period.",
    "For a new run, the change is the new run minus the control run, and the standard deviation "
    "is that of the control run's normal years.",
    "A score needs at least two normal years whose values cover the whole region, and a row "
    "without them gets no score.",
    "A normal year that lacks the period lowers the count of normal years shown.",
    "A row whose normal years hardly vary gets no score, so that a tiny spread cannot give a huge "
    "score.",
    "A change of exactly zero gets no score.",
    "Stores, such as plant carbon, and indices held between fixed bounds are not scored yet, and "
    "their change stays in the table.",
    "Fire rows, the area burned and the carbon lost to fire, get no score.",
    "For a new run, a fire row is marked when it differs from the control run.",
    "For the control run alone, a fire row is marked when it is above the largest value of the "
    "normal years.")
DISPLAY_SENTENCES = (
    "The rows are shown by process group, the groups in the model's order.",
    "Each group shows first its three rows with the largest score in size, the largest first.",
    "A quantity has at most one row there, its period with the largest score, and its other "
    "periods stay in the whole table.",
    "A tie keeps the quantity that comes first in the model's list and, within a quantity, the "
    "period that comes first in the table.",
    "A group without a scored row shows none.",
    "Then come all the marked fire rows, in the table's order.",
    "Then, for each quantity not scored yet, comes its largest change with its period.",
    "Every score is shown with the number of normal years behind it.",
    "A row without a score is never shown first and stays in the whole table.")


# ─────────────────────────────────────────────────────────────────────────────
# WORDS
# ─────────────────────────────────────────────────────────────────────────────
def _p(text) -> str:
    """Recorded prose as a reader sees it: dashes used as punctuation become
    commas. The record itself is untouched."""
    return strip_dashes(str(text if text is not None else ""))


def _fv(v) -> str:
    from agents.analysis.step4_report import format_value
    return format_value(v)


def _join(xs: Sequence[str]) -> str:
    xs = [str(x) for x in xs if str(x)]
    return ("" if not xs else xs[0] if len(xs) == 1 else
            ", ".join(xs[:-1]) + " and " + xs[-1])


def _cap(s: str) -> str:
    s = str(s or "").strip()
    return s[:1].upper() + s[1:]


def _sentence(s: Any) -> str:
    """A text as one sentence: its first letter capital, a full stop at its end."""
    s = _cap(_p(s)).rstrip()
    return s if not s or s[-1] in ".!?" else s + "."


def span_words(years: Any) -> str:
    """Years in words: "2018 to 2020" for a run of three or more, else a list
    ("2018 and 2019")."""
    ys = sorted({int(y) for y in (years or []) if str(y).lstrip("-").isdigit()})
    if not ys:
        return ""
    if len(ys) > 2 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]} to {ys[-1]}"
    return _join([str(y) for y in ys])


def ints(xs: Any) -> List[int]:
    """The whole numbers of a list, each once, in order; anything that is
    not a whole number is left out, so an odd value never stops a page."""
    out: List[int] = []
    for x in xs if isinstance(xs, (list, tuple, set)) else []:
        if isinstance(x, bool):
            continue
        if isinstance(x, int) or (isinstance(x, str) and re.fullmatch(r"\d+", x.strip())):
            n = int(x)
        elif isinstance(x, float) and x.is_integer():
            n = int(x)
        else:
            continue
        if n not in out:
            out.append(n)
    return out


def numbers_words(nums: Sequence[int], word: str = "comparison") -> str:
    """"comparison 5", "comparisons 3, 7 and 8", "comparisons 1 to 4 and 6
    to 8": a run of three or more as a span, every other number alone."""
    ns = sorted(ints(list(nums)))
    if not ns:
        return ""
    runs: List[List[int]] = []
    for n in ns:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts: List[str] = []
    for r in runs:
        parts += [f"{r[0]} to {r[-1]}"] if len(r) > 2 else [str(x) for x in r]
    return f"{word}{'s' if len(ns) > 1 else ''} {_join(parts)}"


def period_words(period: Any) -> str:
    """A period as a person writes it: "September 2018", "summer 2018",
    "winter 2017 to 2018", "water year 2018", "the year 2018"."""
    from agents.analysis import field_words as FW
    p = str(period or "")
    if re.fullmatch(r"\d{4}", p):
        return f"the year {p}"
    return str(FW.period_words(p))


def region_words(region: Any) -> str:
    """A region of the results store in words: "the watershed's vegetated
    land", "the watershed's vegetated land at 1000-1500 m"; an unknown
    region by its name."""
    parts = [x.strip() for x in str(region or "").split(",") if x.strip()]
    if not parts:
        return "the region on record"
    base, rest = parts[0], parts[1:]
    band = next((x for x in rest if re.search(r"\d", x)), None)
    veg = any(x.startswith("vegetated") for x in rest)
    at = "" if not band else (f" {band}" if band.startswith("below") else f" at {band}")
    if base in ("vegetated", "vegetated cells") and not rest:
        return "the vegetated land of the whole grid"
    if base == "watershed":
        return ("the watershed's vegetated land" if veg else "the watershed") + at
    return f"the region `{base}`" + (", its vegetated land" if veg else "") + at


def field_name(field: Any, with_code: bool = True) -> str:
    """A model quantity's short plain name, with its code in brackets."""
    from agents.analysis import field_words as FW
    return str(FW.words(field, with_code=with_code))


def units_name(units: Any) -> str:
    """A unit as a person writes it ("gC/m2" for "gC/m^2/month")."""
    from agents.analysis import field_words as FW
    return str(FW.units_words(units))


def _dicts(xs: Any) -> List[Dict[str, Any]]:
    """The entries of a list that are records (dicts); anything else is left
    out, so an odd value never stops a page."""
    return [x for x in xs if isinstance(x, dict)] if isinstance(xs, list) else []


def identifiers(report: Dict[str, Any]) -> Set[str]:
    """The record's own names: caveats, findings, records, plan and output
    codes. None of them may show on the three short pages."""
    ph = report.get("planned_phase") if isinstance(report.get("planned_phase"), dict) else {}
    ids = {str(c.get("id")) for c in _dicts(report.get("caveats")) if c.get("id")}
    ids |= {str(c.get("finding_id")) for c in _dicts(report.get("claims"))
            + _dicts(report.get("withheld")) if c.get("finding_id")}
    ids |= set(records_of(report))
    for a in _dicts(ph.get("analyses")):
        ids.add(str(a.get("id")))
        ids |= {str(o.get("id")) for o in _dicts(a.get("outputs")) if o.get("id")}
        ids |= {str(o.get("record")) for o in _dicts(a.get("outputs")) if o.get("record")}
    ids |= {str(e.get("id")) for e in _dicts(ph.get("expectations")) if e.get("id")}
    return {i for i in ids if i and i != "None"}


def without_ids(text: Any, ids: Set[str]) -> str:
    """An Analyzer's sentence for a short page: a bracket that holds only the
    record's own names ("(limitation_structural_1)", "(L14, L15)") is taken
    out. Every other word stays as the audit kept it."""
    def drop(m):
        inner = [x.strip() for x in m.group(1).split(",")]
        return "" if inner and all(x in ids for x in inner) else m.group(0)
    return re.sub(r"\s*\(([^()]*)\)", drop, _p(text))


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORDS A PAGE READS
# ─────────────────────────────────────────────────────────────────────────────
def _read_json(path) -> Dict[str, Any]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def study_dir(report: Dict[str, Any], out_dir) -> Path:
    """The study folder: the parent of 04_analysis/, else the record's own."""
    out = Path(out_dir)
    return out.parent if out.name == "04_analysis" else Path(report.get("run_dir") or out)


def records_of(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    et = report.get("evidence_trace") if isinstance(report.get("evidence_trace"), dict) else {}
    recs = et.get("records")
    return {str(k): v for k, v in recs.items() if isinstance(v, dict)} \
        if isinstance(recs, dict) else {}


def question_type(report: Dict[str, Any], plan: Dict[str, Any]) -> Optional[str]:
    return ((report.get("evidence_trace") or {}).get("question_type")
            or plan.get("question_type")
            or (report.get("planned_phase") or {}).get("question_type"))


def analyses(plan: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The approved comparisons in the order the design page numbers them:
    the plan's analyses, or the planned phase's where the plan is missing."""
    rows = [a for a in plan.get("analyses") or [] if isinstance(a, dict)]
    if rows:
        return rows
    return [a for a in (report.get("planned_phase") or {}).get("analyses") or []
            if isinstance(a, dict)]


def numbers_of(plan: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, int]:
    """{analysis id: its number as the design page shows it, from 1}."""
    return {str(a.get("id")): k for k, a in enumerate(analyses(plan, report), 1)}


def claim_comparison(c: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                     numbers: Dict[str, int], lookups: Optional[Dict[str, List[str]]] = None,
                     figure_items: Optional[Dict[str, List[str]]] = None) -> Optional[int]:
    """The comparison a kept finding answers, by the first route that gives
    one: the comparison its finding names ("required_p3" is the plan's
    analysis P3); else the smallest comparison the records it cites were
    made for (each line of lookups.jsonl carries its plan items, and so does
    the trace's copy of it); else the smallest of the plan items of the
    Analyzer's own figure that the finding names (investigation.json).
    None when no route gives one."""
    if not isinstance(c, dict):
        return None
    m = re.fullmatch(r"required_(p\d+)", str(c.get("finding_id") or ""), re.I)
    if m and m.group(1).upper() in numbers:
        return numbers[m.group(1).upper()]
    got = []
    for rid in c.get("evidence") or []:
        items = (lookups or {}).get(str(rid)) or (records.get(str(rid)) or {}).get("plan_items")
        got += [numbers[str(i)] for i in items or [] if str(i) in numbers]
    if not got:
        got = [numbers[str(i)] for i in (figure_items or {}).get(str(c.get("finding_id"))) or []
               if str(i) in numbers]
    return min(got) if got else None


def comparison_of(report: Dict[str, Any], c: Dict[str, Any],
                  numbers: Dict[str, int]) -> Optional[int]:
    """claim_comparison with the study's own records, as write_pages read them."""
    return claim_comparison(c, records_of(report), numbers, report.get("_lookups"),
                            report.get("_figure_items"))


def lookup_items(out_dir) -> Dict[str, List[str]]:
    """{record name: its plan items}, from 04_analysis/lookups.jsonl, whose
    line n is the record "Ln" (blank lines not counted, as the log's reader
    counts them); {} when the log is missing."""
    out: Dict[str, List[str]] = {}
    try:
        lines = [x for x in (Path(out_dir) / "lookups.jsonl").read_text().splitlines()
                 if x.strip()]
    except OSError:
        return out
    for k, line in enumerate(lines, 1):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("plan_items"), list):
            out[f"L{k}"] = [str(x) for x in d["plan_items"]]
    return out


def figure_items(out_dir) -> Dict[str, List[str]]:
    """{the Analyzer's figure id: its plan items}, from investigation.json."""
    inv = _read_json(Path(out_dir) / "investigation.json")
    out: Dict[str, List[str]] = {}
    for key in ("figure_outcomes", "figures"):
        for f in inv.get(key) or []:
            if isinstance(f, dict) and f.get("id") and isinstance(f.get("plan_items"), list):
                out.setdefault(str(f["id"]), [str(x) for x in f["plan_items"]])
    return out


def used_records(report: Dict[str, Any]) -> List[str]:
    """The records the kept findings cite and the tables print, in order."""
    out: List[str] = []
    for c in _dicts(report.get("claims")):
        out += [str(r) for r in c.get("evidence") or [] if str(r) not in out]
    rf = report.get("report_figures") if isinstance(report.get("report_figures"), dict) else {}
    out += [str(r) for r in rf.get("printed_records") or [] if str(r) not in out]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPORT.md
# ─────────────────────────────────────────────────────────────────────────────
def _runs(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The new runs, as the model set-up holds them (the control left out)."""
    return [r for r in ((report.get("model_setup") or {}).get("runs") or [])
            if isinstance(r, dict) and r.get("run") != "control" and r.get("changes")]


def _years(report: Dict[str, Any]) -> List[int]:
    y = (report.get("model_setup") or {}).get("years") or {}
    comp = (report.get("cost") or {}).get("compute") or {}
    return list(y.get("reported") or comp.get("years_planned") or [])


def title_lines(report: Dict[str, Any]) -> List[str]:
    q = _p(report.get("question") or "(no question recorded)")
    name = Path(str(report.get("run_dir") or "")).name
    runs = [f"`{r.get('run')}`" for r in _runs(report)]
    ys = span_words(_years(report))
    who = (f"The new run {runs[0]} against the control run" if len(runs) == 1 else
           f"The new runs {_join(runs)} against the control run" if runs else
           "The control run alone")
    L = [f"# {q}", "", (f"Study `{name}`. " if name else "") + who + (f", {ys}." if ys else ".")]
    return L + [""] + not_tested_lines(report)


def not_tested_lines(report: Dict[str, Any]) -> List[str]:
    """In words, under the title: a required comparison not completed, and a
    required number found not estimable. The codes stay in the record."""
    plan = report.get("_plan") or {}
    numbers = numbers_of(plan, report)
    acc = report.get("acceptance") or {}
    L: List[str] = []
    blockers = acc.get("blockers") or []
    if blockers:
        parts = [f"comparison {numbers.get(str(b.get('analysis')), '?')} "
                 + ("failed" if b.get("status") == "failed" else f"was {b.get('status')}")
                 for b in blockers]
        L += [f"**The hypothesis was not fully tested.** {_cap(_join(parts))}, and the "
              f"plan's expectations need {'them' if len(blockers) > 1 else 'it'}. This does "
              f"not count as a successful test. {PLANNED_MD} says why.", ""]
    un = acc.get("untestable") or []
    if un:
        n = sum(len(u.get("outputs") or []) for u in un)
        comps = [numbers.get(str(u.get("analysis")), "?") for u in un]
        L += [f"**Part of the hypothesis cannot be tested from this run.** {n} required "
              f"number{'s' if n != 1 else ''} of {numbers_words(comps)} could not be "
              f"estimated. The task is complete, and a value that cannot be estimated "
              f"supports no finding either way.", ""]
    return L


def summary_text(report: Dict[str, Any]) -> str:
    """The Analyzer's answer as the Summary shows it: the audited answer
    where the record holds one, else the answer."""
    text = report.get("answer_kept") if "answer_kept" in report else report.get("answer")
    return str(text or "")


def summary_lines(report: Dict[str, Any]) -> List[str]:
    L = [f"## {SECTIONS[0]}", ""]
    if report.get("status") == "inconclusive" and report.get("headline"):
        L += [f"**{_p(report['headline'])}**", ""]
    ids = identifiers(report)
    if "answer_kept" in report:
        n = len(report.get("answer_left_out") or [])
        L += [without_ids(report.get("answer_kept") or "No sentence of the Analyzer's summary "
                                                         "rests on the kept findings alone.",
                          ids), "",
              f"*{n} sentence{'s' if n != 1 else ''} of the Analyzer's summary "
              f"{'are' if n != 1 else 'is'} left out here, because each gives a number that "
              f"no kept finding gives. The details page shows "
              f"{'them' if n != 1 else 'it'}.*", ""]
    else:
        L += [without_ids(report.get("answer") or "The Analyzer gave no answer.", ids), ""]
    kind = QUESTION_KIND.get(question_type(report, report.get("_plan") or {}) or "")
    if kind:
        L += [kind, ""]
    return L


def control_line(sd: Path, report: Dict[str, Any]) -> str:
    """What the control run is, in the design page's first sentence about
    it, else the model server's description, with a pointer to the page."""
    strategy = _read_json(sd / "strategy.json")
    de = strategy.get("domain_experiment") or {}
    name = (de.get("control") or {}).get("name") \
        or ((report.get("model_setup") or {}).get("control") or {}).get("name")
    row = next((c for c in ((strategy.get("domain_rulebook") or {}).get("controls") or {})
                .get("controls") or [] if isinstance(c, dict) and c.get("name") == name), {})
    text = [str(t) for t in ((row.get("page_summary") or {}).get("text") or []) if str(t).strip()]
    first = re.split(r"(?<=\.)\s+", text[0].strip())[0] if text else ""
    if not first:
        desc = str(((report.get("model_setup") or {}).get("control") or {}).get("description")
                   or "").split(":")[0].strip()
        first = (f"The control run is {desc}." if desc else
                 f"The control run is `{name}`." if name else
                 "No description of the control run is on record.")
    if not (sd / DESIGN_MD).is_file():
        return f"{_p(first)} {NO_DESIGN}"
    return (f"{_p(first)} The design page, [{DESIGN_MD}](../{DESIGN_MD}), describes it and sets "
            f"it beside measurements.")


def _hours(seconds: Any) -> str:
    try:
        return f"{_fv(float(seconds) / 3600.0)} hours"
    except (TypeError, ValueError):
        return "not on record"


def run_table(sd: Path, report: Dict[str, Any]) -> List[str]:
    """One row per knob of each new run: what it changes, how, when and where
    (the design page's own rows, 01_inputs/run_given.json), the years it
    simulated and its computing time."""
    runs = _runs(report)
    comp = (report.get("cost") or {}).get("compute") or {}
    secs = comp.get("runs") or {}
    years = span_words(comp.get("years_passed")) or "not on record"
    try:
        from agents import run_figures as RF
        given = RF.load(sd)
    except Exception:                                               # noqa: BLE001
        RF, given = None, None
    meanings = {r: [c.get("meaning") for c in (v or {}).get("changes") or []]
                for r, v in ((report.get("evidence_trace") or {}).get("runs") or {}).items()}
    body = []
    for r in runs:
        name = r.get("run")
        rows = RF.rows(given, name) if RF is not None and given else []
        if not rows:
            words = "; ".join(_p(m) for m in meanings.get(name) or [] if m) or \
                "; ".join(_p(c.get("summary")) if c.get("summary") else f"the change `{c.get('id')}`"
                          for c in _dicts(r.get("changes")))
            rows = [{"knob": "", "change": words, "when": "", "where": ""}]
        for k, row in enumerate(rows):
            cells = [f"`{name}`" if k == 0 else ""] + [
                " ".join(_p(row.get(c)).split()).replace("|", "/")
                for c in ("knob", "change", "when", "where")]
            cells += [years if k == 0 else "", _hours(secs.get(name)) if k == 0 else ""]
            body.append("| " + " | ".join(cells) + " |")
    if not body:
        return []
    return ["| New run | Knob | Change | When | Where | Years simulated | Computing time |",
            "|---|---|---|---|---|---|---|"] + body + [""]


def methods_lines(sd: Path, report: Dict[str, Any]) -> List[str]:
    L = [f"## {SECTIONS[1]}", "", control_line(sd, report), ""]
    runs = _runs(report)
    comp = (report.get("cost") or {}).get("compute") or {}
    if runs:
        L += run_table(sd, report)
        planned, passed = comp.get("years_planned") or [], comp.get("years_passed") or []
        who = "the new run" if len(runs) == 1 else "the new runs"
        if not planned and not passed:
            checks = f"The record holds no check of the years of {who}."
        elif planned and sorted(planned) == sorted(passed) and not comp.get("first_unpassed"):
            checks = (f"Each year of {who}, {span_words(passed)}, passed the checks of its "
                      f"job.")
        else:
            checks = (f"Only {span_words(passed) or 'none'} of the years "
                      f"{span_words(planned) or 'planned'} passed the checks of their jobs"
                      + (f", and the chain stopped at {comp['first_unpassed']}."
                         if comp.get("first_unpassed") else "."))
    else:
        checks = ("No new run was made. The control run's existing output was read"
                  + (f" for {span_words(_years(report))}." if _years(report) else "."))
    handed = ((report.get("analysis_plan") or {}).get("handoff") or {}).get("used")
    if not report.get("planned_phase"):
        how = "No planned comparison is on record for this study."
    elif handed:
        how = (f"The approved comparisons were carried out as the design page listed them. "
               f"[{PLANNED_MD}]({PLANNED_MD}) gives each one and where this report shows it.")
    else:
        how = (f"The Analyzer planned its comparisons before looking up any result, and the "
               f"plan passed its review. [{PLANNED_MD}]({PLANNED_MD}) gives each one and "
               f"where this report shows it.")
    return L + [f"{checks} {how}", ""]


def figures_of(rf: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The figures of the report's figures record, in their order (by their
    number, then as listed); an entry that is not a figure is left out."""
    figs = [f for f in (rf or {}).get("figures") or [] if isinstance(f, dict)]
    return [f for _, f in sorted(enumerate(figs), key=lambda kf: (
        (ints([kf[1].get("number")]) or [10 ** 6])[0], kf[0]))]


def tables_of(rf: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [t for t in (rf or {}).get("tables") or [] if isinstance(t, dict)]


def _rows(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in t.get("rows") or [] if isinstance(r, dict)]


def _records(row: Dict[str, Any]) -> List[str]:
    recs = row.get("records")
    return [str(r) for r in recs] if isinstance(recs, list) else []


def table_records(rf: Dict[str, Any]) -> Set[str]:
    """Every record a table of the report prints."""
    return {r for t in tables_of(rf) for row in _rows(t) for r in _records(row)}


def _fields(rec: Dict[str, Any]) -> Set[str]:
    """The quantities a record reads: its own, and those of the records it
    was made from (an overlap of two maps reads both)."""
    out = {str(rec["field"])} if rec.get("field") else set()
    return out | {str(x.get("field")) for x in _dicts(rec.get("inputs")) if x.get("field")}


def _drawn(f: Dict[str, Any]) -> Set[str]:
    """The quantities a figure's panels draw, a cell-by-cell panel's second
    quantity ("against") with them."""
    return {str(p.get(k)) for p in _dicts(f.get("panels")) for k in ("field", "against")
            if p.get(k)}


def _fit(f: Dict[str, Any], fields: Set[str]) -> int:
    """How well a figure shows a finding's quantities: one for each it draws,
    and one more when a panel sets two of them against each other, cell by
    cell, as a finding on two maps does."""
    pair = any(p.get("field") and p.get("against") and {str(p["field"]), str(p["against"])}
               <= fields for p in _dicts(f.get("panels")))
    return len(_drawn(f) & fields) + (1 if pair else 0)


def choose_findings(report: Dict[str, Any], rf: Dict[str, Any], numbers: Dict[str, int]
                    ) -> Tuple[Dict[str, List[int]], List[Tuple[Optional[int], int]]]:
    """({figure id: the kept findings shown under it, by number from 1},
    [(comparison, finding) for "Other comparisons"]).

    A kept finding belongs to a figure that answers its comparison: the
    one that shows most of the quantities its records read (a panel that
    sets two of them against each other, cell by cell, counts once more),
    the first on a tie, so the first when none draws one. A finding
    whose records all print in a table of the report says nothing the table
    does not, and is left out (a number appears once, in the table). At
    most two under a figure, in the plan's order. A comparison that no
    figure answers shows its first kept finding. A kept finding with no
    comparison on record goes to "Other comparisons" too, so that no kept
    finding is placed nowhere."""
    claims = report.get("claims") or []
    records = records_of(report)
    figs = figures_of(rf)
    printed = table_records(rf)
    answering: Dict[int, List[Dict[str, Any]]] = {}
    for f in figs:
        for n in ints(f.get("comparisons")):
            answering.setdefault(n, []).append(f)
    comp = {i: comparison_of(report, c, numbers) for i, c in enumerate(claims, 1)
            if isinstance(c, dict)}
    order = sorted(comp, key=lambda i: (comp[i] if comp[i] is not None else 10 ** 6, i))
    by_fig: Dict[str, List[int]] = {str(f.get("id")): [] for f in figs}
    others: List[Tuple[Optional[int], int]] = []
    # A FINDING THE SUMMARY ALREADY SAYS is not said again below it
    # (2026-11-14): R2's Summary holds one of its findings nearly word for
    # word, and the reader met it twice, the second time under a figure. When
    # every content word of a finding, its codes in brackets aside, is a word
    # of the Summary, the finding stays on the details page.
    summary = _tokens(_codes_out(summary_text(report)))
    for i in order:
        if _within(_tokens(_codes_out(claims[i - 1].get("claim"))), summary):
            continue
        n = comp[i]
        if n is None:
            others.append((None, i))
            continue
        if n not in answering:
            if n not in {m for m, _ in others}:
                others.append((n, i))
            continue
        ev = [str(r) for r in claims[i - 1].get("evidence") or []]
        if ev and set(ev) <= printed:
            continue
        fields = {x for r in ev for x in _fields(records.get(r) or {})}
        # the figure that shows most of its quantities; the first on a tie
        home = max(answering[n], key=lambda f: _fit(f, fields))
        fid = str(home.get("id"))
        if len(by_fig[fid]) < MAX_PER_FIGURE:
            by_fig[fid].append(i)
    return by_fig, others


def answers_line(f: Dict[str, Any]) -> Optional[str]:
    """The line under a figure's caption that names the comparisons it
    answers, or None when it answers none."""
    comps = ints(f.get("comparisons"))
    return f"This figure answers {numbers_words(comps)}." if comps else None


def _table_lines(t: Dict[str, Any]) -> List[str]:
    cols = [str(c) for c in t.get("columns") or []]
    rows = _rows(t)
    L = [f"**Table {t.get('number')}.** {_p(t.get('caption'))}", ""]
    if not cols or not rows:
        return L + ["The table holds no row.", ""]
    panel = any(r.get("panel") for r in rows)
    head = [str(t.get("corner") or "")] + cols + (["Panel"] if panel else [])
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(c) for c in (r.get("cells") if isinstance(r.get("cells"), list) else [])]
        cells += [""] * len(cols)
        L.append("| " + " | ".join([_p(r.get("label"))] + cells[:len(cols)]
                                   + ([str(r.get("panel") or "")] if panel else [])) + " |")
    return L + [""]


def results_lines(report: Dict[str, Any], shown: Set[int]) -> List[str]:
    """Section 3: a subsection per figure of the report's figures record, in
    its order, each with its caption, the line on the comparisons it
    answers, the table that follows it and at most two findings; then the
    comparisons no figure answers; the key findings when no figure was
    drawn. `shown` gains each finding shown."""
    claims = report.get("claims") or []
    plan = report.get("_plan") or {}
    numbers = numbers_of(plan, report)
    rf = report.get("report_figures") if isinstance(report.get("report_figures"), dict) else {}
    figs = figures_of(rf)
    ids = identifiers(report)
    L = [f"## {SECTIONS[2]}", ""]
    if not figs:
        from agents.analysis import report_views as RV
        keys = RV.key_findings(report)
        if rf:
            L += ["No figure of the report was drawn for this study."] \
                + [_sentence(x) for x in rf.get("not_drawn") or []] + [""]
        if keys:
            L += ["The main findings, in the order the Analyzer ranked them. The details page "
                  "gives every finding with its records.", ""]
            L += [f"- {without_ids(claims[i - 1].get('claim'), ids)}" for i in keys] + [""]
            shown |= set(keys)
        else:
            L += ["The audit kept no finding.", ""]
        return L
    by_fig, others = choose_findings(report, rf, numbers)
    tables = tables_of(rf)
    k = 0
    for f in figs:
        k += 1
        n = (ints([f.get("number")]) or [k])[0]
        L += [f"### 3.{k} {_p(f.get('heading') or 'Figure ' + str(n))}", "",
              f"![Figure {n}]({f.get('file')})", "",
              f"**Figure {n}.** {_p(f.get('caption'))}", ""]
        said = answers_line(f)
        if said:
            L += [said, ""]
        for t in tables:
            if str(t.get("follows")) == str(f.get("id")):
                L += _table_lines(t)
        for i in by_fig.get(str(f.get("id"))) or []:
            L += [without_ids(claims[i - 1].get("claim"), ids), ""]
            shown.add(i)
    placed = {str(f.get("id")) for f in figs}
    for t in tables:
        if str(t.get("follows")) not in placed:
            L += _table_lines(t)
    if others:
        k += 1
        L += [f"### 3.{k} Other comparisons", ""]
        for n, i in others:
            L.append(f"- {without_ids(claims[i - 1].get('claim'), ids)}")
            shown.add(i)
        L.append("")
    # WHAT WAS NOT DRAWN, once: a sentence a caption already says is not
    # said again under the figures (the details page lists every one)
    said = [_tokens(x.get("caption")) for x in figs + tables]
    nd = [_sentence(x) for x in rf.get("not_drawn") or []
          if not any(_within(_tokens(x), c) for c in said)]
    if nd:
        L += [" ".join(nd), ""]
    return L


# words that carry no content of their own, for comparing a sentence with a
# caption ("has no panel over time" and "have no panel" say the same)
GLUE = {"the", "a", "an", "of", "in", "it", "its", "them", "they", "has", "have", "is", "are",
        "over", "time", "and", "to", "by", "with", "so"}


def _tokens(text: Any) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", _p(text).lower()) if w not in GLUE]


def _codes_out(text: Any) -> str:
    """A sentence without the model's codes in brackets ("(QDRAI)"), which a
    finding gives and the Summary may leave out."""
    return re.sub(r"\s*\([A-Z][A-Z0-9_]*\)", "", _p(text))


def _within(words: List[str], caption: List[str]) -> bool:
    """Whether a caption already says a sentence: every word of the
    sentence that carries content is a word of the caption."""
    return bool(words) and set(words) <= set(caption)


# ── Limitations ─────────────────────────────────────────────────────────────
def _norm(text: Any) -> str:
    t = " ".join(_p(text).lower().split()).rstrip(".")
    t = t.replace("this is the planner's judgement", "")
    return t.replace("the control run", "the control").strip()


def _not_meaning_words(statement: Any) -> str:
    """What a run's difference must not be called, in its first clause: the
    statement whole is on the details page."""
    m = re.match(r"run (\S+) \(([^)]*)\): its difference from the control is NOT ([^:;]+)",
                 _p(statement))
    if not m:
        return _sentence(statement)
    return (f"The difference of the new run `{m.group(1)}` from the control run is not "
            f"{m.group(3).strip().rstrip('.')}.")


def design_limits(sd: Path) -> Optional[List[str]]:
    """The design page's "What the answer will not cover", sentence for
    sentence, made again from the study's own records by the design page's
    code; None when the records do not hold a whole-domain design."""
    try:
        from agents import design_page as DP
        reception = _read_json(sd / "reception.json")
        strategy = _read_json(sd / "strategy.json")
        if not isinstance(strategy.get("domain_experiment"), dict):
            return None
        brief = reception.get("brief") or reception
        de = dict(brief.get("domain_experiment") or {})
        de.update(strategy.get("domain_experiment") or {})
        rb = strategy.get("domain_rulebook") or {}
        menu = {c.get("id"): c for c in ((rb.get("changes") or {}).get("changes") or [])
                if isinstance(c, dict)}
        name = (de.get("control") or {}).get("name")
        row = next((c for c in ((rb.get("controls") or {}).get("controls") or [])
                    if isinstance(c, dict) and c.get("name") == name), {})
        question = reception.get("user_request") or brief.get("user_request") or ""
        lines = DP._not_cover_lines(strategy, de, row, sd,
                                    DP.differences(question, DP._runs(de), menu))
    except Exception:                                               # noqa: BLE001
        return None
    return [ln[2:].strip() for ln in lines if ln.startswith("- ")]


def later_start(sd: Path) -> List[Tuple[str, str]]:
    """(the note on a start from a later saved state as one plain sentence,
    the statement it is on record as), for each such note of the server's
    check. The date is the record's own; the note whole is a caveat on the
    details page."""
    exp = _read_json(sd / "experiment.json")
    out = []
    for u in ((exp.get("experiment") or {}).get("check") or {}).get("unusual") or []:
        if isinstance(u, dict) and re.search(r"later saved state", str(u.get("what") or "")):
            m = re.search(r"saved state of (\d{1,2} [A-Z][a-z]+ \d{4})", str(u.get("what")))
            text = LATER_START.format(date=m.group(1)) if m else LATER_START_NO_DATE
            out.append((text, f"{u.get('where')}: {u.get('what')}"))
    return out


def changed_share(sd: Path, regions: Set[str]) -> List[Tuple[str, str]]:
    """(a sentence on the share of a region a regional change covers, the
    caveat's name), for each region a printed record uses that the change
    reaches only in part. The names follow step0_context's count."""
    t = (_read_json(sd / "experiment.json").get("regional") or {})
    several = len(t.get("changed_share") or {}) > 1
    out, k = [], 0
    for run, cs in sorted((t.get("changed_share") or {}).items()):
        where = (cs or {}).get("region")
        for reg, s in sorted(((cs or {}).get("share") or {}).items()):
            if s is None or s >= 0.999:
                continue
            k += 1
            if reg not in regions:
                continue
            who = f"The change of `{run}`" if several else "The new run's change"
            text = (f"{who} applies only in {region_words(where)} and does not reach "
                    f"{region_words(reg)}, so a difference over {region_words(reg)} is no "
                    f"response to the change." if s <= 0.001 else
                    f"{who} applies only in {region_words(where)}, which covers "
                    f"{100 * s:.0f} percent of {region_words(reg)}. A difference over "
                    f"{region_words(reg)} therefore averages changed and unchanged cells.")
            out.append((text, f"changed_share_{k}"))
    return out


def no_mid_colon(text: Any) -> str:
    """A limit with no colon in mid-sentence: the words after a colon start
    a sentence of their own. The words themselves are kept."""
    parts = re.split(r":\s+", _p(text).strip())
    return " ".join(_sentence(x) for x in parts if x.strip())


def limits_of(sd: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    """The limits of section 4, worked out once for both pages:
    {"shown": [(limit, caveats it says)] at most six, "cut": the limits the
    cap leaves out, "more": the caveats on record that no limit says}.

    First the design page's limits for this study in its own sentences, then
    what only the analysis knows, then the note on a later start. Each limit
    once, with no identifier and no colon in mid-sentence."""
    caveats = [c for c in report.get("caveats") or [] if isinstance(c, dict)]
    records = records_of(report)
    used = [records[r] for r in used_records(report) if isinstance(records.get(r), dict)]
    regions = {str(r.get("region")) for r in used if r.get("region")}
    fields = {str(r.get("field")) for r in used if r.get("field")}
    design = design_limits(sd)
    limits: List[Tuple[str, Set[str]]] = []     # (the limit, the caveats it says)

    def add(text: str, ids: Sequence[str] = ()) -> None:
        text = no_mid_colon(text)
        for t, have in limits:
            if _norm(t) == _norm(text):
                have.update(ids)
                return
        if text:
            limits.append((text, set(ids)))
    if design is not None:
        for s in design:
            add(s)
    else:
        # A STUDY WITHOUT A DESIGN PAGE'S RECORDS: the Planner's caveats and
        # what each run's difference must not be called, as recorded
        for c in caveats:
            if str(c.get("id")).startswith(("plan_caveat_", "not_meaning_")):
                add(_not_meaning_words(c.get("statement")), [str(c.get("id"))])
    for text, cid in changed_share(sd, regions):
        add(text, [cid])
    # A BINDING CAVEAT the design page does not hold: one about every claim,
    # or about a quantity a printed record reads
    for c in caveats:
        cid = str(c.get("id"))
        if c.get("severity") != "blocking" or cid.startswith(("not_meaning_", "changed_share_")):
            continue
        codes = set(re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", str(c.get("applies_to") or "")))
        if not codes or codes & fields:
            add(_sentence(c.get("statement")), [cid])
    for text, statement in later_start(sd):
        add(text, [str(c.get("id")) for c in caveats
                   if _norm(c.get("statement")) == _norm(statement)])
    shown, cut = limits[:MAX_LIMITS], limits[MAX_LIMITS:]
    told = [_norm(t) for t, _ in limits]
    named = {i for _, ids in limits for i in ids}
    # a caveat is said by a limit that names it or holds its words whole; one
    # said only by a limit the cap leaves out is counted with that limit
    more = [c for c in caveats if str(c.get("id")) not in named
            and not (_norm(c.get("statement")) and any(_norm(c.get("statement")) in t
                                                        for t in told))]
    return {"shown": shown, "cut": cut, "more": more}


def _many(n: int, one: str, more: str) -> str:
    return f"{n} {one if n == 1 else more}"


def limitation_lines(sd: Path, report: Dict[str, Any],
                     lim: Optional[Dict[str, Any]] = None) -> List[str]:
    """Section 4: at most six limits, then how many more limits and caveats
    the details page gives whole."""
    lim = lim if lim is not None else limits_of(sd, report)
    L = [f"## {SECTIONS[3]}", ""]
    L += [f"- {t}" for t, _ in lim["shown"]] or ["No limit is on record for this study."]
    L.append("")
    parts = ([_many(len(lim["cut"]), "more limit", "more limits")] if lim["cut"] else []) + \
        ([_many(len(lim["more"]), "more caveat on record", "more caveats on record")]
         if lim["more"] else [])
    L.append(f"The details page gives {_join(parts)} in full." if parts else
             "The details page gives each caveat on record in full, with its name.")
    return L + [""]


# ── Next steps ──────────────────────────────────────────────────────────────
def change_words(ch: Dict[str, Any]) -> str:
    """A change of a proposed run in words, never its settings as the server
    takes them."""
    cid, p = ch.get("id"), ch.get("params") or {}
    if cid == "weather_scale":
        parts = []
        t = float(p.get("temperature_K") or 0.0)
        f = p.get("precip_factor")
        if t:
            parts.append(f"the air {abs(t):g} K {'cooler' if t < 0 else 'warmer'}")
        if f not in (None, 1, 1.0):
            parts.append(f"rain and snow times {float(f):g}")
        from agents.analysis.report_views import _months_text
        when = (f" in {_months_text(p.get('months'))}" if p.get("months") else "") + \
            (f" of {span_words(p.get('years'))}" if p.get("years") else "")
        where = f" over {region_words(p['region'])}" if p.get("region") else ""
        return (_join(parts) or "the weather unchanged") + when + where
    if cid == "fire_off":
        return "the model's own fire switched off"
    meaning = re.sub(r"^the difference from the control is\s*", "", _p(ch.get("meaning")),
                     flags=re.I).rstrip(".")
    return meaning or f"the change `{cid}`"


QUESTION_WORDS = ("how", "whether", "what", "which", "why", "when", "where", "if", "who")


def open_words(cause: str) -> str:
    """Step 3's open cause as a sentence: "This study leaves open how ..."
    for a cause that is a question, else the cause as it is."""
    c = _p(cause).strip().rstrip(".?")
    if c.split(" ")[0].lower() in QUESTION_WORDS:
        return f"This study leaves open {c[:1].lower() + c[1:]}."
    return f"Left open by this study. {_sentence(c)}"


def next_step_lines(report: Dict[str, Any]) -> List[str]:
    """Section 5: the question the study leaves open, then the run that would
    answer it, in words; at most three lines."""
    from agents.analysis import report_views as RV
    t = report.get("evidence_trace") or {}
    causes = t.get("open_causes") or report.get("open_causes") or []
    L: List[str] = []
    general = False
    for oc in causes:
        cause = _p(oc.get("cause"))
        # A CAUSE MAY LEAN ON A WITHHELD RESULT (report_views.uncertainty_lines):
        # in step 3's words only when the audit withheld nothing
        if not report.get("withheld") and cause and not re.search(r"\d", cause):
            L.append(f"- {open_words(cause)}")
        elif not general:
            general = True
            L.append("- This study leaves open whether one of these results caused another, "
                     "which the results cannot show. The details page gives the Analyzer's "
                     "own words on it.")
        changes = (oc.get("changes") or (oc.get("proposed_run") or {}).get("changes") or [])
        if changes:
            L.append(f"- A run that would answer it has "
                     f"{_join([change_words(c) for c in changes])}, with every other input as "
                     f"in the control run. It has not been made.")
    if not causes:
        L.append("- No open question was recorded." + (
            "" if any(c.get("evidence_class") == RV.CAUSAL_CLASS
                      for c in report.get("claims") or []) else " " + NO_CAUSE))
    return [f"## {SECTIONS[4]}", ""] + L[:MAX_NEXT] + [""]


# ── Checks and records ──────────────────────────────────────────────────────
def check_sentence(report: Dict[str, Any]) -> str:
    """The second reading of the numbers, in the check's own sentence
    (numbers_check.evidence_sentence), which says what it made again, of how
    many records the findings cite and how many more the report prints, and
    what it did not make again."""
    from agents.analysis import numbers_check as NC
    en = report.get("evidence_numbers")
    rf = report.get("report_figures") if isinstance(report.get("report_figures"), dict) else {}
    return _p(NC.evidence_sentence(en if isinstance(en, dict) else None,
                                   figures=bool(figures_of(rf))))


def _count(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def checks_lines(report: Dict[str, Any], screen: bool, sd: Optional[Path] = None) -> List[str]:
    kept, left = len(report.get("claims") or []), len(report.get("withheld") or [])
    links = [f"[{DETAILS_MD}]({DETAILS_MD}) gives every finding with its records, every "
             f"caveat in full and what code checked."]
    if screen:
        links.append(f"[{SCREEN_MD}]({SCREEN_MD}) gives the first look at every quantity.")
    links.append(f"[{PLANNED_MD}]({PLANNED_MD}) gives each approved comparison.")
    links.append(f"[{DESIGN_MD}](../{DESIGN_MD}) is the design that was approved."
                 if sd is None or (Path(sd) / DESIGN_MD).is_file() else NO_DESIGN)
    return [f"## {SECTIONS[5]}", "", check_sentence(report), "",
            f"The audit kept {kept or 'no'} finding{'s' if kept != 1 else ''} and left out "
            f"{left or 'none'}. "
            f"The details page gives each with its reason.", "",
            JUDGEMENT_WORDS, ""] + [f"- {x}" for x in links] + [""]


def _problem(report: Dict[str, Any], where: str, e: Exception) -> None:
    """A part of a page that could not be written, kept for the details page."""
    probs = report.get("_problems")
    if isinstance(probs, list):
        probs.append(f"{where}: {type(e).__name__}: {e}")


def _section(report: Dict[str, Any], k: int, fn, *args) -> List[str]:
    """A section of REPORT.md, or its heading and one line saying it could
    not be written from the record: an odd value never stops the report.
    The reason goes to the details page."""
    try:
        return fn(*args)
    except Exception as e:                                          # noqa: BLE001
        _problem(report, f"{REPORT_MD}, section {SECTIONS[k]}", e)
        return [f"## {SECTIONS[k]}", "", "This section could not be written from the record. "
                "The details page and analysis.json hold the record.", ""]


def render_report(report: Dict[str, Any], sd: Path, screen: bool) -> Tuple[str, Set[int]]:
    """REPORT.md, and the kept findings it shows (by number from 1)."""
    shown: Set[int] = set()
    try:
        L = title_lines(report)
    except Exception:                                               # noqa: BLE001
        L = [f"# {_p(report.get('question') or 'The report')}", ""]
    L += _section(report, 0, summary_lines, report)
    L += _section(report, 1, methods_lines, sd, report)
    L += _section(report, 2, results_lines, report, shown)
    L += _section(report, 3, limitation_lines, sd, report, report.get("_limits"))
    L += _section(report, 4, next_step_lines, report)
    L += _section(report, 5, checks_lines, report, screen, sd)
    return "\n".join(L).rstrip() + "\n", shown


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN.md
# ─────────────────────────────────────────────────────────────────────────────
def screen_records(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every screen the log holds, in its order."""
    return [r for r in records_of(report).values()
            if isinstance(r, dict) and r.get("call") == "screen"]


def screen_record(report: Dict[str, Any]) -> Dict[str, Any]:
    sr = (report.get("evidence_trace") or {}).get("screen_record")
    return sr if isinstance(sr, dict) else {}


def screen_run_of(sr: Dict[str, Any]) -> str:
    """The run a screen's record looked at (interface I3): its scope's run;
    a record from before the screen named its run in scope gives it in its
    keys, or not at all, and is then read as the control run's."""
    return str((sr.get("scope") or {}).get("run") or (sr.get("keys") or {}).get("run")
               or "control")


def shown_screens(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The screens SCREEN.md shows: the one the screen's record describes
    (the same hash), else those of the run it names. A screen of another
    run that a later screen replaced is listed on the details page only."""
    recs = screen_records(report)
    sr = screen_record(report)
    if not sr:
        return recs
    same = [r for r in recs if sr.get("sha256") and r.get("sha256") == sr.get("sha256")]
    if same:
        return same
    run = screen_run_of(sr)
    return [r for r in recs if str(r.get("run") or (r.get("keys") or {}).get("run")
                                    or "control") == run] or recs


def screen_opening(report: Dict[str, Any], r: Dict[str, Any]) -> str:
    """What the screen compares, from the screen's record (interface I3)."""
    sr = screen_record(report)
    scope = sr.get("scope") or {}
    run = screen_run_of(sr)
    year = scope.get("year") or r.get("period")
    where = region_words(scope.get("region") or r.get("region"))
    if run != "control":
        return (f"The new run `{run}` against the control run over {where}, every quantity "
                f"and period of {year}.")
    normal = span_words(scope.get("normal_years"))
    return (f"{year} against the normal years{(' ' + normal) if normal else ''}, over {where}, "
            f"every quantity and period of the control run.")


def mark_words(x: Dict[str, Any]) -> str:
    """A screen row's mark in words ("above the largest value of the normal
    years"); a mark the list does not hold, with its underscores as spaces."""
    m = x.get("mark")
    return "" if not m else MARK_WORDS.get(str(m), str(m).replace("_", " "))


def status_words(x: Dict[str, Any]) -> str:
    """A screen row's status in words: a marked row by its mark."""
    st = str(x.get("status") or "")
    if st == "marked":
        return mark_words(x) or "marked"
    return STATUS_WORDS.get(st, st.replace("_", " "))


def _drop_empty(head: List[str], rows: List[List[str]]) -> List[str]:
    keep = [k for k in range(len(head)) if k == 0 or any(r[k] for r in rows)]
    out = ["| " + " | ".join(head[k] for k in keep) + " |", "|" + "---|" * len(keep)]
    out += ["| " + " | ".join(r[k] for k in keep) + " |" for r in rows]
    return out + [""]


def render_screen(report: Dict[str, Any]) -> Optional[str]:
    """SCREEN.md, or None when no screen ran."""
    recs = shown_screens(report)
    if not recs:
        return None
    L = ["# The screen", "", f"*{_p(report.get('question') or '')}* Back to "
         f"[the report]({REPORT_MD}).", ""]
    for r in recs:
        L += [screen_opening(report, r), "", f"{SCORE_WORDS} {LOOKING_WORDS}", ""]
        if r.get("planned") is False and r.get("after_close"):
            L += ["This screen was made after the planned comparisons closed, so it is not one "
                  "of the planned lookups.", ""]
        scored, other = [], []
        rows = r.get("rows") if isinstance(r.get("rows"), dict) else {}
        for key, x in rows.items():
            if not str(key).startswith("display.") or not isinstance(x, dict):
                continue
            cells = [field_name(x.get("field")), period_words(x.get("period")),
                     "" if x.get("change") is None else _fv(x["change"]),
                     units_name(x.get("change_units") or x.get("units") or ""),
                     "" if x.get("percent") is None else _fv(x["percent"])]
            if x.get("score") is not None:
                scored.append([_p(x.get("group") or "")] + cells
                              + [_fv(x["score"]), "" if x.get("n_normal") is None
                                 else str(x["n_normal"]), _p(mark_words(x))])
            else:
                other.append(cells + [_p(status_words(x))])
        L += ["## The rows with a score", ""]
        L += (_drop_empty(["Group", "Quantity", "Period", "Change", "Units", "Percent", "Score",
                           "Normal years", "Mark"], scored) if scored else
              ["No row has a score.", ""])
        L += ["## The rows not scored yet", ""]
        L += (_drop_empty(["Quantity", "Period", "Change", "Units", "Percent", "Status"], other)
              if other else ["Every row shown has a score.", ""])
        c = r.get("screen_csv") if isinstance(r.get("screen_csv"), dict) else {}
        L += ([f"[The whole table]({c['file']}) holds every quantity and period of the screen, "
               f"{_fv(c.get('rows'))} rows."] if c.get("file") else
              ["No file of the whole table was written."]) + [""]
    L += ["## How the rows are scored", ""] + [f"- {x}" for x in SCORING_SENTENCES]
    L += ["", "## Which rows are shown", ""] + [f"- {x}" for x in DISPLAY_SENTENCES]
    L += ["", "The details page gives the rules in the words of the model's code."]
    return "\n".join(L).rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# PLANNED_ANALYSES.md
# ─────────────────────────────────────────────────────────────────────────────
def _became(row: Dict[str, Any]) -> str:
    """What became of a comparison, in words, from its outputs."""
    outs = [o for o in row.get("outputs") or [] if isinstance(o, dict)]
    n = len(outs)
    got = sum(o.get("outcome") == "produced" for o in outs)
    ne = sum(o.get("outcome") == "not estimable" for o in outs)
    s = f"{row.get('status') or 'no status'}, {got} of {n} numbers produced"
    if ne:
        s += f" and {ne} not estimable"
    if n - got - ne:
        s += f", {n - got - ne} not produced"
    return s


def shown_in(n: int, a: Dict[str, Any], row: Dict[str, Any], report: Dict[str, Any],
             shown: Set[int], numbers: Dict[str, int]) -> str:
    """Where the report shows a comparison: its figure panels and its table,
    else a finding in the Results, else the details page."""
    rf = report.get("report_figures") if isinstance(report.get("report_figures"), dict) else {}
    fields = {str(f) for f in a.get("fields") or []}
    recs = {str(o.get("record")) for o in row.get("outputs") or []
            if isinstance(o, dict) and o.get("record")}
    parts = []
    for f in figures_of(rf):
        if n in ints(f.get("comparisons")):
            num = f.get("number")
            panels = [p for p in f.get("panels") or [] if isinstance(p, dict)]
            letters = [f"{num}{p.get('letter')}" for p in panels
                       if str(p.get("field")) in fields and p.get("letter")]
            # every panel of the figure is the whole figure
            whole = not letters or len(letters) == len(panels)
            parts.append(f"Figure {num}" if whole else f"Figure {', '.join(letters)}")
    for t in tables_of(rf):
        if any(recs & set(_records(r)) for r in _rows(t)):
            parts.append(f"Table {t.get('number')}")
    if parts:
        return _join(parts)
    claims = report.get("claims") or []
    if any(comparison_of(report, claims[i - 1], numbers) == n for i in shown):
        return "a finding in the Results"
    return "the details page"


def render_planned(report: Dict[str, Any], shown: Set[int]) -> str:
    plan = report.get("_plan") or {}
    ph = report.get("planned_phase") or {}
    numbers = numbers_of(plan, report)
    L = ["# The planned comparisons", "", f"*{_p(report.get('question') or '')}* Back to "
         f"[the report]({REPORT_MD}).", ""]
    from agents import analysis_handoff as _handoff
    said = [_p(x) for x in _handoff.report_lines((report.get("analysis_plan") or {})
                                                 .get("handoff"))]
    if said:
        L += [" ".join(said), ""]
    elif (report.get("analysis_plan") or {}).get("passed"):
        L += ["The Analyzer wrote this plan before looking up any result, and the plan passed "
              "its review. No design page showed it before the yes.", ""]
    rows = {str(r.get("id")): r for r in ph.get("analyses") or [] if isinstance(r, dict)}
    items = analyses(plan, report)
    if not items:
        return "\n".join(L + ["No planned comparison is on record for this study.", ""])
    L += ["| No. | Comparison | What it tells | What became of it | Where the report shows it |",
          "|---|---|---|---|---|"]
    for n, a in enumerate(items, 1):
        row = rows.get(str(a.get("id"))) or {}
        cells = [str(n), _p(a.get("title") or a.get("question") or "untitled"),
                 _p(a.get("tells") or ""), _became(row) if row else "not on record",
                 shown_in(n, a, row, report, shown, numbers)]
        L.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    L.append("")
    competing = [c for c in plan.get("competing") or [] if isinstance(c, dict)]
    if competing:
        L += ["## Other explanations the plan checks", ""]
        for c in competing:
            n = numbers.get(str(c.get("analysis")))
            L.append(f"- **{_p(c.get('title'))}.** {_sentence(c.get('text'))}"
                     + (f" Comparison {n} reads it." if n else ""))
        L.append("")
    exps = [e for e in ph.get("expectations") or plan.get("expectations") or []
            if isinstance(e, dict)]
    if exps:
        claims = report.get("claims") or []
        un = {u.get("expectation"): u for u in (report.get("acceptance") or {})
              .get("untestable") or []}
        L += ["## The expectations", "", SUPPORT_WORDS, "",
              "| Expectation | Comparison | What became of the comparison | Numbers not "
              "estimable | Findings that test it |", "|---|---|---|---|---|"]
        for e in exps:
            n = numbers.get(str(e.get("analysis")))
            row = rows.get(str(e.get("analysis"))) or {}
            tested = sum(c.get("evidence_class") == "tested expectation"
                         and c.get("expectation") == e.get("id") for c in claims)
            ne = len((un.get(e.get("id")) or {}).get("outputs") or [])
            L.append(f"| {_p(e.get('text') or 'no words on record').replace('|', '/')} | "
                     f"{n or 'none'} | {row.get('status') or 'not on record'} | {ne} | "
                     f"{tested} |")
        L.append("")
    L += ["## How the planned phase closed", ""]
    c = ph.get("counts") or {}
    if ph:
        n = sum(int(v or 0) for v in c.values())
        parts = [f"{c.get(k)} {w}" for k, w in (
            ("completed", "were completed"),
            ("completed after an earlier failure", "were completed after an earlier failure"),
            ("failed", "failed"), ("skipped", "were skipped")) if c.get(k)]
        said = (f"all {n} were completed" if c.get("completed") == n and n else
                _join(parts) or "none is on record")
        L.append(f"The planned phase closed {_p(ph.get('closed') or 'at a time not on record')}. "
                 f"Of its {n} comparison{'s' if n != 1 else ''}, {said}.")
    else:
        L.append("The close of the planned phase is not on record.")
    held = ((report.get("analysis_plan") or {}).get("baseline") or {}).get("inclusive")
    if held:
        L += ["", f"A note on the normal years. {_sentence(held)}"]
    acc = report.get("acceptance") or {}
    if acc.get("blockers"):
        L += ["", "The hypothesis was not fully tested, because a comparison its expectations "
                  "need was not completed. This does not count as a successful test."]
    if acc.get("inconclusive"):
        L += ["", "The report is inconclusive, because no finding could head it. This does not "
                  "count as a successful test."]
    if acc.get("not_estimable"):
        k = len(acc["not_estimable"])
        L += ["", f"{k} required number{'s' if k != 1 else ''} could not be estimated. "
                  f"{'Their comparisons are' if k != 1 else 'Its comparison is'} complete, and "
                  f"{'their expectations have' if k != 1 else 'its expectation has'} no estimate."]
    return "\n".join(L).rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# REPORT_DETAILS.md
# ─────────────────────────────────────────────────────────────────────────────
def render_details(report: Dict[str, Any]) -> str:
    """Everything the short page leaves out, under plain headings, from the
    same functions the one page of other reports uses. Each part is written
    on its own, so a part the record cannot give says so and the rest
    stands."""
    L = ["# Details of the report", "", f"*{_p(report.get('question') or '')}*", "",
         f"Everything [the report]({REPORT_MD}) leaves out, from the same record, analysis.json. "
         f"Values are shown to 3 significant figures. The exact values are in analysis.json.",
         "", f"*{report.get('model') or 'model'} run, status **{report.get('status')}**, "
         f"generated {report.get('generated_at')}, "
         f"{Path(str(report.get('run_dir') or '')).name}*", ""]
    for part in (_d_headline, _d_summary, _d_findings, _d_caveats, _d_limits_left,
                 _d_wording, _d_checks, _d_setup, _d_uncertainty, planned_record_lines,
                 _d_drawn, report_figure_lines, _d_screens, _d_provenance):
        try:
            L += part(report)
        except Exception as e:                                      # noqa: BLE001
            _problem(report, f"{DETAILS_MD}, {part.__name__.strip('_')}", e)
            L += [f"A part of this page ({part.__name__.strip('_').replace('_', ' ')}) could "
                  f"not be written from the record ({type(e).__name__}: {e}).", ""]
    probs = report.get("_problems") or []
    if probs:
        L += ["## Parts of the pages that could not be written", ""] + [f"- {_p(x)}" for x in probs]
    return "\n".join(L).rstrip() + "\n"


def _d_headline(report: Dict[str, Any]) -> List[str]:
    from agents.analysis.claim_trace import headline_caveat_line
    L = ["## The headline", ""]
    if report.get("headline"):
        cl = headline_caveat_line(report)
        L += ([f"**{_p(report['headline'])}**  ", f"*{_p(cl)}*"] if cl else
              [f"**{_p(report['headline'])}**"]) + [""]
    else:
        L += ["No headline is on record.", ""]
    rd = report.get("readability") or {}
    if rd.get("problems_after"):
        L += ["> Wording the readability check still flags: "
              + "; ".join(str(x) for x in rd["problems_after"][:6]), ""]
    return L


def _d_summary(report: Dict[str, Any]) -> List[str]:
    L: List[str] = []
    shown = report.get("answer_kept") if "answer_kept" in report else report.get("answer")
    if shown and without_ids(shown, identifiers(report)) != _p(shown):
        L += ["## The summary as the Analyzer wrote it", "",
              "The report takes the record's own names out of the summary. Here it is with "
              "them.", "", _p(shown), ""]
    if report.get("answer_left_out"):
        L += ["## The sentences of the summary left out of the report", ""]
        for x in report["answer_left_out"]:
            L.append(f"- \"{_p(x.get('sentence'))}\" It gives "
                     f"{_join([str(u) for u in x.get('unsupported') or []])}, which no kept "
                     f"finding's record gives.")
        L.append("")
    return L


def _d_findings(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import step4_report as S4
    claims = report.get("claims") or []
    L = [f"## Every kept finding, with its records ({len(claims)})", ""]
    L += S4.claim_lines(report) or ["None.", ""]
    held = report.get("withheld") or []
    L += [f"## The findings left out, and why ({len(held)})", ""]
    L += (S4.withheld_lines(report) + [""]) if held else ["None.", ""]
    return L


def _d_caveats(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import step4_report as S4
    blocking, others = S4.caveat_lines(report)
    L = ["## Every caveat in full", ""]
    if blocking:
        L += ["Caveats that bind the answer:", ""] + blocking + [""]
    if others:
        L += ["Other caveats on record:", ""] + others + [""]
    if not blocking and not others:
        L += ["None.", ""]
    return L


def _d_limits_left(report: Dict[str, Any]) -> List[str]:
    """The limits the six-bullet cap of the report's Limitations leaves out."""
    cut = (report.get("_limits") or {}).get("cut") or []
    if not cut:
        return []
    return (["## Limits the report leaves out", "",
             f"The report shows {MAX_LIMITS} limits. These are the others, in the same order.",
             ""] + [f"- {t}" for t, _ in cut] + [""])


def _d_wording(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import domain_report as DR
    if not report.get("evidence_trace"):
        return []
    return ["## How the claims are worded", "", FIXED_RULES, ""] + DR.wording_lines(report) + [""]


def _d_checks(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import step4_report as S4
    from agents.analysis import numbers_check as NC
    L: List[str] = []
    lim = S4.verification_limits(report)
    if lim:
        L += [lim[0], ""] + lim[2:]
    L += ["## The two numbers checks", ""]
    if "numbers_check" in report:
        L += [NC.page_line(report.get("numbers_check"), S4.NUMBERS_LEAD), ""]
    if "evidence_numbers" in report:
        en = report.get("evidence_numbers")
        L += [NC.evidence_page_line(en if isinstance(en, dict) else None), ""]
        printed = (en or {}).get("printed") if isinstance(en, dict) else None
        if printed:
            # each printed record by whether a number of it was made again,
            # as the check's own sentence counts them (2026-11-14)
            recs = en.get("records") if isinstance(en.get("records"), dict) else {}
            names = sorted_records(printed)
            made = [r for r in names if NC.made_again(recs.get(r))]
            unmade = [r for r in names if r not in set(made)]
            if made:
                L += [f"The records the report's tables print whose numbers were made again: "
                      f"{', '.join(made)}.", ""]
            if unmade:
                L += [f"The records the report's tables print of which no number was made "
                      f"again: {', '.join(unmade)}.", ""]
    if "numbers_check" not in report and "evidence_numbers" not in report:
        L += ["No numbers check is on record.", ""]
    return L


def _d_setup(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import model_setup as MS
    ms = MS.render(report.get("model_setup")) if report.get("model_setup") else []
    return ["## The model set-up in full", ""] + ([_p(x) for x in ms[2:]] if ms else
                                                  ["Not on record.", ""])


def _d_uncertainty(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import domain_report as DR
    L: List[str] = []
    oc = DR.open_cause_lines(report)
    if oc:
        L += ["## The remaining uncertainty, with the runs as the server takes them", ""] \
            + oc + [""]
    timing = DR.timing_lines(report)
    if timing:
        L += ["## Timing", ""] + timing + [""]
    return L


def _d_drawn(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import domain_report as DR
    if not report.get("evidence_trace"):
        return []
    return ["## The figures the Analyzer drew in its own step", ""] \
        + DR.drawn_lines(report, names_only=True) + [""]


def _d_screens(report: Dict[str, Any]) -> List[str]:
    """Every screen on record, which one SCREEN.md shows, and the screen's
    rules as the model server states them, whole."""
    recs = screen_records(report)
    if not recs:
        return []
    shown = {str(r.get("id")) for r in shown_screens(report)}
    L = ["## The screens on record", ""]
    for r in recs:
        c = r.get("screen_csv") if isinstance(r.get("screen_csv"), dict) else {}
        made = ("planned" if r.get("planned") else "not planned") + \
            (", made after the planned phase closed" if r.get("after_close") else "")
        L.append(f"- record {r.get('id')}: run {r.get('run') or (r.get('keys') or {}).get('run')}, "
                 f"year {r.get('period')}, region {r.get('region')}, against "
                 f"{(r.get('comparison') or {}).get('against')}; {made}; its findings are "
                 f"\"{r.get('label') or 'screen'}\""
                 + (f"; its whole table {c.get('file')}, {_fv(c.get('rows'))} rows, sha256 "
                    f"{str(c.get('sha256'))[:12]}" if c.get("file") else "")
                 + ("; shown on SCREEN.md" if str(r.get("id")) in shown else
                    "; not shown on SCREEN.md, which shows the screen the screen's record "
                    "describes"))
    L.append("")
    sr = screen_record(report)
    disp = sr.get("display_rule") or ((recs[-1].get("words") or {}).get("display_rule")
                                      if isinstance(recs[-1].get("words"), dict) else None)
    L += ["## The screen's rules as the model server states them", "",
          f"Scoring rule: {_p(sr.get('scoring_rule') or 'not on record')}", "",
          f"Display rule: {_p(disp or 'not on record')}", ""]
    return L


def _d_provenance(report: Dict[str, Any]) -> List[str]:
    from agents.analysis import step4_report as S4
    return ["## Provenance", ""] + S4.provenance_lines(report) + [""]


def sorted_records(names: Any) -> List[str]:
    """Record names ("L7") each once, by their number; any other name after
    them, as written."""
    xs = [str(x) for x in names or []]
    def key(n: str):
        m = re.fullmatch(r"L(\d+)", n)
        return (0, int(m.group(1)), "") if m else (1, 0, n)
    return sorted(dict.fromkeys(xs), key=key)


def planned_record_lines(report: Dict[str, Any]) -> List[str]:
    """The close of the planned phase as recorded, with each comparison's
    recorded reason by its number and its plan code."""
    ph = report.get("planned_phase") or {}
    if not ph:
        return []
    numbers = numbers_of(report.get("_plan") or {}, report)
    L = ["## The planned phase as recorded", "",
         f"Closed {_p(ph.get('closed'))} by {_p(ph.get('by'))} ({ph.get('file')}, after "
         f"{ph.get('lookups_before')} lookup lines; the plan {ph.get('plan_file')}, sha256 "
         f"{str(ph.get('plan_sha256'))[:12]}). Each comparison is judged by its required "
         f"outputs, each produced, not estimable or unmet; its scope (what it may look up) "
         f"decides nothing.", ""]
    for r in ph.get("analyses") or []:
        L.append(f"- comparison {numbers.get(str(r.get('id')), '?')} ({r.get('id')}): "
                 f"{r.get('status')}; {_p(r.get('reason'))}")
    return L + [""]


def report_figure_lines(report: Dict[str, Any]) -> List[str]:
    """The report's own figures and tables as recorded: their files, the
    records the tables print, what was not drawn and any problem."""
    if not isinstance(report.get("report_figures"), dict):
        return ["## The report's figures", "",
                "No record of the report's figures is on this study.", ""]
    rf = report.get("report_figures") or {}
    L = ["## The report's figures", ""]
    for f in figures_of(rf):
        L.append(f"- Figure {f.get('number')} ({f.get('id')}): {f.get('file')}, comparisons "
                 f"{', '.join(str(x) for x in ints(f.get('comparisons'))) or 'none'}")
    for t in tables_of(rf):
        rows = _rows(t)
        L.append(f"- Table {t.get('number')}, after "
                 f"{'the figure ' + str(t.get('follows')) if t.get('follows') else 'no figure'}, "
                 f"{len(rows)} row{'s' if len(rows) != 1 else ''}:")
        for r in rows:
            L.append(f"  - {_p(r.get('label'))}: records "
                     f"{', '.join(sorted_records(_records(r))) or 'none'}")
    for x in rf.get("not_drawn") or []:
        L.append(f"- not drawn: {_p(x)}")
    for x in rf.get("problems") or []:
        L.append(f"- problem: {_p(x)}")
    if len(L) == 2:
        L.append("Nothing was drawn and nothing is on record of why.")
    return L + [""]


# ─────────────────────────────────────────────────────────────────────────────
# THE FOUR PAGES
# ─────────────────────────────────────────────────────────────────────────────
def applies(report: Dict[str, Any]) -> bool:
    """Whether a report is written as four pages: a whole-domain report that
    is not withheld. Every other report keeps its one page."""
    return report.get("kind") == "whole_domain" and not report.get("withheld_because")


def _page(report: Dict[str, Any], name: str, title: str, fn, *args):
    """A page, or its title and one line saying it could not be written from
    the record (the reason goes to the details page)."""
    try:
        return fn(*args)
    except Exception as e:                                          # noqa: BLE001
        _problem(report, name, e)
        return (f"# {title}\n\nThis page could not be written from the record. "
                f"[The report]({REPORT_MD}) and analysis.json hold the record.\n")


def write_pages(report: Dict[str, Any], out_dir) -> Dict[str, str]:
    """REPORT.md, REPORT_DETAILS.md, PLANNED_ANALYSES.md and, when a screen
    ran, SCREEN.md into out_dir; {page name: path}. {} for a report that is
    not written this way. A SCREEN.md an earlier pass left beside a report
    with no screen is set aside, not deleted.

    AN ODD VALUE NEVER STOPS THE PAGES. A part or a page the record cannot
    give says so in one line, and the details page names why. Only a file
    that cannot be written raises, and step4_report then writes the one
    page and sets the companion pages aside."""
    if not applies(report):
        return {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = study_dir(report, out)
    # THE STUDY'S RECORDS, READ ONCE: the comparisons' numbers and titles
    # (the plan), the plan items of each logged record (the log) and of the
    # Analyzer's own figures (step 2's record), and the limits both pages use
    rep = dict(report, _plan=_read_json(out / "analysis_plan.json"),
               _lookups=lookup_items(out), _figure_items=figure_items(out), _problems=[])
    try:
        rep["_limits"] = limits_of(sd, rep)
    except Exception as e:                                          # noqa: BLE001
        _problem(rep, "the limits", e)
        rep["_limits"] = None
    try:
        screen = render_screen(rep)
    except Exception as e:                                          # noqa: BLE001
        _problem(rep, SCREEN_MD, e)
        screen = (f"# The screen\n\nThis page could not be written from the record. "
                  f"[The report]({REPORT_MD}) and analysis.json hold the record.\n")
    main, shown = render_report(rep, sd, screen is not None)
    planned = _page(rep, PLANNED_MD, "The planned comparisons", render_planned, rep, shown)
    pages = {REPORT_MD: main,
             DETAILS_MD: _page(rep, DETAILS_MD, "Details of the report", render_details, rep),
             PLANNED_MD: planned}
    if screen is not None:
        pages[SCREEN_MD] = screen
    elif (out / SCREEN_MD).is_file():
        from agents.analysis.numbers_check import set_aside
        set_aside(out, (SCREEN_MD,), figures=False)
    written = {}
    for name, text in pages.items():
        (out / name).write_text(text)
        written[name] = str(out / name)
    return written


def words(text: str) -> int:
    """The words of a page outside its tables, figures and captions."""
    n = 0
    for line in str(text).splitlines():
        s = line.strip()
        if s.startswith(("|", "![", "**Figure", "**Table", "#")):
            continue
        n += sum(1 for t in s.split() if re.search(r"[A-Za-z0-9]", t))
    return n
