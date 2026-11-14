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

THE FIGURES ARE PIECE F'S AND THE CHECK'S SENTENCE IS PIECE K'S. Where this
module calls their code, a guard marked "JOIN" lets it work without them.
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

SEASONS = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
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


def numbers_words(nums: Sequence[int], word: str = "comparison") -> str:
    """"comparison 5", "comparisons 1 to 4 and 6 to 8"."""
    ns = sorted({int(n) for n in nums})
    if not ns:
        return ""
    runs: List[List[int]] = []
    for n in ns:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = [f"{r[0]} to {r[-1]}" if len(r) > 2 else _join([str(x) for x in r])
             for r in runs]
    return f"{word}{'s' if len(ns) > 1 else ''} {_join(parts)}"


def period_words(period: Any) -> str:
    """A period as a person writes it: "September 2018", "summer 2018",
    "water year 2018", "the year 2018"."""
    p = str(period or "")
    m = re.fullmatch(r"(\d{4})-(\d{2})", p)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}"
    m = re.fullmatch(r"(\d{4})-([A-Z]{3})", p)
    if m and m.group(2) in SEASONS:
        return f"{SEASONS[m.group(2)]} {m.group(1)}"
    m = re.fullmatch(r"WY(\d{4})", p)
    if m:
        return f"water year {m.group(1)}"
    if re.fullmatch(r"\d{4}", p):
        return f"the year {p}"
    return p


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
    # JOIN: piece F's names in words (interface I4); without it, the code
    try:
        from agents.analysis import field_words as FW
    except ImportError:
        return str(field)
    try:
        return str(FW.words(field, with_code=with_code))
    except Exception:                                               # noqa: BLE001
        return str(field)


def units_name(units: Any) -> str:
    """A unit as a person writes it ("gC/m2" for "gC/m^2/month")."""
    # JOIN: piece F's units in words (interface I4); without it, the powers
    # lose their caret and nothing else changes
    try:
        from agents.analysis import field_words as FW
        return str(FW.units_words(units))
    except Exception:                                               # noqa: BLE001
        return str(units or "").replace("^", "")


def identifiers(report: Dict[str, Any]) -> Set[str]:
    """The record's own names: caveats, findings, records, plan and output
    codes. None of them may show on the three short pages."""
    ids = {str(c.get("id")) for c in report.get("caveats") or [] if c.get("id")}
    ids |= {str(c.get("finding_id")) for c in (report.get("claims") or [])
            + (report.get("withheld") or []) if c.get("finding_id")}
    ids |= set(records_of(report))
    for a in (report.get("planned_phase") or {}).get("analyses") or []:
        ids.add(str(a.get("id")))
        ids |= {str(o.get("id")) for o in a.get("outputs") or [] if o.get("id")}
        ids |= {str(o.get("record")) for o in a.get("outputs") or [] if o.get("record")}
    ids |= {str(e.get("id")) for e in (report.get("planned_phase") or {}).get("expectations")
            or [] if e.get("id")}
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
    return dict(((report.get("evidence_trace") or {}).get("records") or {}))


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
                     numbers: Dict[str, int]) -> Optional[int]:
    """The comparison a kept finding answers: the one its finding names
    ("required_p3" is the plan's analysis P3), else the first comparison
    the records it cites were made for."""
    m = re.fullmatch(r"required_(p\d+)", str(c.get("finding_id") or ""), re.I)
    if m and m.group(1).upper() in numbers:
        return numbers[m.group(1).upper()]
    got = [numbers[str(i)] for rid in c.get("evidence") or []
           for i in (records.get(rid) or {}).get("plan_items") or [] if str(i) in numbers]
    return min(got) if got else None


def used_records(report: Dict[str, Any]) -> List[str]:
    """The records the kept findings cite and the tables print, in order."""
    out: List[str] = []
    for c in report.get("claims") or []:
        out += [r for r in c.get("evidence") or [] if r not in out]
    rf = report.get("report_figures") or {}
    out += [r for r in rf.get("printed_records") or [] if r not in out]
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
    L = [f"# {q}", "", (f"Study {name}. " if name else "") + who + (f", {ys}." if ys else ".")]
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
    where = (f"[{DESIGN_MD}](../{DESIGN_MD})," if (sd / DESIGN_MD).is_file() else
             f"{DESIGN_MD}, which this study does not hold,")
    return f"{_p(first)} The design page, {where} describes it and sets it beside measurements."


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
                "; ".join(str(c.get("summary") or c.get("id")) for c in r.get("changes") or [])
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


def choose_findings(report: Dict[str, Any], rf: Dict[str, Any], numbers: Dict[str, int]
                    ) -> Tuple[Dict[str, List[int]], List[Tuple[int, int]]]:
    """({figure id: the kept findings shown under it, by number from 1},
    [(comparison, finding) for a comparison no figure answers]).

    A kept finding belongs to the first figure that answers its comparison.
    A finding whose records all print in the table that follows that figure
    says nothing the table does not, and is left out. At most two under a
    figure, in the plan's order. A comparison that no figure answers shows
    its first kept finding."""
    claims = report.get("claims") or []
    records = records_of(report)
    figs = [f for f in rf.get("figures") or [] if isinstance(f, dict)]
    table_recs: Dict[str, Set[str]] = {}
    for t in rf.get("tables") or []:
        if isinstance(t, dict):
            s = table_recs.setdefault(str(t.get("follows")), set())
            for row in t.get("rows") or []:
                s |= set((row or {}).get("records") or [])
    first_fig: Dict[int, str] = {}
    for f in figs:
        for n in f.get("comparisons") or []:
            first_fig.setdefault(int(n), str(f.get("id")))
    comp = {i: claim_comparison(c, records, numbers) for i, c in enumerate(claims, 1)}
    order = sorted(comp, key=lambda i: (comp[i] if comp[i] is not None else 10 ** 6, i))
    by_fig: Dict[str, List[int]] = {str(f.get("id")): [] for f in figs}
    others: List[Tuple[int, int]] = []
    for i in order:
        n = comp[i]
        if n is None:
            continue
        fid = first_fig.get(n)
        if fid is None:
            if n not in {m for m, _ in others}:
                others.append((n, i))
            continue
        ev = set(claims[i - 1].get("evidence") or [])
        if ev and ev <= table_recs.get(fid, set()):
            continue
        if len(by_fig[fid]) < MAX_PER_FIGURE:
            by_fig[fid].append(i)
    return by_fig, others


def _table_lines(t: Dict[str, Any]) -> List[str]:
    cols = [str(c) for c in t.get("columns") or []]
    rows = [r for r in t.get("rows") or [] if isinstance(r, dict)]
    L = [f"**Table {t.get('number')}.** {_p(t.get('caption'))}", ""]
    if not cols or not rows:
        return L + ["The table holds no row.", ""]
    panel = any(r.get("panel") for r in rows)
    head = [str(t.get("corner") or "")] + cols + (["Panel"] if panel else [])
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(c) for c in (r.get("cells") or [])] + [""] * len(cols)
        L.append("| " + " | ".join([_p(r.get("label"))] + cells[:len(cols)]
                                   + ([str(r.get("panel") or "")] if panel else [])) + " |")
    return L + [""]


def results_lines(report: Dict[str, Any], shown: Set[int]) -> List[str]:
    """Section 3: a subsection per figure of the report's figures record, its
    table, and at most two findings; the comparisons no figure answers; the
    key findings when no figure was drawn. `shown` gains each finding shown."""
    claims = report.get("claims") or []
    plan = report.get("_plan") or {}
    numbers = numbers_of(plan, report)
    rf = report.get("report_figures") or {}
    figs = [f for f in rf.get("figures") or [] if isinstance(f, dict)]
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
    tables = [t for t in rf.get("tables") or [] if isinstance(t, dict)]
    k = 0
    for f in figs:
        k += 1
        n = f.get("number") or k
        L += [f"### 3.{k} {_p(f.get('heading') or 'Figure ' + str(n))}", "",
              f"![Figure {n}]({f.get('file')})", ""]
        comps = [int(x) for x in f.get("comparisons") or []]
        L += [f"**Figure {n}.** {_p(f.get('caption'))}"
              + (f" It answers {numbers_words(comps)}." if comps else ""), ""]
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
    nd = [_sentence(x) for x in rf.get("not_drawn") or []]
    if nd:
        L += [" ".join(nd), ""]
    return L


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
    """(the server's note on a start from a later saved state, in the design
    page's words; the caveat it is on record as), for each such note."""
    exp = _read_json(sd / "experiment.json")
    out = []
    for u in ((exp.get("experiment") or {}).get("check") or {}).get("unusual") or []:
        if isinstance(u, dict) and re.search(r"later saved state", str(u.get("what") or "")):
            try:
                from agents import design_page as DP
                text = DP._noted(u)[2:]
            except Exception:                                       # noqa: BLE001
                text = _sentence(u.get("what"))
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


def limitation_lines(sd: Path, report: Dict[str, Any]) -> List[str]:
    """Section 4: at most six limits. The design page's limits for this
    study in its own sentences, then what only the analysis knows, then the
    note on a later start. Each limit once, with no identifier. The section
    ends with how many more caveats the details page gives whole."""
    caveats = [c for c in report.get("caveats") or [] if isinstance(c, dict)]
    records = records_of(report)
    used = [records[r] for r in used_records(report) if r in records]
    regions = {str(r.get("region")) for r in used if r.get("region")}
    fields = {str(r.get("field")) for r in used if r.get("field")}
    design = design_limits(sd)
    limits: List[Tuple[str, Set[str]]] = []     # (the limit, the caveats it says)

    def add(text: str, ids: Sequence[str] = ()) -> None:
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
    shown = limits[:MAX_LIMITS]
    told = [_norm(t) for t, _ in shown]
    named = {i for _, ids in shown for i in ids}
    # a caveat is said above when a limit names it or holds its words whole
    more = [c for c in caveats if str(c.get("id")) not in named
            and not (_norm(c.get("statement")) and any(_norm(c.get("statement")) in t
                                                        for t in told))]
    L = [f"## {SECTIONS[3]}", ""]
    L += [f"- {_p(t)}" for t, _ in shown] or ["No limit is on record for this study."]
    L.append("")
    L.append(f"The details page gives {len(more)} more caveat{'s' if len(more) != 1 else ''} "
             f"on record in full." if more else
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
    """The second reading of the numbers, in one sentence."""
    en = report.get("evidence_numbers")
    # JOIN: piece K's sentence (interface I2); without it, this one
    try:
        from agents.analysis import numbers_check as NC
        if en is not None and hasattr(NC, "evidence_sentence"):
            s = NC.evidence_sentence(en)
            if s:
                return _p(s) + _not_checked_words(en)
    except Exception:                                               # noqa: BLE001
        pass
    if not en:
        return "No second reading of the numbers is on record for this study."
    if en.get("error") and not en.get("n_compared"):
        return "The second reading of the numbers did not run to its end, so no number was " \
               "made again."
    w = (en.get("largest_relative_difference") or {}).get("rel_diff")
    head = (f"A second reading of the model output, by code written apart from the analysis, "
            f"made {_count(en.get('n_compared'))} numbers again from the "
            f"{len(en.get('records') or {})} records the findings cite")
    if en.get("pass"):
        s = head + (f", with a largest relative difference of {float(w):.3g}." if w is not None
                    else ".")
    else:
        s = head + (f", and {len(en.get('mismatches') or [])} of them did not match.")
    return s + " The values the figures draw were not made again." + _not_checked_words(en)


def _count(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _not_checked_words(en: Dict[str, Any]) -> str:
    left = (en or {}).get("printed_not_checked") or []
    return (f" The {len(left)} record{'s' if len(left) != 1 else ''} the report's tables print "
            f"beyond those were not made again." if left else "")


def checks_lines(report: Dict[str, Any], screen: bool) -> List[str]:
    kept, left = len(report.get("claims") or []), len(report.get("withheld") or [])
    links = [f"[{DETAILS_MD}]({DETAILS_MD}) gives every finding with its records, every "
             f"caveat in full and what code checked."]
    if screen:
        links.append(f"[{SCREEN_MD}]({SCREEN_MD}) gives the first look at every quantity.")
    links.append(f"[{PLANNED_MD}]({PLANNED_MD}) gives each approved comparison.")
    links.append(f"[{DESIGN_MD}](../{DESIGN_MD}) is the design that was approved.")
    return [f"## {SECTIONS[5]}", "", check_sentence(report), "",
            f"The audit kept {kept or 'no'} finding{'s' if kept != 1 else ''} and left out "
            f"{left or 'none'}. "
            f"The details page gives each with its reason.", "",
            JUDGEMENT_WORDS, ""] + [f"- {x}" for x in links] + [""]


def render_report(report: Dict[str, Any], sd: Path, screen: bool) -> Tuple[str, Set[int]]:
    """REPORT.md, and the kept findings it shows (by number from 1)."""
    shown: Set[int] = set()
    L = title_lines(report)
    L += summary_lines(report)
    L += methods_lines(sd, report)
    L += results_lines(report, shown)
    L += limitation_lines(sd, report)
    L += next_step_lines(report)
    L += checks_lines(report, screen)
    return "\n".join(L).rstrip() + "\n", shown


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN.md
# ─────────────────────────────────────────────────────────────────────────────
def screen_records(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in records_of(report).values() if r.get("call") == "screen"]


def screen_opening(report: Dict[str, Any], r: Dict[str, Any]) -> str:
    """What the screen compares, from its record (interface I3). A record
    from before the screen named its run is read as the control run's."""
    scope = ((report.get("evidence_trace") or {}).get("screen_record") or {}).get("scope") or {}
    run = scope.get("run") or "control"
    year = scope.get("year") or r.get("period")
    where = region_words(scope.get("region") or r.get("region"))
    if run != "control":
        return (f"The new run `{run}` against the control run over {where}, every quantity "
                f"and period of {year}.")
    normal = span_words(scope.get("normal_years"))
    return (f"{year} against the normal years{(' ' + normal) if normal else ''}, over {where}, "
            f"every quantity and period of the control run.")


def _rule_sentences(text: Any) -> List[str]:
    """A rule of the server as short sentences: one per clause, split at a
    semicolon or a full stop outside brackets."""
    t, parts, cur, depth = _p(text), [], "", 0
    for k, ch in enumerate(t):
        depth += (ch == "(") - (ch == ")")
        nxt = t[k + 1:k + 2]
        if depth == 0 and ch in ";." and nxt == " " and (ch == ";" or t[k + 2:k + 3].isupper()):
            parts.append(cur)
            cur = ""
            continue
        cur += ch
    parts.append(cur)
    return [_sentence(x.strip()) for x in parts if x.strip().rstrip(".")]


def _drop_empty(head: List[str], rows: List[List[str]]) -> List[str]:
    keep = [k for k in range(len(head)) if k == 0 or any(r[k] for r in rows)]
    out = ["| " + " | ".join(head[k] for k in keep) + " |", "|" + "---|" * len(keep)]
    out += ["| " + " | ".join(r[k] for k in keep) + " |" for r in rows]
    return out + [""]


def render_screen(report: Dict[str, Any]) -> Optional[str]:
    """SCREEN.md, or None when no screen ran."""
    recs = screen_records(report)
    if not recs:
        return None
    sr = (report.get("evidence_trace") or {}).get("screen_record") or {}
    L = ["# The screen", "", f"*{_p(report.get('question') or '')}* Back to "
         f"[the report]({REPORT_MD}).", ""]
    for r in recs:
        L += [screen_opening(report, r), "", f"{SCORE_WORDS} {LOOKING_WORDS}", ""]
        scored, other = [], []
        for key, x in (r.get("rows") or {}).items():
            if not key.startswith("display.") or not isinstance(x, dict):
                continue
            cells = [field_name(x.get("field")), period_words(x.get("period")),
                     "" if x.get("change") is None else _fv(x["change"]),
                     units_name(x.get("change_units") or x.get("units") or ""),
                     "" if x.get("percent") is None else _fv(x["percent"])]
            if x.get("score") is not None:
                scored.append([_p(x.get("group") or "")] + cells
                              + [_fv(x["score"]), "" if x.get("n_normal") is None
                                 else str(x["n_normal"]), _p(x.get("mark") or "")])
            else:
                other.append(cells + [_p(x.get("status") or ""), _p(x.get("mark") or "")])
        L += ["## The rows with a score", ""]
        L += (_drop_empty(["Group", "Quantity", "Period", "Change", "Units", "Percent", "Score",
                           "Normal years", "Mark"], scored) if scored else
              ["No row has a score.", ""])
        L += ["## The rows not scored yet", ""]
        L += (_drop_empty(["Quantity", "Period", "Change", "Units", "Percent", "Status",
                           "Mark"], other) if other else ["Every row shown has a score.", ""])
        c = r.get("screen_csv") or {}
        L += ([f"The whole table, every quantity and period of the screen "
               f"({_fv(c.get('rows'))} rows), is the file [{c['file']}]({c['file']})."]
              if c.get("file") else ["No file of the whole table was written."]) + [""]
    L += ["## How the rows are scored", ""]
    L += [f"- {s}" for s in _rule_sentences(sr.get("scoring_rule"))] or ["Not on record."]
    L += ["", "## Which rows are shown", ""]
    disp = sr.get("display_rule") or ((recs[0].get("words") or {}).get("display_rule"))
    L += [f"- {s}" for s in _rule_sentences(disp)] or ["Not on record."]
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
    rf = report.get("report_figures") or {}
    fields = {str(f) for f in a.get("fields") or []}
    recs = {str(o.get("record")) for o in row.get("outputs") or [] if o.get("record")}
    parts = []
    for f in rf.get("figures") or []:
        if n in [int(x) for x in f.get("comparisons") or []]:
            num = f.get("number")
            letters = [f"{num}{p.get('letter')}" for p in f.get("panels") or []
                       if str(p.get("field")) in fields and p.get("letter")]
            parts.append(f"Figure {_join(letters)}" if letters else f"Figure {num}")
    for t in rf.get("tables") or []:
        if any(recs & set(r.get("records") or []) for r in t.get("rows") or []):
            parts.append(f"Table {t.get('number')}")
    if parts:
        return _join(parts)
    records = records_of(report)
    claims = report.get("claims") or []
    if any(claim_comparison(claims[i - 1], records, numbers) == n for i in shown):
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
    same functions the one page of other reports uses."""
    from agents.analysis import step4_report as S4
    from agents.analysis import domain_report as DR
    from agents.analysis import model_setup as MS
    from agents.analysis import numbers_check as NC
    from agents.analysis.claim_trace import headline_caveat_line
    L = ["# Details of the report", "", f"*{_p(report.get('question') or '')}*", "",
         f"Everything [the report]({REPORT_MD}) leaves out, from the same record, analysis.json. "
         f"Values are shown to 3 significant figures. The exact values are in analysis.json.",
         "", f"*{report.get('model') or 'model'} run, status **{report.get('status')}**, "
         f"generated {report.get('generated_at')}, "
         f"{Path(str(report.get('run_dir') or '')).name}*", ""]
    L += ["## The headline", ""]
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
    claims = report.get("claims") or []
    L += [f"## Every kept finding, with its records ({len(claims)})", ""]
    L += S4.claim_lines(report) or ["None.", ""]
    held = report.get("withheld") or []
    L += [f"## The findings left out, and why ({len(held)})", ""]
    L += (S4.withheld_lines(report) + [""]) if held else ["None.", ""]
    blocking, others = S4.caveat_lines(report)
    L += ["## Every caveat in full", ""]
    if blocking:
        L += ["Caveats that bind the answer:", ""] + blocking + [""]
    if others:
        L += ["Other caveats on record:", ""] + others + [""]
    if not blocking and not others:
        L += ["None.", ""]
    if report.get("evidence_trace"):
        L += ["## How the claims are worded", "", FIXED_RULES, ""]
        L += DR.wording_lines(report) + [""]
    lim = S4.verification_limits(report)
    if lim:
        L += [lim[0], ""] + lim[2:]
    L += ["## The two numbers checks", ""]
    if "numbers_check" in report:
        L += [NC.page_line(report.get("numbers_check"), S4.NUMBERS_LEAD), ""]
    if "evidence_numbers" in report:
        L += [NC.evidence_page_line(report.get("evidence_numbers")), ""]
        left = (report.get("evidence_numbers") or {}).get("printed_not_checked") or []
        if left:
            L += [f"The records the report's tables print that were not made again: "
                  f"{', '.join(left)}.", ""]
    if "numbers_check" not in report and "evidence_numbers" not in report:
        L += ["No numbers check is on record.", ""]
    ms = MS.render(report.get("model_setup")) if report.get("model_setup") else []
    L += ["## The model set-up in full", ""]
    L += ([_p(x) for x in ms[2:]] if ms else ["Not on record.", ""])
    oc = DR.open_cause_lines(report)
    if oc:
        L += ["## The remaining uncertainty, with the runs as the server takes them", ""] \
            + oc + [""]
    timing = DR.timing_lines(report)
    if timing:
        L += ["## Timing", ""] + timing + [""]
    L += planned_record_lines(report)
    if report.get("evidence_trace"):
        L += ["## The figures the Analyzer drew in its own step", ""]
        L += DR.drawn_lines(report, names_only=True) + [""]
    L += report_figure_lines(report)
    for r in screen_records(report):
        c = r.get("screen_csv") or {}
        L += ["## The screen's record", "",
              f"- record {r.get('id')}: year {r.get('period')}, region {r.get('region')}, "
              f"against {(r.get('comparison') or {}).get('against')}; its findings are "
              f"\"{r.get('label') or 'screen'}\""
              + (f"; its whole table {c.get('file')}, {_fv(c.get('rows'))} rows, sha256 "
                 f"{str(c.get('sha256'))[:12]}" if c.get("file") else ""), ""]
    L += ["## Provenance", ""] + S4.provenance_lines(report) + [""]
    return "\n".join(L).rstrip() + "\n"


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
    for f in rf.get("figures") or []:
        L.append(f"- Figure {f.get('number')}: {f.get('file')}, comparisons "
                 f"{', '.join(str(x) for x in f.get('comparisons') or []) or 'none'}")
    for t in rf.get("tables") or []:
        recs = sorted({x for r in t.get("rows") or [] for x in r.get("records") or []})
        L.append(f"- Table {t.get('number')}: records {', '.join(recs) or 'none'}")
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


def write_pages(report: Dict[str, Any], out_dir) -> Dict[str, str]:
    """REPORT.md, REPORT_DETAILS.md, PLANNED_ANALYSES.md and, when a screen
    ran, SCREEN.md into out_dir; {page name: path}. {} for a report that is
    not written this way. A SCREEN.md an earlier pass left beside a report
    with no screen is set aside, not deleted."""
    if not applies(report):
        return {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = study_dir(report, out)
    # THE PLAN, READ ONCE: the comparisons' numbers and titles come from it
    rep = dict(report, _plan=_read_json(out / "analysis_plan.json"))
    screen = render_screen(rep)
    main, shown = render_report(rep, sd, screen is not None)
    pages = {REPORT_MD: main, DETAILS_MD: render_details(rep),
             PLANNED_MD: render_planned(rep, shown)}
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
