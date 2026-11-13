#!/usr/bin/env python3
"""
What the short design page leaves out: DESIGN_DETAILS.md
src/agents/design_details.py

THE PAGE WAS THE OLD LONG PAGE WITH NEW PARTS ADDED (the user's review of
2026-11-13). It ran to about 3,000 words of bullet lists of raw record text,
sentences cut off with an ellipsis, raw settings and limits printed twice. The
person who reads it approved a draft that follows the short page's seven
sections and holds only what the short page leaves out, in tables and two
figures. This module draws that page:

    Your question: the question, the Planner's answer and its steps
    The control run: its set-up as a table, the maps, and where each
        measurement on the short page's figure comes from
    The new run: one small table per run
    How your question will be answered: every comparison with a number, the
        places the numbers are reported for (a table and a map), what the
        comparisons read, the other explanations, the first look at every
        quantity and the checks on the list
    What the answer will not cover: what stands behind each limit of the
        short page (each change's fuller limit, the control run's tables)
    Choices made for you: the model server's check as a table, each earlier
        version it refused, how the question is read
    Your decision: what follows a yes, and that a no uses no computing time

THE TWO PAGES TOGETHER LOSE NOTHING A DECISION RESTS ON. A limit, a finding or
note of the model server, a remark of the review of the comparisons, a
condition on a comparison, a refused round, a choice made without the person,
what the design requires and a change no run has completed are each on one
of the two pages, whole. What neither page prints is bookkeeping or
description: file hashes, raw settings, the route's name, why this model was
chosen, repeats that were cut off. tests/test_design_details.py holds this.

DRAWN BY CODE FROM RECORDS. No language model writes or reads this page. Every
sentence with a number is filled in by code from a record, and a sentence of
the Planner's is printed as the Planner wrote it. A part whose record is
missing is left out, with its heading when its section would be empty, and
the page is still written. A study made before 2026-11-13 has no set-up rows,
no tables, no fuller limits, no caveats of the Planner's own and no map of
places, and still gets a correct page.

Only a whole-domain design has this page. Any other design keeps the page it
had (design_review.render).
"""
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents import design_page as DP

TITLE = "# Design details: {name}"
OPENING = ("This page holds what the short page `DESIGN_REVIEW.md` leaves out. It has the "
           "same sections in the same order. Your decision is recorded on the short page.")
EVERYTHING_ELSE = "as in the control run"
NOT_MEANING = "The difference from the control run is not {text}."
QUESTION_TYPE = {
    "hypothesis": ("The Planner reads the question as one that proposes an expectation to "
                   "test. The comparisons state it before any result, and the report may call "
                   "it tested."),
    "exploratory": ("The Planner reads the question as one that asks what changed, with no "
                    "expectation to test. The report gives its findings as exploratory.")}
NO_QUESTION_TYPE = ("The Planner gave the question no type the report knows, so the report "
                    "reads it as one that asks what changed and gives its findings as "
                    "exploratory.")
NO_COMPUTE = "A no uses no computing time, and the records stay on disk."
# WHERE THE NORMAL YEARS COME FROM, as the list of comparisons records it
# (analysis_plan.normal_years_from), in the words a person reads
NORMAL_FROM = {"the planner": "chosen by the Planner",
               "the control's own (the planner named none)":
                   "the control run's own, since the Planner named none",
               "none named": ""}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth", "tenth")
NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve")
# THE FIELD THE RESULTS STORE CUTS ITS ELEVATION BANDS BY, as the rule of a
# band names it (the server's domain_analysis writes "..., TOPO below 1000 m")
ELEVATION_FIELD = "TOPO"


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _cell(text: Any) -> str:
    """A record's words as a table cell: one line, no bar."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "/")


def _num(v: Any) -> str:
    """A number of a table as a page prints it: a whole number with its
    thousands, any other at three significant figures; anything else as it
    is written."""
    from agents import control_figures as CF
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}" if v == int(v) and abs(v) < 1e15 else CF.three(v)
    return _cell(v)


def _ordinal(i: int) -> str:
    return ORDINALS[i - 1] if 1 <= i <= len(ORDINALS) else f"number {i}"


def _count(n: int) -> str:
    """A count in a sentence: in words up to twelve, in figures after."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else f"{n:,}"


def _steps(n: int) -> str:
    return f"{_count(n)} step" + ("" if n == 1 else "s")


def _lower_first(text: str) -> str:
    """A knob's name inside a sentence: "Rain and snow" becomes "rain and
    snow", and a name such as "CO2" stays as it is."""
    s = str(text or "")
    return s[0].lower() + s[1:] if len(s) > 1 and s[1].islower() else s


def _list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def _dicts(x: Any) -> List[Dict[str, Any]]:
    return [d for d in _list(x) if isinstance(d, dict)]


def _years_words(years: Sequence[Any]) -> str:
    from agents import analysis_preview as AV
    ys = [int(y) for y in years or [] if str(y).lstrip("-").isdigit()]
    return AV._years_words(ys) if ys else "no year"


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    """A markdown table; an empty head cell is drawn as "| |"."""
    return (["|" + "|".join(f" {h} " if h else " " for h in head) + "|",
             "|" + "---|" * len(head)]
            + ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows] + [""])


# ─────────────────────────────────────────────────────────────────────────────
# YOUR QUESTION
# ─────────────────────────────────────────────────────────────────────────────
def _question_lines(question: Any, strategy: Dict[str, Any]) -> List[str]:
    """The question, each change asked for at the short page, the Planner's
    answer, then the Planner's steps as a numbered list."""
    from agents import design_review as DR
    parts = [p.strip() for p in str(question or "").split(DR.REVISION_MARK)]
    L = [parts[0] or "No question is on record.", ""]
    for i, ch in enumerate(parts[1:], 1):
        L += [f"**Change {i} you asked for at the short page.** {ch}", ""]
    L += [DP.planners_answer(strategy), ""]
    # a step that is not text is left out, never printed as Python text
    goals = [DP._sentence(DP._control_run(g)) for g in _list(strategy.get("goals"))
             if isinstance(g, str) and g.strip()]
    if goals:
        L += [f"The Planner reads the question in {_steps(len(goals))}.", ""]
        L += [f"{i}. {g}" for i, g in enumerate(goals, 1)] + [""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL RUN
# ─────────────────────────────────────────────────────────────────────────────
def setup_rows(record: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(label, words) of the control run's set-up: the rows a person entered
    once in the server's registry (the record's `setup`), or [] where none is
    on record.

    NO ROWS, NO TABLE (2026-11-13). For a record without them the table was
    made of the server's own Grid and Fire sentences, which name files and
    raw settings. The short page already describes the control run."""
    return [(str(r.get("label")).strip(), DP._control_run(str(r.get("words")).strip()))
            for r in _dicts((record or {}).get("setup"))
            if str(r.get("label") or "").strip() and str(r.get("words") or "").strip()]


def _measured_rows(record: Optional[Dict[str, Any]]) -> List[List[str]]:
    """One row for each measurement the server paired with the control run:
    the quantity, where the measurement comes from, and the control run's
    line it is set beside."""
    from agents import control_figures as CF
    out = []
    for p in CF.panels(record):
        mine = (f"`{p.get('field')}`, the monthly {p.get('how') or 'value'} in "
                f"{p.get('units')}, over {p.get('where')}")
        for m in p["measured"]:
            out.append([p.get("title"),
                        DP._control_run(m.get("source") or m.get("label") or "not recorded"),
                        mine])
    return out


def _control_lines(row: Dict[str, Any], figures: Optional[Dict[str, Any]]) -> List[str]:
    """The set-up table, the maps with their sentence, the table of the
    measurements, and why a figure was not drawn. How the control run
    compares with measurements in words is printed where the short page has
    no figure of it."""
    from agents import control_figures as CF
    figures = figures if isinstance(figures, dict) else {}
    rec = figures.get("record") if isinstance(figures.get("record"), dict) else None
    drawn = figures.get("drawn") if isinstance(figures.get("drawn"), dict) else {}
    L: List[str] = []
    setup = setup_rows(rec)
    if setup:
        L += _table(["", "The control run"], setup)
    if drawn.get("maps") and rec:
        m = rec.get("maps") if isinstance(rec.get("maps"), dict) else {}
        outline = (m.get("outline") or {}).get("region") if isinstance(m.get("outline"), dict) \
            else None
        spans = {CF._span(p.get("years") or []) for p in CF._maps(rec)}
        L += [f"![Maps of the control run's mean]({drawn['maps']})", ""]
        L += [(f"Each map is the average of {next(iter(spans))} over the model's grid."
               if len(spans) == 1 else "Each map is an average of the run's years over the "
                                       "model's grid.")
              + (f" The line is the edge of the region `{outline}`." if outline else ""), ""]
    measured = _measured_rows(rec)
    if measured:
        L += [("The figure on the short page sets the control run beside these measurements."
               if drawn.get("series") else
               "The model server holds these measurements beside the control run."), ""]
        L += _table(["Quantity", "Measurement", "Compared with"], measured)
    ev = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    if not drawn.get("series") and str(ev.get("summary") or "").strip():
        # with no figure on the short page, the words the registry holds on
        # how the control run compares with measurements, whole
        L += [f"**Against measurements.** {DP._sentence(DP._control_run(ev['summary']))}"
              + (f" The source is {' '.join(str(ev['source']).split()).rstrip('.')}."
                 if str(ev.get("source") or "").strip() else ""),
              ""]
    probs = [str(p) for p in _list((rec or {}).get("problems")) + _list(drawn.get("problems"))
             if str(p).strip()]
    L += [f"- {DP._sentence(DP._control_run(p))}" for p in probs] + ([""] if probs else [])
    return L


# ─────────────────────────────────────────────────────────────────────────────
# THE NEW RUN
# ─────────────────────────────────────────────────────────────────────────────
def _cells_reached(record: Optional[Dict[str, Any]], name: Any) -> Optional[int]:
    """How many grid cells a run's change reaches: the cells inside the
    outline of the server's map of the run, or None without one."""
    from agents import run_figures as RF
    m = (RF.entry(record, name) or {}).get("map")
    o = m.get("outline") if isinstance(m, dict) else None
    try:
        n = sum(1 for r in o for x in r if x) if isinstance(o, list) else None
    except TypeError:
        return None
    return n or None


def _change_rows(r: Dict[str, Any], record, menu, resolved) -> List[Tuple[str, str]]:
    """The rows "Change" and "Where" of one run: from the server's table of
    what the run is given where it has one, else each change in the server's
    own words."""
    from agents import design_review as DR
    from agents import run_figures as RF
    name = r.get("name")
    if not r.get("changes"):
        return [("Change", "none, so the run repeats the control run")]
    knobs = RF.rows(record, name) if record is not None else []
    if knobs:
        said = []
        for k in knobs:
            change = str(k.get("change") or "").strip()
            said.append(f"{_lower_first(str(k.get('knob') or '').strip())} {change}"
                        + ("," if "," in change else "") + f" in {k.get('when')}")
        wheres = list(dict.fromkeys(str(k.get("where")) for k in knobs if k.get("where")))
        n = _cells_reached(record, name)
        where = DP._join(wheres) + (f", {n:,} grid cells" if n and len(wheres) == 1 else "")
        return [("Change", DP._join(said))] + ([("Where", where)] if wheres else [])
    out = []
    for c in r.get("changes") or []:
        m = menu.get(c.get("id")) or {}
        meaning = resolved.get((name, c.get("id"))) or \
            (DR._with_params(m["meaning"], c.get("params")) if m.get("meaning") else None)
        out.append(("Change", DP._sentence(DP._control_run(meaning)) if meaning else
                    f"`{c.get('id')}`, which the model server's list does not describe"))
    return out


def _new_run_lines(brief, strategy, de, menu, row, preview, run_dir, given,
                   by_hand: bool) -> List[str]:
    """One small table per run: its name, what it starts from, the years it
    simulates, its change, where, everything else, and its cost. [] for a
    design with no new run, whose short page says all there is to say."""
    from agents import design_review as DR
    from agents import run_figures as RF
    from core import analysis_years as AY
    runs = DP._runs(de)
    if not runs:
        return []
    ctl = de.get("control") or {}
    _, how = AY.reported(brief, strategy)
    sim = how["simulated"]
    nh = DP._cost(preview, row.get("facts") or {}, de, sim)
    record, drawn = DP._given(given, de)
    resolved = DR._resolved_meanings(run_dir)
    one = len(runs) == 1
    L: List[str] = []
    for r in runs:
        rows: List[Tuple[str, str]] = [("Name", f"`{r.get('name')}`")] if one else []
        if ctl.get("reuse") and sim:
            rows.append(("Starts from", f"the control run's saved state of 1 January {min(sim)}"))
        if sim:
            rows.append(("Simulates", _years_words(sim)))
        try:
            rows += _change_rows(r, record, menu, resolved)
        except Exception:                                           # noqa: BLE001
            rows += _change_rows(r, None, menu, resolved)
        rows.append(("Everything else", EVERYTHING_ELSE))
        if nh is not None and DP._executed(de) == 1:
            rows.append(("Cost", f"about {DP.hours_words(nh)} node-hours, as the model server "
                                 f"estimates"))
        L += _table(["", "The new run" if one else f"`{r.get('name')}`"], rows)
    if nh is not None and DP._executed(de) > 1:
        L += [f"The model server estimates the cost at about {DP.hours_words(nh)} node-hours "
              f"for {'the runs' if ctl.get('reuse') else 'the runs and the control run'}.", ""]
    probs: List[str] = []
    if record is not None:
        probs += [str(p) for p in _list(record.get("problems")) if str(p).strip()]
        for r in runs:
            probs += [f"For `{r.get('name')}`, {p}" for p in
                      _list((RF.entry(record, r.get("name")) or {}).get("problems"))
                      if str(p).strip()]
        probs += [str(p) for p in _list((drawn or {}).get("problems")) if str(p).strip()]
    L += [f"- {DP._sentence(DP._control_run(p))}" for p in probs] + ([""] if probs else [])
    if by_hand:
        L += [f"The steps to make {'this run' if one else 'these runs'} without the framework "
              f"are in `{DP.BY_HAND_NAME}`.", ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# HOW YOUR QUESTION WILL BE ANSWERED
# ─────────────────────────────────────────────────────────────────────────────
def plan_shown(record: Optional[Dict[str, Any]], run_dir, design_sha256: Optional[str]
               ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """(the list of comparisons the short page shows, its review), or (None,
    {}) where the short page shows none: no record, a record of another
    design, comparisons not planned or failed, or a file that cannot be
    read."""
    from agents import analysis_preview as AV
    if not isinstance(record, dict) or AV._other_design(record, design_sha256) or \
            record.get("status") not in (AV.PLANNED, AV.NOT_PASSED):
        return None, {}
    plan, review = AV._read_plan(record, run_dir)
    if not isinstance(plan, dict) or not plan:
        return None, {}
    return plan, review if isinstance(review, dict) else {}


def plan_regions(record: Optional[Dict[str, Any]], run_dir,
                 design_sha256: Optional[str]) -> List[str]:
    """The places the comparisons read, in the order they first name them,
    with the elevation bands a comparison splits a region by (regions_read).
    [] where the short page shows no comparisons."""
    plan, _ = plan_shown(record, run_dir, design_sha256)
    return regions_read(_dicts((plan or {}).get("analyses")), record)


def _bands(region: str, sizes: Dict[str, Any]) -> List[str]:
    """The results store's elevation bands of a region: the regions whose
    label begins with the region's label and a comma, and whose rule names
    the elevation field."""
    head = f"{region}, "
    return [str(k) for k, g in sizes.items() if str(k).startswith(head) and isinstance(g, dict)
            and re.search(rf"\b{ELEVATION_FIELD}\b", str(g.get("rule") or ""))]


def regions_read(analyses: Sequence[Dict[str, Any]], record: Any) -> List[str]:
    """The places the comparisons read, in the order they first name them.

    THE BANDS A COMPARISON SPLITS BY (2026-11-13). In a live study one
    comparison split the watershed by elevation band and named the watershed
    alone, so the table and the map showed the watershed alone. Where an
    output is made by the method elevation_split over a region, that region's
    bands in the record's `regions` follow it."""
    sizes = record.get("regions") if isinstance(record, dict) and \
        isinstance(record.get("regions"), dict) else {}
    used: List[str] = []
    for a in analyses:
        used += [str(r) for r in _list(a.get("regions")) if str(r) not in used]
        for o in _dicts(a.get("outputs")):
            if o.get("kind") == "elevation_split" and o.get("region") is not None:
                used += [b for b in _bands(str(o["region"]), sizes) if b not in used]
    return used


def _no_number_kind(a: Optional[Dict[str, Any]]) -> str:
    """What the outputs of a comparison that gives no number are, in words:
    a daily line, a map, or either (analysis_preview.analysis_support)."""
    outs = [o for o in _list((a or {}).get("outputs")) if isinstance(o, dict)]
    daily = [o.get("kind") == "series" and o.get("step") == "day" for o in outs]
    return ("a daily line" if outs and all(daily) else "a map" if outs and not any(daily)
            else "a map or a daily line")


def _status_lines(plan: Dict[str, Any], analyses: Sequence[Dict[str, Any]]) -> List[str]:
    """One sentence for each comparison code finds conditional or blocked,
    by its number in the table, with its reason whole."""
    from agents import analysis_preview as AV
    number = {str(a.get("id")): i for i, a in enumerate(analyses, 1)}
    by_id = {str(a.get("id")): a for a in analyses}
    out = []
    for aid, s in AV.analysis_support(plan).items():
        k = number.get(aid)
        if s["status"] == AV.CONDITIONAL:
            out.append(f"Comparison {k} gives figures and no number that the report may state, "
                       f"because each of its outputs is {_no_number_kind(by_id.get(aid))}.")
        elif s["status"] == AV.BLOCKED:
            out.append(f"Comparison {k} cannot be made as planned. "
                       + " ".join(AV._sentence(AV._cell(AV._without_id(aid, w)))
                                  for w in s["why"]))
    return [" ".join(out), ""] if out else []


def _places_lines(analyses, record, places) -> List[str]:
    """The places the comparisons read (regions_read), each with its cells,
    its area and its rule as the control run's results store gives them,
    then the map of them where one was drawn (agents/places_figure)."""
    used = regions_read(analyses, record)
    if not used:
        return []
    sizes = record.get("regions") if isinstance(record.get("regions"), dict) else {}
    rows = []
    for r in used:
        g = sizes.get(r) if isinstance(sizes.get(r), dict) else {}
        cells = g.get("cells")
        area = g.get("area_km2")
        rows.append([f"`{r}`",
                     f"{area:,.0f}" if isinstance(area, (int, float)) and not
                     isinstance(area, bool) else "not on record",
                     f"{cells:,}" if isinstance(cells, int) and not isinstance(cells, bool)
                     else "not on record",
                     g.get("rule") or "not on record"])
    L = ["**Where the numbers are reported.** The comparisons read these places and no "
         "others.", ""]
    L += _table(["Place", "Area (km2)", "Cells", "Which cells belong to it"], rows)
    return L + places_figure_lines(places)


def places_figure_lines(places: Optional[Dict[str, Any]]) -> List[str]:
    """The map of the places, as agents/places_figure gives its line, or []
    where none was drawn. Never raises: a map that fails leaves the table."""
    if not isinstance(places, dict) or not isinstance(places.get("record"), dict):
        return []
    try:
        from agents import places_figure as PF
        got = PF.page_lines(places["record"], places.get("drawn"))
    except Exception:                                               # noqa: BLE001
        return []
    got = [str(x) for x in got or []]
    return got + ([""] if got and got[-1] != "" else [])


def _reads_lines(analyses: Sequence[Dict[str, Any]]) -> List[str]:
    """What the comparisons read, as one plain line of every model output
    in the order the comparisons first name them. Nothing is grouped by
    meaning, since code cannot know it.

    ONE LINE, NOT A TABLE (2026-11-13). The other form allowed, one row per
    comparison with the outputs it reads, names each output at least once
    and adds a row for each comparison and a frame, so it is never the
    shorter of the two."""
    every: List[str] = []
    for a in analyses:
        every += [str(f) for f in _list(a.get("fields")) if str(f).strip() and
                  str(f) not in every]
    if not every:
        return []
    n, k = len(analyses), len(every)
    who = "The comparison reads" if n == 1 else f"The {_count(n)} comparisons read"
    them = DP._join([f"`{f}`" for f in every])
    return [f"**What the comparisons read.** {who} {_count(k)} output{'' if k == 1 else 's'} "
            f"of the model. {'It is' if k == 1 else 'They are'} {them}.", ""]


def _competing_lines(plan: Dict[str, Any]) -> List[str]:
    comp = _dicts(plan.get("competing"))
    if not comp:
        return []
    rows = [[DP._control_run(c.get("title")) or DP._join([f"`{f}`" for f in
                                                          _list(c.get("fields"))])
             or "not named", DP._sentence(DP._control_run(c.get("text"))) or "not given"]
            for c in comp]
    n = len(comp)
    return ([f"**Other explanations.** The comparisons also check "
             f"{'one other explanation' if n == 1 else f'{_count(n)} other explanations'}.",
             ""] + _table(["Explanation", "Why it could matter"], rows))


def _screen_lines(plan: Dict[str, Any]) -> List[str]:
    sc = plan.get("screen") if isinstance(plan.get("screen"), dict) else {}
    if not sc:
        return []
    ny = _list((plan.get("baseline") or {}).get("normal_years"))
    when = {"first": "before", "last": "after"}.get(str(sc.get("when")))
    s = (f"**A first look at every quantity.** Every quantity is screened over "
         f"`{sc.get('region')}` for {sc.get('year')}"
         + (f" {when} the comparisons." if when else ".")
         + " The report marks anything that only this screen finds as found by screening "
           "and not as tested.")
    if ny:
        s += (f" The screen judges the size of each change against the control run's spread "
              f"from year to year over {_years_words(ny)}.")
    return [s, ""]


def _normal_lines(plan: Optional[Dict[str, Any]], de: Dict[str, Any],
                  row: Dict[str, Any]) -> List[str]:
    """The normal years where years are set against them: from the list of
    comparisons when one is shown (with the condition code finds when a year
    is set against normal years that contain it), else from the design."""
    if plan is not None:
        b = plan.get("baseline") if isinstance(plan.get("baseline"), dict) else {}
        ny = _list(b.get("normal_years"))
        against = any(isinstance(c, dict) and c.get("against") == "normal years"
                      for c in _list(plan.get("comparisons")))
        if not ny or not (against or b.get("inclusive")):
            return []
        src = " ".join(str(b.get("normal_years_from") or "").split())
        src = NORMAL_FROM.get(src.lower(), DP._control_run(src))
        s = (f"**The normal years.** The comparisons set years against the normal years "
             f"{_years_words(ny)}" + (f", {src}" if src else "") + ".")
        if b.get("inclusive"):
            s += f" {DP._sentence(DP._control_run(b['inclusive']))}"
        return [s, ""]
    an = de.get("analysis") if isinstance(de.get("analysis"), dict) else {}
    mine = [int(y) for y in _list(an.get("normal_years")) if str(y).isdigit()]
    if not mine:
        return []
    own = [int(y) for y in _list(row.get("normal_years")) if str(y).isdigit()]
    s = (f"**The normal years.** The design compares years with the normal years "
         f"{_years_words(mine)}")
    if own and sorted(own) == sorted(mine):
        s += ", the control run's own."
    elif own:
        s += (f". The control run's own normal years are {_years_words(own)}, so the numbers "
              f"will differ from those of studies that used them.")
    else:
        s += "."
    return [s, ""]


def _design_places(de: Dict[str, Any]) -> List[str]:
    """With no list of comparisons, the places the Planner chose, each with
    its reason as the Planner wrote it."""
    an = de.get("analysis") if isinstance(de.get("analysis"), dict) else {}
    regs = [x if isinstance(x, dict) else {"name": x} for x in _list(an.get("regions"))]
    rows = [[f"`{x.get('name')}`" + (f" within `{x['within']}`" if x.get("within") else ""),
             DP._sentence(DP._control_run(x.get("why"))) or "not given"] for x in regs
            if str(x.get("name") or "").strip()]
    if not rows:
        return []
    if all(r[1] == "not given" for r in rows):
        return [f"**Where the numbers are reported.** The Planner chose "
                f"{DP._join([r[0] for r in rows])}.", ""]
    return (["**Where the numbers are reported.** The Planner chose these places.", ""]
            + _table(["Place", "Why the Planner chose it"], rows))


def _checks_lines(plan: Dict[str, Any], review: Dict[str, Any],
                  record: Dict[str, Any]) -> List[str]:
    """Whether the list passed code's check and the review against the
    question, every remark of the review and every problem code found,
    whole; a run the list takes as made; where the list is kept."""
    from agents import analysis_preview as AV
    again = " on the list's second writing" if int(record.get("attempts") or 1) > 1 else ""
    s = "**Checks on this list.** Code checked the Planner's list, and a second model read " \
        "it again against your question."
    if record.get("status") == AV.PLANNED:
        s += f" Both passed{again}."
    else:
        s += " The list did not pass" + (f" on its second writing" if again else "") + "."
        if review.get("follows") is False:
            s += " The review found that it does not follow from the question."
    remarks = [str(x) for x in _list(review.get("model_problems")) if str(x).strip()]
    if review and not remarks:
        s += " The review made no remark."
    lists: List[str] = []
    if remarks:
        lists += [f"The review made {'this remark' if len(remarks) == 1 else 'these remarks'}.",
                  ""] + [f"- {DP._sentence(DP._control_run(x))}" for x in remarks] + [""]
    problems = [str(x) for x in _list(plan.get("problems")) if str(x).strip()]
    if problems:
        lists += [f"Code's check found "
                  f"{'this problem' if len(problems) == 1 else 'these problems'}.",
                  ""] + [f"- {DP._sentence(DP._control_run(x))}" for x in problems] + [""]
    end = []
    # the runs the list reads that are not made yet, as AV.not_made_yet finds
    # them, in the person's words: "the new run" and never its name
    later = list(dict.fromkeys(str(c.get("run")) for c in _dicts(plan.get("comparisons"))
                               if c.get("run") not in (None, "control")))
    if len(later) == 1:
        end.append("The new run does not exist yet, so the list assumes that it will hold the "
                   "same quantities as the control run. Each year is checked after it runs.")
    elif later:
        end.append("The new runs do not exist yet, so the list assumes that they will hold the "
                   "same quantities as the control run. Each year is checked after it runs.")
    if record.get("plan_file"):
        end.append(f"The full list is in this study's folder, at `{record['plan_file']}`.")
    # one paragraph where there is nothing to list between its parts
    if not lists:
        return [" ".join([s] + end), ""]
    return [s, ""] + lists + ([" ".join(end), ""] if end else [])


def _answer_lines(brief, strategy, de, row, record, run_dir, places) -> List[str]:
    """The section "How your question will be answered", from the list of
    comparisons the short page shows, through agents/analysis_preview's
    helpers. With no list, the places and the normal years of the design."""
    from agents import analysis_preview as AV
    from agents import design_review as DR
    runs = DP._runs(de)
    plan, review = plan_shown(record, run_dir, DR.design_fingerprint(strategy))
    if plan is None:
        return _design_places(de) + _normal_lines(None, de, row)
    analyses = _dicts(plan.get("analyses"))
    said = []
    for c in _dicts(plan.get("comparisons")):
        if c.get("against") == "control" and c.get("run") not in (None, "control"):
            who = "the new run" if len(runs) == 1 else f"`{c.get('run')}`"
            said.append(f"We will compare {who} with the control run for "
                        f"{_years_words(_list(c.get('years')))}.")
    L: List[str] = []
    if analyses:
        main = [a for a in analyses if a.get("main") is True]
        shown = len((main or analyses)[:AV.MAIN_SHOWN])
        n = len(analyses)
        first = "the first " if not main else ""
        said.append(f"The short page lists {first}{_count(shown)} comparisons, and this table "
                    f"lists all {_count(n)}." if shown < n else
                    "This table gives each comparison of the short page a number.")
    L += [" ".join(said), ""] if said else []
    if analyses:
        # a comparison's title and what it tells are a model's words, which
        # can say "the control": each is given the name a person reads
        L += _table(["", "What we compare", "When", "How", "What it tells us"],
                    [[str(i), DP._control_run(AV.compare_words(a)), AV.when_words(a, de, row),
                      AV.how_words(a, len(runs)),
                      DP._control_run(a.get("tells") or a.get("question"))]
                     for i, a in enumerate(analyses, 1)])
        L += _status_lines(plan, analyses)
        L += _places_lines(analyses, record, places)
    L += _normal_lines(plan, de, row)
    L += _reads_lines(analyses)
    L += _competing_lines(plan)
    L += _screen_lines(plan)
    L += _checks_lines(plan, review, record)
    return L


# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE ANSWER WILL NOT COVER
# ─────────────────────────────────────────────────────────────────────────────
def _resolved_changes(run_dir) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """{(run, change id): the change as the server's check resolved it}, read
    from the record that holds each change's short limit (`not_covered`)
    and fuller limit (`in_full`). {} for a record of the wrong shape."""
    from agents import design_review as DR
    rec = DR._load(run_dir, DR.RESOLVED_FILE) if run_dir else None
    chk = rec.get("check") if isinstance(rec, dict) else None
    out = {}
    for run in _dicts(chk.get("runs") if isinstance(chk, dict) else None):
        for c in _dicts(run.get("changes")):
            if c.get("id"):
                out[(run.get("name"), c["id"])] = c
    return out


def change_limits(de: Dict[str, Any], menu: Dict[str, Dict[str, Any]], run_dir
                  ) -> "OrderedDict[str, List[Tuple[Any, Any]]]":
    """{the text that stands behind a change's limit: [(run, change id), ...]}
    for each change a run uses, said once however many runs use it. The text
    is the change's fuller limit (`in_full`) where the server gives one, else
    its `not_meaning` after "The difference from the control run is not"."""
    held = _resolved_changes(run_dir)
    out: "OrderedDict[str, List[Tuple[Any, Any]]]" = OrderedDict()
    for r in DP._runs(de):
        for c in _dicts(r.get("changes")):
            rc = held.get((r.get("name"), c.get("id"))) or {}
            full = [DP._sentence(DP._control_run(t)) for t in _list(rc.get("in_full"))
                    if str(t or "").strip()]
            nm = rc.get("not_meaning") or (menu.get(c.get("id")) or {}).get("not_meaning")
            text = (" ".join(full) if full else
                    NOT_MEANING.format(text=DP._control_run(nm).rstrip(".")) if
                    str(nm or "").strip() else "")
            if text:
                out.setdefault(text, []).append((r.get("name"), c.get("id")))
    return out


def _table_title(about: Any, row: Dict[str, Any]) -> str:
    """The title of a table of the control run: the head of what the control
    run does not contain that names the table's subject, as the short page
    gives it, or "" where none does."""
    word = str(about or "").strip().lower()
    for item in _list(row.get("not_in_it_words")):
        head = str(item).split(":")[0].strip()
        if word and word in head.lower():
            return DP._sentence(head)
    return ""


def table_lines(t: Any) -> List[str]:
    """One table of the control record (`tables`), as the page prints it,
    or [] for a table of the wrong shape."""
    if not isinstance(t, dict):
        return []
    cols = [str(c) for c in _list(t.get("columns"))]
    rows = _dicts(t.get("rows"))
    if not cols or not rows or any(len(_list(r.get("values"))) != len(cols) for r in rows):
        return []
    return _table([_cell(t.get("corner") or "")] + [_cell(c) for c in cols],
                  [[r.get("label")] + [_num(v) for v in r["values"]] for r in rows])


def _not_in_it_rest(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(title, sentence) for what a control's record says it does not contain
    beyond the short page's words: the part of `not_in_it` after the words
    the short page prints, or the whole item where the short page has none
    of it. For a record made before the control run's tables."""
    words = [" ".join(str(w).split()).rstrip(".") for w in _list(row.get("not_in_it_words"))]
    out = []
    for item in _list(row.get("not_in_it")):
        s = " ".join(str(item).split())
        w = next((x for x in words if x and s.startswith(x)), None)
        if w is None:
            if s:
                out.append(("", f"The control run does not contain {s.rstrip('.')}."))
            continue
        rest = s[len(w):].lstrip(" ;,.")
        if rest:
            out.append((DP._sentence(w.split(":")[0].strip()),
                        DP._sentence(DP._control_run(rest))))
    return out


def caveats_left_out(de, row, run_dir, told, menu) -> List[str]:
    """The Planner's caveats that neither the short page nor this page
    prints in other words: for a record made before code put them together,
    where the short page leaves them all to this page. A caveat that is the
    server's own limit of a change, a limit for the question, or a sentence
    the short page prints is not printed again."""
    if DP.caveats_by_code(de):
        return []
    _, every = DP._limits_of_changes(run_dir, DP._runs(de))
    if not every:
        return []                       # the short page prints them (caveats_shown)
    held = _resolved_changes(run_dir)
    servers = {DP._same(x) for x in told} | {DP._same(q) for q in DP.planner_limits(de)}
    for r in DP._runs(de):
        for c in _dicts(r.get("changes")):
            rc = held.get((r.get("name"), c.get("id"))) or {}
            nm = rc.get("not_meaning") or (menu.get(c.get("id")) or {}).get("not_meaning")
            if nm:
                servers |= {DP._same(nm), DP._same(f"not {nm}")}
    caveats = [str(c).strip() for c in _list(de.get("caveats")) if str(c).strip()]
    return [c for c in caveats if DP._same(c) not in servers]


def _not_cover_lines(de, row, run_dir, figures, told, menu) -> List[str]:
    """What stands behind each limit of the short page: each change's fuller
    limit, the control run's tables under their leads (or, for a record made
    before them, the rest of what the control run does not contain), and the
    Planner's caveats where the short page leaves them out."""
    L: List[str] = []
    named = {}
    for r in DP._runs(de):
        named[r.get("name")] = len(_dicts(r.get("changes")))
    for text, uses in change_limits(de, menu, run_dir).items():
        runs = list(dict.fromkeys(n for n, _ in uses))
        ids = list(dict.fromkeys(i for _, i in uses))
        many = any(named.get(n, 0) > 1 for n in runs)
        what = (f"The change `{ids[0]}`" if many and len(ids) == 1 else "The change")
        L += [f"**{what} in {DP._join([f'`{n}`' for n in runs])}.** {text}", ""]
    rec = (figures or {}).get("record") if isinstance(figures, dict) else None
    rec = rec if isinstance(rec, dict) else {}
    # THE CONTROL RUN'S TABLES (interface I2) hold the numbers that a record
    # made before them gives in one sentence (`not_in_it`). That sentence is
    # left out where a table of its subject is printed, and kept otherwise,
    # so a table left out for want of a row loses nothing.
    titles: List[str] = []
    for t in _list(rec.get("tables")):
        body = table_lines(t)
        if not body:
            continue
        title = _table_title(t.get("about"), row)
        titles.append(str((t or {}).get("about") or "").strip().lower())
        lead = DP._sentence(DP._control_run(t.get("lead")))
        head = " ".join(x for x in (f"**{title}**" if title else "", lead) if x)
        L += ([head, ""] if head else []) + body
    for title, s in _not_in_it_rest(row):
        if any(a and a in title.lower() for a in titles):
            continue
        L += [(f"**{title}** " if title else "") + s, ""]
    caveats = caveats_left_out(de, row, run_dir, told, menu)
    if caveats:
        L += ["**The Planner's other limits.** The Planner also wrote these limits, which the "
              "short page leaves out.", ""]
        L += [f"- {DP._sentence(DP._control_run(c))}" for c in caveats] + [""]
    if L:
        L = ["The short page gives each limit in a sentence or two. This section gives more on "
             "them.", ""] + L
    return L


# ─────────────────────────────────────────────────────────────────────────────
# CHOICES MADE FOR YOU
# ─────────────────────────────────────────────────────────────────────────────
def _finding_words(f: Any) -> str:
    from agents import design_review as DR
    return DP._sentence(DP._control_run(DR._finding(f, 10 ** 9)))


def check_rows(preview: Optional[Dict[str, Any]], n_runs: int) -> List[List[str]]:
    """The model server's check as rows of what it checked and the result,
    every finding and note whole."""
    if not isinstance(preview, dict):
        return []
    rows = []
    if preview.get("refused"):
        rows.append(["The design against its list of changes and the control run's facts",
                     "refused. " + " ".join(_finding_words(f) for f in preview["refused"])])
        return rows
    if preview.get("error"):
        rows.append(["The design, built before this page",
                     f"not built. {DP._sentence(DP._control_run(preview['error']))}"])
        return rows
    rows.append(["The design against its list of changes and the control run's facts",
                 "passed"])
    if _list(preview.get("runs")):
        rows.append(["The run's recipe, its checks and its job scripts" if n_runs == 1 else
                     "Each run's recipe, its checks and its job scripts",
                     "built, and nothing was submitted"])
    ic = preview.get("inputs_check") if isinstance(preview.get("inputs_check"), dict) else {}
    if ic:
        n = (ic.get("checks") or {}).get("active_cells") if isinstance(ic.get("checks"), dict) \
            else None
        rows.append(["The surface data, the grid file and the starting state, as one set",
                     ("passed" if ic.get("pass") else
                      "failed. " + DP._sentence(DP._control_run(
                          "; ".join(map(str, _list(ic.get("reasons")))))))
                     + (f", {n:,} cells" if isinstance(n, int) and ic.get("pass") else "")])
    chk = preview.get("check") if isinstance(preview.get("check"), dict) else {}
    for u in _list(chk.get("unusual")):
        where = u.get("where") if isinstance(u, dict) else None
        what = u.get("what") if isinstance(u, dict) else u
        rows.append([f"A note on {where}" if where else "A note",
                     DP._sentence(DP._control_run(what))])
    return rows


def _rounds_lines(preview: Optional[Dict[str, Any]]) -> List[str]:
    rounds = _dicts((preview or {}).get("rounds"))
    if not rounds:
        return []
    n = len(rounds)
    L = [f"**Earlier versions.** The model server refused "
         f"{'an earlier version' if n == 1 else f'{n} earlier versions'} of this design, and the "
         f"Planner changed it {'' if n == 1 else 'each time '}before this page.", ""]
    for i, r in enumerate(rounds, 1):
        kept = f", kept as `{r['design']}`," if r.get("design") else ""
        L.append(f"- The {_ordinal(i)} version{kept} was refused for "
                 f"{'this reason' if len(_list(r.get('findings'))) == 1 else 'these reasons'}.")
        L += [f"  - {_finding_words(f)}" for f in _list(r.get("findings"))]
    return L + [""]


def _choice_lines(brief, strategy, preview, de, menu, question) -> List[str]:
    """The server's check as a table, each earlier version it refused, and
    how the Planner reads the question.

    NO REMARK OF RECEPTION (2026-11-13). The short page prints each of them,
    less one that makes the point of the server's sentence it prints. Such a
    remark said here was a repeat, in Reception's words and with the
    server's identifiers."""
    runs = DP._runs(de)
    L: List[str] = []
    rows = check_rows(preview, len(runs))
    if rows:
        listed = DP._choice_lines(brief, strategy, preview, de, menu, question) != \
            [DP.NO_CHOICE, ""]
        L += [("The short page lists the choices. The model server's check stands behind "
               "them." if listed else "The model server checked the design before the short "
                                      "page was written."), ""]
        L += _table(["The model server checked", "Result"], rows)
    L += _rounds_lines(preview)
    an = de.get("analysis") if isinstance(de.get("analysis"), dict) else {}
    qt = str(an.get("question_type") or "").strip().lower()
    if qt:
        L += [f"**How the question is read.** {QUESTION_TYPE.get(qt, NO_QUESTION_TYPE)}", ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# YOUR DECISION
# ─────────────────────────────────────────────────────────────────────────────
def _decision_lines(de, strategy, preview, record, by_hand: bool, n_comparisons: int
                    ) -> List[str]:
    """What follows a yes, as the short page's own words shortened, and that
    a no uses no computing time."""
    from agents import analysis_handoff as AH
    runs = DP._runs(de)
    ctl = de.get("control") or {}
    if DP.cannot_run(strategy, preview, record):
        return ["Nothing can run until the design changes. Nothing has been submitted, and "
                "the records stay on disk.", ""]
    L: List[str] = []
    made = len(runs) + (0 if ctl.get("reuse") else 1)
    if made:
        one = made == 1
        L.append(f"The model server checks the design and builds "
                 f"{'the run' if one else 'the runs'} again from the same records after a yes. "
                 + ("The run is then submitted as a chain of jobs, one for each year, and each "
                    "year is checked before the next one starts." if one else
                    "Each run is then submitted as a chain of jobs, one for each year, and each "
                    "year is checked before the next one starts.")
                 + (f" `{DP.BY_HAND_NAME}` lists the jobs and their checks." if by_hand else ""))
    else:
        L.append("No model run is made after a yes. The control run's existing output is read.")
    L.append("")
    after = ("once the chain ends" if made == 1 else "once the chains end" if made
             else "after your yes")
    if n_comparisons and AH.promised(record):
        one = n_comparisons == 1
        L += [f"The analysis carries out the {_count(n_comparisons)} comparison"
              f"{'' if one else 's'} as {'it is' if one else 'they are'} listed here {after}. "
              f"The analysis stops, and its report says why, if a comparison cannot be carried "
              f"out as listed."
              + ("" if AH.bound(record) else
                 f" The record of the yes names {'the runs and ' if made else ''}the years "
                 f"and not this list."), ""]
    elif n_comparisons:
        L += [f"The comparisons are chosen again {after}, and they may differ from this list.",
              ""]
    else:
        L += [f"The comparisons are chosen {after}.", ""]
    return L + [NO_COMPUTE, ""]


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render(reception, strategy, run_name="this run", preview=None, run_dir=None,
           analysis_preview=None, by_hand: bool = True,
           control_figures: Optional[Dict[str, Any]] = None,
           run_given: Optional[Dict[str, Any]] = None,
           places: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """DESIGN_DETAILS.md of a whole-domain design, or None for any other
    design, which keeps the page it had. The arguments are the short page's
    (design_page.render), and `places` is the map of the places the numbers
    are reported for as design_review.write_review drew it ({"record",
    "drawn"}; agents/places_figure), or None."""
    from agents import design_review as DR
    brief = reception.get("brief") or reception
    if not isinstance(strategy.get("domain_experiment"), dict):
        return None
    de = dict(brief.get("domain_experiment") or {}) \
        if isinstance(brief.get("domain_experiment"), dict) else {}
    de.update(strategy["domain_experiment"])
    if not de:
        return None
    rb = strategy.get("domain_rulebook") or {}
    menu = {c.get("id"): c for c in ((rb.get("changes") or {}).get("changes") or [])
            if isinstance(c, dict)}
    ctl = de.get("control") or {}
    row = next((c for c in ((rb.get("controls") or {}).get("controls") or [])
                if isinstance(c, dict) and c.get("name") == ctl.get("name")), {})
    question = reception.get("user_request") or brief.get("user_request") or ""
    brief = dict(brief, user_request=question)
    runs = DP._runs(de)
    told = DP.differences(question, runs, menu)
    plan, _ = plan_shown(analysis_preview, run_dir, DR.design_fingerprint(strategy))
    n_comparisons = len(_dicts((plan or {}).get("analyses")))
    sections = [
        ("Your question", lambda: _question_lines(question, strategy)),
        ("The control run", lambda: _control_lines(row, control_figures)),
        ("The new runs" if len(runs) > 1 else "The new run",
         lambda: _new_run_lines(brief, strategy, de, menu, row, preview, run_dir, run_given,
                                by_hand)),
        ("How your question will be answered",
         lambda: _answer_lines(brief, strategy, de, row, analysis_preview, run_dir, places)),
        ("What the answer will not cover",
         lambda: _not_cover_lines(de, row, run_dir, control_figures, told, menu)),
        ("Choices made for you",
         lambda: _choice_lines(brief, strategy, preview, de, menu, question)),
        ("Your decision", lambda: _decision_lines(de, strategy, preview, analysis_preview,
                                                  by_hand, n_comparisons))]
    L = [TITLE.format(name=run_name), "", OPENING, ""]
    for head, part in sections:
        # A RECORD OF THE WRONG SHAPE STOPS NO PAGE: its section is left out,
        # and the others are written
        try:
            body = part()
        except Exception:                                           # noqa: BLE001
            body = []
        if any(str(x).strip() for x in body):
            L += [f"## {head}", ""] + body
    while L and L[-1] == "":
        L.pop()
    return "\n".join(L) + "\n"


def words(text: str) -> int:
    return DP.words(text)
