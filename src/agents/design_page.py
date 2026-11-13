#!/usr/bin/env python3
"""
The design page a person decides on: short, and in full sentences
src/agents/design_page.py

THE PAGE WAS TOO LONG AND READ LIKE A FORM (the user's review of 2026-11-12).
Its first part ran to about 1,700 words for one new run. It repeated the
control's fire set-up, its grid and its burned-area table in every study, and
it was a list of fragments drawn from records.

REBUILT SECTION BY SECTION (2026-11-13). The hydrologist who reads the page
asked for more figures, less text, nothing taken too literally, and every
part very clear and concise. The page now says these things, in this order:

    Your question, and the Planner's direct answer to whether this design
        can answer it
    The control run: a short summary written once, with a figure of the run
        beside its measurements where the model server holds them
    The new run: a table of what it is given, and a figure of it beside the
        control run (agents/run_figures.py)
    How your question will be answered: a table of the comparisons
    What the answer will not cover: a short limit for each knob a run uses,
        the Planner's limits for this question, and what the control run
        does not contain
    Choices made for you: where a run differs from the question's words,
        and what was decided or noted without the person
    Your decision: what each answer at the terminal does

Everything else is kept, in two files the page names: DESIGN_DETAILS.md (what
this page leaves out, in the same seven sections; agents/design_details.py)
and RUN_BY_HAND.md (how to make the runs by hand).

SHORT, BUT NOTHING A DECISION RESTS ON IS LEFT OUT. The page is drawn from
the same records as the details, by code, and no language model writes it. A
limit, a remark or a finding of the model server is printed whole or not at
all: this module never shortens a sentence. What the short page leaves out is
description or bookkeeping (the control's set-up, each change's mechanism,
that the server checked the design, that the plan passed its review), and
the details hold all of it. The Planner's caveats are the one condition left
to the details, and only where every change of every run carries the
server's own short limit, which says the same for the knobs used.
tests/test_design_page.py holds the page to that.

THE PLANNER'S RECORD SINCE 2026-11-13 (`caveats_by` "code"). Code puts the
caveats together from the server's words, so a page prints the server's
limits from the server's own records and, of the Planner's words, its limits
for the question and its own caveats (`planner_caveats`), each as its
judgement. A record without `caveats_by` is read as before.

Only a whole-domain design has this page. Any other design keeps the page it
had (design_review.render).
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

DETAILS_NAME = "DESIGN_DETAILS.md"
BY_HAND_NAME = "RUN_BY_HAND.md"
NO_COMPUTE = ("Nothing is submitted, so no computing time is used. The model calls that "
              "prepared this page are already spent.")
CHANGE_ANSWER = ("The design is made again with your change, and you get a new page. "
                 "Nothing is submitted.")
# THE EXAMPLES OF A CHANGE ARE FIXED TEXT (2026-11-13): two that fit any
# study, with no year in them, so an example never names a year the study
# cannot run.
EXAMPLE = 'An example of a change is "use other years" or "also switch the fire off".'
CANNOT_RUN = "This design cannot run as it stands."
NO_LIMIT = "The design records no limit."
NO_CHOICE = "No choice was made for you."
JUDGEMENT = "This is the Planner's judgement."
# THE PLANNER'S DIRECT ANSWER (2026-11-13). The short page printed the verdict
# only where it was not "full", so a person who asked whether the design can
# answer the question read no answer for the common case. It is now said once,
# directly under the question, on both pages.
ANSWER_HEAD = "**The Planner's answer.**"
VERDICT_WORDS = {"full": "This design can answer the question in full.",
                 "partial": "This design can answer part of the question.",
                 "infeasible": "This design cannot answer the question."}
NO_VERDICT = "The Planner gave no verdict on whether this design can answer the question."


def hours_words(nh: Any) -> str:
    """Node-hours as the page prints them, to a tenth: the cost under "The
    new run" and the limit in the decision table are one number, so they are
    printed by one rule (a review of 2026-11-13 found 1.58333333 in one place
    and 1.58333 in the other)."""
    return f"{round(float(nh), 1):g}"


def _span(years: Any) -> str:
    """Years in words: one year, a span of years in a row ("2031 to 2033"),
    or each year where they are not in a row ("2031 and 2033")."""
    ys = sorted({int(y) for y in years or [] if str(y).lstrip("-").isdigit()})
    if not ys:
        return "no year"
    if ys == list(range(ys[0], ys[-1] + 1)):
        return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]} to {ys[-1]}"
    return _join([str(y) for y in ys])


def _join(xs: Sequence[str]) -> str:
    xs = [str(x) for x in xs if str(x).strip()]
    return ("" if not xs else xs[0] if len(xs) == 1 else
            ", ".join(xs[:-1]) + " and " + xs[-1])


def _sentence(text: Any) -> str:
    """A record's words as one sentence: a capital first, a full stop last.
    Nothing between is changed."""
    s = " ".join(str(text or "").split())
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + "."


def _control_run(text: Any) -> str:
    """The server's words with the name a person reads for its control: "the
    control run" everywhere on the page, at the head of a sentence too. Words
    that already say "the control run" are left as they are."""
    s = " ".join(str(text or "").split())
    return re.sub(r"\b([Tt])he control\b(?! run\b)", r"\1he control run", s)


def _says(question: str, words: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", question, re.IGNORECASE) for w in words)


def _asked_otherwise(question: Any, runs: Sequence[Dict[str, Any]],
                     menu: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """(the server's sentence, its entry) for each place a run does something
    other than the question's own words say (see differences)."""
    q, out = str(question or ""), []
    for r in runs or []:
        for c in (r.get("changes") or []) if isinstance(r, dict) else []:
            entry = menu.get(c.get("id")) or {}
            for note in entry.get("asked_otherwise") or []:
                spec = (entry.get("params") or {}).get(note.get("param")) or {}
                value = (c.get("params") or {}).get(note.get("param"), spec.get("default"))
                if value is None or value == note.get("is_not"):
                    continue
                if _says(q, note.get("question_says") or []) and \
                        not _says(q, note.get("and_not") or []):
                    said = _sentence(note.get("say"))
                    if said and said not in [s for s, _ in out]:
                        out.append((said, note))
    return out


def differences(question: Any, runs: Sequence[Dict[str, Any]],
                menu: Dict[str, Dict[str, Any]]) -> List[str]:
    """Where a run does something other than the question's own words say,
    as the model server's change list declares it (`asked_otherwise`): the
    setting named is in use, the question uses one of the words and none of
    the others. The sentence is the server's."""
    return [s for s, _ in _asked_otherwise(question, runs, menu)]


def _same_point(remark: Any, notes: Sequence[Dict[str, Any]]) -> bool:
    """Whether a remark of Reception's makes the point a difference from the
    question already makes: it uses a word the question used and a word of
    what the run changes besides (2026-11-13). Reception writes "the question
    says rainfall, and the change multiplies rain and snow together" where
    the server's sentence says the same, and the page says it once."""
    r = str(remark or "")
    return any(_says(r, n.get("question_says") or []) and _says(r, n.get("and_not") or [])
               for n in notes)


def _question_lines(text: Any) -> List[str]:
    from agents import design_review as DR
    parts = [p.strip() for p in str(text or "").split(DR.REVISION_MARK)]
    L = [f"**Your question.** {parts[0] or '(none on record)'}"]
    for i, ch in enumerate(parts[1:], 1):
        L.append("")
        L.append(f"**Change {i} you asked for at this page.** {ch}")
    return L


def verdict(strategy: Dict[str, Any]) -> str:
    """The Planner's verdict, "full", "partial" or "infeasible", or "" where
    the record holds none of them."""
    fz = strategy.get("feasibility") if isinstance(strategy.get("feasibility"), dict) else {}
    v = str(fz.get("verdict") or "").strip().lower()
    return v if v in VERDICT_WORDS else ""


def planners_answer(strategy: Dict[str, Any]) -> str:
    """The Planner's answer to "can this design answer my question": the
    verdict in words, then its reason as the Planner wrote it. A reason with
    no verdict on record is still printed, so it is never lost. A reason that
    is not text is left out, so a record of the wrong shape is never printed
    as Python text."""
    fz = strategy.get("feasibility") if isinstance(strategy.get("feasibility"), dict) else {}
    why = _sentence(_control_run(fz.get("why"))) if isinstance(fz.get("why"), str) else ""
    return f"{ANSWER_HEAD} {VERDICT_WORDS.get(verdict(strategy), NO_VERDICT)}" + \
        (f" {why}" if why else "")


def caveats_by_code(de: Dict[str, Any]) -> bool:
    """Whether code put the design's caveats together from the server's words
    (the Planner's record since 2026-11-13). The pages then print the server's
    limits from its own records, and never `caveats`."""
    return de.get("caveats_by") == "code"


def _texts(items: Any) -> List[str]:
    out: List[str] = []
    for q in items if isinstance(items, list) else []:
        s = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        if s and s != "None" and s not in out:
            out.append(s)
    return out


def planner_limits(de: Dict[str, Any]) -> List[str]:
    """The Planner's own words a page prints as its judgement: its limits for
    the question, then, for a record whose caveats code put together, its own
    caveats (`planner_caveats`), each once."""
    out = _texts(de.get("question_limits"))
    if caveats_by_code(de):
        out += [c for c in _texts(de.get("planner_caveats")) if _same(c) not in
                {_same(x) for x in out}]
    return out


def _runs(de: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in de.get("runs") or [] if isinstance(r, dict)]


def _executed(de: Dict[str, Any]) -> int:
    """How many runs a yes submits: each new run, and the control run where
    it is made again."""
    return len(_runs(de)) + (0 if (de.get("control") or {}).get("reuse") else 1)


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL RUN
# ─────────────────────────────────────────────────────────────────────────────
def _control_run_lines(de, row, figures=None) -> List[str]:
    """What the control run is, then how it compares with measurements.

    THE COMPARISON IS A FIGURE WHERE THERE IS ONE (the user's review of
    2026-11-12). The summary's paragraph of numbers was about one or two
    years of the run, and a person who asked about other years read numbers
    that were not about their question. Where the control's measurements can
    be drawn (agents/control_figures), the page shows the whole record with
    the years asked about shaded, and one sentence per quantity for those
    years, filled in by code. The summary's `compared` paragraphs are the
    words for a control that has no figure, and are printed only then.

    WHAT THE RUN DOES NOT CONTAIN IS SAID ONCE, under "What the answer will
    not cover" (_not_cover_lines), where a limit printed there may say it
    too."""
    from agents import control_figures as CF
    ctl = de.get("control") or {}
    ps = row.get("page_summary") if isinstance(row.get("page_summary"), dict) else {}

    def paragraphs(key):
        return [str(t).strip() for t in (ps.get(key) or []) if str(t).strip()]
    text, compared = paragraphs("text"), paragraphs("compared")
    figures = figures or {}
    fig = CF.page_lines(figures.get("record"), figures.get("drawn"), figures.get("years") or [])
    L: List[str] = []
    if text:
        for t in text + ([] if fig else compared):
            L += [t, ""]
        L += fig
        L.append(f"This summary of `{ctl.get('name')}` is version {ps.get('version', '?')}"
                 + (f", written {ps['written']}" if ps.get("written") else "")
                 + (f". The maps of the control run, the sources of the measurements and the "
                    f"run's full record are in `{DETAILS_NAME}`." if fig else
                    f". The run's full record is in `{DETAILS_NAME}`."))
    else:
        L.append(f"The control run is `{ctl.get('name')}`."
                 + (f" The model server describes it as {_control_run(row['description'])}."
                    if row.get("description") else "")
                 + f" No short summary of it is on record. Its full record, with how it "
                   f"compares with measurements, is in `{DETAILS_NAME}`.")
        L += ([""] + fig[:-1]) if fig else []
    return L + [""]


# ─────────────────────────────────────────────────────────────────────────────
# THE NEW RUN
# ─────────────────────────────────────────────────────────────────────────────
def _cost(preview, facts, de, sim) -> Optional[float]:
    """The server's estimate of the node-hours, or the control's own hours a
    model year times the years and the runs, or None."""
    nh = (((preview or {}).get("check") or {}).get("cost") or {}).get("node_hours")
    hp = facts.get("hours_per_model_year")
    if nh is None and hp and sim:
        nh = round(_executed(de) * len(sim) * hp, 1)
    return nh


def _given(given: Optional[Dict[str, Any]], de: Dict[str, Any]):
    """(the record of what the runs are given, what was drawn of it), or
    (None, None) for a record of another design or none."""
    from agents import run_figures as RF
    try:
        rec = given.get("record") if isinstance(given, dict) else None
        if RF.of_design(rec, de):
            return rec, given.get("drawn") or {}
    except Exception:                                               # noqa: BLE001
        pass
    return None, None


def _new_run_lines(brief, strategy, de, menu, row, preview, run_dir,
                   given=None) -> List[str]:
    """The section "The new run": what will run and what it costs, the table
    of the knobs and the figure of what each run is given, from the server's
    record (agents/run_figures.py). A run the record has no row for is said
    in the server's own words for each change, as the page said every run
    before the table, so nothing is lost for an older study."""
    from agents import design_review as DR
    from agents import run_figures as RF
    from core import analysis_years as AY
    runs = _runs(de)
    ctl = de.get("control") or {}
    years, how = AY.reported(brief, strategy)
    sim = how["simulated"]
    nh = _cost(preview, row.get("facts") or {}, de, sim)
    cost = (f" The model server estimates the cost at about {hours_words(nh)} node-hours."
            if nh is not None else "")
    if not runs:
        if ctl and not ctl.get("reuse"):
            return [f"No run with a change is made. The control run is made again from its "
                    f"own recipe and read for {_span(years)}." + cost, ""]
        return ["No new run is made. The control run's existing output is read"
                + (f" for {_span(years)}." if years else "."), ""]
    names = [str(r.get("name")) for r in runs]
    several = len(runs) > 1
    head = (f"We will make one new run, `{names[0]}`." if not several else
            f"We will make {len(runs)} new runs, {_join([f'`{n}`' for n in names])}.")
    if sim:
        # WHERE A RUN STARTS (2026-11-12): from the control run's saved state
        # of its own first year, whichever year that is.
        who = "Each" if several else "It"
        head += (f" {who} starts from the control run's saved state at the beginning of "
                 f"{min(sim)} and simulates {_span(sim)}." if ctl.get("reuse") else
                 f" {who} simulates {_span(sim)}.")
    L = [head + cost, ""]
    record, drawn = _given(given, de)
    try:
        L += RF.table_lines(record, names, several)
    except Exception:                                               # noqa: BLE001
        record, drawn = None, None
    resolved = DR._resolved_meanings(run_dir)
    said: List[str] = []
    for r in runs:
        name = r.get("name")
        if not r.get("changes"):
            said.append(f"`{name}` names no change, so it repeats the control run.")
            continue
        if record is not None and RF.rows(record, name):
            continue
        for c in r.get("changes") or []:
            m = menu.get(c.get("id")) or {}
            meaning = resolved.get((name, c.get("id"))) or \
                (DR._with_params(m["meaning"], c.get("params")) if m.get("meaning") else None)
            said.append(f"In `{name}`, {_control_run(meaning).rstrip('.')}." if meaning else
                        f"`{name}` makes the change `{c.get('id')}`, which the model server's "
                        f"list does not describe.")
    # A CHANGE IN EFFECT IN EVERY SIMULATED YEAR (2026-11-10), said by code
    # for every change that does not set its years itself.
    whole = sorted({c.get("id") for r in runs for c in (r.get("changes") or [])
                    if c.get("id") and not (c.get("params") or {}).get("years")
                    and not any(isinstance(a, dict) and "years" in a
                                for a in (menu.get(c.get("id")) or {}).get("alters") or [])})
    if whole and how["left_out"]:
        said.append(f"The change {_join([f'`{w}`' for w in whole])} is in effect in every "
                    f"simulated year, {_span(sim)}, and not only in the years reported. A "
                    f"reported year carries its effect from {min(sim)} on.")
    if said:
        L += said + [""]
    if record is not None:
        try:
            L += RF.figure_lines(record, drawn, names, several)
        except Exception:                                           # noqa: BLE001
            pass
    return L


# ─────────────────────────────────────────────────────────────────────────────
# HOW YOUR QUESTION WILL BE ANSWERED
# ─────────────────────────────────────────────────────────────────────────────
def _normal_sentence(c: Dict[str, Any], also: bool) -> str:
    """One comparison of years with normal years, in a sentence."""
    from agents import analysis_preview as AV
    control = c.get("run") in (None, "control")
    who = "the control run" if control else f"`{c.get('run')}`"
    ys, ny, held = c.get("years") or [], c.get("normal") or [], c.get("held") or []
    lead = "We will also compare" if also else "We will compare"
    if ys and sorted(set(ys)) == sorted(set(ny)):
        # the years asked about ARE the normal years: said once, plainly
        return f"{lead} each year of {_span(ys)} in {who} with the mean of those same years."
    return (f"{lead} {_span(ys)} in {who} with {'its' if control else 'the'} normal years, "
            f"{_span(ny)}" + (f", which contain {AV._years_words(held)}" if held else "") + ".")


def _answer_lines(brief, strategy, de, row, analysis_preview, run_dir) -> List[str]:
    """The section "How your question will be answered": what is compared with
    what and the years reported, then the comparisons, from the plan written
    before the page when there is one (agents/analysis_preview.short_lines)."""
    from agents import analysis_preview as AV
    from agents import design_review as DR
    from core import analysis_years as AY
    runs = _runs(de)
    years, how = AY.reported(brief, strategy)
    sha = DR.design_fingerprint(strategy)
    normal = AV.normal_comparisons(analysis_preview, run_dir, sha)
    said: List[str] = []
    if runs:
        said.append(f"We will compare {'the new run' if len(runs) == 1 else 'each new run'} "
                    f"with the control run.")
    if normal is None and not runs and how["normal"]:
        # no plan to read: the design's own years against its normal years
        asked = how["asked"] or years
        normal = [{"run": "control", "years": asked, "normal": how["normal"],
                   "held": [y for y in asked if y in how["normal"]]}]
    for c in normal or []:
        said.append(_normal_sentence(c, also=bool(said)))
    if not said:
        said.append("We will read the control run's existing output.")
    if how["simulated"]:
        said.append(f"The numbers are reported for {_span(years)}"
                    + (f", the year{'' if len(years) == 1 else 's'} you asked about."
                       if how["source"] == "request" and set(years) == set(how["asked"])
                       else "."))
        extra = [y for y in how["normal"] if y not in how["asked"]]
        if extra and runs and not normal and how["source"] == "request":
            said.append(f"The design compares them with {_span(extra)}, its normal years.")
    # A COMPARISON'S WORDS ARE A MODEL'S (2026-11-13): its title, question or
    # what it tells can say "the control", so each line, and each cell of the
    # table, is given the name a person reads
    return [" ".join(said), ""] + [_control_run(x) for x in AV.short_lines(
        analysis_preview, run_dir, sha, de, row, n_runs=len(runs))]


# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE ANSWER WILL NOT COVER
# ─────────────────────────────────────────────────────────────────────────────
def _same(text: Any) -> str:
    return " ".join(str(text or "").split()).rstrip(".").lower()


def _limits_of_changes(run_dir, runs: Sequence[Dict[str, Any]]) -> Tuple[List[str], bool]:
    """(the server's short limit of each knob a run uses, once each; whether
    every change of every run carries one). The limits are the key
    `not_covered` of each change the server's check resolved, in
    01_inputs/domain_control_facts.json. A record without the key is one made
    before 2026-11-13, and the page then prints the Planner's caveats."""
    from agents import design_review as DR
    held: Dict[Tuple[Any, Any], Any] = {}
    try:
        rec = DR._load(run_dir, DR.RESOLVED_FILE) if run_dir else None
        for run in ((rec or {}).get("check") or {}).get("runs") or []:
            if isinstance(run, dict):
                for c in run.get("changes") or []:
                    if isinstance(c, dict) and c.get("id"):
                        held[(run.get("name"), c["id"])] = c.get("not_covered")
    except (AttributeError, TypeError):
        # a record of the wrong shape is read as none: the caveats are printed
        held = {}
    out: List[str] = []
    every, n = True, 0
    for r in runs:
        for c in r.get("changes") or []:
            n += 1
            got = held.get((r.get("name"), c.get("id")))
            if not isinstance(got, list):
                every = False
                continue
            for t in got:
                s = _sentence(_control_run(t))
                if s and s not in out:
                    out.append(s)
    return out, every and n > 0


def caveats_shown(de: Dict[str, Any], run_dir, told: Sequence[str] = ()) -> List[str]:
    """The Planner's caveats the short page prints: none for a record whose
    caveats code put together, none where every change of every run carries
    the server's short limit, and otherwise each caveat that is not one of
    the Planner's limits for the question or a sentence `told` already."""
    if caveats_by_code(de):
        return []
    _, every = _limits_of_changes(run_dir, _runs(de))
    if every:
        return []
    caveats = [str(c).strip() for c in de.get("caveats") or [] if str(c).strip()]
    told_set = {_same(x) for x in told} | {_same(q) for q in planner_limits(de)}
    return [c for c in caveats if _same(c) not in told_set]


def _not_cover_lines(strategy, de, row, run_dir, told: Sequence[str] = ()) -> List[str]:
    """The section "What the answer will not cover", in this order: what
    the question would also need; the server's short limit of each knob a
    run uses; the Planner's limits for this question; what the control run
    does not contain, once. The Planner's verdict is said under the question
    since 2026-11-13 (planners_answer), and not here.

    THE PLANNER'S CAVEATS (2026-11-13). They repeated, at length and for every
    knob, what a run is not, and the person called them too long. Where every
    change of every run carries the server's short limit, they are left to
    DESIGN_DETAILS.md, which prints each one whole. Where any change lacks
    one (a record made before), they are printed here as before, so nothing
    is lost. A caveat that is word for word a sentence under "Choices made
    for you" is not printed a second time. A record whose caveats code put
    together (caveats_by_code) has its own caveats printed with its limits
    for the question, and the server's limits come from the server's
    records alone."""
    runs = _runs(de)
    L: List[str] = []
    for need in strategy.get("requires") or []:
        L.append(f"- The question would also need something that is not available here. "
                 f"{_sentence(_control_run(need))}")
    limits, every = _limits_of_changes(run_dir, runs)
    L += [f"- {s}" for s in limits]
    mine = planner_limits(de)
    shown: List[str] = []
    if not every:
        # A LIMIT FOR THE QUESTION IS ALSO A CAVEAT (2026-11-13). The Planner
        # copies each of its limits into the caveats, word for word, so that
        # the report carries it. Where the page prints the caveats (a design
        # with no change, or an older record), such a caveat is the limit
        # printed below as the Planner's judgement, and is not printed twice:
        # a live page of a study of the control run alone gave each one two
        # times.
        shown = caveats_shown(de, run_dir, told)
        L += [f"- {_sentence(_control_run(c))}" for c in shown]
    L += [f"- {_sentence(_control_run(q))} {JUDGEMENT}" for q in mine]
    # WHAT THE CONTROL RUN DOES NOT CONTAIN, ONCE (2026-11-12): in the words of
    # a limit printed above where one names the thing, and in the server's
    # words where none does, so it is never lost.
    near = [x.lower() for x in shown + mine]
    for item in row.get("not_in_it_words") or []:
        head = str(item).split(":")[0].strip().lower()
        if head and not any(head in x for x in near):
            L.append(f"- The control run does not contain {str(item).strip().rstrip('.')}.")
    if not L:
        return [NO_LIMIT, ""]
    return L + [""]


# ─────────────────────────────────────────────────────────────────────────────
# CHOICES MADE FOR YOU
# ─────────────────────────────────────────────────────────────────────────────
def _noted(finding: Any) -> str:
    """A finding of the model server's check, whole: where it is, then what
    the server says of it."""
    if isinstance(finding, dict) and finding.get("where"):
        return (f"- The model server has a note on {finding['where']}. "
                f"{_sentence(_control_run(finding.get('what')))}")
    what = finding.get("what") if isinstance(finding, dict) else finding
    return f"- The model server has a note. {_sentence(_control_run(what))}"


def _untested_noted(preview) -> List[str]:
    """The changes the server's check already calls not demonstrated, by the
    id its note names ("run a, change b"), on a page that prints those notes."""
    if not preview or _blocked(preview):
        return []
    out = []
    for u in ((preview.get("check") or {}).get("unusual") or []):
        if not isinstance(u, dict):
            continue
        m = re.search(r"\bchange (\S+)$", str(u.get("where") or ""))
        if m and str(u.get("what") or "").lower().startswith("not demonstrated"):
            out.append(m.group(1))
    return out


def _blocked(preview) -> bool:
    return bool(preview and (preview.get("refused") or preview.get("error")))


def _choice_lines(brief, strategy, preview, de, menu, question) -> List[str]:
    """The section "Choices made for you", in this order: where a run differs
    from the question's own words; what Reception decided without the
    person, less a remark that makes the same point; what Reception noted;
    the notes of the model server's check; a change no run has completed.

    LEFT TO DESIGN_DETAILS.md (2026-11-13): the Planner's verdict where it is
    "full", that the server checked and built the design, that the plan
    passed its check and its review, the review's remarks and that a run is
    not made yet. They are bookkeeping for a person who decides, and the
    details print each of them."""
    from agents import design_review as DR
    runs = _runs(de)
    rb = strategy.get("domain_rulebook") or {}
    L: List[str] = []
    pairs = _asked_otherwise(question, runs, menu)
    L += [f"- {s}" for s, _ in pairs]
    notes = [n for _, n in pairs]
    rs = brief.get("run_settings") or {}
    for item in list(brief.get("assumptions") or []) + list(rs.get("conflicts") or []):
        if str(item).strip() and not _same_point(item, notes):
            L.append(f"- {_sentence(_control_run(item))}")
    for item in brief.get("known_data_gaps") or []:
        if str(item).strip():
            L.append(f"- Noted by Reception: {_sentence(_control_run(item))}")
    if preview:
        for i, r in enumerate(preview.get("rounds") or [], 1):
            L.append(f"- The model server refused {'an earlier' if i == 1 else 'another'} "
                     f"version of this design, and the Planner changed it before this page:")
            L += [f"  - {_sentence(_control_run(DR._finding(f, 100000)))}"
                  for f in r.get("findings") or []]
        if not _blocked(preview):
            for u in ((preview.get("check") or {}).get("unusual") or []):
                L.append(_noted(u))
            ic = preview.get("inputs_check") or {}
            if ic and not ic.get("pass"):
                L.append("- The input files failed their check. "
                         + _sentence(_control_run("; ".join(map(str, ic.get("reasons") or [])))))
            for p in preview.get("problems") or []:
                L.append(f"- Not shown in the details: {_sentence(_control_run(p))}")
    # A CHANGE NO RUN HAS COMPLETED IS SAID ONCE (2026-11-13). The server's check
    # names it among its notes, under the run that uses it, and a live page then
    # said it a second time in the change list's words. The list's words are
    # kept for a design the server's check did not reach.
    seen = set(_untested_noted(preview))
    for r in runs:
        for c in r.get("changes") or []:
            cid = c.get("id")
            if (menu.get(cid) or {}).get("demonstrated") is False and cid not in seen:
                seen.add(cid)
                L.append(f"- The model server has a note on the change `{cid}`. "
                         f"{_sentence((rb.get('changes') or {}).get('demonstrated_false'))}")
    if not L:
        return [NO_CHOICE, ""]
    return L + [""]


# ─────────────────────────────────────────────────────────────────────────────
# YOUR DECISION
# ─────────────────────────────────────────────────────────────────────────────
def why_not(strategy, preview, analysis_preview) -> List[str]:
    """Why the design cannot run, as the page said it before 2026-11-13,
    whole: the server refuses it, it could not be built, or no comparison a
    yes could approve was planned. [] for a design that can run, or one the
    Planner judges cannot answer the question: that answer and its reason
    are said once, under the question (planners_answer)."""
    from agents import analysis_handoff as AH
    from agents import design_review as DR
    L: List[str] = []
    if preview and preview.get("refused"):
        L.append("- The model server refuses this design as it stands:")
        L += [f"  - {_sentence(_control_run(DR._finding(f, 100000)))}" for f in preview["refused"]]
    elif preview and preview.get("error"):
        L.append(f"- The design could not be built before this page. "
                 f"{_sentence(_control_run(preview['error']))}")
    if AH.unapprovable(analysis_preview):
        L.append("- The comparisons did not pass their check, so a yes could approve none of "
                 "them." if (analysis_preview or {}).get("status") == "not passed" else
                 "- The comparisons could not be planned before this page, so a yes could "
                 "approve none.")
    return L


def cannot_run(strategy, preview, analysis_preview) -> bool:
    """Whether the design has no yes: a reason of why_not, or the Planner's
    answer that it cannot answer the question."""
    return bool(why_not(strategy, preview, analysis_preview)) or \
        verdict(strategy) == "infeasible"


def _decision_lines(de, strategy, preview, by_hand: bool, analysis_preview=None) -> List[str]:
    """The section "Your decision": a table of what each answer at the
    terminal does, or, for a design that cannot run, why and the two answers
    it has."""
    from agents import analysis_handoff as AH
    from core import approved_spending as SP
    reasons = why_not(strategy, preview, analysis_preview)
    n = _executed(de)
    if reasons or cannot_run(strategy, preview, analysis_preview):
        L = reasons + ([""] if reasons else []) + [
                CANNOT_RUN, "", "| You type | What happens |", "|---|---|",
                       "| a change, in your own words | The design is made again with your "
                       "change and checked again. |",
                       "| nothing | The request stops. Nothing has been submitted. |", ""]
    else:
        if n:
            # WHAT A YES ALLOWS (stage 4): the computing the server estimated,
            # as a limit checked before any job (core/approved_spending.py)
            yes = "The run is submitted." if n == 1 else "The runs are submitted."
            nh = SP.estimate(preview)
            if nh is not None:
                yes += (f" {'It' if n == 1 else 'They'} may use up to {hours_words(nh)} "
                        f"node-hours.")
        else:
            yes = ("The analysis of the control run's existing output starts. No model run is "
                   "made, and the analysis uses paid model calls.")
        if AH.bound(analysis_preview):
            yes += " The comparisons above are carried out as listed."
        L = ["Type one of these at the terminal.", "", "| You type | What happens |",
             "|---|---|", f"| yes | {yes} |", f"| no | {NO_COMPUTE} |",
             f"| a change, in your own words | {CHANGE_ANSWER} |", ""]
    # WHAT RUN_BY_HAND.md HOLDS FOLLOWS THE NEW RUNS (2026-11-13): with none it
    # says how the control run was made, so "the runs" named runs it does not hold
    new = len(_runs(de))
    hand = (f" How the control run was made is in `{BY_HAND_NAME}`." if not new else
            f" The steps to make the {'run' if new == 1 else 'runs'} by hand are in "
            f"`{BY_HAND_NAME}`.")
    L += [EXAMPLE, "", f"The design in full is in `{DETAILS_NAME}`." + (hand if by_hand else ""),
          ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render(reception, strategy, run_name="this run", preview=None, run_dir=None,
           analysis_preview=None, by_hand: bool = True,
           control_figures: Optional[Dict[str, Any]] = None,
           run_given: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The short design page of a whole-domain design, or None for any other
    design, which keeps the page it had. `control_figures`: the control's
    figures as design_review.write_review drew them ({"record", "drawn",
    "years"}); `run_given`: what the new runs are given, as it drew them
    ({"record", "drawn"}); either None for a page without them."""
    from agents import design_review as DR
    brief = reception.get("brief") or reception
    de = dict(brief.get("domain_experiment") or {})
    de.update(strategy.get("domain_experiment") or {})
    if not de or not isinstance(strategy.get("domain_experiment"), dict):
        return None
    rb = strategy.get("domain_rulebook") or {}
    menu = {c.get("id"): c for c in ((rb.get("changes") or {}).get("changes") or [])
            if isinstance(c, dict)}
    ctl = de.get("control") or {}
    row = next((c for c in ((rb.get("controls") or {}).get("controls") or [])
                if isinstance(c, dict) and c.get("name") == ctl.get("name")), {})
    question = reception.get("user_request") or brief.get("user_request") or ""
    brief = dict(brief, user_request=question)
    runs = _runs(de)
    L = [f"# Design review: {run_name}", ""]
    L += _question_lines(question) + [""]
    L += [planners_answer(strategy), ""]
    L += ["## The control run", ""] + _control_run_lines(de, row, control_figures)
    L += [f"## {'The new runs' if len(runs) > 1 else 'The new run'}", ""]
    L += _new_run_lines(brief, strategy, de, menu, row, preview, run_dir, run_given)
    L += ["## How your question will be answered", ""]
    L += _answer_lines(brief, strategy, de, row, analysis_preview, run_dir)
    L += ["## What the answer will not cover", ""]
    L += _not_cover_lines(strategy, de, row, run_dir, differences(question, runs, menu))
    L += ["## Choices made for you", ""]
    L += _choice_lines(brief, strategy, preview, de, menu, question)
    L += ["## Your decision", ""] + _decision_lines(de, strategy, preview, by_hand,
                                                    analysis_preview)
    L += ["---", DR._PENDING, ""]
    return "\n".join(L)


def words(text: str) -> int:
    """The words of a page: tokens that hold a letter or a digit."""
    return sum(1 for t in str(text).split() if re.search(r"[A-Za-z0-9]", t))
