#!/usr/bin/env python3
"""
The design page a person decides on: short, and in full sentences
src/agents/design_page.py

THE PAGE WAS TOO LONG AND READ LIKE A FORM (the user's review of 2026-11-12).
Its first part ran to about 1,700 words for one new run. It repeated the
control's fire set-up, its grid and its burned-area table in every study, and
it was a list of fragments drawn from records. A person decides on six
things, and the page now says those and nothing else:

    the question
    what will run, what it costs, and where it differs from what was asked
    the control run, in a short summary written once, with a figure of the
        run beside its measurements where the model server holds them
    how the answer will be read
    what the study cannot tell
    the points to weigh, then the decision

Everything else is kept, in two files the page names: DESIGN_DETAILS.md (the
design and the analysis plan in full, as the page used to print them) and
RUN_BY_HAND.md (how to make the runs by hand).

SHORT, BUT NOTHING A DECISION RESTS ON IS LEFT OUT. The page is drawn from
the same records as the details, by code, and no language model writes it. A
limit, a caveat, a remark of a review or a finding of the model server is
printed whole or not at all: this module never shortens a sentence. What is
left out is description (the control's set-up, each change's mechanism, the
scripts), never a condition. tests/test_design_page.py holds it to that: every
caveat, every server finding, every review remark and every difference from
the question that the details hold is on the page.

Only a whole-domain design has this page. Any other design keeps the page it
had (design_review.render).
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DETAILS_NAME = "DESIGN_DETAILS.md"
BY_HAND_NAME = "RUN_BY_HAND.md"
NOT_BUILT = ("Your yes does not approve this plan. It is shown as a preview, and the "
             "Analyzer writes its plan again after the runs.")
NO_COMPUTE = ("A no submits nothing, so no computing time is used. The model calls that "
              "prepared this page are already spent.")


def _span(years: Any) -> str:
    ys = sorted({int(y) for y in years or [] if str(y).lstrip("-").isdigit()})
    if not ys:
        return "no year"
    return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]} to {ys[-1]}"


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


def differences(question: Any, runs: Sequence[Dict[str, Any]],
                menu: Dict[str, Dict[str, Any]]) -> List[str]:
    """Where a run does something other than the question's own words say,
    as the model server's change list declares it (`asked_otherwise`): the
    setting named is in use, the question uses one of the words and none of
    the others. The sentence is the server's."""
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
                    if said and said not in out:
                        out.append(said)
    return out


def _question_lines(text: Any) -> List[str]:
    from agents import design_review as DR
    parts = [p.strip() for p in str(text or "").split(DR.REVISION_MARK)]
    L = [f"**Your question.** {parts[0] or '(none on record)'}"]
    for i, ch in enumerate(parts[1:], 1):
        L.append("")
        L.append(f"**Change {i} you asked for at this page.** {ch}")
    return L


def _runs_lines(brief, strategy, de, menu, row, preview, run_dir) -> List[str]:
    from agents import design_review as DR
    from core import analysis_years as AY
    rb = strategy.get("domain_rulebook") or {}
    runs = [r for r in de.get("runs") or [] if isinstance(r, dict)]
    ctl = de.get("control") or {}
    facts = row.get("facts") or {}
    resolved = DR._resolved_meanings(run_dir)
    years, how = AY.reported(brief, strategy)
    sim = how["simulated"]
    L: List[str] = []
    if not runs:
        L.append("No new run is made. The control run's existing output is read"
                 + (f" for {_span(years)}." if years else "."))
        if ctl and not ctl.get("reuse"):
            L[-1] = (f"No run with a change is made. The control run is made again from "
                     f"its own recipe and read for {_span(years)}.")
        return L + [""]
    L.append(f"We will make {'one new run' if len(runs) == 1 else f'{len(runs)} new runs'}.")
    for r in runs:
        name = r.get("name")
        for c in r.get("changes") or []:
            m = menu.get(c.get("id")) or {}
            said = resolved.get((name, c.get("id"))) or \
                (DR._with_params(m["meaning"], c.get("params")) if m.get("meaning") else None)
            L.append(f"In `{name}`, {_control_run(said).rstrip('.')}." if said else
                     f"`{name}` makes the change `{c.get('id')}`, which the model server's "
                     f"list does not describe.")
            if m.get("demonstrated") is False:
                L.append(_sentence((rb.get("changes") or {}).get("demonstrated_false")))
        if not r.get("changes"):
            L.append(f"`{name}` names no change, so it repeats the control run.")
    if sim:
        # WHERE A RUN STARTS (2026-11-12): from the control run's saved state
        # of its own first year, whichever year that is. The page said so only
        # of the control run's first year, as the reason for years simulated
        # and not asked about, when every run had to start there.
        first = (f"Each new run starts from the control run's saved state at the "
                 f"beginning of {min(sim)} and simulates {_span(sim)}."
                 if (de.get("control") or {}).get("reuse") else
                 f"Each new run simulates {_span(sim)}.")
        L.append(first + f" The numbers are reported for {_span(years)}"
                 + (f", the year{'' if len(years) == 1 else 's'} you asked about."
                    if how["source"] == "request" and set(years) == set(how["asked"])
                    else "."))
        extra = [y for y in how["normal"] if y not in how["asked"]]
        if extra and how["source"] == "request":
            L.append(f"The design compares them with {_span(extra)}, its normal years.")
    whole = sorted({c.get("id") for r in runs for c in (r.get("changes") or [])
                    if c.get("id") and not (c.get("params") or {}).get("years")
                    and not any(isinstance(a, dict) and "years" in a
                                for a in (menu.get(c.get("id")) or {}).get("alters") or [])})
    if whole and how["left_out"]:
        L.append(f"The change {_join([f'`{w}`' for w in whole])} is in effect in every "
                 f"simulated year, {_span(sim)}, and not only in the years reported. A "
                 f"reported year carries its effect from {min(sim)} on.")
    nh = (((preview or {}).get("check") or {}).get("cost") or {}).get("node_hours")
    hp = facts.get("hours_per_model_year")
    n_exec = len(runs) + (0 if ctl.get("reuse") else 1)
    if nh is None and hp and sim:
        nh = round(n_exec * len(sim) * hp, 1)
    if nh is not None:
        L.append(f"The model server estimates the cost at about {nh} node-hours.")
    diff = differences(brief.get("user_request") or "", runs, menu)
    if diff:
        L += ["", "**Different from what you asked.** " + " ".join(diff)]
    return L + [""]


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

    WHAT THE RUN DOES NOT CONTAIN IS SAID ONCE, under "What the study cannot
    tell you" (_cannot_lines), where the Planner's caveat says it too."""
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


def _same(text: Any) -> str:
    return " ".join(str(text or "").split()).rstrip(".").lower()


def _cannot_lines(de, said: Sequence[str] = (), not_in_it: Sequence[str] = ()) -> List[str]:
    """The design's caveats, each whole. A caveat that is word for word a
    sentence already printed under "Different from what you asked" is not
    printed a second time: the Planner copies the server's sentence, and the
    page gives it once, in the more prominent place."""
    caveats = [str(c).strip() for c in de.get("caveats") or [] if str(c).strip()]
    # WHAT THE CONTROL RUN DOES NOT CONTAIN, ONCE (2026-11-12). The page said
    # it under "The control run" and again here, in the Planner's caveat. It
    # is said here alone: in the Planner's words where a caveat names the
    # thing, and in the server's words where none does, so it is never lost.
    missing = []
    for item in not_in_it or []:
        head = str(item).split(":")[0].strip().lower()
        if head and not any(head in c.lower() for c in caveats):
            missing.append(f"The control run does not contain {str(item).strip().rstrip('.')}.")
    if not caveats and not missing:
        return ["The design records no caveat.", ""]
    told = {_same(x) for x in said}
    kept = [c for c in caveats if _same(c) not in told]
    L = [f"- {_sentence(_control_run(c))}" for c in kept] + [f"- {m}" for m in missing]
    if len(kept) < len(caveats):
        L.append("- The difference from your question is given under \"What will run\".")
    return L + [""]


def _noted(finding: Any) -> str:
    """A finding of the model server's check, whole: where it is, then what
    the server says of it."""
    if isinstance(finding, dict) and finding.get("where"):
        return (f"- The model server has a note on {finding['where']}. "
                f"{_sentence(_control_run(finding.get('what')))}")
    what = finding.get("what") if isinstance(finding, dict) else finding
    return f"- The model server has a note. {_sentence(_control_run(what))}"


def _weigh_lines(brief, strategy, preview, analysis_preview, run_dir) -> List[str]:
    from agents import analysis_preview as AV
    from agents import design_review as DR
    L: List[str] = []
    fz = strategy.get("feasibility") or {}
    verdict = str(fz.get("verdict") or "").strip().lower()
    if verdict == "full":
        L.append("- The Planner judges that this design can answer the question in full.")
    elif verdict == "partial":
        L.append(f"- The Planner judges that this design answers the question only in part. "
                 f"{_sentence(fz.get('why'))}")
    elif verdict == "infeasible":
        L.append(f"- The Planner judges that this design cannot answer the question. "
                 f"{_sentence(fz.get('why'))}")
    else:
        L.append("- The Planner gave no verdict on whether this design can answer the "
                 "question.")
    for need in strategy.get("requires") or []:
        L.append(f"- The question would also need this, which is not available: "
                 f"{_sentence(need)}")
    if preview:
        for i, r in enumerate(preview.get("rounds") or [], 1):
            L.append(f"- The model server refused {'an earlier' if i == 1 else 'another'} "
                     f"version of this design, and the Planner changed it before this page:")
            L += [f"  - {_sentence(DR._finding(f, 100000))}" for f in r.get("findings") or []]
        if preview.get("refused"):
            L.append("- The model server refuses this design as it stands:")
            L += [f"  - {_sentence(DR._finding(f, 100000))}" for f in preview["refused"]]
        elif preview.get("error"):
            L.append(f"- The design could not be built before this page. "
                     f"{_sentence(preview['error'])}")
        else:
            L.append("- The model server checked the design"
                     + (" and built every run without submitting anything."
                        if preview.get("runs") else "."))
            for u in ((preview.get("check") or {}).get("unusual") or []):
                L.append(_noted(u))
            ic = preview.get("inputs_check") or {}
            if ic and not ic.get("pass"):
                L.append("- The input files FAILED their check: "
                         + _sentence("; ".join(map(str, ic.get("reasons") or []))))
            for p in preview.get("problems") or []:
                L.append(f"- Not shown in the details: {_sentence(p)}")
    L += AV.weigh_lines(analysis_preview, run_dir, DR.design_fingerprint(strategy))
    rs = brief.get("run_settings") or {}
    for item in list(brief.get("assumptions") or []) + list(rs.get("conflicts") or []):
        L.append(f"- Decided without you: {_sentence(item)}")
    for item in brief.get("known_data_gaps") or []:
        L.append(f"- Noted by Reception: {_sentence(item)}")
    return L + [""]


def _decision_lines(de, strategy, preview, by_hand: bool, analysis_preview=None) -> List[str]:
    from agents import analysis_handoff as AH
    from core import approved_spending as SP
    runs = de.get("runs") or []
    ctl = de.get("control") or {}
    executes = bool(runs) or not ctl.get("reuse")
    fz = strategy.get("feasibility") or {}
    blocked = bool(preview and (preview.get("refused") or preview.get("error"))) or \
        str(fz.get("verdict") or "").strip().lower() == "infeasible"
    no_plan = AH.unapprovable(analysis_preview)
    # WHAT A YES APPROVES (stage 4): the plan too, where the page shows one
    # and promises it; and the computing it allows, where the server gave an
    # estimate, as a limit checked before any job (core/approved_spending.py)
    covers_plan = AH.bound(analysis_preview)
    nh = SP.estimate(preview)
    limit = (f"It allows the {nh:g} node-hours estimated above, and no more. "
             if executes and nh is not None else "")
    if blocked:
        L = ["This design cannot run as it stands. Type a change to have it made again and "
             "checked again, or stop. Nothing has been submitted."]
    elif no_plan:
        L = [f"This design cannot be approved as it stands, since {no_plan}. Type a change "
             f"to have it planned again, or stop. Nothing has been submitted."]
    elif executes:
        what = "the run" if len(runs) == 1 else "the runs"
        L = [f"A yes submits {what}. "
             + (f"It approves {what}, the years above and the analysis plan shown under "
                f"\"How the answer will be read\". " if covers_plan else
                f"It approves {what} and the years above. It does not approve an analysis "
                f"plan, since the Analyzer writes its own plan afterwards. ")
             + limit + NO_COMPUTE]
    else:
        L = ["A yes starts the analysis of the control run's existing output. No model run "
             "is made, and the analysis itself uses paid model calls. "
             + ("It approves the years above and the analysis plan shown under \"How the "
                "answer will be read\". " if covers_plan else
                "It does not approve an analysis plan, since the Analyzer writes its own "
                "plan afterwards. ") + NO_COMPUTE]
    L += ["", f"The design in full, with the analysis plan when one was written, is in "
              f"`{DETAILS_NAME}`."
              + (f" The steps to make the runs by hand are in `{BY_HAND_NAME}`."
                 if by_hand else "")]
    return L + [""]


def render(reception, strategy, run_name="this run", preview=None, run_dir=None,
           analysis_preview=None, by_hand: bool = True,
           control_figures: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The short design page of a whole-domain design, or None for any other
    design, which keeps the page it had. `control_figures`: the control's
    figures as design_review.write_review drew them ({"record", "drawn",
    "years"}), or None for a page without them."""
    from agents import analysis_preview as AV
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
    L = [f"# Design review: {run_name}", ""]
    L += _question_lines(question) + [""]
    L += ["## What will run", ""]
    L += _runs_lines(dict(brief, user_request=question), strategy, de, menu, row, preview,
                     run_dir)
    L += ["## The control run", ""] + _control_run_lines(de, row, control_figures)
    L += ["## How the answer will be read", ""]
    L += AV.short_lines(analysis_preview, run_dir, DR.design_fingerprint(strategy),
                        de, row)
    L += ["## What the study cannot tell you", ""] + _cannot_lines(
        de, differences(question, [r for r in de.get("runs") or [] if isinstance(r, dict)],
                        menu), row.get("not_in_it_words") or [])
    L += ["## Points to weigh", ""]
    L += _weigh_lines(brief, strategy, preview, analysis_preview, run_dir)
    L += ["## Your decision", ""] + _decision_lines(de, strategy, preview, by_hand,
                                                    analysis_preview)
    L += ["---", DR._PENDING, ""]
    return "\n".join(L)


def words(text: str) -> int:
    """The words of a page: tokens that hold a letter or a digit."""
    return sum(1 for t in str(text).split() if re.search(r"[A-Za-z0-9]", t))
