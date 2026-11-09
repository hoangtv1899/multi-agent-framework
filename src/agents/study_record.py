#!/usr/bin/env python3
"""
The study, from question to answer, on one page: STUDY_RECORD.md.
src/agents/study_record.py

    python src/agents/study_record.py RUN_DIR [--out PATH]

A STUDY RECORD FOR EACH RUN, DRAWN BY CODE FROM ITS FOLDER (2026-11-03).
A finished study leaves some forty to a hundred files behind, each the
authority on one step, and nothing that says in order what happened: what
was asked and what Reception asked back, what the design page said and what
was decided on it, what the model server's check found, which jobs ran and
what each one's check said, what the numbers hold, and what the Analyzer
kept and struck. This page is that account, with a line for every file in
the folder saying what it is.

A RENDERER, NOT A REASONER, like the design page. Every line is quoted or
counted from a file in the run folder; nothing is read from anywhere else
(not the model's output, not the server, not a language model), so the page
can be drawn again at any time from the folder alone and says the same. A
record that is missing or cannot be read is named as such on the page; it
never stops the rest of the page being drawn.

Nothing reads this file. The records it quotes stay the authority.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

RECORD_NAME = "STUDY_RECORD.md"
# The notice tools/notify_study.py writes at the end of job B (its REPORT_NAME).
NOTICE_NAME = "STUDY_REPORT.txt"
# How the coordinator appends a change asked for at the design review to the
# request (workflow.py); the same mark design_review.py splits on.
REVISION_MARK = "\n\nRevision requested at the design review: "
PREVIEW_FILE = "01_inputs/design_preview.json"
DOMAIN_RECORD = "domain_case"
# The report stage's sacct lines in STATUS.log (domain_case.write_stage_script):
# one per job it found, and one with no name for a job the scheduler has no
# account of (cancelled before it started).
_REPORT_LINE = re.compile(
    r"\bREPORT (?:(\S+) )?job=(\S+) state=(\S+)(?: elapsed=(\S+) exit=(\S+))?")


# ─────────────────────────────────────────────────────────────────────
# READING THE FOLDER
# ─────────────────────────────────────────────────────────────────────

class _Unreadable(Exception):
    """A record that exists and could not be parsed."""


def _read_json(path):
    """A JSON record; None when it is absent, _Unreadable when it is broken."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise _Unreadable(f"{p.name} could not be read ({type(e).__name__}: {e})")


def _read_text(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return p.read_text(errors="replace")
    except OSError:
        return None


def _one_line(text, n=400):
    """Text on one line, cut at a word when longer than n."""
    text = " ".join(str(text if text is not None else "").split())
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


def _num(x):
    """A number as a reader wants it: whole above 100, else 3 significant figures."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.3g}"


def _years(ys):
    return ", ".join(str(y) for y in ys) if ys else "none"


def _params(params):
    if not params:
        return ""
    return " (" + ", ".join(f"{k} {json.dumps(v) if isinstance(v, (list, dict)) else v}"
                            for k, v in params.items()) + ")"


def _change_words(change):
    if isinstance(change, dict):
        return f"{change.get('id')}{_params(change.get('params'))}"
    return str(change)


def _answer_paragraph(md):
    """The first paragraph under a report's '## The answer', its bold marks
    taken off; None when the report has no such section."""
    if not md:
        return None
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "## the answer":
            para = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("#"):
                    break
                if not nxt.strip():
                    if para:
                        break
                    continue
                para.append(nxt.strip())
            text = " ".join(para).strip()
            if text.startswith("**") and text.endswith("**"):
                text = text[2:-2].strip()
            return text or None
    return None


def _link(target, page_dir):
    """A markdown link to a file in the run folder, relative to the page."""
    rel = os.path.relpath(str(target), str(page_dir))
    return f"[{Path(target).name}]({Path(rel).as_posix()})"


# ─────────────────────────────────────────────────────────────────────
# THE QUESTION AND RECEPTION'S EXCHANGE
# ─────────────────────────────────────────────────────────────────────

def _question_lines(rd):
    L = ["## The question"]
    try:
        rec = _read_json(rd / "reception.json")
    except _Unreadable as e:
        return L + [f"reception.json is in the folder but {e}.", ""]
    if rec is None:
        return L + ["There is no reception.json in this folder, so the question and "
                    "Reception's exchange are not on record here.", ""]
    brief = rec.get("brief") or {}
    request = rec.get("user_request") or brief.get("user_request") or ""
    first, *revisions = str(request).split(REVISION_MARK)
    L += [f"> {_one_line(first, 2000)}" if first.strip() else
          "reception.json records no question.", ""]
    if revisions:
        L.append(f"Changes asked for at the design review ({len(revisions)}):")
        L += [f"{i}. {_one_line(r, 1000)}" for i, r in enumerate(revisions, 1)]
        L.append("")

    L.append("### Reception's exchange with you")
    route = rec.get("route") or {}
    pairs, seen = [], set()
    for src in ((rec.get("revision_of") or {}).get("user_answers"),
                rec.get("user_answers"), rec.get("exchange")):
        for qa in src or []:
            if not isinstance(qa, dict):
                continue
            key = (str(qa.get("question")), str(qa.get("answer")))
            if key not in seen:
                seen.add(key)
                pairs.append(qa)
    if pairs:
        L.append(f"Reception asked {len(pairs)} question(s), and you answered:")
        for i, qa in enumerate(pairs, 1):
            L.append(f"{i}. **Asked:** {_one_line(qa.get('question'), 800)}")
            L.append(f"   **You answered:** {_one_line(qa.get('answer'), 800) or '(no answer)'}")
    open_qs = [q for q in route.get("questions") or [] if str(q).strip()]
    if open_qs:
        L.append(f"Reception stopped to ask {len(open_qs)} question(s):")
        L += [f"- {_one_line(q, 800)}" for q in open_qs]
    if not pairs and not open_qs:
        # WHAT THE RECORD KEEPS, said plainly (2026-11-03): reception.json
        # drops the tool trace, so a question asked and answered inside one
        # reception round is kept only when a later revision carried the
        # answer forward. "No question" would overstate what is known.
        L.append("reception.json records no question asked of you. It keeps a question "
                 "only when Reception stopped to ask it, or when a change at the design "
                 "review carried your answer forward.")
    if rec.get("json_asked_again"):
        L.append("- Reception's last reply was not readable and was asked for once more.")
    action = route.get("action")
    bits = []
    if action:
        bits.append(f"routed as **{action}**")
    if brief.get("design_archetype"):
        bits.append(f"read as a **{str(brief['design_archetype']).replace('_', '-')}** study")
    if brief.get("model"):
        bits.append(f"for **{str(brief['model']).upper()}**")
    if rec.get("rounds") is not None:
        bits.append(f"in {rec['rounds']} round(s) of Reception's work")
    if bits:
        L.append("- The request was " + ", ".join(bits) + ".")
    de = brief.get("domain_experiment") or {}
    if de:
        ctl = de.get("control") or {}
        runs = [r.get("name") for r in de.get("runs") or [] if isinstance(r, dict)]
        L.append(f"- The brief: control **{ctl.get('name')}**"
                 f"{' (reused)' if ctl.get('reuse') else ''}; "
                 + (f"runs {', '.join(f'**{r}**' for r in runs)}" if runs else "no run besides it")
                 + f"; years {_years(de.get('years'))}.")
        for u in de.get("unresolved") or []:
            L.append(f"- left unresolved in the brief: {_one_line(u, 300)}")
    return L + [""]


# ─────────────────────────────────────────────────────────────────────
# THE DESIGN PAGE AND THE SERVER'S CHECK
# ─────────────────────────────────────────────────────────────────────

def _design_lines(rd, page_dir):
    L = ["## The design and the decision on it"]
    page = rd / "DESIGN_REVIEW.md"
    text = _read_text(page)
    if text is None:
        L.append("There is no DESIGN_REVIEW.md in this folder: no design page was drawn.")
    else:
        dec = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("**Decision:**")]
        L.append(f"The design page: {_link(page, page_dir)}.")
        L.append(f"- {dec[-1]}" if dec else "- The page records no decision line.")
    corrected = rd / "PLAN_CORRECTED.md"
    ctext = _read_text(corrected)
    if ctext is not None:
        title = next((ln.lstrip("# ").strip() for ln in ctext.splitlines() if ln.strip()), "")
        L.append(f"- The plan was changed after the decision: {_link(corrected, page_dir)}"
                 + (f" ({_one_line(title, 200)})" if title else "") + ".")
    if (rd / "strategy.as_accepted.json").is_file():
        L.append("- The design as accepted is kept as `strategy.as_accepted.json`; "
                 "`strategy.json` holds the design as run.")
    return L + [""]


def _finding(f, n=400):
    if isinstance(f, dict):
        return f"{f.get('where', '?')}: {_one_line(f.get('what'), n)}"
    return _one_line(f, n)


def _is_experiment(rd):
    """WHO GETS THE CHECK BEFORE THE PAGE (2026-11-03): the server checks and
    builds a design before its page only for a whole-domain experiment
    (preview_design returns None for any other plan). So a missing preview
    means "made before the check existed" only for such a study: one whose
    design or brief names a domain_experiment, or whose runs were built."""
    if (rd / "01_inputs" / "domain_runs.json").is_file():
        return True
    for name, path in (("strategy.json", ()), ("reception.json", ("brief",))):
        try:
            rec = _read_json(rd / name) or {}
        except _Unreadable:
            continue
        for key in path:
            rec = (rec.get(key) if isinstance(rec, dict) else None) or {}
        if isinstance(rec, dict) and rec.get("domain_experiment"):
            return True
    return False


def _check_lines(rd):
    L = ["## The model server's check"]
    broken = None
    try:
        pv = _read_json(rd / PREVIEW_FILE)
    except _Unreadable as e:
        pv, broken = None, e
    if broken:
        L.append(f"`{PREVIEW_FILE}` is in the folder but {broken}.")
    elif pv is not None:
        L.append(f"From `{PREVIEW_FILE}`, written before the design page"
                 + (f" ({pv['made']})" if pv.get("made") else "") + ":")
        for i, r in enumerate(pv.get("rounds") or [], 1):
            L.append(f"- the server refused design {i} and the Planner changed it "
                     f"(the refused design is kept as `{r.get('design')}`):")
            L += [f"  - {_finding(f, 300)}" for f in r.get("findings") or []]
        if pv.get("refused"):
            L.append("- **Refused.** The server would not build the design:")
            L += [f"  - {_finding(f)}" for f in pv["refused"]]
        elif pv.get("error"):
            L.append(f"- **Not built.** {_one_line(pv['error'], 500)}")
        else:
            n = len(pv.get("runs") or [])
            L.append(f"- **Passed.** The server checked the design and built "
                     f"{n} run(s) without submitting anything.")
        chk = pv.get("check") or {}
        for u in chk.get("unusual") or []:
            L.append(f"- unusual: {_finding(u, 300)}")
        ic = pv.get("inputs_check") or {}
        if ic:
            L.append("- the input set (surface data, grid file, starting state): "
                     + ("passed its check" if ic.get("pass") else
                        "FAILED its check: " + "; ".join(map(str, ic.get("reasons") or []))))
        nh = (chk.get("cost") or {}).get("node_hours")
        if nh is not None:
            L.append(f"- cost: about {nh} node-hours")
        for p in pv.get("problems") or []:
            L.append(f"- problem: {_one_line(p, 300)}")
    elif _is_experiment(rd):
        L.append(f"This run predates the server's check before the design page "
                 f"(there is no `{PREVIEW_FILE}`).")
    else:
        L.append("The server checks a design before the design page only for a "
                 "whole-domain experiment (a control and the runs that differ from it); "
                 "this study is not one.")
    # THE CHECK AT BUILD TIME, which every whole-domain run has: the server's
    # answer when the runs were built after the decision.
    try:
        dr = _read_json(rd / "01_inputs" / "domain_runs.json")
    except _Unreadable as e:
        dr, L = None, L + [f"- 01_inputs/domain_runs.json is in the folder but {e}."]
    if dr is not None:
        cost = dr.get("cost") or {}
        L.append(f"When the runs were built (`01_inputs/domain_runs.json`): "
                 + ("the server accepted the experiment" if dr.get("ok") else
                    "the server did NOT accept the experiment")
                 + ("" if not cost else "; nothing to execute" if cost.get("runs_executed") == 0
                    else f"; {cost.get('runs_executed')} run(s) to execute, "
                         f"{cost.get('model_years')} model year(s), about "
                         f"{cost.get('node_hours')} node-hours")
                 + (f" ({_one_line(cost.get('note'), 200)})" if cost.get("note") else "")
                 + ".")
        for u in dr.get("unusual") or []:
            L.append(f"- unusual: {_finding(u, 300)}")
        for f in dr.get("refused") or dr.get("findings") or []:
            L.append(f"- finding: {_finding(f, 300)}")
    try:
        ic = _read_json(rd / DOMAIN_RECORD / "inputs_check.json")
    except _Unreadable as e:
        ic, L = None, L + [f"- {DOMAIN_RECORD}/inputs_check.json is in the folder but {e}."]
    if ic is not None and pv is None:
        L.append("- the input set (surface data, grid file, starting state): "
                 + ("passed its check" if ic.get("pass") else
                    "FAILED its check: " + "; ".join(map(str, ic.get("reasons") or [])))
                 + f" (`{DOMAIN_RECORD}/inputs_check.json`)")
    return L + [""]


# ─────────────────────────────────────────────────────────────────────
# THE RUNS, THEIR JOBS AND EACH JOB'S CHECK
# ─────────────────────────────────────────────────────────────────────

def report_lines(status_text):
    """The report stage's account of each job, from STATUS.log: {job id:
    {name, state, elapsed, exit}}. A job reported twice keeps its last line."""
    out = {}
    for line in (status_text or "").splitlines():
        m = _REPORT_LINE.search(line)
        if m:
            name, job, state, el, ex = m.groups()
            out[job] = {"name": name, "state": state, "elapsed": el, "exit": ex}
    return out


def _seconds(elapsed):
    try:
        d, _, hms = str(elapsed).rpartition("-")
        h, mi, se = (int(x) for x in hms.split(":"))
        return int(d or 0) * 86400 + h * 3600 + mi * 60 + se
    except (TypeError, ValueError):
        return None


def _hms(secs):
    h, rem = divmod(int(secs), 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}"


def _check_records(record_dir, job_id):
    """{check name: record} for every checks/<name>_<job id>.json; a broken
    one maps to its error text."""
    out = {}
    for p in sorted((record_dir / "checks").glob(f"*_{job_id}.json")):
        name = p.name[: -len(f"_{job_id}.json")]
        try:
            out[name] = _read_json(p)
        except _Unreadable as e:
            out[name] = str(e)
    return out


def _verdict(rec):
    if isinstance(rec, str):
        return "unreadable"
    p = (rec or {}).get("pass")
    return "passed" if p is True else "FAILED" if p is False else "no verdict"


def _check_notes(rec):
    """The reasons a check gives, and what it recorded beside its verdict."""
    if isinstance(rec, str):
        return [rec]
    notes = [_one_line(r, 400) for r in rec.get("reasons") or []]
    for y, v in sorted((rec.get("years") or {}).items()):
        if not isinstance(v, dict):
            continue
        ex = [e for e in v.get("expected") or [] if isinstance(e, dict)]
        if ex:
            held = sum(1 for e in ex if e.get("pass"))
            notes.append(f"the change's expected effects on {y}'s output: "
                         f"{held} of {len(ex)} held")
        ref = v.get("reference") or {}
        if ref and ref.get("gated") is False and ref.get("pass") is False:
            notes.append(f"{y} compared with the control (not counted in the "
                         f"verdict): " + "; ".join(_one_line(r, 200)
                                                  for r in ref.get("reasons") or []))
    return notes


_YEAR = re.compile(r"y\d{4}")


def _stage_rank(stage):
    if stage in ("case", "smoke", "report"):
        return {"case": 0, "smoke": 1, "report": 99999}[stage]
    return int(stage[1:]) + 10 if _YEAR.fullmatch(stage) else 50000


def _stage_pass(record_dir, stage, job_id):
    """A stage's own check record, checks/<stage>_<job id>.json: True or
    False by its "pass", False when it cannot be read, None when absent."""
    try:
        rec = _read_json(record_dir / "checks" / f"{stage}_{job_id}.json")
    except _Unreadable:
        return False
    return None if rec is None else bool((rec or {}).get("pass"))


def _segments(chain):
    """chain.json's armings; a record from before re-arming (no "segments")
    is one arming, as the server reads it."""
    if not isinstance(chain, dict):
        return []
    return chain["segments"] or [] if "segments" in chain else [chain]


def chain_verdict(record_dir, chain):
    """Where a chain stands, by THE SERVER'S RULE (2026-11-03).

    domain_case.chain_records decides this for experiment.json and
    analysis.json, and the page must not say otherwise: each stage is read
    from its LATEST job (a re-armed chain's later arming takes over the
    stages it ran again), a year passes when that job's own check record
    passed, and the chain stops at the first case, smoke or year stage
    without a passing record, except that a case stage with no record counts
    as passed when a later stage left one (it could only run after the case).
    The rule is copied here, not imported, so the page is drawn from the
    folder by framework code alone; a test holds the copy to the server's.
    Returns {stages: [{stage, job_id, segment, check_pass}], years_planned,
    years_passed, first_unpassed}."""
    order, latest = [], {}
    for k, seg in enumerate(_segments(chain)):
        for j in (seg or {}).get("jobs") or []:
            s = str(j.get("stage"))
            if s not in latest:
                order.append(s)
            latest[s] = dict(j, segment=k)
    order.sort(key=_stage_rank)
    rows = [{"stage": s, "job_id": str(latest[s].get("job_id")),
             "segment": latest[s]["segment"],
             "check_pass": _stage_pass(record_dir, s, latest[s].get("job_id"))}
            for s in order]
    later_ran = {r["stage"] for r in rows if r["check_pass"] is not None}
    planned, passed, first = [], [], None
    for r in rows:
        s = r["stage"]
        if _YEAR.fullmatch(s):
            planned.append(int(s[1:]))
            if r["check_pass"]:
                passed.append(int(s[1:]))
        ok = r["check_pass"] is True or (s == "case" and r["check_pass"] is None
                                         and bool(later_ran - {"case"}))
        gated = s in ("case", "smoke") or bool(_YEAR.fullmatch(s))
        if gated and not ok and first is None:
            first = s
    return {"stages": rows, "years_planned": planned, "years_passed": passed,
            "first_unpassed": first}


def _run_lines(rec_dir, heading, rel, sr=None, recorded=None):
    """One run: its changes, each arming of its chain as a table of jobs,
    what each job's check said, and where the chain stands. `rec_dir` is the
    run's record folder, `rel` its path in the run folder (for the page),
    `sr` the run as the strategy names it, `recorded` what experiment.json
    recorded for its chain."""
    L = [f"### {heading}"]
    if sr:
        L.append("- the control with " + "; ".join(_change_words(c) for c in sr.get("changes") or [])
                 if sr.get("changes") else "- no change from the control is recorded")
    try:
        chain = _read_json(rec_dir / "chain.json")
    except _Unreadable as e:
        return L + [f"- {rel}/chain.json is in the folder but {e}.", ""]
    if chain is None:
        return L + [f"- no chain was armed for it (there is no `{rel}/chain.json`).", ""]
    times = report_lines(_read_text(rec_dir / "STATUS.log"))
    total, notes, verdicts = 0, [], {}
    segments = _segments(chain)
    for si, seg in enumerate(segments, 1):
        seg = seg or {}
        if len(segments) > 1:
            L.append(f"- arming {si} of {len(segments)}, armed {seg.get('armed')}:")
        else:
            L.append(f"- its chain, armed {seg.get('armed')}:")
        L += ["", "| stage | job | check | state | time |", "|---|---|---|---|---|"]
        for job in seg.get("jobs") or []:
            stage, jid = job.get("stage"), str(job.get("job_id"))
            checks = _check_records(rec_dir, jid)
            t = times.get(jid) or {}
            if stage == "report":
                verdict = "none (it logs the job times)"
            elif checks:
                verdict = ", ".join(f"{n} {_verdict(c)}" for n, c in checks.items())
            else:
                verdict = "no check record"
            verdicts[jid] = verdict
            for n, c in checks.items():
                who = f"{stage} (job {jid})" + (f", its {n} check" if n != stage else "")
                notes += [f"{who}: {x}" for x in _check_notes(c)]
            s = _seconds(t.get("elapsed"))
            if s is not None and stage != "report":
                total += s
            state = "-" if stage == "report" else (t.get("state") or "not logged")
            L.append(f"| {stage} | {jid} | {verdict} | {state} | "
                     f"{t.get('elapsed') or '-'} |")
        L.append("")
    if not times:
        L.append("- the report stage has logged no job times yet (no REPORT line in "
                 f"`{rel}/STATUS.log`).")
    elif total:
        L.append(f"- model jobs' time as the report stage logged it: {_hms(total)} in all.")
    v = chain_verdict(rec_dir, chain)
    rows = {r["stage"]: r for r in v["stages"]}
    rearmed = sorted((r for r in v["stages"] if r["segment"] > 0),
                     key=lambda r: _stage_rank(r["stage"]))
    if rearmed:
        L.append("- re-armed: each stage is read from its latest job, so "
                 + ", ".join(f"{r['stage']} from job {r['job_id']} (arming {r['segment'] + 1})"
                             for r in rearmed) + ".")
    L.append(f"- years planned: {_years(v['years_planned'])}; years whose check passed: "
             f"{_years(v['years_passed'])}.")
    first = v["first_unpassed"]
    if first:
        r = rows[first]
        state = (times.get(r["job_id"]) or {}).get("state")
        L.append(f"- **The chain stopped at stage {first}** (job {r['job_id']}): its check: "
                 f"{verdicts.get(r['job_id'], 'no check record')}"
                 + (f"; the scheduler's state: {state}" if state else "") + ".")
    else:
        L.append("- every stage's check passed"
                 + (" (a re-armed stage by its latest job)" if rearmed else "") + ".")
        case = rows.get("case")
        if case and case["check_pass"] is None:
            L.append("- the case stage left no check record; it counts as passed because "
                     "a later stage ran after it.")
    # THE FRAMEWORK'S OWN RECORD, quoted where it says otherwise (2026-11-03):
    # experiment.json was written by the same rule when job B finished, so a
    # difference means a check record changed after it, and the page says so.
    if isinstance(recorded, dict) and ("years_passed" in recorded or "first_unpassed" in recorded):
        got = (list(recorded.get("years_passed") or []), recorded.get("first_unpassed"))
        if got != (v["years_passed"], first):
            L.append(f"- experiment.json records otherwise: years whose check passed "
                     f"{_years(got[0])}; first stage without a passing check: "
                     f"{got[1] or 'none'}.")
    if notes:
        L.append("- what the checks said:")
        L += [f"  - {n}" for n in notes]
    return L + [""]


def _brief_runs(rd):
    """The runs the brief in reception.json names, [] when it names none or
    cannot be read."""
    try:
        rec = _read_json(rd / "reception.json") or {}
    except _Unreadable:
        return []
    de = ((rec.get("brief") or {}).get("domain_experiment")) or {}
    return [r.get("name") for r in de.get("runs") or [] if isinstance(r, dict) and r.get("name")]


def _runs_lines(rd):
    L = ["## The runs"]
    try:
        dr = _read_json(rd / "01_inputs" / "domain_runs.json") or {}
    except _Unreadable as e:
        dr, L = {}, L + [f"01_inputs/domain_runs.json is in the folder but {e}."]
    ctl = dr.get("control") or {}
    if ctl:
        L.append(f"- the control: **{ctl.get('name')}**, "
                 + ("reused, so its existing output was read and no job was run for it"
                    if ctl.get("reuse") else "run by its own chain"))
    try:
        strategy = _read_json(rd / "strategy.json") or {}
    except _Unreadable as e:
        strategy, L = {}, L + [f"- strategy.json is in the folder but {e}."]
    try:
        recorded = ((_read_json(rd / "experiment.json") or {}).get("chain") or {}).get("runs") or {}
    except (_Unreadable, AttributeError):
        recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    de = strategy.get("domain_experiment") or {}
    strategy_runs = {r.get("name"): r for r in de.get("runs") or []
                     if isinstance(r, dict) and r.get("name")}
    rec = rd / DOMAIN_RECORD
    # ONE WHOLE-DOMAIN CASE, the layout before experiments (2026-11-03): its
    # record is domain_case/ itself (chain.json, checks/, logs/), so that
    # folder is the one run, and its subfolders are not runs.
    if (rec / "chain.json").is_file():
        case = strategy.get("domain_case") or {}
        name = case.get("case") if isinstance(case, dict) else None
        return L + ["", *_run_lines(rec, "The whole-domain case" + (f" {name}" if name else ""),
                                    DOMAIN_RECORD)]
    # RUN NAMES FROM THE RECORDS (2026-11-03): the server's build, the design,
    # and a record folder only when it holds a chain, so checks/ or logs/ is
    # never taken for a run, and a run the server refused is still named.
    names = [r.get("name") for r in dr.get("runs") or [] if isinstance(r, dict) and r.get("name")]
    names += [n for n in strategy_runs if n not in names]
    armed = sorted(p.name for p in rec.iterdir()
                   if p.is_dir() and (p / "chain.json").is_file()) if rec.is_dir() else []
    if "control" in armed and "control" not in names:
        names.insert(0, "control")
    names += [n for n in armed if n not in names]
    if not names:
        if not dr and not de:
            # NOT A WHOLE-DOMAIN STUDY (2026-11-03): every study ends through
            # the same stage, so a sampled one gets a page too; its cases are
            # counted from RUN_SUMMARY.json rather than left unsaid.
            L.append("No whole-domain run is on record in this folder (no "
                     "01_inputs/domain_runs.json and no domain_case folder).")
            try:
                rs = _read_json(rd / "RUN_SUMMARY.json") or {}
            except _Unreadable:
                rs = {}
            if rs.get("experiments_total") is not None:
                L.append(f"- RUN_SUMMARY.json: {rs.get('experiments_success')} of "
                         f"{rs.get('experiments_total')} case(s) succeeded"
                         + (f", {rs['experiments_failed']} failed"
                            if rs.get("experiments_failed") else "") + ".")
            return L + [""]
        brief = _brief_runs(rd)
        if brief:
            return L + [f"- The brief named run(s) {', '.join(f'**{b}**' for b in brief)}, "
                        f"but the design as run names none and no chain is on record.", ""]
        return L + ["- No run differs from the control: the study read the control's "
                    "output alone, and no job was submitted.", ""]
    L.append("")
    for n in names:
        L += _run_lines(rec / n, f"Run {n}", f"{DOMAIN_RECORD}/{n}",
                        strategy_runs.get(n), recorded.get(n))
    return L


# ─────────────────────────────────────────────────────────────────────
# THE NUMBERS
# ─────────────────────────────────────────────────────────────────────

_HOW = {"sum": "total over the year", "mean": "mean over the year"}


def _numbers_lines(rd, page_dir):
    L = ["## The numbers"]
    md_path = rd / "04_analysis" / "DOMAIN_REPORT.md"
    md = _read_text(md_path)
    try:
        drj = _read_json(rd / "04_analysis" / "domain_report.json") or {}
    except _Unreadable:
        drj = {}
    head = _answer_paragraph(md) or drj.get("headline")
    if md is None and not drj:
        L.append("No numbers report was written (there is no 04_analysis/DOMAIN_REPORT.md).")
    else:
        L.append(f"The numbers report, {_link(md_path, page_dir) if md is not None else 'domain_report.json'}"
                 + (f" (status **{drj['status']}**)" if drj.get("status") else "")
                 + (", says:" if head else ", states no answer."))
        if head:
            L.append(f"> {_one_line(head, 1200)}")
    L.append("")
    L += _numbers_check_lines(rd)
    try:
        reg = _read_json(rd / "03_results" / "regional.json")
    except _Unreadable as e:
        return L + [f"03_results/regional.json is in the folder but {e}.", ""]
    if reg is None:
        return L + ["There is no 03_results/regional.json: no regional numbers were computed.", ""]
    L.append("What `03_results/regional.json` holds:")
    L.append(f"- years: {_years(reg.get('years'))}")
    runs = reg.get("runs") or {}
    for name, r in runs.items():
        r = r or {}
        L.append(f"- run **{name}**, read from `{r.get('run_dir')}`")
        for f, why in (r.get("missing") or {}).items():
            L.append(f"  - missing {f}: {_one_line(why, 200)}")
        for p in r.get("problems") or []:
            L.append(f"  - problem: {_one_line(p, 300)}")
    fields = {}
    for r in runs.values():
        for f, meta in ((r or {}).get("fields") or {}).items():
            fields.setdefault(f, meta or {})
    if fields:
        L.append(f"- fields ({len(fields)}): " + "; ".join(
            f"{f} ({m.get('units')}, {_HOW.get(m.get('how'), m.get('how'))})"
            for f, m in fields.items()))
    regions = reg.get("regions") or {}
    if regions:
        L += ["", f"Regions ({len(regions)}):", "",
              "| region | cells | area (km2) | rule |", "|---|---|---|---|"]
        for name, r in regions.items():
            r = r or {}
            L.append(f"| {name} | {_num(r.get('cells'))} | {_num(r.get('area_km2'))} | "
                     f"{_one_line(r.get('rule'), 200)} |")
        L.append("")
    diffs = reg.get("differences") or {}
    if diffs:
        L.append(f"- differences from the control, for: {', '.join(diffs)}")
    for run, cs in (reg.get("changed_share") or {}).items():
        share = (cs or {}).get("share") or {}
        L.append(f"- share of each region's cells that run {run}'s change reaches"
                 + (f" (its region: {cs.get('region')})" if cs.get("region") else "") + ": "
                 + "; ".join(f"{k} {_num(v)}" for k, v in share.items()))
    ev = reg.get("event") or {}
    if ev:
        L.append(f"- event years {_years(ev.get('target_years'))} against normal years "
                 f"{_years(ev.get('normal_years'))}")
        for p in ev.get("problems") or []:
            L.append(f"  - problem: {_one_line(p, 300)}")
    for key, ob in (reg.get("observed") or {}).items():
        ob = ob or {}
        L.append(f"- observed beside the numbers: {ob.get('what') or key} ({ob.get('units')}), "
                 f"{_one_line(ob.get('source'), 200)}")
    for p in reg.get("problems") or []:
        L.append(f"- problem: {_one_line(p, 300)}")
    return L + [""]


# THE CHECK'S VERDICT (2026-11-03). A passing numbers check left no trace on
# this page; it is quoted from its own record, with its age against the table
# it checked (a check older than the table checked an earlier one).
def _numbers_check_lines(rd):
    p = rd / "04_analysis" / "numbers_check.json"
    try:
        rec = _read_json(p)
    except _Unreadable as e:
        return [f"The numbers check: `04_analysis/numbers_check.json` is in the folder but {e}.", ""]
    if rec is None:
        return ["The numbers check: none on record (there is no "
                "`04_analysis/numbers_check.json`), so the regional numbers were not made a "
                "second time.", ""]
    w = (rec.get("largest_relative_difference") or {}).get("rel_diff")
    n = rec.get("n_compared")
    n = f"{n:,}" if isinstance(n, int) else str(n)
    if rec.get("pass"):
        line = (f"The numbers check, `04_analysis/numbers_check.json`: **passed**. {n} numbers "
                f"of `03_results/regional.json` were made again from the model output by an "
                f"independent reading; the largest relative difference is "
                + (f"{float(w):.3g}" if w is not None else "none") + ".")
        # what the check left (2026-11-03): a passing check is a full one only
        # when it lists no table key it did not check and no note
        left, notes = rec.get("not_checked") or [], rec.get("notes") or []
        if left:
            line += f" Not checked: the table's {', '.join(str(k) for k in left)}."
        if notes:
            line += " The check's notes: " + "; ".join(
                _one_line(x, 300).rstrip(".") for x in notes) + "."
    elif rec.get("error") and not rec.get("n_compared"):
        line = (f"The numbers check, `04_analysis/numbers_check.json`: **did not run to the "
                f"end**: {_one_line(rec['error'], 300)}.")
    else:
        line = (f"The numbers check, `04_analysis/numbers_check.json`: **did not pass**. "
                f"{rec.get('n_mismatched')} of {n} numbers of `03_results/regional.json` did "
                f"not reproduce when made again from the model output.")
    if rec.get("checked_at"):
        line += f" Checked {rec['checked_at']}."
    t = rd / "03_results" / "regional.json"
    try:
        if t.is_file() and p.stat().st_mtime < t.stat().st_mtime:
            line += (" The check is older than `03_results/regional.json`, so it checked an "
                     "earlier table.")
    except OSError:
        pass
    return [line, ""]


# ─────────────────────────────────────────────────────────────────────
# THE ANALYZER
# ─────────────────────────────────────────────────────────────────────

def _analyzer_lines(rd, page_dir):
    L = ["## The Analyzer"]
    try:
        state = _read_json(rd / "run_state.json") or {}
    except _Unreadable:
        state = {}
    st = ((state.get("stages") or {}).get("analyze")) or {}
    status = st.get("status")
    if status == "skipped":
        L.append(f"**Skipped.** {_one_line(st.get('reason'), 400) or 'run_state.json gives no reason.'}")
    elif st.get("analyzer"):
        L.append(f"The Analyzer did not run: {_one_line(st['analyzer'], 300)}.")
    elif status == "failed":
        L.append(f"**The analysis stage failed:** {_one_line(st.get('error'), 400)}"
                 + (f" (steps that failed: {', '.join(st['failed_steps'])})"
                    if st.get("failed_steps") else ""))
    elif status:
        L.append(f"The analysis stage: **{status}**" + (f", at {st['at']}" if st.get("at") else "") + ".")
    else:
        L.append("run_state.json records no analysis stage.")
    try:
        a = _read_json(rd / "04_analysis" / "analysis.json")
    except _Unreadable as e:
        return L + [f"04_analysis/analysis.json is in the folder but {e}.", ""]
    report = rd / "04_analysis" / "REPORT.md"
    text = _read_text(report)
    if a is None and text is None:
        return L + ["There is no Analyzer report (no 04_analysis/analysis.json or REPORT.md)."] \
            + [""] + _suite_lines(rd, None)
    a = a or {}
    # A WITHHELD REPORT IS SAID TO BE WITHHELD (the user's decision on G part
    # 2, 2026-11-08): question 1's page quoted its held-back headline under
    # "Its report" and listed its claims as kept by the audit, as if published
    withheld = a.get("withheld_because")
    head = _answer_paragraph(text) or a.get("headline")
    if withheld:
        L.append(f"The Analyzer's report, "
                 f"{_link(report, page_dir) if text is not None else 'REPORT.md'}, is "
                 f"**withheld**: {_one_line(withheld, 900)}")
        if a.get("headline"):
            L.append(f"The headline held back with it: \"{_one_line(a['headline'], 600)}\"")
    else:
        if text is not None:
            L.append(f"Its report: {_link(report, page_dir)}.")
        if head:
            L.append(f"> {_one_line(head, 1200)}")
    L.append("")
    if a:
        L.append(f"- status **{a.get('status')}**; verdict **{a.get('verdict')}**")
        claims = a.get("claims") or []
        struck = a.get("withheld") or []
        if withheld:
            L.append(f"- claims withheld with the report: {len(claims)}; struck by step 3's "
                     f"audit: {len(struck)}")
        else:
            L.append(f"- claims kept by the audit: {len(claims)}; struck: {len(struck)}")
        for i, c in enumerate(claims, 1):
            c = c or {}
            L.append(f"  {i}. {_one_line(c.get('claim'), 500)}"
                     + (" **(withheld)**" if withheld else ""))
            # EACH CLAIM'S EVIDENCE CLASS AND RECORDS (package F, the user's
            # decision on G part 2, 2026-11-08), with what the trace found
            e = c.get("evidence_trace")
            if isinstance(e, dict):
                L.append(f"     - evidence class: {e.get('class') or 'none'}; records: "
                         f"{', '.join(e.get('records') or []) or 'none'}"
                         + (f"; the evidence trace: fails ({_one_line('; '.join(e['problems']), 400)})"
                            if e.get("problems") else "; the evidence trace: passes"))
        for c in struck:
            c = c or {}
            L.append(f"  - struck ({c.get('struck_by')}): {_one_line(c.get('claim'), 300)} "
                     f"Why: {_one_line(c.get('struck_because'), 300)}")
        llm = ((a.get("cost") or {}).get("llm") or {}).get("total") or {}
        if llm:
            L.append(f"- model calls: {llm.get('calls')}"
                     + (f", {llm.get('prompt_tokens'):,} tokens sent and "
                        f"{llm.get('completion_tokens'):,} received"
                        if isinstance(llm.get("prompt_tokens"), int)
                        and isinstance(llm.get("completion_tokens"), int) else "")
                     + (f", {_num(llm.get('seconds'))} s" if llm.get("seconds") is not None else "")
                     + (f" ({', '.join(llm.get('models') or [])})" if llm.get("models") else ""))
        else:
            L.append("- model calls: not recorded in analysis.json")
    return L + [""] + _suite_lines(rd, a)


def _lookup_counts(rows):
    """How the lookups log's lines fall: answered (planned or not), support
    calls, refused by the plan's gate, refused by the data, failed, and how
    many came after the close of the planned phase."""
    n = {"answered and planned": 0, "answered, not planned": 0, "support calls": 0,
         "refused by the plan's gate": 0, "refused by the data": 0, "failed": 0}
    after = 0
    for r in rows:
        if r.get("support"):
            n["support calls"] += 1
        elif r.get("refused"):
            n["refused by the plan's gate" if str(r["refused"]).startswith("the gate:")
              else "refused by the data"] += 1
        elif r.get("failed"):
            n["failed"] += 1
        elif r.get("planned") is True:
            n["answered and planned"] += 1
        else:
            n["answered, not planned"] += 1
        after += r.get("after_close") is True
    return n, after


def _suite_lines(rd, a):
    """A study with a suite (package F): the recorded analysis plan and each
    version's review, the lookups log, the close of the planned analyses with
    each one's status, the screen's record, the numbers check of the claims'
    evidence and the claims' evidence classes. [] for a study without an
    analysis plan."""
    ad = rd / "04_analysis"
    try:
        plan = _read_json(ad / "analysis_plan.json")
    except _Unreadable as e:
        return ["### The analysis plan and its evidence", f"04_analysis/analysis_plan.json is in "
                f"the folder but {e}.", ""]
    if plan is None:
        return []
    L = ["### The analysis plan and its evidence"]
    sc = plan.get("screen") or {}
    L.append(f"- the plan, `04_analysis/analysis_plan.json` (step 2a, written "
             f"{plan.get('written_at')}): question type **{plan.get('question_type')}** "
             f"({_one_line(plan.get('question_type_from'), 200)}); "
             f"{len(plan.get('analyses') or [])} analyses, "
             f"{len(plan.get('expectations') or [])} expectations, "
             f"{len(plan.get('competing') or [])} competing explanations; the screen "
             f"{sc.get('when') or 'not declared'}"
             + (f", over {sc.get('region')} in {sc.get('year')}" if sc.get("region") else ""))
    for name, label in (("analysis_plan_review.v1.json", "the first version's review"),
                        ("analysis_plan_review.json", "its review")):
        try:
            rv = _read_json(ad / name)
        except _Unreadable as e:
            L.append(f"- {label}: `04_analysis/{name}` is in the folder but {e}.")
            continue
        if rv is None:
            if name == "analysis_plan_review.json":
                L.append("- its review: none on record")
            continue
        if name.endswith(".v1.json"):
            L.append("- the first version, `04_analysis/analysis_plan.v1.json`, did not pass "
                     "its review and was rewritten once, before any lookup")
        code = rv.get("code_problems")
        if code is None:
            code = [p[6:] for p in rv.get("problems") or [] if str(p).startswith("code: ")]
        model = rv.get("model_problems")
        if model is None:
            model = [p for p in rv.get("problems") or [] if not str(p).startswith("code: ")]
        verdict = {True: "follows from the question", False: "does not follow from the "
                   "question"}.get(rv.get("follows"), "no verdict on record")
        L.append(f"- {label}, `04_analysis/{name}`: **{'passed' if rv.get('passed') else 'did not pass'}**; "
                 f"the code check: {_one_line('; '.join(code), 500) or 'no problem'}; the "
                 f"model review: {verdict}"
                 + (f"; its problems: {_one_line('; '.join(model), 500)}" if model else ""))
    p = ad / "lookups.jsonl"
    if p.is_file():
        rows = []
        for ln in (_read_text(p) or "").splitlines():
            try:
                rows.append(json.loads(ln)) if ln.strip() else None
            except ValueError:
                rows.append({"failed": "an unreadable line"})
        n, after = _lookup_counts(rows)
        L.append(f"- the lookups, `04_analysis/lookups.jsonl`: {len(rows)} lines: "
                 + ", ".join(f"{v} {k}" for k, v in n.items() if v)
                 + f"; {after} after the close of the planned analyses")
    else:
        L.append("- the lookups: none (there is no `04_analysis/lookups.jsonl`)")
    try:
        ph = _read_json(ad / "planned_phase.json")
    except _Unreadable as e:
        ph, L = None, L + [f"- 04_analysis/planned_phase.json is in the folder but {e}."]
    if ph:
        from agents.analysis import plan_contract as PC
        c = ph.get("counts") or {}
        L.append(f"- the planned analyses, `04_analysis/planned_phase.json` (closed "
                 f"{ph.get('closed')}, after {ph.get('lookups_before')} lookup lines), each "
                 f"judged by its required outputs: "
                 + ", ".join(f"{c.get(s, 0)} {s}" for s in PC.STATUSES))
        L += ["", "| analysis | status | required | produced | not estimable | unmet | reason |",
              "|---|---|---|---|---|---|---|"]
        for r in ph.get("analyses") or []:
            L.append(f"| {r.get('id')} | {r.get('status')} | "
                     f"{'yes' if r.get('required') else 'no'} | "
                     f"{', '.join(r.get('produced') or []) or 'none'} | "
                     f"{', '.join(r.get('not_estimable') or []) or 'none'} | "
                     f"{', '.join(r.get('unmet') or []) or 'none'} | "
                     f"{_one_line(r.get('reason'), 300).replace('|', '/')} |")
        L.append("")
    elif plan.get("question_type"):
        L.append("- the planned analyses: no close on record (there is no "
                 "`04_analysis/planned_phase.json`)")
    acc = (a or {}).get("acceptance")
    if isinstance(acc, dict):
        if acc.get("ok"):
            L.append("- acceptance: every required analysis completed")
        else:
            bl = acc.get("blockers") or []
            L.append("- acceptance: **not met**: " + (
                "; ".join(f"{b.get('analysis')} {b.get('status')} ({_one_line(b.get('reason'), 200)})"
                          for b in bl) if bl else _one_line(acc.get("why"), 300)))
        for u in acc.get("untestable") or []:
            L.append(f"- untestable from this run: {u.get('expectation')} (analysis "
                     f"{u.get('analysis')}): " + "; ".join(
                         f"{o.get('output')} not estimable ({o.get('record')}: "
                         f"{_one_line(o.get('why'), 160)})" for o in u.get("outputs") or [])
                     + "; its task is complete")
    try:
        sr = _read_json(ad / "screen_record.json")
    except _Unreadable:
        sr = None
    if sr:
        scope = sr.get("scope") or {}
        L.append(f"- the screen, `04_analysis/screen_record.json`: {scope.get('region')} in "
                 f"{scope.get('year')}, {scope.get('rows')} rows, labelled "
                 f"\"{sr.get('label')}\"")
    try:
        en = _read_json(ad / "evidence_numbers_check.json")
    except _Unreadable:
        en = None
    if en:
        L.append(f"- the numbers the claims cite, `04_analysis/evidence_numbers_check.json`: "
                 f"{'passed' if en.get('pass') else 'did not pass'}, {en.get('n_compared')} "
                 f"numbers made again")
    classes = {}
    for c in (a or {}).get("claims") or []:
        e = (c or {}).get("evidence_trace")
        if isinstance(e, dict):
            k = e.get("class") or "none"
            classes[k] = classes.get(k, 0) + 1
    if classes:
        L.append("- the claims' evidence classes: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    return L + [""]


# ─────────────────────────────────────────────────────────────────────
# EVERY FILE IN THE FOLDER, WITH ITS ROLE
# ─────────────────────────────────────────────────────────────────────
# FIRST MATCH WINS, so the specific patterns come before the general ones.
# A file no pattern names is listed too, and said to have no role on record:
# the index is of the folder, not of what this table knows.

ROLES = (
    (RECORD_NAME, "this page"),
    ("reception.json", "Reception's package: the question, how it was routed, the brief and what it gathered"),
    ("reception_brief.json", "the brief alone, a copy of reception.json's brief for the standalone tools"),
    ("strategy.as_accepted.json", "the Planner's design as accepted on the design page, before a later correction"),
    ("strategy.json", "the Planner's design, as run"),
    ("strategy.refused_*.json", "a design the model server refused before the design page, kept as it was"),
    ("plan.json", "the design as the Experiment Manager was handed it"),
    ("run_plan.json", "the design merged with what the Experiment Manager resolved from it"),
    ("assumptions.json", "the assumptions ledger: what was assumed without you"),
    ("columns.json", "the frame the plan was built on (for a whole-domain study, the control's inputs)"),
    ("run_state.json", "each pipeline stage's status and time"),
    ("experiment.json", "the Experiment Manager's product: the experiment, its chains and its numbers"),
    ("RUN_SUMMARY.json", "the run's summary: start, end and each case's status"),
    ("ERROR_LOG.json", "the Experiment Manager's account of the error that stopped the study"),
    # written by tools/notify_study.py after the manager draws this page; which
    # pass it is from is read from its time (_notice_role, _index_lines)
    (NOTICE_NAME, "the notice tools/notify_study.py wrote when the study ended"),
    ("DESIGN_REVIEW.md", "the design page shown before any compute, with the decision on it"),
    ("PLAN_CORRECTED.md", "a note of a correction made to the plan after the decision"),
    ("ensemble_A.sbatch", "the script of job A, which builds the model's cases and runs them"),
    ("ensemble_A.log", "job A's log: each case built and run, and how it ended"),
    ("ensemble_B.sbatch", "the script of job B: reads the results and writes the analysis after the model jobs"),
    ("ensemble_B.log", "job B's log"),
    ("sampling_design.png", "the map of the columns the model runs, drawn before any compute"),
    ("conus2_subsurface.npz", "the ParFlow CONUS2 subsurface data over the study area, fetched for the run"),
    ("conus2_subsurface.json", "where the CONUS2 subsurface data came from and the grid it covers"),
    ("wtd_conus2.tif", "CONUS2's water table depth over the study area, as a map"),
    ("warmstart/warmstart.json", "the warm start's record: which cell of the donor run each column was cut from"),
    ("warmstart/finidat_*.nc", "one column's starting state, cut from the donor run's saved state"),
    ("warmstart/surfdata_*.nc", "one column's surface data, cut to match its starting state"),
    ("warmstart/*", "a file of the warm start"),
    ("02_setup_plots/*", "a figure drawn while the inputs were set up"),
    ("01_inputs/design_preview.json", "the model server's check and build of the design before the design page"),
    ("01_inputs/control_recipe.json", "how the control was made, read from its case"),
    ("01_inputs/source_changes.patch", "the model code's local changes, saved as a patch"),
    ("01_inputs/case_inputs.json", "one row per case the model server builds, with its inputs and settings (for a whole-domain study, one per run, the control's included)"),
    ("01_inputs/built_cases.json", "the cases job A built, and each one's case folder"),
    ("01_inputs/elm_columns.json", "each column as ELM received it: place, elevation band and starting water table"),
    ("01_inputs/decks/*.in", "one column's PFLOTRAN input deck"),
    ("01_inputs/decks/*.out", "one column's PFLOTRAN log: what it printed as it ran"),
    ("01_inputs/decks/*.err", "the error output of one column's PFLOTRAN run"),
    ("01_inputs/decks/*-mas.dat", "one column's mass balance through the run"),
    ("01_inputs/decks/*.tec", "one column's state at one output time"),
    ("01_inputs/decks/*.regression", "one column's regression summary of its output"),
    ("01_inputs/decks/*", "a file of one column's PFLOTRAN run"),
    ("01_inputs/columns.json", "the frame the inputs were built on"),
    ("01_inputs/domain_experiment.json", "the experiment as handed to the model server: control, runs, years"),
    ("01_inputs/domain_control_facts.json", "the server's facts about the control and its check of the experiment"),
    ("01_inputs/domain_runs.json", "the server's build of the runs: the control, each run, the cost and its findings"),
    ("01_inputs/domain_recipe.json", "the control's recipe, for a control that runs too"),
    ("01_inputs/domain_chain.json", "the years and checks asked for, for a single whole-domain case"),
    ("01_inputs/experiment_summary.json", "a summary of the experiment's inputs"),
    ("01_inputs/runs/*/domain_recipe.json", "the run's recipe: the control's recipe plus its changes"),
    ("01_inputs/runs/*/domain_chain.json", "the run's planned jobs"),
    ("01_inputs/runs/*/domain_expectations.json", "what each of the run's changes must do to the output, checked every year"),
    ("01_inputs/runs/*/weather/*.nc", "the weather change file the run reads"),
    ("01_inputs/runs/*/weather/*", "a weather stream description for the run's change file"),
    ("01_inputs/runs/*", "a file the model server wrote for this run"),
    (f"{DOMAIN_RECORD}/inputs_check.json", "the check of the input set: surface data, grid file, starting state"),
    (f"{DOMAIN_RECORD}/*/inputs_check.json", "the check of the run's input set"),
    (f"{DOMAIN_RECORD}/*/chain.json", "the run's submitted jobs and their ids"),
    (f"{DOMAIN_RECORD}/*/STATUS.log", "the run's jobs, line by line, and the job times the report stage logged"),
    (f"{DOMAIN_RECORD}/*/domain_recipe.json", "the recipe the run's case was built from"),
    (f"{DOMAIN_RECORD}/*/expectations.json", "the change's expected effects, fixed when the chain was armed"),
    (f"{DOMAIN_RECORD}/*/create_case.sh", "the script that creates and builds the run's case"),
    (f"{DOMAIN_RECORD}/*/run_stage.sh", "the script each job of the run's chain runs"),
    (f"{DOMAIN_RECORD}/*/checks/case_*.json", "the case job's check: the model code it was built from"),
    (f"{DOMAIN_RECORD}/*/checks/settings_*.json", "the check of the run's settings against the control's, before its model was built"),
    (f"{DOMAIN_RECORD}/*/checks/smoke_*.json", "the short first run's check: its logs, daily output and what it carried from the saved state"),
    (f"{DOMAIN_RECORD}/*/checks/y[0-9][0-9][0-9][0-9]_*.json", "one model year's check: its logs, its output and the change's expected effects"),
    (f"{DOMAIN_RECORD}/*/checks/*", "a check record of one of the run's jobs"),
    (f"{DOMAIN_RECORD}/*/logs/*", "one job's log from the scheduler"),
    (f"{DOMAIN_RECORD}/*/SourceMods/*", "a model source file copied into the run's case"),
    (f"{DOMAIN_RECORD}/*/user_files/*", "a file copied into the run's case (a weather stream description)"),
    # ONE WHOLE-DOMAIN CASE keeps the same files in domain_case/ itself.
    (f"{DOMAIN_RECORD}/chain.json", "the case's submitted jobs and their ids"),
    (f"{DOMAIN_RECORD}/STATUS.log", "the case's jobs, line by line, and the job times the report stage logged"),
    (f"{DOMAIN_RECORD}/domain_recipe.json", "the recipe the case was built from"),
    (f"{DOMAIN_RECORD}/create_case.sh", "the script that creates and builds the case"),
    (f"{DOMAIN_RECORD}/run_stage.sh", "the script each job of the case's chain runs"),
    (f"{DOMAIN_RECORD}/checks/case_*.json", "the case job's check: the model code it was built from"),
    (f"{DOMAIN_RECORD}/checks/smoke_*.json", "the short first run's check: its logs, daily output and what it carried from the saved state"),
    (f"{DOMAIN_RECORD}/checks/y[0-9][0-9][0-9][0-9]_*.json", "one model year's check: its logs and its output"),
    (f"{DOMAIN_RECORD}/checks/*", "a check record of one of the case's jobs"),
    (f"{DOMAIN_RECORD}/logs/*", "one job's log from the scheduler"),
    (f"{DOMAIN_RECORD}/SourceMods/*", "a model source file copied into the case"),
    ("03_results/regional.json", "the numbers by region, field and year, computed by the model server from the model's output"),
    ("03_results/regional_monthly.csv", "every month and season of 03_results/regional.json as one table, written with the numbers report"),
    ("03_results/extracted.json", "the record of what was read from the model's output"),
    ("03_results/execution_report.txt", "the Experiment Manager's account of each case"),
    ("03_results/results_summary.csv", "one row per case, from the results"),
    ("04_analysis/DOMAIN_REPORT.md", "the numbers report, written by code from the records"),
    ("04_analysis/domain_report.json", "the numbers report's record"),
    ("04_analysis/domain_fig_*.png", "a figure of the numbers report"),
    ("04_analysis/numbers_check.json", "the numbers check: the numbers of 03_results/regional.json made again from the model output by an independent reading, how the two compare, and what it did not check"),
    ("04_analysis/superseded/*", "a file of an earlier analysis, set aside when the numbers did not reproduce"),
    ("04_analysis/REPORT.md", "the Analyzer's report"),
    ("04_analysis/analysis.json", "the Analyzer's record: answer, audited claims, struck claims, cost"),
    ("04_analysis/analysis.pptx", "the Analyzer's slides"),
    ("04_analysis/comparison.json", "Analyzer step 1: the comparison with observations"),
    ("04_analysis/investigation.json", "Analyzer step 2: the figures proposed and what each found"),
    ("04_analysis/investigation_plan.md", "Analyzer step 2's plan, in words"),
    ("04_analysis/interpretation.json", "Analyzer step 3: the claims, the audit and the verdict"),
    # PACKAGE F'S FILES (the user's decision on G part 2, 2026-11-08): the
    # recorded analysis plan, its review, the lookups and the checks of a
    # study with a suite were all "no role on record for this file".
    ("04_analysis/analysis_plan.json", "the recorded analysis plan (step 2a): comparisons, analyses, expectations, competing explanations and the screen, written before any lookup"),
    ("04_analysis/analysis_plan.v1.json", "the first analysis plan, kept as it was: it failed its review and was rewritten once, before any lookup"),
    ("04_analysis/analysis_plan_review.json", "step 3's review of the analysis plan: what the code check and the model review found, tied to the plan's hash"),
    ("04_analysis/analysis_plan_review.v1.json", "the review of the first analysis plan, kept as it was"),
    ("04_analysis/analysis_plan_amendments.jsonl", "the amendments to the analysis plan, appended after outcomes were seen, never planned"),
    ("04_analysis/lookups.jsonl", "every call made on the suite, the framework's and each figure's, refused or answered: the evidence records the claims name"),
    ("04_analysis/planned_phase.json", "the close of the planned analyses: each analysis of the plan completed, failed or skipped, with its reason"),
    ("04_analysis/required_outputs.json", "the framework's own calls of the plan's required outputs: each call as the brief wrote it, its lookup line, and what it came to"),
    ("04_analysis/required_outputs.pkl", "the results of those calls, which the framework draws the figures from"),
    ("04_analysis/screen_record.json", "the plan's screen: what it ran over, its scoring rule and its display rule"),
    ("04_analysis/screen_L*.csv", "one screen's whole table (lookup L<n>): every field and period with every column"),
    ("04_analysis/contract_reads.jsonl", "the method contracts the scripts read (no lookup)"),
    ("04_analysis/evidence_numbers_check.json", "the numbers check of the claims' evidence: each number of every record a claim cites, made again from the model output"),
    ("04_analysis/step*_round*_prompt.txt", "what a language model was sent, one step and round"),
    ("04_analysis/step*_round*_reply.txt", "a language model's reply, one step and round"),
    ("04_analysis/compare_*.png", "a figure of the comparison with observations"),
    ("04_analysis/*.spec.json", "the specification of the figure of the same name, which the framework drew"),
    ("04_analysis/*.py", "the script that drew the figure of the same name"),
    ("04_analysis/*.png", "a figure the Analyzer drew (its script beside it)"),
)


def file_role(rel):
    """The role of a file, by its path relative to the run folder."""
    for pattern, role in ROLES:
        if fnmatch.fnmatchcase(rel, pattern):
            return role
    return "no role on record for this file"


# WHICH PASS THE NOTICE IS FROM, BY ITS TIME (2026-11-03). The manager draws
# this page inside job B and the notice is written after it, so a notice on the
# folder then is an earlier pass's; but the page can be drawn again later
# (python src/agents/study_record.py RUN_DIR), after this pass's notice. A
# notice newer than the last stage run_state.json records is this pass's.
def _last_stage_time(rd):
    try:
        st = _read_json(rd / "run_state.json") or {}
    except _Unreadable:
        return None
    times = []
    for e in ((st.get("stages") or {}) if isinstance(st, dict) else {}).values():
        try:
            times.append(datetime.fromisoformat(str((e or {}).get("at"))))
        except (TypeError, ValueError, AttributeError):
            pass
    return max(times) if times else None


def _notice_role(rd):
    base = file_role(NOTICE_NAME)
    last = _last_stage_time(rd)
    try:
        written = datetime.fromtimestamp((rd / NOTICE_NAME).stat().st_mtime)
    except OSError:
        return base
    if last is None:
        return (f"{base}; run_state.json records no stage time, so which pass it is from "
                f"is not on record")
    which, than = (("this pass", "newer") if written >= last else
                   ("an earlier pass", "older"))
    return (f"the notice tools/notify_study.py wrote when {which} of the study ended (it is "
            f"{than} than the last stage run_state.json records, "
            f"{last.isoformat(timespec='seconds')})")


def _index_lines(rd, page):
    files = set()
    for root, dirs, names in os.walk(rd):        # links to folders are not followed
        for n in names:
            files.add(Path(root, n).relative_to(rd).as_posix())
    try:                                         # the page being drawn now
        files.add(page.resolve().relative_to(rd.resolve()).as_posix())
    except ValueError:
        pass                                     # it is written outside the folder
    L = [f"## Every file in this folder ({len(files)})", "",
         "| file | what it is |", "|---|---|"]
    for rel in sorted(files):
        L.append(f"| `{rel}` | {_notice_role(rd) if rel == NOTICE_NAME else file_role(rel)} |")
    # WRITTEN AFTER THIS PAGE (2026-11-03): job B's notice is written once
    # this page is, so a fresh study's index cannot hold it; it is named here.
    if NOTICE_NAME not in files and (rd / "ensemble_B.sbatch").is_file():
        L += ["", f"Written after this page: `{NOTICE_NAME}`, the notice tools/notify_study.py "
                  f"writes when job B ends (which stages ran, which reports exist, and "
                  f"whether the study is usable)."]
    return L + [""]


# ─────────────────────────────────────────────────────────────────────
# THE PAGE
# ─────────────────────────────────────────────────────────────────────

def render(run_dir, out=None):
    """The page, as text. `out` is where it will be written (links are made
    relative to it); the run folder by default."""
    rd = Path(run_dir)
    page = Path(out) if out else rd / RECORD_NAME
    page_dir = page.parent
    L = [f"# Study record: {rd.name}", "",
         f"*Drawn by code from the files in this folder, with no language model: the "
         f"files are the authority, and this page only quotes or counts them. "
         f"{datetime.now().isoformat(timespec='seconds')}*", ""]
    for part in (lambda: _question_lines(rd),
                 lambda: _design_lines(rd, page_dir),
                 lambda: _check_lines(rd),
                 lambda: _runs_lines(rd),
                 lambda: _numbers_lines(rd, page_dir),
                 lambda: _analyzer_lines(rd, page_dir),
                 lambda: _index_lines(rd, page)):
        # ONE BROKEN RECORD COSTS ONE SECTION (2026-11-03), never the page: the
        # section says what failed and the rest is still drawn.
        try:
            L += part()
        except Exception as e:                               # noqa: BLE001
            L += [f"*This part could not be drawn: {type(e).__name__}: {_one_line(e, 300)}*", ""]
    return "\n".join(L).rstrip() + "\n"


def write_record(run_dir, out=None):
    """Draw STUDY_RECORD.md for a run folder and write it (into the folder
    unless `out` names another path); returns the path written."""
    rd = Path(run_dir)
    if not rd.is_dir():
        raise FileNotFoundError(f"no run folder at {rd}")
    path = Path(out) if out else rd / RECORD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(rd, out=path))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a run folder's STUDY_RECORD.md, "
                                             "drawn by code from the folder alone.")
    ap.add_argument("run_dir", help="the run folder")
    ap.add_argument("--out", help=f"where to write the page (default RUN_DIR/{RECORD_NAME})")
    a = ap.parse_args(argv)
    print(write_record(a.run_dir, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
