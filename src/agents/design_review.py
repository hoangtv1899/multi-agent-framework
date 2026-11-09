#!/usr/bin/env python3
"""
The design, explained before it runs — DESIGN_REVIEW.md.
src/agents/design_review.py

    python src/agents/design_review.py RUN_DIR      # render for an existing run

After the planner writes strategy.json, the coordinator writes this file
beside it: one page a person can read through and understand before
agreeing to spend compute. In an interactive session the pipeline PAUSES
on it; unattended runs record it and continue.

A RENDERER, NOT A REASONER. Everything on the page is quoted or counted
out of reception.json and strategy.json — the planner's own recorded
reasons, verbatim. No language model writes here: a review that could
say things the records do not is the exact failure the guard rails
exist to prevent, in the one file made for human eyes. When a section
reads thin, the fix is upstream — make the planner record the reason —
never a second pass of prose.

Nothing downstream reads this file. The Analyzer's inputs are unchanged
(experiment.json, reception.json, strategy.json); this is a leaf for
people, and the two JSON records stay the authority.

A PAGE A PERSON COULD MAKE THE RUNS FROM (2026-10-28). For a whole-domain
experiment the model server checks the design and builds every run (its
recipe, its checks, the scripts its jobs run) before this page is drawn,
submitting nothing, and lists what it wrote in 01_inputs/design_preview.json.
The page then carries the server's verdict in its first part and, after the
decision line, a second part: how to make these runs by hand (the code and
its local changes, the control, each run's changes from the control's recipe,
the files written for it, its case and stage scripts verbatim, its jobs and
what each checks, the input files and weather, and the regions and fields
the answer is read from). Still quoted or counted, never written: the
server's files are the third authority beside the two records.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REVIEW_NAME = "DESIGN_REVIEW.md"
_PENDING = "**Decision:** pending — the coordinator records it here."
# How the coordinator appends a change asked for at this review to the request
# (workflow.py), so the page can show the question and each change apart.
REVISION_MARK = "\n\nRevision requested at the design review: "
# THE PAGE'S SECOND PART (2026-10-28): how to make the runs by hand, drawn
# from what the model server wrote before the review. The terminal shows the
# page down to this heading and names the file for the rest.
APPENDIX_MARK = "# How to make these runs by hand"
PREVIEW_FILE = "01_inputs/design_preview.json"


def _trim(text, n=200):
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    # END AT A SENTENCE when one falls in the back half of the budget — a
    # quote that stops at a full stop needs no ellipsis and drops nothing
    # mid-thought. Only the back half: cutting at a period near the start
    # would throw away most of what the budget allows.
    dot = max(cut.rfind(". "), cut.rfind("; "))
    if dot >= n // 2:
        # A full stop ends the quote; a semicolon only ends a clause, and a
        # quote ending in ";" reads as broken (2026-10-27), so it is marked
        # as shortened.
        return cut[: dot + 1] if cut[dot] == "." else cut[:dot] + "…"
    # NEVER MID-WORD: "a pillow measures i…" reads as a glitch where
    # "a pillow measures…" reads as a quote that was shortened.
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


def _period_line(brief):
    rs = (brief.get("run_settings") or {}).get("resolved_period") or {}
    y0, y1 = rs.get("yr_start"), rs.get("yr_end")
    if not y0:
        return "not resolved"
    span = f"{y0}" if (y1 in (None, y0)) else f"{y0}–{y1}"
    src = str(rs.get("source") or "").lower()
    said = {"user": "you stated it",
            "carried": "carried from the prior run",
            "default": "a DEFAULT the model chose, recorded as such"}
    return f"{span} ({said.get(src, src or 'source unrecorded')})"


def _domain_line(brief):
    d = brief.get("domain") or {}
    if not isinstance(d, dict) or not d.get("name"):
        return "no basin — this study has no place, by design"
    bits = [d["name"]]
    if d.get("huc"):
        bits.append(f"HUC {d['huc']}")
    if d.get("area_km2"):
        bits.append(f"{round(float(d['area_km2']))} km²")
    if d.get("source"):
        bits.append(f"resolved via {d['source']}")
    return ", ".join(bits)


def _with_params(text, params):
    """A change's published text with a run's parameters where it names them
    ({ppmv}); a placeholder the run does not give stays as written."""
    out = str(text or "")
    for k, v in (params or {}).items():
        out = out.replace("{" + str(k) + "}", f"{v:g}" if isinstance(v, float) else str(v))
    return out


# THE CONTROL'S FIRE AND GRID ON THE PAGE (2026-11-03), quoted from the
# sentences the model server read from the control's own files, as the
# card shows them: a reviewer sees whose fire burns in it and how fine its
# grid is before saying yes. Facts that lack a part say so.
def _control_setup_lines(facts):
    out = []
    for part, label in (("fire", "its fire"), ("grid", "its grid")):
        v = (facts or {}).get(part)
        if isinstance(v, dict) and v.get("sentence"):
            out.append(f"- {label}: {v['sentence']}")
        elif facts:
            out.append(f"- {label}: not in the control's facts"
                       + (f" ({v['error']})" if isinstance(v, dict) and v.get("error")
                          else ""))
    return out


# THE QUESTION TYPE ON THE PAGE (package E, 2026-11-05). The Planner records
# analysis.question_type from the question as asked; the page shows it, since
# it decides what the report may call a tested expectation.
QUESTION_TYPE_WORDS = {
    "hypothesis": "the question proposes an expectation to test, so the analysis plan "
                  "states it before any result and the report may call it tested",
    "exploratory": "the question asks what changed or happened, without an expectation "
                   "to test, so the findings are reported as exploratory"}


def question_type_line(analysis: dict) -> str:
    qt = str(analysis.get("question_type") or "").strip().lower()
    words = QUESTION_TYPE_WORDS.get(qt, "not one of hypothesis or exploratory; the Analyzer "
                                        "will read it as exploratory")
    return f"question type: {qt or 'none recorded'} ({words})"


def _whole_domain_lines(brief, strategy):
    """A control and runs of named changes (docs/ELM_DOMAIN_EXPERIMENTS.md),
    every line quoted or counted from the records: the experiment Reception
    settled, the analysis the planner chose, and the server's rulebook the
    plan was made under (strategy.domain_rulebook), which says what each
    change means and does not mean."""
    de = dict(brief.get("domain_experiment") or {})
    de.update(strategy.get("domain_experiment") or {})
    rb = strategy.get("domain_rulebook") or {}
    menu = {c.get("id"): c for c in ((rb.get("changes") or {}).get("changes") or [])}
    ctl = de.get("control") or {}
    row = next((c for c in ((rb.get("controls") or {}).get("controls") or [])
                if c.get("name") == ctl.get("name")), {})
    facts = row.get("facts") or {}
    out = [f"- the control: **{ctl.get('name', '?')}**"
           + ("; reused, so its existing output is compared and nothing is spent on it"
              if ctl.get("reuse") else "; run again from its recipe")
           + (f"; it is {_trim(row.get('description'), 170)}" if row.get("description")
              else "")]
    oy = facts.get("output_years") or []
    if oy:
        out.append(f"- its output covers {oy[0]}–{oy[-1]}, from the saved state of "
                   f"{facts.get('start_year')}")
    out += _control_setup_lines(facts)
    ev = row.get("evaluation") or {}
    if isinstance(ev, dict) and ev.get("summary"):
        # measured outside the server and entered with its source: what the
        # scientist should know about the run before trusting an answer from it
        out.append(f"- against observations: {_trim(ev['summary'], 420)}"
                   + (f" ({ev['source']})" if ev.get("source") else ""))
    for item in row.get("not_in_it") or []:
        out.append(f"- not in this run: {_trim(item, 420)}")
    runs = de.get("runs") or []
    if not runs:
        out.append("- no run differs from the control: its output is read as it is, "
                   "at no cost")
    for r in runs:
        ch = r.get("changes") or []
        out.append(f"- run **{r.get('name')}**: the control with "
                   + (", ".join(c.get("id", "?") + (f" {json.dumps(c['params'])}"
                                                    if c.get("params") else "")
                                for c in ch) or "no change"))
        for c in ch:
            m = menu.get(c.get("id")) or {}
            if m.get("meaning"):
                # the menu's text with this run's parameters in it; the
                # not-meaning in full, since it says what the difference must
                # never be called
                out.append(f"  - {c.get('id')} means "
                           f"{_trim(_with_params(m['meaning'], c.get('params')), 240)}")
                if m.get("not_meaning"):
                    out.append(f"  - it is NOT {' '.join(str(m['not_meaning']).split())}")
    years = de.get("years") or []
    n_exec = len(runs) + (0 if ctl.get("reuse") else 1)
    hp = facts.get("hours_per_model_year")
    if years:
        span = (str(years[0]) if years[0] == years[-1] else f"{years[0]}–{years[-1]}")
        out.append(f"- {len(years)} model year(s), {span}, for "
                   f"{n_exec} run(s) to execute"
                   + (f": about {round(n_exec * len(years) * hp, 1)} node-hours"
                      if hp else ""))
    an = de.get("analysis") or {}
    if an.get("regions"):
        regs = [x if isinstance(x, dict) else {"name": x} for x in an["regions"][:5]]
        out.append("- reported over: " + "; ".join(
            str(x.get("name")) + (f" within {x['within']}" if x.get("within") else "")
            + (f" ({_trim(x.get('why'), 90)})" if x.get("why") else "")
            for x in regs))
    if an.get("normal_years"):
        out.append(f"- compared with the normal years {an['normal_years']}")
    if an.get("question_type"):
        out.append("- " + question_type_line(an))
    for cv in (de.get("caveats") or [])[:3]:
        out.append(f"- caveat: {_trim(cv, 200)}")
    if runs:
        out.append("- each run's settings are compared with the control's before its "
                   "model is built, and each change's rule on the output is checked "
                   "every year")
    return out


def _design_lines(brief, strategy):
    """The archetype-specific middle: what will actually be built."""
    arch = str(brief.get("design_archetype")
               or strategy.get("archetype") or "site").lower()
    out = []
    if arch == "conceptual":
        sw = brief.get("sweep") or {}
        for f in sw.get("factors") or []:
            out.append(f"- vary **{f.get('name')}** over {f.get('levels')}"
                       + (f" (settled by {f.get('settled_by')})"
                          if f.get("settled_by") else ""))
        hf = sw.get("held_fixed") or {}
        by = sw.get("held_fixed_settled_by") or {}
        for k, v in hf.items():
            out.append(f"- hold **{k}** fixed at {json.dumps(v)}"
                       + (f" (settled by {by[k]})" if by.get(k) else ""))
        n = (strategy.get("sampling") or {}).get("n_columns")
        if n:
            out.append(f"- {n} column(s), one per level combination — "
                       f"no basin, no grid, no station sampling")
        return "controlled sweep", out
    if arch == "coupling":
        c = dict(strategy.get("coupling") or {})
        c.update(brief.get("coupling") or {})
        out.append(f"- a follow-up on **{c.get('prior_experiment', '?')}** — "
                   f"its columns reused verbatim"
                   + (f" ({c.get('n_columns_prior')} of them)"
                      if c.get("n_columns_prior") else ""))
        if c.get("from_model") or c.get("to_model"):
            out.append(f"- direction: **{c.get('from_model', '?')} → "
                       f"{c.get('to_model', '?')}**")
        if c.get("coupling_variable"):
            out.append(f"- what travels: {_trim(c['coupling_variable'], 230)}")
        out.append("- nothing is fetched: domain, grid, observations and rain "
                   "are carried from the prior run's record")
        return "coupling follow-up", out
    if arch == "whole_domain":
        return "whole-domain experiment", _whole_domain_lines(brief, strategy)
    # site
    sm = strategy.get("sampling") or {}
    if sm.get("approach"):
        out.append(f"- **{sm['approach']}** — "
                   f"{sm.get('n_columns', '?')} column(s)"
                   + (f": {_trim(sm.get('justification'), 220)}"
                      if sm.get("justification") else ""))
    for v in strategy.get("validation") or []:
        st = v.get("stations")
        # THE COMPARISON NOTE IS THE TRUTH, so it is quoted, not paraphrased.
        # An empty station list has two different meanings — nothing exists
        # (no flux tower in the basin) and exists-but-cannot-be-pinned (a
        # gauge integrates its whole upstream area) — and the canned phrase
        # "no in-basin station" was flatly wrong for the second. The planner's
        # own sentence says which one it is, and why stations were dropped.
        # A one-word cadence ("daily") is not a sentence and is not quoted.
        note = _trim(v.get("comparison"), 220)
        if st:
            line = (f"- check **{v.get('variable')}** against {len(st)} "
                    f"station(s): {', '.join(map(str, st[:4]))}")
            if " " in note:
                line += f" — {note}"
        elif " " in note:
            line = f"- **{v.get('variable')}**: {note}"
        else:
            line = (f"- check **{v.get('variable')}** against "
                    f"no in-basin station")
        out.append(line)
    # WHAT THE DESIGN SEPARATES vs WHAT IT ONLY CARRIES, when the planner
    # recorded it. A basin study stratified by elevation still has soil and
    # the water table varying underneath — recorded per column, but their
    # contribution entangled with the stratified axis. Saying so here is
    # what keeps "why this can answer it" honest. Absent in older records;
    # rendered only when present.
    dr = strategy.get("drivers") or {}
    if dr.get("controlled"):
        out.append("- **separated on purpose:** "
                   + "; ".join(_trim(x, 120) for x in dr["controlled"][:4]))
    if dr.get("carried"):
        out.append("- **varies underneath, recorded but not separated:** "
                   + "; ".join(_trim(x, 120) for x in dr["carried"][:4]))
    return "real-basin study", out


# ─────────────────────────────────────────────────────────────────────────────
# THE SERVER'S CHECK, AND HOW TO MAKE THE RUNS BY HAND (2026-10-28)
# ─────────────────────────────────────────────────────────────────────────────
# The test for this page: a person could make the runs from it alone.
# The model server now checks the design and builds every run (its recipe,
# its checks, the scripts its jobs run) before the page is drawn, submitting
# nothing, and lists what it wrote in 01_inputs/design_preview.json. Every
# line below is quoted or counted from that record and the files it names;
# the files stay the authority.

def _load(run_dir, rel, as_json=True):
    """A file the preview names, read; None when it cannot be."""
    if not run_dir or not rel:
        return None
    p = Path(run_dir) / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()) if as_json else p.read_text()
    except (OSError, ValueError):
        return None


def _finding(f, n=400):
    if isinstance(f, dict):
        return f"{f.get('where', '?')}: {_trim(f.get('what'), n)}"
    return _trim(f, n)


def _check_lines(preview):
    """The server's verdict on the design, in the part the decision rests on."""
    if not preview:
        return []
    L = ["## The model server's check"]
    for i, r in enumerate(preview.get("rounds") or [], 1):
        L.append(f"- the server refused the planner's {'first' if i == 1 else 'next'} design, "
                 f"and the planner changed it before this page (the refused design is kept "
                 f"as `{r.get('design')}`):")
        L.extend(f"  - {_finding(f, 300)}" for f in r.get("findings") or [])
    if preview.get("refused"):
        L.append("**Refused.** The server will not build this design as it stands, so "
                 "nothing was built for its runs:")
        L.extend(f"- {_finding(f)}" for f in preview["refused"])
        return L + [""]
    if preview.get("error"):
        L.append(f"**Not built.** {_trim(preview['error'], 500)}")
        return L + [""]
    runs = preview.get("runs") or []
    L.append("**Passed.** The server checked this design against its change list and the "
             "control's facts"
             + (", and built every run's recipe, checks and job scripts without "
                "submitting anything" if runs else "")
             + ". The second part of this page, how to make these runs by hand, is drawn "
               "from what it wrote.")
    chk = preview.get("check") or {}
    for u in chk.get("unusual") or []:
        L.append(f"- unusual: {_finding(u, 300)}")
    ic = preview.get("inputs_check") or {}
    if ic:
        n = (ic.get("checks") or {}).get("active_cells")
        L.append("- the input set (surface data, grid file, starting state) "
                 + ("passed its check" if ic.get("pass") else
                    "FAILED its check: " + "; ".join(map(str, ic.get("reasons") or [])))
                 + (f", {n:,} cells" if isinstance(n, int) else ""))
    nh = (chk.get("cost") or {}).get("node_hours")
    if nh is not None:
        L.append(f"- cost: about {nh} node-hours" if (runs or nh) else
                 "- cost: none, since nothing is run")
    for p in preview.get("problems") or []:
        L.append(f"- not on this page: {_trim(p, 300)}")
    return L + [""]


def _quote_path(p):
    return f"`{p}`" if p else "(none)"


def _nl_files(recipe):
    """(key, path) for every file the land namelist names."""
    out = []
    for line in (recipe.get("user_nl") or {}).get("elm") or []:
        m = re.match(r"\s*(\w+)\s*=\s*'([^']*/[^']*)'", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _recipe_commands(recipe):
    """A recipe as the commands its case script runs, in the order the
    server's own script writer uses (domain_case.write_case_script), for a
    recipe the server wrote no script for: the control's, read from its case.
    The case name is left to the reader: the control's own case exists."""
    create = (f"{recipe.get('srcroot')}/cime/scripts/create_newcase --case \"$CASEDIR\" "
              f"--compset {recipe.get('compset')} --res {recipe.get('res')} "
              f"--machine {recipe.get('mach')} --compiler {recipe.get('compiler')} "
              f"--project {recipe.get('project')}")
    for k, v in (recipe.get("create_options") or {}).items():
        create += f" --{k}" + ("" if v is True else f" {v}")
    L = ["```bash", "CASEDIR=<a new case directory>", create, 'cd "$CASEDIR"']
    for e in recipe.get("xml") or []:
        opts = e[2] if len(e) > 2 and isinstance(e[2], dict) else {}
        flags = (f"--subgroup {opts['subgroup']} " if opts.get("subgroup") else "") + \
            ("--append " if opts.get("append") else "")
        L.append(f"./xmlchange {flags}{e[0]}={e[1]}")
    for comp, body in (recipe.get("user_nl") or {}).items():
        L.append(f"cat > user_nl_{comp} <<'NL_END_OF_RECIPE'")
        L.extend(body)
        L.append("NL_END_OF_RECIPE")
    for name, m in (recipe.get("user_files") or {}).items():
        L.append(f"cp {m.get('from')} {name}    # sha256 {m.get('sha256')}")
    for rel, m in (recipe.get("sourcemods") or {}).items():
        L.append(f"mkdir -p SourceMods/{Path(rel).parent}")
        L.append(f"cp {m.get('from')} SourceMods/{rel}    # sha256 {m.get('sha256')}")
    L += ["./case.setup", "./preview_namelists", "```"]
    return L


def _recipe_diff(run, ctl):
    """What a run's recipe changes from the control's, setting by setting
    and line by line, so the reader sees the change and nothing else."""
    L = []
    for k in ("compset", "res", "mach", "compiler", "project", "srcroot", "start"):
        if run.get(k) != ctl.get(k):
            L.append(f"- {k}: `{ctl.get(k)}` in the control, `{run.get(k)}` here")
    cx = {str(e[0]): str(e[1]) for e in ctl.get("xml") or []}
    rx = {str(e[0]): str(e[1]) for e in run.get("xml") or []}
    for n in rx:
        if cx.get(n) != rx[n]:
            L.append(f"- `./xmlchange {n}={rx[n]}`"
                     + (f" (the control's: {cx[n]})" if n in cx else " (not set in the control)"))
    for n in cx:
        if n not in rx:
            L.append(f"- {n} is not set here (the control's: {cx[n]})")
    body = []
    comps = list(dict.fromkeys(list((ctl.get("user_nl") or {})) +
                               list((run.get("user_nl") or {}))))
    for comp in comps:
        a = [l.rstrip() for l in (ctl.get("user_nl") or {}).get(comp) or [] if l.strip()]
        b = [l.rstrip() for l in (run.get("user_nl") or {}).get(comp) or [] if l.strip()]
        gone, new = [l for l in a if l not in b], [l for l in b if l not in a]
        if gone or new:
            body.append(f"user_nl_{comp}")
            body += [f"- {l}" for l in gone] + [f"+ {l}" for l in new]
    if body:
        L += ["```diff", *body, "```"]
    for kind, key in (("user file", "user_files"), ("source modification", "sourcemods")):
        a, b = ctl.get(key) or {}, run.get(key) or {}
        for name, m in b.items():
            if (a.get(name) or {}).get("sha256") != m.get("sha256"):
                L.append(f"- {kind} `{name}`, sha256 `{m.get('sha256')}`"
                         + (" (the control has none)" if name not in a else
                            " (the control's differs)"))
        for name in a:
            if name not in b:
                L.append(f"- the control's {kind} `{name}` is not copied here")
    return L or ["- nothing: the run's recipe is the control's under a new case name"]


def _weather_lines(w):
    """One weather file the server wrote for a run, as counted from the file."""
    rec = w.get("records") or {}
    g = w.get("grid") or {}
    L = [f"- `{w.get('file')}` (sha256 `{w.get('sha256')}`): {rec.get('n')} monthly "
         f"records, {rec.get('first')} to {rec.get('last')}, each at {rec.get('each')}, on "
         f"the control's {' x '.join(map(str, g.get('shape') or []))} grid"]
    for var, f in (w.get("fields") or {}).items():
        act = ("added to the air temperature" if f.get("how") == "offset"
               else "a factor on precipitation" if f.get("how") == "factor" else f.get("how"))
        if f.get("units") and f.get("units") != "1":
            act = f"{f['units']}, {act}"
        if f.get("months"):
            L.append(f"  - `{var}` ({act}, read as `{f.get('acts_on')}`): "
                     f"{', '.join(f'{v:g}' for v in f.get('values') or [])} in "
                     f"{f.get('cells'):,} cells in {', '.join(f['months'])}; "
                     f"{f.get('identity'):g} everywhere else")
        else:
            L.append(f"  - `{var}` ({act}): {f.get('identity'):g} everywhere, so it changes "
                     f"nothing")
    return L


def _settings_lines(settings):
    """What the case job requires of the run's resolved namelists beyond the
    control's (the expectations' settings), with per-stream lists that grow
    alike put on one line."""
    L = []
    for nl, keys in (settings or {}).items():
        grows = {}
        for k, v in (keys or {}).items():
            if isinstance(v, dict) and set(v) == {"appends"}:
                grows.setdefault(v["appends"], []).append(k)
                continue
            if isinstance(v, dict):
                bits = []
                if "appends" in v:
                    bits.append(f"grows by {v['appends']}")
                if v.get("starts"):
                    bits.append("with " + ", ".join(map(str, v["starts"])))
                if v.get("values"):
                    bits.append("values " + ", ".join(map(str, v["values"])))
                L.append(f"  - `{nl}` {k}: " + "; ".join(bits))
            else:
                L.append(f"  - `{nl}` {k} = {json.dumps(v)}")
        for n, ks in grows.items():
            L.append(f"  - `{nl}` {', '.join(ks)}: each grows by {n}")
    return L


def _rule_line(o):
    fields = o.get("fields") or ([o["field"]] if o.get("field") else [])
    how = [o.get("rule")]
    if o.get("how"):
        how.append(str(o["how"]))
    if o.get("tolerance") is not None:
        how.append(f"tolerance {o['tolerance']:g}")
    where = f" on {', '.join(fields)} in {o.get('stream')}" if fields else \
        (f" on {o.get('stream')}" if o.get("stream") else "")
    against = (f", against `{o.get('anomaly_var')}` in `{Path(o['anomaly_file']).name}`"
               if o.get("anomaly_file") else "")
    return f"  - {o.get('why')} ({'; '.join(map(str, how))}{where}{against})"


def _run_lines(e, run_dir, ctl_recipe, words):
    """One executed run: what it changes, the files written for it, its case
    script, its jobs and what they check."""
    L = [f"## Run {e.get('name')}: case `{e.get('case')}`", "",
         f"Built as `{e.get('case_dir')}`; its changes: "
         + (", ".join(map(str, e.get("changes") or [])) or "none")
         + (f"; its jobs wait for run {e['after']}'s case job" if e.get("after") else "")
         + ".", ""]
    recipe = _load(run_dir, e.get("recipe")) or {}
    if ctl_recipe and recipe:
        L += ["### What it changes from the control's recipe", ""]
        L += _recipe_diff(recipe, ctl_recipe) + [""]
    if e.get("weather_files") or recipe.get("user_files"):
        L += ["### Files written for it", ""]
        for w in e.get("weather_files") or []:
            L += _weather_lines(w)
        for name, m in (recipe.get("user_files") or {}).items():
            text = _load(run_dir, Path(e.get("case_script") or "").parent / "user_files" / name,
                         as_json=False) if e.get("case_script") else None
            L.append(f"- `{name}` (sha256 `{m.get('sha256')}`)"
                     + (", as written:" if text else f", from `{m.get('from')}`"))
            if text:
                L += ["```xml", text.rstrip(), "```"]
        L.append("")
    script = _load(run_dir, e.get("case_script"), as_json=False)
    if script:
        L += [f"### The case, as its first job creates it (`{e['case_script']}`)", "",
              "```bash", script.rstrip(), "```", ""]
    jobs = e.get("jobs") or []
    if jobs:
        L += ["### Its jobs", "", "| stage | job name | queue | wall time | waits for |",
              "|---|---|---|---|---|"]
        # the report job waits on every other job with afterany
        # (domain_case.submit_chain), which its plan leaves as no `after`
        L += [f"| {j.get('stage')} | {j.get('name')} | {j.get('partition')} | "
              f"{j.get('walltime')} | "
              f"{j.get('after') or ('every job above' if j.get('stage') == 'report' else 'nothing')} |"
              for j in jobs]
        L += ["", "Each job starts only when the one it waits for ended well, and a failure "
                  "cancels the jobs after it; the report job runs whatever happens, and writes "
                  "each job's state and time.", ""]
    stage = _load(run_dir, e.get("stage_script"), as_json=False)
    if stage:
        L += [f"Each job runs `{e['stage_script']}` with its stage's name, as follows:", "",
              "```bash", stage.rstrip(), "```", ""]
    ex = _load(run_dir, e.get("expectations")) or {}
    spec = _load(run_dir, e.get("chain_spec")) or {}
    words = words or {}
    L += ["### What its jobs check", ""]
    if words.get("case"):
        L.append(f"- the case job: {words['case']}. For this run:")
        L += _settings_lines(ex.get("settings")) or ["  - (no setting differs)"]
    if words.get("smoke") and spec.get("smoke", True):
        L.append(f"- the smoke job: {words['smoke']}")
    if words.get("year"):
        L.append(f"- each year's job: {words['year']}")
    if ex.get("output"):
        L.append("- each year's job, this run's changes:")
        L += [_rule_line(o) for o in ex["output"]]
    for y, sets in sorted((ex.get("year_settings") or {}).items()):
        L.append(f"- the {y} job first sets: "
                 + ", ".join(f"{s[0]}={s[1]}" for s in sets if len(s) > 1))
    ref = spec.get("reference") or {}
    if ref and words.get("reference"):
        L.append(f"- each year's job, against the control: {words['reference']} "
                 f"(`{ref.get('run_dir')}`, stream {ref.get('stream')})")
    return L + [""]


# MONTHS AND SEASONS (2026-11-03). The field table gives a year's value only,
# while the server also reports each field by month and by season; a scientist
# approving a question about a summer should see that it is answered from the
# summer's numbers.
MONTHS_AND_SEASONS_LINE = (
    "Each field is also reported by month and by season (MAM, JJA and SON within "
    "the year): a total over the year becomes the total over the month or season, "
    "a mean over the year the mean over it. DJF belongs to the year of its January "
    "and needs the December before it, so a run whose output starts in January "
    "has no DJF in its first year.")


def _period_cell(f, period: str) -> str:
    how = f.get(f"by_{period}")
    units = f.get(f"{period}_units")
    return f"{how} ({units})" if how and units else (how or "")


def _periods_line(periods) -> str:
    """The periods every field is reported over, in the server's words
    (domain_analysis.periods_plan); empty when the plan carries none."""
    p = periods or {}
    seasons = p.get("seasons") or []
    if not seasons:
        return ""
    return ("Each field is reported for " + ", ".join(
        x for x in (p.get("year"), p.get("month")) if x)
        + " and each season: "
        + "; ".join(f"{s.get('name')} ({', '.join(s.get('months') or [])})" for s in seasons)
        + (f". {p['season_left_out']}." if p.get("season_left_out") else "."))


def _appendix(preview, run_dir, strategy, brief):
    """The page's second part. Empty when the server built nothing."""
    if not preview or preview.get("refused") or preview.get("error"):
        return []
    L = ["", "---", "", APPENDIX_MARK, "",
         "*Drawn by code from what the model server wrote before this review "
         f"(`{PREVIEW_FILE}` and the files it names), so the runs can be made from this "
         "page alone. Nothing here was written by a language model, and nothing has been "
         "submitted; the files are the authority.*", ""]
    runs = preview.get("runs") or []
    ctl = preview.get("control") or {}
    ctl_recipe = _load(run_dir, preview.get("control_recipe")) or {}
    first = (_load(run_dir, runs[0].get("recipe")) if runs else None) or ctl_recipe

    L += ["## The code", ""]
    src = preview.get("source") or {}
    if src.get("error"):
        L.append(f"- the E3SM checkout could not be read: {src['error']}")
    elif src:
        L.append(f"- E3SM checkout `{src.get('srcroot')}` at commit `{src.get('commit')}`")
        # git status --short lines: the path after the status letters
        ch = [str(c).split(None, 1)[-1] for c in src.get("changed") or []]
        # THE PATCH, AS THE SERVER WRITES IT (2026-11-03): the top-level and
        # submodule changes with paths from the checkout root, and untracked
        # files as new files, except any too large to carry, listed here.
        L.append((f"- local changes to {len(ch)} path(s): " + ", ".join(f"`{c}`" for c in ch)
                  if ch else "- no local changes to tracked files")
                 + (f"; the local changes are saved as `{src['patch']}`, to apply with "
                    "`git apply` at the checkout root once that commit and its submodules "
                    "are checked out (untracked files are in it as new files)"
                    if src.get("patch") else ""))
        if src.get("patch_left_out"):
            L.append("- left out of the patch, too large to carry, so copy them by hand: "
                     + ", ".join(f"`{u}`" for u in src["patch_left_out"]))
        if src.get("untracked"):
            L.append("- untracked files in the checkout (counted in the fingerprint): "
                     + ", ".join(f"`{u}`" for u in src["untracked"][:10]))
        L.append(f"- fingerprint of that code: `{src.get('fingerprint')}` (each case job "
                 f"records the fingerprint of the code it built, to compare with)")
        rep = ctl.get("reproduced") or {}
        if not runs and rep.get("fingerprint"):
            L.append("- the control was reproduced bit for bit from code with fingerprint "
                     f"`{rep['fingerprint']}`: "
                     + ("the same code" if rep["fingerprint"] == src.get("fingerprint")
                        else "NOT the checkout's code now"))
    for rel, m in (first.get("sourcemods") or {}).items():
        L.append(f"- source modification {'every run copies in' if runs else 'the control holds'}"
                 f": `{rel}` from `{m.get('from')}` (sha256 `{m.get('sha256')}`)")
    L.append("")

    L += ["## The control", ""]
    L.append(f"- **{ctl.get('name', '?')}**: case `{ctl.get('case_dir')}`, output "
             f"`{ctl.get('run_dir')}`"
             + ("; reused: its existing output is read, and nothing is run for it"
                if ctl.get("reuse") else "; it runs too, as run control"))
    if ctl_recipe and not runs:
        L += ["- its recipe, read from its case: the commands that make a case like it", ""]
        L += _recipe_commands(ctl_recipe)
    elif ctl_recipe:
        L.append(f"- its recipe, read from its case, is `{preview.get('control_recipe')}`; "
                 f"each run below is that recipe with the run's changes")
    L.append("")

    for e in runs:
        L += _run_lines(e, run_dir, ctl_recipe, preview.get("checks_in_words"))

    ic = preview.get("inputs_check") or {}
    if first or ic:
        L += ["## The input files every run reads" if runs else
              "## The input files the control read", ""]
        if not ic:
            # no run, so no input check: the files as the control's recipe names them
            dom = {str(e[0]): str(e[1]) for e in first.get("xml") or []}
            nl = dict(_nl_files(first))
            L.append(f"- surface data {_quote_path(nl.get('fsurdat'))}")
            if dom.get("LND_DOMAIN_FILE"):
                L.append(f"- grid (domain) file `{dom.get('LND_DOMAIN_PATH', '')}/"
                         f"{dom['LND_DOMAIN_FILE']}`")
            L.append(f"- starting state {_quote_path(nl.get('finidat'))}")
        if ic:
            L.append(f"- surface data {_quote_path(ic.get('surface'))}")
            L.append(f"- grid (domain) file {_quote_path(ic.get('domain'))}")
            L.append(f"- starting state {_quote_path(ic.get('restart'))}")
            c = ic.get("checks") or {}
            L.append(f"- the three checked as one set (`{ic.get('path')}`): "
                     + ("passed" if ic.get("pass") else "FAILED")
                     + (f"; {c['active_cells']:,} cells" if isinstance(c.get("active_cells"), int)
                        else "")
                     + (f"; the starting state's subgrid matches the surface data's"
                        if c.get("layout_matches") else ""))
        named = [(k, p) for k, p in _nl_files(first) if k not in ("fsurdat", "finidat")]
        if named:
            L.append("- other files the land namelist names: "
                     + "; ".join(f"{k} `{p}`" for k, p in named))
        for s in preview.get("forcing") or []:
            y = [x for x in s.get("years") or [] if x is not None]
            span = (f"years {y[1]}-{y[2]}, aligned to model year {y[0]}" if len(y) == 3
                    else "years " + " ".join(map(str, y)))
            if s.get("error"):
                L.append(f"- weather stream `{s['stream']}` ({span}): not readable: {s['error']}")
                continue
            L.append(f"- weather stream `{s['stream']}` ({span}): {s.get('n_files')} file(s), "
                     f"`{s.get('first')}`"
                     + (f" to `{s.get('last')}`" if s.get("last") != s.get("first") else "")
                     + f", in `{s.get('data_dir')}`, on grid `{s.get('domain')}`; fields "
                     + ", ".join(s.get("fields") or []))
        L.append("")

    an = preview.get("analysis") or {}
    de = dict(brief.get("domain_experiment") or {})
    de.update(strategy.get("domain_experiment") or {})
    plan_an = de.get("analysis") or {}
    if an or plan_an:
        L += ["## The numbers the answer is read from", ""]
        if de.get("years"):
            L.append(f"- model years: {', '.join(map(str, de['years']))}")
        if plan_an.get("normal_years"):
            L.append(f"- normal years: {', '.join(map(str, plan_an['normal_years']))}")
        if plan_an.get("question_type"):
            L.append("- " + question_type_line(plan_an))
        if an.get("regions"):
            L += ["", "| region | cells | area (km²) | rule | drawn from |", "|---|---|---|---|---|"]
            L += [f"| {r['name']} | {r['cells']:,} | {r['area_km2']:,} | {r['rule']} | "
                  f"{', '.join(r.get('sources') or [])} |" for r in an["regions"]]
        if an.get("fields"):
            # A MONTH'S AND A SEASON'S VALUE, AS THE SERVER SAYS (2026-11-03):
            # each field's how and units for a month and a season come from the
            # server's analysis plan, and so does the line on the periods; a
            # plan written before it carried them keeps the page's own line.
            per = any(f.get("by_month") for f in an["fields"])
            if per:
                L += ["", "| field | read from | a year's value | units | a month's value | "
                          "a season's value | note |", "|---|---|---|---|---|---|---|"]
                L += [f"| {f['name']} | {f.get('stream') or 'the field above'} | {f['how']} | "
                      f"{f.get('units') or ''} | {_period_cell(f, 'month')} | "
                      f"{_period_cell(f, 'season')} | {f.get('note') or ''} |"
                      for f in an["fields"]]
            else:
                L += ["", "| field | read from | a year's value | units | note |",
                      "|---|---|---|---|---|"]
                L += [f"| {f['name']} | {f.get('stream') or 'the field above'} | {f['how']} | "
                      f"{f.get('units') or ''} | {f.get('note') or ''} |" for f in an["fields"]]
            L += ["", _periods_line(an.get("periods")) or MONTHS_AND_SEASONS_LINE]
        for p in an.get("problems") or []:
            L.append(f"- cannot be made: {_trim(p, 300)}")
        L.append("")
    return L


def render(reception, strategy, run_name="this run", preview=None, run_dir=None):
    """reception.json + strategy.json (+ the server's preview) -> markdown."""
    brief = reception.get("brief") or reception
    kind, design = _design_lines(brief, strategy)
    L = []
    L.append(f"# Design review — {run_name}")
    L.append("")
    if preview:
        # the server's records join the two the page always quoted
        L.append(f"*Rendered from `reception.json`, `strategy.json` and what the model "
                 f"server wrote before this review (`{PREVIEW_FILE}`); those records are "
                 f"the authority, and this page only quotes them. "
                 f"{datetime.now().isoformat(timespec='seconds')}*")
    else:
        L.append(f"*Rendered from `reception.json` and `strategy.json` — those "
                 f"records are the authority; this page only quotes them. "
                 f"{datetime.now().isoformat(timespec='seconds')}*")
    L.append("")
    L.append("## The question")
    # THE QUESTION, THEN EACH CHANGE ASKED FOR AT THIS REVIEW. A revision
    # appends its change to the request; quoted as one 400-character block, a
    # second change fell off the end of the page (2026-10-27).
    question, *changes = str(reception.get("user_request")
                             or brief.get("user_request") or "").split(REVISION_MARK)
    L.append(f"> {_trim(question, 400)}")
    if changes:
        L.append("")
        L.append("Changes you asked for at this review:")
        L.extend(f"{i}. {_trim(c, 200)}" for i, c in enumerate(changes, 1))
    L.append("")
    model = brief.get("model")
    read_as = f"Read as a **{kind}**"
    if model:
        read_as += f" for **{model.upper()}**"
    L.append(read_as + ".")
    if brief.get("model_rationale"):
        # 420, NOT 260: reception writes two or three sentences here and the
        # old cap cut the second one mid-thought on every live run. With the
        # sentence-aware _trim, a typical rationale now survives whole and a
        # long one ends at a full stop.
        L.append(f"Why this model: {_trim(brief['model_rationale'], 420)}")
    L.append(f"- **Where:** {_domain_line(brief)}")
    L.append(f"- **When:** {_period_line(brief)}")
    L.append("")
    L.append("## The design")
    L.extend(design or ["- (the strategy records no design lines)"])
    L.append("")
    L.append("## Why this can answer it")
    for g in (strategy.get("goals") or [])[:4]:
        L.append(f"- {_trim(g, 240)}")
    fz = strategy.get("feasibility") or {}
    verdict = str(fz.get("verdict") or "unstated")
    L.append("")
    L.append(f"**Feasibility: {verdict}.** {_trim(fz.get('why'), 320)}")
    L.append("")
    gaps = (brief.get("known_data_gaps") or [])
    conflicts = ((brief.get("run_settings") or {}).get("conflicts") or [])
    assumptions = (brief.get("assumptions") or [])
    if gaps or conflicts or assumptions:
        L.append("## Assumed or missing — decided without you, and recorded")
        for item in (list(assumptions) + list(conflicts))[:5]:
            L.append(f"- {_trim(item, 240)}")
        for item in gaps[:3]:
            L.append(f"- data gap: {_trim(item, 200)}")
        L.append("")
    L.extend(_check_lines(preview))
    L.append("## What happens next")
    whole = str(brief.get("design_archetype") or strategy.get("archetype") or "") \
        .lower() == "whole_domain"
    runs = ((strategy.get("domain_experiment") or brief.get("domain_experiment")
             or {}).get("runs") or [])
    built = bool(preview) and not preview.get("refused") and not preview.get("error")
    if preview and not built:
        L.append("Nothing can run until the design changes: type the change at the "
                 "prompt, and the design is made again and checked again. Declining "
                 "costs nothing: the records above stay on disk.")
    elif whole and runs and built:
        L.append("On a yes, the Experiment Manager has the server check the design and "
                 "build every run again from the same records, then submits each run as "
                 "a chain of one-year jobs, each year checked before the next starts, as "
                 "the second part of this page shows. When the chains end, the numbers by "
                 "region, year, month and season are written from the runs' own records, "
                 "and the "
                 "Analyzer interprets them against the question, under the caveats above. "
                 "Declining here costs nothing: nothing has been submitted, and the "
                 "records stay on disk.")
    elif whole and not runs:
        # THE CONTROL ALONE: nothing is built or run, so the chains below
        # would describe work that never happens (2026-10-27).
        L.append("Nothing is run: the control's existing output is read, the "
                 "numbers by region, year, month and season are written from it, "
                 "and the "
                 "Analyzer interprets them against the question, under the "
                 "caveats above. Declining here costs nothing: the records "
                 "above stay on disk.")
    elif whole:
        L.append("The Experiment Manager has the model's server check this "
                 "experiment against its change list and the control's facts, "
                 "and the input set, before any compute; then each run is built "
                 "as its own case and run as a chain of one-year jobs, each year "
                 "checked before the next starts. When the chains end, the numbers "
                 "by region, year, month and season are written from the runs' own "
                 "records, and "
                 "the Analyzer interprets them against the question, under the "
                 "caveats above. Declining here costs nothing: the records above "
                 "stay on disk.")
    else:
        L.append("The Experiment Manager re-checks this design against the data "
                 "record before any compute (it stops on a design that cannot be "
                 "placed and records what it corrects), then builds real columns "
                 "and runs them. Declining here costs nothing — the records above "
                 "stay on disk.")
    L.append("")
    L.append("---")
    L.append(_PENDING)
    L.append("")
    L.extend(_appendix(preview, run_dir, strategy, brief))
    return "\n".join(L)


# A PREVIEW GIVEN AS NONE MEANS NONE (2026-11-03). The record on disk is read
# only when no preview is passed at all (a page redrawn from a run directory).
# The coordinator always passes its own, so a page is never drawn from an
# older refused record while the terminal offers a yes for the design now.
_NOT_GIVEN = object()


def write_review(run_dir, reception=None, strategy=None, preview=_NOT_GIVEN):
    """Render and write DESIGN_REVIEW.md into run_dir; returns its path. The
    server's preview is read from 01_inputs/design_preview.json only when the
    argument is left out (a study the server previews nothing for has none)."""
    rd = Path(run_dir)
    if reception is None:
        reception = json.loads((rd / "reception.json").read_text())
    if strategy is None:
        strategy = json.loads((rd / "strategy.json").read_text())
    if preview is _NOT_GIVEN:
        preview = _load(rd, PREVIEW_FILE)
    path = rd / REVIEW_NAME
    path.write_text(render(reception, strategy, run_name=rd.name, preview=preview,
                           run_dir=rd))
    return path


def decision_part(text):
    """The page down to its second part, for the terminal: the decision rests
    on it, and the recipe below it is long. (text, True) when a second part
    was cut off."""
    head, sep, _ = text.partition("\n---\n\n" + APPENDIX_MARK)
    return head.rstrip() + "\n", bool(sep)


def record_decision(run_dir, decision):
    """Replace the pending line with what was actually decided."""
    path = Path(run_dir) / REVIEW_NAME
    if not path.is_file():
        return None
    stamp = datetime.now().isoformat(timespec="seconds")
    text = path.read_text().replace(
        _PENDING, f"**Decision:** {decision} at {stamp}.")
    path.write_text(text)
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} RUN_DIR")
    print(write_review(sys.argv[1]))
