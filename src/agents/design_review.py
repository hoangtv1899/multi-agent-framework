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
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REVIEW_NAME = "DESIGN_REVIEW.md"
_PENDING = "**Decision:** pending — the coordinator records it here."


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
        return cut[: dot + 1]
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


def render(reception, strategy, run_name="this run"):
    """reception.json + strategy.json -> one page of markdown."""
    brief = reception.get("brief") or reception
    kind, design = _design_lines(brief, strategy)
    L = []
    L.append(f"# Design review — {run_name}")
    L.append("")
    L.append(f"*Rendered from `reception.json` and `strategy.json` — those "
             f"records are the authority; this page only quotes them. "
             f"{datetime.now().isoformat(timespec='seconds')}*")
    L.append("")
    L.append("## The question")
    L.append(f"> {_trim(reception.get('user_request') or brief.get('user_request'), 400)}")
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
    L.append("## What happens next")
    L.append("The Experiment Manager re-checks this design against the data "
             "record before any compute (it stops on a design that cannot be "
             "placed and records what it corrects), then builds real columns "
             "and runs them. Declining here costs nothing — the records above "
             "stay on disk.")
    L.append("")
    L.append("---")
    L.append(_PENDING)
    L.append("")
    return "\n".join(L)


def write_review(run_dir, reception=None, strategy=None):
    """Render and write DESIGN_REVIEW.md into run_dir; returns its path."""
    rd = Path(run_dir)
    if reception is None:
        reception = json.loads((rd / "reception.json").read_text())
    if strategy is None:
        strategy = json.loads((rd / "strategy.json").read_text())
    path = rd / REVIEW_NAME
    path.write_text(render(reception, strategy, run_name=rd.name))
    return path


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
