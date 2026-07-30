#!/usr/bin/env python3
"""
Analyzer step 2 — decide what to plot, then plot it
src/agents/analysis/step2_investigate.py

    in   ctx (plan, data, caveats) + the step-1 comparison records
    out  {findings, caveats, figures} — at most MAX_PLOTS of each

WHAT THE LLM DECIDES, AND WHAT IT DOES NOT.

    decides   which ELM variables constitute the process the user asked about,
              which places are worth featuring, what is worth NOT computing,
              and what this experiment cannot answer
    does not  produce a single number. Every value comes from pandas, run by
              script_runner in a subprocess. The model writes the analysis; the
              machine does the arithmetic.

That split is why step 3 can review this at all. If step 2 emitted prose,
nothing downstream could check a claim against the data it came from.

THE MAPPING IS THE JOB. "Recharge" is QCHARGE, QDRAI and SOILLIQ; "snowmelt" is
H2OSNO, its daily decrease, and what QOVER does with it; "water availability" is
something else again. That translation is real domain knowledge, it differs for
every question, and no fixed function covers it — which is why step2_derive's
menu of four hardcoded analyses produced the same output whatever was asked.

TWO SCALES, ALWAYS.

    overall   the DISTRIBUTION across sampled points and how it varies along the
              sampling axis — never a basin total. Nineteen unrouted 1-D columns
              do not sum to a watershed: "recharge ranges 12-95 mm/yr, median 31,
              increasing with elevation" is supportable, "the watershed recharges
              31 mm/yr" is not, and the second is the sentence a model writes
              unless the brief forbids it.
    places    per-band exemplars plus any column departing from its band's trend.
              The departure is usually the finding — a mid-elevation band that
              fails to interpolate between the high and low bands is a mechanism
              showing itself.

CAVEATS BIND MECHANICALLY. The blocking ones are quoted into the brief with
their `applies_to` scope, and a finding that lands inside a blocked scope is
tagged whatever the model concluded. Prompting alone would make the caveat
system decorative: an unrespected caveat looks exactly like a respected one.

FIVE PLOTS IS A CEILING, NOT A QUOTA. A thin run supports fewer, and a model
told to produce five will pad to five — which is fishing with extra steps.
"""
import json
import re
from typing import Any, Dict, List, Optional

from agents.analysis import script_runner as _runner   # noqa: E402

FILENAME = "investigation.json"

MAX_PLOTS = 5

# The gateway serves claude-opus-5-project; the rest of the framework defaults
# to claude-opus-4-8-project. Step 2 is the most judgement-heavy call in the
# pipeline, so it takes the stronger model rather than inheriting the default.
DEFAULT_MODEL = "claude-opus-5-project"


# ─────────────────────────────────────────────────────────────────────
# THE BRIEF — deterministic, so it can be asserted on without an API call
# ─────────────────────────────────────────────────────────────────────
def _frame_units(ctx) -> Dict[str, str]:
    """Units as they appear in `df` — the thing the script actually reads.

    NOT ctx.data["variable_units"]. That map describes the RAW ELM variable
    (QOVER in mm/s) and its own docstring says so. The daily series are not
    raw: _daily() resampled 3-hourly output, converted fluxes to mm/day, and
    wrote "mm/day" alongside each series — which step0.series() carries into
    the frame's `units` column.

    Both maps are correct about their own subject. Printing the RAW one under a
    heading about `df` is what broke: the model believed the brief over the
    frame, multiplied a year of already-per-day values by 86400, and plotted
    1.1e6 mm/yr of runoff. The frame had said mm/day the whole time, and so had
    the per-series block it came from. There was never a bug in the data.
    """
    df = ctx.series()
    if df is None or "variable" not in getattr(df, "columns", []):
        return dict(ctx.data.get("variable_units") or {})
    out: Dict[str, str] = {}
    for v, u in df[["variable", "units"]].drop_duplicates().values:
        out.setdefault(str(v), str(u))
    return out


def _variable_catalog(ctx) -> List[str]:
    """Frame variables, then the derived metrics and what they come FROM.

    field_semantics is keyed by DERIVED METRIC (annual_recharge_mm_yr), not by
    raw variable, and each entry records its `from` list. That mapping is the
    framework's own record of what a metric means, and it is where I personally
    got runoff_fraction wrong — assuming it was a fraction of precipitation when
    it is QOVER/(QCHARGE+QOVER). Showing the derivation, not just the name,
    stops the model repeating that.
    """
    units = _frame_units(ctx)
    sem = (ctx.data.get("field_semantics") or {})

    out = ["  variables in `df` (one row per day; units exactly as the frame's",
           "  `units` column reports them — already daily, do not rescale):"]
    for v in sorted(units):
        out.append(f"    {v:10s} {units[v]}")

    if sem:
        out.append("")
        out.append("  derived per-column metrics (in columns[i]['metrics']) — "
                   "ALREADY UNIT-CORRECT. Prefer these over re-deriving from df:")
        for k in sorted(sem):
            e = sem[k] or {}
            frm = ", ".join(e.get("from") or []) or "?"
            out.append(f"    {k:26s} {str(e.get('units')):8s} from {frm}")
    return out


def _live_metadata_keys(ctx) -> List[str]:
    """Metadata keys that are populated on at least one column.

    A key that is None everywhere is worse than absent: `soil` sits beside
    `soil_profile`, is null on all 19 rows, and is the exact field the old
    soil_attribution read before returning {} for the study's entire history.
    Advertising it invites a generated script to read it and get nothing.
    """
    live = set()
    for c in ctx.columns:
        for k, v in c.items():
            if v is not None:
                live.add(k)
    return sorted(live)


def _elevation_span(ctx):
    es = [c.get("elevation_m") for c in ctx.columns
          if isinstance(c.get("elevation_m"), (int, float))]
    return (min(es), max(es)) if es else (None, None)


def context_brief(ctx, step1: Optional[Dict[str, Any]] = None) -> str:
    """Everything the model is allowed to reason from, as text.

    Built from ctx alone so it works for any US watershed — nothing here names
    a basin, a variable set, or a station network. A run with no SNOTEL in the
    basin simply carries a caveat saying so, and the model plans around it.
    """
    plan = ctx.plan or {}
    lo, hi = _elevation_span(ctx)
    period = plan.get("period") or {}

    blocking = [c for c in (ctx.caveats or []) if c.get("severity") == "blocking"]
    other = [c for c in (ctx.caveats or []) if c.get("severity") != "blocking"]

    lines = [
        "THE USER ASKED:",
        f"    {plan.get('question') or '(no question recorded)'}",
        "",
        "THE PLANNER DECOMPOSED THAT INTO:",
    ]
    for g in (plan.get("goals") or []):
        lines.append(f"    - {g}")

    lines += [
        "",
        "WHAT THIS EXPERIMENT IS:",
        f"    {len(ctx.columns)} independent 1-D ELM columns, "
        f"elevation {lo:.0f}-{hi:.0f} m." if lo is not None else
        f"    {len(ctx.columns)} independent 1-D ELM columns.",
        f"    Period {period.get('yr_start')}-{period.get('yr_end')}.",
        "    No lateral flow, no routing, no run-on: the columns do not exchange",
        "    water. They are a gradient experiment, not a distributed basin model.",
        "    Anything requiring connectivity between columns, basin routing, or a",
        "    basin water balance is OUT OF SCOPE and must not be attempted.",
        "",
        "VARIABLES AVAILABLE (name, units, meaning):",
    ] + _variable_catalog(ctx) + [
        "",
        "PER-COLUMN METADATA (keys on each entry of `columns`):",
        "    " + ", ".join(_live_metadata_keys(ctx)),
        "",
    ]

    if blocking:
        lines.append("BLOCKING CAVEATS — a claim of these kinds must NOT be made:")
        for c in blocking:
            lines.append(f"    [{c.get('id')}] applies to: {c.get('applies_to')}")
            lines.append(f"        {c.get('statement')}")
        lines.append("")
    if other:
        lines.append("OTHER CAVEATS — claims may be made, stated with these attached:")
        for c in other[:12]:
            lines.append(f"    [{c.get('id')}] {str(c.get('statement'))[:180]}")
        lines.append("")

    if step1:
        lines.append("WHAT THE STEP-1 COMPARISONS FOUND:")
        for name, rec in step1.items():
            if not isinstance(rec, dict):
                continue
            bits = []
            for k in ("has_measured_wtd", "n_static_columns", "window"):
                if k in rec:
                    bits.append(f"{k}={rec[k]}")
            for k in ("gauges", "wells", "model", "columns", "fan", "observed"):
                if isinstance(rec.get(k), list):
                    bits.append(f"{k}={len(rec[k])}")
            lines.append(f"    {name}: " + ", ".join(bits))
        lines.append("")

    return "\n".join(lines)


TASK = f"""\
You are a hydrologist analysing this ELM ensemble. Decide which figures answer
what the user asked, and write the code that draws them.

Report the process the user asked about at TWO scales:
  1. overall — the DISTRIBUTION across the sampled columns and how it varies
     along the sampling axis (usually elevation). NEVER a basin total: these are
     unrouted 1-D columns and they do not sum to a watershed.
  2. representative places — per-band exemplars, and any column that departs
     from its band's trend. A departure is usually the interesting result.

Rules:
  - At most {MAX_PLOTS} figures. Fewer is correct if the run supports fewer;
    say why in `notes` rather than padding to {MAX_PLOTS}.
  - Do not propose anything a blocking caveat forbids. If the user's question
    cannot be addressed by this experiment, say so in `notes` and propose only
    what can.
  - Report as-is. No interpretation drawn on the figure, no titles that state a
    conclusion. Axis labels >= 16pt, tick labels >= 13pt.
  - Sanity-check magnitudes before saving. An annual flux in a mountain
    watershed is order 10-2000 mm/yr and a fraction is between 0 and 1. If your
    numbers land orders of magnitude outside that, something is wrong — say so
    in `notes` rather than plotting it.

Each figure's `code` is a Python snippet run with these already bound:
    df        tidy long frame: columns date | entity | variable | value | units | source
              `entity` is the column case_name for model rows.
    columns   list of per-column metadata dicts (keys listed above)
    caveats   the caveat records
    plt, np, pd, out_path
It MUST save the figure to out_path, and MUST assign a dict named `result`
containing the numbers it plotted plus an integer `n` = how many data points
the figure rests on. A result with n below 3, or with no n, is rejected.

Return ONLY JSON:
{{"notes": "<what you could and could not address, and why>",
  "figures": [
    {{"id": "snake_case_name",
      "question": "<the question this figure answers>",
      "scale": "overall" | "places",
      "variables": ["ELM variable names used"],
      "code": "<python>"}}
  ]}}
"""


# ─────────────────────────────────────────────────────────────────────
# PROPOSE + EXECUTE
# ─────────────────────────────────────────────────────────────────────
def _parse(reply: str) -> Dict[str, Any]:
    """The model's JSON, tolerating a fenced block around it."""
    m = re.search(r"```(?:json)?\s*(.+?)```", reply, re.S)
    text = m.group(1) if m else reply
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in the reply")
    blob = text[i:j + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # The `code` field carries generated Python — raw newlines inside
        # string literals, echoed comments, trailing commas — which naive
        # json.loads rejects. A live run died here on a reply that was
        # otherwise fine. llm_agent already solved this for the planner, so
        # reuse its DETERMINISTIC repairs rather than write a third parser.
        # Not parse_json_resilient: that adds an LLM self-repair round, and a
        # step that silently spends another call to fix its own output is a
        # step whose cost accounting lies.
        from agents.llm_agent import LLMAgent
        repaired = LLMAgent._escape_raw_newlines(
            LLMAgent._strip_json_comments(blob))
        return json.loads(repaired)


def propose(ctx, step1=None, model: str = DEFAULT_MODEL,
            client=None, feedback=None) -> Dict[str, Any]:
    """Ask for figure specs. Returns {notes, figures} with figures capped.

    `feedback` is step 3's verdict from the previous round: what it judged
    unsupported, unreadable or unanswered. It is appended AFTER the task so the
    stable part of the prompt stays byte-identical between rounds.
    """
    if client is None:
        from agents.llm_agent import SimpleLLMClient
        client = SimpleLLMClient(model=model)
        client.label = "step2_investigate"      # so step 4 can attribute the spend
    prompt = context_brief(ctx, step1) + "\n" + TASK
    if feedback:
        prompt += ("\n\nA PREVIOUS ROUND OF THESE FIGURES WAS REVIEWED AND "
                   "JUDGED INSUFFICIENT. Address this and propose a revised "
                   "set — keep what worked, do not simply re-send it:\n"
                   + str(feedback))
    reply = client.ask([{"role": "user", "content": prompt}])
    spec = _parse(reply)
    figs = [f for f in (spec.get("figures") or []) if isinstance(f, dict)]

    seen, kept = set(), []
    for f in figs:
        fid = str(f.get("id") or "").strip() or f"figure_{len(kept)+1}"
        if fid in seen:                      # a duplicate id would overwrite
            continue                         # the earlier figure's PNG
        seen.add(fid)
        kept.append(dict(f, id=fid))
    return {"notes": spec.get("notes"), "figures": kept[:MAX_PLOTS],
            "raw": reply}


def _blocked_by(variables, caveats) -> List[str]:
    """Blocking caveats whose SCOPE names a variable this figure used.

    Mechanical, not model-declared: asking step 2 to tag its own figures makes
    the tag a claim rather than a check, and the reason caveats are uniform
    records is that an unrespected one must not look like a respected one.

    MATCHED AGAINST `applies_to` ONLY, on word boundaries. A first version
    searched the full statement too, and struck every claim in a live run:
    limitation_structural_3 is scoped to the water table but mentions QDRAI in
    passing ("ELM parameterizes lateral losses (QDRAI)"), so every drainage
    figure inherited a water-table caveat. An audit that strikes everything
    tells you nothing — it is indistinguishable from an audit that is broken,
    which is what it was.

    `applies_to` is the field that exists to say what a caveat governs. The
    statement is prose about why.
    """
    used = {str(v) for v in (variables or []) if v}
    if not used:
        return []
    out = []
    for c in (caveats or []):
        if c.get("severity") != "blocking":
            continue
        scope = str(c.get("applies_to") or "")
        # CASE-SENSITIVE. ELM variable names collide with ordinary English:
        # matching SNOW case-insensitively tagged every precipitation figure
        # with "snow (SWE) at stations", and RAIN would do the same. Variable
        # names are uppercase and caveat prose is not, so case is the
        # discriminator that separates "(QOVER)" from "runoff".
        if any(re.search(rf"\b{re.escape(v)}\b", scope) for v in used):
            out.append(c.get("id"))
    return out


def investigate(ctx, out_dir, step1=None, model: str = DEFAULT_MODEL,
                client=None, feedback=None, round_no: int = 1) -> Dict[str, Any]:
    """Propose, execute, and report — including what failed and why.

    A script that fails becomes a CAVEAT, not a silent gap. That is the whole
    lesson of soil_attribution: the analysis that quietly produced nothing was
    indistinguishable, from the outside, from one that was never planned.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = propose(ctx, step1=step1, model=model, client=client,
                   feedback=feedback)
    all_caveats = list(ctx.caveats or []) + list((step1 or {}).get("caveats") or [])
    findings, caveats = [], []

    for f in spec["figures"]:
        fid = f["id"]
        run = _runner.run(f.get("code") or "", ctx,
                          out_dir / f"{fid}.png",
                          script_path=out_dir / f"{fid}.py")
        if run["ok"]:
            findings.append({
                "id": fid, "question": f.get("question"),
                "scale": f.get("scale"), "variables": f.get("variables"),
                "result": run["result"], "figure": run["figure"],
                "script": run.get("script_path"),
                "n": (run["result"] or {}).get("n"),
                "blocked_by": _blocked_by(f.get("variables"), all_caveats)})
        else:
            caveats.append({
                "id": f"figure_failed:{fid}", "severity": "context",
                "statement": (f"The planned figure '{fid}' "
                              f"({f.get('question')}) did not produce a usable "
                              f"result: {run['error']}"),
                "applies_to": "completeness of the step 2 figure set",
                "source": "step2_investigate"})

    out = {"round": round_no, "notes": spec.get("notes"),
           "findings": findings, "caveats": caveats,
           "figures": [f["figure"] for f in findings],
           "n_proposed": len(spec["figures"]), "n_succeeded": len(findings),
           "responded_to_feedback": feedback or None}

    # WRITTEN DOWN, not just returned. A round costs an LLM call plus one
    # subprocess per figure; if step 3 then fails, an in-memory-only result
    # throws all of it away. It is also the provenance record — findings, their
    # n, their figure and the script that drew each one.
    (out_dir / FILENAME).write_text(json.dumps(out, indent=2, default=str))
    return out


def load(out_dir) -> Dict[str, Any]:
    """Read back a previous investigation, or {} if step 2 has not run."""
    from pathlib import Path
    p = Path(out_dir) / FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
