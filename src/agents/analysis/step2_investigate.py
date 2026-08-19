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
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.analysis import script_runner as _runner   # noqa: E402

FILENAME = "investigation.json"

# THE SAME ROUND, WRITTEN FOR A READER. investigation.json is the machine
# record: findings, results, script paths. This is the plan behind it — how the
# user's question was mapped onto ELM variables, which figures were chosen,
# what each was meant to answer, and what came back. It is RENDERED from the
# structured record rather than written by the model, so it cannot disagree
# with investigation.json; the only prose in it is the two fields the model
# fills in (`reasoning` and each figure's `why`).
#
# It goes to step 3 as evidence about the INVESTIGATION, not about the run: a
# reviewer that sees only the findings cannot tell a thin figure set that is
# thin because the run is thin from one that is thin because the plan was.
PLAN_MD = "investigation_plan.md"

MAX_PLOTS = 5

# The gateway serves claude-opus-5-project; the rest of the framework defaults
# to claude-opus-4-8-project. Step 2 is the most judgement-heavy call in the
# pipeline, so it takes the stronger model rather than inheriting the default.
DEFAULT_MODEL = "claude-opus-5-project"



def preflight_of(ctx) -> Dict[str, Any]:
    """ctx.preflight(), or {} for a context-like object that has none.

    FAILS OPEN, deliberately. Preflight is a diagnostic and a cost saving, not
    a safety check: with no information the gate does not gate and the filter
    does not filter, which is exactly the behaviour that existed before it. A
    context-shaped stub — a test, a standalone CLI, a future backend building
    its own — must not lose the whole Analyzer to a missing optional method.
    """
    fn = getattr(ctx, "preflight", None)
    try:
        return fn() if callable(fn) else {}
    except Exception:                                           # noqa: BLE001
        return {}

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

    field_semantics is keyed by DERIVED METRIC (water_budget, precip_mm_yr),
    not by raw variable, and each entry records its `from` list. That mapping is
    the framework's own record of what a metric means, and it is where I
    personally got `runoff_fraction` wrong — assuming it was a fraction of
    precipitation when it was QOVER/(QCHARGE+QOVER). That metric was deleted on
    2026-08-13 for exactly that reason; every surviving ratio carries its
    denominator in its name (`runoff_frac_of_P`). Showing the derivation, not
    just the name, is what stops the next one.
    """
    units = _frame_units(ctx)
    sem = (ctx.data.get("field_semantics") or {})

    # WHICH FRAME THIS RUN ACTUALLY HAS. A backend without a daily series gets
    # `df = None`, and describing `df` anyway would send the model to write
    # scripts against it — the failure would surface as an AttributeError on
    # None inside generated code rather than as anything a reader could act on.
    out: List[str] = []
    if ctx.series() is not None:
        out += ["  variables in `df` (one row per day; units exactly as the "
                "frame's",
                "  `units` column reports them — already daily, do not "
                "rescale):"]
        for v in sorted(units):
            out.append(f"    {v:10s} {units[v]}")
    else:
        out.append("  `df` is None for this run — it has NO daily series. Do "
                   "not use df.")

    # WHAT IS DELIBERATELY ABSENT, and why. Silence here reads as "the variable
    # is not in this run", and the model plans around an absence perfectly
    # well. It does NOT plan around a variable it believes is present: told
    # nothing, two rounds of figures at Brandywine went into re-deriving soil
    # moisture from a variable that cannot be in a daily frame at all, and both
    # were correctly rejected by the reviewer as physically impossible.
    withheld = getattr(ctx, "series_withheld", None) or {}
    broken = {k: v for k, v in withheld.items() if not v.get("in_frame")}
    if broken:
        out += ["",
                "  NOT in any frame — the packaged block is malformed. Do not "
                "use these:"]
        for v in sorted(broken):
            out.append(f"    {v:10s} {broken[v]['why']}")

    soil = getattr(ctx, "soil", lambda: None)()
    if soil is not None:
        out += ["",
                "  `soil` — the SOIL COLUMN THROUGH TIME for this run:",
                "    entity | date | layer | depth_m | thickness_m | "
                "depth_top_m | depth_bottom_m |",
                "    variable | value | units | source",
                "    variables: " + ", ".join(
                    sorted(str(v) for v in soil["variable"].unique())),
                f"    {soil['layer'].nunique()} layers per column"]
        # THE GEOMETRY IS DESCRIBED ONLY IF IT IS THERE. DZSOI is in every h0
        # file this pipeline writes, but a run extracted before 2026-08-13 has
        # no thickness — and a catalog that states a weighting rule for a
        # column the frame cannot weight sends the model to write code that
        # divides by nothing.
        geom = (soil[soil["entity"] == soil["entity"].iloc[0]]
                .drop_duplicates("layer").sort_values("layer"))
        thick = geom["thickness_m"].dropna()
        bottom = geom["depth_bottom_m"].dropna()
        if len(thick) and len(bottom):
            active = bottom[bottom <= 3.81]
            out += [
                f"    {thick.min():.4f} m thick at the surface to "
                f"{thick.max():.2f} m at the bottom, "
                f"{bottom.max():.2f} m in total",
                "",
                "    A COLUMN MEAN IS THICKNESS-WEIGHTED. The layers differ by "
                "three orders of",
                "    magnitude, so an unweighted mean over `layer` reports the "
                "bedrock:",
                "        (g['value'] * g['thickness_m']).sum() / "
                "g['thickness_m'].sum()",
                "",
                f"    The HYDROLOGICALLY ACTIVE soil is depth_bottom_m <= "
                f"{active.max():.4f} (the first {len(active)} layers).",
                "    Below that ELM diagnoses the water table from an aquifer "
                "store rather than",
                "    simulating it, which is what the water-table caveat is "
                "about. A soil-water",
                "    claim about the whole column is not a claim about the "
                "soil."]
        else:
            out += ["    NO LAYER THICKNESSES in this run's package, so a "
                    "column mean cannot be",
                    "    weighted. Report per-layer or per-depth, never a mean "
                    "over `layer`."]

    # getattr, because `profiles` is newer than this function's other callers:
    # a context object without it has no depth data by definition, which is
    # the same answer as a context whose profiles() returns None.
    prof = getattr(ctx, "profiles", lambda: None)()
    if prof is not None:
        times = sorted(prof["time_y"].dropna().unique().tolist())
        out += ["",
                "  `prof` — the DEPTH frame for this run:",
                "    entity | time_y | depth_m | variable | value | units | "
                "source",
                f"    output times (years): {times}",
                f"    depth_m is positive DOWNWARD from the surface, "
                f"{prof['depth_m'].max():.1f} m at the deepest",
                "    variables: " + ", ".join(
                    sorted(str(v) for v in prof["variable"].unique())),
                "    Each column has its OWN depth grid — column depth is "
                "capped per column, so",
                "    do not assume a shared depth axis across entities; group "
                "by entity first."]

    if sem:
        # WHERE EACH DECLARED FIELD LIVES, read off the rows rather than
        # assumed. ELM's semantics describe derived metrics that sit in
        # columns[i]['metrics']; PFLOTRAN's describe the variables of `prof`
        # and the INPUTS carried on the row itself (water_table_m,
        # unsaturated_m, transient ...). Telling the model that water_table_m
        # is in `metrics` sends it to a key that is empty. Entries whose name
        # starts with "_" are notes about the run — what was NOT computed, how
        # it was forced — and are printed as such.
        rows = [c for c in ctx.columns if isinstance(c, dict)]
        in_metrics = {k for c in rows for k in (c.get("metrics") or {})}
        on_row = {k for c in rows for k, v in c.items() if v is not None}
        prof_vars = set()
        if prof is not None:
            prof_vars = {str(v) for v in prof["variable"].unique()}
        notes, fields = [], []
        for k in sorted(sem):
            e = sem[k] or {}
            if k.startswith("_"):
                notes.append((k, e)); continue
            if k in prof_vars:
                where = "a variable of `prof`"
            elif k in in_metrics:
                where = "columns[i]['metrics'][k]"
            elif k in on_row:
                where = "columns[i][k]  (an INPUT to the column, not an output)"
            elif k in ("depth_m", "times_y", "time_y", "date"):
                where = "an axis of the frame"
            else:
                where = "declared but on NO row of this run — do not use"
            fields.append((k, e, where))
        if fields:
            out += ["", "  declared fields (units, where they live, meaning) — "
                        "ALREADY UNIT-CORRECT; prefer these over re-deriving:"]
            for k, e, where in fields:
                # THE DERIVATION STAYS ON THE LINE: runoff_fraction is
                # QOVER/(QCHARGE+QOVER), not a fraction of P, and reading the
                # name instead of the derivation is how that was once got wrong.
                frm = ", ".join(e.get("from") or []) or "?"
                out.append(f"    {k:22s} {str(e.get('units')):10s} from {frm} — {where}")
                if e.get("note"):
                    out.append(f"    {'':22s} {str(e['note'])[:220]}")
        for k, e in notes:
            out += ["", f"  {k.strip('_').upper()}: {str(e.get('note') or e)[:400]}"]
            if e.get("fields"):
                out.append(f"    not computed: {', '.join(e['fields'])}")
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


def _treatment_labels(ctx) -> List[str]:
    """The sweep's factor levels, named, and the instruction to use the names.

    Empty on a site run, which varies nothing deliberately and has no
    treatments to name.

    WHAT THIS IS FOR. `treatment` is a nested dict, and a script told only that
    the key exists prints the dict — which on the 2026-08-16 sweep produced
    four x-axis ticks reading `{'soil_texture': 5, 'prescribed_weather':
    {'fill': 'scale', 'values': {'PRECTmms': 2.0}}}` and a figure whose bars
    were squeezed into one corner. The label already exists on the row by then;
    all that was missing was saying so.
    """
    seen: Dict[str, str] = {}
    for c in ctx.columns:
        lab = c.get("treatment_label")
        if lab:
            seen[str(c.get("case_name") or c.get("scenario_name"))] = str(lab)
    if not seen:
        return []
    out = ["THIS IS A CONTROLLED SWEEP. Each column is one combination of the",
           "factor levels below. `treatment_label` on the row is the SHORT NAME",
           "for that combination — use it verbatim for every legend entry, axis",
           "tick and annotation. Never print the raw `treatment` dict: it is",
           "nested, it is long, and it wrecks the layout of the figure it labels.",
           "",
           "A label is a NAME, not a layout. These names are words, not codes,",
           "so give them room: rotate the ticks, wrap at the comma, or move them",
           "into a legend. Two labels that overlap are two labels nobody can",
           "read, and an annotation printed over a bar hides the bar.",
           ""]
    for name, lab in seen.items():
        out.append(f"    {name:10s} {lab}")
    out.append("")
    return out


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

    model_name = str((ctx.data or {}).get("model") or "model").upper()
    lines += [
        "",
        "WHAT THIS EXPERIMENT IS:",
        f"    {len(ctx.columns)} independent 1-D {model_name} columns, "
        f"elevation {lo:.0f}-{hi:.0f} m." if lo is not None else
        f"    {len(ctx.columns)} independent 1-D {model_name} columns.",
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
    ] + _treatment_labels(ctx)

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

    # WHAT THE PLANNER ALREADY JUDGED, and what the design did not deliver.
    # Both were computed and then read by nobody: `feasibility` has sat in
    # ctx.plan since step 0 was written, and planned_vs_actual() had tests and
    # no production caller. The stage whose job is deciding whether a study can
    # answer the question published a verdict — "streamflow can only be a
    # first-order water-balance comparison, the model has no routing" — and the
    # stage that plans the figures never saw it.
    pre = preflight_of(ctx)
    feas = pre.get("feasibility")
    if isinstance(feas, dict) and feas.get("verdict"):
        lines += [f"THE PLANNER JUDGED THIS STUDY {str(feas['verdict']).upper()}:",
                  f"    {feas.get('why')}",
                  "(This is about the DESIGN, not a result. Do not propose a",
                  " figure it says the run cannot support.)", ""]
    elif isinstance(feas, str) and feas:
        lines += [f"THE PLANNER JUDGED THIS STUDY: {feas}", ""]

    unmet = pre.get("unmet_plan_targets") or []
    if unmet:
        lines.append("PLANNED AND NOT DELIVERED — the design named these and "
                     "the run does not have them:")
        for c in unmet[:8]:
            lines.append(f"    {c.get('claim')}: planned {c.get('planned')}, "
                         f"got {c.get('actual')}  (from {c.get('planned_in')})")
        lines.append("")

    # THE NUMBERS, NOT THE COUNTS (2026-08-12). This block used to print how
    # many entries each step-1 record held — "swe: model=17, observed=2" — and
    # nothing about what they said, so the one part of the analysis that had
    # touched an observation reached the planner of the figures as a pair of
    # list lengths. It now shows the comparison's own summary, rendered by the
    # step that owns the shape.
    if step1:
        from agents.analysis import step1_compare as _cmp
        text = _cmp.format_comparison(step1.get("summary") or {})
        if text:
            lines += ["WHAT THE MODEL-vs-OBSERVATION COMPARISON MEASURED",
                      "(step 1; these are measurements, not verdicts — no",
                      " metric below is a statement that the model is good):",
                      text, ""]

    return "\n".join(lines)


TASK = f"""\
You are a hydrologist analysing this ensemble of 1-D columns — the model is
named in the brief above. Decide which figures answer what the user asked, and
write the code that draws them.

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
              None when this run has no daily series — the catalog above says
              which frames exist. Use only the ones it lists.
    prof      tidy DEPTH frame: entity | time_y | depth_m | variable | value |
              units | source. None when this run has no depth output.
    soil      tidy SOIL frame: entity | date | layer | depth_m | thickness_m |
              depth_top_m | depth_bottom_m | variable | value | units | source.
              One row per day per layer. None when the run has no layered
              output. Any mean over layers must be weighted by thickness_m.
    columns   list of per-column metadata dicts (keys listed above)
    caveats   the caveat records
    plt, np, pd, out_path
It MUST save the figure to out_path, and MUST assign a dict named `result`
containing the numbers it plotted plus an integer `n` = how many data points
the figure rests on. A result with n below 3, or with no n, is rejected.

Return ONLY JSON:
{{"reasoning": "<2-5 sentences: which of this model's variables you decided
                constitute the process the user asked about, and why those and
                not others.
                Name anything you considered computing and rejected, with the
                reason. This is read by the reviewer, so say what you actually
                decided rather than restating the question.>",
  "notes": "<what you could and could not address, and why>",
  "figures": [
    {{"id": "snake_case_name",
      "question": "<the question this figure answers>",
      "why": "<one sentence: what this figure adds that the others do not>",
      "scale": "overall" | "places",
      "variables": ["model variable names used"],
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
    # `prompt` travels back beside `raw` so the caller can write the PAIR.
    # What the model was shown is half the record: the worst bug this pipeline
    # has had was a silent cap on the evidence, invisible because nobody could
    # see what was sent.
    return {"notes": spec.get("notes"), "reasoning": spec.get("reasoning"),
            "figures": kept[:MAX_PLOTS], "raw": reply, "prompt": prompt}


# ─────────────────────────────────────────────────────────────────────
# THE PLAN, AS A DOCUMENT
# ─────────────────────────────────────────────────────────────────────
def render_plan_md(ctx, spec: Dict[str, Any], findings: List[Dict[str, Any]],
                   caveats: List[Dict[str, Any]], round_no: int,
                   feedback=None) -> str:
    """The round as Markdown: what was planned, why, and what came back.

    DERIVED, NOT DICTATED. Every outcome line is read off the same `findings`
    and `caveats` this round produced, so the document cannot claim a figure
    was drawn that was not. The model contributes exactly two prose fields —
    `reasoning` and each figure's `why` — and both are clearly attributed.

    A proposal that produced nothing still appears, with its error. A plan that
    lists only what worked is a plan rewritten after the fact.
    """
    by_id = {f["id"]: f for f in findings}
    failed = {c["id"].split(":", 1)[-1]: c for c in caveats
              if str(c.get("id", "")).startswith("figure_failed:")}
    figs = spec.get("figures") or []

    L = [f"# Investigation plan — round {round_no}", "",
         "*Rendered from `investigation.json`. The reasoning and the per-figure "
         "*why* are the model's words; every outcome below is measured.*", "",
         "## The question", "",
         f"> {(ctx.plan or {}).get('question')}", ""]

    if feedback:
        L += ["## What the reviewer asked for after the previous round", "",
              f"{feedback}", ""]

    if spec.get("reasoning"):
        L += ["## How the question was mapped onto the model's variables", "",
              str(spec["reasoning"]), ""]

    L += [f"## Analysis steps ({len(figs)} proposed, {len(findings)} produced "
          f"a result)", ""]

    for i, f in enumerate(figs, 1):
        fid = f.get("id")
        got = by_id.get(fid)
        L += [f"### {i}. `{fid}` — {f.get('scale') or 'unspecified scale'}", "",
              f"**Answers.** {f.get('question')}", ""]
        if f.get("why"):
            L += [f"**Why this one.** {f['why']}", ""]
        v = ", ".join(f"`{x}`" for x in (f.get("variables") or [])) or "—"
        L += [f"**Model variables read.** {v}", ""]
        if got:
            L += [f"**Outcome.** Drew `{Path(got['figure']).name}` "
                  f"from `{Path(got['script']).name}` "
                  f"— {got.get('n')} data points.", ""]
            if got.get("blocked_by"):
                L += ["**Falls inside a blocking caveat.** "
                      + ", ".join(f"`{c}`" for c in got["blocked_by"])
                      + " — a claim from this figure must carry the id.", ""]
        elif fid in failed:
            L += [f"**Outcome — no result.** "
                  f"{failed[fid].get('statement')}", ""]
        else:
            L += ["**Outcome — no result**, and no reason was recorded. "
                  "That is a defect in this renderer's inputs, not a "
                  "property of the run.", ""]

    if spec.get("notes"):
        L += ["## What this round could not address", "",
              str(spec["notes"]), ""]

    return "\n".join(L)


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

    # WHAT THIS RUN ACTUALLY HOLDS, asked once. The filter below refuses a
    # figure whose variables are in no frame BEFORE spending a subprocess on
    # it — and, more usefully, says exactly which name was missing. The same
    # figure failing inside the runner returns "KeyError" from somewhere in
    # generated pandas, which is a far worse thing to hand to round 2.
    pre = preflight_of(ctx)
    withheld = pre.get("withheld") or {}
    # FAILS OPEN, AND THAT TAKES A LINE TO GET RIGHT. `set(x or [])` collapses
    # "no inventory available" and "an inventory that is empty" into the same
    # empty set — and then EVERY figure naming any variable is refused, because
    # none of them are in it. That is the opposite of the intent, and it is
    # what a context without preflight() got until this was caught. An absent
    # or empty inventory means the filter does not know, so it does not judge.
    have = set(pre.get("variables") or ())
    can_filter = bool(have)

    for f in spec["figures"]:
        fid = f["id"]
        wanted = [str(v) for v in (f.get("variables") or []) if v]
        missing = [v for v in wanted if v not in have]
        if can_filter and wanted and len(missing) == len(wanted):
            # EVERY variable it named is absent. A figure missing one of five
            # may still be worth drawing; one missing all of them cannot be.
            why = "; ".join(f"{v}: {withheld[v]}" for v in missing
                            if v in withheld) or \
                  f"not present in any frame of this run"
            caveats.append({
                "id": f"figure_failed:{fid}",
                "severity": "context",
                "statement": (f"The planned figure '{fid}' "
                              f"({f.get('question')}) was not run: it reads "
                              f"{', '.join(missing)}, and {why}. Available "
                              f"variables: {', '.join(sorted(have)) or 'none'}."),
                "applies_to": "completeness of the step 2 figure set",
                "source": "step2_investigate:preflight"})
            continue

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

    # THE PLAN, WRITTEN DOWN. Rendered after the figures run, so each step
    # carries its outcome rather than its intention. Step 3 reads this file —
    # see review_brief — which is why the path is recorded rather than the text:
    # step 3 can then be re-run alone against an archived study and still see
    # what step 2 was trying to do.
    plan_md = out_dir / PLAN_MD
    plan_md.write_text(render_plan_md(ctx, spec, findings, caveats,
                                      round_no, feedback))

    # VERBATIM, BESIDE THE PARSED RECORD. investigation.json holds what the
    # reply became; these hold what it was. See script_runner.save_exchange.
    exchange = _runner.save_exchange(out_dir, "step2", round_no,
                                     spec.get("prompt"), spec.get("raw"))

    out = {"round": round_no, "notes": spec.get("notes"),
           "reasoning": spec.get("reasoning"),
           "findings": findings, "caveats": caveats,
           "figures": [f["figure"] for f in findings],
           "plan_md": str(plan_md),
           "exchange": exchange or None,
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
