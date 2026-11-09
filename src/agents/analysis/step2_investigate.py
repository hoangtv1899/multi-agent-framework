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
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.analysis import script_runner as _runner   # noqa: E402
from agents.analysis.dashes import strip_dashes, strip_dashes_in          # noqa: E402

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



def _param_text(k, v) -> str:
    """A change's parameter in words: a map as "2016 to 2015", a list as
    "6, 7, 8", rather than a Python literal."""
    if isinstance(v, dict):
        return f"{k} " + ", ".join(f"{a} to {b}" for a, b in v.items())
    if isinstance(v, (list, tuple)):
        return f"{k} " + ", ".join(str(x) for x in v)
    return f"{k} {v}"


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
    if getattr(ctx, "is_domain", False):
        from agents.analysis import analysis_plan as _plan
        if _plan.suite_bearing(ctx):
            return suite_brief(ctx, step1)
        return domain_brief(ctx, step1)
    plan = ctx.plan or {}
    lo, hi = _elevation_span(ctx)
    period = plan.get("period") or {}

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
    return "\n".join(lines + _brief_tail(ctx, step1))


def _brief_tail(ctx, step1: Optional[Dict[str, Any]] = None) -> List[str]:
    """The half of a brief every study shares: the caveats, the planner's
    verdict on the design, what the design named and the run lacks, and what
    the comparison with observations measured."""
    blocking = [c for c in (ctx.caveats or []) if c.get("severity") == "blocking"]
    other = [c for c in (ctx.caveats or []) if c.get("severity") != "blocking"]
    lines: List[str] = []
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
    return lines


def _span(years) -> str:
    from agents.analysis.domain_report import span
    return span(years)


def _period_brief(t: Dict[str, Any]) -> List[str]:
    """MONTHS AND SEASONS (2026-11-03): what `reg_monthly` and `reg_seasonal`
    hold, read off the table, so a question about a season is drawn from the
    season and not from the year. [] for a table without months."""
    runs = [r for r in (t.get("runs") or {}).values() if (r or {}).get("monthly")]
    if not runs:
        return []
    units: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        for v, u in (r.get("period_units") or {}).items():
            units.setdefault(v, u or {})
    lines = ["", "MONTHS AND SEASONS IN `reg_monthly` AND `reg_seasonal` (name, units "
             "over a month, units over a season):"]
    for v, u in units.items():
        lines.append(f"    {v:16s} {str(u.get('month')):16s} {u.get('season')}")
    left = sorted({k for r in runs for k in (r.get("seasons_left_out") or {})})
    if left:
        lines.append(f"    Not made: {', '.join(left)} (the run's output does not hold "
                     f"the December before).")
    return lines


def domain_brief(ctx, step1: Optional[Dict[str, Any]] = None) -> str:
    """The brief for a whole-domain experiment: a control run over a whole
    prepared domain and, for a what-if question, runs that differ from it by
    named changes, reported by region, year, month and season. Built from
    ctx alone, like
    context_brief; what a change means and does not mean is quoted from the
    model server's change list, never written here."""
    plan = ctx.plan or {}
    de = plan.get("experiment") or {}
    t = (ctx.data or {}).get("regional") or {}
    model_name = str((ctx.data or {}).get("model") or "model").upper()
    ctl = de.get("control") or {}
    years = [int(y) for y in (t.get("years") or de.get("years") or [])]
    lines = ["THE USER ASKED:",
             f"    {plan.get('question') or '(no question recorded)'}", "",
             "THE PLANNER DECOMPOSED THAT INTO:"]
    for g in (plan.get("goals") or []):
        lines.append(f"    - {g}")
    lines += ["", "WHAT THIS EXPERIMENT IS:",
              f"    One {model_name} case over the whole prepared domain of the control "
              f"\"{ctl.get('name') or '?'}\", model years {_span(years)}.",
              ("    The control was REUSED: its existing output was read, and nothing "
               "was run for it." if ctl.get("reuse") else
               "    The control was run again from its recorded recipe.")]
    changed = [r for r in (de.get("runs") or []) if isinstance(r, dict)
               and not r.get("reused") and (r.get("run") or r.get("name")) != "control"]
    if changed:
        lines.append("    Runs that differ from the control by named changes, and what "
                     "each difference means, as the model's server states it:")
        for r in changed:
            name = r.get("run") or r.get("name")
            ch = r.get("changes") or []
            what = ", ".join(str(c.get("id")) + (" " + ", ".join(
                _param_text(k, v) for k, v in (c.get("params") or {}).items())
                if c.get("params") else "") for c in ch) or "no change"
            lines.append(f"      run {name}: the control with {what}"
                         + (f" (status {r.get('status')})" if r.get("status") else ""))
            for c in ch:
                if c.get("meaning"):
                    lines.append(f"          {c.get('id')} MEANS: {c['meaning']}")
                if c.get("not_meaning"):
                    lines.append(f"          it does NOT mean: {c['not_meaning']}")
    else:
        lines.append("    No run differs from the control: the question is about the "
                     "control itself, its years against each other.")
    lines += ["", "REGIONS (where the numbers are reported; every one is restricted to "
                  "the vegetated cells):"]
    for reg, info in (t.get("regions") or {}).items():
        lines.append(f"    {reg}: {info.get('cells')} cells, {info.get('area_km2')} km2 "
                     f"({info.get('rule')})")
    ev = t.get("event") or {}
    if ev.get("normal_years"):
        lines += ["", f"NORMAL YEARS (the planner's, for a question about an event): "
                      f"{_span(ev['normal_years'])}; the years compared with them: "
                      f"{', '.join(str(y) for y in ev.get('target_years') or [])}."]
    lines += ["", "VARIABLES IN `reg` (name, units, how a year's value is made):"]
    fields: Dict[str, Dict[str, Any]] = {}
    for r in (t.get("runs") or {}).values():
        for v, meta in ((r or {}).get("fields") or {}).items():
            fields.setdefault(v, meta or {})
    for v, meta in fields.items():
        how = "total over the year" if meta.get("how") == "sum" else "mean over the year"
        lines.append(f"    {v:16s} {str(meta.get('units')):16s} {how}")
        if meta.get("note"):
            lines.append(f"    {'':16s} NOTE: {str(meta['note'])[:300]}")
    for name, ob in (t.get("observed") or {}).items():
        lines.append(f"    {name:16s} {str(ob.get('units')):16s} total over the year: the "
                     f"observed {ob.get('what') or name} (run \"observed\", source "
                     f"\"observation\": {ob.get('source')}) and each run's own (source "
                     f"\"model\", from {ob.get('model_field')})")
    lines += _period_brief(t)
    lines.append("")
    return "\n".join(lines + _brief_tail(ctx, step1))


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
    conclusion, and no caveat, note or other sentence written on the plot:
    caveats travel in the records. Axis labels >= 16pt, tick labels >= 13pt.
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


# THE SAME JOB FOR A WHOLE DOMAIN (docs/ELM_DOMAIN_EXPERIMENTS.md, step 5b).
# Not a column ensemble: one case over a prepared domain, reported by region
# and year, so a regional total is the answer rather than a sum the brief must
# forbid; and a run's difference from the control means exactly what the
# server's change list says it means, which the brief quotes.
DOMAIN_TASK = f"""\
You are an earth-system scientist analysing this whole-domain experiment of the
model named in the brief above: a control run and, for a what-if question, runs
that differ from it by named changes, reported by region, year, month and
season. Decide which figures answer what the user asked,
and write the code that draws them.

Report at TWO scales:
  1. overall: the domain as a whole (its largest region) through the years,
     with each run against the control, or each year against the normal years.
  2. regions: where in the domain the answer differs (the elevation bands, a
     watershed). A region that departs from the domain-wide pattern is usually
     the interesting result.

Rules:
  - At most {MAX_PLOTS} figures. Fewer is correct if the experiment supports
    fewer; say why in `notes` rather than padding to {MAX_PLOTS}.
  - Do not propose anything a blocking caveat forbids. A run's difference from
    the control means what WHAT THIS EXPERIMENT IS says it means, and never
    what it says it does NOT mean. If the question cannot be addressed by this
    experiment, say so in `notes` and propose only what can.
  - Report as-is. No interpretation drawn on the figure, no titles that state a
    conclusion, and no caveat, note or other sentence written on the plot:
    caveats travel in the records. Axis labels >= 16pt, tick labels >= 13pt.
  - Region and variable names are long. Give every label room: rotate the
    ticks, wrap them, or move them into a legend. Two labels that overlap are
    two labels nobody can read.
  - The values in `reg` are already annual: a total over the year or a mean
    over it, per region, as the catalog says. Do not rescale them. Compute a
    difference between runs, or a year against the normal years, from `reg`.
  - A "mean over the year" is a stock. Label a difference in it as a
    difference in the stock, never as an amount removed or added that year.
  - A question about a month or a season (a summer, a spring, a winter) is
    answered from `reg_monthly` or `reg_seasonal`, never from `reg`: a year's
    value mixes that season with the rest of the year. A rate's value there
    is its total over the month or the season, a state's its mean over it.

Each figure's `code` is a Python snippet run with these already bound:
    reg       tidy frame: run | region | variable | year | value | units | how |
              source. One row per run, region, variable and model year; `run`
              is "control" or a run's name, and an observation's rows have run
              "observed" and source "observation".
    reg_monthly   tidy frame: run | region | variable | month | value | units |
              how | minus_control. One row per run, region, variable and
              month; `month` is "YYYY-MM"; `minus_control` is the run's
              difference from the control that month (None for the control).
              None when the experiment has no months.
    reg_seasonal  tidy frame: run | region | variable | year | season | value |
              units | how | minus_control. `season` is DJF, MAM, JJA or SON;
              DJF belongs to the year of its January and holds the December
              before it, so a run whose output starts in January has no DJF
              in its first year (that row is absent, not zero).
    regions   dict: region name -> {{"cells", "area_km2", "rule"}}
    runs      list of runs: {{"run", "reused", "status", "changes": [{{"id",
              "params", "meaning", "not_meaning"}}]}}
    caveats   the caveat records
    plt, np, pd, out_path
    df, prof and soil are None for this experiment: do not use them.
It MUST save the figure to out_path, and MUST assign a dict named `result`
containing the numbers it plotted plus an integer `n` = how many data points
(region-years, or run-region-years) the figure rests on. A result with n below
3, or with no n, is rejected.

Return ONLY JSON:
{{"reasoning": "<2-5 sentences: which of this model's variables and regions you
                decided answer what the user asked, and why those and not
                others. Name anything you considered computing and rejected,
                with the reason. This is read by the reviewer, so say what you
                actually decided rather than restating the question.>",
  "notes": "<what you could and could not address, and why>",
  "figures": [
    {{"id": "snake_case_name",
      "question": "<the question this figure answers>",
      "why": "<one sentence: what this figure adds that the others do not>",
      "scale": "overall" | "regions",
      "variables": ["variable names used, as `reg` holds them"],
      "code": "<python>"}}
  ]}}
"""


# STEP 2b'S REPLY, DEFINED ONCE (the user's decision on G part 2, 2026-11-08).
# investigate reads `plan_items`, `amendment` (analysis_plan.SCOPE_KEYS and
# settings, with its why) and `skipped`, and SUITE_TASK's reply template is
# rendered from the same parts, so the task cannot ask for one shape while
# the code reads another.
from agents.analysis.analysis_plan import Part as _Part             # noqa: E402
from agents.analysis.analysis_plan import SCOPE_KEYS as _SCOPE_KEYS  # noqa: E402

AMENDMENT_KEYS = ("why",) + _SCOPE_KEYS + ("settings",)
# A FIGURE IS A SPECIFICATION THE FRAMEWORK DRAWS (the user's decision of
# 2026-11-08, after the third acceptance batch; no fallback to a script in
# this phase): its panels name outputs by label and options from a closed
# list (figure_spec.PANEL_KINDS); an output the plan does not ask for is an
# extra, in a plan output's own form, made under the figure's amendment.
SUITE_FIGURE_PARTS = (
    _Part("id", "name", "its id", "snake_case_name"),
    _Part("question", "words", "the question this figure answers",
          "<the question this figure answers>"),
    _Part("why", "words", "what this figure adds that the others do not",
          "<one sentence: what this figure adds that the others do not>"),
    _Part("scale", "choice", "its scale", choices=("overall", "regions")),
    _Part("plan_items", "list", "the plan's analyses it shows, by id, or [\"screen\"] "
          "for the one figure that runs the plan's screen and draws nothing else", ["P1"]),
    _Part("amendment", "amendment", "null, or the scope its extra outputs add to the plan, "
          "with why", None),
    _Part("extra", "list", "outputs the plan does not ask for, each in a plan output's own "
          "form with an id X1, X2, ...; [] for none",
          '[] | [{"id": "X1", "kind": "anomaly", "field": "TSA", "period": "2015-JJA", '
          '"region": "..."}]'),
    _Part("panels", "list", "its panels, at most four (THE PANELS above)",
          '[{"kind": "map", "output": "P2.O1"}, {"kind": "bars", "outputs": ["P1.O1", '
          '"P1.O2"]}]'),
)
SUITE_REPLY_PARTS = (
    _Part("reasoning", "words", "which of the plan's analyses these figures carry out",
          "<2-5 sentences: which of the plan's analyses these figures carry out, and "
          "anything you considered and rejected, with the reason.>"),
    _Part("notes", "words", "what you could and could not address",
          "<what you could and could not address, and why>"),
    _Part("figures", "figures", "the figures", None),
    _Part("skipped", "skipped", "each analysis of the plan this round does not carry out, "
          "with why", None),
)
SKIPPED_KEYS = ("analysis", "why")


def _suite_reply_lines() -> List[str]:
    """SUITE_TASK's reply template, rendered from SUITE_REPLY_PARTS,
    SUITE_FIGURE_PARTS, AMENDMENT_KEYS and SKIPPED_KEYS."""
    from agents.analysis.analysis_plan import _shown
    am = ", ".join(f'"{k}": ' + ('"..."' if k == "why" else "{}" if k == "settings" else "[]")
                   for k in AMENDMENT_KEYS)
    L = ["Return ONLY JSON:"]
    for i, p in enumerate(SUITE_REPLY_PARTS):
        lead = "{" if i == 0 else "  "
        tail = "}" if i == len(SUITE_REPLY_PARTS) - 1 else ","
        if p.holds == "figures":
            L.append(f'{lead}"{p.name}": [')
            for j, f in enumerate(SUITE_FIGURE_PARTS):
                shown = f"null | {{{am}}}" if f.holds == "amendment" else _shown(f)
                L += textwrap.wrap(("    {" if j == 0 else "      ") + f'"{f.name}": {shown}'
                                   + ("}" if j == len(SUITE_FIGURE_PARTS) - 1 else ","),
                                   79, subsequent_indent="          ",
                                   break_long_words=False, break_on_hyphens=False)
            L.append("  ]" + tail)
        elif p.holds == "skipped":
            L += textwrap.wrap(f'{lead}"{p.name}": [{{"analysis": "P6", "why": "<why this '
                               f'round does not carry out that analysis of the plan>"}}]'
                               + tail, 79, subsequent_indent="      ",
                               break_long_words=False, break_on_hyphens=False)
        else:
            L += textwrap.wrap(f'{lead}"{p.name}": {_shown(p)}{tail}', 79,
                               subsequent_indent="      ", break_long_words=False,
                               break_on_hyphens=False)
    return L


# THE REQUIRED OUTPUTS, in plan_contract's words (the user's decision after G
# part 2's fixes, 2026-11-08): what counts as carrying out an analysis
from agents.analysis import plan_contract as _PC                    # noqa: E402

_OUTPUT_RULE = "\n".join(textwrap.wrap("  - " + _PC.STEP2_WORDS, 79, subsequent_indent="    ",
                                       break_long_words=False, break_on_hyphens=False))

SUITE_TASK_BODY = f"""\
You are an earth-system scientist choosing how the RECORDED ANALYSIS PLAN
above is shown, for this whole-domain experiment of the model named in the
brief. The framework has made every required output the plan names (listed
above by label, such as P1.O1, each with the call that made it). You write no
code: each figure is a specification, and the framework draws it.

Report at TWO scales: overall (the domain or its largest region) and regions
(the elevation bands, a watershed), as the plan names them.

Rules:
  - At most {MAX_PLOTS} figures. Fewer is correct if the plan needs fewer.
  - Follow the plan's order by question type; the framework keeps it. For a
    hypothesis question the planned analyses come first: the framework makes
    every required output's call, draws every figure that adds no output,
    then closes the planned phase (planned_phase.json: each analysis of the
    plan marked by its required outputs), then makes the extra outputs and
    runs the plan's screen for the figure that draws it; the screen is
    refused before that close. Every lookup after the close, an amendment's
    included, is exploratory and never planned, and anything the screen
    shows that the plan did not predict is "found by screening, not tested".
    For an exploratory question the framework runs the plan's screen first,
    with its scope, scoring rule and display rule recorded, then makes the
    required outputs' calls; a figure may draw the screen; then follow what
    it points to.
{_OUTPUT_RULE}
  - Each figure names the plan's analyses it shows (`plan_items`), or
    ["screen"] for the one figure that runs the plan's screen and draws
    nothing else.
  - An output the plan does not ask for is an EXTRA: give it in `extra`, in a
    plan output's own form (its kind and keys) with an id X1, X2, ..., and an
    `amendment` with `why` and the scope it adds (a key it leaves out is the
    plan's). The amendment is appended to the record with a time stamp, the
    framework makes the extra's call and logs it under the figure, and its
    records never count as planned. A lookup outside the plan's scope and
    the amendment's is refused, and its figure is not drawn.
  - An extra's settings are its amendment's, else the method's defaults
    (with the contract's alternatives computed beside them). A timing
    measure has only the settings timing.nc holds.
  - Every output a figure draws is an evidence record (its id, e.g. "L7",
    comes back with the finding, with the warnings it carries): a claim later
    names its records and carries their warnings. Every number a claim cites
    is made again from the model output before the report is written.
  - Do not propose anything a blocking caveat forbids. A run's difference
    from the control means what the configuration says it means, never what
    it says it does NOT mean.
  - The framework writes every title and label from the outputs' own keys
    and nothing more: no interpretation is drawn on a figure. A panel whose
    output the method found not estimable shows the method's own words; a
    panel with nothing to draw, or whose output was not made, fails its
    figure, and the caveat names the panel.
  - Each figure is checked before anything is called or drawn: a panel kind,
    an option or an output that does not exist, or an output of a kind its
    panel does not draw, refuses the figure with the reason. A figure is a
    specification only: a "code" field refuses it, and no script is run.

"""


def suite_task() -> str:
    """Step 2b's task for a study with a suite: SUITE_TASK_BODY, the panels
    (figure_spec.panel_lines, which names the timing measures) and the reply
    template, rendered from SUITE_REPLY_PARTS and SUITE_FIGURE_PARTS."""
    from agents.analysis import analysis_plan as _plan
    from agents.analysis import figure_spec as _fs
    _plan._elm()
    import domain_methods as _m                                        # noqa: N812
    return (SUITE_TASK_BODY + "\n".join(_fs.panel_lines(_m.TIMING_IDS) + [""]
                                          + _suite_reply_lines()) + "\n")


def __getattr__(name: str):
    # SUITE_TASK IS RENDERED, NOT WRITTEN (2026-11-08): its panel list reads
    # domain_methods, which this module imports only when needed
    if name == "SUITE_TASK":
        return suite_task()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def task_for(ctx) -> str:
    """The task a study's brief is followed by: the whole-domain one for a
    whole-domain experiment (the suite's for one whose control has a suite),
    the column one otherwise."""
    if getattr(ctx, "is_domain", False):
        from agents.analysis import analysis_plan as _plan
        return suite_task() if _plan.suite_bearing(ctx) else DOMAIN_TASK
    return TASK


# STEP 2b FOR A SUITE-BEARING STUDY (package E, 2026-11-05). domain_brief is
# replaced by the data catalog, the compact methods view and the recorded
# analysis plan (with its review and amendments); the brief's rules stay (two
# scales, caveats bind mechanically, five plots is a ceiling). The scripts
# reach the results only through `suite`, whose calls are gated by the plan's
# review and logged; anything the plan did not name is an amendment.
def suite_brief(ctx, step1: Optional[Dict[str, Any]] = None) -> str:
    from agents.analysis import analysis_plan as _plan
    out_dir = _plan.analysis_dir(ctx)
    parts = _plan.brief_parts(ctx, out_dir)
    plan = ctx.plan or {}
    lines = ["THE USER ASKED:", f"    {plan.get('question') or '(no question recorded)'}", "",
             "THE PLANNER DECOMPOSED THAT INTO:"]
    lines += [f"    - {g}" for g in (plan.get("goals") or [])]
    lines += ["", "THE RECORDED ANALYSIS PLAN (step 2a, written before any result was "
              "seen, and passed by step 3's review):", parts["plan"], ""]
    # EACH REQUIRED OUTPUT AS THE CALL THAT PRODUCES IT (2026-11-08), from
    # plan_contract: what step 2b is told to call is what the close accepts
    lines += _plan.output_calls(_plan.load_plan(out_dir)) + [""]
    lines += ["ITS REVIEW:", parts["review"], "",
              "AMENDMENTS SO FAR (each appended after results were seen; never planned):",
              parts["amendments"], "", "THE DATA CATALOG:", parts["catalog"], "",
              "THE METHODS:", parts["methods"], ""]
    return "\n".join(lines + _brief_tail(ctx, step1))


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


class ReplyError(ValueError):
    """Step 2b's reply could not be read: cut off at its output limit, or not
    JSON. The reply is saved before this is raised (`saved` names the file)."""

    def __init__(self, message: str, prompt: str = None, raw: str = None,
                 saved: Optional[Dict[str, str]] = None, cut: bool = False):
        super().__init__(message)
        self.prompt, self.raw, self.saved, self.cut = prompt, raw, saved or {}, cut


# STEP 2b'S OUTPUT LIMIT (2026-11-08): question 1 of G part 2's run again was
# cut off at the client's 16,384 tokens. The required outputs' calls are now
# the framework's, so the reply holds drawing code only; this ceiling is the
# backstop, not the fix.
STEP2_MAX_TOKENS = 32000


def propose(ctx, step1=None, model: str = DEFAULT_MODEL,
            client=None, feedback=None, out_dir=None, round_no: int = 1) -> Dict[str, Any]:
    """Ask for figure specs. Returns {notes, figures} with figures capped.

    `feedback` is step 3's verdict from the previous round: what it judged
    unsupported, unreadable or unanswered. It is appended AFTER the task so the
    stable part of the prompt stays byte-identical between rounds.

    A REPLY THAT CANNOT BE READ IS SAVED FIRST (2026-11-08): cut off at its
    output limit (finish_reason "length") or not JSON, the prompt and the reply
    go to step2_round<n>_*.txt under `out_dir` before ReplyError is raised,
    so the cut reply is on record rather than lost.
    """
    if client is None:
        from agents.llm_agent import SimpleLLMClient
        client = SimpleLLMClient(model=model)
        client.label = "step2_investigate"      # so step 4 can attribute the spend
        client.max_tokens = STEP2_MAX_TOKENS
    prompt = context_brief(ctx, step1) + "\n" + task_for(ctx)
    if feedback:
        prompt += ("\n\nA PREVIOUS ROUND OF THESE FIGURES WAS REVIEWED AND "
                   "JUDGED INSUFFICIENT. Address this and propose a revised "
                   "set — keep what worked, do not simply re-send it:\n"
                   + str(feedback))
    reply = client.ask([{"role": "user", "content": prompt}])
    cut = getattr(client, "last_finish_reason", None) == "length"
    try:
        spec = _parse(reply)
    except (ValueError, json.JSONDecodeError) as e:
        saved = _runner.save_exchange(out_dir, "step2", round_no, prompt, reply) \
            if out_dir is not None else {}
        limit = getattr(client, "max_tokens", None)
        raise ReplyError(
            ("step 2b's reply was cut off at its output limit"
             + (f" of {limit} tokens" if limit else "") if cut else
             f"step 2b's reply is not readable JSON ({e})")
            + (f"; it is saved as {', '.join(sorted(Path(v).name for v in saved.values()))}"
               if saved else ""), prompt=prompt, raw=reply, saved=saved, cut=cut) from None
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
    skipped = [{k: (x or {}).get(k) for k in SKIPPED_KEYS}
               for x in (spec.get("skipped") or []) if isinstance(x, dict)
               and (x or {}).get("analysis")] if isinstance(spec.get("skipped"), list) else []
    return {"notes": spec.get("notes"), "reasoning": spec.get("reasoning"),
            "figures": kept[:MAX_PLOTS], "skipped": skipped, "raw": reply, "prompt": prompt,
            **({"finish_reason": "length"} if cut else {})}


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
        # a specification names outputs, not fields: its finding lists the
        # fields of the outputs it drew
        v = ", ".join(f"`{x}`" for x in ((got or {}).get("variables") or f.get("variables")
                                         or [])) or "—"
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


def _runs_screen(f: Dict[str, Any]) -> bool:
    """A figure that runs the plan's screen: it names "screen" in its
    plan_items or draws a screen panel. For the order only: the gate still
    decides what each call may do."""
    return "screen" in (f.get("plan_items") or []) or \
        any(isinstance(p, dict) and p.get("kind") == "screen" for p in f.get("panels") or [])


def _script_figure(ctx, f: Dict[str, Any], out_dir, have, withheld, can_filter: bool,
                   all_caveats) -> tuple:
    """One figure of a study without a suite, drawn by its script (TASK,
    DOMAIN_TASK): (its finding or None, a caveat or None), as before."""
    fid = f["id"]
    wanted = [str(v) for v in (f.get("variables") or []) if v]
    missing = [v for v in wanted if v not in have]
    if can_filter and wanted and len(missing) == len(wanted):
        # EVERY variable it named is absent. A figure missing one of five
        # may still be worth drawing; one missing all of them cannot be.
        why = "; ".join(f"{v}: {withheld[v]}" for v in missing
                        if v in withheld) or \
              f"not present in any frame of this run"
        return None, {
            "id": f"figure_failed:{fid}",
            "severity": "context",
            "statement": (f"The planned figure '{fid}' "
                          f"({f.get('question')}) was not run: it reads "
                          f"{', '.join(missing)}, and {why}. Available "
                          f"variables: {', '.join(sorted(have)) or 'none'}."),
            "applies_to": "completeness of the step 2 figure set",
            "source": "step2_investigate:preflight"}
    run = _runner.run(f.get("code") or "", ctx, Path(out_dir) / f"{fid}.png",
                      script_path=Path(out_dir) / f"{fid}.py")
    if run["ok"]:
        return {"id": fid, "question": f.get("question"),
                "scale": f.get("scale"), "variables": f.get("variables"),
                "result": run["result"], "figure": run["figure"],
                "script": run.get("script_path"),
                "n": (run["result"] or {}).get("n"),
                "blocked_by": _blocked_by(f.get("variables"), all_caveats)}, None
    return None, {
        "id": f"figure_failed:{fid}", "severity": "context",
        "statement": (f"The planned figure '{fid}' "
                      f"({f.get('question')}) did not produce a usable "
                      f"result: {run['error']}"),
        "applies_to": "completeness of the step 2 figure set",
        "source": "step2_investigate"}


SPEC_SUFFIX = ".spec.json"


def _spec_figure(ctx, f: Dict[str, Any], out_dir, produced: Optional[Dict[str, Any]],
                 plan: Dict[str, Any], round_no: int, all_caveats) -> tuple:
    """One figure of a study with a suite, drawn by the framework from its
    specification (figure_spec): checked against the plan, its amendment
    appended, its extra outputs and the plan's screen called through the
    gated, logged `suite` under the figure's id, then drawn from the calls'
    own results. Returns (its outcome for the close, its finding or None, a
    caveat or None); a figure refused or failed is a caveat naming why."""
    from agents.analysis import analysis_plan as _plan
    from agents.analysis import claim_trace as _ct
    from agents.analysis import figure_spec as _fs
    _plan._elm()
    import domain_methods as _m                                        # noqa: N812
    out_dir = Path(out_dir)
    fid, items = f["id"], list(f.get("plan_items") or [])

    def fail(why, lines=None, panels=None):
        return ({"id": fid, "plan_items": items, "ok": False, "round": round_no, "error": why,
                 **({"lines": lines} if lines else {}), **({"panels": panels} if panels else {})},
                None,
                {"id": f"figure_failed:{fid}", "severity": "context",
                 "statement": f"The planned figure '{fid}' ({f.get('question')}) was not drawn: "
                              f"{why}",
                 "applies_to": "completeness of the step 2 figure set",
                 "source": "step2_investigate:figure_spec"})

    probs = _fs.check_figure(f, plan, _m.TIMING_IDS)
    if probs:
        return fail("its specification was refused: " + "; ".join(probs))
    am = f.get("amendment") if isinstance(f.get("amendment"), dict) and f.get("amendment") \
        else None
    am_id = None
    if am:
        try:
            am_id = _plan.append_amendment(out_dir, am.get("why") or "", am)["id"]
        except (ValueError, _plan.PlanNotApproved) as e:
            return fail(f"its amendment was not recorded ({e})")
    n_before = len(_ct._read_log(out_dir))
    acc = _plan.make_access(_plan.access_spec(ctx), script=fid, amendment=am_id)
    index = _PC.plan_index(plan)
    extra = {str(xo["id"]): xo for xo in f.get("extra") or []}
    values = dict((produced or {}).get("values") or {})
    rows = {r["label"]: r for r in ((produced or {}).get("record") or {}).get("calls") or []}
    names = list(dict.fromkeys(n for p in f["panels"] for n in _fs.panel_refs(p)))

    def made(label, call):
        try:
            values[label] = call()
            rows[label] = {"label": label, "line": len(_ct._read_log(out_dir)), "ok": True}
        except Exception as e:                                      # noqa: BLE001
            rows[label] = {"label": label, "line": len(_ct._read_log(out_dir)), "ok": False,
                           "error": f"{type(e).__name__}: {e}"[:300]}

    # ITS EXTRA OUTPUTS, made by the framework under its amendment, each as a
    # plan output's own call (plan_contract.call_spec), an input map from the
    # output it names
    for xid, xo in extra.items():
        cs = _PC.call_spec(xo, {"settings": (am or {}).get("settings") or {}},
                           _m.TIMING_IDS, _plan._method_defaults)
        args, miss = [], None
        for v in cs["args"]:
            if isinstance(v, dict) and "input" in v:
                io = index.get(v["input"]) or extra.get(v["input"]) or {}
                src, path = values.get(v["input"]), _PC.MAP_PATH.get(io.get("kind"))
                mp = src if path is None or src is None else (src or {}).get(path)
                if mp is None:
                    miss = v["input"]
                    break
                args.append(mp)
            else:
                args.append(v)
        if miss:
            rows[xid] = {"label": xid, "line": None, "ok": False,
                         "error": f"not called: its input map {miss} was not made"}
            continue
        made(xid, lambda cs=cs, args=args: getattr(acc, cs["fn"])(*args, **cs["kwargs"]))
    # THE PLAN'S SCREEN: a hypothesis study's runs here, after the close; an
    # exploratory study's ran first, by the framework, and is drawn as it is
    sc = plan.get("screen") or {}
    if _fs.SCREEN in names and _fs.SCREEN not in values:
        made(_fs.SCREEN, lambda: acc.screen(int(sc.get("year")), sc.get("region")))
    ll = None
    if any(p.get("kind") == "map" for p in f["panels"]):
        try:
            ll = acc.lat_lon()
        except Exception:                                           # noqa: BLE001
            ll = None
    log = _ct._read_log(out_dir)
    refs = {}
    for n in names:
        r = rows.get(n) or {}
        ln = log[r["line"] - 1] if r.get("ok") and r.get("line") else None
        answered = ln is not None and "sha256" in ln
        outcome, why = ((_PC.PRODUCED, None) if n == _fs.SCREEN else _PC.answer(ln)) \
            if answered else (None, None)
        o = index.get(n) or extra.get(n) or (
            {"kind": "screen", "period": str(sc.get("year")), "region": sc.get("region")}
            if n == _fs.SCREEN else {})
        units = {i: ((values.get(i) or {}).get("change_units") or (values.get(i) or {})
                     .get("units")) if isinstance(values.get(i), dict) else None
                 for i in o.get("inputs") or []}
        refs[n] = {"ref": n, "kind": o.get("kind"), "out": o, "value": values.get(n),
                   "input_units": units, "made": answered, "outcome": outcome, "why": why,
                   "line": r.get("line") if answered else None,
                   "error": (ln or {}).get("refused") or r.get("error")
                   or (None if r else "no call made it")}
    spec_path = out_dir / f"{fid}{SPEC_SUFFIX}"
    spec_path.write_text(json.dumps(strip_dashes_in({p.name: f.get(p.name)
                                                     for p in SUITE_FIGURE_PARTS}),
                                    indent=1, default=str))
    drawn = _fs.render_figure(f, refs, out_dir / f"{fid}.png", lat_lon=ll,
                              normal_years=(plan.get("baseline") or {}).get("normal_years")
                              or ())
    lines = [n_before + 1, len(_ct._read_log(out_dir))]
    outcome_rec = {"id": fid, "plan_items": items, "ok": bool(drawn["ok"]), "round": round_no,
                   "error": drawn["error"], "lines": lines, "panels": drawn["panels"]}
    if not drawn["ok"]:
        return fail(drawn["error"], lines, drawn["panels"])
    ids = [f"L{refs[n]['line']}" for n in names if refs[n].get("line")]
    by_id = {x["id"]: x for x in _ct.lookups_since(out_dir, 0)}
    fields = sorted({str(refs[n]["out"].get("field")) for n in names
                     if refs[n]["out"].get("field")})
    finding = {"id": fid, "question": f.get("question"), "scale": f.get("scale"),
               "variables": fields,
               "result": {n: (_ct.citable(log[refs[n]["line"] - 1].get("result"))
                              if refs[n].get("line") else {"not made": refs[n]["error"]})
                          for n in names},
               "figure": drawn["figure"], "script": str(spec_path), "spec": str(spec_path),
               "n": drawn["n"], "panels": drawn["panels"],
               "blocked_by": _blocked_by(fields, all_caveats),
               "plan_items": items, "amendment": am_id,
               "lookups": [by_id[i] for i in ids if i in by_id]}
    return outcome_rec, finding, None


def investigate(ctx, out_dir, step1=None, model: str = DEFAULT_MODEL,
                client=None, feedback=None, round_no: int = 1,
                history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Propose, execute, and report — including what failed and why.

    A script that fails becomes a CAVEAT, not a silent gap. That is the whole
    lesson of soil_attribution: the analysis that quietly produced nothing was
    indistinguishable, from the outside, from one that was never planned.

    A HYPOTHESIS STUDY WITH A SUITE CLOSES ITS PLANNED PHASE BEFORE ITS SCREEN
    (the user's decision on G part 2, 2026-11-08): every figure that does not
    run the screen runs first, then the framework writes planned_phase.json
    (analysis_plan.close_planned_phase), then the figure or figures that run
    the screen. `history` holds the earlier rounds' figures and step 2b's
    stated reasons, which the close counts too.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A SUITE-BEARING STUDY PROPOSES NOTHING BEFORE ITS PLAN HAS PASSED step
    # 3's review (package E, 2026-11-05): no model call, no lookup, and the
    # reason recorded as a caveat, since every lookup would be refused anyway.
    from agents.analysis import analysis_plan as _plan
    on_suite = _plan.suite_bearing(ctx)
    if on_suite:
        ok, why = _plan.plan_passed(out_dir)
        if not ok:
            out = {"round": round_no, "notes": why, "reasoning": None, "findings": [],
                   "caveats": [{"id": "analysis_plan_not_approved", "severity": "blocking",
                                "statement": f"No figure was proposed and no lookup made: "
                                             f"{why}.",
                                "applies_to": "every finding of this study",
                                "source": "step2_investigate:analysis_plan"}],
                   "figures": [], "plan_md": None, "exchange": None, "n_proposed": 0,
                   "n_succeeded": 0, "responded_to_feedback": feedback or None}
            (out_dir / FILENAME).write_text(json.dumps(out, indent=2, default=str))
            return out

    # THE FRAMEWORK'S CALLS FIRST (2026-11-08): the plan's required outputs
    # (and, in an exploratory study, its screen) are made by the framework
    # before step 2b's model call, once per study, so a reply that cannot be
    # read loses none of them; the figures read them from `outputs`
    produced = _plan.produce_required(ctx, out_dir, round_no) if on_suite else None
    plan_rec = _plan.load_plan(out_dir) if on_suite else {}
    reply_failed = None
    try:
        spec = propose(ctx, step1=step1, model=model, client=client,
                       feedback=feedback, out_dir=out_dir, round_no=round_no)
    except ReplyError as e:
        if not on_suite:
            raise
        reply_failed = str(e)
        spec = {"notes": None, "reasoning": None, "figures": [], "skipped": [],
                "raw": e.raw, "prompt": e.prompt}
    if on_suite:
        # THE MODEL'S WORDS KEPT IN THE RECORDS CARRY NO DASHES (fix 6 of G
        # part 2, 2026-11-08); the exchange files stay verbatim, and the code
        # is code
        spec = dict(spec, notes=strip_dashes(spec.get("notes")) or None,
                    reasoning=strip_dashes(spec.get("reasoning")) or None,
                    skipped=[dict(x, why=strip_dashes(x.get("why")))
                             for x in spec.get("skipped") or []],
                    figures=[dict(f, **{k: strip_dashes(f[k]) for k in ("question", "why")
                                        if isinstance(f.get(k), str)})
                             for f in spec["figures"]])
    all_caveats = list(ctx.caveats or []) + list((step1 or {}).get("caveats") or [])
    findings, caveats = [], []
    outcomes: List[Dict[str, Any]] = []        # each figure run, for the planned phase
    if produced and (produced["record"] or {}).get("round") == round_no:
        outcomes.append(_plan.required_attempt(produced["record"], plan_rec))
    if reply_failed:
        caveats.append({
            "id": "step2b_reply_unreadable", "severity": "context",
            "statement": (f"No figure was drawn this round: {reply_failed}. The required "
                          f"outputs the framework made stand, and step 3 reads them."),
            "applies_to": "the figures of this round",
            "source": "step2_investigate:reply"})

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
    # the suite checks its own fields, by name, at the call
    can_filter = bool(have) and not on_suite

    # THE ORDER OF A HYPOTHESIS STUDY: the planned analyses, the close, the
    # screen. Ordering only; a figure that calls the screen before the close
    # is refused by the gate.
    figs = list(spec["figures"])
    qt = _plan.load_plan(out_dir).get("question_type") if on_suite else None
    closing = bool(on_suite and qt == "hypothesis" and not _plan.load_planned_phase(out_dir))
    # A FIGURE THAT ADDS AN OUTPUT COMES AFTER THE CLOSE (2026-11-08): its
    # extra outputs are exploratory, so in a hypothesis study they are made
    # after the planned phase is closed, then the screen
    adds = [f for f in figs if f.get("extra") and not _runs_screen(f)] if closing else []
    screen_figs = adds + ([f for f in figs if _runs_screen(f)] if closing else [])
    figs = [f for f in figs if not any(f is g for g in screen_figs)] + screen_figs
    stated = {**((history or {}).get("skipped") or {}),
              **{x["analysis"]: x["why"] for x in spec.get("skipped") or []}}

    for f in figs:
        fid = f["id"]
        if screen_figs and f is screen_figs[0]:
            _plan.close_planned_phase(
                out_dir, when="before the screen",
                by=f"the framework, in round {round_no} of step 2b: after every figure that "
                   f"does not run the screen, before the figure '{fid}' that runs it",
                figures=list((history or {}).get("figures") or []) + outcomes,
                skipped=stated)
        if on_suite:
            drew = _spec_figure(ctx, f, out_dir, produced, plan_rec, round_no, all_caveats)
            outcomes.append(drew[0])
            if drew[1] is not None:
                findings.append(drew[1])
            if drew[2] is not None:
                caveats.append(drew[2])
            continue
        found, cav = _script_figure(ctx, f, out_dir, have, withheld, can_filter, all_caveats)
        if found is not None:
            findings.append(found)
        if cav is not None:
            caveats.append(cav)

    # THE FRAMEWORK'S FINDINGS, BESIDE THE FIGURES' (2026-11-08): each
    # analysis's required outputs as the framework made them, with their
    # evidence records, whatever the figures did (plotting independence)
    if produced:
        findings = _plan.required_findings(out_dir, produced["record"], plan_rec) + findings

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
           "responded_to_feedback": feedback or None,
           # A STUDY WITH A SUITE (2026-11-08): each figure's outcome and the
           # analyses step 2b says it did not carry out, with its reasons, for
           # the close of the planned phase
           **({"figure_outcomes": outcomes, "skipped": spec.get("skipped") or [],
               "required_outputs": _plan.REQUIRED_FILE,
               **({"reply_failed": reply_failed} if reply_failed else {}),
               **({"finish_reason": spec["finish_reason"]} if spec.get("finish_reason")
                  else {})}
              if on_suite else {})}

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
