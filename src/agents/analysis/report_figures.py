#!/usr/bin/env python3
"""
The figures and tables a whole-domain report leads with, drawn by code
src/agents/analysis/report_figures.py

    build(study_dir, plan, out_dir) -> record          never raises

WHY CODE DRAWS THEM (the report rebuild, 2026-11-13). The model chose a
report's figures after the yes, and twice they were not drawn: a rain-cut
study's elevation figure was refused, and its answer then could not say where
growth fell; a CO2 study planned maps and drew none of them. A reader asked for
the figures to lead and the text to support them. So the report's figures
are now drawn by code, for every whole-domain study, from the approved
comparisons (the plan, `plan`) and the results store, with the shared figure
tool (figure_spec.render_figure). No model call writes or chooses one. The
model's own figures of step 2 are left as they are.

WHERE THE VALUES COME FROM. Through the model server's analysis access
(domain_access.SuiteAccess) over the study's stores, built as
analysis_plan.make_access builds it, with two differences:

    the gate    this module's own (_gate): it allows the four calls made here
                (CALLS) and marks each not planned. The plan's gate
                (analysis_plan.check_lookup) keeps the MODEL's lookups to what
                was declared before any result was seen; code that draws the
                report's figures is not the model and is not held to it.
    the log     this module's own (ReportLog): every call, in the format of
                lookups.jsonl, into report_lookups.jsonl beside it. Nothing
                here writes lookups.jsonl, whose line numbers are the records
                ("L12") the claims cite.

A figure may draw values that are not planned lookups. A TABLE PRINTS ONLY
PLANNED RECORDS: each cell is the value (and its percent) of one line of
lookups.jsonl marked planned, read and never written, formatted as the report
formats values (step4_report.format_value), so the second reading of the
numbers can make each one again. `printed_records` names them.

FIGURE 1, OVER TIME. One panel per quantity the approved comparisons name,
over the region most of them name, for each month of the years they cover:
the new run minus the control run as bars, the months of the change shaded
(from the design records, 01_inputs/run_given.json, else the change's own
months). The quantity the run changes comes first, then the catalog's order
of groups, the order the screen uses. A quantity the two runs hold alike in
every month (SAME_SHARE) gets no panel and the caption names it. Over eight
panels the figure splits in two, water and energy first, then plants, carbon,
nitrogen and fire. A study of the control run alone shows each month against
the normal years' range and mean instead.

FIGURE 2, OVER SPACE. The first quantity of the plan's first comparison. The
upper row maps its change in each cell over the season of the change (the
whole year for a change in every month), one panel per year, at most four, on
one scale centred on zero, with the region's edge. The lower row sets the
region's cells against the change of the input the run changes, where that
change differs from cell to cell (VARIES_SHARE), coloured by the control
run's air temperature; where it does not, against the pair of one of the
plan's own "overlap" outputs; and is left out otherwise. A study of the
control run alone maps the departure from the normal years' mean, with no
lower row.

TABLE 1 holds the planned period records of Figure 1's quantities over its
region (differences, or anomalies for the control run alone): seasons in
time order, then whole years, at most six periods, those with the most
records first. TABLE 2 holds the planned elevation splits over that region:
a row per band with its area, a column per split.

Every sentence here (a heading, a caption, a reason something is not drawn)
says what is drawn, never what it shows.
"""
import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.analysis import analysis_plan as AP
from agents.analysis import field_words as FW
from agents.analysis import figure_spec as FS

RECORD_FILE = "report_figures.json"
LOG_FILE = "report_lookups.jsonl"
SCRIPT = "report_figures"
LOG_LABEL = "the report's figures, drawn by code: not a planned lookup"
CALLS = ("get_series", "get_map", "region_mask", "lat_lon")
CONTROL = "control"
OVER_TIME, OVER_SPACE = "over_time", "over_space"
PERIODS, ELEVATION = "periods", "elevation"
PNG_PATTERN = "figure_over_*.png"                # what an earlier build drew, removed first
MAX_MAP_YEARS = 4
MAX_PERIODS = 6
# the inputs a run may change that the store holds, each a sum of its fields
# (rain and snow are both water, so their change is one amount)
INPUTS = (("RAIN", "SNOW"), ("TBOT",), ("FSDS",))
AIR = "TBOT"
WEATHER = "weather in"
WATER_GROUPS = ("weather in", "snow", "soil water and groundwater", "runoff",
                "water to the air", "energy")
PART_HEADINGS = {"water": "Water and energy", "plants": "Plants, carbon, nitrogen and fire"}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
# THE WEATHER A RUN CHANGES IS READ FROM ITS DESIGN, NOT FROM ITS OUTPUT. A run
# started from a later saved state differs from the control run in its first
# month even where its weather is the control's: in the CO2 study the new
# run's rain over the watershed differs in its first January alone, by the
# model's start step. Read from the output, the CO2 run would seem to change
# the rain. So a weather input the run's changes do not alter (altered_inputs)
# gets no panel, and the caption says the run does not change it. The server's
# change list says which changes write weather (an "inputs" or "years"
# writer); weather_scale says by its settings which inputs, as the design
# page draws them (domain_given.KNOBS and QUANTITY): rain and snow for
# precip_factor, the air temperature for temperature_K. Another year's
# weather (weather_repeat_year) alters every weather input.
WEATHER_KNOBS = {"precip_factor": ("RAIN", "SNOW"), "temperature_K": ("TBOT",)}
# Any other quantity gets no panel only where the two runs hold it alike in
# every month: no month's change beyond the rounding of a sum of stored values.
SAME_SHARE = 1e-9
# A CHANGE DIFFERS FROM CELL TO CELL when its spread over the region's cells is
# more than one hundredth of its largest size: a uniform change, such as two
# kelvin added everywhere, still varies by the rounding of the stored values.
VARIES_SHARE = 0.01


class NoStore(Exception):
    """The study holds no results store to draw from."""


# ─────────────────────────────────────────────────────────────────────────────
# THE DOOR: the server's access, with this module's own gate and log
# ─────────────────────────────────────────────────────────────────────────────
class ReportLog(AP.LookupLog):
    """The log of the report's figures: AP.LookupLog's lines, written to
    report_lookups.jsonl and never to lookups.jsonl."""

    def __init__(self, out_dir):
        super().__init__(out_dir, script=SCRIPT)
        self.path = Path(out_dir) / LOG_FILE


def _gate(ask: Dict[str, Any]) -> Dict[str, Any]:
    """Allows the calls this module makes, each marked not planned."""
    if ask.get("call") not in CALLS:
        raise PermissionError(f"the report's figures make only {', '.join(CALLS)}, "
                              f"not {ask.get('call')}")
    return {"planned": False, "plan_items": [], "label": LOG_LABEL}


def make_access(study_dir, out_dir, plan: Dict[str, Any], ctx=None,
                suites: Optional[Dict[str, str]] = None):
    """The server's SuiteAccess over the study's stores, as
    analysis_plan.make_access builds it, with this module's gate and log.
    `suites` ({run: store}) in place of the study's own, for a test."""
    AP._elm()
    import domain_access as DA                                      # noqa: N812
    if suites is None:
        if ctx is None:
            from agents.analysis import step0_context
            ctx = step0_context.load(str(study_dir))
        suites = AP.suite_dirs(ctx)
    if not suites or CONTROL not in suites:
        raise NoStore("the study has no results store of the control run")
    ny = [int(y) for y in (plan.get("baseline") or {}).get("normal_years") or []]
    if not ny and ctx is not None:
        ny = [int(y) for y in AP.planner_normal_years(ctx)]
    return DA.SuiteAccess(dict(suites), gate=_gate, log=ReportLog(out_dir), normal_years=ny,
                          study_dir=str(study_dir))


# ─────────────────────────────────────────────────────────────────────────────
# THE STUDY, from its records: no store is read here
# ─────────────────────────────────────────────────────────────────────────────
def _read(p) -> Dict[str, Any]:
    try:
        d = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _list(x) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def analyses(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in _list(plan.get("analyses")) if isinstance(a, dict)]


def new_run(plan: Dict[str, Any]) -> Optional[str]:
    """The new run of the first approved comparison that names one, else of
    the first output that names one; None for a study of the control run
    alone."""
    for c in _list(plan.get("comparisons")):
        r = c.get("run") if isinstance(c, dict) else None
        if r and str(r) != CONTROL:
            return str(r)
    for a in analyses(plan):
        for o in _list(a.get("outputs")):
            if isinstance(o, dict) and o.get("run") and str(o["run"]) != CONTROL:
                return str(o["run"])
    return None


def _year_of(period) -> Optional[int]:
    m = re.match(r"^(?:WY)?(\d{4})", str(period))
    return int(m.group(1)) if m else None


def study_years(plan: Dict[str, Any]) -> List[int]:
    """The years the approved comparisons cover, else the years of the
    analyses' periods."""
    ys = {int(y) for c in _list(plan.get("comparisons")) if isinstance(c, dict)
          for y in _list(c.get("years")) if str(y).lstrip("-").isdigit()}
    if not ys:
        ys = {y for a in analyses(plan) for p in _list(a.get("periods"))
              for y in [_year_of(p)] if y is not None}
    return sorted(ys)


def main_region(plan: Dict[str, Any]) -> Optional[str]:
    """The region most of the plan's analyses name, each analysis counted
    once; the first named on a tie."""
    counts: Counter = Counter()
    first: List[str] = []
    for a in analyses(plan):
        regs = [str(r) for r in _list(a.get("regions")) if r]
        if not regs:
            regs = [str(o["region"]) for o in _list(a.get("outputs"))
                    if isinstance(o, dict) and o.get("region")]
        for r in dict.fromkeys(regs):
            counts[r] += 1
            if r not in first:
                first.append(r)
    if not counts:
        return None
    top = max(counts.values())
    return next(r for r in first if counts[r] == top)


def plan_fields(plan: Dict[str, Any]) -> List[str]:
    return list(dict.fromkeys(str(f) for a in analyses(plan) for f in _list(a.get("fields"))
                              if f))


def run_changes(study_dir, run: Optional[str]):
    """(the run's entry of 01_inputs/run_given.json or None, its changes
    [{"id", "params"}] from that entry or 01_inputs/domain_experiment.json,
    or None where neither record holds them)."""
    if not run:
        return None, None
    from agents import run_figures as RF
    given = RF.entry(RF.load(study_dir), run)
    changes = given.get("changes") if isinstance(given, dict) else None
    if not isinstance(changes, list):
        de = _read(Path(study_dir) / "01_inputs" / "domain_experiment.json")
        hit = [r for r in _list(de.get("runs")) if isinstance(r, dict) and r.get("name") == run]
        changes = hit[0].get("changes") if hit else None
    return (given if isinstance(given, dict) else None,
            changes if isinstance(changes, list) else None)


def altered_inputs(changes: Optional[Sequence[Any]]) -> Optional[List[str]]:
    """The weather fields of the store a run's changes alter (see
    WEATHER_KNOBS): [] for changes that leave the weather alone, None where
    it cannot be said (no record of the changes, or a change the server does
    not list)."""
    if not isinstance(changes, (list, tuple)):
        return None
    try:
        AP._elm()
        import domain_changes as DCH                                # noqa: N812
        out: List[str] = []
        for c in changes:
            cid = c.get("id") if isinstance(c, dict) else c
            decl = DCH.get(str(cid))
            if not any(isinstance(a, dict) and ("inputs" in a or "years" in a)
                       for a in decl.get("alters") or []):
                continue                                  # a setting, not the weather
            params = (c.get("params") if isinstance(c, dict) else None) or {}
            knobs = [k for k in WEATHER_KNOBS if k in (decl.get("params") or {})]
            if not knobs:                                  # another year's weather: all of it
                out += [f for g in INPUTS for f in g]
                continue
            for k in knobs:
                neutral = decl["params"][k].get("default")
                if float(params.get(k, neutral)) != float(neutral):
                    out += list(WEATHER_KNOBS[k])
        return list(dict.fromkeys(out))
    except Exception:                                               # noqa: BLE001
        return None


def change_months(study_dir, run: Optional[str],
                  years: Sequence[int]) -> Optional[Dict[str, Any]]:
    """When the run's change applies: {"labels": ["2033-06", ...]} for a
    change in some months, {"every": True} for a change in every month, None
    where the design records do not say. Read from 01_inputs/run_given.json
    (the months the design page shaded), else the changes' own "months" and
    "years", from that record or 01_inputs/domain_experiment.json."""
    if not run:
        return None
    given, changes = run_changes(study_dir, run)
    labels = sorted({str(x) for p in _list((given or {}).get("panels")) if isinstance(p, dict)
                     for x in _list(p.get("changed"))})
    if labels:
        return {"labels": labels}
    if not isinstance(changes, list) or not changes:
        return None
    for c in changes:
        params = c.get("params") if isinstance(c, dict) else None
        params = params if isinstance(params, dict) else {}
        months = [int(m) for m in _list(params.get("months")) if str(m).isdigit()]
        ys = [int(y) for y in _list(params.get("years")) if str(y).isdigit()] or list(years)
        labels += [f"{y}-{m:02d}" for y in ys for m in months if 1 <= m <= 12]
    return {"labels": sorted(set(labels))} if labels else {"every": True}


def _season_of(months) -> Optional[str]:
    """The season that holds every one of these calendar months, or None."""
    ms = {int(m) for m in months}
    for s, held in FW.SEASON_MONTHS.items():
        if ms and ms <= set(held):
            return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# THE CATALOG'S ORDER
# ─────────────────────────────────────────────────────────────────────────────
def catalog() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """The field notes and the catalog's groups in order; ({}, []) where the
    notes cannot be read (the plan's own order is kept then)."""
    try:
        AP._elm()
        import domain_catalog as DC                                 # noqa: N812
        return DC.load_notes(), list(DC.GROUPS)
    except Exception:                                               # noqa: BLE001
        return {}, []


def ordered(fields: Sequence[str], notes: Dict[str, Dict[str, Any]], groups: Sequence[str],
            changed: Sequence[str] = ()) -> List[str]:
    """The quantity the run changes first, then the catalog's groups in
    order, each group's fields in the catalog's order; a field the catalog
    does not hold last, in the plan's order."""
    cat = list(notes)
    gs = list(groups)
    given = list(fields)

    def key(f):
        g = (notes.get(f) or {}).get("group")
        return (0 if f in changed else 1, gs.index(g) if g in gs else len(gs),
                cat.index(f) if f in cat else len(cat), given.index(f))
    return sorted(given, key=key)


def parts_of(fields: Sequence[str],
             notes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """The figures Figure 1's panels make: one figure up to STACK_MAX_PANELS,
    else water and energy first, then plants, carbon, nitrogen and fire, a
    part longer than the limit cut into even figures. [(part, fields)]."""
    lim = FS.STACK_MAX_PANELS
    fields = list(fields)
    if len(fields) <= lim:
        return [("all", fields)] if fields else []
    water = [f for f in fields if (notes.get(f) or {}).get("group") in WATER_GROUPS]
    plants = [f for f in fields if f not in water]
    out = []
    for name, part in (("water", water), ("plants", plants)):
        if not part:
            continue
        k = -(-len(part) // lim)
        size = -(-len(part) // k)
        out += [(name, part[i:i + size]) for i in range(0, len(part), size)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1, OVER TIME
# ─────────────────────────────────────────────────────────────────────────────
def _by_label(series: Dict[str, Any]) -> Dict[str, float]:
    labs = [str(x) for x in list(series.get("labels") or [])]
    vals = np.asarray(series.get("values"), dtype=float).ravel()
    return {lab: float(v) for lab, v in zip(labs, vals)}


def same_in_every_month(diff: np.ndarray, ref: np.ndarray) -> bool:
    """Whether the two runs hold a quantity alike in every month: no month's
    difference is larger than SAME_SHARE of the quantity's largest month."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if not d.size:
        return False
    r = np.asarray(ref, dtype=float)
    scale = float(np.nanmax(np.abs(r))) if np.isfinite(r).any() else 0.0
    return float(np.max(np.abs(d))) <= SAME_SHARE * scale


def _reason(e: Exception) -> str:
    return re.sub(r"\s+", " ", str(e)).strip().rstrip(".") or type(e).__name__


def _months_record(acc, run: Optional[str], fields: Sequence[str], region: str,
                   years: Sequence[int], normal_years: Sequence[int]):
    """Each quantity's month values over the region: {field: {"values",
    "units", "normal"}}, the reasons each other field has no panel, and the
    fields the two runs hold alike."""
    labels = [f"{y}-{m:02d}" for y in years for m in range(1, 13)]
    got: Dict[str, Dict[str, Any]] = {}
    no_values: List[Tuple[str, str]] = []
    unchanged: List[str] = []
    for f in fields:
        try:
            ctl = acc.get_series(CONTROL, f, region, "month")
            new = acc.get_series(run, f, region, "month") if run else None
        except Exception as e:                                      # noqa: BLE001
            no_values.append((f, f"the results store gives no month by month values of it "
                                 f"({_reason(e)})"))
            continue
        c = _by_label(ctl)
        cv = np.array([c.get(lab, np.nan) for lab in labels], dtype=float)
        if run:
            n = _by_label(new)
            nv = np.array([n.get(lab, np.nan) for lab in labels], dtype=float)
            d = nv - cv
            if not np.isfinite(d).any():
                no_values.append((f, "the two runs hold no value of it in the same month"))
                continue
            if same_in_every_month(d, cv):
                unchanged.append(f)
                continue
            got[f] = {"values": d, "units": new.get("units") or ctl.get("units")}
        else:
            if not np.isfinite(cv).any():
                no_values.append((f, "the control run holds no value of it in those months"))
                continue
            lo, hi, mean = [], [], []
            for lab in labels:
                xs = np.array([c.get(f"{y}-{lab[5:]}", np.nan) for y in normal_years],
                              dtype=float)
                xs = xs[np.isfinite(xs)]
                lo.append(float(xs.min()) if xs.size else np.nan)
                hi.append(float(xs.max()) if xs.size else np.nan)
                mean.append(float(xs.mean()) if xs.size else np.nan)
            got[f] = {"values": cv, "units": ctl.get("units"),
                      "normal": {"low": lo, "high": hi, "mean": mean}}
    return labels, got, no_values, unchanged


def _ref(ref: str, kind: str, out: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """One output as the figure tool takes it (agents/run_figures.py)."""
    return {"ref": ref, "kind": kind, "out": out, "value": value, "made": True, "error": None,
            "outcome": None, "why": None, "line": None}


def _answers(plan: Dict[str, Any], test) -> List[int]:
    """The comparisons' numbers, as the design page shows them (1 for the
    plan's first analysis), whose outputs pass `test`."""
    return [i for i, a in enumerate(analyses(plan), 1)
            if any(isinstance(o, dict) and test(o) for o in _list(a.get("outputs")))]


def _over_time_answers(plan, fields: Sequence[str], region: str) -> List[int]:
    fs = set(fields)
    return _answers(plan, lambda o: o.get("field") in fs and o.get("region") == region and
                    o.get("kind") in ("difference", "anomaly", "series"))


def panel_sentence(panels: Sequence[Dict[str, Any]]) -> str:
    """"Panels: (a) rain, (b) topsoil water." as the interface's example
    writes it, or the one quantity of a figure of one panel."""
    if len(panels) == 1:
        return f"The quantity drawn is {panels[0]['words']}."
    return "Panels: " + ", ".join(f"({p['letter']}) {p['words']}" for p in panels) + "."


NOT_CHANGED, SAME = "not changed", "same"


def _why_words(why: str, one: bool) -> str:
    if why == NOT_CHANGED:
        return f"the new run does not change {'it' if one else 'them'}"
    return (f"the two runs hold the same value of {'it' if one else 'each'} in every "
            f"month")


def no_panel_sentence(fields: Sequence[str], why: str) -> str:
    """"Air temperature (TBOT) has no panel, because the new run does not
    change it." for the quantities of one reason."""
    one = len(fields) == 1
    return (f"{FW.capital(FW.join(FW.words(f) for f in fields))} "
            f"{'has' if one else 'have'} no panel, because {_why_words(why, one)}.")


def over_time(acc, plan, facts, notes, groups, out: Path, number: int,
              rec) -> List[Dict[str, Any]]:
    """Figure 1 (one figure, or two or more after a split): drawn, and their
    records."""
    run, region, years = facts["run"], facts["region"], facts["years"]
    if not region or not years or not facts["fields"]:
        rec["not_drawn"].append("No figure over time was drawn, because the approved plan names "
                                "no region, year or quantity to draw it for.")
        return []
    altered = facts.get("altered")
    weather = [f for f in facts["fields"] if (notes.get(f) or {}).get("group") == WEATHER]
    kept = [f for f in weather if f in altered] if run and altered is not None else []
    left = {f: NOT_CHANGED for f in weather if run and altered is not None and f not in altered}
    labels, got, no_values, unchanged = _months_record(
        acc, run, [f for f in facts["fields"] if f not in left], region, years,
        facts["normal_years"])
    left.update({f: SAME for f in unchanged})
    for f, why in no_values:
        rec["not_drawn"].append(f"{FW.capital(FW.words(f))} has no panel over time, because "
                                f"{why}.")
    for f, why in left.items():
        rec["not_drawn"].append(f"{FW.capital(FW.words(f))} has no panel over time, because "
                                f"{_why_words(why, True)}.")
    if run and altered is None:                   # no design record: the output says it
        kept = [f for f in got if (notes.get(f) or {}).get("group") == WEATHER]
    parts = parts_of(ordered(list(got), notes, groups, kept), notes)
    if not parts:
        rec["not_drawn"].append("No figure over time was drawn, because no quantity of the plan "
                                "has a month to draw.")
        return []
    shade = facts["shade"]
    shaded = sorted(set((shade or {}).get("labels") or []) & set(labels))
    if run and shade is None:
        rec["not_drawn"].append("No month is shaded over time, because the study's records do "
                                "not say which months the change alters.")
    # each quantity left out is named under the figure of its own part
    names = [p for p, _ in parts]
    by_part: Dict[str, Dict[str, List[str]]] = {}
    for f, why in left.items():
        p = "water" if (notes.get(f) or {}).get("group") in WATER_GROUPS else "plants"
        by_part.setdefault(p if p in names else names[-1], {}).setdefault(why, []).append(f)
    figures: List[Dict[str, Any]] = []
    for k, (part, fields) in enumerate(parts):
        fid = OVER_TIME if k == 0 else f"{OVER_TIME}_{k + 1}"
        png = f"figure_{fid}.png"
        spec = {"id": f"report_{fid}", "layout": "stack", "letters": True, "panels": []}
        refs: Dict[str, Dict[str, Any]] = {}
        for i, f in enumerate(fields, 1):
            ref = f"M{i}"
            refs[ref] = _ref(ref, "months", {"field": f, "units": got[f]["units"]},
                             {"labels": labels, "values": [float(x) for x in got[f]["values"]]})
            panel: Dict[str, Any] = {"kind": "months", "output": ref}
            if shaded:
                panel["shade_labels"] = shaded
            if got[f].get("normal"):
                panel["normal"] = got[f]["normal"]
            spec["panels"].append(panel)
        done = FS.render_figure(spec, refs, out / png)
        if not done.get("ok"):
            rec["problems"].append(f"a figure over time was not drawn: {done.get('error')}")
            continue
        panels = [{"letter": FS.LETTERS[i] if len(fields) > 1 else "", "field": f,
                   "words": FW.words(f, with_code=False)} for i, f in enumerate(fields)]
        last = part not in [q for q, _ in parts[k + 1:]]
        rw, yw = FW.region_words(region), FW.years_words(years)
        if run:
            cap = (f"The new run minus the control run over {rw}, for each month of {yw}. "
                   f"{panel_sentence(panels)}")
            if shaded:
                cap += " The shaded months are those with the change."
            elif shade and shade.get("every"):
                cap += " The change holds in every month, so no month is shaded."
            if last:
                for why, fs in (by_part.get(part) or {}).items():
                    cap += " " + no_panel_sentence(fs, why)
        else:
            cap = (f"The control run over {rw}, for each month of {yw}. "
                   f"{panel_sentence(panels)} The grey band is the range of the normal years "
                   f"{FW.years_words(facts['normal_years'])} in each month, and the dashed "
                   f"line is their mean.")
        head = PART_HEADINGS.get(part, "Each quantity") + " month by month"
        if not run:
            head += ", against the normal years"
        if any(q == part for q, _ in parts[:k]):
            head += " (continued)"
        figures.append({"id": fid, "file": png, "number": number + len(figures),
                        "heading": head, "caption": cap,
                        "comparisons": _over_time_answers(plan, fields, region),
                        "panels": panels})
    return figures


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2, OVER SPACE
# ─────────────────────────────────────────────────────────────────────────────
def _period(kind: str, year: int) -> str:
    return {"year": str(year), "wateryear": f"WY{year}"}.get(kind, f"{year}-{kind}")


def _normal(kind: str) -> str:
    return {"year": "normal", "wateryear": "normal-WY"}.get(kind, f"normal-{kind}")


def _step(period: str) -> str:
    p = str(period)
    if re.fullmatch(r"WY\d{4}", p) or p == "normal-WY":
        return "wateryear"
    if re.search(r"-(DJF|MAM|JJA|SON)$", p):
        return "season"
    if re.fullmatch(r"\d{4}-\d{2}", p):
        return "month"
    return "year"


def _span(period: str) -> str:
    """The part of the year a period covers, in words."""
    m = re.search(r"(DJF|MAM|JJA|SON)$", str(period))
    if m:
        return FW.months_words(FW.SEASON_MONTHS[m.group(1)])
    return "the water year" if _step(period) == "wateryear" else "the whole year"


def map_kind(plan: Dict[str, Any], run: Optional[str], shade: Optional[Dict[str, Any]]) -> str:
    """The period of Figure 2's maps: the season that holds every changed
    month, else the whole year; for a study of the control run alone, the
    kind of the first comparison's first period."""
    if run:
        labels = (shade or {}).get("labels") or []
        months = {int(str(x)[5:7]) for x in labels if re.fullmatch(r"\d{4}-\d{2}", str(x))}
        return _season_of(months) or "year"
    first = analyses(plan)[0] if analyses(plan) else {}
    for p in _list(first.get("periods")):
        if str(p).startswith("normal"):
            continue
        s = _step(str(p))
        if s == "season":
            return str(p)[-3:]
        if s in ("year", "wateryear"):
            return s
    return "year"


def _units(acc, cache, run, field, region, period) -> str:
    """A field's unit at a period's step, as the store gives it."""
    key = (run, field, region, _step(period))
    if key not in cache:
        try:
            cache[key] = str(acc.get_series(run, field, region, _step(period)).get("units")
                             or "")
        except Exception:                                           # noqa: BLE001
            cache[key] = ""
    return cache[key]


def _with_span(units: str, period: str) -> str:
    """"gC/m2 over June to August" for a total over a season or a year; the
    unit alone for a mean."""
    u = FW.units_words(units)
    if FW.units_period(units) in ("season", "year", "water year") and u:
        return f"{u} over {_span(period)}"
    return u


def _cell_map(acc, o: Dict[str, Any]) -> np.ndarray:
    """A per-cell output of the plan as the map it names: a difference (the
    run minus the control), an anomaly (minus its normal) or a map."""
    k, f, p = o.get("kind"), str(o.get("field")), str(o.get("period"))
    run = str(o.get("run") or CONTROL)
    if k == "difference":
        return np.asarray(acc.get_map(run, f, p), dtype=float) - \
            np.asarray(acc.get_map(CONTROL, f, p), dtype=float)
    if k == "anomaly":
        kind = p[-3:] if _step(p) == "season" else _step(p)
        return np.asarray(acc.get_map(run, f, p), dtype=float) - \
            np.asarray(acc.get_map(run, f, _normal(kind)), dtype=float)
    if k == "map":
        return np.asarray(acc.get_map(run, f, p), dtype=float)
    raise ValueError(f"a {k} is not a map")


def _input_of(x: Any, own: Dict[str, Dict[str, Any]], index: Dict[str, Dict[str, Any]]):
    x = str(x)
    return own.get(x) if "." not in x else index.get(x)


def overlap_pairs(plan: Dict[str, Any], main: str) -> List[Dict[str, Any]]:
    """The plan's "overlap" outputs that set the main quantity's map against
    another map: [{"number" (its comparison), "y" (the main quantity's
    output), "x" (the other), "region"}], in the plan's order."""
    from agents.analysis import plan_contract as PC
    index = PC.plan_index(plan)
    out = []
    for i, a in enumerate(analyses(plan), 1):
        own = {str(o.get("id")): o for o in _list(a.get("outputs")) if isinstance(o, dict)}
        for o in own.values():
            if o.get("kind") != "overlap":
                continue
            ins = [_input_of(x, own, index) for x in _list(o.get("inputs"))]
            if len(ins) != 2 or not all(isinstance(x, dict) for x in ins):
                continue
            if not all(x.get("kind") in ("difference", "anomaly", "map") and
                       x.get("region") is None and x.get("field") and x.get("period")
                       for x in ins):
                continue
            fields = [str(x.get("field")) for x in ins]
            if main not in fields:
                continue
            y = fields.index(main)
            out.append({"number": i, "y": ins[y], "x": ins[1 - y], "region": o.get("region")})
    return out


def _change_words(o: Dict[str, Any]) -> str:
    k, w = o.get("kind"), FW.words(o.get("field"), with_code=False)
    if k == "difference":
        return f"{w}, new run minus control run"
    if k == "anomaly":
        return f"{w}, minus the normal years' mean"
    return f"{w}, {'control run' if str(o.get('run') or CONTROL) == CONTROL else 'new run'}"


def _wrap(text: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _celsius(a: np.ndarray, units: str) -> Tuple[np.ndarray, str]:
    if str(units).strip() == "K":
        return a - 273.15, "°C"
    return a, FW.units_words(units)


def _air(acc, cache, region, period) -> Tuple[np.ndarray, str]:
    """The control run's air temperature over a period, in °C."""
    t = np.asarray(acc.get_map(CONTROL, AIR, period), dtype=float)
    return _celsius(t, _units(acc, cache, CONTROL, AIR, region, period))


def over_space(acc, plan, facts, out: Path, number: int, rec) -> Optional[Dict[str, Any]]:
    """Figure 2: drawn, and its record, or None."""
    run, region, years = facts["run"], facts["region"], facts["years"]
    first = analyses(plan)[0] if analyses(plan) else {}
    main = next((str(f) for f in _list(first.get("fields")) if f), None)
    if not main or not region or not years:
        rec["not_drawn"].append("No figure over space was drawn, because the plan's first "
                                "comparison names no quantity, region or year.")
        return None
    w = FW.words(main, with_code=False)
    mapped, more = list(years[:MAX_MAP_YEARS]), list(years[MAX_MAP_YEARS:])
    if more:
        rec["not_drawn"].append(f"The map of {w} in {FW.years_words(more)} is not drawn, "
                                f"because a figure over space holds at most {MAX_MAP_YEARS} "
                                f"years.")
    kind = map_kind(plan, run, facts["shade"])
    cache: Dict[Any, str] = {}
    ll = acc.lat_lon()
    mask = np.asarray(acc.region_mask(region), dtype=bool)
    upper = []
    for y in mapped:
        p = _period(kind, y)
        try:
            if run:
                a = np.asarray(acc.get_map(run, main, p), dtype=float) - \
                    np.asarray(acc.get_map(CONTROL, main, p), dtype=float)
            else:
                a = np.asarray(acc.get_map(CONTROL, main, p), dtype=float) - \
                    np.asarray(acc.get_map(CONTROL, main, _normal(kind)), dtype=float)
        except Exception as e:                                      # noqa: BLE001
            rec["not_drawn"].append(f"The map of {w} in {FW.period_words(p)} is not drawn, "
                                    f"because the results store gives none ({_reason(e)}).")
            continue
        upper.append((y, p, a))
    if not upper:
        return None
    span = _span(upper[0][1])
    u = _with_span(_units(acc, cache, CONTROL, main, region, upper[0][1]), upper[0][1])
    scale = (f"{FW.capital(w)}, new run minus control run" if run
             else f"{FW.capital(w)}, minus the normal years' mean") + (f" ({u})" if u else "")
    # a row of maps over a row of cells; maps alone, two to a row from three on
    lower_due = bool(run)
    cols = len(upper) if lower_due or len(upper) <= 2 else 2
    spec = {"id": "report_over_space", "layout": "grid", "columns": cols,
            "letters": True, "panels": []}
    refs: Dict[str, Dict[str, Any]] = {}
    panels: List[Dict[str, Any]] = []
    for i, (y, p, a) in enumerate(upper, 1):
        refs[f"U{i}"] = _ref(f"U{i}", "shared_map",
                             {"field": main, "title": FW.capital(FW.period_words(p)),
                              "scale_label": _wrap(scale, 28)}, a)
        spec["panels"].append({"kind": "shared_map", "output": f"U{i}"})
        panels.append({"field": main, "words": w, "period": p, "kind": "map"})
    lower, lower_words, numbers = _lower_row(acc, plan, facts, main, upper, mask, cache, rec)
    for i, (x_, y_, c_, labels) in enumerate(lower, 1):
        refs[f"C{i}"] = _ref(f"C{i}", "cells", labels, {"x": x_, "y": y_, "c": c_})
        spec["panels"].append({"kind": "cells", "output": f"C{i}"})
        panels.append({"field": main, "words": w, "against": labels["against"],
                       "kind": "cells"})
    png = f"figure_{OVER_SPACE}.png"
    done = FS.render_figure(spec, refs, out / png,
                            lat_lon={"lat": ll["lat"], "lon": ll["lon"], "outline": mask})
    if not done.get("ok"):
        rec["problems"].append(f"the figure over space was not drawn: {done.get('error')}")
        return None
    several = len(panels) > 1
    for i, p in enumerate(panels):
        p["letter"] = FS.LETTERS[i] if several else ""

    def tag(kind_):
        ls = [p["letter"] for p in panels if p["kind"] == kind_ and p["letter"]]
        return f"({', '.join(ls)}) " if ls else ""
    ys = FW.years_words([y for y, _, _ in upper])
    rw = FW.region_words(region)
    when = (f"over each water year (October to September) of {ys}"
            if _step(upper[0][1]) == "wateryear" else f"over {span} in {ys}")
    if run:
        cap = (f"{tag('map')}Each cell's {w} {when}, the new run minus the control run. The "
               f"line is the edge of {rw}.")
    else:
        cap = (f"{tag('map')}Each cell's {w} {when}, minus the mean of the normal years "
               f"{FW.years_words(facts['normal_years'])}. The line is the edge of {rw}.")
    if lower:
        cap += f" {tag('cells')}{lower_words}"
    answers = _answers(plan, lambda o: (o.get("field") == main and o.get("region") is None and
                                        o.get("kind") in ("difference", "anomaly", "map"))
                       or (o.get("kind") == "elevation_split" and o.get("region") == region))
    return {"id": OVER_SPACE, "file": png, "number": number,
            "heading": f"{FW.capital(w)} in each cell", "caption": cap,
            "comparisons": sorted(set(answers) | set(numbers)), "panels": panels}


def _input_change(acc, run, group, upper, mask):
    """The change of one input over each mapped period ([(year, period, main
    change, input change or None)]), the fields of it that changed, and
    whether the change differs from cell to cell in any period."""
    per_year, changed, varies = [], [], False
    for y, p, a in upper:
        total = None
        for f in group:
            try:
                d = np.asarray(acc.get_map(run, f, p), dtype=float) - \
                    np.asarray(acc.get_map(CONTROL, f, p), dtype=float)
            except Exception:                                       # noqa: BLE001
                continue                                            # a field the store lacks
            inside = d[mask & np.isfinite(d)]
            if inside.size and float(np.max(np.abs(inside))) > 0.0 and f not in changed:
                changed.append(f)
            total = d if total is None else total + d
        if total is not None:
            v = total[mask & np.isfinite(total)]
            top = float(np.abs(v).max()) if v.size else 0.0
            if top > 0.0 and float(v.max() - v.min()) > VARIES_SHARE * top:
                varies = True
        per_year.append((y, p, a, total))
    return per_year, [f for f in group if f in changed], varies


def _lower_row(acc, plan, facts, main, upper, mask, cache, rec):
    """The lower row of Figure 2: [(x, y, colour, labels)], the caption's
    words for it, and the comparisons it draws. Against the input the run
    changes where that change differs from cell to cell; else against the
    pair of the plan's own overlap; else none."""
    run, region = facts["run"], facts["region"]
    w = FW.words(main, with_code=False)
    if not run:
        rec["not_drawn"].append("The figure over space has no lower row, because a study of the "
                                "control run alone has no change to set the cells against.")
        return [], "", []
    altered = facts.get("altered")
    for group in INPUTS:
        if altered is not None:                   # only the weather the design changes
            group = tuple(f for f in group if f in altered)
            if not group:
                continue
        per_year, used, varies = _input_change(acc, run, group, upper, mask)
        if not (varies and used):
            continue
        iw = FW.join(FW.words(f, with_code=False) for f in used)
        out = []
        for y, p, a, t in per_year:
            if t is None:
                continue
            try:
                temp, tu = _air(acc, cache, region, p)
            except Exception as e:                                  # noqa: BLE001
                rec["not_drawn"].append(f"The lower panel of {FW.period_words(p)} is not drawn, "
                                        f"because the results store gives no air temperature "
                                        f"for it ({_reason(e)}).")
                continue
            sel = mask & np.isfinite(t) & np.isfinite(a) & np.isfinite(temp)
            u_x = _with_span(_units(acc, cache, CONTROL, used[0], region, p), p)
            u_y = _with_span(_units(acc, cache, CONTROL, main, region, p), p)
            labels = {"x_label": _wrap(f"{FW.capital(iw)}, new run minus control run"
                                       + (f" ({u_x})" if u_x else ""), 40),
                      "y_label": _wrap(f"{FW.capital(w)}, new run minus control run"
                                       + (f" ({u_y})" if u_y else ""), 40),
                      "c_label": _wrap(f"Air temperature in the control run ({tu})", 26),
                      "against": "+".join(used)}
            out.append((t[sel], a[sel], temp[sel], labels))
        words = (f"The cells of {FW.region_words(region)}, the same change against the change "
                 f"in {iw} over the same period, coloured by the control run's air "
                 f"temperature.")
        return out, words, []
    pairs = overlap_pairs(plan, main)[:len(upper)]
    if not pairs:
        rec["not_drawn"].append(f"The figure over space has no lower row, because the change is "
                                f"the same in every cell and the plan sets no other map against "
                                f"{w}.")
        return [], "", []
    out, numbers, xs = [], [], []
    for pr in pairs:
        reg = str(pr["region"] or region)
        py, px = str(pr["y"]["period"]), str(pr["x"]["period"])
        try:
            ym, xm = _cell_map(acc, pr["y"]), _cell_map(acc, pr["x"])
            m = mask if reg == region else np.asarray(acc.region_mask(reg), dtype=bool)
            temp, tu = _air(acc, cache, reg, py)
        except Exception as e:                                      # noqa: BLE001
            rec["not_drawn"].append(f"A lower panel over space is not drawn, because the results "
                                    f"store gives no map for it ({_reason(e)}).")
            continue
        sel = m & np.isfinite(xm) & np.isfinite(ym) & np.isfinite(temp)
        u_y = _with_span(_units(acc, cache, CONTROL, main, reg, py), py)
        u_x = _with_span(_units(acc, cache, CONTROL, str(pr["x"]["field"]), reg, px), px)
        labels = {"x_label": _wrap(FW.capital(_change_words(pr["x"]))
                                   + (f" ({u_x})" if u_x else ""), 40),
                  "y_label": _wrap(FW.capital(_change_words(pr["y"]))
                                   + (f" ({u_y})" if u_y else ""), 40),
                  "c_label": _wrap(f"Air temperature in the control run ({tu})", 26),
                  "against": str(pr["x"]["field"])}
        out.append((xm[sel], ym[sel], temp[sel], labels))
        numbers.append(pr["number"])
        xs.append(pr["x"])
    if not out:
        return [], "", []
    x0 = xs[0]
    xw = FW.words(x0.get("field"), with_code=False)
    what = (f"the change in {xw}" if x0.get("kind") == "difference" else
            f"the departure of {xw} from the normal years" if x0.get("kind") == "anomaly" else
            f"the {'control' if str(x0.get('run') or CONTROL) == CONTROL else 'new'} run's {xw}")
    words = (f"The cells of {FW.region_words(region)}, the change in {w} against {what} over "
             f"{_span(str(x0.get('period')))}, as the plan's comparison"
             f"{'s' if len(set(numbers)) > 1 else ''} "
             f"{FW.join(str(n) for n in dict.fromkeys(numbers))} set{'' if len(set(numbers)) > 1 else 's'} "
             f"them, coloured by the control run's air temperature.")
    return out, words, numbers


# ─────────────────────────────────────────────────────────────────────────────
# THE TABLES: planned records only
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(x) -> str:
    from agents.analysis.step4_report import format_value
    return format_value(x)


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def _time_key(period: str) -> Tuple[int, int, int]:
    """Seasons in time order (a winter before the spring of its year), then
    whole years, a year before its water year; anything else after."""
    p = str(period)
    m = re.fullmatch(r"(\d{4})-(DJF|MAM|JJA|SON)", p)
    if m:
        return (0, int(m.group(1)), SEASON_ORDER.index(m.group(2)))
    m = re.fullmatch(r"(WY)?(\d{4})", p)
    if m:
        return (1, int(m.group(2)), 1 if m.group(1) else 0)
    return (2, 0, 0)


def _column_words(period: str) -> str:
    p = str(period)
    if re.fullmatch(r"\d{4}", p):
        return f"Year {p}"
    return FW.capital(FW.period_words(p))


def period_table(lookups: Sequence[Dict[str, Any]], run: Optional[str], region: str,
                 panel_of: Dict[str, str], rec) -> Optional[Dict[str, Any]]:
    """Table 1: the planned difference records (anomaly records for the
    control run alone) of Figure 1's quantities over its region, by period."""
    kind = "difference" if run else "anomaly"
    want = run or CONTROL
    found: Dict[Tuple[str, str], Tuple[int, Dict[str, Any]]] = {}
    for i, ln in enumerate(lookups, 1):
        if not isinstance(ln, dict) or ln.get("planned") is not True or ln.get("call") != kind:
            continue
        k, res = ln.get("keys") or {}, ln.get("result")
        if not isinstance(res, dict) or k.get("region") != region or \
                str(k.get("run") or CONTROL) != want:
            continue
        if kind == "difference" and str(k.get("control") or CONTROL) != CONTROL:
            continue
        f, p = str(k.get("field")), str(k.get("period"))
        if f not in panel_of or _time_key(p)[0] > 1 or not _number(res.get(kind)):
            continue
        found.setdefault((f, p), (i, res))
    if not found:
        return None
    counts = Counter(p for _, p in found)
    chosen = sorted(counts, key=lambda p: (-counts[p], _time_key(p)))[:MAX_PERIODS]
    left = sorted(set(counts) - set(chosen), key=_time_key)
    columns = sorted(chosen, key=_time_key)
    rows = []
    for f in panel_of:
        hits = [found.get((f, p)) for p in columns]
        if not any(hits):
            continue
        units = {FW.units_words(h[1].get("change_units") or h[1].get("units"))
                 for h in hits if h}
        one = len(units) == 1
        cells = []
        for h in hits:
            if not h:
                cells.append("")
                continue
            res = h[1]
            t = _fmt(res[kind]) + (f" ({_fmt(res['percent'])} %)"
                                   if _number(res.get("percent")) else "")
            u = FW.units_words(res.get("change_units") or res.get("units"))
            cells.append(t if one or not u else f"{t} {u}")
        u = next(iter(units)) if one else ""
        rows.append({"label": FW.capital(FW.words(f)) + (f", {u}" if u else ""),
                     "cells": cells, "panel": panel_of[f],
                     "records": [f"L{h[0]}" for h in hits if h]})
    if not rows:
        return None
    if left:
        rec["not_drawn"].append(f"The table by period holds the {MAX_PERIODS} periods with the "
                                f"most records, so "
                                f"{FW.join(_column_words(p)[:1].lower() + _column_words(p)[1:] for p in left)} "
                                f"{'is' if len(left) == 1 else 'are'} not in it.")
    rw = FW.region_words(region)
    caption = (f"The new run minus the control run over {rw}, over each period. In brackets, "
               f"the change as a percent of the control run." if run else
               f"Each period of the control run minus the mean of the normal years, over {rw}. "
               f"In brackets, the departure as a percent of that mean.")
    return {"id": PERIODS, "caption": caption, "corner": "Quantity",
            "columns": [_column_words(p) for p in columns], "rows": rows}


def _maps_by_hash(lookups: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[int, Dict[str, Any]]]:
    """Every per-cell map a line of the log returned, by its hash: a
    method's map (its "arrays") or a map read whole (get_map's sha256)."""
    out: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for i, ln in enumerate(lookups, 1):
        if not isinstance(ln, dict):
            continue
        for h in (ln.get("arrays") or {}).values():
            out.setdefault(str(h), (i, ln))
        if ln.get("call") == "get_map" and ln.get("sha256"):
            out.setdefault(str(ln["sha256"]), (i, ln))
    return out


def elevation_table(lookups: Sequence[Dict[str, Any]], region: str,
                    rec) -> Optional[Dict[str, Any]]:
    """Table 2: the planned elevation splits over the region, a row per band
    with its area, a column per split, named by the map it splits."""
    by_hash = _maps_by_hash(lookups)
    splits = []
    for i, ln in enumerate(lookups, 1):
        if not isinstance(ln, dict) or ln.get("planned") is not True or \
                ln.get("call") != "elevation_split":
            continue
        k, res = ln.get("keys") or {}, ln.get("result")
        if not isinstance(res, dict) or k.get("region") != region:
            continue
        hit = by_hash.get(str(k.get("x")))
        if hit is None or hit[0] >= i:
            rec["not_drawn"].append("One elevation split is not in the table by elevation, "
                                    "because the map it splits is not on the record.")
            continue
        splits.append((i, res, hit[1]))
    if not splits:
        return None
    bands: List[str] = []
    for _, res, _ in splits:
        for key in res:
            m = re.fullmatch(r"parts\.(.+)\.mean", str(key))
            if m and m.group(1) not in bands:
                bands.append(m.group(1))
    if not bands:
        return None
    fields = list(dict.fromkeys(str((src.get("keys") or {}).get("field"))
                                for _, _, src in splits))
    one_field = len(fields) == 1
    columns = ["Area, km2"]
    for _, _, src in splits:
        k, sres = src.get("keys") or {}, src.get("result") or {}
        u = FW.units_words(sres.get("change_units") or sres.get("units"))
        name = (FW.capital(FW.period_words(k.get("period"))) if one_field else
                f"{FW.capital(FW.words(k.get('field')))}, {FW.period_words(k.get('period'))}")
        columns.append(name + (f", {u}" if u else ""))
    rows = []
    for b in bands:
        area = next((res.get(f"parts.{b}.area_km2") for _, res, _ in splits
                     if _number(res.get(f"parts.{b}.area_km2"))), None)
        cells = [_fmt(area) if _number(area) else ""]
        cells += [_fmt(res.get(f"parts.{b}.mean")) if _number(res.get(f"parts.{b}.mean"))
                  else "" for _, res, _ in splits]
        rows.append({"label": FW.capital(FW.band_words(b)), "cells": cells,
                     "records": [f"L{i}" for i, res, _ in splits
                                 if _number(res.get(f"parts.{b}.mean"))
                                 or _number(res.get(f"parts.{b}.area_km2"))]})
    what = {"difference": "the new run minus the control run",
            "anomaly": "the departure from the normal years' mean"}.get(
                splits[0][2].get("call"), "the map")
    rw = FW.region_words(region)
    caption = (f"{FW.capital(FW.words(fields[0]))}, {what}, in each elevation band of {rw}, the "
               f"mean over the band's cells, with each band's area." if one_field else
               f"Each map, {what}, in each elevation band of {rw}, the mean over the band's "
               f"cells, with each band's area.")
    return {"id": ELEVATION, "caption": caption, "corner": "Elevation band", "columns": columns,
            "rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORD
# ─────────────────────────────────────────────────────────────────────────────
def empty_record() -> Dict[str, Any]:
    return {"figures": [], "tables": [], "printed_records": [], "not_drawn": [], "problems": []}


def study_facts(study_dir, plan: Dict[str, Any]) -> Dict[str, Any]:
    """What the figures are drawn for, from the plan and the design records."""
    run = new_run(plan)
    years = study_years(plan)
    return {"run": run, "years": years, "region": main_region(plan),
            "altered": altered_inputs(run_changes(study_dir, run)[1]) if run else None,
            "fields": plan_fields(plan),
            "normal_years": [int(y) for y in (plan.get("baseline") or {}).get("normal_years")
                             or []],
            "shade": change_months(study_dir, run, years)}


def _clear(out: Path) -> None:
    """The figures an earlier build drew here (this module's own files only),
    so none is left from another draw."""
    for p in out.glob(PNG_PATTERN):
        try:
            p.unlink()
        except OSError:
            pass


def _build(study_dir: Path, plan: Dict[str, Any], out: Path, ctx, suites, access):
    rec = empty_record()
    facts = study_facts(study_dir, plan)
    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    try:
        acc = access if access is not None else make_access(study_dir, out, plan, ctx, suites)
    except NoStore as e:
        rec["not_drawn"].append(f"No figure or table was drawn, because {e}.")
        return rec
    notes, groups = catalog()
    figures = over_time(acc, plan, facts, notes, groups, out, 1, rec)
    space = over_space(acc, plan, facts, out, len(figures) + 1, rec)
    lookups = AP.lookups(out)
    panel_of = {p["field"]: f"{fig['number']}{p['letter']}" for fig in figures
                for p in fig["panels"]}
    tables: List[Dict[str, Any]] = []
    t1 = period_table(lookups, facts["run"], facts["region"], panel_of, rec) \
        if facts["region"] else None
    if t1:
        tables.append(dict(t1, number=len(tables) + 1,
                           follows=figures[-1]["id"] if figures else None))
    t2 = elevation_table(lookups, facts["region"], rec) if facts["region"] else None
    if t2:
        tables.append(dict(t2, number=len(tables) + 1,
                           follows=OVER_SPACE if space else
                           (figures[-1]["id"] if figures else None)))
    rec["figures"] = figures + ([space] if space else [])
    rec["tables"] = tables
    names = {n for t in tables for r in t["rows"] for n in r["records"]}
    rec["printed_records"] = sorted(names, key=lambda n: int(n[1:]))
    if not rec["figures"] and not tables and not rec["not_drawn"]:
        rec["not_drawn"].append("No figure or table was drawn, because the study holds nothing "
                                "the approved comparisons name to draw.")
    return rec


def build(study_dir, plan: Optional[Dict[str, Any]], out_dir, ctx=None,
          suites: Optional[Dict[str, str]] = None, access=None) -> Dict[str, Any]:
    """Draw the report's figures and tables into `out_dir` (the study's
    04_analysis), write their record there as report_figures.json, and
    return it (interface I1 of the report rebuild). Never raises: a failure
    is named under "problems" and the report is written without the figure.
    `ctx` (the study's AnalysisContext) saves reading the study's records
    again; `suites` or `access` stand in for the study's stores in a test."""
    out = Path(out_dir)
    try:
        rec = _build(Path(study_dir), plan if isinstance(plan, dict) else {}, out, ctx, suites,
                     access)
    except Exception as e:                                          # noqa: BLE001
        rec = empty_record()
        rec["problems"].append(f"the report's figures were not drawn ({type(e).__name__}: "
                               f"{_reason(e)})")
        rec["not_drawn"].append("No figure or table was drawn, because of the problem the "
                                "record names.")
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / RECORD_FILE).write_text(json.dumps(rec, indent=1, default=str))
    except Exception as e:                                          # noqa: BLE001
        rec["problems"].append(f"{RECORD_FILE} was not written ({type(e).__name__}: {e})")
    return rec
