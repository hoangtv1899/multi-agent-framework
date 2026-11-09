"""Step 2b's figures as specifications the framework draws (the user's
decision of 2026-11-08, after the third acceptance batch: plotting
specifications, with no fallback to a model-written script in this phase).

A study with a suite has its required outputs made by the framework before
step 2b is asked (analysis_plan.produce_required). Step 2b now says only how
to show them. A figure is one to MAX_PANELS panels, each one of PANEL_KINDS,
naming outputs by label ("P1.O1"), with options from a closed list:

    {"id": "fig_snow", "plan_items": ["P1"], "panels": [
        {"kind": "series", "outputs": ["P1.O4"], "years": [2015], "normal": true},
        {"kind": "map", "output": "P1.O5"},
        {"kind": "bars", "outputs": ["P1.O1", "P1.O2"]}]}

The framework checks a figure against the plan before anything is called or
drawn (check_figure), makes the calls a figure adds (an exploratory output it
declares under an amendment, or the plan's screen), and draws it
(render_figure). Every title and label comes from the outputs' own keys,
never a sentence of the model's. A panel whose output the method found not
estimable shows the method's own words; a panel with nothing to draw, or
whose output was not made, fails the figure and names the panel, so a blank
panel is caught by code and not only by step 3's eyes.
"""
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.analysis import plan_contract as PC

MAX_PANELS = 4
SCREEN = "screen"
EXTRA_ID = re.compile(r"^X\d+$")
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")


@dataclass(frozen=True)
class Option:
    """One option of a panel kind: its name, what it does, and the values it
    may take (a tuple of choices, or "years" or "bool")."""
    name: str
    means: str
    allows: Any


@dataclass(frozen=True)
class PanelKind:
    """One kind of panel: the output kinds it draws ("timing" stands for
    every timing measure), whether it names one output, several, or none
    (the screen's), its options and what it draws."""
    id: str
    accepts: Tuple[str, ...]
    takes: str                                  # "output", "outputs" or "none"
    options: Tuple[Option, ...]
    means: str


_VALUE = Option("value", "for an anomaly or a difference: its change, or its percent of the "
                "reference where the field allows one", ("change", "percent"))
PANEL_KINDS: Dict[str, PanelKind] = {k.id: k for k in (
    PanelKind("map", ("map", "anomaly", "difference"), "output", (_VALUE,),
              "one per-cell map: a map, or a per-cell anomaly or difference (its region "
              "null), a change drawn on a scale centred on zero"),
    PanelKind("series", ("series",), "outputs", (
        Option("years", "the years drawn on one seasonal axis; left out, the whole record "
               "is drawn", "years"),
        Option("normal", "true: the normal years' range and mean drawn beside them", "bool")),
              "region series of one field and one step"),
    PanelKind("bars", ("anomaly", "difference", "timing", "elevation_split"), "outputs",
              (_VALUE,),
              "one bar per output: a region's anomaly or difference, or a timing measure's "
              "median shift with its settings' range; or one elevation split, a bar per band"),
    PanelKind("overlap", ("overlap",), "output", (
        Option("by_band", "true: each band's correlation in place of the quarters", "bool"),),
              "one overlap: the first map's mean in each quarter of the second, with the "
              "correlation"),
    PanelKind("timing", ("timing",), "output", (),
              "one timing measure: its median shift at each stored setting, the default "
              "marked"),
    PanelKind(SCREEN, (SCREEN,), "none", (),
              "the plan's screen, drawn by its own display rule; a figure with it draws "
              "nothing else"),
)}


# ONE UNIT PER PANEL (the user's decision of 2026-11-08, preparing the phase's
# close): every figure that failed in the accepted fourth batch, eight of
# them, was a bars panel whose outputs were in different units (a season's
# total beside a year's, a fraction beside a proportion), seen only when
# drawing. The rule below is the one the check applies, the words the task
# shows (panel_lines) and the unit the brief lists beside each output
# (output_unit): a panel of several units is split into one panel per unit,
# in the order its outputs come, while the figure keeps to MAX_PANELS;
# otherwise the figure is refused before anything is drawn. Nothing is
# dropped, two units never share an axis, and no value is rescaled. An
# output that draws nothing on the axis (not made, not estimable, or a bar
# with no value) takes no unit, as the drawing's own check does (drawn_unit).
UNIT_KINDS = ("bars", "series")
UNIT_RULE = (f"a bars or series panel draws outputs of one unit, the unit the brief lists "
             f"beside each output (percent, for every output of a panel drawn with "
             f"\"value\": \"percent\"); a panel of several units is split into one panel per "
             f"unit, in the order its outputs come, while the figure keeps to {MAX_PANELS} "
             f"panels, and otherwise the figure is refused, naming the units")
NO_UNIT = "no unit"


def output_unit(r: Dict[str, Any], opts: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The unit an output is drawn in by a bars or series panel (`opts`, the
    panel's options): a series' own unit; an anomaly's or a difference's
    change unit, or "percent" for its percent; a timing measure's days; an
    elevation split's input map's unit. None for an output not made, of a
    kind no bars or series panel draws, or an elevation split whose input's
    unit is not recorded (a map read as it is)."""
    v, k = r.get("value"), str(r.get("kind"))
    if not isinstance(v, dict):
        return None
    if k == "series":
        return v.get("units") or NO_UNIT
    if k in PC.NULL_REGION:
        if (opts or {}).get("value") == "percent":
            return "percent"
        return v.get("change_units") or v.get("units") or NO_UNIT
    if k == "elevation_split":
        ins = list((r.get("out") or {}).get("inputs") or [])
        return (r.get("input_units") or {}).get(ins[0]) if ins else None
    if "variants" in v or "across_settings" in v:          # a timing measure
        return "days"
    return None


def drawn_unit(r: Optional[Dict[str, Any]], opts: Optional[Dict[str, Any]] = None
               ) -> Optional[str]:
    """The unit an output takes on its panel's axis: its output_unit, or
    None when it draws nothing there (not made, not estimable, or a bar
    with no value), so it neither splits a panel nor clashes on one."""
    if not r or not r.get("made") or r.get("outcome") == PC.NOT_ESTIMABLE:
        return None
    if r.get("kind") != "series" and _bar_value(r, opts or {})[0] is None:
        return None
    return output_unit(r, opts)


def split_by_unit(f: Dict[str, Any], refs: Dict[str, Dict[str, Any]]
                  ) -> Tuple[Dict[str, Any], List[str]]:
    """The figure with each bars or series panel of several units split into
    one panel per unit (UNIT_RULE), each split panel keeping the panel's
    options and the number of the panel it came from; or the figure as it
    was and the problems, when the split would give more than MAX_PANELS. An
    output that takes no unit on the axis (drawn_unit: not made, not
    estimable, no value) stays in its panel's first part, so a failure is
    still named when drawing. After a split every panel keeps the number it
    was asked as ("asked_as"), which a drawing failure names."""
    out: List[Tuple[int, Dict[str, Any]]] = []
    mixed: List[str] = []
    for i, p in enumerate(f.get("panels") or [], 1):
        names = panel_refs(p)
        if p.get("kind") not in UNIT_KINDS or len(names) < 2:
            out.append((i, p))
            continue
        groups: Dict[str, List[str]] = {}
        no_axis: List[str] = []
        for n in names:
            u = drawn_unit(refs.get(n), p)
            if u is None:
                no_axis.append(n)
            else:
                groups.setdefault(u, []).append(n)
        if len(groups) < 2:
            out.append((i, p))
            continue
        mixed.append(f"panel {i} ({p.get('kind')}): " + ", ".join(
            f"{u} ({', '.join(ns)})" for u, ns in groups.items()))
        for j, (u, ns) in enumerate(groups.items()):
            out.append((i, dict(p, outputs=ns + (no_axis if j == 0 else []), split_from=i)))
    if len(out) > MAX_PANELS:
        return f, [f"its outputs are in different units within one panel ({'; '.join(mixed)}), "
                   f"and one panel per unit would give it {len(out)} panels where a figure "
                   f"holds {MAX_PANELS}: put each unit's outputs in a panel of their own, in "
                   f"this figure or another"]
    if not mixed:
        return f, []
    return dict(f, panels=[dict(p, asked_as=i) for i, p in out]), []


class FigureFailed(ValueError):
    """A figure the framework cannot draw, with the panel and the reason."""


class NothingToDraw(ValueError):
    """A panel whose output answered with nothing to draw, in the method's
    own words: drawn as those words, never as an empty frame."""


def panel_refs(p: Dict[str, Any]) -> List[str]:
    """The outputs a panel names: its "output", its "outputs", or the screen."""
    k = PANEL_KINDS.get(str(p.get("kind")))
    if k is not None and k.takes == "none":
        return [SCREEN]
    if isinstance(p.get("outputs"), list):
        return [str(x) for x in p["outputs"]]
    return [str(p["output"])] if p.get("output") is not None else []


def _accepts(kind: PanelKind, out_kind: str, timing_ids: Sequence[str]) -> bool:
    return out_kind in kind.accepts or ("timing" in kind.accepts
                                        and out_kind in tuple(timing_ids))


def check_figure(f: Dict[str, Any], plan: Dict[str, Any],
                 timing_ids: Sequence[str]) -> List[str]:
    """What is wrong with a figure's specification, before anything is
    called or drawn: panels and options the kinds do not have, outputs the
    plan and the figure's own `extra` do not hold, an output of a kind its
    panel does not draw, and a screen panel beside anything else."""
    probs: List[str] = []
    fid = f.get("id")
    if f.get("code"):
        probs.append(f"{fid}: a figure here is a specification the framework draws; a "
                     f"script is not run")
    index = PC.plan_index(plan)
    extra = {}
    for x in f.get("extra") or []:
        if not isinstance(x, dict) or not EXTRA_ID.match(str(x.get("id"))):
            probs.append(f"{fid}: an extra output's id is X1, X2, ..., not "
                         f"{(x or {}).get('id') if isinstance(x, dict) else x!r}")
            continue
        d = PC.kind_of(x, timing_ids)
        if d is None:
            probs.append(f"{fid}: extra {x['id']}: no kind {x.get('kind')!r}")
            continue
        miss = [n for n in d["names"] if n not in x]
        if miss:
            probs.append(f"{fid}: extra {x['id']}: names no {', '.join(miss)}")
        for i in x.get("inputs") or []:
            if str(i) not in index and str(i) not in extra:
                probs.append(f"{fid}: extra {x['id']}: no input {i!r} (a plan output's label "
                             f"or an earlier extra's id)")
        extra[str(x["id"])] = x
    panels = f.get("panels")
    if not isinstance(panels, list) or not panels:
        return probs + [f"{fid}: no panels"]
    if len(panels) > MAX_PANELS:
        probs.append(f"{fid}: {len(panels)} panels; at most {MAX_PANELS}")
    for i, p in enumerate(panels, 1):
        where = f"{fid}, panel {i}"
        if not isinstance(p, dict) or p.get("kind") not in PANEL_KINDS:
            probs.append(f"{where}: no panel kind {(p or {}).get('kind') if isinstance(p, dict) else p!r}"
                         f" (the kinds: {', '.join(PANEL_KINDS)})")
            continue
        k = PANEL_KINDS[p["kind"]]
        allowed = {"kind", "output", "outputs"} | {o.name for o in k.options}
        for key in sorted(set(p) - allowed):
            probs.append(f"{where}: {k.id} has no option {key!r}"
                         + (f" (its options: {', '.join(o.name for o in k.options)})"
                            if k.options else ""))
        for o in k.options:
            if o.name not in p:
                continue
            v = p[o.name]
            ok = (v in o.allows if isinstance(o.allows, tuple) else
                  isinstance(v, bool) if o.allows == "bool" else
                  isinstance(v, list) and v and all(isinstance(y, int) and not
                                                    isinstance(y, bool) for y in v))
            if not ok:
                probs.append(f"{where}: {o.name}={v!r} is not "
                             + (" or ".join(map(repr, o.allows)) if isinstance(o.allows, tuple)
                                else "true or false" if o.allows == "bool"
                                else "a list of years"))
        if k.takes == "none":
            if len(panels) > 1 or list(f.get("plan_items") or []) != [SCREEN]:
                probs.append(f"{where}: the screen panel is its figure's only panel, and the "
                             f"figure's plan_items is [\"screen\"]")
            continue
        refs = panel_refs(p)
        if k.takes == "output" and (len(refs) != 1 or isinstance(p.get("outputs"), list)):
            probs.append(f"{where}: {k.id} names one \"output\"")
        if k.takes == "outputs" and not isinstance(p.get("outputs"), list):
            probs.append(f"{where}: {k.id} names a list of \"outputs\"")
        kinds = []
        for r in refs:
            o = index.get(r) or extra.get(r)
            if o is None:
                probs.append(f"{where}: no output {r!r} (a plan output's label such as P1.O1, "
                             f"or this figure's extra X1, ...)")
                continue
            ok_kind = str(o.get("kind"))
            kinds.append(ok_kind)
            if not _accepts(k, ok_kind, timing_ids):
                probs.append(f"{where}: {k.id} does not draw a {ok_kind} ({r})")
            elif k.id == "map" and ok_kind in PC.NULL_REGION and o.get("region") is not None:
                probs.append(f"{where}: {r} is a region's {ok_kind}, one number: draw it in "
                             f"bars; a map draws a per-cell {ok_kind} (its region null)")
            elif k.id == "bars" and ok_kind in PC.NULL_REGION and o.get("region") is None:
                probs.append(f"{where}: {r} is a per-cell {ok_kind}, a map: draw it in a map "
                             f"panel; bars draw a region's {ok_kind}")
        if k.id == "bars" and "elevation_split" in kinds and len(refs) > 1:
            probs.append(f"{where}: an elevation split is drawn alone in its bars, a bar per "
                         f"band")
        if k.id == "series":
            steps = {str((index.get(r) or extra.get(r) or {}).get("step")) for r in refs}
            if len(steps) > 1:
                probs.append(f"{where}: series of one step only, not {sorted(steps)}")
    return probs


# ── drawing ─────────────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "axes.titlesize": 13, "font.size": 13, "legend.fontsize": 12})
    return plt


def _short(o: Dict[str, Any]) -> str:
    """An output in the words of its own keys: field, kind, period, region."""
    k = str(o.get("kind"))
    bits = [str(o["field"])] if o.get("field") else []
    if k not in ("map", "series") or not bits:
        bits.append(k.replace("_", " "))
    for k in ("period", "step"):
        if o.get(k) is not None:
            bits.append(str(o[k]))
    if o.get("region"):
        bits.append(str(o["region"]))
    return " ".join(bits)


def _title(ax, refs: Sequence[str], outs: Dict[str, Dict[str, Any]]):
    """The panel's title from its outputs' own keys: each label and what it
    is; a region every output shares is said once, at the end."""
    os_ = [outs.get(r) or {"kind": r} for r in refs]
    regions = {o.get("region") for o in os_}
    shared = next(iter(regions)) if len(refs) > 1 and len(regions) == 1 else None
    t = "; ".join(("the plan's screen, " + " ".join(str(o[k]) for k in ("period", "region")
                                                    if o.get(k)))
                  if r == SCREEN else
                  f"{r}: {_short(dict(o, region=None) if shared else o)}"
                  for r, o in zip(refs, os_)) + (f"; {shared}" if shared else "")
    ax.set_title("\n".join(textwrap.wrap(t, 46)), loc="left")


def _note(ax, text: str):
    """A panel that says why it holds no picture, in the method's words."""
    ax.axis("off")
    ax.text(0.02, 0.5, "\n".join(textwrap.wrap(text, 40)), va="center", ha="left",
            transform=ax.transAxes, fontsize=13)


def _blank(where: str, why: str):
    raise FigureFailed(f"{where}: {why}")


def _finite(a) -> int:
    try:
        x = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return 0
    return int(np.isfinite(x).sum())


def _draw_map(plt, ax, r, opts, ll, where):
    v, k = r["value"], r["kind"]
    if k == "map":
        x, units, centred = v, str(r["out"].get("field") or ""), False
    else:
        pct = opts.get("value") == "percent"
        key = "percent" if pct else PC.MAP_PATH[k]
        x = (v or {}).get(key)
        if pct and x is None:
            _note(ax, f"no percent: {(v or {}).get('percent_why_not') or 'its field gives none'}")
            return 0
        units, centred = ("percent" if pct else str((v or {}).get("change_units")
                                                   or (v or {}).get("units") or "")), True
    n = _finite(x)
    if not n:
        _blank(where, f"{r['ref']} holds no cell with a value")
    a = np.asarray(x, dtype=float)
    kw = {"cmap": "viridis"}
    if centred:
        lim = float(np.nanmax(np.abs(a))) or 1.0
        kw = {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}
    m = ax.pcolormesh(ll["lon"], ll["lat"], a, shading="nearest", **kw)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label(units)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return n


def _seasonal(labels: Sequence[str], step: str):
    """(year, position on the seasonal axis) of each label."""
    out = []
    for lab in labels:
        s = str(lab)
        if step == "day":
            out.append((int(s[:4]), s[5:]))               # "MM-DD"
        elif step == "month":
            out.append((int(s[:4]), int(s[5:7])))
        elif step == "season":
            out.append((int(s[:4]), SEASON_ORDER.index(s[5:]) if s[5:] in SEASON_ORDER else -1))
        else:
            out.append((int(re.sub(r"\D", "", s)[:4]), 0))
    return out


def _draw_series(ax, rs, opts, normal_years, where):
    years, normal = opts.get("years"), bool(opts.get("normal"))
    n, units = 0, set()
    for r in rs:
        v = r["value"] or {}
        labs, vals = list(v.get("labels") or []), np.asarray(v.get("values"), dtype=float)
        step = str(v.get("step") or r["out"].get("step"))
        u = drawn_unit(r, opts)
        if u:
            units.add(u)
        name = f"{r['ref']} {r['out'].get('field')}"
        if not years and not normal:
            ok = np.isfinite(vals)
            n += int(ok.sum())
            ax.plot(np.arange(len(vals)), vals, lw=1.2, label=name)
            ticks = [i for i, lab in enumerate(labs) if str(lab)[5:] in ("01-01", "01", "DJF")
                     or step in ("year", "wateryear")]
            ax.set_xticks(ticks[:: max(1, len(ticks) // 9)])
            ax.set_xticklabels([str(labs[i])[:4] for i in ticks[:: max(1, len(ticks) // 9)]])
            continue
        pos = _seasonal(labs, step)
        keys = sorted({p for _, p in pos})
        at = {k: i for i, k in enumerate(keys)}
        by_year: Dict[int, Dict[Any, float]] = {}
        for (y, p), x in zip(pos, vals):
            by_year.setdefault(y, {})[p] = x
        if normal and normal_years:
            stack = np.array([[by_year.get(int(y), {}).get(k, np.nan) for k in keys]
                              for y in normal_years], dtype=float)
            if np.isfinite(stack).any():
                lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                ax.fill_between(range(len(keys)), lo, hi, alpha=0.25, color="0.5",
                                label=f"{name}, normal years' range")
                ax.plot(range(len(keys)), np.nanmean(stack, axis=0), color="0.3", ls="--",
                        lw=1.2, label=f"{name}, normal years' mean")
                n += int(np.isfinite(stack).sum())
        for y in years or []:
            row = np.array([by_year.get(int(y), {}).get(k, np.nan) for k in keys], dtype=float)
            if not np.isfinite(row).any():
                _blank(where, f"{r['ref']} holds no value in {y}")
            n += int(np.isfinite(row).sum())
            ax.plot(range(len(keys)), row, lw=1.8, label=f"{name}, {y}")
        step_ticks = max(1, len(keys) // 6)
        ax.set_xticks(range(0, len(keys), step_ticks))
        ax.set_xticklabels([str(keys[i]) if step != "season" else SEASON_ORDER[keys[i]]
                            for i in range(0, len(keys), step_ticks)])
    if not n:
        _blank(where, "the series hold no value to draw")
    if len(units) > 1:
        _blank(where, f"its series are in different units ({', '.join(sorted(units))})")
    ax.set_ylabel(", ".join(sorted(u for u in units if u != NO_UNIT)) or "value")
    ax.legend(loc="best")
    return n


def _of(r, i: int, what: str) -> str:
    """'<what> of <the input map's label>' with its units, for a panel drawn
    from an overlap's or an elevation split's input maps."""
    ins = list(r["out"].get("inputs") or [])
    units = (r.get("input_units") or {}).get(ins[i]) if i < len(ins) else None
    return f"{what} of {ins[i]}" + (f" ({units})" if units else "") if i < len(ins) else what


def _bar_value(r, opts) -> Tuple[Optional[float], Optional[Tuple[float, float]], str]:
    """A bar's height, its settings' range (a timing measure's) and its unit
    (output_unit, the rule the check splits panels by)."""
    v, k = r["value"] or {}, r["kind"]
    unit = output_unit(r, opts) or NO_UNIT
    if k in PC.NULL_REGION:
        if opts.get("value") == "percent":
            return v.get("percent"), None, unit
        return v.get(PC.MAP_PATH[k]), None, unit
    sh = (((v.get("variants") or {}).get("default") or {}).get("shift") or {}).get("median")
    ac = v.get("across_settings") or {}
    lo = (ac.get("minimum") or {}).get("median_shift")
    hi = (ac.get("maximum") or {}).get("median_shift")
    return sh, ((lo, hi) if lo is not None and hi is not None else None), unit


def _draw_bars(ax, rs, opts, where):
    if len(rs) == 1 and rs[0]["kind"] == "elevation_split":
        parts = (rs[0]["value"] or {}).get("parts") or {}
        names = [b for b, x in parts.items() if isinstance(x, dict)]
        hs = [parts[b].get("mean") for b in names]
        if not any(h is not None and np.isfinite(h) for h in hs):
            _blank(where, f"{rs[0]['ref']} holds no band with a value")
        ax.barh(range(len(names)), [h if h is not None else 0.0 for h in hs], color="0.45")
        for i, h in enumerate(hs):
            if h is None or not np.isfinite(h):
                ax.text(0, i, "no value", va="center")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([str(b) for b in names])
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(_of(rs[0], 0, "band mean"))
        return sum(1 for h in hs if h is not None and np.isfinite(h))
    vals, units = [], set()
    for r in rs:
        h, rng, u = _bar_value(r, opts)
        if r["outcome"] == PC.NOT_ESTIMABLE:
            h, rng = None, None
        vals.append((r, h, rng))
        if h is not None:
            units.add(u)
    if len(units) > 1:
        _blank(where, f"its bars are in different units ({', '.join(sorted(units))})")
    if not any(h is not None for _, h, _ in vals):
        return None
    for i, (r, h, rng) in enumerate(vals):
        if h is None:
            ax.text(i, 0, "not estimable" if r["outcome"] == PC.NOT_ESTIMABLE else "no value",
                    ha="center", va="bottom", rotation=90)
            continue
        ax.bar(i, h, color="0.45")
        if rng:
            ax.errorbar(i, h, yerr=[[h - min(rng)], [max(rng) - h]], color="k", capsize=6)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([r["ref"] for r, _, _ in vals])
    ax.axhline(0, color="k", lw=0.8)
    u = next(iter(units))
    ax.set_ylabel(u if u != NO_UNIT else "value")
    return sum(1 for _, h, _ in vals if h is not None)


def _draw_overlap(ax, r, opts, where):
    v = r["value"] or {}
    if opts.get("by_band"):
        bands = v.get("by_band") or {}
        names = list(bands)
        cs = [(bands[b] or {}).get("correlation") for b in names]
        if not names:
            _blank(where, f"{r['ref']} was made without its bands")
        if not any(c is not None for c in cs):
            return None
        ax.barh(range(len(names)), [c if c is not None else 0.0 for c in cs], color="0.45")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([str(b) for b in names])
        ax.invert_yaxis()
        ax.set_xlabel("correlation by band")
        ax.axvline(0, color="k", lw=0.8)
        return len([c for c in cs if c is not None])
    qs = v.get("quarters_by_second_map") or []
    if not qs:
        return None
    ms = [q.get("first_map_mean") for q in qs]
    ax.bar(range(len(qs)), [m if m is not None else 0.0 for m in ms], color="0.45")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"{q.get('second_map_from'):.3g} to\n{q.get('second_map_to'):.3g}"
                        for q in qs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(_of(r, 1, "quarter"))
    ax.set_ylabel(_of(r, 0, "mean"))
    if v.get("correlation") is not None:
        ax.text(0.98, 0.04, f"r = {v['correlation']:.3g}", transform=ax.transAxes, ha="right",
                va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    return int(v.get("cells") or 0)


def _draw_timing(ax, r, where):
    ac = (r["value"] or {}).get("across_settings") or {}
    ev = [e for e in ac.get("evaluated") or [] if isinstance(e, dict)]
    pts = [(str(e.get("variant")), e.get("median_shift")) for e in ev]
    if not any(s is not None for _, s in pts):
        return None
    for i, (lab, s) in enumerate(pts):
        if s is None:
            ax.text(0, i, "not estimable", va="center")
            continue
        ax.plot([s], [i], "o", ms=10, color="k" if lab == "default" else "0.5")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([lab for lab, _ in pts])
    ag = str((r["value"] or {}).get("against") or "the normal years")
    ax.set_xlabel("median shift, days, against " + re.sub(r" \[.*$", "", ag).replace(
        "the ", "", 1))
    return int((ac.get("comparable") or {}).get("cells") or 0)


def _draw_screen(ax, r, where):
    display = (r["value"] or {}).get("display") or {}
    disp = display.get("groups") or {}
    rows = [(g, x) for g, xs in disp.items() for x in xs or [] if isinstance(x, dict)]
    rows = [(g, x) for g, x in rows if x.get("score") is not None]
    if not rows:
        gone = [x for x in display.get("not_scored") or [] if isinstance(x, dict)]
        why = sorted({str(x.get("status") or x.get("why") or "not scored") for x in gone})
        raise NothingToDraw(f"the screen scored no row: {len(gone)} rows not scored"
                            + (f" ({'; '.join(why)})" if why else "")
                            + f"; {len(display.get('events') or [])} events")
    ax.barh(range(len(rows)), [x["score"] for _, x in rows], color="0.45")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{g}: {x.get('field')} {x.get('period')}" for g, x in rows])
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("score (the screen's rule)")
    return len(rows)


def render_figure(f: Dict[str, Any], refs: Dict[str, Dict[str, Any]], out_path,
                  lat_lon=None, normal_years: Sequence[int] = ()) -> Dict[str, Any]:
    """Draw one checked figure. `refs`: each output the figure names, by
    label, as {"ref", "kind", "out" (its spec), "value" (what its call
    returned), "made", "error", "outcome", "why", "line"}; `lat_lon` gives
    the grid's axes for a map. Returns {"ok", "error", "figure", "panels",
    "n"}; a panel that cannot be drawn fails the whole figure, naming it."""
    plt = _plt()
    panels = list(f.get("panels") or [])
    cols = 2 if len(panels) == 4 else len(panels)
    rows = 2 if len(panels) == 4 else 1
    height = 4.6
    if [p.get("kind") for p in panels] == [SCREEN]:   # a row per shown field, 13 pt each
        disp = (((refs.get(SCREEN) or {}).get("value") or {}).get("display") or {})
        height = max(height, 0.42 * sum(len(x or []) for x in (disp.get("groups") or {})
                                        .values()) + 1.5)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols if height == 4.6 else 9.0,
                                                  height * rows), squeeze=False)
    axes = [a for row in axes for a in row]
    record, total = [], 0
    try:
        for i, (p, ax) in enumerate(zip(panels, axes), 1):
            # the panel as the figure asked for it, after a split by unit
            where = f"panel {p.get('asked_as') or i} ({p.get('kind')})"
            names = panel_refs(p)
            rs = []
            for n in names:
                r = refs.get(n)
                if r is None or not r.get("made"):
                    _blank(where, f"{n} was not made: "
                                  f"{(r or {}).get('error') or 'no call made it'}")
                rs.append(r)
            outs = {r["ref"]: r["out"] for r in rs}
            _title(ax, names, outs)
            gone = [r for r in rs if r["outcome"] == PC.NOT_ESTIMABLE]
            row = {"panel": i, "kind": p.get("kind"), "outputs": names,
                   "records": [f"L{r['line']}" for r in rs if r.get("line")],
                   **({"split_from": p["split_from"]} if p.get("split_from") else {}),
                   **({"asked_as": p["asked_as"]} if p.get("asked_as") else {})}
            if gone and len(gone) == len(rs) and p.get("kind") != "bars":
                _note(ax, "not estimable: " + "; ".join(f"{r['ref']}: {r['why']}" for r in gone))
                record.append(dict(row, status="not estimable",
                                   why={r["ref"]: r["why"] for r in gone}))
                continue
            k = p.get("kind")
            if k == "map":
                if lat_lon is None:
                    _blank(where, "the grid's latitude and longitude were not read")
                got = _draw_map(plt, ax, rs[0], p, lat_lon, where)
            elif k == "series":
                got = _draw_series(ax, rs, p, normal_years, where)
            elif k == "bars":
                got = _draw_bars(ax, rs, p, where)
            elif k == "overlap":
                got = _draw_overlap(ax, rs[0], p, where)
            elif k == "timing":
                got = _draw_timing(ax, rs[0], where)
            else:
                try:
                    got = _draw_screen(ax, rs[0], where)
                except NothingToDraw as e:
                    _note(ax, str(e))
                    record.append(dict(row, status="not estimable", why={rs[0]["ref"]: str(e)}))
                    continue
            if got is None:
                _note(ax, "not estimable: " + "; ".join(
                    f"{r['ref']}: {r.get('why') or 'no value'}" for r in rs))
                record.append(dict(row, status="not estimable",
                                   why={r["ref"]: r.get("why") or "no value" for r in rs}))
                continue
            total += int(got or 0)
            record.append(dict(row, status="drawn", n=int(got or 0),
                               **({"not_estimable": [r["ref"] for r in gone]} if gone else {})))
        for ax in axes[len(panels):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except FigureFailed as e:
        plt.close(fig)
        return {"ok": False, "error": str(e), "figure": None, "panels": record, "n": total}
    plt.close(fig)
    return {"ok": True, "error": None, "figure": str(out_path), "panels": record, "n": total}


# ── the words step 2b is shown ──────────────────────────────────────────────

def _allows(o: Option) -> str:
    return (" | ".join(f'"{x}"' for x in o.allows) if isinstance(o.allows, tuple)
            else "true | false" if o.allows == "bool" else "[2015, ...]")


def panel_lines(timing_ids: Sequence[str]) -> List[str]:
    """THE PANELS step 2b may ask for, rendered from PANEL_KINDS: each kind,
    what it draws, the outputs it takes and its options."""
    L = textwrap.wrap(f"THE PANELS (at most {MAX_PANELS} a figure). Each panel is "
                      "{\"kind\": ..., and \"output\" (one label) or \"outputs\" (a list of "
                      "labels), and its options}; a label is a plan output's (P1.O1), or an "
                      "extra output's of this figure (X1):", 79, break_long_words=False,
                      break_on_hyphens=False)
    L += textwrap.wrap("ONE UNIT PER PANEL: " + UNIT_RULE + ".", 79, break_long_words=False,
                       break_on_hyphens=False)
    for k in PANEL_KINDS.values():
        acc = [("a timing measure (" + ", ".join(timing_ids) + ")") if a == "timing" else a
               for a in k.accepts]
        takes = {"output": "one \"output\"", "outputs": "\"outputs\", a list",
                 "none": "no output"}[k.takes]
        ops = "; ".join(f"\"{o.name}\": {_allows(o)} ({o.means})" for o in k.options)
        L += textwrap.wrap(f"  - \"{k.id}\": {k.means}. It draws {', '.join(acc)}; it takes "
                           f"{takes}" + (f"; options: {ops}" if ops else "") + ".", 79,
                           subsequent_indent="      ", break_long_words=False,
                           break_on_hyphens=False)
    return L
