#!/usr/bin/env python3
"""A whole-domain experiment's report, rendered from its records. No LLM.

docs/ELM_DOMAIN_EXPERIMENTS.md, step 5b. The numbers of a whole-domain
experiment, laid out like the Analyzer's report (step4_report): the question,
one headline, the numbers, what was run, the caveats, and where every number
came from. Written before the Analyzer runs and beside its REPORT.md, under
names of its own, so an Analyzer that fails or is switched off leaves this.

A RENDERER, NOT A REASONER, like the design review. Every number is read from
03_results/regional.json (domain_analysis, made from each run's own history)
or from experiment.json; nothing is computed here but the percentage a
recorded difference is of the control's recorded value. What a run's
difference MEANS is quoted from the model server's change list, with what it
does NOT mean beside it. Values are shown to 3 significant figures
(step4_report.format_value); recorded prose is shown with dashes used as
punctuation turned into commas (step4_report._plain); the records keep the
exact values and words.

  build(run_dir)    04_analysis/DOMAIN_REPORT.md, domain_report.json and
                    the domain_fig_*.png figures
  render(record)    REPORT.md from the report record. Pure.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.analysis.step4_report import _plain, format_value

REPORT_MD, REPORT_JSON, FIG_PREFIX = "DOMAIN_REPORT.md", "domain_report.json", "domain_fig_"
CONTROL = "control"
# The standard regions domain_analysis makes, and the suffix it gives a
# registered region (every region is restricted to the vegetated cells).
VEGETATED, BANDS, VEG_SUFFIX = "vegetated cells", "elevation bands", ", vegetated"
MAX_PANELS = 6


def _param_text(k, v) -> str:
    """A change's parameter in words: a map as "2016 to 2015", a list as
    "6, 7, 8", rather than a Python literal."""
    if isinstance(v, dict):
        return f"{k} " + ", ".join(f"{a} to {b}" for a, b in v.items())
    if isinstance(v, (list, tuple)):
        return f"{k} " + ", ".join(str(x) for x in v)
    return f"{k} {v}"


def _read(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text()) if p.is_file() else {}


def _fv(x, units: str = "") -> str:
    if x is None:
        return "n/a"
    s = format_value(x)
    return f"{s} {_shown(units)}".strip()


def _shown(units) -> str:
    """A unit as a reader reads it after a number: the CF unit "1" is a pure
    number, so nothing. The records keep the file's own string."""
    u = str(units or "").strip()
    return "" if u == "1" else u


def _pct(diff: Optional[float], base: Optional[float]) -> Optional[float]:
    if diff is None or base in (None, 0):
        return None
    return 100.0 * diff / base


def _signed(x: Optional[float], units: str = "") -> str:
    if x is None:
        return "n/a"
    return ("+" if x > 0 else "") + _fv(x, units)


def region_label(region) -> str:
    """A planned region in words: its name, or "elevation bands within
    watershed" for {"name": "elevation bands", "within": "watershed"}."""
    if isinstance(region, dict):
        n, w = str(region.get("name") or "").strip(), region.get("within")
        return f"{n} within {str(w).strip()}" if w else n
    return str(region).strip()


def region_matches(region, have: List[str]) -> List[str]:
    """The table's regions one planned region stands for: a plan names
    "watershed" and the table holds "watershed, vegetated"; "elevation bands"
    stands for every band the table holds over the domain, and with within W
    for every band inside W ("W, vegetated, 1000-1500 m"). [] when the table
    has none. `region` is a name or the plan's {"name", "within"} object."""
    n = str(region.get("name") if isinstance(region, dict) else region).strip()
    w = str(region.get("within") or "").strip() if isinstance(region, dict) else ""
    if n.lower() in (BANDS, "elevation_bands", "bands"):
        if w:
            return [k for k in have if k.startswith(f"{w}{VEG_SUFFIX}, ")]
        return [k for k in have if k != VEGETATED and VEG_SUFFIX not in k]
    if w:
        return []
    if n.lower() in (VEGETATED, "vegetated"):
        return [VEGETATED] if VEGETATED in have else []
    if n in have:
        return [n]
    return [n + VEG_SUFFIX] if n + VEG_SUFFIX in have else []


def region_keys(table: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
    """The table's regions in the plan's order (region_matches), or every
    region the table holds when the plan names none it has."""
    have = list((table.get("regions") or {}).keys())
    out: List[str] = []
    for r in analysis.get("regions") or []:
        out += region_matches(r, have)
    seen, keep = set(), []
    for k in out or have:
        if k not in seen:
            seen.add(k)
            keep.append(k)
    return keep


def _fields(table: Dict[str, Any]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in (table.get("runs") or {}).values():
        for v, meta in (r.get("fields") or {}).items():
            out.setdefault(v, meta)
    return out


def _val(table, run, region, var, year) -> Optional[float]:
    return ((((table.get("runs") or {}).get(run) or {}).get("values") or {}).get(region)
            or {}).get(var, {}).get(str(year))


def _diff(table, run, region, var, year) -> Optional[float]:
    return ((((table.get("differences") or {}).get(run) or {}).get(region) or {})
            .get(var) or {}).get(str(year))


def _what(meta: Dict) -> str:
    return "total over the year" if meta.get("how") == "sum" else "mean over the year"


def _changes_text(run: Dict) -> str:
    out = []
    for c in run.get("changes") or []:
        p = c.get("params") or {}
        out.append(c.get("id", "?") + (" " + ", ".join(_param_text(k, v)
                                                       for k, v in p.items())
                                       if p else ""))
    return ", ".join(out) or "no change"


def headline(table: Dict[str, Any], runs: List[Dict], regions: List[str],
             variables: List[str], years: List[int]) -> Tuple[str, Dict]:
    """One claim, its numbers the claim's own: for a what-if experiment the
    first run's difference from the control, for an event the target years
    against the normal, otherwise the control's values; always the plan's
    first variable over its first region."""
    fields = _fields(table)
    var = next((v for v in variables if v in fields), None)
    reg = regions[0] if regions else None
    if var is None or reg is None:
        return "The experiment produced no regional numbers to report.", {}
    units = fields[var].get("units", "")
    changed = [r for r in runs if not r.get("reused") and r.get("run") != CONTROL]
    if changed and changed[0]["run"] in (table.get("differences") or {}):
        run = changed[0]
        parts, data = [], {}
        for y in years:
            d, c = _diff(table, run["run"], reg, var, y), _val(table, CONTROL, reg, var, y)
            if d is None:
                continue
            p = _pct(d, c)
            data[str(y)] = {"difference": d, "control": c, "percent": p}
            parts.append(f"{_signed(d, units)} in {y}"
                         + (f" ({_signed(p)} %)" if p is not None else ""))
        if parts:
            return (f"With {_changes_text(run)} (run {run['run']}), {var} over {reg} "
                    f"differed from the control by {_join(parts)}.",
                    {"kind": "difference", "run": run["run"], "variable": var,
                     "region": reg, "units": units, "years": data})
    ev = (table.get("event") or {})
    a = ((ev.get("anomalies") or {}).get(reg) or {}).get(var)
    if a:
        n = ev.get("normal_years") or []
        parts, data = [], {}
        for y in ev.get("target_years") or []:
            yy = _year(a, y)
            # a percent where the record has one; an anomaly in the field's
            # units where it has none (a temperature in K)
            p, x = yy.get("anomaly_pct"), yy.get("anomaly")
            if x is None:
                continue
            what = f"its {span(n)} normal ({_fv(a['normal'], units)})" if not parts else "it"
            if x == 0:
                parts.append(f"equal to {what} in {y}")
            else:
                amount = f"{_fv(abs(p))} %" if p is not None else _fv(abs(x), units)
                parts.append(f"{amount} {'above' if x > 0 else 'below'} {what} in {y}")
            data[str(y)] = yy
        if parts:
            return (f"{var} over {reg} was {_join(parts)}.",
                    {"kind": "event", "variable": var, "region": reg, "units": units,
                     "normal": a["normal"], "normal_years": n, "years": data})
    vals = [(y, _val(table, CONTROL, reg, var, y)) for y in years]
    vals = [(y, v) for y, v in vals if v is not None]
    return (f"{var} over {reg}: " + _join([f"{_fv(v, units)} in {y}" for y, v in vals]) + ".",
            {"kind": "values", "variable": var, "region": reg, "units": units,
             "years": {str(y): v for y, v in vals}})


def _join(parts: List[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def span(years) -> str:
    """Years as a reader says them: 2016, 2016-2020, or 2016, 2018 and 2020."""
    ys = sorted(int(y) for y in years or [])
    if not ys:
        return "?"
    if len(ys) == 1:
        return str(ys[0])
    if ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]}-{ys[-1]}"
    return _join([str(y) for y in ys])


def _year(a: Dict, y) -> Dict:
    """One target year of an anomaly record; its keys are ints in process and
    strings once the record has been through JSON."""
    ys = a.get("years") or {}
    return ys.get(y) or ys.get(str(y)) or {}


def _tables(table, runs, regions, variables, years) -> List[str]:
    fields = _fields(table)
    changed = [r for r in runs if not r.get("reused") and r.get("run") != CONTROL]
    ev = table.get("event") or {}
    lines: List[str] = []
    for var in [v for v in variables if v in fields]:
        meta = fields[var]
        units = meta.get("units", "")
        lines += [f"### {var} ({_shown(units) or 'dimensionless'}, {_what(meta)})", ""]
        if meta.get("note"):
            lines += [f"*{_plain(meta['note'])}*", ""]
        head = "| region | " + " | ".join(str(y) for y in years) + " |"
        rule = "|---|" + "---|" * len(years)
        lines += ["The control:", "", head, rule]
        for reg in regions:
            lines.append(f"| {reg} | " + " | ".join(_fv(_val(table, CONTROL, reg, var, y))
                                                   for y in years) + " |")
        lines.append("")
        for run in changed:
            lines += [f"Run {run['run']} minus the control (percent of the control):", "",
                      head, rule]
            for reg in regions:
                cells = []
                for y in years:
                    d = _diff(table, run["run"], reg, var, y)
                    p = _pct(d, _val(table, CONTROL, reg, var, y))
                    cells.append(_signed(d) + (f" ({_signed(p)} %)" if p is not None else ""))
                lines.append(f"| {reg} | " + " | ".join(cells) + " |")
            lines.append("")
        if ev.get("normal_years") and (ev.get("anomalies") or {}):
            tys = ev.get("target_years") or []
            lines += [f"The control against its {span(ev['normal_years'])} normal (the "
                      f"anomaly, and the year's rank among the years with a value, 1 the "
                      f"lowest):", "",
                      "| region | normal | " + " | ".join(str(y) for y in tys) + " |",
                      "|---|---|" + "---|" * len(tys)]
            for reg in regions:
                a = ((ev["anomalies"].get(reg) or {}).get(var)) or {}
                cells = []
                for y in tys:
                    yy = _year(a, y)
                    p, rank = yy.get("anomaly_pct"), yy.get("rank_lowest_first")
                    cells.append((_signed(p) + " %" if p is not None else _signed(
                        yy.get("anomaly"))) + (f" (rank {rank} of {yy.get('of')})"
                                               if rank else ""))
                lines.append(f"| {reg} | {_fv(a.get('normal'))} | " + " | ".join(cells) + " |")
            lines.append("")
    # AN OBSERVATION BESIDE THE MODEL'S OWN, as the server put them: the
    # observed values per region and year, and each run's own value of the
    # same quantity (for a burned area, the model's burned fraction times the
    # region's area, made in the server, which knows the field).
    for name, ob in (table.get("observed") or {}).items():
        model = ob.get("model") or {}
        names = [r for r in [CONTROL] + [c["run"] for c in changed] if r in model]
        lines += [f"### {str(ob.get('what') or name).capitalize()}: the model's own and "
                  f"the observed ({ob.get('units') or '?'})", "",
                  f"*observed: {_plain(ob.get('source'))}*", "",
                  "| region | year | " + " | ".join(
                      "model" if r == CONTROL and len(names) == 1 else f"model, {r}"
                      for r in names) + " | observed |",
                  "|---|---|" + "---|" * len(names) + "---|"]
        for reg in regions:
            for y in years:
                ms = [_fv(((model.get(r) or {}).get(reg) or {}).get(str(y))) for r in names]
                o = ((ob.get("values") or {}).get(reg) or {}).get(str(y))
                lines.append(f"| {reg} | {y} | " + " | ".join(ms) + f" | {_fv(o)} |")
        lines.append("")
    return lines


def _caveats(exp, plan, table) -> List[str]:
    out: List[str] = []
    for r in ((exp.get("experiment") or {}).get("runs") or []):
        for c in r.get("changes") or []:
            if c.get("not_meaning"):
                out.append(f"Run {r['run']}, {c.get('id')}: its difference from the control "
                           f"is NOT {c['not_meaning']}")
    out += [str(c) for c in (plan.get("caveats") or [])]
    for u in (((exp.get("experiment") or {}).get("check") or {}).get("unusual") or []):
        out.append(f"{u.get('where')}: {u.get('what')}")
    for v, meta in _fields(table).items():
        if meta.get("note"):
            out.append(f"{v}: {meta['note']}")
    out += [f"Not reported: {p}" for p in table.get("problems") or []]
    for run, r in (table.get("runs") or {}).items():
        out += [f"Run {run}, not reported: {p}" for p in r.get("problems") or []]
    seen, keep = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            keep.append(c)
    return keep


def _status(exp) -> str:
    """Complete only when every planned year passed: the notifier's rule
    (tools/notify_study.py), so the page and the mail agree."""
    ch = exp.get("chain") or {}
    planned, passed = ch.get("years_planned") or [], ch.get("years_passed") or []
    if ch.get("first_unpassed") or not planned or sorted(planned) != sorted(passed):
        years = f"years passed {passed} of {planned}"
        return (f"incomplete: the chain stopped at {ch['first_unpassed']} ({years})"
                if ch.get("first_unpassed") else f"incomplete: {years}")
    return "complete"


def _figure_style():
    try:
        import figstyle
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        import figstyle
    return figstyle


def _label(reg: str) -> str:
    """A region as a figure names it. Every region is restricted to the
    vegetated cells, so ", vegetated" is not repeated on each label:
    "watershed, vegetated, 1000-1500 m" is "watershed, 1000-1500 m"."""
    return reg if reg == VEGETATED else reg.replace(VEG_SUFFIX, "")


def _no_percent(units) -> bool:
    """A field in kelvin has no percent: +2 K is not 0.7 % of anything a
    reader means. The server's own rule for an anomaly (domain_analysis.event)."""
    return str(units or "").strip().upper().startswith("K")


def _pct_of(diff: Optional[float], base: Optional[float]) -> Optional[float]:
    """A percent the figure may draw: none over a base at or below zero, where
    a percent flips the sign (a sink weakened from -100 by +10 is not -10 %)."""
    if diff is None or base is None or base <= 0:
        return None
    return 100.0 * diff / base


def _bars(ax, xs, vals, width, **kw) -> None:
    """Bars at xs, a missing value drawn as no bar with "n/a" on the zero line.
    MISSING IS NOT ZERO (2026-11-03): a missing value drawn at 0 read as "no
    change" on the slide."""
    ax.bar(xs, [float("nan") if x is None else x for x in vals], width=width, **kw)
    for x, v in zip(xs, vals):
        if v is None:
            ax.text(x, 0.0, "n/a", ha="center", va="bottom", rotation=90,
                    fontsize="x-small", color="0.3")


def figures(table, runs, regions, var, years, out_dir: Path) -> List[str]:
    """The headline variable by year, a panel per region, a line per run; and
    each changed run's difference from the control, or the control's anomaly
    against its normal. Measurements only: no claim is written on a plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = _figure_style()
    meta = _fields(table).get(var) or {}
    units = meta.get("units", "")
    regs = regions[:MAX_PANELS]
    names = [CONTROL] + [r["run"] for r in runs
                         if not r.get("reused") and r.get("run") != CONTROL
                         and r["run"] in (table.get("runs") or {})]
    paths = []
    fs.manuscript(fs.WIDTH["double"], base_pt=9)
    ncol = min(3, len(regs)) or 1
    nrow = (len(regs) + ncol - 1) // ncol or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(fs.WIDTH["double"], 2.2 * nrow),
                             constrained_layout=True, squeeze=False)
    for k, reg in enumerate(regs):
        ax = axes[k // ncol][k % ncol]
        for n in names:
            ys = [(y, _val(table, n, reg, var, y)) for y in years]
            ys = [(y, v) for y, v in ys if v is not None]
            if ys:
                ax.plot([y for y, _ in ys], [v for _, v in ys], marker="o", label=n)
        # a band within a region: the region over the band, on two lines
        ax.set_title(_label(reg).replace(", ", "\n", 1))
        ax.set_xticks(years)
        ax.set_xticklabels([str(y)[2:] for y in years])
        if k % ncol == 0:
            ax.set_ylabel(_shown(units) or "dimensionless")
    for k in range(len(regs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if len(names) > 1:
        axes[0][0].legend()
    fig.suptitle(f"{var}, {_what(meta)}")
    fs.check_titles(fig)
    p = out_dir / f"{FIG_PREFIX}{var}_by_year.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))
    changed = names[1:]
    ev = table.get("event") or {}
    if changed or ev.get("anomalies"):
        fig, ax = plt.subplots(figsize=(fs.WIDTH["double"], 2.6), constrained_layout=True)
        width = 0.8 / max(1, len(regs))
        # a difference in every year, an anomaly in the target years only: one
        # evenly spaced group of bars per year shown, the year beneath it
        xs = years if changed else [int(y) for y in ev.get("target_years") or []]
        for k, reg in enumerate(regs):
            if changed:
                vs = [_diff(table, changed[0], reg, var, y) for y in xs]
            else:
                a = ((ev["anomalies"].get(reg) or {}).get(var)) or {}
                vs = [_year(a, y).get("anomaly") for y in xs]
            _bars(ax, [i - 0.4 + width * (k + 0.5) for i in range(len(xs))], vs, width,
                  label=_label(reg))
        ax.axhline(0.0, color="0.3", linewidth=0.6)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(y) for y in xs])
        ax.set_ylabel(_shown(units) or "dimensionless")
        # outside the axes, in space the layout keeps for it: a legend
        # anchored below the axes is cut off at the canvas edge
        fig.legend(ncol=min(3, len(regs)), loc="outside lower center", frameon=False)
        ax.set_title(f"{var}: run {changed[0]} minus the control" if changed else
                     f"{var}: the control minus its {span(ev['normal_years'])} normal")
        fs.check_titles(fig)
        p = out_dir / (f"{FIG_PREFIX}{var}_{'difference' if changed else 'anomaly'}.png")
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(str(p))
    return paths


MAX_OVERVIEW = 5


def overview_figures(table, runs, regions, variables, years, out_dir: Path) -> List[str]:
    """Every variable at a glance, for the first region the plan reports, in
    talk-sized type on one row (2026-10-27: the deck showed one figure of one
    variable for a question about carbon AND water). Two figures:

      overview   each year of each run, one panel per variable, the normal-
                 year mean dashed when the question names normal years
      changes    every variable in one panel: each event year against its
                 normal, or each changed run minus the control, in percent;
                 a kelvin field is left out, and a missing percent is n/a

    Measurements only: no claim is written on a plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = _fields(table)
    vs = [v for v in variables if v in fields][:MAX_OVERVIEW]
    if not regions or not vs:
        return []
    fs = _figure_style()
    reg = regions[0]
    names = [CONTROL] + [r["run"] for r in runs
                         if not r.get("reused") and r.get("run") != CONTROL
                         and r["run"] in (table.get("runs") or {})]
    ev = table.get("event") or {}
    normal = ev.get("normal_years") or []
    paths: List[str] = []
    fs.manuscript(fs.WIDTH["double"], base_pt=10)

    fig, axes = plt.subplots(1, len(vs), figsize=(fs.WIDTH["double"], 2.3),
                             constrained_layout=True, squeeze=False)
    # a narrow panel holds every other year's label, and a long field name in
    # a smaller face, so neither runs into its neighbour
    ticks = years if len(years) <= 5 else years[::2]
    for ax, v in zip(axes[0], vs):
        for n in names:
            ys = [(y, _val(table, n, reg, v, y)) for y in years]
            ys = [(y, x) for y, x in ys if x is not None]
            if ys:
                ax.plot([y for y, _ in ys], [x for _, x in ys], marker="o", label=n)
        a = ((ev.get("anomalies") or {}).get(reg) or {}).get(v) or {}
        if normal and a.get("normal") is not None:
            ax.axhline(a["normal"], color="0.4", linestyle="--", linewidth=1,
                       label=f"{span(normal)} mean")
        ax.set_title(v, fontsize=("x-small" if len(v) > 10 else
                                  "small" if len(v) > 8 else None))
        ax.set_ylabel(_shown((fields.get(v) or {}).get("units")) or "dimensionless")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(y) if len(ticks) <= 3 else str(y)[2:] for y in ticks])
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(labels) > 1:
        fig.legend(handles, labels, ncol=len(labels), loc="outside lower center",
                   frameon=False)
    fig.suptitle(_label(reg))
    fs.check_titles(fig)
    p = out_dir / f"{FIG_PREFIX}overview.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    paths.append(str(p))

    # ONLY WHAT A PERCENT MEANS (2026-11-03): a kelvin field drew +2 K as
    # +0.7 % of absolute temperature, so it is left out of the percent panel;
    # a bar over a control or normal at or below zero, or with no value, is
    # drawn as no bar marked n/a (_pct_of, _bars), never as 0.
    changed = names[1:]
    pv = [v for v in vs if not _no_percent((fields.get(v) or {}).get("units"))]
    series: List[Tuple[str, List[Optional[float]]]] = []
    if changed:
        for n in changed:
            for y in years:
                vals = []
                for v in pv:
                    d, c = _diff(table, n, reg, v, y), _val(table, CONTROL, reg, v, y)
                    vals.append(_pct_of(d, c))
                series.append((f"{n}, {y}" if len(changed) > 1 else str(y), vals))
        ylab = "run minus the control, % of the control"
    elif ev.get("anomalies"):
        for y in ev.get("target_years") or []:
            vals = []
            for v in pv:
                a = (ev["anomalies"].get(reg) or {}).get(v) or {}
                d = _year(a, y).get("anomaly_pct")
                n0 = a.get("normal")
                vals.append(d if d is not None and n0 is not None and n0 > 0 else None)
            series.append((str(y), vals))
        ylab = f"% of the {span(normal)} mean"
    if pv and any(x is not None for _, vals in series for x in vals):
        fig, ax = plt.subplots(figsize=(fs.WIDTH["double"], 2.5), constrained_layout=True)
        width = 0.8 / len(series)
        for k, (lab, vals) in enumerate(series):
            _bars(ax, [i - 0.4 + width * (k + 0.5) for i in range(len(pv))], vals, width,
                  label=lab)
        ax.axhline(0.0, color="0.3", linewidth=0.6)
        ax.set_xticks(range(len(pv)))
        ax.set_xticklabels(pv)
        ax.set_ylabel(ylab)
        ax.set_title(_label(reg))
        fig.legend(ncol=len(series), loc="outside lower center", frameon=False)
        fs.check_titles(fig)
        p = out_dir / f"{FIG_PREFIX}changes.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(str(p))
    return paths


# MONTHS, SEASONS AND WHERE THE NUMBERS COME FROM (2026-11-03). The page showed
# each calendar year's value only, so a question about a summer was answered
# from a year that holds its winters too. The server now makes each run's
# months and seasons (domain_analysis.months_and_seasons); the page draws the
# months of up to four fields per region and writes every month and season to
# a table of its own, linked, rather than into the page. It also says, from
# the server's provenance block, which files each run's numbers were read
# from and what each region is drawn from. Nothing is computed here: every
# value is the server's, copied or drawn.
MONTHLY_CSV = "regional_monthly.csv"
MAX_MONTHLY_FIELDS = 4
MONTH_LETTERS = "JFMAMJJASOND"


def _periods(table, run, kind) -> Dict[str, Dict]:
    """A run's months (kind "monthly") or seasons ("seasonal"): {region:
    {field: {period: value}}}, {} when the table has none."""
    return (((table.get("runs") or {}).get(run) or {}).get(kind) or {})


def _period_units(table, var, period) -> str:
    """A field's units over a month or a season, as the server recorded them."""
    for r in (table.get("runs") or {}).values():
        u = ((r.get("period_units") or {}).get(var) or {}).get(period)
        if u is not None:
            return u
    return ""


def has_months(table) -> bool:
    return any(r.get("monthly") for r in (table.get("runs") or {}).values())


def monthly_rows(table) -> List[Dict[str, Any]]:
    """Every month and season of every run, region and field, with each
    run's difference from the control and the control's normal-year means,
    as rows: the full table the page links to. Copied from the records."""
    rows: List[Dict[str, Any]] = []
    for run, r in (table.get("runs") or {}).items():
        for kind, diffs, period in (("monthly", "monthly_differences", "month"),
                                    ("seasonal", "seasonal_differences", "season")):
            d = ((table.get(diffs) or {}).get(run) or {})
            for reg, fields in (r.get(kind) or {}).items():
                for var, vals in fields.items():
                    units = _period_units(table, var, period)
                    for p, x in vals.items():
                        rows.append({"run": run, "region": reg, "variable": var,
                                     "period_kind": period, "period": p, "value": x,
                                     "units": units,
                                     "minus_control": ((d.get(reg) or {}).get(var) or {})
                                     .get(p)})
    ev = table.get("event") or {}
    if ev.get("normal_years"):
        tag = f"mean of {span(ev['normal_years'])}"
        # a normal over fewer years (the DJF after an event year is left out,
        # 2026-11-03) names the years it is the mean of
        used = ev.get("normal_years_used") or {}
        for kind, period in (("monthly_normal", "month"), ("seasonal_normal", "season")):
            for reg, fields in (ev.get(kind) or {}).items():
                for var, vals in fields.items():
                    for p, x in vals.items():
                        ptag = (f"mean of {span(used[p])}" if used.get(p) else
                                "mean of no year" if p in used else tag)
                        rows.append({"run": CONTROL, "region": reg, "variable": var,
                                     "period_kind": f"{period}, {ptag}", "period": p,
                                     "value": x,
                                     "units": _period_units(table, var, period),
                                     "minus_control": None})
    return rows


def monthly_csv(table, path: Path) -> Optional[str]:
    """03_results/regional_monthly.csv from monthly_rows, or None when the
    table holds no months (a table made before 2026-11-03)."""
    import csv
    rows = monthly_rows(table)
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run", "region", "variable", "period_kind", "period", "value", "units",
            "minus_control"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    return str(path)


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower() or "region"


def _month_axis(ax, years: List[int]) -> None:
    """Months along the x axis: a letter per month for one year, the year at
    each January for more."""
    n = 12 * len(years)
    if len(years) == 1:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(MONTH_LETTERS))
    else:
        ax.set_xticks(range(0, n, 12))
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xticks(range(0, n, 3), minor=True)
    ax.set_xlim(-0.5, n - 0.5)


def monthly_figures(table, runs, regions, variables, years, out_dir: Path) -> List[str]:
    """The months of up to four fields (the plan's first ones the table has
    months of), one figure per region (at most MAX_PANELS):

      runs       each run and the control through every month of the years,
                 and beneath, each run minus the control
      an event   the control alone with normal years: each target year's
                 months against the normal years' mean of the same month
      otherwise  the control through every month

    A month without a value is a gap, never a zero. Measurements only: no
    claim is written on a plot."""
    if not has_months(table) or not regions:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = _figure_style()
    ctl = _periods(table, CONTROL, "monthly")
    vs = [v for v in variables if any(v in (ctl.get(r) or {}) for r in regions)]
    vs = vs[:MAX_MONTHLY_FIELDS]
    if not vs:
        return []
    names = [CONTROL] + [r["run"] for r in runs
                         if not r.get("reused") and r.get("run") != CONTROL
                         and r["run"] in (table.get("runs") or {})]
    changed = names[1:]
    ev = table.get("event") or {}
    event = (not changed and ev.get("normal_years") and ev.get("monthly_normal")
             and ev.get("target_years"))
    keys = [f"{int(y)}-{m:02d}" for y in years for m in range(1, 13)]

    def _series(run, reg, var, ks):
        vals = (_periods(table, run, "monthly").get(reg) or {}).get(var) or {}
        return [float("nan") if vals.get(k) is None else vals[k] for k in ks]

    paths: List[str] = []
    fs.manuscript(fs.WIDTH["double"], base_pt=9)
    for reg in regions[:MAX_PANELS]:
        if not any(var in (ctl.get(reg) or {}) for var in vs):
            continue
        nrow = 2 if changed else 1
        fig, axes = plt.subplots(nrow, len(vs), figsize=(fs.WIDTH["double"], 2.2 * nrow),
                                 constrained_layout=True, squeeze=False)
        for j, var in enumerate(vs):
            units = _shown(_period_units(table, var, "month")) or "dimensionless"
            ax = axes[0][j]
            if event:
                normal = ((ev["monthly_normal"].get(reg) or {}).get(var)) or {}
                for y in ev["target_years"]:
                    ax.plot(range(12), _series(CONTROL, reg, var,
                                               [f"{int(y)}-{m:02d}" for m in range(1, 13)]),
                            marker="o", markersize=3, label=str(y))
                ax.plot(range(12), [float("nan") if normal.get(f"{m:02d}") is None
                                    else normal[f"{m:02d}"] for m in range(1, 13)],
                        color="0.4", linestyle="--",
                        label=f"{span(ev['normal_years'])} mean")
                _month_axis(ax, [0])
            else:
                for k, n in enumerate(names):
                    ax.plot(range(len(keys)), _series(n, reg, var, keys), label=n,
                            color=f"C{k}")
                _month_axis(ax, years)
            ax.set_title(var, fontsize=("x-small" if len(var) > 10 else
                                        "small" if len(var) > 8 else None))
            ax.set_ylabel(units)
            if changed:
                bx = axes[1][j]
                # each run's difference in the run's own colour above
                for i, n in enumerate(changed, start=1):
                    d = ((((table.get("monthly_differences") or {}).get(n) or {})
                          .get(reg) or {}).get(var)) or {}
                    bx.plot(range(len(keys)), [float("nan") if d.get(k) is None else d[k]
                                               for k in keys], color=f"C{i}")
                bx.axhline(0.0, color="0.3", linewidth=0.6)
                _month_axis(bx, years)
                if j == 0:
                    bx.set_ylabel(f"minus the control\n{units}")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if len(labels) > 1:
            fig.legend(handles, labels, ncol=min(4, len(labels)),
                       loc="outside lower center", frameon=False)
        fig.suptitle(_label(reg) + (", by month" if not event else ", by month of the year"))
        fs.check_titles(fig)
        p = out_dir / f"{FIG_PREFIX}monthly_{_slug(_label(reg))}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _mb(n) -> str:
    return f"{format_value(float(n) / 1e6)} MB" if n is not None else "? MB"


def provenance_lines(table, csv_link: Optional[str] = None) -> List[str]:
    """Which files each run's numbers were read from, and what each region is
    drawn from, from the server's provenance block. [] for a table without
    one (made before 2026-11-03)."""
    pv = table.get("provenance") or {}
    out: List[str] = []
    if csv_link:
        out += [f"- every month and season of every run, region and field, with each "
                f"run's difference from the control: [{csv_link}](../{csv_link})"]
    if not pv:
        return out + ([""] if out else [])
    out += ["", "### What each run was read from", ""]
    for run, r in (pv.get("runs") or {}).items():
        out.append(f"- run **{run}**: `{r.get('run_dir')}`")
        for st, s in (r.get("streams") or {}).items():
            per = s.get("records_per_year") or {}
            counts = ", ".join(f"{n} in {y}" for y, n in per.items())
            files = s.get("files") or []
            out.append(f"  - stream {st} ({s.get('cadence')}, {s.get('calendar')} calendar): "
                       f"{', '.join(s.get('fields') or [])}; records read: {counts or 'none'}; "
                       f"{len(files)} file{'' if len(files) == 1 else 's'}"
                       + (f", {s['records_left_out']} record(s) left out by the reader"
                          if s.get("records_left_out") else ""))
            for f in files:
                out.append(f"    - {f.get('name')} ({_mb(f.get('bytes'))}, modified "
                           f"{str(f.get('modified') or '?').replace('T', ' ').replace('+00:00', ' UTC')})")
    regs = pv.get("regions") or {}
    if regs:
        out += ["", "### What each region is drawn from", "",
                "| region | cells | area (km2) | rule | drawn from |", "|---|---|---|---|---|"]
        for n, g in regs.items():
            out.append(f"| {n} | {g.get('cells')} | {_fv(g.get('area_km2'))} | "
                       f"{g.get('rule')} | {', '.join(g.get('sources') or [])} |")
    if pv.get("surface_file"):
        out += ["", f"The vegetated cells and the elevations are read from the control's "
                    f"surface file, {Path(str(pv['surface_file'])).name}."]
    return out + [""]


def build(run_dir) -> Dict[str, Any]:
    """The report of the whole-domain experiment in run_dir: report.json (the
    record, with every number the text uses), REPORT.md and the figures, in
    04_analysis/. Raises when the run holds no regional table."""
    rd = Path(run_dir)
    exp = _read(rd / "experiment.json")
    strategy = _read(rd / "strategy.json")
    reception = _read(rd / "reception.json")
    table = exp.get("regional") or _read(rd / "03_results" / "regional.json")
    if not table or table.get("error"):
        raise ValueError("no regional table to report: "
                         + str((table or {}).get("error") or "03_results/regional.json absent"))
    plan = strategy.get("domain_experiment") or {}
    analysis = plan.get("analysis") or {}
    runs = (exp.get("experiment") or {}).get("runs") or []
    years = [int(y) for y in table.get("years") or []]
    regions = region_keys(table, analysis)
    variables = [v for v in (analysis.get("variables") or []) if v in _fields(table)] or \
        list(_fields(table))
    text, claim = headline(table, runs, regions, variables, years)
    out = rd / "04_analysis"
    out.mkdir(parents=True, exist_ok=True)
    figs: List[str] = []
    try:
        if claim.get("variable"):
            figs = figures(table, runs, regions, claim["variable"], years, out)
    except Exception as e:                                          # noqa: BLE001
        figs = []
        claim["figure_error"] = f"{type(e).__name__}: {e}"
    try:
        figs += overview_figures(table, runs, regions, variables, years, out)
    except Exception as e:                                          # noqa: BLE001
        claim["overview_error"] = f"{type(e).__name__}: {e}"
    try:
        figs += monthly_figures(table, runs, regions, variables, years, out)
    except Exception as e:                                          # noqa: BLE001
        claim["monthly_figure_error"] = f"{type(e).__name__}: {e}"
    ctl = (exp.get("experiment") or {}).get("control") or {}
    record = {
        "kind": "whole_domain_report",
        "question": (reception.get("brief") or {}).get("user_request")
        or reception.get("user_request") or strategy.get("question"),
        "status": _status(exp),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(rd), "control": ctl, "years": years, "regions": regions,
        "variables": variables, "headline": text, "claim": claim,
        "runs": runs, "feasibility": strategy.get("feasibility"),
        "requires": strategy.get("requires"),
        "caveats": _caveats(exp, plan, table),
        "figures": [Path(f).name for f in figs],
        "sources": {"numbers": "03_results/regional.json",
                    "experiment": "experiment.json", "design": "strategy.json"},
    }
    record["markdown_tables"] = _tables(table, runs, regions, variables, years)
    # months, seasons and where the numbers come from (2026-11-03)
    csv_path = monthly_csv(table, rd / "03_results" / MONTHLY_CSV)
    record["monthly_csv"] = f"03_results/{MONTHLY_CSV}" if csv_path else None
    if csv_path:
        record["markdown_tables"] += [
            f"Every month and season of every run, region and field, with each run's "
            f"difference from the control: [{record['monthly_csv']}](../{record['monthly_csv']}).",
            ""]
    record["provenance_lines"] = provenance_lines(table, record["monthly_csv"])
    (out / REPORT_JSON).write_text(json.dumps(record, indent=1, default=str))
    (out / REPORT_MD).write_text(render(record))
    return {"report": str(out / REPORT_MD), "record": str(out / REPORT_JSON),
            "figures": figs, "headline": text}


def render(record: Dict[str, Any]) -> str:
    """REPORT.md from the report record. Pure."""
    ctl = record.get("control") or {}
    lines = [f"# {_plain(record.get('question') or '(no question recorded)')}", "",
             f"*Whole-domain experiment on the control {ctl.get('name') or '?'}, status "
             f"**{record.get('status')}**, rendered {record.get('generated_at')} from the "
             f"run's records ({Path(str(record.get('run_dir') or '')).name})*", "",
             "## The answer", "", f"**{_plain(record.get('headline'))}**", ""]
    fz = record.get("feasibility") or {}
    if fz.get("verdict") and fz.get("verdict") != "full":
        lines += [f"The design was judged **{fz['verdict']}**: {_plain(fz.get('why'))}"
                  + (" It needs: " + "; ".join(_plain(x) for x in record.get("requires") or [])
                     + "." if record.get("requires") else ""), ""]
    lines += ["## What was run", ""]
    lines.append(f"- the control: **{ctl.get('name') or '?'}**"
                 + ("; reused, so its existing output was read and nothing was spent on it"
                    if ctl.get("reuse") else "; run again from its recipe"))
    for r in record.get("runs") or []:
        if r.get("reused") or r.get("run") == CONTROL:
            continue
        lines.append(f"- run **{r['run']}**: the control with {_changes_text(r)}"
                     + (f" (status {r.get('status')})" if r.get("status") else ""))
        for c in r.get("changes") or []:
            if c.get("meaning"):
                lines.append(f"  - {c.get('id')} means {_plain(c['meaning'])}; it is NOT "
                             f"{_plain(c.get('not_meaning'))}")
    ys = record.get("years") or []
    if ys:
        lines.append(f"- {'year' if len(ys) == 1 else 'years'} {span(ys)}; regions: "
                     + "; ".join(record.get("regions") or []))
    lines += ["", "## The numbers", ""] + list(record.get("markdown_tables") or [])
    for f in record.get("figures") or []:
        lines += [f"![{f}]({f})", ""]
    lines += ["## Caveats", ""] + [f"- {_plain(c)}" for c in record.get("caveats") or []]
    if not record.get("caveats"):
        lines.append("- none recorded")
    src = record.get("sources") or {}
    lines += ["", "## Where the numbers come from", "",
              f"- every number above: {src.get('numbers')}, made from each run's own history",
              f"- the experiment, its runs and their checks: {src.get('experiment')}",
              f"- the design and its caveats: {src.get('design')}", "",
              "*Values are shown to 3 significant figures; the exact values are in "
              f"{src.get('numbers')}.*", ""]
    lines += list(record.get("provenance_lines") or [])
    return "\n".join(lines)
