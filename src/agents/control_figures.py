#!/usr/bin/env python3
"""
The control run in two figures, for the design pages
src/agents/control_figures.py

A design page told a person how the control run compares with measurements in
one paragraph of numbers, written once, about one or two years of the run. A
person who asked about other years read numbers that were not about their
question (the user's review of 2026-11-12).

The pages now show the whole record instead:

    the control run beside its measurements, month by month, one panel per
    quantity, with the years the person asked about shaded
    maps of the control run's mean over its years

and say, in one sentence per panel, what the two lines are in the years asked
about. The sentences are filled in by code from the same values the figure is
drawn from. No language model writes or reads any of this.

WHO DOES WHAT. The model server owns the values: it pairs each measurement
with the control run's own monthly series and keeps the pairs in the run's
results store, and it averages the store's year maps (its domain_measured).
This module asks the server for them, saves what it was given beside the
design (01_inputs/control_figures.json, so a page can be drawn again from its
records), and draws with the Analyzer's own figure tool
(agents/analysis/figure_spec.render_figure). One tool draws the design pages'
figures and the reports'.

Nothing here may stop a design page. A control with no measurements, a store
out of reach or a figure that cannot be drawn leaves the page as it was
without figures, and the reason is kept in the record.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RECORD_FILE = "01_inputs/control_figures.json"
FIG_DIR = "design_figures"
SERIES_PNG = "control_run_series.png"
MAPS_PNG = "control_run_maps.png"
MODEL_LABEL = "control run"
HEADING = "The control run in figures"


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORD: asked of the model server, kept beside the design
# ─────────────────────────────────────────────────────────────────────────────
def fetch(control_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """What the model server holds for the control's figures, or None where
    it holds nothing to draw. Never raises."""
    if not control_name:
        return None
    try:
        from agents.analysis import analysis_plan as AP
        entry = AP._registry_entry(control_name)
        if not entry:
            return None
        import domain_measured as DM                                # noqa: N812
        rec = DM.page_record(dict(entry, name=control_name))
    except Exception as e:                                          # noqa: BLE001
        return {"control": control_name, "measured": None, "maps": None,
                "problems": [f"the model server could not give the figures' values: "
                             f"{type(e).__name__}: {e}"]}
    # a record with nothing to draw is still kept for the set-up rows and the
    # tables the details page prints from it (2026-11-13)
    if not _series(rec) and not _maps(rec) and not rec.get("setup") \
            and not rec.get("tables"):
        return None if not rec.get("problems") else rec
    return rec


def save(run_dir, record: Optional[Dict[str, Any]]) -> None:
    """Keep the record beside the design. None removes one an earlier design
    left, so a page is never drawn from another design's record."""
    p = Path(run_dir) / RECORD_FILE
    if record is None:
        if p.is_file():
            p.unlink()
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record))


def load(run_dir) -> Optional[Dict[str, Any]]:
    p = Path(run_dir) / RECORD_FILE
    try:
        d = json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _series(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(((record or {}).get("measured") or {}).get("series") or [])


def _maps(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(((record or {}).get("maps") or {}).get("panels") or [])


def record_years(record: Optional[Dict[str, Any]]) -> List[int]:
    """The years the paired series cover."""
    labels = ((record or {}).get("measured") or {}).get("labels") or []
    return sorted({int(str(lab)[:4]) for lab in labels if str(lab)[:4].isdigit()})


def asked_years(brief, strategy, record: Optional[Dict[str, Any]]) -> List[int]:
    """The years the page sets off: the ones the person asked about that the
    record holds. None when the request names no year, or names the whole
    record."""
    from core import analysis_years as AY
    _, how = AY.reported(brief, strategy)
    held = record_years(record)
    want = [int(y) for y in how.get("asked") or [] if int(y) in held]
    return [] if set(want) == set(held) else sorted(set(want))


# ─────────────────────────────────────────────────────────────────────────────
# THE PANELS: one per quantity, the model's line and its measurements
# ─────────────────────────────────────────────────────────────────────────────
def panels(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The paired series as panels. Two measurements of one quantity (the
    same field over the same cells, as the server paired them) are one panel
    with one line of the model's."""
    labels = list(((record or {}).get("measured") or {}).get("labels") or [])
    out: List[Dict[str, Any]] = []
    at: Dict[Any, Dict[str, Any]] = {}
    for s in _series(record):
        key = (s.get("field"), s.get("where"), json.dumps(s.get("model")))
        p = at.get(key)
        if p is None:
            p = at[key] = {"title": s.get("title") or f"{s.get('field')} over {s.get('where')}",
                           "field": s.get("field"), "how": s.get("how"),
                           "units": s.get("units"), "where": s.get("where"),
                           "cells": s.get("cells"), "labels": labels,
                           "model": list(s.get("model") or []),
                           "model_words": s.get("model_words") or
                           f"{s.get('field')} over {s.get('where')}",
                           "measured": []}
            out.append(p)
        p["measured"].append({"id": s.get("id"), "label": s.get("label") or "measured",
                              "words": s.get("measured_words") or "measured",
                              "source": s.get("source"),
                              "values": list(s.get("measured") or [])})
    return out


def _ref(ref: str, kind: str, out: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """One output as the figure tool takes it."""
    return {"ref": ref, "kind": kind, "out": out, "value": value, "made": True,
            "error": None, "outcome": None, "why": None, "line": None}


def series_figure(record: Optional[Dict[str, Any]], shade: Sequence[int] = ()):
    """(the figure's specification, its outputs) for the paired series."""
    spec = {"id": "control_run_series", "layout": "stack", "panels": []}
    refs: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(panels(record), 1):
        names = [f"S{i}.model"]
        refs[names[0]] = _ref(names[0], "series",
                              {"field": p["field"], "step": "month", "title": p["title"],
                               "label": MODEL_LABEL},
                              {"labels": p["labels"], "values": p["model"], "step": "month",
                               "units": p["units"]})
        for j, m in enumerate(p["measured"], 1):
            n = f"S{i}.M{j}"
            names.append(n)
            refs[n] = _ref(n, "series",
                           {"field": p["field"], "step": "month", "title": p["title"],
                            "label": m["label"], "measured": True},
                           {"labels": p["labels"], "values": m["values"], "step": "month",
                            "units": p["units"]})
        spec["panels"].append({"kind": "series", "outputs": names,
                               **({"shade": [int(y) for y in shade]} if shade else {})})
    return spec, refs


def _span(years: Sequence[int]) -> str:
    ys = sorted({int(y) for y in years})
    if not ys:
        return "no year"
    if ys == list(range(ys[0], ys[-1] + 1)):
        return str(ys[0]) if len(ys) == 1 else f"{ys[0]} to {ys[-1]}"
    return ", ".join(str(y) for y in ys[:-1]) + f" and {ys[-1]}"


def maps_figure(record: Optional[Dict[str, Any]]):
    """(the specification, its outputs, the grid's axes) for the mean maps."""
    spec = {"id": "control_run_maps", "panels": []}
    refs: Dict[str, Dict[str, Any]] = {}
    m = (record or {}).get("maps") or {}
    for i, p in enumerate(_maps(record), 1):
        n = f"M{i}"
        title = f"{p.get('words')}, {'sum' if p.get('how') == 'total' else 'mean'} of a year"
        refs[n] = _ref(n, "map", {"field": "+".join(p.get("fields") or []), "title": title,
                                  "units": p.get("units")}, p.get("values"))
        spec["panels"].append({"kind": "map", "output": n})
    ll = {"lat": m.get("lat"), "lon": m.get("lon")}
    if isinstance(m.get("outline"), dict) and m["outline"].get("mask"):
        ll["outline"] = m["outline"]["mask"]
    return spec, refs, ll


def draw(record: Optional[Dict[str, Any]], run_dir, shade: Sequence[int] = ()
         ) -> Dict[str, Any]:
    """Draw the two figures into the study's folder. {"series", "maps"}: each
    the figure's path from the study folder, or None where it was not drawn,
    and `problems` says why."""
    out: Dict[str, Any] = {"series": None, "maps": None, "problems": []}
    if not record:
        return out
    from agents.analysis import figure_spec as FS
    d = Path(run_dir) / FIG_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        if _series(record):
            spec, refs = series_figure(record, shade)
            got = FS.render_figure(spec, refs, d / SERIES_PNG)
            if got.get("ok"):
                out["series"] = f"{FIG_DIR}/{SERIES_PNG}"
            else:
                out["problems"].append(f"the series were not drawn: {got.get('error')}")
        if _maps(record):
            spec, refs, ll = maps_figure(record)
            got = FS.render_figure(spec, refs, d / MAPS_PNG, lat_lon=ll)
            if got.get("ok"):
                out["maps"] = f"{FIG_DIR}/{MAPS_PNG}"
            else:
                out["problems"].append(f"the maps were not drawn: {got.get('error')}")
    except Exception as e:                                          # noqa: BLE001
        out["problems"].append(f"the figures were not drawn: {type(e).__name__}: {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE SENTENCES: the two lines in the years asked about, by code
# ─────────────────────────────────────────────────────────────────────────────
def three(x: float) -> str:
    """x at three significant figures, as a page prints it: 12.3, 457, 1,230."""
    if x == 0 or not math.isfinite(x):
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    r = round(x, 2 - e)
    if abs(r) >= 10 ** (e + 1):                   # 99.96 rounds up into the next decade
        e += 1
    return f"{r:,.0f}" if e >= 2 else f"{r:.{2 - e}f}"


def _units_words(units: str, total: bool) -> str:
    u = str(units or "").strip()
    if total and u.endswith("/month"):
        u = u[: -len("/month")]
    return u.replace("^", "")                      # gC/m^2 is written gC/m2 in a sentence


def panel_numbers(p: Dict[str, Any], years: Sequence[int]) -> Optional[Dict[str, Any]]:
    """The model's value and each measurement's over the years given, like
    with like: a month counts when the model and every measurement of the
    panel have a value in it. A total is a year's total, so a year counts
    when all its twelve months do, and the value is the mean of those years.
    A mean is over the months that count. None when nothing counts."""
    want = {int(y) for y in years}
    series = [p["model"]] + [m["values"] for m in p["measured"]]
    by_year: Dict[int, List[int]] = {}
    for i, lab in enumerate(p["labels"]):
        y = int(str(lab)[:4])
        if y in want and all(i < len(s) and s[i] is not None for s in series):
            by_year.setdefault(y, []).append(i)
    total = p.get("how") == "total"
    if total:
        by_year = {y: ix for y, ix in by_year.items() if len(ix) == 12}
    idx = [i for ix in by_year.values() for i in ix]
    if not idx:
        return None

    def value(s):
        return (sum(s[i] for i in idx) / len(by_year)) if total else \
            (sum(s[i] for i in idx) / len(idx))
    return {"years": sorted(by_year), "total": total, "model": value(p["model"]),
            "measured": [value(m["values"]) for m in p["measured"]]}


def sentences(record: Optional[Dict[str, Any]], years: Sequence[int]) -> List[str]:
    """One sentence per panel: the control run's value and the measured ones
    in the years given. A panel with no value in those years has none."""
    out = []
    for p in panels(record):
        n = panel_numbers(p, years)
        if n is None:
            continue
        u = _units_words(p["units"], n["total"])
        per = " a year" if n["total"] else ""
        ys = n["years"]
        when = f"in {ys[0]}" if len(ys) == 1 else f"over {_span(ys)}"
        verb = "totals" if n["total"] else "averages"
        those = " and ".join(f"{three(v)} {u}{per} {m['words']}"
                             for v, m in zip(n["measured"], p["measured"]))
        out.append(f"{p['model_words']} {verb} {three(n['model'])} {u}{per} in the control "
                   f"run {when}, compared with {those}.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGES' LINES
# ─────────────────────────────────────────────────────────────────────────────
def page_lines(record: Optional[Dict[str, Any]], drawn: Optional[Dict[str, Any]],
               years: Sequence[int]) -> List[str]:
    """The short page's lines: the series figure, what it shows, and the
    sentences for the years asked about. [] when the figure was not drawn."""
    if not (drawn or {}).get("series"):
        return []
    held = record_years(record)
    L = [f"![The control run and the measurements, month by month]({drawn['series']})", ""]
    said = (f"The figure sets the control run beside the measurements for each month of "
            f"{_span(held)}.")
    if years:
        said += (f" The shaded year is the one you asked about." if len(years) == 1 else
                 f" The shaded years are the ones you asked about.")
    L += [said + (" " + " ".join(sentences(record, years or held)) if panels(record) else ""),
          ""]
    return L


def details_lines(record: Optional[Dict[str, Any]], drawn: Optional[Dict[str, Any]],
                  years: Sequence[int]) -> List[str]:
    """The details page's section: both figures, and where each line of the
    series comes from."""
    drawn = drawn or {}
    if not drawn.get("series") and not drawn.get("maps"):
        return []
    L = [f"## {HEADING}", ""]
    m = (record or {}).get("maps") or {}
    if drawn.get("maps"):
        L += [f"![Maps of the control run's mean]({drawn['maps']})", ""]
        outline = (m.get("outline") or {}).get("region")
        spans = {_span(p.get("years") or []) for p in _maps(record)}
        L += [(f"Each map is the average of {next(iter(spans))} over the model's grid."
               if len(spans) == 1 else "Each map is an average of the run's years over the "
                                       "model's grid.")
              + (f" The line is the edge of the region `{outline}`." if outline else ""), ""]
    if drawn.get("series"):
        L += [f"![The control run and the measurements, month by month]({drawn['series']})", ""]
        if years:
            L += [f"The shaded years are {_span(years)}, the years of the question.", ""]
        L += ["| Panel | The control run's line | Measured line | Where the measurement "
              "comes from |", "|---|---|---|---|"]
        for p in panels(record):
            mine = (f"`{p['field']}`, monthly {p.get('how') or 'value'} in {p['units']}, over "
                    f"{p['where']}")
            for x in p["measured"]:
                L.append(f"| {p['title']} | {mine} | {x['label']} | "
                         f"{x.get('source') or 'not recorded'} |")
        L.append("")
    for prob in (record or {}).get("problems") or []:
        L.append(f"- Not shown: {prob}")
    for prob in drawn.get("problems") or []:
        L.append(f"- Not shown: {prob}")
    return L + [""] if L[-1] != "" else L
