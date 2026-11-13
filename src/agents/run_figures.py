#!/usr/bin/env python3
"""
What a new run is given, beside the control run, for the design page
src/agents/run_figures.py

A design page is written before any run is made, so it cannot show what a run
produces. It can show what a run is given: the inputs its changes alter,
beside the control run's own (the user's review of 2026-11-12, the page
rebuilt on 2026-11-13). The page's section on the new runs holds

    a small table, one row per knob a run changes: what it is, how it is
        changed, when and where
    a figure per run, one panel per input a change alters, the control run's
        line beside the new run's, the changed months shaded
    a map of the mean change, where the change covers part of the grid

WHO DOES WHAT, as for the control run's figures (agents/control_figures.py).
The model server owns the values: it reads the control run's history and each
run's own weather file and gives the rows, the series and the map
(domain_given.page_record). This module asks the server for them, keeps what
it was given beside the design (01_inputs/run_given.json, so a page can be
drawn again from its records), and draws with the Analyzer's own figure tool
(agents/analysis/figure_spec.render_figure).

Nothing here may stop a design page. A server that cannot give the record, a
run it holds nothing for, or a figure that cannot be drawn leaves that part
out. The page then says what each run changes in the server's own words, as
it did before the table, and a figure not drawn is named under `problems`.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RECORD_FILE = "01_inputs/run_given.json"
FIG_DIR = "design_figures"
CONTROL_LABEL = "control run"
COLUMNS = ("knob", "change", "when", "where")
HEADS = ("Knob", "Change", "When", "Where")
STEP_WORDS = {"month": "month by month", "year": "year by year"}
SHADED = {"month": "months", "year": "years"}


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORD: asked of the model server, kept beside the design
# ─────────────────────────────────────────────────────────────────────────────
def fetch(strategy: Optional[Dict[str, Any]], run_dir) -> Optional[Dict[str, Any]]:
    """What the model server gives for the new runs of a design, or None: no
    whole-domain design, a control it does not know, a server without the
    tool, or any error. Never raises: a page is written without the figures
    rather than not at all."""
    try:
        de = (strategy or {}).get("domain_experiment")
        if not isinstance(de, dict) or not de.get("runs"):
            return None
        ctl = de.get("control") or {}
        name = ctl.get("name")
        if not name:
            return None
        from agents.analysis import analysis_plan as AP
        entry = AP._registry_entry(name)
        if not entry:
            return None
        import domain_controls as DCT                               # noqa: N812
        import domain_given as DG                                   # noqa: N812
        experiment = {"control": dict(ctl), "runs": list(de.get("runs") or []),
                      "years": list(de.get("years") or [])}
        control = dict(entry, name=name, facts=DCT.facts(name))
        rec = DG.page_record(experiment, control,
                             Path(run_dir) / "01_inputs" if run_dir is not None else None)
    except Exception:                                               # noqa: BLE001
        return None
    return rec if isinstance(rec, dict) else None


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
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _list(x: Any) -> List[Any]:
    """A record's list, or [] where the record holds anything else: a record
    of the wrong shape leaves its part of the page out and stops nothing."""
    return list(x) if isinstance(x, list) else []


def _years(xs: Any) -> List[int]:
    return sorted({int(y) for y in _list(xs) if str(y).lstrip("-").isdigit()})


def of_design(record: Optional[Dict[str, Any]], de: Optional[Dict[str, Any]]) -> bool:
    """Whether a record was made for this design: its control, its years,
    and every run it holds is a run of the design, with the same changes
    where the record says what they were. A record an earlier design left is
    not drawn on this design's page, even one with the same control, years
    and run names. A run of the design the record does not hold is said on
    the page in the server's words, as before."""
    if not isinstance(record, dict) or not isinstance(de, dict):
        return False
    ctl = de.get("control") if isinstance(de.get("control"), dict) else {}
    if record.get("control") != ctl.get("name") or \
            _years(record.get("years")) != _years(de.get("years")):
        return False

    def same(a, b):
        return json.dumps(a, sort_keys=True, default=str) == \
            json.dumps(b, sort_keys=True, default=str)
    mine = {r.get("name"): [c for c in _list(r.get("changes")) if isinstance(c, dict)]
            for r in _list(de.get("runs")) if isinstance(r, dict)}
    theirs = {r.get("name"): r for r in _list(record.get("runs")) if isinstance(r, dict)}
    return bool(theirs) and all(n in mine and ("changes" not in r or same(r["changes"], mine[n]))
                                for n, r in theirs.items())


def entry(record: Optional[Dict[str, Any]], name: Any) -> Optional[Dict[str, Any]]:
    """The record's entry for one run, or None."""
    for r in _list((record or {}).get("runs") if isinstance(record, dict) else None):
        if isinstance(r, dict) and r.get("name") == name:
            return r
    return None


def rows(record: Optional[Dict[str, Any]], name: Any) -> List[Dict[str, Any]]:
    return [x for x in _list((entry(record, name) or {}).get("rows")) if isinstance(x, dict)]


def _panels(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [p for p in _list(run.get("panels")) if isinstance(p, dict)]


# ─────────────────────────────────────────────────────────────────────────────
# THE FIGURES: a stacked series figure and a map, per run
# ─────────────────────────────────────────────────────────────────────────────
def _file_name(name: Any) -> str:
    """A run's name as a part of a file's name (the server's rule already
    allows only letters, digits and _)."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)) or "run"


def series_png(name: Any) -> str:
    return f"new_run_{_file_name(name)}_series.png"


def map_png(name: Any) -> str:
    return f"new_run_{_file_name(name)}_map.png"


def _ref(ref: str, kind: str, out: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """One output as the figure tool takes it."""
    return {"ref": ref, "kind": kind, "out": out, "value": value, "made": True,
            "error": None, "outcome": None, "why": None, "line": None}


def series_figure(run: Dict[str, Any]):
    """(the figure's specification, its outputs) for one run: a panel per
    input, the control run's line and the run's, its changed labels shaded."""
    name = str(run.get("name"))
    spec = {"id": f"new_run_{_file_name(name)}_series", "layout": "stack", "panels": []}
    refs: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(_panels(run), 1):
        step = str(p.get("step") or "month")
        out = {"field": p.get("title"), "step": step, "title": p.get("title")}
        for key, label, values in ((f"G{i}.control", CONTROL_LABEL, p.get("control")),
                                   (f"G{i}.run", name, p.get("run"))):
            refs[key] = _ref(key, "series", dict(out, label=label),
                             {"labels": list(p.get("labels") or []),
                              "values": list(values or []), "step": step,
                              "units": p.get("units")})
        panel = {"kind": "series", "outputs": [f"G{i}.control", f"G{i}.run"]}
        if p.get("changed"):
            panel["shade_labels"] = [str(x) for x in p["changed"]]
        spec["panels"].append(panel)
    return spec, refs


def map_figure(run: Dict[str, Any]):
    """(the specification, its output, the grid's axes) for one run's map: the
    run minus the control run, on a scale centred on zero, with the edge of
    the cells the change reaches."""
    m = run.get("map") or {}
    spec = {"id": f"new_run_{_file_name(run.get('name'))}_map",
            "panels": [{"kind": "map", "output": "D1"}]}
    refs = {"D1": _ref("D1", "difference",
                       {"field": m.get("title"), "title": m.get("title"),
                        "units": m.get("units")},
                       {"difference": m.get("values"), "units": m.get("units")})}
    ll = {"lat": m.get("lat"), "lon": m.get("lon")}
    if m.get("outline"):
        ll["outline"] = m["outline"]
    return spec, refs, ll


def draw(record: Optional[Dict[str, Any]], run_dir) -> Dict[str, Any]:
    """Draw each run's figures into the study's folder. {"runs": {name:
    {"series", "map"}}, "problems"}: each figure's path from the study folder,
    or None where it was not drawn, and `problems` says why."""
    out: Dict[str, Any] = {"runs": {}, "problems": []}
    if not isinstance(record, dict):
        return out
    d = Path(run_dir) / FIG_DIR
    for run in _list(record.get("runs")):
        if not isinstance(run, dict) or not run.get("name"):
            continue
        name = str(run["name"])
        got = out["runs"][name] = {"series": None, "map": None}
        try:
            from agents.analysis import figure_spec as FS
            if _panels(run):
                d.mkdir(parents=True, exist_ok=True)
                spec, refs = series_figure(run)
                done = FS.render_figure(spec, refs, d / series_png(name))
                if done.get("ok"):
                    got["series"] = f"{FIG_DIR}/{series_png(name)}"
                else:
                    out["problems"].append(f"the figure of `{name}` was not drawn: "
                                           f"{done.get('error')}")
            if isinstance(run.get("map"), dict):
                d.mkdir(parents=True, exist_ok=True)
                spec, refs, ll = map_figure(run)
                done = FS.render_figure(spec, refs, d / map_png(name), lat_lon=ll)
                if done.get("ok"):
                    got["map"] = f"{FIG_DIR}/{map_png(name)}"
                else:
                    out["problems"].append(f"the map of `{name}` was not drawn: "
                                           f"{done.get('error')}")
        except Exception as e:                                      # noqa: BLE001
            out["problems"].append(f"the figures of `{name}` were not drawn: "
                                   f"{type(e).__name__}: {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGE'S LINES: the table, the figures, what they show
# ─────────────────────────────────────────────────────────────────────────────
def _cell(text: Any) -> str:
    return " ".join(str(text if text is not None else "").split()).replace("|", "/")


def table_lines(record: Optional[Dict[str, Any]], names: Sequence[str],
                several: bool) -> List[str]:
    """The table of the knobs, one row per knob of each run named that the
    record holds rows for, with a first column for the run where the design
    has several. [] when no run named has a row."""
    body = []
    for name in names:
        for r in rows(record, name):
            cells = [_cell(r.get(k)) for k in COLUMNS]
            body.append("| " + " | ".join(([f"`{name}`"] if several else []) + cells) + " |")
    if not body:
        return []
    heads = (["Run"] if several else []) + list(HEADS)
    return ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)] + body + [""]


def _join(xs: Sequence[str]) -> str:
    xs = [x for x in xs if x]
    return ("" if not xs else xs[0] if len(xs) == 1 else
            ", ".join(xs[:-1]) + " and " + xs[-1])


def shown_lines(run: Dict[str, Any], drawn: Dict[str, Any], several: bool) -> List[str]:
    """What one run's figures show, in sentences: the figure's lines and its
    step, the shaded labels, and the map, only for what was drawn."""
    who = f"`{run.get('name')}`" if several else "the new run"
    said = []
    if drawn.get("series"):
        steps = list(dict.fromkeys(str(p.get("step") or "month") for p in _panels(run)))
        said.append(f"The figure shows what {who} is given beside the control run, "
                    f"{_join([STEP_WORDS.get(s, s) for s in steps])}.")
        shaded = list(dict.fromkeys(str(p.get("step") or "month") for p in _panels(run)
                                    if p.get("changed")))
        if shaded:
            said.append(f"The shaded {_join([SHADED.get(s, s) for s in shaded])} are the "
                        f"changed ones.")
    if drawn.get("map"):
        said.append("The map shows the mean change in those months."
                    if drawn.get("series") and "month" in [str(p.get("step") or "month")
                                                           for p in _panels(run)]
                    else f"The map shows the mean change in the months {who} changes.")
    return said


def figure_lines(record: Optional[Dict[str, Any]], drawn: Optional[Dict[str, Any]],
                 names: Sequence[str], several: bool) -> List[str]:
    """Each run's figures under the table, with the sentences that say what
    they show. A run whose figures were not drawn has none. With several
    runs, each run's figures are introduced by its name."""
    L: List[str] = []
    done = drawn.get("runs") if isinstance(drawn, dict) else None
    for name in names:
        run = entry(record, name)
        got = done.get(name) if isinstance(done, dict) else None
        got = got if isinstance(got, dict) else {}
        if run is None or not (got.get("series") or got.get("map")):
            continue
        if several:
            L += [f"**`{name}`**", ""]
        what = f"What {name} is given" if several else "What the new run is given"
        if got.get("series"):
            L += [f"![{what}]({got['series']})", ""]
        if got.get("map"):
            L += [f"![Where the change applies]({got['map']})", ""]
        L += [" ".join(shown_lines(run, got, several)), ""]
    return L


def details_lines(record: Optional[Dict[str, Any]], drawn: Optional[Dict[str, Any]],
                  names: Sequence[str]) -> List[str]:
    """The details page's section: the same table and figures, and why any
    part could not be given or drawn. [] with no record."""
    if not isinstance(record, dict):
        return []
    several = len(names) > 1
    L = ["## What the new runs are given", ""]
    L += table_lines(record, names, several) + figure_lines(record, drawn, names, several)
    probs = [str(p) for p in _list(record.get("problems")) if str(p).strip()]
    for name in names:
        probs += [f"`{name}`: {p}" for p in _list((entry(record, name) or {}).get("problems"))
                  if str(p).strip()]
    probs += [str(p) for p in _list((drawn or {}).get("problems")) if str(p).strip()]
    L += [f"- Not shown: {p}" for p in probs]
    if len(L) == 2:
        return []
    return L + [""] if L[-1] != "" else L
