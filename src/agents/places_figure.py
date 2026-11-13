#!/usr/bin/env python3
"""
The places the numbers are reported for, as one map, for the details page
src/agents/places_figure.py

A study's comparisons read the model's numbers over named places: the
vegetated cells of a watershed, its elevation bands, the cells that burned. A
reader of the details page saw them as names and areas in a table, and could
not see where they lie (the user's review of 2026-11-13). The page now shows
them on the model's grid:

    each place that holds no other named place, one colour of an ordered
        scale, with its area in the legend
    a place that holds another entirely, its edge as a line
    the rest of the grid, light grey

WHO DOES WHAT, as for the control run's figures (agents/control_figures.py).
The model server owns the places: it reads each region's mask and area from
the control run's results store and decides which are filled and which are
outlines (domain_places.page_record). This module asks the server for that
record, keeps it beside the design (01_inputs/places.json, so a page can be
drawn again from its records) and draws it with the shared figure tool
(agents/analysis/figure_spec.render_figure, its map of classes).

Nothing here may stop a page. A server without the module, a control with no
results store or a figure that cannot be drawn leaves the map out, and draw()
says why.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RECORD_FILE = "01_inputs/places.json"
FIG_DIR = "design_figures"
PNG = f"{FIG_DIR}/places_reported.png"
OTHER = "the rest of the model grid"
ALT = "The places the numbers are reported for"
NOT_DRAWN = "the map of the places was not drawn"


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORD: asked of the model server, kept beside the design
# ─────────────────────────────────────────────────────────────────────────────
def _labels(regions: Any) -> List[str]:
    if not isinstance(regions, (list, tuple)):
        return []
    return [str(r) for r in regions if isinstance(r, str) and r.strip()]


def fetch(control_name: Optional[str], regions: Sequence[str]) -> Optional[Dict[str, Any]]:
    """What the model server holds for the places of a control's results
    store, or None: no control, no region, a control the registry does not
    know, a server without domain_places, or any error. Never raises."""
    wanted = _labels(regions)
    if not control_name or not wanted:
        return None
    try:
        from agents.analysis import analysis_plan as AP
        entry = AP._registry_entry(control_name)
        if not entry:
            return None
        import domain_places as DPL                                 # noqa: N812
        rec = DPL.page_record(dict(entry, name=control_name), wanted)
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
    p.write_text(json.dumps(record, separators=(",", ":")))


def load(run_dir) -> Optional[Dict[str, Any]]:
    """The saved record, or None for a file that is missing or broken."""
    p = Path(run_dir) / RECORD_FILE
    try:
        d = json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def of_design(record: Optional[Dict[str, Any]], control_name: Optional[str],
              regions: Sequence[str]) -> bool:
    """Whether a record was made for this design: the same control and the
    same set of regions. A record an earlier design left is not drawn."""
    if not isinstance(record, dict) or not control_name:
        return False
    theirs = record.get("regions")
    if not isinstance(theirs, list):
        return False
    return record.get("control") == control_name and \
        set(_labels(theirs)) == set(_labels(regions)) and bool(_labels(regions))


# ─────────────────────────────────────────────────────────────────────────────
# THE FIGURE: a map of classes in the shared figure tool
# ─────────────────────────────────────────────────────────────────────────────
def _filled(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    xs = record.get("filled")
    return [x for x in xs if isinstance(x, dict)] if isinstance(xs, list) else []


def _outlines(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    xs = record.get("outlines")
    return [x for x in xs if isinstance(x, dict) and x.get("mask") is not None] \
        if isinstance(xs, list) else []


def legend(record: Dict[str, Any]) -> Dict[str, Any]:
    """The legend's words: {"title", "entries", "lines"}. Where every filled
    label begins with one outline's label, each entry is the rest of its
    label, the title is that outline's label with its area, and its edge
    needs no entry of its own. Otherwise each entry is its whole label, there
    is no title, and every outline is named beside its line."""
    filled, outlines = _filled(record), _outlines(record)
    labels = [str(f.get("label") or "") for f in filled]
    head = None
    for o in sorted(outlines, key=lambda x: -len(str(x.get("label") or ""))):
        lab = str(o.get("label") or "")
        if lab and labels and all(x.startswith(lab) and _rest(x, lab) for x in labels):
            head = o
            break
    entries = [{"label": _rest(x, head["label"]) if head else x, "area_km2": f.get("area_km2")}
               for x, f in zip(labels, filled)]
    lines = [{"label": None if o is head else o.get("label"), "area_km2": o.get("area_km2"),
              "mask": o.get("mask")} for o in outlines]
    title = None
    if head is not None:
        from agents.analysis import figure_spec as FS
        area = FS.area_words(head.get("area_km2"))
        title = f"{head['label']} ({area})" if area else str(head["label"])
    return {"title": title, "entries": entries, "lines": lines}


def _rest(label: str, head: str) -> str:
    """A label without the outline's label it begins with: "watershed,
    vegetated, below 1000 m" after "watershed, vegetated" is "below 1000 m"."""
    return label[len(head):].lstrip(" ,;:-")


# AN EDGE NEEDS TWO CELLS A SIDE (2026-11-13). An outline is drawn as a
# contour, and a contour needs a grid of at least two cells each way. On a
# grid one cell wide the classes are drawn alone, and draw() names each edge
# it left out.
def _too_narrow_for_edges(record: Dict[str, Any]) -> bool:
    lat, lon = record.get("lat"), record.get("lon")
    return isinstance(lat, list) and isinstance(lon, list) and min(len(lat), len(lon)) < 2


def figure(record: Dict[str, Any]):
    """(the figure's specification, its output, the grid's axes) for the
    map of classes. Raises ValueError for a record whose grid does not hold
    together, which draw() turns into a reason."""
    lat, lon = record.get("lat"), record.get("lon")
    if not isinstance(lat, list) or not isinstance(lon, list) or not lat or not lon:
        raise ValueError("the record holds no grid")
    lon = [float(x) - 360.0 if float(x) > 180.0 else float(x) for x in lon]
    classes = record.get("classes")
    if not isinstance(classes, list) or len(classes) != len(lat) or \
            any(not isinstance(row, list) or len(row) != len(lon) for row in classes):
        raise ValueError(f"its classes are not on the grid of {len(lat)} by {len(lon)} cells")
    words = legend(record)
    spec = {"id": "places_reported", "panels": [{
        "kind": "map", "output": "C1",
        "classes": {"entries": words["entries"], "title": words["title"],
                    "other": OTHER if words["entries"] else "the model grid",
                    "lines": [] if _too_narrow_for_edges(record) else words["lines"]}}]}
    refs = {"C1": {"ref": "C1", "kind": "map", "out": {"field": "places", "title": ALT},
                   "value": classes, "made": True, "error": None, "outcome": None,
                   "why": None, "line": None}}
    return spec, refs, {"lat": [float(x) for x in lat], "lon": lon}


def draw(record: Optional[Dict[str, Any]], run_dir) -> Dict[str, Any]:
    """Draw the map into the study's folder. {"png": its path from the study
    folder, or None, "problems": why it was not drawn}. Never raises. A
    record with no filled region and no outline gives no figure."""
    out: Dict[str, Any] = {"png": None, "problems": []}
    if not isinstance(record, dict):
        return out
    if not _filled(record) and not _outlines(record):
        out["problems"].append(f"{NOT_DRAWN}: the record holds no place to draw")
        return out
    try:
        from agents.analysis import figure_spec as FS
        spec, refs, ll = figure(record)
        if _too_narrow_for_edges(record):
            for o in _outlines(record):
                name = repr(o["label"]) if o.get("label") else "a region"
                out["problems"].append(f"the edge of {name} is left out of the map of the "
                                       f"places, because the grid is one cell wide and an "
                                       f"edge needs two cells a side")
        (Path(run_dir) / FIG_DIR).mkdir(parents=True, exist_ok=True)
        got = FS.render_figure(spec, refs, Path(run_dir) / PNG, lat_lon=ll)
        if got.get("ok"):
            out["png"] = PNG
        else:
            out["problems"].append(f"{NOT_DRAWN}: {got.get('error')}")
    except Exception as e:                                          # noqa: BLE001
        out["problems"].append(f"{NOT_DRAWN}: {type(e).__name__}: {e}")
    return out


def page_lines(record: Optional[Dict[str, Any]], drawn: Optional[Dict[str, Any]]) -> List[str]:
    """The image line for a page, or [] when nothing was drawn."""
    png = (drawn or {}).get("png") if isinstance(drawn, dict) else None
    if not png or not isinstance(record, dict):
        return []
    return [f"![{ALT}]({png})", ""]
