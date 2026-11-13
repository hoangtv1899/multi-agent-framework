#!/usr/bin/env python3
"""
A control run beside its measurements, month by month
mcp/elm-mcp/src/domain_measured.py

A control's results store (domain_suite) holds the model's output alone. How
the control compares with measurements was worked out by hand, outside the
server, and reached a person as a paragraph of numbers for one or two years.
A person who asks about other years then reads numbers that are not about
their question.

This module puts the measurements beside the model's own values, one pair of
monthly series per measurement, and keeps them in the store. A figure of the
whole record then shows every year, and the numbers of any year can be read
from the same file.

WHAT THE REGISTRY SAYS. A control may list its measurements under `measured`:

    {"id": "snow",
     "field": "H2OSNO",                  the model's field the store holds
     "points": [{"id", "lat", "lon"}]    the model is read at the cells that
                                         hold these points, and averaged
       or "region": "<a region of the store>"
     "table": "<csv: month,value>",      the measured values, one per month
     "units": "mm",                      must be the store's monthly units
     "title": "...", "label": "...",     the panel's title, the line's name
     "model_words": "...",               the sentence's subject
     "measured_words": "...",            what follows the measured number
     "source": "..."}                    where the measured values came from

THE TABLE IS PREPARED OUTSIDE THE SERVER. Fetching a network's files and
turning them into one value a month is not done here: each network has its
own files and gaps. The table is a plain file, `month,value`, with the month
as YYYY-MM and an empty value for a month that has none. Its units must be
the units the store gives the field by month, and the build refuses a pair
whose units differ, so a conversion is never guessed.

THE MODEL'S SIDE IS READ BY THE SERVER, never taken from the table: from the
store's regional series for a region, and from the monthly history for
points. Points are for a field the store holds as a mean (a state such as
snow water); a monthly total at single cells is not read here.

TWO MEASUREMENTS OF ONE THING SHARE THE MODEL'S SERIES. Two entries with the
same field and the same region or points are one panel of a figure with two
measured lines (a regulated river: the flow measured, and the flow with the
regulation removed).

    python domain_measured.py <control>          builds the file in the store

The file is `measured_month.json` in the control's store. Nothing else in the
store is changed, and the store's own manifest lists only its netCDF files.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FILE = "measured_month.json"
VERSION = 1
STEP = "month"
WORDS = ("title", "label", "model_words", "measured_words", "source")


class MeasuredError(Exception):
    """A measurement cannot be put beside the model, and the message says why."""


def read_table(path) -> Dict[str, Optional[float]]:
    """{month: value}, the month as YYYY-MM; None for a month with no value."""
    p = Path(str(path))
    if not p.is_file():
        raise MeasuredError(f"no table at {p}")
    out: Dict[str, Optional[float]] = {}
    with open(p, newline="") as fh:
        rows = csv.DictReader(fh)
        if not rows.fieldnames or [c.strip() for c in rows.fieldnames[:2]] != ["month", "value"]:
            raise MeasuredError(f"{p.name}: its first two columns must be month,value")
        for r in rows:
            m = str(r.get("month") or "").strip()
            if not m:
                continue
            if len(m) != 7 or m[4] != "-" or not (m[:4] + m[5:]).isdigit() \
                    or not 1 <= int(m[5:]) <= 12:
                raise MeasuredError(f"{p.name}: {m!r} is not a month written YYYY-MM")
            if m in out:
                raise MeasuredError(f"{p.name}: {m} appears twice")
            v = str(r.get("value") or "").strip()
            try:
                x = float(v) if v else None
            except ValueError:
                raise MeasuredError(f"{p.name}: {m} has the value {v!r}, which is no number")
            out[m] = x if x is not None and np.isfinite(x) else None
    if not out:
        raise MeasuredError(f"{p.name} holds no month")
    return out


def _sha256(path) -> str:
    return hashlib.sha256(Path(str(path)).read_bytes()).hexdigest()


def _clean(values: Sequence[Any]) -> List[Optional[float]]:
    """Numbers a JSON file can hold: None where there is no value."""
    out = []
    for v in values:
        x = None if v is None else float(v)
        out.append(x if x is not None and np.isfinite(x) else None)
    return out


def where_key(m: Dict[str, Any]) -> Tuple:
    """What two entries share when they are one panel: the field and the
    region, or the field and the points."""
    if m.get("region"):
        return (m.get("field"), "region", str(m["region"]))
    return (m.get("field"), "points",
            tuple(sorted((round(float(p["lat"]), 6), round(float(p["lon"]), 6))
                         for p in m.get("points") or [])))


def _region_series(suite, field: str, region: str, labels: Sequence[str]
                   ) -> Tuple[List[Optional[float]], Dict[str, Any]]:
    if region not in suite.masks:
        raise MeasuredError(f"the store has no region {region!r} "
                            f"(it has {', '.join(sorted(suite.masks))})")
    vals = [suite.value(field, region, lab)[0] for lab in labels]
    return vals, {"where": region, "cells": int(suite.masks[region].sum())}


def _point_series(run_dir, suite, field: str, points: Sequence[Dict[str, Any]],
                  labels: Sequence[str]) -> Tuple[List[Optional[float]], Dict[str, Any]]:
    """The field's monthly value at the cells that hold the points, averaged
    over the cells, from the monthly history. A month in which one of the
    cells has no value has none."""
    import gridded as G
    import regions as R
    if suite.how(field) != "mean":
        raise MeasuredError(f"{field} is a monthly {suite.how(field) or 'value'} in the store; "
                            f"points are read for a field it holds as a mean")
    if not points:
        raise MeasuredError("no point is listed")
    stream = G.open_stream(run_dir, "h0")
    if stream.cadence != "monthly":
        raise MeasuredError(f"the run's h0 history is {stream.cadence}, not monthly")
    if field not in stream.variables:
        raise MeasuredError(f"the monthly history holds no {field}")
    cells = R.station_cells(stream.grid, [dict(p, id=p.get("id") or i)
                                          for i, p in enumerate(points)])
    outside = [str(c["id"]) for c in cells if c["containing"] is None]
    if outside:
        raise MeasuredError(f"outside the model's grid: {', '.join(outside)}")
    flat = [int(c["containing"]["flat"]) for c in cells]
    at: Dict[str, float] = {}
    for y in sorted({int(lab[:4]) for lab in labels}):
        if y not in stream.years():
            continue
        b = stream.read(field, y, require_complete=False)
        if b.per_second:
            raise MeasuredError(f"{field} is a rate in the history; points are read for "
                                f"a state or a mean")
        data = np.asarray(b.data, dtype=np.float64).reshape(len(b.labels), -1)[:, flat]
        for lab, row in zip(b.labels, data):
            at[str(lab)] = float(row.mean()) if np.isfinite(row).all() else np.nan
    return ([at.get(lab) for lab in labels],
            {"where": f"the {len(flat)} cell{'' if len(flat) == 1 else 's'} that hold the points",
             "cells": len(set(flat)), "points": len(flat),
             "history_units": stream.units(field)})


def _pair(m: Dict[str, Any], suite, run_dir, labels: Sequence[str],
          models: Dict[Tuple, Tuple]) -> Dict[str, Any]:
    for k in ("id", "field", "table", "units"):
        if not m.get(k):
            raise MeasuredError(f"no {k}")
    field = str(m["field"])
    if field not in suite.fields:
        raise MeasuredError(f"the store holds no {field}")
    if bool(m.get("region")) == bool(m.get("points")):
        raise MeasuredError("name a region or list points, one of the two")
    units = suite.units(field, STEP)
    if str(m["units"]).strip() != units:
        raise MeasuredError(f"the table is in {m['units']} and the store holds {field} by "
                            f"month in {units}; convert the table")
    table = read_table(m["table"])
    key = where_key(m)
    if key not in models:
        models[key] = (_region_series(suite, field, str(m["region"]), labels)
                       if m.get("region") else
                       _point_series(run_dir, suite, field, m["points"], labels))
    model, where = models[key]
    hu = where.get("history_units")
    if hu is not None and str(hu).strip() != units:
        raise MeasuredError(f"the history gives {field} in {hu} and the store in {units}")
    measured = [table.get(lab) for lab in labels]
    mo, me = _clean(model), _clean(measured)
    both = sum(1 for a, b in zip(mo, me) if a is not None and b is not None)
    if not both:
        raise MeasuredError("no month has both a measured value and the model's")
    return {"id": str(m["id"]), "field": field, "how": suite.how(field), "units": units,
            "where": where["where"], "cells": where["cells"],
            **{k: (str(m[k]) if m.get(k) else None) for k in WORDS},
            "table": {"path": str(m["table"]), "sha256": _sha256(m["table"]),
                      "months": len(table)},
            "measured": me, "model": mo, "months_both": both}


def build(entry: Dict[str, Any], store_dir=None, write: bool = True) -> Dict[str, Any]:
    """The control's measurements beside its own values, written into its
    store. A measurement that cannot be paired is left out and named under
    `problems`, with the reason; the others are kept."""
    import domain_methods as M
    sd = Path(str(store_dir or entry.get("suite_dir") or ""))
    if not (store_dir or entry.get("suite_dir")) or not sd.is_dir():
        raise MeasuredError(f"{entry.get('name')}: no results store at {sd}")
    suite = M.Suite(sd)
    labels = suite.labels(STEP)
    rec: Dict[str, Any] = {"version": VERSION, "control": entry.get("name"),
                           "built_at": datetime.now().isoformat(timespec="seconds"),
                           "step": STEP, "labels": labels, "series": [], "problems": []}
    models: Dict[Tuple, Tuple] = {}
    seen = set()
    for m in entry.get("measured") or []:
        name = str((m or {}).get("id") or "a measurement with no id")
        try:
            if name in seen:
                raise MeasuredError("its id is used twice")
            rec["series"].append(_pair(m, suite, entry.get("run_dir"), labels, models))
            seen.add(name)
        except MeasuredError as e:
            rec["problems"].append(f"{name}: {e}")
        except Exception as e:                                  # noqa: BLE001
            rec["problems"].append(f"{name}: {type(e).__name__}: {e}")
    if write:
        (sd / FILE).write_text(json.dumps(rec, indent=1))
    return rec


def load(store_dir) -> Optional[Dict[str, Any]]:
    """The file build() wrote, or None where the store has none."""
    p = Path(str(store_dir or "")) / FILE
    if not store_dir or not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except ValueError:
        return None
    return d if isinstance(d, dict) and d.get("version") == VERSION else None


# ─────────────────────────────────────────────────────────────────────────────
# THE MAPS OF THE RUN'S MEAN
# ─────────────────────────────────────────────────────────────────────────────
def _axes(store_dir) -> Tuple[List[float], List[float]]:
    import netCDF4
    with netCDF4.Dataset(Path(str(store_dir)) / "maps_year.nc") as d:
        d.set_auto_mask(False)
        if "lat" not in d.variables or "lon" not in d.variables:
            raise MeasuredError("the store's year maps carry no latitude and longitude")
        return ([float(x) for x in np.asarray(d["lat"][:])],
                [float(x) for x in np.asarray(d["lon"][:])])


def _grid_list(a: np.ndarray, digits: int = 4) -> List[List[Optional[float]]]:
    """A map as a JSON file holds it: rounded to `digits` figures, None where
    a cell has no value."""
    out = []
    for row in np.asarray(a, dtype=np.float64):
        out.append([None if not np.isfinite(x) else float(f"{x:.{digits}g}") for x in row])
    return out


def mean_maps(suite, specs: Sequence[Dict[str, Any]], outline: Optional[str] = None
              ) -> Dict[str, Any]:
    """Each spec's map: the sum of its fields' year maps, averaged over the
    years every one of them has a map for. {"words", "fields"} per spec."""
    lat, lon = _axes(suite.dir)
    years = [str(y) for y in suite.labels("year")]
    out: Dict[str, Any] = {"lat": lat, "lon": lon, "panels": [], "problems": []}
    for s in specs or []:
        fields = [str(f) for f in (s or {}).get("fields") or []]
        words = str((s or {}).get("words") or " + ".join(fields))
        try:
            if not fields:
                raise MeasuredError("no field")
            lacking = [f for f in fields if f not in suite.fields]
            if lacking:
                raise MeasuredError(f"the store holds no {', '.join(lacking)}")
            units = {suite.units(f, "year") for f in fields}
            if len(units) > 1:
                raise MeasuredError(f"its fields are in different units "
                                    f"({', '.join(sorted(units))})")
            stack, used = [], []
            for y in years:
                maps = [suite.map(f, y) for f in fields]
                if any(m is None for m in maps):
                    continue
                stack.append(np.sum(maps, axis=0))
                used.append(int(y))
            if not stack:
                raise MeasuredError("no year has a map of every field")
            hows = {suite.how(f) for f in fields}
            out["panels"].append({
                "words": words, "fields": fields, "units": next(iter(units)),
                "how": next(iter(hows)) if len(hows) == 1 else "value",
                "years": used, "values": _grid_list(np.nanmean(np.array(stack), axis=0))})
        except MeasuredError as e:
            out["problems"].append(f"map of {words}: {e}")
    if outline:
        if outline in suite.masks:
            out["outline"] = {"region": outline,
                              "mask": [[int(bool(x)) for x in row]
                                       for row in suite.masks[outline]]}
        else:
            out["problems"].append(f"outline: the store has no region {outline!r}")
    return out


def page_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    """What the design pages draw of a control: its measurements beside its
    own values (the file in the store) and the maps its registry entry asks
    for. Every part that cannot be given is named under `problems`."""
    import domain_methods as M
    rec: Dict[str, Any] = {"control": entry.get("name"), "measured": None, "maps": None,
                           "problems": []}
    sd = entry.get("suite_dir")
    if not sd or not Path(str(sd)).is_dir():
        rec["problems"].append("the control has no results store")
        return rec
    rec["measured"] = load(sd)
    if rec["measured"] is None and entry.get("measured"):
        rec["problems"].append(f"the store holds no {FILE}: python domain_measured.py "
                               f"{entry.get('name')} builds it")
    spec = entry.get("page_figures") or {}
    if spec.get("maps"):
        try:
            rec["maps"] = mean_maps(M.Suite(sd), spec["maps"], spec.get("outline"))
            rec["problems"] += rec["maps"].pop("problems")
        except Exception as e:                                  # noqa: BLE001
            rec["problems"].append(f"maps: {type(e).__name__}: {e}")
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import domain_controls as DCT
    ap = argparse.ArgumentParser(description="Put a control's measurements beside its "
                                             "own monthly values, in its results store.")
    ap.add_argument("control")
    ap.add_argument("--registry", default=None)
    a = ap.parse_args(argv)
    try:
        entry = DCT.get(a.control, a.registry)
        rec = build(entry)
    except Exception as e:                                      # noqa: BLE001
        print(f"not built: {e}")
        return 1
    for s in rec["series"]:
        print(f"  {s['id']}: {s['field']} over {s['where']}, {s['months_both']} months "
              f"with both values, in {s['units']}")
    for p in rec["problems"]:
        print(f"  LEFT OUT  {p}")
    print(f"wrote {Path(str(entry['suite_dir'])) / FILE}")
    return 0 if rec["series"] and not rec["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
