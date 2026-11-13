#!/usr/bin/env python3
"""
The places a study's numbers are reported for, for the design pages
mcp/elm-mcp/src/domain_places.py

A design page named the regions its numbers are reported for, such as
"watershed, vegetated" and its elevation bands, by their labels alone. A
person could not see where they lie or how large they are. The details page
now shows them on a map, with each region's area in a legend (the review of
2026-11-13).

WHO DOES WHAT. This module is the model server's half: it reads the regions
from the control's results store (domain_suite), which keeps one mask per
region in masks.nc with each cell's area, and works out what a map shows.
The design page's code draws it (agents/places_figure.py). The record is the
contract between the two halves:

    page_record(entry, regions) -> None for a control with no results store,
      else {"control", "regions", "lat", "lon", "filled", "classes",
            "outlines", "problems"}

    regions   the labels asked for, as asked
    lat, lon  the cell centres in degrees, lon between -180 and 180
    filled    [{"label", "cells", "area_km2"}]: the regions drawn as colours
    classes   lat by lon: 0 where a cell is in no filled region, k where it is
              in filled[k-1]
    outlines  [{"label", "cells", "area_km2", "mask"}]: the regions drawn as
              an edge, each mask lat by lon of 0 and 1

THE RULES. A region that holds another region asked for entirely is drawn as
an outline, since its colour would hide the other. A region that holds no
other is filled, and so is a region asked for alone. A cell in two filled
regions is drawn as the one with the smaller area, and `problems` says that
the two overlap. A label the store does not know is named in `problems` and
left out, as is a region with no cell.

THE STORE'S OWN NAMES AND AREAS. The labels are the store's, and each area is
the one the store wrote beside its mask, which its manifest and the catalog
give too, so a page never shows a second area for the same region.

NOTHING HERE RAISES. A part that cannot be given is named under `problems`.
The record goes to the design pages alone, saved as JSON beside each design,
so it is kept small: masks as 0 and 1, and the coordinates to six decimals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# cell centres are kept to this many decimals of a degree (about 0.1 m)
DIGITS = 6


def _store_masks(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, int, float]]]:
    """(lat, lon, {region: (mask, cells, area_km2)}) from the store's
    masks.nc, each area the one the store wrote beside the mask."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        lat = np.asarray(d["lat"][:], dtype=np.float64)
        lon = np.asarray(d["lon"][:], dtype=np.float64)
        weight = np.asarray(d["weight_km2"][:], dtype=np.float64) \
            if "weight_km2" in d.variables else None
        masks: Dict[str, Tuple[np.ndarray, int, float]] = {}
        for vn, v in d.variables.items():
            if not vn.startswith("mask_"):
                continue
            m = np.asarray(v[:]) == 1
            attrs = set(v.ncattrs())
            name = str(v.region) if "region" in attrs else vn[len("mask_"):]
            cells = int(v.n_cells) if "n_cells" in attrs else int(m.sum())
            if "area_km2" in attrs:
                area = float(v.area_km2)
            elif weight is not None:
                area = float(weight[m].sum())
            else:
                area = float("nan")
            masks[name] = (m, cells, area)
    return lat, lon, masks


def _holds(outer: np.ndarray, inner: np.ndarray) -> bool:
    """True when `outer` holds every cell of `inner` and at least one more.
    Two regions of the same cells hold neither the other: both are filled,
    and their overlap is named."""
    return bool(np.all(outer[inner]) and np.any(outer & ~inner))


def _grid(a: np.ndarray) -> List[List[int]]:
    return [[int(x) for x in row] for row in np.asarray(a)]


def page_record(entry: Optional[Dict[str, Any]], regions: Sequence[str]
                ) -> Optional[Dict[str, Any]]:
    """The places the numbers are reported for, as a map shows them: see the
    module's notes for the record. None for a control with no results store
    (no suite_dir, or no masks.nc in it). Never raises."""
    if not isinstance(entry, dict):
        return None
    import domain_suite as S
    sd = entry.get("suite_dir")
    path = Path(str(sd)) / S.MASKS_FILE if sd else None
    if path is None or not path.is_file():
        return None
    if isinstance(regions, str):
        regions = [regions]
    asked = list(regions) if isinstance(regions, (list, tuple)) else []
    rec: Dict[str, Any] = {"control": entry.get("name"), "regions": asked,
                           "lat": [], "lon": [], "filled": [], "classes": [],
                           "outlines": [], "problems": []}
    if not asked:
        rec["problems"].append("no region was asked for")
        return rec
    try:
        lat, lon, masks = _store_masks(path)
    except Exception as e:                                          # noqa: BLE001
        rec["problems"].append(f"the results store's regions cannot be read "
                               f"({type(e).__name__}: {e})")
        return rec

    # the labels to draw: the store's, once each, in the order asked
    use: List[str] = []
    for lab in asked:
        if not isinstance(lab, str) or not lab.strip():
            rec["problems"].append(f"{lab!r} is not a region's name, so it is left out")
        elif lab in use:
            continue
        elif lab not in masks:
            rec["problems"].append(f"the results store has no region {lab!r}, so it is left "
                                   f"out (it has {', '.join(sorted(masks))})")
        elif not masks[lab][0].any():
            rec["problems"].append(f"the region {lab!r} holds no cell, so it is left out")
        else:
            use.append(lab)

    regular = lat.ndim == 1 and lon.ndim == 1
    shape = (len(lat), len(lon)) if regular else None
    for lab in list(use):
        if regular and masks[lab][0].shape != shape:
            rec["problems"].append(f"the region {lab!r} is not on the grid of the store's "
                                   f"latitudes and longitudes, so it is left out")
            use.remove(lab)
    if regular:
        import gridded as G
        rec["lat"] = [round(float(x), DIGITS) for x in lat]
        # the store writes them in [-180, 180) already; a store written
        # otherwise is brought there the way the history reader does it
        rec["lon"] = [round(float(x), DIGITS) for x in G.to_pm180(lon)]
    else:
        rec["problems"].append("the store's grid has no axes of latitude and longitude, so "
                               "the regions cannot be drawn on a map")

    outline = {a for a in use if any(_holds(masks[a][0], masks[b][0]) for b in use if b != a)}
    filled = [a for a in use if a not in outline]
    for a in use:
        m, cells, area = masks[a]
        row = {"label": a, "cells": cells, "area_km2": area}
        if a in outline:
            rec["outlines"].append(dict(row, mask=_grid(m.astype(np.int8)) if regular else []))
        else:
            rec["filled"].append(row)

    # a cell in two filled regions goes to the smaller: the larger regions are
    # painted first, and of two of one area the one asked for first is
    # painted last
    for i, a in enumerate(filled):
        for b in filled[i + 1:]:
            shared = int(np.sum(masks[a][0] & masks[b][0]))
            if shared:
                small = a if masks[a][2] <= masks[b][2] else b
                rec["problems"].append(f"the regions {a!r} and {b!r} overlap in {shared} "
                                       f"cell{'' if shared == 1 else 's'}, which are drawn as "
                                       f"{small!r}, the smaller")
    if regular and filled:
        classes = np.zeros(shape, dtype=np.int64)
        order = sorted(range(len(filled)), key=lambda k: (-masks[filled[k]][2], -k))
        for k in order:
            classes[masks[filled[k]][0]] = k + 1
        rec["classes"] = _grid(classes)
    elif regular:
        rec["classes"] = _grid(np.zeros(shape, dtype=np.int64))
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    """python domain_places.py <control> <region> [<region> ...]: the record
    for a registered control, without its grids, to read by eye."""
    import json
    import sys
    import domain_controls as DCT
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 2:
        print("usage: domain_places.py <control> <region> [<region> ...]")
        return 2
    rec = page_record(DCT.get(a[0]), a[1:])
    if rec is None:
        print(f"{a[0]} has no results store")
        return 1
    shown = {k: v for k, v in rec.items() if k not in ("lat", "lon", "classes")}
    shown["outlines"] = [{k: v for k, v in o.items() if k != "mask"} for o in rec["outlines"]]
    print(json.dumps(shown, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
