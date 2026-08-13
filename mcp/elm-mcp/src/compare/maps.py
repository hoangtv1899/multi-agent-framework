#!/usr/bin/env python3
"""The spatial comparison: one ROW per observable, observation left, ELM right.

PORTED FROM step1_maps.py (2026-08-12, the user's decision). This file used to
draw a SITING map — where the columns and the stations are, with no values on it
at all — because when comparison moved into this package that is what got
written, and the value map stayed behind in the legacy analyzer where it has
never once rendered. Two maps, one slot, and the one in the slot answered the
smaller question.

    swe          SNOTEL                    |  ELM      mm
    streamflow   USGS gauges               |  ELM      mm/day, log
    water_table  Fan 2013 (+ well overlay) |  ELM      m, log
    et           AmeriFlux                 |  ELM      mm/day

A SCALE PER ROW, NEVER PER FIGURE. The rows are millimetres, millimetres per day
and metres; one scale across them would be arithmetic on unlike quantities and a
colour would mean three things at once. Within a row the two panels DO share, and
that is the entire reason to draw them adjacent — independent scales map each
field's own maximum to the same colour and make fields that differ by orders of
magnitude look alike.

NO "OBSERVED" / "MODELLED" COLUMN HEADERS. Fan 2013 is a compilation of
long-term means, not this year's measurement, and a header claiming otherwise
would misrepresent the one row where there is usually no observation at all.
Each panel names its own source; the row is identified by its colourbar.

THE OVERLAY exists because water table has three sources for two slots: Fan
takes the panel because it exists nearly everywhere, and the recorder wells ride
on top in their own marker, sharing the row's scale so the values stay
comparable. A basin with no well gets no overlay rather than an empty panel.

THIS FILE DECIDES LAYOUT AND NOTHING ELSE. Every point comes from the
observable's own map_points(), so the module that knows what H2OSNO means is the
module that averages it, and this figure can never disagree with the record
beside it about where a station is or what it read.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import _common as C

MAP_CMAP = "viridis"
OBS_MARKER = "o"
MODEL_MARKER = "s"


def _row_points(row: Dict) -> List:
    out = []
    for p in row.get("panels") or []:
        out += list(p.get("points") or [])
        out += list((p.get("overlay") or {}).get("points") or [])
    return out


def plot_all(records: Dict[str, Dict], rows: List[Dict], meta: Dict,
             out_path: str, series: Optional[Dict] = None,
             reception_json: Optional[str] = None,
             order: Optional[List[str]] = None) -> Optional[str]:
    """The grid. `records` is compare_all's per-observable output.

    A row is drawn when EITHER panel has a point. An observable with no
    observation still shows its model field beside an empty panel, labelled —
    which states the coverage gap far better than a row silently dropped. Two
    of thirteen chain-eval basins had a flux tower; that absence is a finding.
    """
    from . import OBSERVABLES

    spec_rows = []
    for name in (order or ["swe", "water_table", "streamflow", "et"]):
        mod = OBSERVABLES.get(name)
        rec = (records or {}).get(name)
        if mod is None or rec is None or not hasattr(mod, "map_points"):
            continue
        try:
            row = mod.map_points(rec, rows, series or {}, meta,
                                 reception_json=reception_json)
        except Exception as e:                                  # noqa: BLE001
            print(f"   map_points failed for {name}: {type(e).__name__}: {e}")
            continue
        if row and _row_points(row):
            row["name"] = name
            spec_rows.append(row)
    if not spec_rows:
        return None

    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "axes.titlesize": 12, "legend.fontsize": 9})
    nrow = len(spec_rows)

    # THE FIGURE IS SIZED FROM THE BASIN'S SHAPE, not fixed. A map axis holds a
    # fixed aspect, so a width chosen in advance leaves the difference as dead
    # space: Brandywine is twice as tall as it is wide, and at a fixed 12.4 in
    # the two panels sat marooned with three inches of white between them. Hold
    # the panel HEIGHT and let the width follow the extent.
    xs = [q[0] for r in spec_rows for q in _row_points(r)]
    ys = [q[1] for r in spec_rows for q in _row_points(r)]
    dlon = (max(xs) - min(xs)) if xs else 1.0
    dlat = (max(ys) - min(ys)) if ys else 1.0
    mid = (sum(ys) / len(ys)) if ys else 40.0
    wh = max(0.25, min(4.0, (dlon * math.cos(math.radians(mid))) / (dlat or 1)))
    panel_h = 4.0
    fig_w = max(6.0, 2 * panel_h * wh + 3.0)
    fig, axes = plt.subplots(nrow, 2, figsize=(fig_w, panel_h * nrow),
                             squeeze=False)
    # Room for the tick labels between rows: each row is its own map and keeps
    # its own longitude axis, so without this the labels land on the title of
    # the row below.
    fig.subplots_adjust(hspace=0.34, wspace=0.12)

    for ri, row in enumerate(spec_rows):
        vals = [q[2] for q in _row_points(row) if q[2] is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        norm = None
        if row.get("log"):
            pos = [v for v in vals if v > 0]
            if pos and max(pos) / min(pos) > 20:
                # FLOORED AT FOUR DECADES, the same cap streamflow's own panel
                # uses. Naches columns run down to 1e-6 mm/day, and honouring
                # that put every real value in the top sixth of the colourbar.
                norm = LogNorm(vmin=max(min(pos), max(pos) / 1e4),
                               vmax=max(pos))
        if norm is None:
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)

        sc = None
        for ci, panel in enumerate(row["panels"]):
            ax = axes[ri][ci]
            # The hillshade is fetched once per process and reused, so eight
            # panels over one basin cost one trip to the tile server.
            C.basin_backdrop(ax, reception_json, scale_bar=(ri == 0 and ci == 0),
                             avoid=[(q[0], q[1]) for q in _row_points(row)])
            pts = list(panel.get("points") or [])
            if pts:
                sizes = panel.get("sizes")
                # SMALLER WHEN THERE ARE MANY. Fan hands over 131 sites inside
                # Brandywine and at one fixed size they merge into a single
                # blob that hides the basin under them.
                s_default = 150 if len(pts) <= 40 else 55
                sc = ax.scatter([q[0] for q in pts], [q[1] for q in pts],
                                c=[q[2] for q in pts], cmap=MAP_CMAP, norm=norm,
                                s=(list(sizes) if sizes else s_default),
                                marker=(MODEL_MARKER if ci else OBS_MARKER),
                                edgecolor="white", linewidth=1.2, zorder=6)
            else:
                ax.text(0.5, 0.04, "none in this basin", ha="center",
                        va="bottom", transform=ax.transAxes, fontsize=10,
                        color="#8A5A44",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
            ov = panel.get("overlay") or {}
            if ov.get("points"):
                o = ov["points"]
                sc = ax.scatter([q[0] for q in o], [q[1] for q in o],
                                c=[q[2] for q in o], cmap=MAP_CMAP, norm=norm,
                                s=170, marker=ov.get("marker", "D"),
                                edgecolor="#111", linewidth=1.4, zorder=7,
                                label=f"{ov.get('title')} ({len(o)})")
                C.legend(ax)
            n = len(pts)
            ax.set_title(f"{panel['title']}  ({n})" if n else panel["title"],
                         fontsize=11)
            if ci == 0:
                ax.set_ylabel("latitude")
            if ri == nrow - 1:
                ax.set_xlabel("longitude")
            from matplotlib.ticker import MaxNLocator
            # THREE, and rotated. A narrow panel fitted four longitude labels
            # by running them together: "-76.0-75.8-75.6".
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=10)

        if sc is not None:
            cb = fig.colorbar(sc, ax=list(axes[ri]), fraction=0.030, pad=0.02)
            cb.set_label(row["label"])

    # BBOX TIGHT, not tight_layout: the colourbars are attached to the axes
    # list rather than to one axis, and tight_layout measures neither them nor
    # a legend anchored outside. The first render of this figure cropped its
    # own y-axis label away.
    from pathlib import Path as _P
    _P(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
