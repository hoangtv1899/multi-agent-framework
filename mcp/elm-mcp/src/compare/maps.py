#!/usr/bin/env python3
"""One figure, every observable over the same ground.

Separate from the per-observable plots because it answers a different question:
those ask "does the model match here", this asks "where are we even looking".
A basin where every station sits in the valley and every column on the ridge is
a design problem no metric will surface, and it is obvious in one glance here.

No basemap tiles. Fetching them needs the network and an SSL trust store, and a
figure that silently renders without its backdrop is worse than one that never
had one — this draws the columns and the stations and nothing else.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import _common as C

MARK = {"swe": ("o", "#2C6A5C"), "wtd": ("s", "#3A5B78"),
        "streamflow": ("^", "#6B4A86"), "et": ("D", "#A4522A")}


def plot_all(records: Dict[str, Dict], rows: List[Dict], meta: Dict,
             out_path: str) -> Optional[str]:
    """Columns as grey points, stations coloured by observable.

    Station markers are ringed when the station was ASSIGNED to a column, plain
    when it was not — an unassigned station is present in the basin and absent
    from every comparison, which is exactly the thing worth spotting.
    """
    cols = [(r.get("lon"), r.get("lat"), r.get("case_name"))
            for r in rows or [] if r.get("lat") is not None]
    have = {k: v for k, v in (records or {}).items() if not v.get("error")}
    if not cols and not have:
        return None

    fig, ax = C.new_figure(ncols=1, width=7.4, height=6.2)
    if cols:
        ax.scatter([c[0] for c in cols], [c[1] for c in cols], s=26,
                   color="#9AA7B0", zorder=2, label=f"columns ({len(cols)})")

    for name, rec in have.items():
        marker, colour = MARK.get(name, ("o", "#444444"))
        # THREE RECORD SHAPES, ALL READ (2026-08-12). swe and wtd match their
        # stations to columns FIRST and report only the matches, so their
        # `pairs` are already pairs and the stations that missed out sit in
        # `unpaired_stations`. streamflow does not pair at all — a gauge
        # measures an area, so it stands against the ensemble mean and every
        # in-basin gauge is used, which is why its `gauges` all count as
        # assigned and none is ever unassigned. et still compares everything
        # against everything and reports the bijection under `assignment`. This
        # reads any of the three, so the map keeps working while the observables
        # are converted one at a time.
        assigned = ({a["station_id"] for a in
                     (rec.get("assignment") or {}).get("pairs") or []}
                    or {e["station_id"] for e in (rec.get("pairs") or [])
                        if e.get("case_name")}
                    or {g["station_id"] for g in (rec.get("gauges") or [])})
        seen = (list(rec.get("pairs") or [])
                + list(rec.get("unpaired_stations") or [])
                + list(rec.get("gauges") or []))
        xs, ys, ring_x, ring_y = [], [], [], []
        for e in seen:
            sid = e.get("station_id")
            m = meta.get((sid, name)) or {}
            if not sid or m.get("lat") is None:
                continue
            (ring_x if sid in assigned else xs).append(m["lon"])
            (ring_y if sid in assigned else ys).append(m["lat"])
        if ring_x:
            ax.scatter(ring_x, ring_y, s=90, marker=marker, facecolors="none",
                       edgecolors=colour, linewidths=1.8, zorder=4,
                       label=f"{name} · assigned ({len(ring_x)})")
        if xs:
            ax.scatter(xs, ys, s=46, marker=marker, color=colour, alpha=0.55,
                       zorder=3, label=f"{name} · unassigned ({len(xs)})")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("columns and observation sites")
    ax.set_aspect("equal", adjustable="datalim")
    C.legend(ax)
    return C.save(fig, out_path)
