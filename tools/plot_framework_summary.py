#!/usr/bin/env python3
"""What 13 one-sentence requests produced: a CONUS locator, and every design.

Not a results figure. Nothing has been simulated — every number describes a
DESIGN and the inputs built from it, so this says what was asked and what came
back, and nothing about whether the answers would be right. That belongs in the
caption: it is a caveat about what the figure omits, so it is not drawn on the
axes.

    python3 tools/plot_framework_summary.py --out draft.png

The verbatim requests were panel (b) and are now tools/make_request_table.py.
Thirteen rows of full sentences is a table; as a panel it forced 4 pt type and
ate two thirds of the canvas.

LEADER LINES ARE PLACED BY BEARING, THEN FIXED BY COUNTING. Ordering both the
basins and the ring slots by angle about the map centre gets most of the way,
and it is tempting to call that a guarantee — I did, twice, and it was wrong
twice. Angle ordering says nothing about RADIUS: St Vrain and Naches differ by
3.4 degrees of bearing but sit at very different distances from the centre, and
their lines crossed. So the bearing order is only a starting point, and
`untangle` then counts crossings exactly and swaps pairs while that count goes
down. A property worth claiming is worth measuring.

THE TWO NACHES YEARS ARE ONE RING ENTRY, stacked, sharing one leader line. Two
thumbnails on one dot cannot have distinct bearings, which is what broke the
placement in the first place. 1988, 1995 and 2020 came back identical to 2023
and four copies of one picture would read as repetition, not reproducibility.

EACH BOX IS SHAPED LIKE ITS BASIN, at equal MAP area. Fixed boxes left Smoky
filling 37% of its slot and Brandywine 42%, wasting half the ring and making the
tall basins read as the unimportant ones. Equal area, not equal box, so no basin
is given weight it has not earned. The thumbnails are shaped correctly but are
NOT to a common scale, so each carries its own area in km2.
"""
import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import figstyle                                                # noqa: E402
import plot_sampling_design as psd                             # noqa: E402

SCRATCH = ("/tmp/claude-297133/-qfs-people-tran289-IDEAS/"
           "4cb5f884-d1c3-431f-b573-ca37cad8f651/scratchpad/ib")

# Newest first: the verify re-run supersedes the 2026-08-08 chain where both have
# the case.
SOURCES = [("workflow_outputs/verify_20260809", f"{SCRATCH}/verify_runs"),
           ("workflow_outputs/chain_inbasin_20260808", f"{SCRATCH}/runs")]

CASES = ["brandywine_2010", "centralcoast_1998", "chattahoochee_2000",
         "chicopee_2018", "gunnison_2015", "naches_1979", "naches_1988",
         "naches_1995", "naches_2020", "naches_2023", "smoky_2012",
         "stvrain_2013", "verde_2005"]

# One ring entry per WATERSHED, holding the designs that watershed actually has.
ENTRIES = [["brandywine_2010"], ["centralcoast_1998"], ["chattahoochee_2000"],
           ["chicopee_2018"], ["gunnison_2015"],
           ["naches_1979", "naches_2023"],
           ["smoky_2012"], ["stvrain_2013"], ["verde_2005"]]

# Display only; the full names are in the request table. A 0.8 inch box cannot
# hold "Northern Big Smoky Valley" at a legible size.
SHORT = {"Northern Big Smoky Valley": "Big Smoky Valley",
         "Brandywine-Christina": "Brandywine",
         "Middle Chattahoochee": "Chattahoochee",
         "Chicopee River": "Chicopee"}

CONUS = [-125.0, -66.5, 23.5, 50.5]
# The locator wants RELIEF. Measured against eight alternatives at this extent:
# World Hillshade is a near-white box with no coast (relief and nothing else,
# which is right for a basin panel and useless for a country); Dark Gray inverts
# the leader lines' contrast at the map frame; NatGeo and Topo are unreadable
# text at 3 inches; Terrain Base's cyan ocean out-shouts the whole page. Shaded
# Relief is the one backdrop that adds the quantity this figure is about — the
# planner sizes each ensemble on relief — while staying low enough in saturation
# that the red dots remain the loudest thing on it.
LOCATOR_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}")

FIGW, FIGH = 7.2, 8.5
MAP_BOX = [0.27, 0.420, 0.46, 0.225]        # x0, y0, w, h, figure coordinates

# Nine slots around the map for nine watersheds. (x, y) is the slot CENTRE;
# `side` is the box edge a leader line attaches to, always the one facing the
# map. Boxes are sized per basin, so these are positions only.
SLOTS = [(0.185, 0.840, "bottom"), (0.500, 0.840, "bottom"),
         (0.815, 0.840, "bottom"),
         (0.910, 0.630, "left"), (0.910, 0.350, "left"),
         (0.770, 0.185, "top"), (0.480, 0.185, "top"), (0.185, 0.185, "top"),
         (0.090, 0.490, "right")]

MAP_AREA = 1.15              # square inches of MAP per entry, titles excluded
ASPECT_CLAMP = (0.55, 1.80)  # a sliver and a letterbox are both unreadable


def load(cid):
    for art_dir, run_dir in SOURCES:
        a = Path(art_dir) / f"{cid}.json"
        c = Path(run_dir) / cid / "01_inputs" / "columns.json"
        if a.is_file() and c.is_file():
            return json.loads(a.read_text()), json.loads(c.read_text())["columns"]
    raise FileNotFoundError(cid)


def stations(art):
    """(in basin, outside basin) over every station reception offered."""
    obs = ((art["reception"].get("brief") or {})
           .get("observations_summary")) or {}
    ins = out = 0
    for var, key in (("streamflow", "stations"), ("water_table", "wells"),
                     ("swe", "stations"), ("et", "towers")):
        for s in (obs.get(var) or {}).get(key) or []:
            ins += s.get("in_basin") is True
            out += s.get("in_basin") is False
    return ins, out


def refusals():
    """What the framework was offered and did not use.

    Counts only. WHY a gauge is never pinnable, and why ET never was, are
    claims and belong in the caption.
    """
    from collections import Counter
    offered, pinned_var = Counter(), Counter()
    n_col = n_pin = out_of_basin = 0
    for cid in CASES:
        art, cols = load(cid)
        obs = ((art["reception"].get("brief") or {})
               .get("observations_summary")) or {}
        for var, key in (("streamflow", "stations"), ("water_table", "wells"),
                         ("swe", "stations"), ("et", "towers")):
            offered[var] += len((obs.get(var) or {}).get(key) or [])
        n_col += len(cols)
        for c in cols:
            if c.get("pinned"):
                n_pin += 1
                pinned_var[c.get("station_variable")] += 1
        out_of_basin += stations(art)[1]
    return [
        f"{len(CASES)} requests · {n_col} columns · {n_pin} pinned "
        f"({pinned_var['swe']} SWE, {pinned_var['water_table']} water table)",
        f"{offered['streamflow']} streamflow gauges and {offered['et']} ET "
        f"towers offered · 0 pinned",
        f"{out_of_basin} station records tagged outside the basin · 0 pinned",
        "Naches 1988, 1995 and 2020 produced designs identical to 2023",
    ]


def boundary(art, cols):
    """The basin's largest ring, or the column cloud when there is no polygon."""
    rings = (art["reception"].get("grid") or {}).get("boundary")
    if rings:
        big = max(rings, key=len)
        return [p[0] for p in big], [p[1] for p in big]
    return [c["lon"] for c in cols], [c["lat"] for c in cols]


def area_km2(bx, by):
    """Shoelace on a local equal-distance projection. Good to ~1% at HUC8 size."""
    R, la0 = 6371.0, sum(by) / len(by)
    xs = [math.radians(x) * R * math.cos(math.radians(la0)) for x in bx]
    ys = [math.radians(y) * R for y in by]
    s = sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
            for i in range(len(xs)))
    return abs(s) / 2.0


def extent(bx, by):
    mx, my = 0.06 * (max(bx) - min(bx)), 0.06 * (max(by) - min(by))
    return [min(bx) - mx, max(bx) + mx, min(by) - my, max(by) + my]


def page_aspect(ext):
    """Width/height the basin occupies ON PAPER, after the latitude stretch."""
    lat = (ext[2] + ext[3]) / 2
    return (ext[1] - ext[0]) / ((ext[3] - ext[2]) / math.cos(math.radians(lat)))


def wrap_for(text, box_w_in, pt):
    """Wrap to a box `box_w_in` wide.

    0.64 em per character, not the 0.55 em DejaVu Sans averages over its whole
    table: these strings are digits, capitals and middots, every one of them
    wider than the average letter. _check_titles caught the difference at 110%.
    """
    return textwrap.wrap(text, max(8, int(box_w_in / (0.64 * pt / 72.0)))) \
        or [text]


def _crosses(p1, p2, q1, q2):
    """Do open segments p1-p2 and q1-q2 properly intersect?"""
    def side(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))
    d1, d2 = side(q1, q2, p1), side(q1, q2, p2)
    d3, d4 = side(p1, p2, q1), side(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def untangle(dots, bearings):
    """Assign entries to slots: no crossing leader lines, geography preserved.

    Crossings are counted exactly and dominate the cost, because a crossed pair
    is simply wrong. But removing a crossing is not the only goal: an
    arrangement can be crossing-free and still put the Arizona basin at the
    top-left, which is worse to read than the tangle it fixed. So the tie-break
    is the total angular mismatch against each basin's true bearing, and the
    search only ever moves to a strictly cheaper arrangement.

    Returns (assignment, crossings remaining).
    """
    slot_xy = [(x, y) for x, y, _ in SLOTS]
    slot_ang = []
    for x, y in slot_xy:
        slot_ang.append(math.atan2(x - (MAP_BOX[0] + MAP_BOX[2] / 2),
                                   y - (MAP_BOX[1] + MAP_BOX[3] / 2))
                        % (2 * math.pi))

    def crossings(a):
        ks = list(a)
        return sum(_crosses(dots[i], slot_xy[a[i]], dots[j], slot_xy[a[j]])
                   for m, i in enumerate(ks) for j in ks[m + 1:])

    def cost(a):
        c = crossings(a)
        ang = 0.0
        for i, s in a.items():
            d = abs(bearings[i] - slot_ang[s]) % (2 * math.pi)
            ang += min(d, 2 * math.pi - d)
        return 1000.0 * c + ang, c

    order = sorted(range(len(dots)), key=lambda i: bearings[i])
    best_assign, best = None, (1e18, 99)
    for rot in range(len(SLOTS)):        # the bearing seed, every rotation
        a = {ei: (rot + k) % len(SLOTS) for k, ei in enumerate(order)}
        if cost(a) < best:
            best_assign, best = a, cost(a)

    assign, improved = best_assign, True
    while improved:
        improved = False
        for i in assign:
            for j in assign:
                if i >= j:
                    continue
                assign[i], assign[j] = assign[j], assign[i]
                c = cost(assign)
                if c < best:
                    best, improved = c, True
                else:
                    assign[i], assign[j] = assign[j], assign[i]
    return assign, best[1]


def check_overlaps(boxes):
    """Warn when two thumbnail cells overlap. Reports; never moves anything.

    The first taller layout put Chattahoochee through Brandywine, and I found
    it by looking at the PNG. Boxes are sized from the DATA — a basin's shape
    decides its cell — so whether the ring fits is a property of the data and
    has to be checked on every figure, exactly like the titles.
    """
    over = []
    for m, (n1, x1, y1, w1, h1) in enumerate(boxes):
        for n2, x2, y2, w2, h2 in boxes[m + 1:]:
            dx = min(x1 + w1, x2 + w2) - max(x1, x2)
            dy = min(y1 + h1, y2 + h2) - max(y1, y2)
            if dx > 0 and dy > 0:
                over.append(f"{n1} overlaps {n2} "
                            f"({dx / min(w1, w2):.0%} x {dy / min(h1, h2):.0%})")
    for o in over:
        print(f"   \u26a0\ufe0f  {o}")
    return over


def thumbnail(ax, art, cols, label, pt, area=None, basemap=True):
    """One design: hillshade, basin outline, columns coloured by elevation."""
    import numpy as np
    bx, by = boundary(art, cols)
    ext = extent(bx, by)
    lat = np.array([c["lat"] for c in cols], float)
    lon = np.array([c["lon"] for c in cols], float)
    e = np.array([c["elevation_m"] for c in cols], float)
    e = (e - e.min()) / (np.ptp(e) or 1.0)
    pin = np.array([bool(c.get("pinned")) for c in cols])

    if basemap:
        psd._basemap(ax, ext, max_tiles=8)   # 1 inch cannot use zoom-10 detail
    if (art["reception"].get("grid") or {}).get("boundary"):
        ax.plot(bx, by, color="0.1", lw=0.5, zorder=5)
    ax.scatter(lon[~pin], lat[~pin], c=e[~pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=5, edgecolor="k", linewidth=0.2, zorder=3)
    ax.scatter(lon[pin], lat[pin], c=e[pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=22, marker="*", edgecolor="k", linewidth=0.3, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / math.cos(math.radians(float(lat.mean()))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    if label:
        ax.set_title(label, fontsize=pt, pad=1.6, linespacing=1.15)
    if area:
        # Inside the panel: the thumbnails are not to a common scale, and in the
        # title this one string cost every box two extra wrapped lines.
        ax.text(0.03, 0.03, area, transform=ax.transAxes, fontsize=pt,
                ha="left", va="bottom", color="0.15", zorder=6,
                bbox=dict(facecolor="w", alpha=0.7, pad=0.8, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="framework_summary.png")
    ap.add_argument("--width", type=float, default=FIGW)
    ap.add_argument("--font", type=float, default=8.0)
    ap.add_argument("--no-basemap", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch
    from reception_cases import CASES as CASE_DEFS

    name = {c["id"]: c["name"] for c in CASE_DEFS}
    small = max(6.0, a.font - 2.0)     # the journal floor, never scaled past
    line_h = small * 1.25 / 72.0       # one line of title, inches

    figstyle.manuscript(a.width, a.font)
    fig = plt.figure(figsize=(a.width, FIGH * a.width / FIGW))
    fw, fh = fig.get_size_inches()

    # ── the locator ──────────────────────────────────────────────────────────
    axm = fig.add_axes(MAP_BOX)
    if not a.no_basemap:
        psd._basemap(axm, CONUS, tiles=LOCATOR_TILES, max_tiles=90)
    axm.set_xlim(CONUS[0], CONUS[1]); axm.set_ylim(CONUS[2], CONUS[3])
    axm.set_aspect(1 / math.cos(math.radians(37.0)))
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_linewidth(0.4)

    data = {cid: load(cid) for cids in ENTRIES for cid in cids}

    # Geometry per entry: where its dot is, and how tall a stack of correctly
    # shaped maps plus their titles comes to.
    geom = []
    for cids in ENTRIES:
        bx, by = boundary(*data[cids[0]])
        r = page_aspect(extent(bx, by))
        r = min(max(r, ASPECT_CLAMP[0]), ASPECT_CLAMP[1])
        mw = math.sqrt(MAP_AREA / len(cids) * r)
        mh = math.sqrt(MAP_AREA / len(cids) / r)
        base = name.get(cids[0], cids[0])
        head = SHORT.get(base, base)
        labels, head_lines = [], []
        if len(cids) > 1:
            head_lines = wrap_for(head, mw, small)
        for cid in cids:
            cols = data[cid][1]
            npin = sum(1 for c in cols if c.get("pinned"))
            txt = f"{len(cols)} col · {npin} pinned"
            if len(cids) == 1:
                labels.append(wrap_for(f"{head} {cid.split('_')[1]}", mw, small)
                              + wrap_for(txt, mw, small))
            else:
                labels.append(wrap_for(f"{cid.split('_')[1]} · {txt}", mw, small))
        head_h = len(head_lines) * line_h + (0.01 if head_lines else 0.0)
        geom.append({"cids": cids, "mw": mw, "mh": mh, "labels": labels,
                     "head": head_lines, "head_h": head_h,
                     "lon": sum(bx) / len(bx), "lat": sum(by) / len(by),
                     "km2": area_km2(bx, by),
                     "h": head_h + sum(len(l) * line_h + 0.05 + mh
                                       for l in labels)})

    cx = MAP_BOX[0] + MAP_BOX[2] / 2
    cy = MAP_BOX[1] + MAP_BOX[3] / 2
    dots, bearings = [], []
    for g in geom:
        fx = MAP_BOX[0] + (g["lon"] - CONUS[0]) / (CONUS[1] - CONUS[0]) * MAP_BOX[2]
        fy = MAP_BOX[1] + (g["lat"] - CONUS[2]) / (CONUS[3] - CONUS[2]) * MAP_BOX[3]
        dots.append((fx, fy))
        bearings.append(math.atan2(fx - cx, fy - cy) % (2 * math.pi))
    slot_of, left = untangle(dots, bearings)
    print(f"   leader lines: {left} crossing(s)")

    cells = []
    for i, g in enumerate(geom):
        sx, sy, side = SLOTS[slot_of[i]]
        w, h = g["mw"] / fw, g["h"] / fh
        x0, top = sx - w / 2, sy + h / 2
        cells.append((g["cids"][0], x0, top - h, w, h))
        if g["head"]:
            fig.text(sx, top - g["head_h"] / fh + 0.004, "\n".join(g["head"]),
                     ha="center", va="bottom", fontsize=small,
                     linespacing=1.15)
        y = top - g["head_h"] / fh
        for j, cid in enumerate(g["cids"]):
            lab_h = (len(g["labels"][j]) * line_h + 0.05) / fh
            y -= lab_h + g["mh"] / fh
            thumbnail(fig.add_axes([x0, y, w, g["mh"] / fh]), *data[cid],
                      "\n".join(g["labels"][j]), small,
                      area=(f"{g['km2']:,.0f} km$^2$"
                            if j == len(g["cids"]) - 1 else None),
                      basemap=not a.no_basemap)

        axm.plot([g["lon"]], [g["lat"]], marker="o", ms=2.6, mfc="#d62728",
                 mec="k", mew=0.3, zorder=6)
        anchor = {"bottom": (sx, top - h), "top": (sx, top),
                  "left": (x0, sy), "right": (x0 + w, sy)}[side]
        fig.add_artist(ConnectionPatch(
            xyA=(g["lon"], g["lat"]), coordsA=axm.transData,
            xyB=anchor, coordsB=fig.transFigure,
            lw=0.4, color="0.35", zorder=1))

    # ── scale, key, and what was refused ─────────────────────────────────────
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1), cmap=psd.CMAP)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.44, 0.046, 0.235, 0.006]),
                      orientation="horizontal")
    cb.set_label("elevation, normalised per basin", fontsize=small,
                 labelpad=1.5)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=small, length=1.5, pad=1.0)

    for k, line in enumerate(refusals()):
        fig.text(0.025, 0.058 - 0.0155 * k, line, fontsize=small, color="0.25",
                 va="center")

    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=3,
               label="stratified column"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=6,
               label="pinned at an observation station"),
    ], loc="lower right", bbox_to_anchor=(0.985, 0.012), ncol=1, frameon=False,
        fontsize=small, handletextpad=0.4, labelspacing=0.35)

    fig.text(0.5, 0.978, "13 requests · 9 watersheds · 1979–2023",
             ha="center", va="center", fontsize=a.font + 2)

    check_overlaps(cells)
    psd._check_titles(fig)
    fig.savefig(Path(a.out))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
