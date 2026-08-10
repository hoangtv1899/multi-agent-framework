#!/usr/bin/env python3
"""
DRAFT — what 13 one-sentence requests produced.

Not a results figure. Nothing has been simulated: every number describes a
DESIGN and the inputs built from it, so this says what was asked and what came
back, and nothing about whether the answers would be right.

    (a) CONUS locator, with a leader line from each watershed to its design
    (b) every request verbatim, against the ensemble it produced

Measurements only: counts, relief, station tallies. No claim is drawn on the
axes.

THE THUMBNAILS ARE PLACED BY BEARING. Each sits in the ring slot whose angle
from the map centre best matches its basin's, so leader lines cannot cross and
adding a basin later does not require re-tuning by hand.

THE NACHES APPEARS TWICE, 1979 and 2023, because those are the two designs it
has: 1988, 1995 and 2020 are identical to 2023, and four copies of one picture
would read as repetition rather than reproducibility.

    python3 tools/plot_framework_summary.py --out draft.png
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

GALLERY = ["brandywine_2010", "centralcoast_1998", "chattahoochee_2000",
           "chicopee_2018", "gunnison_2015", "naches_1979", "naches_2023",
           "smoky_2012", "stvrain_2013", "verde_2005"]

CONUS = [-125.0, -66.5, 23.5, 50.5]
# The LOCATOR needs outlines, not relief: at CONUS scale World Hillshade is a
# white box with no coast and no state lines. Light Gray Base carries exactly
# the outlines and nothing else. Thumbnails keep the hillshade, where the relief
# IS the information.
LOCATOR_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

# Ten slots around the map, clockwise from the top-left. (x, y) is the slot
# centre in figure coordinates; `side` is the thumbnail edge a leader line
# should attach to, which is always the one facing the map.
SLOTS = [(0.20, 0.905, "bottom"), (0.50, 0.905, "bottom"), (0.80, 0.905, "bottom"),
         (0.885, 0.745, "left"), (0.885, 0.605, "left"),
         (0.80, 0.445, "top"), (0.50, 0.445, "top"), (0.20, 0.445, "top"),
         (0.115, 0.605, "right"), (0.115, 0.745, "right")]
TW, TH = 0.150, 0.088          # thumbnail size, figure coordinates
MAP_BOX = [0.275, 0.545, 0.45, 0.26]     # x0, y0, w, h


def load(cid):
    for art_dir, run_dir in SOURCES:
        a = Path(art_dir) / f"{cid}.json"
        c = Path(run_dir) / cid / "01_inputs" / "columns.json"
        if a.is_file() and c.is_file():
            return json.loads(a.read_text()), json.loads(c.read_text())["columns"]
    raise FileNotFoundError(cid)


def stations(art):
    obs = ((art["reception"].get("brief") or {})
           .get("observations_summary")) or {}
    ins = out = 0
    for var, key in (("streamflow", "stations"), ("water_table", "wells"),
                     ("swe", "stations"), ("et", "towers")):
        for s in (obs.get(var) or {}).get(key) or []:
            ins += s.get("in_basin") is True
            out += s.get("in_basin") is False
    return ins, out


def centroid(art, cols):
    rings = (art["reception"].get("grid") or {}).get("boundary")
    if rings:
        big = max(rings, key=len)
        return (sum(p[0] for p in big) / len(big),
                sum(p[1] for p in big) / len(big))
    return (sum(c["lon"] for c in cols) / len(cols),
            sum(c["lat"] for c in cols) / len(cols))


def assign_slots(bearings):
    """Slot index per basin, chosen so leader lines do not cross.

    Both basins and slots are ordered by angle about the map centre; the only
    freedom left is where the two sequences start relative to each other, so
    try all ten rotations and keep the one with the least total angular
    mismatch. Cheap, and it removes the hand-tuning that makes callout figures
    fragile.
    """
    n = len(bearings)
    order = sorted(range(n), key=lambda i: bearings[i])
    slot_ang = []
    for x, y, _ in SLOTS:
        dx = x - (MAP_BOX[0] + MAP_BOX[2] / 2)
        dy = y - (MAP_BOX[1] + MAP_BOX[3] / 2)
        slot_ang.append(math.atan2(dx, dy) % (2 * math.pi))
    best, best_cost = None, 1e9
    for rot in range(len(SLOTS)):
        cost, trial = 0.0, {}
        for k, bi in enumerate(order):
            s = (rot + k) % len(SLOTS)
            d = abs(bearings[bi] - slot_ang[s]) % (2 * math.pi)
            cost += min(d, 2 * math.pi - d)
            trial[bi] = s
        if cost < best_cost:
            best, best_cost = trial, cost
    return best


def thumbnail(ax, art, cols, label, basemap=True):
    """One basin: hillshade, boundary, columns coloured by within-basin elevation."""
    import numpy as np
    rings = (art["reception"].get("grid") or {}).get("boundary")
    lat = np.array([c["lat"] for c in cols], float)
    lon = np.array([c["lon"] for c in cols], float)
    e = np.array([c["elevation_m"] for c in cols], float)
    e = (e - e.min()) / (np.ptp(e) or 1.0)
    pin = np.array([bool(c.get("pinned")) for c in cols])

    if rings:
        big = max(rings, key=len)
        bx = [p[0] for p in big]; by = [p[1] for p in big]
    else:
        bx, by = list(lon), list(lat)
    mx, my = 0.06 * (max(bx) - min(bx)), 0.06 * (max(by) - min(by))
    ext = [min(bx) - mx, max(bx) + mx, min(by) - my, max(by) + my]
    if basemap:
        psd._basemap(ax, ext)
    if rings:
        ax.plot(bx, by, color="0.1", lw=0.6, zorder=5)
    ax.scatter(lon[~pin], lat[~pin], c=e[~pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=5, edgecolor="k", linewidth=0.2, zorder=3)
    ax.scatter(lon[pin], lat[pin], c=e[pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=20, marker="*", edgecolor="k", linewidth=0.3, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / math.cos(math.radians(float(lat.mean()))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.set_title(label, fontsize=4.6, pad=1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="framework_summary_draft.png")
    ap.add_argument("--width", type=float, default=7.2)
    ap.add_argument("--font", type=float, default=6.0)
    ap.add_argument("--no-basemap", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch
    from reception_cases import CASES as CASE_DEFS

    query = {c["id"]: c["query"] for c in CASE_DEFS}
    name = {c["id"]: c["name"] for c in CASE_DEFS}

    figstyle.manuscript(a.width, a.font)
    fig = plt.figure(figsize=(a.width, 10.2))

    # (a) LOCATOR ─────────────────────────────────────────────────────────────
    axm = fig.add_axes(MAP_BOX)
    psd.MAX_TILES = 40
    if not a.no_basemap:
        hill, psd.TILES = psd.TILES, LOCATOR_TILES
        psd._basemap(axm, CONUS)
        psd.TILES = hill
    axm.set_xlim(CONUS[0], CONUS[1]); axm.set_ylim(CONUS[2], CONUS[3])
    axm.set_aspect(1 / math.cos(math.radians(37.0)))
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_linewidth(0.4)

    data = {cid: load(cid) for cid in GALLERY}
    cents = {cid: centroid(*data[cid]) for cid in GALLERY}
    cx = MAP_BOX[0] + MAP_BOX[2] / 2
    cy = MAP_BOX[1] + MAP_BOX[3] / 2
    bear = []
    for cid in GALLERY:
        lo, la = cents[cid]
        fx = (lo - CONUS[0]) / (CONUS[1] - CONUS[0])
        fy = (la - CONUS[2]) / (CONUS[3] - CONUS[2])
        bear.append(math.atan2(MAP_BOX[0] + fx * MAP_BOX[2] - cx,
                               MAP_BOX[1] + fy * MAP_BOX[3] - cy) % (2 * math.pi))
    slot_of = assign_slots(bear)

    psd.MAX_TILES = 8           # a 1 inch thumbnail cannot use zoom-10 detail
    for i, cid in enumerate(GALLERY):
        art, cols = data[cid]
        sx, sy, side = SLOTS[slot_of[i]]
        ax = fig.add_axes([sx - TW / 2, sy - TH / 2, TW, TH])
        n_pin = sum(1 for c in cols if c.get("pinned"))
        yr = cid.split("_")[1]
        thumbnail(ax, art, cols,
                  f"{name.get(cid, cid)} {yr}\n{len(cols)} col · {n_pin} pinned",
                  basemap=not a.no_basemap)
        lo, la = cents[cid]
        axm.plot([lo], [la], marker="o", ms=2.4, mfc="#d62728", mec="k",
                 mew=0.3, zorder=6)
        anchor = {"bottom": (0.5, 0.0), "top": (0.5, 1.0),
                  "left": (0.0, 0.5), "right": (1.0, 0.5)}[side]
        fig.add_artist(ConnectionPatch(
            xyA=(lo, la), coordsA=axm.transData,
            xyB=anchor, coordsB=ax.transAxes,
            lw=0.4, color="0.35", zorder=1))

    # (b) THE REQUEST, AND WHAT IT PRODUCED ───────────────────────────────────
    axc = fig.add_axes([0.40, 0.055, 0.40, 0.315])
    labels, S, P, extra = [], [], [], []
    for cid in CASES:
        art, cols = load(cid)
        p = sum(1 for c in cols if c.get("pinned"))
        P.append(p); S.append(len(cols) - p)
        i_in, i_out = stations(art)
        rel = (art["reception"].get("grid") or {}).get("relief_m") or 0
        extra.append(f"{len(cols)} col · relief {rel:.0f} m · "
                     f"{i_in} stations in basin, {i_out} out")
        labels.append("\n".join(textwrap.wrap(query[cid], 74)))
    y = np.arange(len(CASES))[::-1]
    axc.barh(y, S, 0.55, color="0.75")
    axc.barh(y, P, 0.55, left=S, color="#2c7fb8")
    for yy, s, p, x_ in zip(y, S, P, extra):
        axc.text(s + p + 0.5, yy, x_, va="center", fontsize=4.2, color="0.25")
    axc.set_yticks(y)
    axc.set_yticklabels(labels, fontsize=4.2, linespacing=1.1)
    axc.set_xlim(0, 20); axc.set_xlabel("columns built", fontsize=a.font - 1, labelpad=1.5)
    axc.tick_params(axis="x", labelsize=a.font - 2)
    axc.set_title("(b) every request, verbatim", loc="left", fontsize=a.font,
              pad=3)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    fig.text(0.02, 0.965, "(a)", fontsize=a.font + 1, fontweight="bold")
    fig.text(0.5, 0.395, "1988, 1995 and 2020 produced designs identical to "
                     "Naches 2023", ha="center", fontsize=4.6, color="0.35")

    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1), cmap=psd.CMAP)
    cax = fig.add_axes([0.035, 0.835, 0.115, 0.006])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("elevation, normalised within each basin", fontsize=4.4,
                 labelpad=1.5)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=4.4, length=1.5, pad=1.0)

    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=3,
               label="stratified column"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=6,
               label="column pinned at an observation station"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False,
        fontsize=a.font - 1)

    fig.suptitle("13 requests · 9 watersheds · 1979-2023 · nothing simulated yet",
                 fontsize=a.font + 2, y=0.985)
    fig.savefig(Path(a.out))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
