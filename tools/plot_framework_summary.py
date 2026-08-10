#!/usr/bin/env python3
"""
DRAFT — what 13 one-sentence requests produced.

Not a results figure. Nothing has been simulated: every number here describes a
DESIGN and the inputs built from it, so the figure says what was asked and what
came back, and nothing about whether the answers would be right.

    (a) one thumbnail per distinct watershed — 9 of them
    (b) the Naches, five years apart
    (c) every request verbatim, against the ensemble it produced

Measurements only, per the figure rules: counts, relief, station tallies. No
claim is drawn on the axes.

    python3 tools/plot_framework_summary.py --out draft.png
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import figstyle                                                # noqa: E402
import plot_sampling_design as psd                             # noqa: E402

SCRATCH = ("/tmp/claude-297133/-qfs-people-tran289-IDEAS/"
           "4cb5f884-d1c3-431f-b573-ca37cad8f651/scratchpad/ib")

# Newest first: the verify re-run supersedes the 2026-08-08 chain for the four
# basins it covers.
SOURCES = [("workflow_outputs/verify_20260809", f"{SCRATCH}/verify_runs"),
           ("workflow_outputs/chain_inbasin_20260808", f"{SCRATCH}/runs")]

CASES = ["brandywine_2010", "centralcoast_1998", "chattahoochee_2000",
         "chicopee_2018", "gunnison_2015", "naches_1979", "naches_1988",
         "naches_1995", "naches_2020", "naches_2023", "smoky_2012",
         "stvrain_2013", "verde_2005"]

# One per distinct watershed. The Naches appears five times in CASES and gets
# its own panel, so a single year represents it here.
GALLERY = ["brandywine_2010", "centralcoast_1998", "chattahoochee_2000",
           "chicopee_2018", "gunnison_2015", "naches_2023", "smoky_2012",
           "stvrain_2013", "verde_2005"]
NACHES = ["naches_1979", "naches_1988", "naches_1995", "naches_2020",
          "naches_2023"]


def load(cid):
    """(artifact, built columns) for one case, from the newest run that has both."""
    for art_dir, run_dir in SOURCES:
        a = Path(art_dir) / f"{cid}.json"
        c = Path(run_dir) / cid / "01_inputs" / "columns.json"
        if a.is_file() and c.is_file():
            return json.loads(a.read_text()), json.loads(c.read_text())["columns"]
    raise FileNotFoundError(cid)


def stations(art):
    """(in-basin, outside) station counts across all four observation types."""
    obs = ((art["reception"].get("brief") or {})
           .get("observations_summary")) or {}
    ins = out = 0
    for var, key in (("streamflow", "stations"), ("water_table", "wells"),
                     ("swe", "stations"), ("et", "towers")):
        for s in (obs.get(var) or {}).get(key) or []:
            ins += s.get("in_basin") is True
            out += s.get("in_basin") is False
    return ins, out


def thumbnail(ax, art, cols, label, basemap=True):
    """One basin: hillshade, boundary, columns.

    Colour is elevation NORMALISED WITHIN THE BASIN. Nine basins spanning
    255-2297 m of relief cannot share an absolute scale, and nine separate
    colorbars would be unreadable, so the scale is per-basin and says so.
    """
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
        ax.plot(bx, by, color="0.1", lw=0.7, zorder=5)
    ax.scatter(lon[~pin], lat[~pin], c=e[~pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=7, edgecolor="k", linewidth=0.25, zorder=3)
    ax.scatter(lon[pin], lat[pin], c=e[pin], cmap=psd.CMAP, vmin=0, vmax=1,
               s=26, marker="*", edgecolor="k", linewidth=0.35, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians(float(lat.mean()))))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=5.5, pad=1.5)


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
    from reception_cases import CASES as CASE_DEFS

    query = {c["id"]: c["query"] for c in CASE_DEFS}
    name = {c["id"]: c["name"] for c in CASE_DEFS}

    # Fewer tiles than the full-size figure: a 1.6 inch thumbnail cannot show
    # zoom-10 detail, and 9 basins at 25 tiles each is 225 requests for nothing.
    psd.MAX_TILES = 8

    figstyle.manuscript(a.width, a.font)
    fig = plt.figure(figsize=(a.width, 9.3), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.85, 3.3])

    # (a) NINE WATERSHEDS ─────────────────────────────────────────────────────
    gsa = gs[0].subgridspec(3, 3, hspace=0.16, wspace=0.06)
    for i, cid in enumerate(GALLERY):
        art, cols = load(cid)
        ax = fig.add_subplot(gsa[divmod(i, 3)])
        n_pin = sum(1 for c in cols if c.get("pinned"))
        thumbnail(ax, art, cols,
                  f"{name.get(cid, cid)}  {len(cols)} col, {n_pin} pinned",
                  basemap=not a.no_basemap)
        if i == 0:
            ax.set_ylabel("(a)", rotation=0, ha="right", va="top",
                          fontsize=a.font + 1, labelpad=6)

    # (b) ONE WATERSHED, FIVE YEARS ───────────────────────────────────────────
    axb = fig.add_subplot(gs[1])
    yrs, strat, pins, ins_b = [], [], [], []
    for cid in NACHES:
        art, cols = load(cid)
        yrs.append(cid.split("_")[1])
        p = sum(1 for c in cols if c.get("pinned"))
        pins.append(p); strat.append(len(cols) - p)
        ins_b.append(stations(art)[0])
    x = np.arange(len(NACHES))
    axb.bar(x, strat, 0.55, color="0.75", label="stratified")
    axb.bar(x, pins, 0.55, bottom=strat, color="#2c7fb8", label="pinned")
    for i, (s, p, n) in enumerate(zip(strat, pins, ins_b)):
        axb.text(i, s + p + 0.5, f"{s + p}", ha="center", fontsize=a.font - 1)
        axb.text(i, -3.4, f"{n} in basin", ha="center", fontsize=a.font - 1.5,
                 color="0.35")
    axb.set_xticks(x); axb.set_xticklabels(yrs)
    axb.set_ylim(0, 24); axb.set_ylabel("columns")
    axb.set_title("(b) the Naches, five years", loc="left")
    axb.legend(loc="upper right", ncol=2, fontsize=a.font - 1)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # (c) THE REQUEST, AND WHAT IT PRODUCED ───────────────────────────────────
    axc = fig.add_subplot(gs[2])
    labels, S, P, extra = [], [], [], []
    for cid in CASES:
        art, cols = load(cid)
        p = sum(1 for c in cols if c.get("pinned"))
        P.append(p); S.append(len(cols) - p)
        i_in, i_out = stations(art)
        rel = (art["reception"].get("grid") or {}).get("relief_m") or 0
        extra.append(f"relief {rel:.0f} m · {i_in} stations in basin, {i_out} out")
        labels.append("\n".join(textwrap.wrap(query[cid], 92)))
    y = np.arange(len(CASES))[::-1]
    axc.barh(y, S, 0.55, color="0.75")
    axc.barh(y, P, 0.55, left=S, color="#2c7fb8")
    for yy, s, p, x_ in zip(y, S, P, extra):
        axc.text(s + p + 0.4, yy, f"{s + p} columns · {x_}",
                 va="center", fontsize=a.font - 1.5, color="0.25")
    axc.set_yticks(y)
    axc.set_yticklabels(labels, fontsize=a.font - 1.5, linespacing=1.15)
    axc.set_xlim(0, 34); axc.set_xlabel("columns built")
    axc.set_title("(c) every request, verbatim", loc="left")
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1), cmap=psd.CMAP)
    cb = fig.colorbar(sm, ax=fig.axes[:9], location="right", fraction=0.02,
                      shrink=0.5, pad=0.01)
    cb.set_label("elevation, normalised within each basin", fontsize=a.font - 1)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])

    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=3,
               label="stratified column"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=6,
               label="column pinned at an observation station"),
    ], loc="outside lower center", ncol=2, frameon=False)

    fig.suptitle("13 requests · 9 watersheds · 1979-2023 · nothing simulated yet",
                 fontsize=a.font + 2)
    out = Path(a.out)
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
