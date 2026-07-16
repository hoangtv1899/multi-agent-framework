#!/usr/bin/env python3
"""
Cold-start vs Fan-informed warm-start comparison (paper Fig 4) — same columns,
same forcing, same year; only the initial water table differs. Neutral,
publication-toned titles; the editorial reading belongs in the caption.

    python3 tools/make_cold_vs_warm.py \
        [--cold workflow_outputs/naches_nldas_20col] \
        [--warm workflow_outputs/naches_warmstart]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def by_col(hs, key):
    return {r["case_name"]: r["metrics"].get(key)
            for r in hs["experiments"] if r["status"] == "ok"}


def zwt(hs, which):
    return {r["case_name"]: (r["variables"].get("ZWT") or {}).get(which)
            for r in hs["experiments"] if r["status"] == "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", default="workflow_outputs/naches_nldas_20col")
    ap.add_argument("--warm", default="workflow_outputs/naches_warmstart")
    args = ap.parse_args()

    cold = json.loads(Path(args.cold, "04_analysis", "hydro_summary.json").read_text())
    warm = json.loads(Path(args.warm, "04_analysis", "hydro_summary.json").read_text())
    cj = json.loads(Path(args.warm, "columns.json").read_text())
    fan = {c["id"]: c.get("fan_wtd_m") for c in cj["columns"]}
    elev = {r["case_name"]: r.get("elevation_m") for r in warm["experiments"]}

    rc, rw = by_col(cold, "annual_recharge_mm_yr"), by_col(warm, "annual_recharge_mm_yr")
    zi, ze = zwt(warm, "first_m"), zwt(warm, "last_m")
    cols = sorted(set(rc) & set(rw), key=lambda c: elev.get(c) or 0)
    x = np.arange(len(cols))
    names = [f"{c.split('_')[1]}\n{elev[c]:.0f} m" for c in cols]

    fig, ax = plt.subplots(2, 1, figsize=(13, 7.6))
    w = 0.38
    ax[0].bar(x - w / 2, [rc[c] for c in cols], w, color="#9ecae1",
              edgecolor="#222", label="cold start (uniform 8.8 m default)")
    ax[0].bar(x + w / 2, [rw[c] for c in cols], w, color="#08519c",
              edgecolor="#222", label="warm start (Fan-2013 water table)")
    ax[0].axhline(0, color="#444", lw=.8)
    ax[0].set_ylabel("recharge (mm yr$^{-1}$)")
    ax[0].set_title("(a) Annual recharge per column under the two initializations "
                    "(negative = aquifer discharge to the root zone)",
                    fontweight="bold", fontsize=11.5, loc="left")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].scatter(x, [np.clip(fan.get(c), .05, None) for c in cols], marker="o",
                  s=52, color="#8856a7", edgecolor="#222",
                  label="Fan (2013) equilibrium water-table depth", zorder=4)
    ax[1].scatter(x, [zi[c] for c in cols], marker="s", s=40, color="#d95f0e",
                  label="warm start — initial model state", zorder=3)
    ax[1].scatter(x, [ze[c] for c in cols], marker="x", s=56, color="#08519c",
                  label="warm start — end of year", zorder=5)
    ax[1].axhline(8.802, color="#777", ls="--", lw=1.2,
                  label="cold-start initial state (uniform)")
    ax[1].set_yscale("log")
    ax[1].invert_yaxis()
    ax[1].set_ylabel("water-table depth (m, log)")
    ax[1].set_title("(b) Water-table state: initialization and end-of-year "
                    "position relative to the Fan prior",
                    fontweight="bold", fontsize=11.5, loc="left")
    ax[1].legend(frameon=False, fontsize=8.5, ncol=2)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=7)
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cold start vs Fan-informed warm start — identical columns, "
                 "forcing, and year; only the initial water table differs",
                 fontweight="bold")
    fig.tight_layout()
    out = Path(args.warm, "04_analysis", "cold_vs_warm")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"✓ {out}.png (300 dpi) + .pdf")


if __name__ == "__main__":
    main()
