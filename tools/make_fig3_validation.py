#!/usr/bin/env python3
"""
Consolidated observation-validation figure (paper Fig 3) — cold start vs
Fan warm start side by side, from the STORED validation.json of both studies
(no network). Three comparison families: water-table distributions, water
yield vs gauges, peak SWE vs SNOTEL.

    python3 tools/make_fig3_validation.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLD = Path("workflow_outputs/naches_nldas_20col/04_analysis/validation.json")
WARM = Path("workflow_outputs/naches_warmstart/04_analysis/validation.json")


def wtd_panel(ax, val, title):
    wtd = val["wtd_comparison"]
    groups = [("model", wtd["model_zwt_m"], "#2c7fb8"),
              ("Fan 2013", wtd["fan_at_columns_m"], "#8856a7"),
              ("USGS wells", wtd["observed_wells_m"], "#31a354")]
    rng = np.random.default_rng(0)
    for i, (lab, vals, c) in enumerate(groups):
        v = np.array([x for x in vals if x is not None and x > 0], float)
        if not len(v):
            continue
        ax.scatter(np.full(len(v), i) + rng.uniform(-.12, .12, len(v)), v,
                   s=34, color=c, edgecolor="#222", zorder=3, alpha=.85)
        ax.hlines(np.median(v), i - .25, i + .25, color=c, lw=2.6, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", fontsize=10)


def swe_panel(ax, val, title):
    swe = val["swe_context"] or []
    mswe = val.get("model_peak_swe_mm") or []
    names = [s["name"][:12] for s in swe]
    ax.bar(range(len(swe)), [s["peak_swe_mm"] for s in swe], color="#756bb1",
           edgecolor="#222", label="SNOTEL observed")
    if mswe:
        ax.axhspan(min(mswe), max(mswe), color="#2c7fb8", alpha=.3,
                   label="model columns (range)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(title, fontweight="bold", fontsize=10)


def main():
    cold, warm = json.loads(COLD.read_text()), json.loads(WARM.read_text())
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for row, (val, tag) in enumerate([(cold, "cold start"), (warm, "Fan warm start")]):
        wtd_panel(ax[row][0], val, f"({'ad'[row]}) water table — {tag}")
        # yield panel
        a = ax[row][1]
        sf = val["streamflow_comparison"]
        vals = [sf["modeled_yield_mm_yr"]] + [g["specific_discharge_mm_yr"]
                                              for g in sf["gauges"]]
        labels = ["model"] + [g["name"][:14] for g in sf["gauges"]]
        a.bar(range(len(vals)), [v or 0 for v in vals],
              color=["#2c7fb8"] + ["#d95f0e"] * (len(vals) - 1), edgecolor="#222")
        for i, v in enumerate(vals):
            if v:
                a.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=8)
        if not sf["gauges"]:
            a.text(.5, .82, "no in-domain gauge with\ndaily records this year\n(context-only)",
                   transform=a.transAxes, ha="center", fontsize=8,
                   color="#334155", style="italic",
                   bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        a.set_title(f"({'be'[row]}) water yield (mm/yr) — {tag}",
                    fontweight="bold", fontsize=10)
        swe_panel(ax[row][2], val, f"({'cf'[row]}) peak SWE (mm) — {tag}")
    for a in ax.ravel():
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[0][0].set_ylabel("WTD (m, log)"); ax[1][0].set_ylabel("WTD (m, log)")
    fig.suptitle("Observation validation — cold start (top) vs Fan-informed "
                 "warm start (bottom), identical columns/forcing/year",
                 fontweight="bold")
    fig.tight_layout()
    out = Path("docs/paper/fig3_validation")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"✓ {out}.png + .pdf")


if __name__ == "__main__":
    main()
