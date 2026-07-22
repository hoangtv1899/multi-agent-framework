#!/usr/bin/env python3
"""Standalone PFLOTRAN equilibrium water table vs the Fan (2013) prior, Gunnison.

Honest framing: Fan is BOTH the initial condition and the bottom boundary of the
1-D column, so in-domain agreement is a consistency/equilibration check, NOT an
independent validation. The informative content is the departure at the deep
columns where the 50 m domain cap prevents the column from holding the Fan
water table.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "workflow_outputs/gunnison_pflotran/r100/pflotran_summary.json"
IN, CAP = "#2c7fb8", "#d95f0e"


def main():
    cols = json.loads(SW.read_text())["columns"]
    fan_in, wtd_in, fan_cap, cap_dom = [], [], [], []
    for c in cols:
        fan, dom, wf = c["fan_wtd_m"], c["depth_m"], c.get("wtd_final_m")
        if isinstance(wf, (int, float)):
            fan_in.append(fan); wtd_in.append(wf)
        else:
            fan_cap.append(fan); cap_dom.append(dom)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lim = [0.05, 300]
    ax.plot(lim, lim, "--", color="#94a3b8", lw=1.1, zorder=1, label="1:1 line")
    ax.scatter(fan_in, wtd_in, s=70, color=IN, edgecolor="#222", zorder=3,
               label="in-domain columns (equilibrated)")
    # capped columns pin at the domain bottom; draw them at the cap depth
    ax.scatter(fan_cap, cap_dom, s=90, marker="v", color=CAP, edgecolor="#222",
               zorder=3, label="Fan deeper than 50 m cap (pinned at bottom)")
    for x, y in zip(fan_cap, cap_dom):
        ax.annotate("", xy=(x, y * 1.9), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=CAP, lw=1.1))
    ax.axhline(50, color=CAP, ls=":", lw=1, alpha=.6)
    ax.text(0.07, 53, "50 m domain cap", fontsize=7.5, color=CAP)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Fan (2013) water-table depth (m)", fontsize=9.5)
    ax.set_ylabel("PFLOTRAN equilibrium water-table depth (m)", fontsize=9.5)
    ax.set_title("Standalone PFLOTRAN water table vs the Fan (2013) prior "
                 "(Upper Gunnison)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(.03, .04,
            "Fan is the column's initial condition AND bottom boundary, so\n"
            "in-domain agreement is a consistency check, not an independent\n"
            "validation. The informative result is the departure at the three\n"
            "deepest columns, where the 50 m cap cannot hold the Fan water table.",
            transform=ax.transAxes, fontsize=7.3, color="#374151")
    ax.grid(alpha=.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "paper" / "fig_pflotran_wtd_fan"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"OK {out}.png  ({len(fan_in)} in-domain, {len(fan_cap)} capped)")


if __name__ == "__main__":
    main()
