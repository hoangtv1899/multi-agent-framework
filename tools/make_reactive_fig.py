#!/usr/bin/env python3
"""
Figure: reactive-transport proof of concept on a real Gunnison column.

Organic-matter degradation front depth as a function of recharge rate, from
the LAMBDA reaction sandbox running on col_01 (26 m Fan water table, SSURGO
layering, 31 m domain). Demonstration configuration, not a calibration.

    python3 tools/make_reactive_fig.py
"""
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "workflow_outputs" / "gunnison_reactive"
INK, MUT = "#111827", "#475569"
STYLE = {"r100": ("#2563EB", "100 mm/yr recharge"),
         "r10": ("#94A3B8", "10 mm/yr recharge")}
CH2O_0 = 110.0


def read(p):
    ls = Path(p).read_text().splitlines()
    names = re.findall(r'"([^"]+)"', ls[1])
    rows = []
    for l in ls[2:]:
        s = l.split()
        if len(s) == len(names):
            try:
                rows.append([float(x) for x in s])
            except ValueError:
                pass
    return names, np.array(rows)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for rate, (color, label) in STYLE.items():
        fs = sorted(glob.glob(str(RUN / rate / "*.tec")))
        names, a = read(fs[-1])
        z = a[:, names.index("Z [m]")]
        depth = z.max() - z
        ch = a[:, names.index("Total CH2O(s) [M]")]
        ph = a[:, names.index("pH")]
        o = np.argsort(depth)
        ax[0].plot(ch[o], depth[o], "o-", color=color, lw=1.8, ms=3.5,
                   label=label, zorder=3)
        ax[1].plot(ph[o], depth[o], "o-", color=color, lw=1.8, ms=3.5,
                   label=label, zorder=3)
        frac = 100 * (1 - ch.mean() / CH2O_0)
        print(f"{rate}: {frac:.1f}% of column organic carbon consumed")

    ax[0].axvline(CH2O_0, ls="--", color=MUT, lw=1.1, zorder=2)
    ax[0].text(CH2O_0 - 3, 28, "initial", rotation=90, fontsize=7.5,
               color=MUT, ha="right", va="center")
    ax[0].set_xlabel("organic carbon CH2O(s)  [M]", fontsize=9)
    ax[0].set_title("(a) organic-matter degradation front", fontsize=10,
                    fontweight="bold", loc="left")
    ax[1].set_xlabel("pH", fontsize=9)
    ax[1].set_title("(b) geochemical signature", fontsize=10,
                    fontweight="bold", loc="left")
    for a_ in ax:
        a_.invert_yaxis()
        a_.set_ylabel("depth below surface  [m]", fontsize=9)
        a_.grid(alpha=.25, zorder=0)
        a_.tick_params(labelsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Reactive transport on a sampled Upper Gunnison column "
                 "(col_01, 26 m water table): recharge sets the depth "
                 "organic-matter respiration reaches",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "docs" / "paper" / "fig_reactive_demo"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"OK {out}.png + .pdf")


if __name__ == "__main__":
    main()
