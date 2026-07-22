#!/usr/bin/env python3
"""Daily SWE seasonal cycle (model column range) vs observed SNOTEL peaks, 1995 Naches.

Model daily H2OSNO from the warm-start run's h0 history (still on scratch);
observed peak SWE and peak date per SNOTEL station from the committed
validation record. Shows accumulation and melt TIMING, not just the peak.
"""
import glob, json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "workflow_outputs/naches_warmstart"
VAL = ROOT / "workflow_outputs/naches_nldas_20col/04_analysis/validation.json"
MOD, OBS = "#2c7fb8", "#b91c1c"


def model_swe():
    cases = json.loads((RUN / "cases.json").read_text())
    series = []
    for c in cases:
        f = [x for x in sorted(glob.glob(f"{c}/run/*.elm.h0.*.nc"))
             if ".h0.1995-01-01" in x]
        if not f:
            continue
        d = Dataset(f[0])
        series.append(np.asarray(d.variables["H2OSNO"][:365]).reshape(-1))
    a = np.array(series)                       # (cols, 365)
    day = np.arange(1, a.shape[1] + 1)
    return day, a.min(0), a.mean(0), a.max(0), a.shape[0]


def main():
    day, lo, mean, hi, ncol = model_swe()
    obs = json.loads(VAL.read_text())["swe_context"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.fill_between(day, lo, hi, color=MOD, alpha=.20,
                    label=f"model SWE range ({ncol} columns)")
    ax.plot(day, mean, color=MOD, lw=1.8, label="model SWE (column mean)")
    for i, s in enumerate(obs):
        yday = datetime.strptime(s["peak_date"], "%Y-%m-%d").timetuple().tm_yday
        ax.scatter(yday, s["peak_swe_mm"], s=55, marker="v", color=OBS,
                   edgecolor="#222", zorder=4,
                   label="observed SNOTEL peak" if i == 0 else None)
        ax.annotate(s["name"], (yday, s["peak_swe_mm"]), fontsize=6.5,
                    color="#374151", xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("day of 1995"); ax.set_ylabel("snow water equivalent (mm)")
    ax.set_title("Snow water equivalent: model seasonal cycle vs observed "
                 "SNOTEL peaks (Naches, 1995)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(.02, .74, "high-elevation SNOTEL peaks exceed the model range:\n"
                      "the 12 km forcing cell averages over ridgetop terrain",
            transform=ax.transAxes, fontsize=7.6, color="#475569")
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "paper" / "fig_swe_validation"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"OK {out}.png  ({ncol} model columns, {len(obs)} SNOTEL stations)")


if __name__ == "__main__":
    main()
