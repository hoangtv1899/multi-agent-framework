#!/usr/bin/env python3
"""Streamflow comparison figure (an honest limitation, not a validation success)."""
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "workflow_outputs/naches_nldas_20col/04_analysis/validation.json"
OBS, MOD, MUT = "#334155", "#d95f0e", "#475569"


def main():
    h = json.loads(SRC.read_text())["hydrograph"]
    d = [datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday for x in h["days"]]
    obs, mod = np.array(h["obs"]), np.array(h["mod"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(d, obs, color=OBS, lw=1.3, label="observed specific discharge")
    ax[0].plot(d, mod, color=MOD, lw=1.5, label="modeled water yield (column mean)")
    ax[0].set_xlabel("day of 1995"); ax[0].set_ylabel("flux (mm/day)")
    ax[0].set_title("(a) magnitude: routed discharge vs local generation",
                    fontsize=10, fontweight="bold", loc="left")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].text(.03, .80, f"model peak {mod.max():.1f} vs observed {obs.max():.0f} mm/day\n"
                         f"bias ratio {h['beta_bias_ratio']:.2f}: 1-D columns carry no\n"
                         "routing or upstream contribution",
               transform=ax[0].transAxes, fontsize=7.6, color=MUT)
    on, mn = obs / obs.max(), mod / max(mod.max(), 1e-9)
    ax[1].plot(d, on, color=OBS, lw=1.3, label="observed (normalized)")
    ax[1].plot(d, mn, color=MOD, lw=1.5, label="modeled (normalized)")
    ax[1].set_xlabel("day of 1995"); ax[1].set_ylabel("fraction of annual max")
    ax[1].set_title("(b) timing: partial, snowmelt onset only",
                    fontsize=10, fontweight="bold", loc="left")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    ax[1].text(.03, .80, f"NSE {h['NSE']:.2f}   KGE {h['KGE']:.2f}   r {h['r']:.2f}\n"
                         "some melt-onset correspondence,\n"
                         "no routed recession",
               transform=ax[1].transAxes, fontsize=7.6, color=MUT)
    for a in ax:
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Streamflow at {h['gauge']} ({h['gauge_id']}, "
                 f"{h['drainage_mi2']:.0f} mi2): a structural limit of 1-D columns, "
                 "stated not hidden", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "docs" / "paper" / "fig_discharge_validation"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"OK {out}.png  (NSE {h['NSE']}, KGE {h['KGE']}, bias {h['beta_bias_ratio']})")


if __name__ == "__main__":
    main()
