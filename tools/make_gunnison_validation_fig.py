#!/usr/bin/env python3
"""Observation validation for the Upper Gunnison warm run (clean, em-dash-free).

Two panels: water-table depth (model vs Fan prior vs USGS wells) and peak SWE
(model column range vs SNOTEL). Streamflow is omitted: no in-domain gauge had
daily records for the simulation years, which is stated in the text.
"""
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "workflow_outputs/pipeline_20260716_200457_warmconus3yr/04_analysis/validation.json"
MOD, FAN, OBS = "#2c7fb8", "#7b3fa0", "#2ca25f"


def main():
    v = json.loads(V.read_text())
    w = v["wtd_comparison"]
    mz = np.array(w["model_zwt_m"]); fan = np.array(w["fan_at_columns_m"])
    wells = np.array(w["observed_wells_m"])
    sw = v["swe_context"]
    mp = v.get("model_peak_swe_mm", [])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) water table
    def strip(a, x, c, lab):
        ax[0].scatter(np.full(len(a), x) + np.random.RandomState(1).uniform(-.06, .06, len(a)),
                      a, s=48, color=c, edgecolor="#222", zorder=3, label=lab)
        ax[0].plot([x - .18, x + .18], [np.median(a)] * 2, color=c, lw=2.4, zorder=4)
    strip(mz, 0, MOD, "model (ELM warm)")
    strip(fan, 1, FAN, "Fan (2013) at columns")
    strip(wells, 2, OBS, "USGS wells (observed)")
    ax[0].set_yscale("log"); ax[0].invert_yaxis()
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["model\n(ELM warm)", "Fan 2013\n(at columns)",
                           "USGS wells\n(observed)"], fontsize=8.5)
    ax[0].set_ylabel("water-table depth (m, log)", fontsize=9.5)
    ax[0].set_title("(a) water table: model vs Fan vs wells", fontsize=10,
                    fontweight="bold", loc="left")
    ax[0].text(.03, .05, f"medians: model {np.median(mz):.0f} m, Fan "
                         f"{np.median(fan):.0f} m, wells {np.median(wells):.0f} m; "
                         f"only {len(wells)} wells had records",
               transform=ax[0].transAxes, fontsize=7.3, color="#475569")

    # (b) peak SWE
    names = [s["name"][:12] for s in sw]
    obs = [s["peak_swe_mm"] for s in sw]
    x = np.arange(len(sw))
    if mp:
        ax[1].fill_between([-.5, len(sw) - .5], min(mp), max(mp), color=MOD,
                           alpha=.18, label=f"model range ({len(mp)} cols)")
    ax[1].bar(x, obs, width=.6, color=FAN, alpha=.85, label="SNOTEL observed")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7.5, rotation=25, ha="right")
    ax[1].set_ylabel("peak SWE (mm)", fontsize=9.5)
    ax[1].set_title("(b) peak SWE: model range vs SNOTEL", fontsize=10,
                    fontweight="bold", loc="left")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")

    for a in ax:
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Observation validation, Upper Gunnison warm run "
                 "(no in-domain daily gauge, so streamflow is not shown)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "docs" / "paper" / "fig_gunnison_validation"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"OK {out}.png  (WTD {len(mz)} cols / {len(wells)} wells, SWE {len(sw)} stations)")


if __name__ == "__main__":
    main()
