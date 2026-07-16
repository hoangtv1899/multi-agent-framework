#!/usr/bin/env python3
"""
Combined cross-model result (paper Fig 5): the vadose zone as a low-pass
filter whose cutoff is set by water-table depth. From STORED summaries only:

  (a) standalone PFLOTRAN recharge sweep — climate sensitivity of vadose
      wetness vs the Fan water-table depth
  (b) coupled ELM->PFLOTRAN — infiltration-to-water-table lag vs depth
  (c) coupled — attenuation (signal surviving) vs depth

    python3 tools/make_fig5_lowpass.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = "workflow_outputs/pflotran_naches"
COUPLED = "workflow_outputs/pflotran_coupled/pflotran_summary_coupled.json"


def main():
    scen = {}
    for f in sorted(glob.glob(f"{SWEEP}/r*/pflotran_summary.json")):
        s = json.loads(Path(f).read_text())
        scen[s["scenario"]["recharge_mm_yr"]] = {r["id"]: r for r in s["columns"]}
    rates = sorted(scen)
    base = scen[rates[len(rates) // 2]]
    sens = {c: scen[rates[-1]][c]["sat_vadose_mean"] - scen[rates[0]][c]["sat_vadose_mean"]
            for c in base
            if all(scen[r][c].get("sat_vadose_mean") is not None for r in rates)}

    cpl = json.loads(Path(COUPLED).read_text())["columns"]

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return float(np.corrcoef(x, y)[0, 1])

    fig, ax = plt.subplots(1, 3, figsize=(11.8, 3.9))

    # (a) sweep sensitivity
    w = [base[c]["fan_wtd_m"] for c in sens]
    y = list(sens.values())
    capped = [base[c]["wtd_final_m"] is None for c in sens]
    ax[0].scatter([v for v, b in zip(w, capped) if not b],
                  [v for v, b in zip(y, capped) if not b],
                  s=42, color="#2c7fb8", edgecolor="#222", label="WT in domain")
    ax[0].scatter([v for v, b in zip(w, capped) if b],
                  [v for v, b in zip(y, capped) if b],
                  s=42, marker="v", color="#d95f0e", edgecolor="#222",
                  label="WT below 50 m cap")
    r_a = corr(np.log10(w), y)
    ax[0].set_ylabel(f"Δ vadose saturation ({rates[-1]:.0f} vs {rates[0]:.0f} mm/yr)")
    ax[0].set_title(f"(a) Standalone sweep — climate sensitivity\n"
                    f"r(Δsat, log₁₀ WTD) = {r_a:+.2f}", fontweight="bold", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7.5)

    # (b) coupled lag
    ok = [r for r in cpl if r["lag_days"] is not None]
    ax[1].scatter([r["fan_wtd_m"] for r in ok], [r["lag_days"] for r in ok],
                  s=42, color="#2c7fb8", edgecolor="#222")
    ax[1].set_ylabel("lag (days)")
    ax[1].set_title("(b) Coupled — infiltration → water-table lag\n"
                    f"{len(ok)} columns with coherent response",
                    fontweight="bold", fontsize=10)

    # (c) coupled attenuation
    w3 = [r["fan_wtd_m"] for r in cpl]
    a3 = [r["attenuation"] for r in cpl]
    r_c = corr(np.log10(w3), a3)
    ax[2].scatter(w3, a3, s=42, color="#31a354", edgecolor="#222")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("attenuation (std ratio)")
    ax[2].set_title(f"(c) Coupled — signal surviving the vadose zone\n"
                    f"r(attenuation, log₁₀ WTD) = {r_c:+.2f}",
                    fontweight="bold", fontsize=10)

    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("Fan (2013) water-table depth (m)")
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The vadose zone as a low-pass filter — cutoff set by "
                 "water-table depth (neither model shows this alone)",
                 fontweight="bold")
    fig.tight_layout()
    out = Path("docs/paper/fig5_lowpass")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"✓ {out}.png + .pdf")


if __name__ == "__main__":
    main()
