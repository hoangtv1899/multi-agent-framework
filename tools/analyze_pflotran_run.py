#!/usr/bin/env python3
"""
Analyze a standalone PFLOTRAN ensemble (build_pflotran_cases.py output).

Reads each column's TECPLOT snapshots (1-D profiles of pressure/saturation),
extracts the water-table depth through time, and writes:
    pflotran_summary.json   per-column WTD initial/final, storage, scenario
    pflotran_profiles.png   saturation profiles + WTD-vs-Fan drift figure

    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/analyze_pflotran_run.py --run-dir workflow_outputs/pflotran_naches
"""
import argparse
import glob
import json
import re
from pathlib import Path

ATM = 101325.0


def read_tec(path):
    """One TECPLOT POINT snapshot -> (time_yr, z[], pressure[], saturation[])."""
    lines = Path(path).read_text().splitlines()
    t = float(re.search(r'"\s*([\d.E+-]+)\s*\[y\]"', lines[0]).group(1))
    z, p, s = [], [], []
    for ln in lines[3:]:
        v = ln.split()
        if len(v) >= 5:
            z.append(float(v[2])); p.append(float(v[3])); s.append(float(v[4]))
    return t, z, p, s


def wtd_from_profile(z, p, height):
    """Water-table depth below the surface = highest elevation where P>=atm,
    linearly interpolated; None if the whole column is unsaturated."""
    zw = None
    for i in range(len(z) - 1):
        if (p[i] - ATM) * (p[i + 1] - ATM) <= 0 and p[i] >= ATM:
            zw = z[i] + (ATM - p[i]) * (z[i + 1] - z[i]) / (p[i + 1] - p[i])
    if zw is None and p and p[-1] >= ATM:            # saturated to the top
        zw = z[-1]
    return None if zw is None else round(height - zw, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plot", action="store_true", default=True)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    meta = json.loads((rd / "pflotran_cases.json").read_text())
    rows, profiles = [], {}
    for m in meta["cases"]:
        tecs = sorted(glob.glob(f"{m['case_dir']}/{m['id']}-*.tec"))
        if not tecs:
            continue
        H = m["depth_m"]
        t0, z0, p0, _ = read_tec(tecs[0])
        tN, zN, pN, sN = read_tec(tecs[-1])
        vad = [sv for sv, pv in zip(sN, pN) if pv < ATM]
        rows.append({**{k: m[k] for k in ("id", "elevation_m", "fan_wtd_m", "depth_m")},
                     "wtd_initial_m": wtd_from_profile(z0, p0, H),
                     "wtd_final_m": wtd_from_profile(zN, pN, H),
                     "sat_top": round(sN[-1], 4),
                     "sat_vadose_mean": round(sum(vad) / len(vad), 4) if vad else None,
                     "years": tN})
        profiles[m["id"]] = (m.get("elevation_m"), [H - zi for zi in zN], sN)

    out = {"scenario": meta["scenario"], "columns": rows}
    (rd / "pflotran_summary.json").write_text(json.dumps(out, indent=2))

    print(f"{'column':<8}{'elev_m':>8}{'Fan_WTD':>9}{'domain':>8}{'WTD_t0':>8}{'WTD_end':>9}")
    print("-" * 52)
    for r in rows:
        f = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else ">bottom"
        print(f"{r['id']:<8}{r['elevation_m']:>8.0f}{r['fan_wtd_m']:>9.1f}"
              f"{r['depth_m']:>8.1f}{f(r['wtd_initial_m']):>8}{f(r['wtd_final_m']):>9}")

    if args.plot and profiles:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
        elevs = [e for e, _, _ in profiles.values() if e is not None]
        norm = plt.Normalize(min(elevs), max(elevs))
        for cid, (e, depth, sat) in profiles.items():
            ax[0].plot(sat, depth, lw=1.1, color=plt.cm.viridis(norm(e or 0)))
        ax[0].invert_yaxis()
        ax[0].set_xlabel("liquid saturation (–)"); ax[0].set_ylabel("depth (m)")
        ax[0].set_title(f"Saturation profiles at t={rows[0]['years']:.0f} y "
                        f"(colour = elevation)", fontweight="bold", fontsize=11)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax[0],
                     label="elevation (m)")
        fan = [r["fan_wtd_m"] for r in rows]
        end = [r["wtd_final_m"] if r["wtd_final_m"] is not None else r["depth_m"]
               for r in rows]
        below = [r["wtd_final_m"] is None for r in rows]
        ax[1].scatter([f for f, b in zip(fan, below) if not b],
                      [e for e, b in zip(end, below) if not b],
                      s=46, color="#2c7fb8", edgecolor="#222", label="WT resolved")
        ax[1].scatter([f for f, b in zip(fan, below) if b],
                      [e for e, b in zip(end, below) if b],
                      s=46, marker="v", color="#d95f0e", edgecolor="#222",
                      label="WT below capped domain")
        lim = [0.1, max(max(fan), max(end)) * 1.3]
        ax[1].plot(lim, lim, ls="--", color="#888", lw=1)
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("Fan 2013 WTD (m)"); ax[1].set_ylabel(f"PFLOTRAN WTD at t_end (m)")
        ax[1].set_title("Simulated water table vs the Fan prior", fontweight="bold", fontsize=11)
        ax[1].legend(frameon=False, fontsize=8.5)
        for a in ax:
            a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
        sc = meta["scenario"]
        fig.suptitle(f"Standalone PFLOTRAN ensemble — recharge {sc['recharge_mm_yr']:.0f} mm/yr, "
                     f"bottom BC: {sc['bottom_bc']}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(rd / "pflotran_profiles.png", dpi=300, bbox_inches="tight")
        print(f"\n   ✓ figure: {rd / 'pflotran_profiles.png'}")


if __name__ == "__main__":
    main()
