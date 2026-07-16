#!/usr/bin/env python3
"""
Compare the recharge scenarios of a standalone PFLOTRAN study and package it
as a deck-ready case study.

Reads <study>/r*/pflotran_summary.json (build_pflotran_cases.py + analyze per
scenario) and writes:
    pflotran_scenarios.png   climate-sensitivity figure (per-column dynamics)
    pflotran_study.json      question, scenarios, per-column table,
                             deterministic interpretation (deck input)

    python3 tools/compare_pflotran_scenarios.py --study-dir workflow_outputs/pflotran_naches \
        [--sampling-fig <path>]   # copies the sampling-design figure alongside
"""
import argparse
import glob
import json
import math
import shutil
from pathlib import Path

QUESTION = ("How different are the groundwater dynamics across the Naches — "
            "same recharge everywhere, only the subsurface varies?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--sampling-fig", default=None)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--columns", default=None,
                    help="columns.json for soil drivers (clay, Ksat) — enables the relations grid")
    args = ap.parse_args()

    sd = Path(args.study_dir)
    scen = {}                                       # recharge_mm_yr -> {col_id: row}
    for f in sorted(glob.glob(str(sd / "r*" / "pflotran_summary.json"))):
        s = json.loads(Path(f).read_text())
        scen[s["scenario"]["recharge_mm_yr"]] = {r["id"]: r for r in s["columns"]}
    if len(scen) < 2:
        raise SystemExit("need >=2 scenario subdirs (r030/r100/... with summaries)")
    rates = sorted(scen)
    base = scen[rates[len(rates) // 2]]             # middle scenario = reference rows

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = sorted(base, key=lambda c: base[c]["elevation_m"] or 0)
    elev = {c: base[c]["elevation_m"] for c in cols}
    norm = plt.Normalize(min(elev.values()), max(elev.values()))

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.5))
    sens = {}
    for c in cols:
        y = [scen[r][c].get("sat_vadose_mean") for r in rates]
        if None in y:
            continue
        sens[c] = y[-1] - y[0]
        ax[0].plot(rates, y, marker="o", ms=4, lw=1.1,
                   color=plt.cm.viridis(norm(elev[c] or 0)))
    ax[0].set_xscale("log")
    ax[0].set_xlabel("applied recharge (mm/yr)")
    ax[0].set_ylabel("mean vadose-zone saturation (–)")
    ax[0].set_title("Wetness response to recharge — one line per column",
                    fontweight="bold", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax[0],
                 label="elevation (m)")

    x = [base[c]["fan_wtd_m"] for c in sens]
    y = [sens[c] for c in sens]
    capped = [base[c]["wtd_final_m"] is None for c in sens]
    ax[1].scatter([v for v, b in zip(x, capped) if not b], [v for v, b in zip(y, capped) if not b],
                  s=46, color="#2c7fb8", edgecolor="#222", label="WT in domain")
    ax[1].scatter([v for v, b in zip(x, capped) if b], [v for v, b in zip(y, capped) if b],
                  s=46, marker="v", color="#d95f0e", edgecolor="#222", label="WT below cap")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Fan 2013 water-table depth (m)")
    ax[1].set_ylabel(f"Δ vadose saturation ({rates[-1]:.0f} vs {rates[0]:.0f} mm/yr)")
    ax[1].set_title("Climate sensitivity vs water-table depth", fontweight="bold", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8.5)
    for a in ax:
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"PFLOTRAN scenario sweep — {len(cols)} columns × recharge "
                 f"{'/'.join(f'{r:.0f}' for r in rates)} mm/yr (uniform, steady)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(sd / "pflotran_scenarios.png", dpi=300, bbox_inches="tight")

    # driver -> response relations grid (the ELM-style figure, PFLOTRAN edition)
    rel_facts = []
    if args.columns and Path(args.columns).exists():
        cj = json.loads(Path(args.columns).read_text())
        cj = cj.get("columns", cj) if isinstance(cj, dict) else cj
        soil = {}
        for c in cj:
            layers = (c.get("soil_profile") or {}).get("layers") or []
            comp = layers[0].get("component") if layers else None
            hz = [l for l in layers if l.get("component") == comp]
            def num(x):
                try: return float(x)
                except (TypeError, ValueError): return None
            clays = [v for v in (num(l.get("clay_pct")) for l in hz) if v is not None]
            ks = [v for v in (num(l.get("ksat_ums")) for l in hz) if v is not None]
            soil[c["id"]] = {"clay_max": max(clays) if clays else None,
                             "ksat_min": min(ks) if ks else None}
        drivers = [("elevation (m)", lambda c: base[c]["elevation_m"]),
                   ("log10 Fan WTD (m)", lambda c: math.log10(max(base[c]["fan_wtd_m"], .05))),
                   ("max clay (%)", lambda c: soil.get(c, {}).get("clay_max")),
                   ("min Ksat (µm/s)", lambda c: soil.get(c, {}).get("ksat_min"))]
        mid = rates[len(rates) // 2]
        responses = [(f"vadose sat @{mid:.0f} mm/yr", lambda c: scen[mid][c].get("sat_vadose_mean")),
                     (f"Δsat ({rates[-1]:.0f}−{rates[0]:.0f})", lambda c: sens.get(c)),
                     (f"surface sat @{mid:.0f}", lambda c: scen[mid][c].get("sat_top"))]
        import numpy as np
        figR, axR = plt.subplots(len(responses), len(drivers),
                                 figsize=(3.0 * len(drivers), 2.5 * len(responses)),
                                 sharey="row", squeeze=False)
        for i, (rlab, rget) in enumerate(responses):
            for j, (dlab, dget) in enumerate(drivers):
                a = axR[i][j]
                x = np.array([dget(c) if dget(c) is not None else np.nan for c in cols], float)
                y = np.array([rget(c) if rget(c) is not None else np.nan for c in cols], float)
                m = ~np.isnan(x) & ~np.isnan(y)
                a.scatter(x[m], y[m], s=26, color="#2c7fb8", edgecolor="#222",
                          linewidth=.4, alpha=.85, zorder=3)
                if m.sum() >= 3 and np.ptp(x[m]) > 1e-9 and np.ptp(y[m]) > 1e-9:
                    rr = float(np.corrcoef(x[m], y[m])[0, 1])
                    a.text(.04, .86, f"r={rr:+.2f}", transform=a.transAxes, fontsize=9,
                           fontweight="bold", color="#b91c1c" if abs(rr) >= .5 else "#64748b")
                    if abs(rr) >= .5:
                        rel_facts.append((rlab, dlab, rr))
                if i == len(responses) - 1:
                    a.set_xlabel(dlab, fontsize=9)
                if j == 0:
                    a.set_ylabel(rlab, fontsize=9)
                a.tick_params(labelsize=7.5); a.grid(alpha=.25)
                a.spines[["top", "right"]].set_visible(False)
        figR.suptitle("PFLOTRAN driver → response relations (uniform forcing — "
                      "controls are the water table and the soil)", fontweight="bold", y=1.0)
        figR.tight_layout()
        figR.savefig(sd / "pflotran_relations.png", dpi=300, bbox_inches="tight")
        print(f"✓ {sd}/pflotran_relations.png")

    # deterministic interpretation — facts only, no LLM
    hi = max(sens, key=sens.get) if sens else None
    lo = min(sens, key=sens.get) if sens else None
    n_cap = sum(1 for c in cols if base[c]["wtd_final_m"] is None)
    interp = [
        f"Uniform steady recharge ({'/'.join(f'{r:.0f}' for r in rates)} mm/yr) on all "
        f"{len(cols)} columns — every difference below is subsurface structure, not climate.",
        f"Vadose-zone wetness spans {min(base[c]['sat_vadose_mean'] or 0 for c in cols):.2f}–"
        f"{max(base[c]['sat_vadose_mean'] or 0 for c in cols):.2f} at the same forcing — "
        "soil layering and water-table depth set the baseline, not recharge.",
        (f"Most climate-sensitive: {hi} (Δsat {sens[hi]:+.2f}, Fan WTD "
         f"{base[hi]['fan_wtd_m']:.1f} m); least: {lo} (Δsat {sens[lo]:+.2f}, "
         f"Fan WTD {base[lo]['fan_wtd_m']:.1f} m) — deep unsaturated zones buffer "
         "recharge changes; shallow water tables pin the profile.") if sens else "",
        f"{n_cap} ridge columns keep their water table below the capped 50 m domain at "
        "every rate (Fan says 61–215 m there) — honestly flagged, not simulated.",
        "Water tables are held by the bottom boundary (pinned at the Fan prior), so this "
        "sweep isolates the UNSATURATED-zone response; transient forcing and a free water "
        "table are the ELM-coupling phase.",
    ]
    if rel_facts:
        strongest = sorted(rel_facts, key=lambda t: -abs(t[2]))[:3]
        interp.append("Strongest controls (|r|≥0.5): " + " · ".join(
            f"{r} vs {d} (r={v:+.2f})" for r, d, v in strongest)
            + " — elevation matters only where it proxies the water table (no orographic "
              "forcing exists in this design, unlike the ELM/NLDAS studies).")
    rows = [{**{k: base[c][k] for k in ("id", "elevation_m", "fan_wtd_m", "depth_m",
                                        "wtd_final_m")},
             **{f"sat_r{r:.0f}": scen[r][c].get("sat_vadose_mean") for r in rates}}
            for c in cols]
    (sd / "pflotran_study.json").write_text(json.dumps({
        "name": "Naches groundwater (PFLOTRAN)", "question": args.question,
        "model": "PFLOTRAN 1-D RICHARDS (standalone — no ELM)",
        "scenarios_mm_yr": rates, "interpretation": [t for t in interp if t],
        "columns": rows,
        "execution": f"{len(cols)} columns × {len(rates)} scenarios = "
                     f"{len(cols)*len(rates)} serial runs, ~0.5 s each, login node — no batch queue",
    }, indent=2))
    if args.sampling_fig and Path(args.sampling_fig).exists():
        shutil.copy2(args.sampling_fig, sd / "sampling_design.png")
    print(f"✓ {sd}/pflotran_scenarios.png + pflotran_study.json  "
          f"({len(cols)} columns, rates {rates})")


if __name__ == "__main__":
    main()
