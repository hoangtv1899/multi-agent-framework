#!/usr/bin/env python3
"""
Evidence: what the ELM->PFLOTRAN weak coupling adds over each standalone model
— built entirely from existing outputs (no new runs).

For every column, the SAME lag/attenuation metrics are computed through two
pathways driven by the SAME infiltration forcing (ELM daily QINFL):

    ELM alone     QINFL -> QCHARGE   recharge at the 3.8 m soil bottom into a
                                     bucket aquifer — no vadose zone to cross
    coupled       QINFL -> flux delivered at the REAL water table
                                     (PFLOTRAN Richards column, coupled run)

If the vadose zone matters, the two pathways must diverge with water-table
depth — and only the coupled one can show it.

    python3 tools/coupling_evidence.py --elm-run workflow_outputs/naches_warmstart \
        --coupled-run workflow_outputs/pflotran_coupled
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "tools")
import numpy as np
from analyze_pflotran_coupled import read_mas, lag_and_attenuation, daily


def elm_daily(elm_run):
    """col id -> dict(day[], qinfl[], qcharge[]) daily means in mm/day."""
    import xarray as xr
    rd = Path(elm_run)
    cf = next((rd / n for n in ("cases_all.json", "cases.json") if (rd / n).exists()))
    out = {}
    for c in json.loads(cf.read_text()):
        name = c.split(".")[-1]
        fs = sorted(Path(c, "run").glob("*.elm.h0.*.nc"))
        if not fs:
            continue
        ds = xr.open_mfdataset([str(f) for f in fs], combine="by_coords",
                               decode_times=True, engine="netcdf4", data_vars="all",
                               coords="different", compat="no_conflicts", join="outer")
        qi = np.ravel(ds["QINFL"].values) * 86400.0        # mm/s -> mm/d
        qc = np.ravel(ds["QCHARGE"].values) * 86400.0
        doy = np.asarray(ds["time"].dt.dayofyear.values)[:len(qi)]
        ds.close()
        days = sorted(set(int(d) for d in doy))[:365]
        out[name] = {
            "day": np.array(days, float),
            "qinfl": np.array([qi[doy == d].mean() for d in days]),
            "qcharge": np.array([qc[doy == d].mean() for d in days]),
        }
    return out


def series_lag(F, B):
    """Same cross-correlation lag/attenuation as the coupled analyzer, on
    already-daily arrays."""
    t = np.arange(len(F)) / 365.0
    return lag_and_attenuation(t, F * 365.0, -B * 365.0, 0.0, len(F) / 365.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elm-run", default="workflow_outputs/naches_warmstart")
    ap.add_argument("--coupled-run", default="workflow_outputs/pflotran_coupled")
    ap.add_argument("--example-col", default="col_10")
    args = ap.parse_args()

    cr = Path(args.coupled_run)
    meta = json.loads((cr / "pflotran_cases.json").read_text())
    spin = meta["scenario"].get("spin_years") or 10.0
    fan = {m["id"]: m["fan_wtd_m"] for m in meta["cases"]}
    ok_ids = [m["id"] for m in meta["cases"] if m.get("run_ok") is not False]

    print("reading ELM daily QINFL/QCHARGE (20 columns) ...")
    elm = elm_daily(args.elm_run)

    rows = []
    ts_example = None
    for cid in ok_ids:
        if cid not in elm:
            continue
        e = elm[cid]
        elm_lag, elm_att = series_lag(e["qinfl"], -e["qcharge"])
        mas = read_mas(next(m["case_dir"] for m in meta["cases"] if m["id"] == cid), cid)
        if not mas:
            continue
        t, top, bot = mas
        pf_lag, pf_att = lag_and_attenuation(t, top, bot, spin, spin + 1.0)
        rows.append({"id": cid, "fan_wtd_m": fan[cid],
                     "elm_lag_d": elm_lag, "elm_att": elm_att,
                     "coupled_lag_d": pf_lag, "coupled_att": pf_att})
        if cid == args.example_col:
            g, B = daily(t, -bot, spin, spin + 1.0)
            ts_example = (cid, fan[cid], e["day"], e["qinfl"], e["qcharge"],
                          B / 365.0)                        # mm/yr -> mm/d

    print(f"\n{'column':<8}{'Fan_WTD':>8} | {'ELM lag':>8}{'ELM att':>9} | "
          f"{'CPL lag':>8}{'CPL att':>9}")
    print("-" * 58)
    fmt = lambda x: f"{x}" if x is not None else "—"
    for r in sorted(rows, key=lambda r: r["fan_wtd_m"]):
        print(f"{r['id']:<8}{r['fan_wtd_m']:>8.1f} | {fmt(r['elm_lag_d']):>8}"
              f"{r['elm_att']:>9.3f} | {fmt(r['coupled_lag_d']):>8}{r['coupled_att']:>9.3f}")

    def _corr(key):
        w = np.array([np.log10(r["fan_wtd_m"]) for r in rows])
        y = np.array([r[key] for r in rows])
        return float(np.corrcoef(w, y)[0, 1]) if len(rows) > 2 else float("nan")
    r_cpl_att, r_elm_att = _corr("coupled_att"), _corr("elm_att")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

    if ts_example:
        cid, wtd, day, qi, qc, pf = ts_example
        ax[0].plot(day, qi, color="#94a3b8", lw=.9, label="infiltration (shared forcing)")
        ax[0].plot(day, qc, color="#2c7fb8", lw=1.6,
                   label="ELM recharge (QCHARGE @3.8 m — bucket)")
        ax[0].plot(np.arange(len(pf)) + 1, pf, color="#d95f0e", lw=1.8,
                   label=f"coupled: delivered at the water table ({wtd:.0f} m)")
        ax[0].set_xlabel("day of year"); ax[0].set_ylabel("flux (mm/day)")
        ax[0].set_title(f"{cid} — same forcing, two recharge stories",
                        fontweight="bold", fontsize=10.5)
        ax[0].legend(frameon=False, fontsize=7.5)

    okE = [r for r in rows if r["elm_lag_d"] is not None]
    okP = [r for r in rows if r["coupled_lag_d"] is not None]
    ax[1].scatter([r["fan_wtd_m"] for r in okE], [r["elm_lag_d"] for r in okE],
                  s=46, color="#2c7fb8", edgecolor="#222", label="ELM alone (bucket)")
    ax[1].scatter([r["fan_wtd_m"] for r in okP], [r["coupled_lag_d"] for r in okP],
                  s=52, marker="D", color="#d95f0e", edgecolor="#222",
                  label="coupled (real water table)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Fan water-table depth (m)"); ax[1].set_ylabel("lag (days)")
    ax[1].set_title("Infiltration → recharge lag", fontweight="bold", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=8)
    n_none = sum(1 for r in rows if r["elm_lag_d"] is None)
    ax[1].text(.03, .60, f"ELM alone: no coherent lag detectable\n"
                         f"in {n_none}/{len(rows)} columns (r≤0.2) — the bucket's\n"
                         "recharge is not a delayed copy of the\n"
                         "forcing; travel time is UNDEFINED",
               transform=ax[1].transAxes, fontsize=8, color="#2c7fb8")

    ax[2].scatter([r["fan_wtd_m"] for r in rows], [r["elm_att"] for r in rows],
                  s=46, color="#2c7fb8", edgecolor="#222", label="ELM alone")
    ax[2].scatter([r["fan_wtd_m"] for r in rows], [r["coupled_att"] for r in rows],
                  s=52, marker="D", color="#d95f0e", edgecolor="#222", label="coupled")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("Fan water-table depth (m)")
    ax[2].set_ylabel("attenuation (std ratio)")
    ax[2].set_title("Signal surviving to the 'recharge' point",
                    fontweight="bold", fontsize=10.5)
    ax[2].legend(frameon=False, fontsize=8, loc="lower left")
    ax[2].text(.03, .05, f"coupled: decay with depth (r={r_cpl_att:+.2f}); soil texture adds scatter\n"
                         f"ELM: no relation to WTD (r={r_elm_att:+.2f}) — recharge fixed at 3.8 m",
               transform=ax[2].transAxes, fontsize=8, color="#374151")
    for a in ax:
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Why couple? ELM's recharge point is 3.8 m (no vadose zone to cross) — "
                 "the coupled system carries the same water to the REAL water table",
                 fontweight="bold")
    fig.tight_layout()
    out = cr / "coupling_evidence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    (cr / "coupling_evidence.json").write_text(json.dumps(rows, indent=2))
    print(f"\n   ✓ {out}")


if __name__ == "__main__":
    main()
