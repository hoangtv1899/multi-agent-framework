#!/usr/bin/env python3
"""
Per-column DEBUG plots — visualise what was actually BUILT and RUN, to catch
problems at the two steps that otherwise have no picture.

  --surfaces    read each case's generated FSURDAT and plot its soil profile
                (PCT_CLAY / PCT_SAND vs depth) — verifies every column got a
                DISTINCT, correct soil (catches surface-file collisions etc.).
                Run this AFTER build_cases.py, BEFORE the salloc run.

  --timeseries  read each case's history files and plot the daily water-balance
                fluxes (QCHARGE recharge, QOVER runoff) over the run — verifies
                the run produced sensible dynamics. Run this AFTER the salloc run.

Reads cases.json in the run dir (written by build_cases.py). NOTHING is executed.

    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/plot_columns.py --run-dir <dir> --cases-file cases.json --surfaces
    python3 tools/plot_columns.py --run-dir <dir> --cases-file cases.json --timeseries
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")


def _fsurdat(case_dir):
    try:
        t = (Path(case_dir) / "run" / "lnd_in").read_text()
        m = re.search(r"fsurdat\s*=\s*'([^']+)'", t)
        return m.group(1) if m else None
    except OSError:
        return None


def _colors(n):
    import matplotlib.pyplot as plt
    import numpy as np
    return plt.cm.viridis(np.linspace(0, 1, n))


def plot_surfaces(cases, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from core.elm_surface_generator import ELM_LEVEL_NODE_DEPTH_M

    depth = np.array(ELM_LEVEL_NODE_DEPTH_M)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 5.2))
    mismatches = 0
    for c, color in zip(cases, _colors(len(cases))):
        name = c.split(".")[-1]
        fs = _fsurdat(c)
        if not fs or not Path(fs).exists():
            print(f"  ! {name}: surface file not found")
            continue
        d = xr.open_dataset(fs, decode_times=False)

        # coordinate consistency: the surface's lat/lon must match the case's
        # domain file, or ELM aborts at init (surfdata/fatmgrid mismatch)
        try:
            dom = re.search(r"fatmlndfrc\s*=\s*'([^']+)'",
                            (Path(c) / "run" / "lnd_in").read_text())
            if dom and Path(dom.group(1)).exists():
                dd = xr.open_dataset(dom.group(1), decode_times=False)
                slat = float(np.asarray(d["LATIXY"].values).flat[0])
                dlat = float(np.asarray(dd["yc"].values).flat[0])
                slon = float(np.asarray(d["LONGXY"].values).flat[0]) % 360
                dlon = float(np.asarray(dd["xc"].values).flat[0]) % 360
                dd.close()
                if abs(slat - dlat) > 0.01 or abs(slon - dlon) > 0.01:
                    mismatches += 1
                    print(f"  ✗ {name}: SURFACE/DOMAIN COORD MISMATCH — "
                          f"surface ({slat:.2f},{slon:.2f}) vs domain ({dlat:.2f},{dlon:.2f}) "
                          f"— this column WILL abort at init")
        except Exception:
            pass

        def prof(v):
            if v not in d:
                return None
            a = np.asarray(d[v].values)
            return a.reshape(a.shape[0], -1)[:len(depth), 0]
        clay, sand = prof("PCT_CLAY"), prof("PCT_SAND")
        d.close()
        if clay is not None:
            ax[0].plot(clay, depth[:len(clay)], "-o", color=color, label=name, ms=3)
        if sand is not None:
            ax[1].plot(sand, depth[:len(sand)], "-o", color=color, ms=3)

    ax[0].set_title("Soil clay profile", fontweight="bold")
    ax[0].set_xlabel("PCT_CLAY (%)"); ax[0].set_ylabel("depth (m)")
    ax[1].set_title("Soil sand profile", fontweight="bold")
    ax[1].set_xlabel("PCT_SAND (%)")
    for a in ax:
        a.invert_yaxis(); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, ncol=2, title="column")
    fig.suptitle("Per-column soil — verify each column got a distinct profile",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ {out_path}")
    if mismatches:
        print(f"   ✗ {mismatches} column(s) with surface/domain coordinate mismatch — "
              f"FIX BEFORE RUNNING")
    else:
        print("   ✓ surface/domain coordinates consistent for all columns")


def plot_timeseries(cases, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    S = 86400.0
    # third panel (SWE) only when the runs carry H2OSNO (output since 2026-07)
    probe = sorted(glob.glob(cases[0] + "/run/*.elm.h0.*.nc"))
    has_swe = False
    if probe:
        d0 = xr.open_dataset(probe[0], decode_times=False)
        has_swe = "H2OSNO" in d0
        d0.close()
    npan = 3 if has_swe else 2
    fig, ax = plt.subplots(1, npan, figsize=(5.6 * npan, 4.2))
    for c, color in zip(cases, _colors(len(cases))):
        name = c.split(".")[-1]
        fhs = sorted(glob.glob(c + "/run/*.elm.h0.*.nc"))
        if not fhs:
            print(f"  ! {name}: no history files")
            continue
        qc, qo, sw = [], [], []
        for f in fhs:
            d = xr.open_dataset(f, decode_times=False)
            qc.append(np.ravel(d["QCHARGE"].values) if "QCHARGE" in d else [])
            qo.append(np.ravel(d["QOVER"].values) if "QOVER" in d else [])
            if has_swe:
                sw.append(np.ravel(d["H2OSNO"].values) if "H2OSNO" in d else [])
            d.close()
        qc = np.concatenate(qc) * S
        qo = np.concatenate(qo) * S
        x = np.arange(len(qc))
        ax[0].plot(x, qc, color=color, lw=.8, label=name)
        ax[1].plot(x, qo, color=color, lw=.8)
        if has_swe and sw:
            ax[2].plot(np.arange(len(np.concatenate(sw))), np.concatenate(sw),
                       color=color, lw=.8)

    ax[0].set_title("Recharge  QCHARGE (mm/day)", fontweight="bold")
    ax[1].set_title("Runoff  QOVER (mm/day)", fontweight="bold")
    if has_swe:
        ax[2].set_title("Snowpack  H2OSNO / SWE (mm)", fontweight="bold")
    for a in ax:
        a.set_xlabel("timestep"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, ncol=2, title="column")
    fig.suptitle("Per-column output dynamics — verify the run is sensible",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Per-column debug plots (read-only)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-file", default="cases.json")
    ap.add_argument("--surfaces", action="store_true", help="plot built soil profiles")
    ap.add_argument("--timeseries", action="store_true", help="plot run output fluxes")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cases = json.loads((run_dir / args.cases_file).read_text())
    out = run_dir / "04_analysis"
    out.mkdir(parents=True, exist_ok=True)
    if not (args.surfaces or args.timeseries):
        args.surfaces = args.timeseries = True   # default: both
    if args.surfaces:
        plot_surfaces(cases, out / "debug_surfaces.png")
    if args.timeseries:
        plot_timeseries(cases, out / "debug_timeseries.png")


if __name__ == "__main__":
    main()
