#!/usr/bin/env python3
"""
Sampling-design figure, drawn from the columns the MODEL will run.

Every panel uses post-warm-start values: `elevation_m` is the donor gridcell's
TOPO, the soil is that gridcell's own profile, and the initial soil water is
read out of the finidat each column will start from. So the figure describes the
ensemble ELM integrates rather than the one that was sampled — the 3DEP
elevations did their job choosing the columns and do not appear here.

This is a DESIGN figure. No model output is plotted anywhere in it.

    python3 tools/plot_sampling_design.py --run-dir <dir> [--reception <json>]
                                          [--year 2015] [--out design.png]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(run_dir: Path, reception: Path | None):
    """The columns, the basin polygon, and the basin's name.

    The polygon and the name both come from reception when it is given: a run
    dir written straight from the sampler carries the columns and the boundary
    but not what the place is called.
    """
    cj = json.loads((run_dir / "01_inputs" / "columns.json").read_text())
    rings, name = cj.get("boundary"), (cj.get("sampling_domain") or {}).get("name")
    if reception and reception.is_file():
        rec = json.loads(reception.read_text())
        rec = rec.get("reception", rec)   # a chain artifact nests it; a saved
        rings = rings or (rec.get("grid") or {}).get("boundary")   # package doesn't
        name = name or ((rec.get("brief") or {}).get("domain") or {}).get("name")
    return cj, rings, name or run_dir.name


def _precip(cols, year):
    """Annual NLDAS precipitation at each column.

    NLDAS is 12 km against 1 km columns, so columns sharing a forcing cell share
    a value. That is the forcing the run is actually driven by, and coarse
    forcing over a finer model grid is the ordinary situation in hydrology, not
    a defect to hide.
    """
    try:
        import expand_sampling as exp
        return exp.nldas_annual_precip(cols, year)
    except Exception as e:                                   # noqa: BLE001
        print(f"   precip unavailable: {str(e)[:90]}")
        return None


def _init_soil_water(run_dir: Path, col):
    """Volumetric soil water the column STARTS from, layer by layer.

    The warm start wrote one finidat per column, subset from the CONUS restart.
    H2OSOI_LIQ and H2OSOI_ICE are kg/m2 over levtot = 5 snow layers + 15 ground
    layers, for every column of the gridcell; the vegetated soil column is the
    one with cols1d_ityplun == 1. Liquid and ice are summed because in a
    January restart over snow country most of the near-surface water is frozen
    and plotting liquid alone would show an empty soil.

    Dividing by 1000 kg/m3 and the layer thickness gives volumetric water. The
    thicknesses come from the column's own soil_profile, whose bounds ARE the
    first ten ELM ground layers, so this panel and the soil panel share a depth
    axis. Returns (None, None) when the file or the profile is missing.
    """
    import numpy as np
    f = run_dir / "warmstart" / f"finidat_{col['id']}.nc"
    layers = (col.get("soil_profile") or {}).get("layers") or []
    if not f.is_file() or not layers:
        return None, None
    import netCDF4 as nc
    with nc.Dataset(f) as d:
        soil = np.where(np.asarray(d.variables["cols1d_ityplun"][:]) == 1)[0]
        if soil.size == 0:
            return None, None
        i = int(soil[0])
        w = (np.asarray(d.variables["H2OSOI_LIQ"][i, 5:], float)
             + np.asarray(d.variables["H2OSOI_ICE"][i, 5:], float))
    n = min(len(layers), len(w))
    dz = np.array([(l["depth_bot_cm"] - l["depth_top_cm"]) / 100.0
                   for l in layers[:n]])
    mid = np.array([(l["depth_bot_cm"] + l["depth_top_cm"]) / 2.0
                    for l in layers[:n]])
    return w[:n] / (1000.0 * dz), mid


def _depth_axis(ax):
    """Depth downward, on a log scale, for the two profile panels.

    ELM's layers are exponentially spaced — the top one is 1.8 cm thick and the
    tenth is 1.5 m — so a linear axis spends 99% of its height on layers that
    barely differ and collapses the near-surface, where both the texture change
    and the frozen initial water actually are.
    """
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.set_yscale("log")
    ax.set_ylim(400, 0.7)
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reception", default="")
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    rd = Path(a.run_dir)
    cj, rings, name = _load(rd, Path(a.reception) if a.reception else None)
    cols = cj["columns"]
    grid = cj.get("grid") or []
    elev = np.array([c["elevation_m"] for c in cols], float)
    lat = np.array([c["lat"] for c in cols], float)
    lon = np.array([c["lon"] for c in cols], float)
    pin = np.array([bool(c.get("pinned")) for c in cols])

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16,
                         "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5), layout="constrained")
    cmap, vmin, vmax = "terrain", elev.min(), elev.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    def points(ax, x, y):
        """Every panel draws the same ensemble: stars pinned, circles not."""
        ax.scatter(x[~pin], y[~pin], c=elev[~pin], cmap=cmap, norm=norm, s=110,
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.scatter(x[pin], y[pin], c=elev[pin], cmap=cmap, norm=norm, s=300,
                   marker="*", edgecolor="k", linewidth=0.9, zorder=4)

    # (a) WHERE THE COLUMNS ARE ────────────────────────────────────────────────
    ax = axes[0, 0]
    if rings:
        xs, ys = zip(*[(p[0], p[1]) for p in max(rings, key=len)])
        ax.plot(xs, ys, color="0.35", lw=1.6, zorder=1)
    if grid:
        ax.scatter([p["lon"] for p in grid], [p["lat"] for p in grid], s=6,
                   c="0.8", zorder=2)
    points(ax, lon, lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("(a) columns")
    ax.set_aspect(1 / np.cos(np.radians(float(lat.mean()))))
    # Below the axes, not inside them: `loc="best"` parked the key on top of two
    # columns, and a legend that hides the data it explains is worse than none.
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=9,
               label="stratified"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=16,
               label="pinned at station"),
        Line2D([], [], ls="", marker=".", color="0.8", ms=12, label="DEM sample"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)

    # (b) FORCING ACROSS THE GRADIENT ──────────────────────────────────────────
    ax = axes[0, 1]
    pr = _precip(cols, a.year) if a.year else None
    if pr:
        p = np.array([pr.get(c["id"], np.nan) for c in cols], float)
        points(ax, p, elev)
        n_cell = len(set(np.round(p[np.isfinite(p)], 1)))
        ax.set_xlabel(f"NLDAS annual precipitation {a.year} (mm)")
        ax.set_title(f"(b) forcing — {n_cell} distinct values "
                     f"for {len(cols)} columns")
    else:
        ax.text(0.5, 0.5, "pass --year for the NLDAS panel", ha="center",
                transform=ax.transAxes, color="0.4")
        ax.set_title("(b) forcing")
    ax.set_ylabel("donor elevation (m)")

    # (c) THE SOIL THE MODEL RUNS ──────────────────────────────────────────────
    ax = axes[1, 0]
    n_prof = 0
    for c in cols:
        sp = (c.get("soil_profile") or {}).get("layers") or []
        v = [l.get("sand_pct") for l in sp]
        if not sp or any(x is None for x in v):
            continue
        d = [(l["depth_top_cm"] + l["depth_bot_cm"]) / 2 for l in sp]
        ax.plot(v, d, color=sm.to_rgba(c["elevation_m"]),
                lw=2.4 if c.get("pinned") else 1.2,
                alpha=0.95 if c.get("pinned") else 0.7)
        n_prof += 1
    _depth_axis(ax)
    ax.set_xlabel("sand (%)"); ax.set_ylabel("depth (cm)")
    ax.set_title(f"(c) donor soil — {n_prof} profiles")

    # (d) THE STATE THE RUN STARTS FROM ────────────────────────────────────────
    ax = axes[1, 1]
    n_init = 0
    for c in cols:
        theta, mid = _init_soil_water(rd, c)
        if theta is None:
            continue
        ax.plot(theta, mid, color=sm.to_rgba(c["elevation_m"]),
                lw=2.4 if c.get("pinned") else 1.2,
                alpha=0.95 if c.get("pinned") else 0.7)
        n_init += 1
    if n_init:
        _depth_axis(ax)
        ax.set_xlabel("initial soil water, liquid + ice (m$^3$/m$^3$)")
    else:
        ax.text(0.5, 0.5, "no finidat in warmstart/", ha="center",
                transform=ax.transAxes, color="0.4")
    ax.set_ylabel("depth (cm)")
    ax.set_title(f"(d) initial condition — {n_init} profiles")

    # ONE colorbar for the whole figure: elevation is the same variable in every
    # panel, and it had been drawn twice.
    fig.colorbar(sm, ax=axes, label="donor elevation (m)", fraction=0.035,
                 shrink=0.6, pad=0.015)
    d = cj.get("sampling_design") or {}
    fig.suptitle(a.title or
                 f"{name} — {len(cols)} columns "
                 f"({d.get('n_pinned', 0)} pinned, {d.get('n_stratified', 0)} stratified)",
                 fontsize=18)
    out = Path(a.out) if a.out else rd / "sampling_design.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
