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


def _soil_layers(col, var):
    """(values, layer-midpoint depths) for one donor profile, or (None, None)."""
    sp = (col.get("soil_profile") or {}).get("layers") or []
    v = [l.get(var) for l in sp]
    if not sp or any(x is None for x in v):
        return None, None
    return v, [(l["depth_top_cm"] + l["depth_bot_cm"]) / 2 for l in sp]


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


# Everything the CONUS surfdata carries per soil layer, which is everything ELM
# is given: PCT_SAND, PCT_CLAY, ORGANIC and PCT_GRVL over nlevsoi=10. Porosity,
# conductivity, retention and the thermal properties are not inputs — ELM derives
# them from these four at runtime.
SOIL_VARS = {"sand_pct": "sand (%)",
             "clay_pct": "clay (%)",
             "organic_kg_m3": "soil organic matter (kg m$^{-3}$)",
             "gravel_pct": "gravel (%)"}


# Esri's World Hillshade: relief and nothing else. Measured against the
# alternatives over this basin it is both the most detailed and the least
# coloured — mean saturation 9 against World Shaded Relief's 18, at 33% more
# contrast — so it needs no muting, and muting is what made the first attempt
# bland. World Light Gray Base is neutral but carries no relief at all, which
# for a design stratified BY elevation throws away the context that matters.
TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}")
MAX_TILES = 48

# VIRIDIS, NOT TERRAIN, for the columns. `terrain` runs blue-green-yellow-brown
# -white, which is every colour a landscape has: on a hillshade its browns sat
# in the shadows and its white top end — the highest columns, the ones a
# snow-dominated design is most about — vanished into the light valleys. It
# disappeared into the white background of panels (c) and (d) too.
#
# Desaturating the basemap "fixed" that by taking the relief out with the hue.
# The honest fix is to colour the DATA in something a landscape never contains.
# Viridis is perceptually uniform, colourblind-safe, and dark at the low end and
# bright at the high one, so it still reads as elevation.
CMAP = "viridis"


def _tile_xy(lon, lat, z):
    """Web Mercator tile coordinates, the slippy-map convention every XYZ
    service uses."""
    import math
    n = 2.0 ** z
    r = math.radians(lat)
    return ((lon + 180.0) / 360.0 * n,
            (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)


def _tile_lonlat(x, y, z):
    import math
    n = 2.0 ** z
    return (x / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))))


def _basemap(ax, extent):
    """Paste an XYZ hillshade under the map panel, in plain lon/lat.

    Each tile is drawn with imshow at its own lon/lat bounds rather than through
    cartopy. Cartopy would be the obvious tool and was tried first: its GeoAxes
    mis-places itself inside a gridspec — the panel hangs off the canvas and its
    title disappears — because the fixed aspect is applied at draw time, after
    the layout engine has sized the cell. Drawing the tiles directly keeps every
    panel on ordinary axes and every coordinate in degrees.

    Tiles are square in Mercator and we draw them as latitude rectangles, so
    each one is stretched by how much sec(lat) varies across it: 0.4% over a
    0.3-degree tile at 38N, well under a pixel.

    A missing basemap is not an error. Compute nodes have no network, and a
    figure without a backdrop beats one that cannot be drawn.
    """
    import io
    import urllib.request
    import numpy as np
    from PIL import Image

    for z in range(11, 5, -1):
        x0, y0 = _tile_xy(extent[0], extent[3], z)
        x1, y1 = _tile_xy(extent[1], extent[2], z)
        tiles = [(tx, ty) for tx in range(int(x0), int(x1) + 1)
                 for ty in range(int(y0), int(y1) + 1)]
        if len(tiles) <= MAX_TILES:
            break
    try:
        for tx, ty in tiles:
            url = TILES.format(z=z, x=tx, y=ty)
            with urllib.request.urlopen(url, timeout=20) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            w, n = _tile_lonlat(tx, ty, z)
            e, s = _tile_lonlat(tx + 1, ty + 1, z)
            ax.imshow(np.asarray(img), extent=[w, e, s, n], origin="upper",
                      zorder=0, interpolation="bilinear")
    except Exception as ex:                                  # noqa: BLE001
        print(f"   basemap unavailable ({type(ex).__name__}: {str(ex)[:60]})")
        return
    print(f"   basemap: {len(tiles)} tiles at zoom {z}")


def _map_axes(fig, cell, extent, basemap=True):
    """The map panel: lon/lat at the right aspect, over a hillshade."""
    import math
    ax = fig.add_subplot(cell)
    if basemap:
        _basemap(ax, extent)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect(1 / math.cos(math.radians((extent[2] + extent[3]) / 2)))
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reception", default="")
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--soil-vars", default="sand_pct,clay_pct,organic_kg_m3",
                    help=f"1-3 of {', '.join(sorted(SOIL_VARS))}")
    ap.add_argument("--no-basemap", action="store_true")
    a = ap.parse_args()

    soil_vars = [v.strip() for v in a.soil_vars.split(",") if v.strip()]
    bad = [v for v in soil_vars if v not in SOIL_VARS]
    if bad or not 1 <= len(soil_vars) <= 3:
        ap.error(f"--soil-vars takes 1-3 of {sorted(SOIL_VARS)}, got {soil_vars}")

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
    # Top row: where the columns are, what drives them, what state they start
    # in. Bottom row: the static soil the model was handed. Three columns, so
    # the soil row holds three properties instead of one.
    fig = plt.figure(figsize=(20, 13), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    cmap, vmin, vmax = CMAP, elev.min(), elev.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    def points(ax, x, y):
        """Every panel draws the same ensemble: stars pinned, circles not."""
        ax.scatter(x[~pin], y[~pin], c=elev[~pin], cmap=cmap, norm=norm, s=110,
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.scatter(x[pin], y[pin], c=elev[pin], cmap=cmap, norm=norm, s=300,
                   marker="*", edgecolor="k", linewidth=0.9, zorder=4)

    def profiles(ax, values_of):
        """One depth profile per column, coloured by elevation, pinned heavier.

        `values_of(col)` returns (values, depths_cm) or (None, None) when that
        column has nothing to draw. Returns how many were drawn — the count only
        earns a place in the title when it is short of the ensemble.
        """
        n = 0
        for c in cols:
            v, d = values_of(c)
            if v is None:
                continue
            ax.plot(v, d, color=sm.to_rgba(c["elevation_m"]),
                    lw=2.4 if c.get("pinned") else 1.2,
                    alpha=0.95 if c.get("pinned") else 0.7)
            n += 1
        if n:
            _depth_axis(ax)
        return n

    def short(n):
        return f" — {n} of {len(cols)}" if n < len(cols) else ""

    # (a) WHERE THE COLUMNS ARE ────────────────────────────────────────────────
    # Extent from the basin when we have it, since the polygon is the thing the
    # panel is about; from the columns otherwise. 5% of margin either way.
    bx = [p[0] for p in max(rings, key=len)] if rings else list(lon)
    by = [p[1] for p in max(rings, key=len)] if rings else list(lat)
    mx, my = 0.05 * (max(bx) - min(bx)), 0.05 * (max(by) - min(by))
    extent = [min(bx) - mx, max(bx) + mx, min(by) - my, max(by) + my]

    ax = _map_axes(fig, gs[0, 0], extent, basemap=not a.no_basemap)
    axes = [ax]
    if rings:
        xs, ys = zip(*[(p[0], p[1]) for p in max(rings, key=len)])
        ax.plot(xs, ys, color="0.15", lw=1.8, zorder=5)
    if grid:
        ax.scatter([p["lon"] for p in grid], [p["lat"] for p in grid], s=6,
                   c="0.45", zorder=2)
    points(ax, lon, lat)
    ax.set_title("(a) columns")

    # (b) FORCING ACROSS THE GRADIENT ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); axes.append(ax)
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

    # (c) THE STATE THE RUN STARTS FROM ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); axes.append(ax)
    n = profiles(ax, lambda c: _init_soil_water(rd, c))
    if not n:
        ax.text(0.5, 0.5, "no finidat in warmstart/", ha="center",
                transform=ax.transAxes, color="0.4")
    ax.set_ylabel("depth (cm)")
    ax.set_title("(c) initial soil water, liquid + ice "
                 "(m$^3$/m$^3$)" + short(n))

    # (d-f) THE SOIL THE MODEL WAS HANDED ──────────────────────────────────────
    # The bottom row shares one depth axis, so it is labelled once.
    first = None
    for j, var in enumerate(soil_vars):
        ax = fig.add_subplot(gs[1, j], sharey=first)
        axes.append(ax)
        n = profiles(ax, lambda c, v=var: _soil_layers(c, v))
        ax.set_title(f"({'def'[j]}) "
                     + ("donor soil — " if j == 0 else "")
                     + SOIL_VARS[var] + short(n))
        if first is None:
            first = ax
            ax.set_ylabel("depth (cm)")
        else:
            ax.tick_params(labelleft=False)

    # ONE colorbar for the whole figure: elevation is the same variable in every
    # panel, and it had been drawn twice.
    fig.colorbar(sm, ax=axes, label="donor elevation (m)", fraction=0.035,
                 shrink=0.6, pad=0.015)
    # The marker key belongs to the whole figure too. Inside panel (a) it landed
    # on top of columns; anchored under panel (a) it landed on panel (c)'s title.
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=9,
               label="stratified"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=16,
               label="pinned at station"),
        Line2D([], [], ls="", marker=".", color="0.45", ms=12,
               label="DEM sample"),
    ], loc="outside lower center", ncol=3, frameon=False)
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
