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

import figstyle                                              # noqa: E402


def _load(run_dir: Path, reception: Path | None):
    """The columns, the basin polygon, and the basin's name.

    The polygon and the name both come from reception when it is given: a run
    dir written straight from the sampler carries the columns and the boundary
    but not what the place is called.
    """
    # elm_columns.json FIRST: it is the MCP's post-warm-start columns, which is
    # what actually ran. columns.json is what was sampled, and the warm start
    # moves every column off it. Older runs have only columns.json, which the
    # MCP used to overwrite in place — for those the fallback IS the snapped
    # version, so both eras read correctly here.
    cj = next(json.loads(p.read_text())
              for n in ("elm_columns.json", "columns.json")
              for p in [run_dir / "01_inputs" / n] if p.is_file())
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


def _check_titles(fig):
    """Warn when a panel's titles are wider than the panel.

    Drawing at printed size makes overflow visible, but visible is not the same
    as noticed — the first version of this figure shipped with (b)'s title
    running through (c)'s and I only caught it by looking. A panel here is about
    28 characters wide at 8 pt, and basin names and column counts vary, so
    whether a title fits is a property of the DATA and has to be measured on
    every figure rather than decided once.

    Reports; never changes the figure. Titles are the author's call.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    over = []
    for ax in fig.axes:
        w = ax.get_window_extent(r).width
        used = sum(t.get_window_extent(r).width
                   for t in (ax.title, ax._left_title, ax._right_title)
                   if t.get_text())
        if used > w:
            over.append(f"title {ax.get_title('left') or ax.get_title()!r} "
                        f"({used / w:.0%} of its panel)")
        xl = ax.xaxis.label
        if xl.get_text() and xl.get_window_extent(r).width > w:
            over.append(f"x label {xl.get_text()!r} "
                        f"({xl.get_window_extent(r).width / w:.0%})")
    for o in over:
        print(f"   ⚠️  title overflows: {o}")
    return over


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
SOIL_VARS = {"sand_pct": ("sand", "%"),
             "clay_pct": ("clay", "%"),
             "organic_kg_m3": ("organic matter", "kg m$^{-3}$"),
             "gravel_pct": ("gravel", "%")}


# The hillshade itself, and why it is that one, moved to src/core/basemap.py.

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


def _basemap(ax, extent, tiles=None, max_tiles=None):
    """Paste an XYZ hillshade under the map panel. See src/core/basemap.py.

    MOVED OUT (2026-08-12). The tile maths, the certifi context and the zoom
    search now live in core.basemap, because the ELM MCP's comparison figures
    draw their column maps over the same ground and a second copy of this would
    drift from the one that draws the design. This is the same function it
    always was, one import further away.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from core.basemap import paste
    return paste(ax, extent, tiles=tiles, max_tiles=max_tiles)


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


# The CLI's defaults, in one place so the programmatic entry point below and
# `--help` cannot drift apart.
DEFAULTS = {"reception": "", "year": 0, "out": "", "title": "",
            "soil_vars": "sand_pct,clay_pct,organic_kg_m3",
            "no_basemap": False, "width": figstyle.WIDTH["double"], "font": 8.0}


def render_run(run_dir, **kw):
    """Draw the figure for a run directory. The pipeline's entry point.

    Exists so the Experiment Manager can call this renderer directly instead of
    shelling out. Before 2026-08-13 it could not: everything lived inside
    main(), so the manager drew its own 2x3 from expand_sampling.plot_columns —
    a figure with its own rcParams, a title at 15 pt on a 17.5-inch canvas
    (about 6 pt on the page), and a panel reading `fan_wtd_m`, a field whose
    producer was removed on 2026-08-07. This one is the styled twin nothing
    ever called.
    """
    from types import SimpleNamespace
    opts = dict(DEFAULTS, **kw)
    opts["run_dir"] = str(run_dir)
    for k in ("reception", "out", "title"):
        opts[k] = str(opts[k] or "")
    return render(SimpleNamespace(**opts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reception", default=DEFAULTS["reception"])
    ap.add_argument("--year", type=int, default=DEFAULTS["year"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--title", default=DEFAULTS["title"])
    ap.add_argument("--soil-vars", default=DEFAULTS["soil_vars"],
                    help=f"1-3 of {', '.join(sorted(SOIL_VARS))}")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--width", type=float, default=DEFAULTS["width"],
                    help="printed width in inches (the canvas IS the page)")
    ap.add_argument("--font", type=float, default=DEFAULTS["font"],
                    help="body text size in points, on the page")
    a = ap.parse_args()
    a.run_dir = a.run_dir
    a.soil_vars = a.soil_vars
    print(f"saved {render(a)}")


def render(a):
    """Draw it. `a` carries the CLI's fields; see DEFAULTS."""
    soil_vars = [v.strip() for v in a.soil_vars.split(",") if v.strip()]
    bad = [v for v in soil_vars if v not in SOIL_VARS]
    if bad or not 1 <= len(soil_vars) <= 3:
        raise ValueError(
            f"soil_vars takes 1-3 of {sorted(SOIL_VARS)}, got {soil_vars}")

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

    # Sizes come from figstyle, which draws at the PRINTED width so the point
    # sizes here are the point sizes on the page. `k` scales what has to shrink
    # with the canvas: marker areas as k**2, line widths as k.
    k = figstyle.manuscript(a.width, a.font)

    # Top row: where the columns are, what drives them, what state they start
    # in. Bottom row: the static soil the model was handed. Three columns, so
    # the soil row holds three properties instead of one.
    fig = plt.figure(figsize=(a.width, a.width * 0.65), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    cmap, vmin, vmax = CMAP, elev.min(), elev.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Marker AREA scales as the square of the canvas; edges and rules as the
    # canvas, with a floor so nothing drops out of print. The weights are set
    # for the PRINTED panel — a 1.4 pt profile line looked right on a 6 inch
    # panel and is a bar across a 2 inch one.
    s_col, s_pin = 110 * (20 * k) ** 2 / 400, 300 * (20 * k) ** 2 / 400
    lw_edge, lw_line = max(0.3, 1.4 * k), max(0.5, 2.0 * k)

    def tag(ax, letter, text, note=""):
        """Panel title left, measurement right, on the one title line.

        LEFT-aligned, and short. At 7.2 inches a panel is about 28 characters
        wide, so a centred title long enough to say anything runs into its
        neighbour — "(b) forcing — 14 distinct values for 17 columns" ran
        straight through (c)'s. Splitting the measurement onto matplotlib's
        right-hand title keeps both on one line and inside the panel; the
        title says what the panel is, the note says what was counted.
        """
        ax.set_title(f"({letter}) {text}", loc="left")
        if note:
            ax.set_title(note, loc="right", fontsize=a.font - 1, color="0.35")

    def points(ax, x, y):
        """Every panel draws the same ensemble: stars pinned, circles not."""
        ax.scatter(x[~pin], y[~pin], c=elev[~pin], cmap=cmap, norm=norm,
                   s=s_col, edgecolor="k", linewidth=lw_edge, zorder=3)
        ax.scatter(x[pin], y[pin], c=elev[pin], cmap=cmap, norm=norm, s=s_pin,
                   marker="*", edgecolor="k", linewidth=lw_edge * 1.4, zorder=4)

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
                    lw=lw_line * (2.0 if c.get("pinned") else 1.0),
                    alpha=0.95 if c.get("pinned") else 0.7)
            n += 1
        if n:
            _depth_axis(ax)
        return n

    def short(n):
        return f"{n} of {len(cols)} columns" if n < len(cols) else ""

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
        ax.plot(xs, ys, color="0.15", lw=max(0.6, 2.5 * k), zorder=5)
    if grid:
        ax.scatter([p["lon"] for p in grid], [p["lat"] for p in grid],
                   s=max(1.0, s_col * 0.055), c="0.45", zorder=2)
    points(ax, lon, lat)
    tag(ax, "a", "columns")

    # (b) FORCING ACROSS THE GRADIENT ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); axes.append(ax)
    pr = _precip(cols, a.year) if a.year else None
    if pr:
        p = np.array([pr.get(c["id"], np.nan) for c in cols], float)
        points(ax, p, elev)
        n_cell = len(set(np.round(p[np.isfinite(p)], 1)))
        ax.set_xlabel("NLDAS precipitation (mm)")
        tag(ax, "b", "forcing", f"{n_cell} distinct")
    else:
        ax.text(0.5, 0.5, "pass --year for the NLDAS panel", ha="center",
                transform=ax.transAxes, color="0.4")
        tag(ax, "b", "forcing")
    ax.set_ylabel("donor elevation (m)")

    # (c) THE STATE THE RUN STARTS FROM ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); axes.append(ax)
    n = profiles(ax, lambda c: _init_soil_water(rd, c))
    if not n:
        ax.text(0.5, 0.5, "no finidat in warmstart/", ha="center",
                transform=ax.transAxes, color="0.4")
    ax.set_ylabel("depth (cm)")
    ax.set_xlabel("m$^3$ m$^{-3}$")
    tag(ax, "c", "initial soil water", short(n))

    # (d-f) THE SOIL THE MODEL WAS HANDED ──────────────────────────────────────
    # The bottom row shares one depth axis, so it is labelled once.
    first = None
    for j, var in enumerate(soil_vars):
        ax = fig.add_subplot(gs[1, j], sharey=first)
        axes.append(ax)
        n = profiles(ax, lambda c, v=var: _soil_layers(c, v))
        ax.set_xlabel(SOIL_VARS[var][1])
        tag(ax, "def"[j], SOIL_VARS[var][0], short(n))
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
    ms = s_col ** 0.5                    # legend markers are DIAMETERS, not areas
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=ms,
               label="stratified"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=ms * 1.7,
               label="pinned at station"),
        Line2D([], [], ls="", marker=".", color="0.45", ms=ms,
               label="DEM sample"),
    ], loc="outside lower center", ncol=3, frameon=False)
    # COUNTED FROM THE COLUMNS, not read off a summary block. This used to read
    # cj["sampling_design"]["n_pinned"], which the MCP's elm_columns.json does
    # not carry — `sampling_design` is null there — so the title said "0 pinned,
    # 0 stratified" while panel (a) drew four stars from the same file's
    # `pinned` flags. The columns are the authority and are always present;
    # anything derivable from them is derived here.
    n_pin = int(pin.sum())
    fig.suptitle(a.title or
                 f"{name}{f' {a.year}' if a.year else ''} — {len(cols)} columns "
                 f"({n_pin} pinned at a station, {len(cols) - n_pin} stratified)",
                 fontsize=a.font + 2)
    _check_titles(fig)
    out = Path(a.out) if a.out else rd / "sampling_design.png"
    fig.savefig(out)
    plt.close(fig)          # the manager draws in-process; do not leak figures
    return out


if __name__ == "__main__":
    main()
