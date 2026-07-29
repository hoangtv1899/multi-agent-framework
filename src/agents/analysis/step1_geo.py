#!/usr/bin/env python3
"""
Shared geography for the step 1 validators
src/agents/analysis/step1_geo.py

Two things every observable needs and none of them should own:

    in_polygon     is this station inside the watershed?
    plot_two_maps  observed on the left, modelled on the right, one scale

The first exists because reception fetches observations by BBOX, and a bbox is
the rectangle around a basin — so stations in its corners are outside the
watershed. The DEM grid gets clipped to the boundary; the stations never were.
On the 2019 Upper Gunnison run that left 3 of 5 SNOTEL sites and 12 of 19
gauges outside the basin, and the SWE pairing dropped an in-basin station in
favour of an out-of-basin one before anyone noticed.

The second exists because the maps are what caught it. A scatter and a
hydrograph both answer "do the numbers agree"; only a map answers "are these
two networks even sampling the same place", and that question turned out to
be the one that mattered.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _num(x) -> Optional[float]:
    try:
        f = float(x)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def in_polygon(lat: float, lon: float, rings) -> bool:
    """Ray casting against the watershed rings.

    No rings means no polygon to test against — a run with no resolved HUC.
    Returns False, and callers must treat "no boundary" as "exclude nothing"
    rather than "exclude everything".
    """
    for ring in (rings or []):
        try:
            n, hit = len(ring), False
            for i in range(n):
                x1, y1 = ring[i][0], ring[i][1]
                x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
                if (y1 > lat) != (y2 > lat):
                    xi = x1 + (lat - y1) * (x2 - x1) / ((y2 - y1) or 1e-12)
                    if lon < xi:
                        hit = not hit
            if hit:
                return True
        except Exception:
            continue
    return False


def split_by_basin(items: Sequence[Dict[str, Any]], rings
                   ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(inside, names_of_those_outside).

    An item with no coordinates is KEPT — it cannot be shown to be outside,
    and dropping it would confuse "outside the basin" with "we do not know
    where this is". The 2019 Gunnison run has one such gauge.
    """
    if not rings:
        return list(items), []
    inside, outside = [], []
    for it in items:
        lat, lon = _num(it.get("lat")), _num(it.get("lon"))
        if lat is None or lon is None:
            inside.append(it)
            continue
        (inside if in_polygon(lat, lon, rings) else outside).append(it)
    return inside, [str(o.get("id") or o.get("name") or o.get("triplet"))
                    for o in outside]


def plot_two_maps(obs_points:  Sequence[Tuple[float, float, float, str]],
                  mod_points:  Sequence[Tuple[float, float, float, str]],
                  boundary,
                  out_path,
                  label:       str,
                  obs_title:   str = "observed",
                  mod_title:   str = "modelled",
                  basemap:     bool = True,
                  zoom:        int = 9,
                  obs_sizes:   Optional[Sequence[float]] = None,
                  log:         bool = False) -> str:
    """Observed left, modelled right, ONE shared colour scale.

    Points are (lon, lat, value, name).

    The shared scale is the whole point. Independent scales would map the
    observed maximum and the modelled maximum to the same colour and make two
    very different fields look alike — exactly the comparison the figure is
    supposed to make impossible to fudge.

    OSM underneath at reduced alpha. Terrain is why stations sit where they
    do, so it is context rather than decoration; but a full-strength basemap
    competes with the data for the eye. Non-fatal if the tiles cannot be
    fetched — a figure must not depend on a third-party raster being
    reachable, and the watershed outline alone is enough to read it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allv = [p[2] for p in list(obs_points) + list(mod_points)]
    vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)

    # SymLog, not Log. A linear scale is unreadable when one value is orders
    # of magnitude above the rest — col_19's 5.2 mm/day flattened all seven
    # gauges into one indistinguishable dark purple. But plain LogNorm cannot
    # take a zero, and 15 of 19 columns produced EXACTLY zero runoff, which is
    # a qualitatively different result from "very small" and must stay visible
    # as such. SymLogNorm is linear below linthresh and logarithmic above, so
    # zero keeps its own place at the bottom.
    norm = None
    if log and allv:
        try:
            from matplotlib.colors import SymLogNorm
            pos = [v for v in allv if v > 0]
            lt = (min(pos) if pos else 1e-3)
            norm = SymLogNorm(linthresh=max(lt, 1e-6), vmin=0.0,
                              vmax=max(vmax, lt * 10), base=10)
        except Exception:
            norm = None

    xs = [p[0] for p in list(obs_points) + list(mod_points)]
    ys = [p[1] for p in list(obs_points) + list(mod_points)]
    for ring in (boundary or []):
        try:
            xs += [q[0] for q in ring]
            ys += [q[1] for q in ring]
        except Exception:
            pass
    if not xs:
        xs, ys = [-108.0, -106.5], [37.5, 39.0]
    mx = 0.06 * (max(xs) - min(xs) or 1)
    my = 0.06 * (max(ys) - min(ys) or 1)
    extent = [min(xs) - mx, max(xs) + mx, min(ys) - my, max(ys) + my]

    proj = tiler = None
    if basemap:
        try:
            import cartopy.crs as ccrs
            import cartopy.io.img_tiles as cimgt
            tiler, proj = cimgt.OSM(), ccrs.PlateCarree()
        except Exception:
            proj = None

    fig, axes = plt.subplots(
        1, 2, figsize=(17.5, 8.4),
        subplot_kw={"projection": proj} if proj is not None else None)

    sc = None
    for ax, points, title, sizes in (
            (axes[0], list(obs_points), obs_title, obs_sizes),
            (axes[1], list(mod_points), mod_title, None)):
        if proj is not None:
            import cartopy.crs as ccrs
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                ax.add_image(tiler, zoom, alpha=0.45, zorder=0)
            except Exception:
                pass
            gl = ax.gridlines(draw_labels=True, alpha=0.25, zorder=1)
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 13}
            kw = {"transform": ccrs.PlateCarree()}
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("longitude", fontsize=18)
            ax.set_ylabel("latitude", fontsize=18)
            ax.tick_params(labelsize=13)
            ax.grid(alpha=0.25)
            kw = {}

        for ring in (boundary or []):
            try:
                ax.plot([q[0] for q in ring], [q[1] for q in ring],
                        color="#111", lw=2.2, zorder=3, **kw)
            except Exception:
                pass
        if points:
            ckw = ({"norm": norm} if norm is not None
                   else {"vmin": vmin, "vmax": vmax})
            sc = ax.scatter([p[0] for p in points], [p[1] for p in points],
                            c=[p[2] for p in points], cmap="viridis",
                            s=(list(sizes) if sizes else 300),
                            edgecolor="#fff", linewidth=1.8, zorder=5,
                            **ckw, **kw)
        # Inside the axes, not above them. With cartopy the title sits beyond
        # the gridline labels and reads as floating text belonging to neither
        # panel; a reader could not tell which map was which.
        ax.set_title("")
        ax.text(0.5, 0.965, f"{title}   (n={len(points)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold", color="#111", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333",
                          alpha=0.92))

    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes), fraction=0.032, pad=0.04)
        cb.set_label(label, fontsize=19)
        cb.ax.tick_params(labelsize=14)

    # No bbox_inches="tight". With cartopy GeoAxes the tight box is computed
    # before the tiles and the frame exist, and cropped a whole figure down to
    # its colourbar. Draw, then save the full canvas.
    try:
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return str(out_path)


def plot_panels(panels, boundary, out_path, label: str,
                basemap: bool = True, zoom: int = 9,
                log: bool = False) -> str:
    """N map panels, one per network, ONE shared colour scale.

    panels is [(title, points, sizes_or_None)] with points as
    (lon, lat, value, name). The panel COUNT follows the data: a basin with no
    in-domain wells gets two panels rather than an empty third, because an
    empty axis reads as "measured nothing" instead of "nothing to measure".

    The shared scale is the reason this exists. Independent scales would map
    each network's own maximum to the same colour and make fields that differ
    by two orders of magnitude look alike.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [pn for pn in panels if pn[1]]
    if not panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "NOTHING TO MAP", ha="center", va="center",
                transform=ax.transAxes, fontsize=15, color="#b30000",
                fontweight="bold")
        fig.savefig(out_path, dpi=135)
        plt.close(fig)
        return str(out_path)

    allv = [q[2] for _t, pts, _s in panels for q in pts]
    vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)

    norm = None
    if log and allv:
        try:
            from matplotlib.colors import SymLogNorm
            pos = [v for v in allv if v > 0]
            lt = min(pos) if pos else 1e-3
            norm = SymLogNorm(linthresh=max(lt, 1e-6), vmin=0.0,
                              vmax=max(vmax, lt * 10), base=10)
        except Exception:
            norm = None

    xs = [q[0] for _t, pts, _s in panels for q in pts]
    ys = [q[1] for _t, pts, _s in panels for q in pts]
    for ring in (boundary or []):
        try:
            xs += [q[0] for q in ring]; ys += [q[1] for q in ring]
        except Exception:
            pass
    if not xs:
        xs, ys = [-108.0, -106.5], [37.5, 39.0]
    mx = 0.06 * (max(xs) - min(xs) or 1)
    my = 0.06 * (max(ys) - min(ys) or 1)
    extent = [min(xs) - mx, max(xs) + mx, min(ys) - my, max(ys) + my]

    proj = tiler = None
    if basemap:
        try:
            import cartopy.crs as ccrs
            import cartopy.io.img_tiles as cimgt
            tiler, proj = cimgt.OSM(), ccrs.PlateCarree()
        except Exception:
            proj = None

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(8.8 * n, 8.4), squeeze=False,
        subplot_kw={"projection": proj} if proj is not None else None)
    axes = [a for row in axes for a in row]

    sc = None
    for ax, (title, pts, sizes) in zip(axes, panels):
        if proj is not None:
            import cartopy.crs as ccrs
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            try:
                ax.add_image(tiler, zoom, alpha=0.45, zorder=0)
            except Exception:
                pass
            gl = ax.gridlines(draw_labels=True, alpha=0.25, zorder=1)
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 13}
            kw = {"transform": ccrs.PlateCarree()}
        else:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("longitude", fontsize=18)
            ax.set_ylabel("latitude", fontsize=18)
            ax.tick_params(labelsize=13); ax.grid(alpha=0.25)
            kw = {}

        for ring in (boundary or []):
            try:
                ax.plot([q[0] for q in ring], [q[1] for q in ring],
                        color="#111", lw=2.2, zorder=3, **kw)
            except Exception:
                pass
        ckw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
        sc = ax.scatter([q[0] for q in pts], [q[1] for q in pts],
                        c=[q[2] for q in pts], cmap="viridis",
                        s=(list(sizes) if sizes else 300),
                        edgecolor="#fff", linewidth=1.8, zorder=5,
                        **ckw, **kw)
        ax.text(0.5, 0.965, f"{title}   (n={len(pts)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold", color="#111", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333",
                          alpha=0.92))

    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes), fraction=0.032, pad=0.04)
        cb.set_label(label, fontsize=19)
        cb.ax.tick_params(labelsize=14)
    try:
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return str(out_path)
