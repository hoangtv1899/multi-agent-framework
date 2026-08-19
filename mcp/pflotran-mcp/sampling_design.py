#!/usr/bin/env python3
"""
Sampling-design figure, drawn from the columns PFLOTRAN will run.
mcp/pflotran-mcp/sampling_design.py

Every panel uses the columns AS BUILT — columns.json after create_decks_from_
columns has joined the site data and written the decks: the water table each
column was given, the depth its domain runs to, the rain that will drive it,
whether it runs transient or steady. So the figure describes the ensemble
PFLOTRAN integrates rather than the one that was sampled, exactly as ELM's
design figure describes its post-warm-start columns.

This is a DESIGN figure. No model output is plotted anywhere in it.

    (a) columns          where they are, over the terrain, with the basin divide
    (b) rain vs elevation  the year's Daymet total at each column (a borrowed
                         series is drawn open, and counted in the title)
    (c) rain through the year  every column's daily series, thin; the ensemble
                         mean, heavy
    (d) water table      the CONUS2 depth each column was given, against
                         elevation; steady columns are drawn open
    (e) the columns      each column as a bar from the surface to its domain
                         bottom on the CONUS2 layer ladder, its water table
                         marked — which layers each column spans
    (f) unsaturated column  how much column above the water table each run
                         starts with, against elevation

    A CONTROLLED SWEEP has no place, so it gets its own four panels
    (render_sweep), chosen when columns.json says approach: factor_sweep:
    (a) the design      one row per column: its treatment, in the factors' words
    (b) the columns     each column on the CONUS2 layer ladder with its water
                        table — what a water_table_m level does to the depth
    (c) water in        the written daily rain per column, or the steady rate
    (d) the soils       the retention curve of every texture class in the
                        design — what a soil level means

    python3 mcp/pflotran-mcp/sampling_design.py --run-dir <dir> [--reception <json>]
                                                [--out design.png]

Reaches back for two things, neither of them framework LOGIC: figstyle (the
house style — printed width and point sizes) and core.basemap (the hillshade
tiles), the same two ELM's design figure uses.
"""
import argparse
import json
import sys
from pathlib import Path

_FW = Path(__file__).resolve().parents[2]           # multi-agent-framework/
for _d in (_FW / "tools", _FW / "src"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import figstyle                                              # noqa: E402

# CONUS2's layer boundaries below ground — the depths a domain may end on. The
# server states them (create_decks_from_columns.BOUNDS); drawn here as the
# ladder the columns hang on.
CONUS2_BOUNDS_M = (0.1, 0.4, 1.0, 2.0, 7.0, 17.0, 42.0, 92.0, 192.0, 392.0)
CMAP = "viridis"


def _load(run_dir: Path, reception: Path | None):
    """The columns as built, the basin polygon, the rain series, the name."""
    cj = next(json.loads(p.read_text())
              for n in ("columns.json",)
              for p in [run_dir / "01_inputs" / n, run_dir / n] if p.is_file())
    rings = cj.get("boundary")
    name = (cj.get("sampling_domain") or {}).get("name")
    rain, calendar = {}, {}
    if reception and reception.is_file():
        rec = json.loads(reception.read_text())
        rec = rec.get("reception", rec)
        rings = rings or (rec.get("grid") or {}).get("boundary")
        name = name or ((rec.get("brief") or {}).get("domain") or {}).get("name")
        pr = rec.get("precipitation") or {}
        rain, calendar = (pr.get("series") or {}), (pr.get("calendar") or {})
    return cj, rings, rain, calendar, name or run_dir.name


def _rain_key(lat, lon):
    return f"{round(float(lat), 5)},{round(float(lon), 5)}"


def _series_for(col, rain):
    """(daily series, borrowed?) for a column: its own grid point's, or the
    one the server borrowed for it — recorded on the column as
    rain_borrowed_from."""
    own = rain.get(_rain_key(col["lat"], col["lon"]))
    if own is not None:
        return own, False
    key = col.get("rain_borrowed_from")
    if key and rain.get(key) is not None:
        return rain[key], True
    return None, False


def _basemap(ax, extent):
    from core.basemap import paste          # framework src is on sys.path above
    return paste(ax, extent)


def _map_axes(fig, cell, extent, basemap=True):
    import math
    ax = fig.add_subplot(cell)
    if basemap:
        _basemap(ax, extent)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect(1 / math.cos(math.radians((extent[2] + extent[3]) / 2)))
    return ax


DEFAULTS = {"reception": "", "out": "", "title": "", "no_basemap": False,
            "width": figstyle.WIDTH["double"], "font": 8.0}


def render_run(run_dir, **kw):
    """Draw the figure for a run directory. The Experiment Manager's entry point."""
    from types import SimpleNamespace
    opts = dict(DEFAULTS, **kw)
    opts["run_dir"] = str(run_dir)
    for k in ("reception", "out", "title"):
        opts[k] = str(opts[k] or "")
    return render(SimpleNamespace(**opts))


def render(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    rd = Path(a.run_dir)
    cj, rings, rain, calendar, name = _load(rd, Path(a.reception) if a.reception else None)
    cols = cj["columns"]
    grid = cj.get("grid") or []
    if not cols:
        raise ValueError("no columns to draw")
    if cj.get("approach") == "factor_sweep" or all(
            not isinstance(c.get("lat"), (int, float)) for c in cols):
        return render_sweep(a, rd, cj)

    def num(c, k):
        v = c.get(k)
        return float(v) if isinstance(v, (int, float)) else np.nan

    elev = np.array([num(c, "elevation_m") for c in cols])
    lat = np.array([num(c, "lat") for c in cols])
    lon = np.array([num(c, "lon") for c in cols])
    pin = np.array([bool(c.get("pinned")) for c in cols])
    trans = np.array([bool(c.get("transient")) for c in cols])
    wt = np.array([num(c, "water_table_m") for c in cols])
    dom = np.array([num(c, "depth_m") for c in cols])         # the deck's domain depth
    uns = np.array([num(c, "unsaturated_m") for c in cols])
    by_elevation = np.isfinite(elev).all()
    if not by_elevation:
        elev = np.zeros(len(cols))

    k = figstyle.manuscript(a.width, a.font)
    fig = plt.figure(figsize=(a.width, a.width * 0.65), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    vmin, vmax = float(elev.min()), float(elev.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP)
    NEUTRAL = "#4C6A8A"

    def colour_of(i):
        return sm.to_rgba(elev[i]) if by_elevation else NEUTRAL

    s_col, s_pin = 110 * (20 * k) ** 2 / 400, 300 * (20 * k) ** 2 / 400
    lw_edge, lw_line = max(0.3, 1.4 * k), max(0.5, 2.0 * k)

    def tag(ax, letter, text, note=""):
        ax.set_title(f"({letter}) {text}", loc="left")
        if note:
            ax.set_title(note, loc="right", fontsize=a.font - 1, color="0.35")

    def points(ax, x, y, open_mask=None):
        """Stars pinned, circles not; open markers where open_mask is True."""
        om = np.zeros(len(cols), bool) if open_mask is None else open_mask
        for star in (False, True):
            for opn in (False, True):
                m = (pin == star) & (om == opn) & np.isfinite(x) & np.isfinite(y)
                if not m.any():
                    continue
                if opn:
                    ax.scatter(x[m], y[m], s=(s_pin if star else s_col),
                               marker=("*" if star else "o"), facecolors="none",
                               edgecolors=[colour_of(i) for i in np.where(m)[0]],
                               linewidth=lw_edge * 1.6, zorder=(4 if star else 3))
                else:
                    ax.scatter(x[m], y[m], c=(elev[m] if by_elevation else NEUTRAL),
                               cmap=CMAP, norm=norm, s=(s_pin if star else s_col),
                               marker=("*" if star else "o"), edgecolor="k",
                               linewidth=lw_edge * (1.4 if star else 1),
                               zorder=(4 if star else 3))

    # (a) WHERE THE COLUMNS ARE ────────────────────────────────────────────────
    ring = max(rings, key=len) if rings else None
    bx = [p[0] for p in ring] if ring else list(lon[np.isfinite(lon)])
    by = [p[1] for p in ring] if ring else list(lat[np.isfinite(lat)])
    mx, my = 0.05 * (max(bx) - min(bx)), 0.05 * (max(by) - min(by))
    extent = [min(bx) - mx, max(bx) + mx, min(by) - my, max(by) + my]
    ax = _map_axes(fig, gs[0, 0], extent, basemap=not a.no_basemap)
    axes = [ax]
    if ring:
        xs, ys = zip(*[(p[0], p[1]) for p in ring])
        ax.plot(xs, ys, color="0.15", lw=max(0.6, 2.5 * k), zorder=5)
    if grid:
        ax.scatter([p["lon"] for p in grid], [p["lat"] for p in grid],
                   s=max(1.0, s_col * 0.055), c="0.45", zorder=2)
    points(ax, lon, lat)
    tag(ax, "a", "columns", f"{int(pin.sum())} pinned" if pin.any() else "")

    # (b) RAIN ACROSS THE GRADIENT ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); axes.append(ax)
    tot = np.full(len(cols), np.nan)
    borrowed = np.zeros(len(cols), bool)
    series_by_col = {}
    for i, c in enumerate(cols):
        s, b = _series_for(c, rain)
        if s:
            vals = [v for v in s if isinstance(v, (int, float))]
            tot[i] = float(sum(vals))
            borrowed[i] = b
            series_by_col[i] = vals
    if np.isfinite(tot).any():
        points(ax, tot, elev, open_mask=borrowed)
        yr = calendar.get("year") or []
        year = f"{min(yr)}" if yr else "the year"
        ax.set_xlabel("Daymet rain (mm)")
        note = (f"{year}" + (f" · {int(borrowed.sum())} borrowed" if borrowed.any() else ""))
        tag(ax, "b", "rain", note)
    else:
        ax.text(0.5, 0.5, "no rain series in reception.json", ha="center",
                transform=ax.transAxes, color="0.4")
        tag(ax, "b", "rain")
    ax.set_ylabel("elevation (m)")

    # (c) RAIN THROUGH THE YEAR ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); axes.append(ax)
    if series_by_col:
        n_days = max(len(v) for v in series_by_col.values())
        days = np.arange(n_days)
        arr = np.full((len(series_by_col), n_days), np.nan)
        for j, (i, vals) in enumerate(series_by_col.items()):
            arr[j, :len(vals)] = vals
            ax.plot(days[:len(vals)], vals, color=colour_of(i), lw=lw_line * 0.5,
                    alpha=0.35)
        ax.plot(days, np.nanmean(arr, axis=0), color="k", lw=lw_line * 1.2)
        ax.set_xlabel("day of year")
        ax.set_ylabel("mm day$^{-1}$")
        tag(ax, "c", "rain, daily",
            f"{len(series_by_col)} series" if len(series_by_col) < len(cols) else "")
    else:
        ax.text(0.5, 0.5, "no rain series", ha="center", transform=ax.transAxes,
                color="0.4")
        tag(ax, "c", "rain, daily")

    # (d) THE WATER TABLE EACH COLUMN WAS GIVEN ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); axes.append(ax)
    if np.isfinite(wt).any():
        x = np.where(wt > 0.05, wt, 0.05)                 # log axis; surface at the edge
        points(ax, x, elev, open_mask=~trans)
        ax.set_xscale("log")
        ax.set_xlabel("CONUS2 water table (m)")
        n_steady = int((~trans).sum())
        tag(ax, "d", "water table", f"{n_steady} steady" if n_steady else "")
    else:
        ax.text(0.5, 0.5, "no water_table_m on the columns", ha="center",
                transform=ax.transAxes, color="0.4")
        tag(ax, "d", "water table")
    ax.set_ylabel("elevation (m)")

    # (e) THE COLUMNS ON THE CONUS2 LADDER ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); axes.append(ax)
    order = np.argsort(elev)
    for b in CONUS2_BOUNDS_M:
        ax.axhline(b, color="0.85", lw=max(0.4, 1.2 * k), zorder=1)
    top = 0.05
    for xi, i in enumerate(order):
        if np.isfinite(dom[i]):
            ax.plot([xi, xi], [top, dom[i]], color=colour_of(i),
                    lw=lw_line * (2.2 if pin[i] else 1.4), solid_capstyle="butt",
                    zorder=3)
        if np.isfinite(wt[i]):
            ax.plot([xi], [max(wt[i], top)], marker=("*" if pin[i] else "o"),
                    color=colour_of(i), mec="k", mew=lw_edge,
                    ms=(9 if pin[i] else 5) * max(0.6, 20 * k / 4), zorder=4)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_ylim(CONUS2_BOUNDS_M[-1] * 1.15, top)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(cols[i].get("id", "")).replace("col_", "")
                        for i in order], fontsize=max(4, a.font - 3))
    ax.set_xlabel("column, by elevation")
    ax.set_ylabel("depth (m)")
    tag(ax, "e", "column depth",
        f"{int(np.isfinite(dom).sum())} decks" if np.isfinite(dom).sum() < len(cols) else "")

    # (f) HOW MUCH UNSATURATED COLUMN ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); axes.append(ax)
    if np.isfinite(uns).any():
        x = np.where(uns > 0.01, uns, 0.01)
        points(ax, x, elev, open_mask=~trans)
        ax.set_xscale("log")
        ax.set_xlabel("unsaturated at start (m)")
        n_sat = int(np.sum(np.isfinite(uns) & (uns < 0.5)))
        tag(ax, "f", "unsaturated",
            f"{n_sat} < 0.5 m" if n_sat else "")
    else:
        ax.text(0.5, 0.5, "no unsaturated_m on the columns", ha="center",
                transform=ax.transAxes, color="0.4")
        tag(ax, "f", "unsaturated")
    ax.set_ylabel("elevation (m)")

    # ── legend and colourbar ─────────────────────────────────────────────────
    handles = [Line2D([], [], marker="o", ls="", mfc="0.6", mec="k", ms=6, label="column"),
               Line2D([], [], marker="*", ls="", mfc="0.6", mec="k", ms=9, label="pinned"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=6,
                      label="open: steady / borrowed rain")]
    fig.legend(handles=handles, loc="outside upper right", ncol=3)
    if by_elevation:
        cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.01, location="right")
        cb.set_label("elevation (m)")
    fig.suptitle(a.title or f"{name} — {len(cols)} PFLOTRAN columns as built",
                 fontsize=a.font + 1)
    figstyle.check_titles(fig)
    out = Path(a.out) if a.out else rd / "sampling_design.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)



def _treatment_words(t):
    """A treatment dict as short words: 'wt 2 m · loam · storms 500 mm'."""
    if not isinstance(t, dict):
        return ""
    parts = []
    for k, v in t.items():
        if k == "water_table_m":
            parts.append(f"wt {v:g} m")
        elif k == "soil_depth_m":
            parts.append(f"soil {v:g} m")
        elif k == "recharge_mm_yr":
            parts.append(f"{v:g} mm/yr")
        elif k == "rain" and isinstance(v, dict):
            parts.append(f"{v.get('fill')} {v.get('mm_yr'):g} mm")
        else:
            parts.append(str(v))
    return " · ".join(parts)


def render_sweep(a, rd, cj):
    """The design figure for a controlled sweep — no place, so no map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cols = cj["columns"]
    design = cj.get("sampling_design") or {}
    factors = [f.get("name") for f in (design.get("factors") or [])]
    fixed = design.get("held_fixed") or {}
    n = len(cols)

    def num(c, k):
        v = c.get(k)
        return float(v) if isinstance(v, (int, float)) else np.nan

    wt = np.array([num(c, "water_table_m") for c in cols])
    dom = np.array([num(c, "depth_m") for c in cols])
    k = figstyle.manuscript(a.width, a.font)
    fig = plt.figure(figsize=(a.width, a.width * 0.72), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    lw_line = max(0.5, 2.0 * k)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colour = [cmap(i % cmap.N) for i in range(n)]

    def tag(ax, letter, text, note=""):
        ax.set_title(f"({letter}) {text}", loc="left")
        if note:
            ax.set_title(note, loc="right", fontsize=a.font - 1, color="0.35")

    # (a) THE DESIGN, one row per column ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    heads = ["column"] + factors
    body = []
    for c in cols:
        t = c.get("treatment") or {}
        row = [str(c.get("id", "")).replace("col_", "")]
        for f in factors:
            row.append(_treatment_words({f: t.get(f)}) if f in t else "")
        body.append(row)
    tbl = ax.table(cellText=body, colLabels=heads, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(max(4, a.font - 2))
    tbl.scale(1, max(0.6, min(1.3, 14.0 / max(n, 1))))
    for (r, cidx), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_text_props(weight="bold")
        elif cidx == 0:
            cell.set_facecolor(matplotlib.colors.to_rgba(colour[r - 1], alpha=0.35))
    held = " · ".join(f"{kk} {_treatment_words({kk: v}) if kk in ('water_table_m','soil_depth_m','recharge_mm_yr','rain') else v}"
                      for kk, v in fixed.items())
    tag(ax, "a", "the design", f"{n} columns")

    # (b) THE COLUMNS ON THE CONUS2 LADDER ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for b in CONUS2_BOUNDS_M:
        ax.axhline(b, color="0.88", lw=max(0.4, 1.2 * k), zorder=1)
    top = 0.05
    cap = 0.18                                            # half-width of an end tick, in x units
    for i in range(n):
        if np.isfinite(dom[i]):
            # the column: surface (top) to its domain bottom, with a cap at
            # each end so the reader sees exactly where it stops
            ax.plot([i, i], [top, dom[i]], color=colour[i], lw=lw_line * 1.3,
                    solid_capstyle="butt", zorder=3)
            ax.plot([i - cap, i + cap], [dom[i], dom[i]], color=colour[i],
                    lw=lw_line * 1.3, zorder=3)
        if np.isfinite(wt[i]):
            # the water table it is anchored to
            ax.plot([i], [max(wt[i], top)], marker="o", color=colour[i], mec="k",
                    mew=max(0.3, 1.4 * k), ms=6 * max(0.6, 20 * k / 4), zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(max(np.nanmax(dom) if np.isfinite(dom).any() else 10.0, 10.0) * 1.25, top)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(c.get("id", "")).replace("col_", "") for c in cols],
                       fontsize=max(4, a.font - 3))
    ax.set_xlabel("column"); ax.set_ylabel("depth below surface (m)")
    ax.plot([], [], marker="o", ls="", color="0.4", mec="k",
            label="water table (anchor)")
    ax.plot([], [], color="0.4", lw=lw_line * 1.3, label="column to its bottom")
    ax.legend(fontsize=max(4, a.font - 3), loc="lower left", framealpha=0.9)
    tag(ax, "b", "column depth and water table")

    # (c) WATER IN ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    trans = [c for c in cols if c.get("transient")]
    if trans:
        # the written series, as the server left it on the row (a sweep row
        # keeps precipitation_mm_day — the design is its only source)
        drew = 0
        for i, c in enumerate(cols):
            y = c.get("precipitation_mm_day")
            if not y:
                continue
            y = np.array([float(v) for v in y])
            ax.plot(np.arange(1, len(y) + 1), y, color=colour[i], lw=lw_line * 0.8,
                    label=str(c.get("id", "")).replace("col_", ""))
            drew += 1
        ax.set_xlabel("day of year"); ax.set_ylabel("rain (mm/day)")
        ax.set_xlim(1, 365)
        tag(ax, "c", "written rain", f"{drew} series")
    else:
        rate = np.array([num(c, "recharge_mm_yr") for c in cols])
        ax.bar(range(n), rate, color=colour, edgecolor="k", lw=max(0.3, 1.0 * k))
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(c.get("id", "")).replace("col_", "") for c in cols],
                           fontsize=max(4, a.font - 3))
        ax.set_xlabel("column"); ax.set_ylabel("steady recharge (mm/yr)")
        tag(ax, "c", "water in — steady")

    # (d) THE SOILS: retention curves of every class in the design ───────────
    ax = fig.add_subplot(gs[1, 1])
    seen = {}
    for c in cols:
        for L in ((c.get("subsurface_profile") or {}).get("layers") or []):
            m = L.get("material")
            if m and m not in seen:
                seen[m] = L
    h = np.logspace(-2, 3, 200)                        # suction head, m
    for j, (m, L) in enumerate(seen.items()):
        alpha, nn, sres = float(L["vg_alpha"]), float(L["vg_n"]), float(L["sres"])
        S = sres + (1.0 - sres) / (1.0 + (alpha * h) ** nn) ** (1.0 - 1.0 / nn)
        ax.plot(h, S, lw=lw_line, label=f"{m}  (Ks {float(L['permeability_z'])*24:.2g} m/d)")
    ax.set_xscale("log")
    ax.set_xlabel("suction head (m)"); ax.set_ylabel("saturation")
    ax.set_ylim(0, 1.02)
    if seen:
        ax.legend(fontsize=max(4, a.font - 3), loc="lower left")
    tag(ax, "d", "the soils", f"{len(seen)} class(es)")

    top_line = a.title or f"controlled sweep — {n} PFLOTRAN columns as built"
    fig.suptitle(top_line + (f"\nheld fixed: {held}" if held else ""),
                 fontsize=a.font + 1)
    figstyle.check_titles(fig)
    out = Path(a.out) if a.out else rd / "sampling_design.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)

def main():
    ap = argparse.ArgumentParser(description="PFLOTRAN sampling-design figure")
    ap.add_argument("--run-dir", required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    a = ap.parse_args()
    print(render(a))


if __name__ == "__main__":
    main()
