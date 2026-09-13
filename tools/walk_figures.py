#!/usr/bin/env python3
"""Two figures from a walk's own records, measurements only.

    python tools/walk_figures.py WALK_DIR [--against TWIN_DIR] [--out DIR]

  walk_design.png   what the walk was set up with: each column's depth, its
                    starting water table and the sink's datum and band
                    (walk.json), and the gauge recession the timescale came
                    from (walk.json sink.tau_cross_check).
  walk_year.png     what the year did: each column's water table from start
                    to end against its datum (walk_summary.json), the sink's
                    e-folding time against the dial (walk_report.json), the
                    monthly forward flux at the highest column for this walk
                    and the twin, and ELM's year residual per column for both.

Reads only what the walk wrote (walk.json, walk_summary.json, walk_report.json
from tools/walk_report.py) and the ELM source run's elevation per column.
Draws nothing it did not read; labels state quantities, not conclusions.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "mcp" / "elm-mcp" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "figure.dpi": 150})
INK, MUT, TEAL, RUST, GREY = "#1d2e3a", "#68798a", "#0e7490", "#b45309", "#c9d2d8"


def load(walk_dir):
    w = Path(walk_dir)
    walk = json.loads((w / "walk.json").read_text())
    summary = json.loads((w / "walk_summary.json").read_text())
    rep_p = w / "walk_report.json"
    report = json.loads(rep_p.read_text()) if rep_p.is_file() else None
    return walk, summary, report


def elevations(walk):
    """Column elevation from the ELM source run's own finding, else None."""
    src = walk.get("elm_run_source")
    try:
        inv = json.loads((Path(src) / "04_analysis" / "investigation.json").read_text())
        f = [x for x in inv["findings"] if x["id"] == "partition_vs_elevation"][0]["result"]
        return dict(zip(f["entity"], f["elevation_m"]))
    except Exception:                                             # noqa: BLE001
        return {}


def ordered(walk, elev):
    cols = walk["columns"]
    return sorted(cols, key=lambda c: (elev.get(c["id"], 1e9), c["id"]))


def year_residual(elm):
    """ELM's year residual P - ET - runoff - dTWS (mm): the walk's own record
    when the walk wrote one, else the same sum over the daily series it
    kept (older walks predate the balance record; the two agree to 0.1 mm
    where both exist, checked on walk_naches_sink_wt col_17: 119.5)."""
    b = (elm or {}).get("balance") or {}
    y = b.get("year") or {}
    if isinstance(y.get("residual_mm"), (int, float)):
        return y["residual_mm"], y.get("P_mm")
    elm = elm or {}
    need = ("RAIN", "SNOW", "QOVER", "QDRAI", "QVEGE", "QVEGT", "QSOIL", "TWS")
    if not all(isinstance(elm.get(v), list) and elm.get(v) for v in need):
        return None, None
    tot = lambda v: float(sum(x for x in elm[v] if isinstance(x, (int, float))))  # noqa: E731
    P = tot("RAIN") + tot("SNOW")
    ET = tot("QVEGE") + tot("QVEGT") + tot("QSOIL")
    tws = [x for x in elm["TWS"] if isinstance(x, (int, float))]
    dTWS = tws[-1] - tws[0]
    return round(P - ET - tot("QOVER") - tot("QDRAI") - dTWS, 1), round(P, 1)


BALANCE_VARS = ("RAIN", "SNOW", "QOVER", "QDRAI", "QCHARGE",
                "QVEGE", "QVEGT", "QSOIL", "TWS")


def complete_elm_daily(walk_dir, walk, summary):
    """A walk written before the balance record kept only ZWT, QDRAI and
    QCHARGE per day. The rest is still in its ELM case histories, so read
    them the way tools/walk_job.py does, once, into
    WALK_DIR/figures/elm_daily_recomputed.json, and merge."""
    elm = summary.get("elm_daily") or {}
    missing = [cid for cid, d in elm.items()
               if isinstance(d, dict) and not all(isinstance(d.get(v), list) for v in BALANCE_VARS)]
    if not missing:
        return elm
    cache = Path(walk_dir) / "figures" / "elm_daily_recomputed.json"
    if cache.is_file():
        extra = json.loads(cache.read_text())
    else:
        import extract as elm_extract                          # noqa: E402
        extra = {}
        cases = {c["id"]: c["elm_case_dir"] for c in walk["columns"]}
        for cid in missing:
            try:
                data, _m = elm_extract.extract_column(
                    cases[cid], variables=["ZWT", *BALANCE_VARS], spinup_days=0)
            except Exception as e:                            # noqa: BLE001
                extra[cid] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue
            vs = data.get("variables") or {}
            extra[cid] = {"dates": data.get("dates"),
                          **{v: (vs.get(v) or {}).get("values") for v in ("ZWT", *BALANCE_VARS)}}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(extra))
        print(f"read {len(extra)} column(s) of ELM history for {walk_dir}; cached at {cache}")
    out = dict(elm)
    for cid, d in extra.items():
        if "error" not in d:
            out[cid] = {**elm.get(cid, {}), **d}
    return out


def monthly_means(elm, var):
    """Monthly mean of a daily series (mm/day), by calendar month."""
    dates, vals = elm.get("dates") or [], elm.get(var) or []
    acc = {}
    for d, v in zip(dates, vals):
        if isinstance(v, (int, float)):
            acc.setdefault(d[5:7], []).append(float(v))
    return [np.mean(acc[m]) if m in acc else np.nan
            for m in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")]


def xlabels(cols, elev):
    return [f"{elev[c['id']]:.0f}" if c["id"] in elev else c["id"] for c in cols]


def design_figure(walk, elev, out):
    cols = ordered(walk, elev)
    x = np.arange(len(cols))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [3, 1.35]})
    ax = axes[0]
    depth = [c["depth_m"] for c in cols]
    datum = [c["sink_datum_applied_m"] for c in cols]
    band = [c.get("sink_band_m", 1.0) for c in cols]
    wt0 = [c["water_table_m"] for c in cols]
    ax.bar(x, depth, width=0.62, color=GREY, edgecolor="none", label="column depth")
    for i, (d, b) in enumerate(zip(datum, band)):
        ax.add_patch(plt.Rectangle((i - 0.31, d - b), 0.62, b, color=TEAL, alpha=0.9))
    ax.scatter(x, wt0, marker="_", s=420, color=RUST, linewidths=3, label="starting water table", zorder=5)
    ax.scatter([], [], marker="s", s=110, color=TEAL, label="seepage band above the datum")
    ax.set_ylim(max(depth) * 1.04, 0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels(cols, elev), rotation=90)
    ax.set_xlabel("column, by elevation (m)")
    ax.set_ylabel("depth below the surface (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
    ax.set_title("(a) each column's depth, starting water table and sink datum")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    cc = (walk.get("sink") or {}).get("tau_cross_check") or {}
    pooled = cc.get("pooled") or {}
    segs = []
    for g in cc.get("gauges") or []:
        for sgm in g.get("segments") or []:
            segs.append((sgm.get("n_days"), sgm.get("tau_days"), g.get("id")))
    dial = (walk.get("sink") or {}).get("tau_days")
    if segs:
        ax.scatter([s[0] for s in segs], [s[1] for s in segs], s=140, color=INK, zorder=5,
                   label="one falling segment")
    if pooled.get("tau_days"):
        ax.axhspan(pooled.get("tau_lo_days", pooled["tau_days"]),
                   pooled.get("tau_hi_days", pooled["tau_days"]), color=TEAL, alpha=0.18, lw=0)
        ax.axhline(pooled["tau_days"], color=TEAL, lw=2.2,
                   label=f"pooled {pooled['tau_days']:.1f} d ({pooled.get('n_days')} days)")
    if dial:
        ax.axhline(dial, color=RUST, lw=2.2, ls="--", label=f"dial {dial} d")
    ax.set_xlabel("segment length (days)")
    ax.set_ylabel("e-folding time of gauge flow (days)")
    ax.set_ylim(0, max([s[1] for s in segs] + [dial or 0]) * 1.25)
    ax.legend(loc="upper left", frameon=False)
    gname = ", ".join(sorted({s[2] for s in segs})) or "no gauge"
    ax.set_title(f"(b) recession at {gname}")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def year_figure(walk, summary, report, twin, elev, out):
    cols = ordered(walk, elev)
    x = np.arange(len(cols))
    traj = summary.get("solved_trajectories") or {}
    elm = summary.get("elm_daily") or {}
    telm = (twin[1].get("elm_daily") if twin else None) or {}
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    ax = axes[0, 0]
    for i, c in enumerate(cols):
        t = traj.get(c["id"]) or {}
        wts = t.get("wt_solved_m") or []
        if not wts:
            continue
        ax.plot([i, i], [c["water_table_m"], wts[-1]], color=INK, lw=2.2, zorder=3)
        ax.scatter([i], [c["water_table_m"]], marker="_", s=380, color=RUST, linewidths=3, zorder=4)
        ax.scatter([i], [wts[-1]], marker="v", s=90, color=INK, zorder=5)
        ax.scatter([i], [c["sink_datum_applied_m"]], marker="_", s=380, color=TEAL, linewidths=3, zorder=4)
    ax.scatter([], [], marker="_", s=200, color=RUST, linewidths=3, label="start")
    ax.scatter([], [], marker="v", s=80, color=INK, label="end of the year")
    ax.scatter([], [], marker="_", s=200, color=TEAL, linewidths=3, label="sink datum")
    ax.set_ylim(max(c["depth_m"] for c in cols) * 1.04, 0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels(cols, elev), rotation=90)
    ax.set_xlabel("column, by elevation (m)")
    ax.set_ylabel("water table depth (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
    ax.set_title("(a) water table, start to end")

    ax = axes[0, 1]
    dial = None
    if report:
        rows = report.get("columns") or {}
        ef = [(i, rows[c["id"]].get("outflow_efold_days")) for i, c in enumerate(cols)
              if c["id"] in rows and (rows[c["id"]].get("outflow_efold_days") or 0) > 0]
        dial = next((rows[c["id"]].get("tau_dial_days") for c in cols if c["id"] in rows), None)
        if ef:
            ax.scatter([e[0] for e in ef], [e[1] for e in ef], s=120, color=INK, zorder=5,
                       label="sink outflow, first two windows")
    if dial:
        ax.axhline(dial, color=RUST, lw=2.2, ls="--", label=f"dial {dial} d")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels(cols, elev), rotation=90)
    ax.set_xlabel("column, by elevation (m)")
    ax.set_ylabel("e-folding time (days)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(b) outflow e-folding time against the dial")

    ax = axes[1, 0]
    top = cols[-1]["id"]
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    m_this = monthly_means(elm.get(top) or {}, "QCHARGE")
    w = 0.38
    if telm.get(top):
        m_twin = monthly_means(telm.get(top) or {}, "QCHARGE")
        ax.bar(np.arange(12) - w / 2, m_twin, w, color=GREY, edgecolor=MUT,
               label=f"twin: {twin[0]}")
        ax.bar(np.arange(12) + w / 2, m_this, w, color=TEAL, label=f"this walk: {walk.get('return_leg', 'wt+profile')}")
    else:
        ax.bar(np.arange(12), m_this, 0.6, color=TEAL, label="this walk")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(months)
    ax.set_ylabel("recharge handed to PFLOTRAN (mm/day)")
    ax.set_title(f"(c) monthly forward flux at the highest column, "
                 f"{elev.get(top, 0):.0f} m")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1, 1]
    r_this = [year_residual(elm.get(c["id"]))[0] for c in cols]
    if telm:
        r_twin = [year_residual(telm.get(c["id"]))[0] for c in cols]
        ax.bar(x - w / 2, [v if v is not None else np.nan for v in r_twin], w, color=GREY,
               edgecolor=MUT, label=f"twin: {twin[0]}")
        ax.bar(x + w / 2, [v if v is not None else np.nan for v in r_this], w, color=TEAL,
               label=f"this walk: {walk.get('return_leg', 'wt+profile')}")
    else:
        ax.bar(x, [v if v is not None else np.nan for v in r_this], 0.6, color=TEAL, label="this walk")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels(cols, elev), rotation=90)
    ax.set_xlabel("column, by elevation (m)")
    ax.set_ylabel("residual (mm); negative = water appeared")
    ax.set_title("(d) ELM year residual, P - ET - runoff - dTWS")
    ax.legend(loc="lower left", frameon=False)
    for ax in axes.flat:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walk_dir")
    ap.add_argument("--against", default=None, help="twin walk dir, drawn beside this one in (c) and (d)")
    ap.add_argument("--out", default=None, help="output dir (default WALK_DIR/figures)")
    a = ap.parse_args()
    walk, summary, report = load(a.walk_dir)
    if report is None:
        sys.exit(f"{a.walk_dir}/walk_report.json is missing; run tools/walk_report.py --json first")
    twin = None
    if a.against:
        tw, ts, _ = load(a.against)
        ts = dict(ts, elm_daily=complete_elm_daily(a.against, tw, ts))
        twin = (tw.get("return_leg") or "wt+profile", ts)
    summary = dict(summary, elm_daily=complete_elm_daily(a.walk_dir, walk, summary))
    elev = elevations(walk)
    out = Path(a.out or Path(a.walk_dir) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    design_figure(walk, elev, out / "walk_design.png")
    year_figure(walk, summary, report, twin, elev, out / "walk_year.png")
    print(f"written {out / 'walk_design.png'} and {out / 'walk_year.png'}")


if __name__ == "__main__":
    main()
