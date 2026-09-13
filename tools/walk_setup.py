#!/usr/bin/env python3
"""Build a walk directory: cloned ELM cases + PFLOTRAN spin decks + walk.json.

    python tools/walk_setup.py --elm-run RUN_DIR --pf-run RUN_DIR \
        --out WALK_DIR (--months 1 | --days 7) [--year YYYY] \
        [--bottom none] [--forward qcharge] [--negative-forward clip] \
        [--sink-datum hand|conus2 --sink-tau-days D [--sink-band-m B] \
         [--sink-sy S]]

Login-node safe: clones cases (--keepexe, no compile), writes decks, runs
NOTHING. The job that runs the walk is tools/walk_job.py, submitted with
permission.

Where the pieces come from:
  --elm-run   a finished ELM leg whose built cases (01_inputs/
              built_cases.json) are cloned per column; each clone inherits
              the leg's finidat (the chain's stamped state) and fsurdat,
              and is configured to stop at the FIRST window.
  --pf-run    the PFLOTRAN leg those columns coupled to: its columns.json
              carries each column's daily flux (the spin rate is its mean)
              and solved anchor, and its site files feed the deck builder's
              join. The spin decks are built steady at that mean with
              checkpoint=True — the walk's clock starts where they end.
  The spin rate is the FORWARD variable's mean (--forward, QDRAI or
  QCHARGE) over the ELM run's own 03_results/extracted.json after the
  negative-day policy; the lateral sink (--sink-datum) rides each column
  as the contract keys of docs/coupling/lateral_sink_design.md.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import walk_lib                                        # noqa: E402
import elm_wrapper as ew                               # noqa: E402

# THE SINK'S TWO HELPERS ARE OPTIONAL IMPORTS: a walk without --sink-datum
# must still build on a checkout that lacks them.
try:
    import drainage_datum                              # noqa: E402
except ImportError:
    drainage_datum = None
try:
    import recession_tau                               # noqa: E402
except ImportError:
    recession_tau = None


def _sh(args, cwd):
    r = subprocess.run(args, cwd=cwd, env=ew._cime_env(),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args))} in {cwd}:\n"
                           f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


def _clone_elm(ref, dest, first_window):
    cimeroot = _sh(["./xmlquery", "CIMEROOT", "--value"], cwd=ref).strip()
    _sh([str(Path(cimeroot) / "scripts" / "create_clone"),
         "--case", str(dest), "--clone", str(ref), "--keepexe"], cwd=ref)
    for kv in (f"STOP_N={first_window['stop_n']}",
               f"STOP_OPTION={first_window['stop_option']}",
               f"REST_N={first_window['stop_n']}",
               f"REST_OPTION={first_window['stop_option']}",
               "CONTINUE_RUN=FALSE"):
        _sh(["./xmlchange", kv], cwd=dest)
    _sh(["./case.setup"], cwd=dest)
    return dest


def _own_finidat(dest):
    """Copy the source finidat into the clone and point user_nl_elm at it.

    Window 1 replays the starting file's relaxation as January weather:
    measured at Naches (job 775060), the UNCOUPLED warm start burst
    648-1795 mm/day of drainage — against 2.5 mm/day annual means — into
    sealed columns and killed the solver at day 22. walk_job therefore
    stamps PFLOTRAN's spun state onto the starting file before the first
    window (the same return leg every later window gets), and the file it
    stamps must be the CLONE'S OWN COPY: the source run's finidat is a
    finished study's record, and is never written."""
    unl = dest / "user_nl_elm"
    text = unl.read_text()
    m = re.search(r"finidat\s*=\s*'([^']+)'", text)
    if not m:
        raise RuntimeError(f"{dest}: user_nl_elm names no finidat — the walk "
                           f"stamps the starting state and needs one")
    src = Path(m.group(1))
    own = dest / f"finidat_{src.name}"
    shutil.copy(src, own)
    unl.write_text(text.replace(m.group(0), f"finidat = '{own}'"))
    # the run-dir namelists were generated at case.setup, BEFORE the
    # rewrite — without this the run still reads the source file (the
    # slice-proof lesson, job 774959)
    _sh(["./preview_namelists"], cwd=dest)
    return own


def _fsurdat_of(case_dir):
    m = re.search(r"fsurdat\s*=\s*'([^']+)'",
                  (Path(case_dir) / "user_nl_elm").read_text())
    if not m:
        raise RuntimeError(f"{case_dir}: user_nl_elm names no fsurdat — "
                           f"apply_profile needs the column's own surfdata")
    return m.group(1)


def _read_elm_extract(elm_run):
    """The ELM source run's own daily record, {cid: {dates, variables}}."""
    p = Path(elm_run) / "03_results" / "extracted.json"
    if not p.is_file():
        raise SystemExit(f"{elm_run}: no 03_results/extracted.json; the spin "
                         f"rate is the forward variable's mean over that "
                         f"record, so the elm run must have finished its "
                         f"extract")
    return (json.loads(p.read_text()) or {}).get("data") or {}


def _spin_rate(cid, elm_data, col, forward_var, policy):
    """The steady rate a column spins at, mm/yr, with its source and clip.

    The mean of the forward variable over the ELM source run's own extract
    after the negative policy (design, section 5): the pf run's
    daily_flux_mm_day is QDRAI whatever the walk forwards. A column whose
    extract has no series for the variable spins at the pf run's QDRAI
    mean and is said to; walk_job records its departure at window 1.
    """
    vals = walk_lib.forward_values(elm_data.get(cid), forward_var)
    if vals:
        vals, clipped = walk_lib.apply_negative_policy(vals, policy)
        return (round(sum(vals) / len(vals) * 365.0, 4),
                f"mean {forward_var} over the ELM source run's "
                f"03_results/extracted.json ({len(vals)} days) after "
                f"negative_forward={policy}", clipped)
    flux = col.get("daily_flux_mm_day") or []
    if not flux:
        raise SystemExit(f"{cid}: neither the ELM source run's extract "
                         f"({forward_var}) nor the pf run's column "
                         f"(daily_flux_mm_day) carries a series to spin at")
    return (round(sum(flux) / len(flux) * 365.0, 4),
            f"mean of the pf run's daily_flux_mm_day (QDRAI): the ELM "
            f"source run's extract has no {forward_var} for this column, "
            f"so the walk will record its departure at window 1", 0.0)


def _sink_dials(a):
    """The sink dials as given (walk.json's `sink`), refused when they clash."""
    if a.sink_datum == "none":
        for name, v in (("--sink-tau-days", a.sink_tau_days),
                        ("--sink-sy", a.sink_sy)):
            if v is not None:
                sys.exit(f"{name} {v} given with --sink-datum none: the dial "
                         f"would do nothing; name a datum or drop it")
        return {"datum": "none", "tau_days": None, "band_m": a.sink_band_m,
                "sy": None}
    if a.sink_tau_days is None:
        sys.exit(f"--sink-datum {a.sink_datum} needs --sink-tau-days, the "
                 f"basin's recession timescale (tools/recession_tau.py; the "
                 f"reader pass gave Naches 30, Brandywine 35); there is no "
                 f"hidden default")
    for name, v, top in (("--sink-tau-days", a.sink_tau_days, None),
                         ("--sink-band-m", a.sink_band_m, None),
                         ("--sink-sy", a.sink_sy, 1.0)):
        if v is not None and (v <= 0 or (top is not None and v > top)):
            sys.exit(f"{name} {v} is out of range "
                     f"({'0 < x <= 1' if top else 'x > 0'})")
    return {"datum": a.sink_datum, "tau_days": a.sink_tau_days,
            "band_m": a.sink_band_m, "sy": a.sink_sy,
            "pick": getattr(a, "sink_pick", "smallest")}


def _sink_sy(cid, dial, layer, applied_m):
    """(sink_sy, sink_sy_source): the dial, else min(0.2, porosity at the datum)."""
    if dial is not None:
        return float(dial), "--sink-sy"
    phi = (layer or {}).get("porosity")
    if not isinstance(phi, (int, float)):
        sys.exit(f"{cid}: no CONUS2 layer of its profile holds the applied "
                 f"datum depth {applied_m} m, so the default Sy "
                 f"min({walk_lib.SY_CAP}, porosity) has nothing to read; "
                 f"pass --sink-sy")
    return (round(min(walk_lib.SY_CAP, float(phi)), 6),
            f"min({walk_lib.SY_CAP}, CONUS2 porosity {phi} of the layer at "
            f"{applied_m} m); {walk_lib.SY_CAP} is ELM's own aquifer "
            f"specific yield")


def _sink_column(cid, datum, sink, depth_m, profile, water_table_m):
    """Every contract key of one column's sink (design, sections 3 and 4)."""
    if not isinstance(datum.get("sink_datum_m"), (int, float)):
        sys.exit(f"{cid}: drainage_datum.compute gave no sink_datum_m (it "
                 f"gave {sorted(datum)})")
    applied, clipped, raised = walk_lib.datum_in_domain(
        datum["sink_datum_m"], depth_m, sink["band_m"])
    layer = walk_lib.layer_at_depth(profile, applied)
    sy, sy_src = _sink_sy(cid, sink["sy"], layer, applied)
    path_km = datum.get("hand_path_km")
    head = (applied - float(water_table_m)
            if isinstance(water_table_m, (int, float)) else None)
    rec = {k: datum.get(k) for k in walk_lib.DATUM_MODULE_KEYS}
    rec.update({
        "sink_datum_source": datum.get("sink_datum_source") or sink["datum"],
        "sink_datum_m": round(float(datum["sink_datum_m"]), 4),
        "sink_datum_applied_m": applied,
        "sink_datum_clipped_to_domain": clipped,
        "sink_datum_raised_to_band": raised,
        "sink_band_m": float(sink["band_m"]),
        "sink_sy": sy,
        "sink_sy_source": sy_src,
        "sink_tau_days": float(sink["tau_days"]),
        "sink_tau_source": "--sink-tau-days",
        "sink_conductance_m": walk_lib.sink_conductance_m(
            sy, sink["tau_days"], sink["band_m"]),
        # the Dupuit timescale the column's own material implies at its
        # starting head, for inspection beside the recession the sink uses
        "sink_implied_tau_days": walk_lib.implied_tau_days(
            sy, (layer or {}).get("permeability_z"),
            (path_km * 1000.0 if isinstance(path_km, (int, float))
             else None), head),
    })
    return rec


def _datum_report(elm_run, source, pick="smallest"):
    """drainage_datum's report for the ELM source run: (rows by id, basin facts).

    THE ELM RUN, NOT THE PF RUN: the module reads STD_ELEV off each
    column's warm-start surfdata (warmstart/surfdata_<cid>.nc, which only
    the ELM leg writes) and, for conus2, samples the run's own raster when
    columns.json carries no water_table_m. The pf run's water_table_m is
    the driving model's solved mean, not CONUS2's, so it must not be the
    one read. The report's rows sit under "columns"; the basin-level facts
    (the gauge check that fixed the stream threshold, the datum rule) are
    kept beside the dials so a reader can see what A was and why.
    """
    if drainage_datum is None:
        sys.exit(f"--sink-datum {source} needs tools/drainage_datum.py, "
                 f"which this checkout cannot import")
    # The dial is additive: the default asks compute() for nothing new.
    kw = {} if pick in (None, "smallest") else {"pick": pick}
    report = drainage_datum.compute(str(elm_run), source, **kw) or {}
    rows = report.get("columns") if isinstance(report, dict) else None
    by_id = ({r["id"]: r for r in rows} if isinstance(rows, list)
             else dict(report))
    facts = {k: report.get(k) for k in (
        "run_dir", "basin", "surfdata_files", "gauge_check",
        "hand_threshold_km2", "std_elev_floor", "sink_datum_rule",
        "n_columns", "n_columns_without_stream")}
    return by_id, json.loads(json.dumps(facts, default=str))


def _sink_columns(sink, elm_run, pf_run, pf_cols, rows, site_data):
    """{cid: contract keys} for every walked column, {} without a sink."""
    if sink["datum"] == "none":
        return {}
    datums, sink["datum_report"] = _datum_report(elm_run, sink["datum"],
                                                 sink.get("pick") or "smallest")
    joined = site_data.join(str(pf_run), [{"id": c["id"], "lat": c.get("lat"),
                                           "lon": c.get("lon")}
                                          for c in pf_cols])
    prof_of = {c["id"]: c.get("subsurface_profile")
               for c in joined["columns"]}
    out = {}
    for c in pf_cols:
        cid = c["id"]
        if cid not in datums:
            sys.exit(f"{cid}: drainage_datum.compute({sink['datum']!r}) "
                     f"returned no row for it (it has {sorted(datums)})")
        out[cid] = _sink_column(cid, datums[cid], sink,
                                float(rows[cid]["depth_m"]),
                                prof_of.get(cid), c.get("water_table_m"))
    return out


def _tau_cross_check(elm_run):
    """tools/recession_tau's own fit of the run's gauges, for inspection.

    The dial is what the conductance used; this sits beside it so a reader
    can see how far the two stand apart. The ELM run is read because its
    experiment.json carries the forcing the rain-day exclusion needs.
    Nothing here can stop the walk.
    """
    if recession_tau is None:
        return {"status": "not computed: tools/recession_tau.py is not "
                          "importable on this checkout"}
    try:
        got = recession_tau.compute(str(elm_run), exclude=[])
    except Exception as e:                              # noqa: BLE001
        return {"status": f"not computed: {type(e).__name__}: {e}"[:300]}
    return json.loads(json.dumps(
        {"status": "computed with exclude=[] (no regulated gauge named); "
                   "inspection only, the dial is what was used",
         **(got or {})}, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elm-run", required=True)
    ap.add_argument("--pf-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--year", type=int, default=None)
    # THE BOTTOM OF THE WALK'S PFLOTRAN COLUMNS. "water_table" holds the
    # pressure at the bottom fixed to the starting anchor — which PINS the
    # solved water table there (measured 2026-08-25: 100x the recharge moved
    # it 11 cm), so the walk exchanges a number that cannot change. "none"
    # seals the bottom: no water passes, and the level must genuinely rise
    # and fall with the season's drainage. The SPIN decks stay anchored
    # either way — a sealed column fed a steady inflow never settles, so the
    # warm-up needs the anchor; the seal applies from the first window on.
    ap.add_argument("--bottom", choices=("water_table", "none"),
                    default="water_table")
    # WALK A SUBSET, NAMED. A coupled run may carry columns whose driver
    # sends exactly nothing (Naches: 9 of 18 with QDRAI identically zero at
    # water tables >= 12 m) — a walk window there re-proves a known flat
    # line. Names must exist in the pf run; an unknown one is refused, not
    # skipped, so a typo cannot silently shrink the study.
    ap.add_argument("--columns", default=None,
                    help="comma-separated column ids to walk (default: all)")
    # THE FORWARD FLUX. "qdrai" is ELM's sub-surface drainage, what left its
    # soil column downward (the walk's meaning until now, kept as the
    # default so old walk dirs read the same). "qcharge" is ELM's recharge
    # to the aquifer, positive INTO it (SoilWaterMovementMod.F90:759), which
    # goes negative on days the aquifer feeds the soil; --negative-forward
    # says what such a day becomes: "clip" zeroes it (a negative top flux
    # would extract water at an unsaturated face, double-counting the
    # capillary rise ELM already resolved) and records the clipped total
    # per window; "pass" hands it to the deck for the experiment that
    # wants it.
    ap.add_argument("--forward", choices=("qdrai", "qcharge"),
                    default="qdrai")
    ap.add_argument("--negative-forward", choices=("clip", "pass"),
                    default="clip")
    ap.add_argument("--return", dest="return_leg",
                    choices=walk_lib.RETURN_LEGS, default="wt+profile",
                    help="what PFLOTRAN hands back to ELM after each window: "
                         "the water table and the saturation profile (today's "
                         "behaviour), or the water table alone. The window-0 "
                         "pre-stamp keeps the profile either way; it cures "
                         "the start-up burst. Stamping the profile every "
                         "window was measured to CREATE water (2026-09-13).")
    # THE LATERAL SINK (docs/coupling/lateral_sink_design.md). A sealed
    # column has nowhere to put the drainage it receives; a thin band of
    # side faces above a drainage datum leaks outward while the water table
    # stands above the datum, at a conductance matched to the basin's
    # recession. The datum is read by tools/drainage_datum.py: "hand" is
    # height above the nearest drainage from the 1-km TOPO, "conus2" the
    # CONUS2 water-table depth the column already carries. The recession
    # timescale is REQUIRED with a sink (no hidden default); Sy defaults to
    # min(0.2, the porosity at the datum). The spin decks stay anchored and
    # sink-free; the sink applies from the first window on.
    ap.add_argument("--sink-datum", choices=("none", "hand", "conus2"),
                    default="none")
    ap.add_argument("--sink-tau-days", type=float, default=None,
                    help="recession timescale, days (required with a sink)")
    ap.add_argument("--sink-band-m", type=float, default=1.0)
    ap.add_argument("--sink-sy", type=float, default=None,
                    help="specific yield override (default: min(0.2, the "
                         "porosity at the datum))")
    ap.add_argument("--sink-pick", choices=("smallest", "largest"),
                    default="smallest",
                    help="which stream threshold wins a tied gauge check: "
                         "the densest network (smallest) or the sparsest "
                         "(largest); stream sets nest, so smallest is "
                         "always the first candidate")
    a = ap.parse_args()
    sink = _sink_dials(a)
    forward_var = a.forward.upper()

    elm_run = Path(a.elm_run).resolve()
    pf_run = Path(a.pf_run).resolve()
    out = Path(a.out).resolve()
    # A WALK DIR IS BUILT ONCE. Re-running setup into a live walk would
    # rewrite the spin decks under a mid-walk job's feet and then die on the
    # first existing ELM clone anyway — refuse up front instead.
    if (out / "walk.json").exists() or (out / "elm").exists():
        sys.exit(f"{out} already holds a walk — pick a fresh --out (a walk "
                 f"dir is built once; the job resumes from its state file)")
    out.mkdir(parents=True, exist_ok=True)

    built = walk_lib.read_built_cases(elm_run)
    pf_cols = json.loads((pf_run / "columns.json").read_text())
    pf_cols = pf_cols.get("columns", pf_cols) \
        if isinstance(pf_cols, dict) else pf_cols
    if a.columns:
        want = [w.strip() for w in a.columns.split(",") if w.strip()]
        have = {c["id"] for c in pf_cols}
        missing = [w for w in want if w not in have]
        if missing:
            sys.exit(f"--columns names {missing} but the pf run carries "
                     f"{sorted(have)}")
        pf_cols = [c for c in pf_cols if c["id"] in want]

    # The year is the forcing year the reference cases already run
    ref0 = next(iter(built.values()))
    start = _sh(["./xmlquery", "RUN_STARTDATE", "--value"], cwd=ref0).strip()
    year = a.year or int(start[:4])
    windows = walk_lib.window_edges(year, months=a.months, days=a.days)

    # ── PFLOTRAN spin decks: steady at each column's own mean flux ──────
    pf, _cb, _sim, sd = walk_lib.load_pf_server()
    # THE SPIN RATE IS THE FORWARD VARIABLE'S OWN MEAN, read off the ELM
    # source run's extract after the negative policy: the pf run's
    # daily_flux_mm_day is QDRAI whatever the walk forwards.
    elm_data = _read_elm_extract(elm_run)
    spin_cols, spin_src = [], {}
    for c in pf_cols:
        rate, source, clipped = _spin_rate(c["id"], elm_data, c, forward_var,
                                           a.negative_forward)
        spin_src[c["id"]] = {"spin_rate_source": source,
                             "spin_clipped_mm": clipped}
        spin_cols.append({
            "id": c["id"], "lat": c.get("lat"), "lon": c.get("lon"),
            "water_table_m": c.get("water_table_m"),
            "recharge_mm_yr": rate,
            # AN EMPTY LIST, NOT AN ABSENCE — the join fills only fields that
            # are None, so this stops it attaching reception's rain and
            # silently flipping the spin deck TRANSIENT (final_time 11 y, not
            # the steady 20 the walk's clock is anchored to). The server's
            # `if pr:` treats [] as no series, so the steady branch runs at
            # the mean-drainage rate above.
            "precipitation_mm_day": [],
        })
    spin = pf.create_decks_from_columns(
        columns=spin_cols, out_dir=str(out / "pf_spin"),
        site_dir=str(pf_run), checkpoint=True)
    rows = {(r.get("id") or r.get("case_name")): r
            for r in (spin.get("decks")
                      or spin.get("run_plan", {}).get("CONDITIONS_COUPLERS")
                      or [])}
    bad = [cid for cid, r in rows.items() if r.get("status") != "built"
           and r.get("deck_status") != "built"]
    if bad:
        raise SystemExit(f"spin decks not built for {bad}: "
                         f"{[rows[c].get('reason') for c in bad]}")
    # THE CLOCK'S GUARANTEE. t0_y=20.0 below is true ONLY of the steady
    # branch (final_time = years = 20.0); a spin deck that came out
    # transient would checkpoint at t=11 and desynchronize every window.
    wet = [cid for cid, r in rows.items() if r.get("transient")]
    if wet:
        raise SystemExit(
            f"spin deck(s) for {wet} were built TRANSIENT — a rain series "
            f"reached them through the join, so their checkpoints would sit "
            f"at t=11 y, not the t=20 the walk's clock assumes. Nothing may "
            f"proceed on a broken clock.")

    # ── the lateral sink, per column (design, section 4) ────────────────
    sink_cols = _sink_columns(sink, elm_run, pf_run, pf_cols, rows, sd)
    if sink["datum"] != "none":
        sink = {**sink, "tau_source": "--sink-tau-days",
                "conductance_formula": "C = Sy * mu / (tau * rho * g * B)",
                "constants": {"mu_Pa_s": walk_lib.MU_PA_S,
                              "rho_kg_m3": walk_lib.RHO_KG_M3,
                              "g_m_s2": walk_lib.G_M_S2},
                "tau_cross_check": _tau_cross_check(elm_run)}

    # ── ELM clones, configured for the first window ─────────────────────
    cols = []
    for c in pf_cols:
        cid = c["id"]
        if cid not in built:
            raise SystemExit(f"{cid}: the elm run built no case for it")
        dest = _clone_elm(Path(built[cid]), out / "elm" / cid, windows[0])
        finidat = _own_finidat(dest)
        r = rows[cid]
        cols.append({
            "id": cid,
            "elm_case_dir": str(dest),
            "finidat": str(finidat),
            "fsurdat": _fsurdat_of(dest),
            "water_table_m": c.get("water_table_m"),
            "depth_m": r.get("depth_m"),
            "n_cells": r.get("n_cells"),
            "spin_input_file": r.get("input_file"),
            "spin_checkpoint": r.get("restart_file_expected"),
            "spin_rate_mm_yr": next(s["recharge_mm_yr"] for s in spin_cols
                                    if s["id"] == cid),
            **spin_src[cid],
            "lat": c.get("lat"), "lon": c.get("lon"),
            # the sink's contract keys, when a sink was asked for
            **sink_cols.get(cid, {}),
        })

    walk = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "year": year,
        "window": ({"months": a.months} if a.months else {"days": a.days}),
        "n_windows": len(windows),
        # The walk's clock origin: the spin runs 20 years steady (the
        # builder's steady final_time), so its checkpoint carries t=20 y.
        "t0_y": 20.0,
        "pf_bottom": a.bottom,
        "forward_var": forward_var,
        "negative_forward": a.negative_forward,
        "return_leg": a.return_leg,
        "sink": sink,
        "site_dir": str(pf_run),
        "elm_run_source": str(elm_run),
        "columns": cols,
    }
    (out / "walk.json").write_text(json.dumps(walk, indent=1))
    print(f"walk.json written: {len(cols)} column(s), {len(windows)} "
          f"window(s) of {a.months or a.days} "
          f"{'month(s)' if a.months else 'day(s)'}, year {year}, "
          f"window bottom {a.bottom!r}, forward {forward_var} "
          f"({a.negative_forward}), sink {sink['datum']!r}")
    print(f"next: submit tools/walk_job.py {out} inside a SLURM job "
          f"(ask first)")


if __name__ == "__main__":
    main()
