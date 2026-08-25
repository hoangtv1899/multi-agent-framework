#!/usr/bin/env python3
"""Build a walk directory: cloned ELM cases + PFLOTRAN spin decks + walk.json.

    python tools/walk_setup.py --elm-run RUN_DIR --pf-run RUN_DIR \
        --out WALK_DIR (--months 1 | --days 7) [--year YYYY]

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
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import walk_lib                                        # noqa: E402
import elm_wrapper as ew                               # noqa: E402


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


def _fsurdat_of(case_dir):
    m = re.search(r"fsurdat\s*=\s*'([^']+)'",
                  (Path(case_dir) / "user_nl_elm").read_text())
    if not m:
        raise RuntimeError(f"{case_dir}: user_nl_elm names no fsurdat — "
                           f"apply_profile needs the column's own surfdata")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elm-run", required=True)
    ap.add_argument("--pf-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--year", type=int, default=None)
    a = ap.parse_args()

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

    # The year is the forcing year the reference cases already run
    ref0 = next(iter(built.values()))
    start = _sh(["./xmlquery", "RUN_STARTDATE", "--value"], cwd=ref0).strip()
    year = a.year or int(start[:4])
    windows = walk_lib.window_edges(year, months=a.months, days=a.days)

    # ── PFLOTRAN spin decks: steady at each column's own mean flux ──────
    pf, _cb, _sim, _sd = walk_lib.load_pf_server()
    spin_cols = []
    for c in pf_cols:
        flux = c.get("daily_flux_mm_day") or []
        if not flux:
            raise SystemExit(f"{c.get('id')}: the pf run's column carries no "
                             f"daily_flux_mm_day — nothing to spin at")
        spin_cols.append({
            "id": c["id"], "lat": c.get("lat"), "lon": c.get("lon"),
            "water_table_m": c.get("water_table_m"),
            "recharge_mm_yr": round(sum(flux) / len(flux) * 365.0, 4),
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

    # ── ELM clones, configured for the first window ─────────────────────
    cols = []
    for c in pf_cols:
        cid = c["id"]
        if cid not in built:
            raise SystemExit(f"{cid}: the elm run built no case for it")
        dest = _clone_elm(Path(built[cid]), out / "elm" / cid, windows[0])
        r = rows[cid]
        cols.append({
            "id": cid,
            "elm_case_dir": str(dest),
            "fsurdat": _fsurdat_of(dest),
            "water_table_m": c.get("water_table_m"),
            "depth_m": r.get("depth_m"),
            "n_cells": r.get("n_cells"),
            "spin_input_file": r.get("input_file"),
            "spin_checkpoint": r.get("restart_file_expected"),
            "spin_rate_mm_yr": next(s["recharge_mm_yr"] for s in spin_cols
                                    if s["id"] == cid),
            "lat": c.get("lat"), "lon": c.get("lon"),
        })

    walk = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "year": year,
        "window": ({"months": a.months} if a.months else {"days": a.days}),
        "n_windows": len(windows),
        # The walk's clock origin: the spin runs 20 years steady (the
        # builder's steady final_time), so its checkpoint carries t=20 y.
        "t0_y": 20.0,
        "site_dir": str(pf_run),
        "elm_run_source": str(elm_run),
        "columns": cols,
    }
    (out / "walk.json").write_text(json.dumps(walk, indent=1))
    print(f"walk.json written: {len(cols)} column(s), {len(windows)} "
          f"window(s) of {a.months or a.days} "
          f"{'month(s)' if a.months else 'day(s)'}, year {year}")
    print(f"next: submit tools/walk_job.py {out} inside a SLURM job "
          f"(ask first)")


if __name__ == "__main__":
    main()
