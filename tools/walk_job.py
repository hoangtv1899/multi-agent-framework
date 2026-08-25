#!/usr/bin/env python3
"""Run the walk: both models march the same year, exchanging every window.

    python tools/walk_job.py WALK_DIR

Run INSIDE a SLURM allocation (ELM runs via srun). WALK_DIR comes from
tools/walk_setup.py. Per window:

    ELM runs the window        (slice: proven exact, job 774960)
      -> its daily drainage    (extract_column, spin-drop disabled)
    PFLOTRAN continues the window from its checkpoint
                               (proven to 6e-4 / 62 Pa)
      -> its solved water table + saturation profile
    stamped onto ELM's next-slice restart
                               (set_water_table.apply + apply_profile,
                                the proven return leg, mid-year now)

State (walk_state.json) advances after each window, so a killed job
resumes at the window it died in. Every number exchanged is logged to
walk_log.jsonl; the trajectories land in walk_summary.json.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))       # compare/ imports agents.analysis
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import walk_lib                                        # noqa: E402
import elm_wrapper as ew                               # noqa: E402
import extract as elm_extract                          # noqa: E402
import set_water_table as swt                          # noqa: E402


def _pf():
    pf_server, cb, sim, site_data = walk_lib.load_pf_server()
    return (pf_server, cb.build_column_deck, cb.conus2_cells,
            sim.run_simulation, site_data)


def _solved_fn():
    import importlib.util
    pkg = ROOT / "mcp" / "pflotran-mcp" / "compare"
    spec = importlib.util.spec_from_file_location(
        "compare_pflotran", pkg / "__init__.py",
        submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_pflotran", mod)
    spec.loader.exec_module(mod)
    from compare_pflotran import water_table as wt
    return wt.solved_water_table_m


def _log(walk_dir, row):
    with open(Path(walk_dir) / "walk_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def main(walk_dir):
    wd = Path(walk_dir).resolve()
    walk = json.loads((wd / "walk.json").read_text())
    state = walk_lib.load_state(wd)
    pf_server, build_deck, conus2_cells, run_sim, site_data = _pf()
    solved_of = _solved_fn()
    EXE = os.environ["PFLOTRAN_EXECUTABLE"]
    t0, year = float(walk["t0_y"]), int(walk["year"])
    win = walk["window"]
    windows = walk_lib.window_edges(year, months=win.get("months"),
                                    days=win.get("days"))
    cols = walk["columns"]

    def pf_run_ok(input_file):
        r = run_sim(input_file=input_file, executable=EXE, timeout=1800)
        codes = r.get("exit_codes") or [r.get("exit_code")]
        return all(c == 0 for c in codes), r

    # ── spin once: PFLOTRAN's clock starts where these end ──────────────
    if not state.get("spun"):
        for c in cols:
            print(f"== spin {c['id']}", flush=True)
            ok, r = pf_run_ok(c["spin_input_file"])
            assert ok, (c["id"], r)
            ck = Path(c["spin_checkpoint"])
            assert ck.is_file(), f"{c['id']}: spin wrote no checkpoint {ck}"
            state["checkpoints"][c["id"]] = str(ck)
        state["spun"] = True
        walk_lib.save_state(wd, state)

    # the deck geometry each continuation must reproduce exactly
    joined = site_data.join(walk["site_dir"],
                            [{"id": c["id"], "lat": c["lat"],
                              "lon": c["lon"]} for c in cols])
    prof_of = {c["id"]: c.get("subsurface_profile")
               for c in joined["columns"]}
    cells_of = {}
    for c in cols:
        cells = conus2_cells(prof_of[c["id"]], float(c["depth_m"]),
                             max_cell_m=0.5)
        assert len(cells) == int(c["n_cells"]), (
            f"{c['id']}: continuation would have {len(cells)} cells against "
            f"the spin's {c['n_cells']} — the grids must be identical for "
            f"the restart to mean anything")
        cells_of[c["id"]] = cells

    # ── the walk ────────────────────────────────────────────────────────
    for w in windows[int(state["windows_done"]):]:
        print(f"== window {w['i'] + 1}/{len(windows)} "
              f"(days {w['d0']}..{w['d1']})", flush=True)
        row = {"window": w["i"], "d0": w["d0"], "d1": w["d1"], "columns": {}}

        expected_end = walk_lib.window_end_date(year, w["d1"])
        for c in cols:
            case = c["elm_case_dir"]
            # RESUME-SAFE: state advances only after a FULL window, so a job
            # killed mid-window re-enters it — and continuing a case that
            # already ran this window would advance it a second one, silently
            # overrunning the year. The restart's own date says where the
            # case stands: at the window's end, skip; past it, abort loudly.
            try:
                at = walk_lib.restart_date(ew.latest_restart(case).name)
            except ValueError:
                at = None                              # fresh case, no restart
            if at == expected_end:
                print(f"   {c['id']}: already at {at} — window re-entered, "
                      f"skipping its ELM run", flush=True)
                continue
            if at is not None and at > expected_end:
                raise RuntimeError(
                    f"{c['id']}: the case stands at {at}, PAST this window's "
                    f"end {expected_end} — the walk lost track; refusing to "
                    f"advance it further")
            if w["i"] > 0:
                ew.configure_continuation(case, stop_n=w["stop_n"],
                                          stop_option=w["stop_option"])
            assert ew.run_built_case(case), f"{c['id']}: ELM window {w['i']}"

        for c in cols:
            cid = c["id"]
            data, meta = elm_extract.extract_column(
                c["elm_case_dir"], variables=["QDRAI"], spinup_days=0)
            qd = ((data.get("variables") or {}).get("QDRAI") or {})
            vals = walk_lib.window_slice(data.get("dates"),
                                         qd.get("values") or [],
                                         year, w["d0"], w["d1"])
            series = walk_lib.flux_series(vals, w["d0"], t0)
            end = round(t0 + w["d1"] / 365.0, 8)
            built = build_deck(
                column={"id": cid, "soil_profile": {"layers": []}},
                out_dir=str(wd / "pf" / f"w{w['i']:02d}"),
                water_table_m=float(c["water_table_m"]),
                recharge_series=series, cells=cells_of[cid],
                restart_from=state["checkpoints"][cid],
                final_time_y=end, output_times_y=[end], checkpoint=True)
            ok, r = pf_run_ok(built["input_file"])
            assert ok, (cid, w["i"], r)
            out_file = str(wd / "pf" / f"w{w['i']:02d}" / f"x_{cid}.json")
            pf_server.extract_column_series(
                cases=[{"id": cid, "case_dir": built["case_dir"]}],
                out_file=out_file)
            block = json.loads(Path(out_file).read_text())["columns"][cid]
            wt_solved = solved_of(block)
            assert wt_solved is not None, (cid, "no solved water table")
            ck = Path(built["restart_file_expected"])
            assert ck.is_file(), (cid, f"window wrote no checkpoint {ck}")

            # SATURATED TO THE SURFACE IS AN ANSWER, NOT A CRASH: solved 0.0
            # is a real winter state, and apply() rightly refuses 0 as "not
            # below ground". Stamp a hair below the surface and say so —
            # the saturation profile (which apply_profile carries whole) is
            # the state that matters; the ZWT number is its label.
            saturated = float(wt_solved) <= 0.0
            wt_stamp = max(float(wt_solved), 0.01)
            rest = ew.latest_restart(c["elm_case_dir"])
            a1 = swt.apply(str(rest), wt_stamp, quiet=True)
            a2 = swt.apply_profile(str(rest), block["depth_m"],
                                   block["saturation"][-1], c["fsurdat"],
                                   quiet=True)
            state["checkpoints"][cid] = str(ck)
            row["columns"][cid] = {
                "flux_mean_mm_day": round(sum(vals) / len(vals), 4),
                "wt_solved_m": round(float(wt_solved), 4),
                "saturated_to_surface": saturated,
                "stamped": rest.name,
                "zwt_written_m": a1.get("written_m"),
                "zwt_regime": a1.get("regime"),
                "layers_written": a2.get("layers_written"),
            }

        state["windows_done"] = w["i"] + 1
        state["log"].append({"window": w["i"], "done": True})
        # LOG BEFORE STATE: a kill between the two then leaves a duplicate
        # row (which the summary's d1-keyed dedup collapses) instead of a
        # window that ran, stamped, and is missing from every trajectory.
        _log(wd, row)
        walk_lib.save_state(wd, state)

    # ── the record: one trajectory per column ───────────────────────────
    # keyed by window end so a window RE-ENTERED after a kill (logged twice,
    # the later row the one that stamped) appears once, as its final values
    by_day = {c["id"]: {} for c in cols}
    for line in (wd / "walk_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        for cid, v in r["columns"].items():
            by_day[cid][r["d1"]] = v["wt_solved_m"]
    traj = {cid: {"day": sorted(d), "wt_solved_m": [d[k] for k in sorted(d)]}
            for cid, d in by_day.items()}
    # ELM's own view of the year, for the cadence figure
    elm_series = {}
    for c in cols:
        data, _m = elm_extract.extract_column(
            c["elm_case_dir"], variables=["ZWT", "QDRAI"], spinup_days=0)
        elm_series[c["id"]] = {
            "dates": data.get("dates"),
            "ZWT": ((data.get("variables") or {}).get("ZWT") or {}).get("values"),
            "QDRAI": ((data.get("variables") or {}).get("QDRAI") or {}).get("values"),
        }
    (wd / "walk_summary.json").write_text(json.dumps({
        "year": year, "window": win, "n_windows": len(windows),
        "t0_y": t0, "solved_trajectories": traj, "elm_daily": elm_series,
        "window_0_note": (
            "the first ~14 days carry ELM's warm-start relaxation (the "
            "day-1 drainage spike the site extractor normally trims — see "
            "extract.py SPINUP_DAYS); January's exchange is start-up "
            "adjustment, not weather"),
    }, indent=1))
    print("WALK COMPLETE:", len(windows), "windows,",
          len(cols), "columns", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
