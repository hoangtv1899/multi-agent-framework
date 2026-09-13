#!/usr/bin/env python3
"""Run the walk: both models march the same year, exchanging every window.

    python tools/walk_job.py WALK_DIR

Run INSIDE a SLURM allocation (ELM runs via srun). WALK_DIR comes from
tools/walk_setup.py. Per window:

    ELM runs the window        (slice: proven exact, job 774960)
      -> its daily forward flux (QDRAI or QCHARGE, walk_setup --forward;
                               extract_column, spin-drop disabled)
    PFLOTRAN continues the window from its checkpoint
                               (proven to 6e-4 / 62 Pa)
      -> its solved water table + saturation profile, and with a lateral
         sink the window's outflow off its -mas.dat
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


def _leave(state, row, cid, w, reason):
    """A column's departure, recorded: the state, the window's row, the line."""
    state["dead"][cid] = {"window": w["i"], "window_end_day": w["d1"],
                          "reason": reason}
    row["columns"][cid] = {"failed": reason}
    print(f"   ✗ {cid} leaves the walk at window {w['i'] + 1}: {reason}",
          flush=True)


def _sink_kwargs(c):
    """The deck builder's three sink kwargs for a column that carries one, else {}.

    A column without sink_conductance_m (walk_setup --sink-datum none, or a
    walk dir from before the sink) gets the call it always got, so a server
    without the sink still runs a sink-free walk.
    """
    if not isinstance(c.get("sink_conductance_m"), (int, float)):
        return {}
    return {"lateral_sink_depth_m": float(c["sink_datum_applied_m"]),
            "lateral_sink_band_m": float(c["sink_band_m"]),
            "lateral_sink_conductance": float(c["sink_conductance_m"])}


def _lateral_outflow(case_dir, cid):
    """The window's lateral outflow off <cid>-mas.dat as row keys; {} without.

    Positive = leaving; kg == mm over the 1 m^2 column; the window runs in
    its own case directory so the file's last cumulative value is its
    total. A file that cannot be read is reported as the error, not as a
    number: this is a record of the window, not a gate on it.
    """
    try:
        block = walk_lib.lateral_outflow_block(
            Path(case_dir) / f"{cid}-mas.dat")
    except ValueError as e:
        return {"lateral_outflow_error": str(e)[:200]}
    if block is None:
        return {}
    return {"lateral_outflow_window_mm": block["window_total_mm"]}


def main(walk_dir):
    wd = Path(walk_dir).resolve()
    walk = json.loads((wd / "walk.json").read_text())
    state = walk_lib.load_state(wd)
    pf_server, build_deck, conus2_cells, run_sim, site_data = _pf()
    solved_of = _solved_fn()
    EXE = os.environ["PFLOTRAN_EXECUTABLE"]
    t0, year = float(walk["t0_y"]), int(walk["year"])
    # walk_setup.py --bottom: "water_table" re-anchors every window's deck to
    # the starting level (which pins the answer there); "none" seals the
    # bottom so the level must move with the drainage. The spin decks are
    # anchored regardless — the seal starts at window 0, whose state comes
    # from the anchored spin's checkpoint.
    bottom = walk.get("pf_bottom", "water_table")
    win = walk["window"]
    windows = walk_lib.window_edges(year, months=win.get("months"),
                                    days=win.get("days"))
    cols = walk["columns"]
    # walk_setup.py --forward / --negative-forward: which ELM flux drives
    # the top face (QDRAI, or QCHARGE for the aquifer's own recharge) and
    # what a negative day becomes. A walk dir from before the dials carries
    # neither and means QDRAI, clip (QDRAI is never negative, so the clip
    # changes nothing there). The sink rides each column as the three
    # numbers the deck builder takes; the spin decks never see it.
    forward_var = str(walk.get("forward_var") or "QDRAI").upper()
    negative_forward = walk.get("negative_forward") or "clip"
    # walk_setup --return: a walk dir from before the dial stamped the
    # profile every window, so that is what its absence means.
    return_leg = walk.get("return_leg") or "wt+profile"
    if return_leg not in walk_lib.RETURN_LEGS:
        sys.exit(f"return_leg={return_leg!r} is not one of {walk_lib.RETURN_LEGS}")
    sink_kw = {c["id"]: _sink_kwargs(c) for c in cols}

    def pf_run_ok(input_file):
        r = run_sim(input_file=input_file, executable=EXE, timeout=1800)
        codes = r.get("exit_codes") or [r.get("exit_code")]
        return all(c == 0 for c in codes), r

    # A COLUMN MAY LEAVE THE WALK. A sealed column that fills has nowhere
    # to put arriving water — the solver dies mid-window and PFLOTRAN still
    # exits 0 at its step cap (measured: Naches col_10, job 775060, dead at
    # day 22 after 166k timestep cuts). One column's death is recorded with
    # its reason and window, and the others walk on.
    state.setdefault("dead", {})

    # ── spin once: PFLOTRAN's clock starts where these end ──────────────
    if not state.get("spun"):
        for c in cols:
            print(f"== spin {c['id']}", flush=True)
            ok, r = pf_run_ok(c["spin_input_file"])
            assert ok, (c["id"], r)
            ck = Path(c["spin_checkpoint"])
            assert ck.is_file(), f"{c['id']}: spin wrote no checkpoint {ck}"
            state["checkpoints"][c["id"]] = str(ck)
            # THE RETURN LEG, BEFORE THE FIRST WINDOW. An uncoupled warm
            # start replays its relaxation as January weather (Naches, job
            # 775060: 648-1795 mm/day day-one bursts against 2.5 mm/day
            # annual means — a solver-killing slug into a sealed column).
            # Stamping the spun state onto the clone's OWN starting file is
            # the same repair every later window gets; walk_setup made the
            # copy, and old walk dirs without `finidat` are left unstamped.
            if c.get("finidat"):
                out_file = str(wd / "pf_spin" / f"x_{c['id']}.json")
                pf_server.extract_column_series(
                    cases=[{"id": c["id"],
                            "case_dir": str(Path(c["spin_input_file"]).parent)}],
                    out_file=out_file)
                block = json.loads(
                    Path(out_file).read_text())["columns"][c["id"]]
                wt0 = solved_of(block)
                assert wt0 is not None, (c["id"], "spin solved no water table")
                a1 = swt.apply(c["finidat"], max(float(wt0), 0.01), quiet=True)
                a2 = swt.apply_profile(c["finidat"], block["depth_m"],
                                       block["saturation"][-1], c["fsurdat"],
                                       quiet=True)
                print(f"   {c['id']}: spun state stamped onto its starting "
                      f"file (wt {a1.get('written_m')} m, "
                      f"{a2.get('layers_written')} layers)", flush=True)
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
        alive = [c for c in cols if c["id"] not in state["dead"]]
        if not alive:
            print("== every column has left the walk — stopping", flush=True)
            break
        print(f"== window {w['i'] + 1}/{len(windows)} "
              f"(days {w['d0']}..{w['d1']}, {len(alive)} column(s))",
              flush=True)
        row = {"window": w["i"], "d0": w["d0"], "d1": w["d1"], "columns": {}}

        expected_end = walk_lib.window_end_date(year, w["d1"])
        for c in alive:
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

        for c in alive:
            cid = c["id"]
            data, meta = elm_extract.extract_column(
                c["elm_case_dir"], variables=[forward_var], spinup_days=0)
            raw = walk_lib.forward_values(data, forward_var)
            # ELM WROTE NO SERIES: a recorded departure, never a traceback
            # (design, section 5; Naches col_04, where QCHARGE is fill on a
            # cell ELM never solved it for). Before window_slice, which
            # would refuse the empty record as a month of missing days.
            if not raw:
                _leave(state, row, cid, w,
                       walk_lib.no_forward_reason(forward_var, meta))
                continue
            vals = walk_lib.window_slice(data.get("dates"), raw,
                                         year, w["d0"], w["d1"])
            vals, clipped_mm = walk_lib.apply_negative_policy(
                vals, negative_forward)
            series = walk_lib.flux_series(vals, w["d0"], t0)
            end = round(t0 + w["d1"] / 365.0, 8)
            built = build_deck(
                column={"id": cid, "soil_profile": {"layers": []}},
                out_dir=str(wd / "pf" / f"w{w['i']:02d}"),
                water_table_m=float(c["water_table_m"]),
                recharge_series=series, cells=cells_of[cid],
                restart_from=state["checkpoints"][cid], bottom=bottom,
                final_time_y=end, output_times_y=[end], checkpoint=True,
                **sink_kw[cid])
            ck = Path(built["restart_file_expected"])
            out_file = str(wd / "pf" / f"w{w['i']:02d}" / f"x_{cid}.json")

            def _window_block():
                pf_server.extract_column_series(
                    cases=[{"id": cid, "case_dir": built["case_dir"]}],
                    out_file=out_file)
                return json.loads(
                    Path(out_file).read_text())["columns"].get(cid)

            # THE PF MIRROR OF THE ELM SKIP-GUARD (adversarial review,
            # 2026-08-26): a re-entered window may already hold this leg's
            # finished result — the kill gap between _log and save_state
            # re-enters the window, and re-running a finished deterministic
            # leg only gives a transient failure (a slow node, a timeout)
            # the chance to disown it and record a healthy column dead. The
            # window's own checkpoint name is the proof: a step-cap death
            # writes <cid>-restart-max-ts.h5 and no snapshot, so it cannot
            # pass this gate.
            block = None
            if ck.is_file():
                block = _window_block()
                if block:
                    print(f"   {cid}: window checkpoint and output already "
                          f"present — reused, not re-run", flush=True)
            # THE EXIT CODE IS NOT THE VERDICT: PFLOTRAN exits 0 when it
            # gives up at its step cap. The window's checkpoint and its
            # output time are what say the window was actually reached.
            reason = None
            if block is None:
                ok, r = pf_run_ok(built["input_file"])
                if not ok:
                    # exit codes + the run layer's own error sentence — not
                    # str(dict) truncated into a list of file paths (the
                    # review measured the 'error' field always cut off)
                    reason = ("the window's PFLOTRAN run failed: exit_codes="
                              f"{r.get('exit_codes') or [r.get('exit_code')]}"
                              + (f"; {r.get('error')}" if r.get("error")
                                 else ""))
                elif not ck.is_file():
                    reason = ("the solver never reached the window's end — "
                              "no checkpoint written (a sealed column that "
                              "fills has nowhere to put arriving water)")
                else:
                    block = _window_block()
                    if not block:
                        reason = ("the run wrote no output snapshot — "
                                  "nothing extracted at the window's end")
            if reason is None:
                wt_solved = solved_of(block)
                if wt_solved is None:
                    reason = "no solved water table in the window's extract"
            if reason:
                _leave(state, row, cid, w, reason)
                continue

            # SATURATED TO THE SURFACE IS AN ANSWER, NOT A CRASH: solved 0.0
            # is a real winter state, and apply() rightly refuses 0 as "not
            # below ground". Stamp a hair below the surface and say so —
            # the saturation profile (which apply_profile carries whole) is
            # the state that matters; the ZWT number is its label.
            saturated = float(wt_solved) <= 0.0
            wt_stamp = max(float(wt_solved), 0.01)
            rest = ew.latest_restart(c["elm_case_dir"])
            a1 = swt.apply(str(rest), wt_stamp, quiet=True)
            if return_leg == "wt":
                # THE PROFILE IS NOT STAMPED after window 0: copying
                # PFLOTRAN's saturation into ELM's soil every window put
                # water into ELM that PFLOTRAN kept, and ELM drained it
                # back down as recharge (walk_lib.elm_balance records the
                # jump that would show). The water table alone is a state
                # both models own once.
                a2 = {"layers_written": 0, "profile": "not stamped (return_leg=wt)"}
            else:
                a2 = swt.apply_profile(str(rest), block["depth_m"],
                                       block["saturation"][-1], c["fsurdat"],
                                       quiet=True)
            state["checkpoints"][cid] = str(ck)
            row["columns"][cid] = {
                "flux_mean_mm_day": round(sum(vals) / len(vals), 4),
                # the window's forward total after the policy, and what the
                # policy removed (both mm over the window)
                "forward_mm_window": round(sum(vals), 4),
                "forward_clipped_mm": clipped_mm,
                **_lateral_outflow(built["case_dir"], cid),
                "wt_solved_m": round(float(wt_solved), 4),
                "saturated_to_surface": saturated,
                "stamped": rest.name,
                "zwt_written_m": a1.get("written_m"),
                "zwt_regime": a1.get("regime"),
                "layers_written": a2.get("layers_written"),
                "return_leg": return_leg,
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
    out_by_day = {c["id"]: {} for c in cols}
    for line in (wd / "walk_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        for cid, v in r["columns"].items():
            # "the later row wins" must hold in BOTH directions: a success
            # overwrites, and a failure TOMBSTONES the day — otherwise a
            # window logged success-then-failure (killed in the log/state
            # gap, then disowned on re-entry) would keep a trajectory point
            # at the very day the death record says nothing was solved.
            if isinstance(v, dict) and "wt_solved_m" in v:
                by_day[cid][r["d1"]] = v["wt_solved_m"]
                out_by_day[cid][r["d1"]] = v.get("lateral_outflow_window_mm")
            elif isinstance(v, dict) and "failed" in v:
                by_day[cid].pop(r["d1"], None)
                out_by_day[cid].pop(r["d1"], None)
    traj = {cid: {"day": sorted(d), "wt_solved_m": [d[k] for k in sorted(d)],
                  # None on a window without a sink: an absence, not zero
                  "lateral_outflow_window_mm": [out_by_day[cid][k]
                                                for k in sorted(d)]}
            for cid, d in by_day.items()}
    # ELM's own view of the year, for the cadence figure — partial for a
    # column that left the walk (its case stands at the window it died in)
    elm_series = {}
    for c in cols:
        try:
            data, _m = elm_extract.extract_column(
                c["elm_case_dir"],
                variables=["ZWT", *walk_lib.BALANCE_VARS], spinup_days=0)
        except Exception as e:                              # noqa: BLE001
            elm_series[c["id"]] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        vs = data.get("variables") or {}
        got = {v: (vs.get(v) or {}).get("values") for v in ("ZWT", *walk_lib.BALANCE_VARS)}
        # ELM'S OWN WATER BALANCE, per window and for the year, with the
        # storage jump at every stamp. A conserving exchange keeps the
        # residual near zero; water created by the return leg shows here.
        balance = walk_lib.elm_balance(got, [(w_["d0"], w_["d1"]) for w_ in windows])
        elm_series[c["id"]] = {"dates": data.get("dates"), **got,
                               "balance": balance}
        yr = (balance or {}).get("year") or {}
        frac = yr.get("residual_fraction_of_P")
        if frac is not None and abs(frac) > 0.10:
            kind = ("WATER CREATED (the return leg?)" if frac < 0 else
                    "water left through a flux this history does not carry, "
                    "or the sink's removal mirrored by the stamp")
            print(f"   ⚠️  {c['id']}: ELM's year does not close: residual "
                  f"{yr['residual_mm']:+.0f} mm against {yr['P_mm']:.0f} mm "
                  f"of precipitation ({frac:+.0%}); stamp jumps "
                  f"{yr['stamp_jumps_mm']:+.0f} mm; {kind}", flush=True)
    (wd / "walk_summary.json").write_text(json.dumps({
        "year": year, "window": win, "n_windows": len(windows),
        "t0_y": t0, "pf_bottom": bottom,
        "forward_var": forward_var, "negative_forward": negative_forward,
        "return_leg": return_leg,
        "sink": walk.get("sink"),
        "columns_left_the_walk": state["dead"],
        "solved_trajectories": traj, "elm_daily": elm_series,
        "window_0_note": (
            "the first ~14 days carry ELM's warm-start relaxation (the "
            "day-1 drainage spike the site extractor normally trims — see "
            "extract.py SPINUP_DAYS); January's exchange is start-up "
            "adjustment, not weather"),
    }, indent=1))
    n_dead = len(state["dead"])
    print(f"WALK COMPLETE: {len(windows)} windows, "
          f"{len(cols) - n_dead}/{len(cols)} columns walked the year"
          + (f" ({n_dead} left early: "
             f"{', '.join(sorted(state['dead']))})" if n_dead else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
