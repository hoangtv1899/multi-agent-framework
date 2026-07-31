#!/usr/bin/env python3
"""
PFLOTRAN Experiment Manager
src/core/pflotran_exp_manager.py

The subsurface backend: one 1-D Richards-flow column per sampled point, with
the Fan 2013 equilibrium water table as both the initial condition and the
bottom boundary.

WHAT IT SHARES WITH ELM, AND WHY THAT IS THE POINT. Sampling is identical —
`_materialize` in the base resolves the bbox, clips to the HUC, stratifies by
elevation, and enriches each column with soil and `fan_wtd_m`. PFLOTRAN needs
every one of those fields, so a PFLOTRAN run over the columns an ELM run
produced is comparing two models at the same points rather than at two
different samples of the same basin. That is the whole reason the manager was
split.

WHERE IT DIFFERS, DECLARED RATHER THAN STUBBED:

    NEEDS_PREPARE = False     ELM compiles CIME cases (~8 min for the first,
                              clones after). PFLOTRAN writes a text deck; deck
                              generation IS the build, so there is no separate
                              prepare stage. Declaring that is honest; a no-op
                              _prepare() would report "prepared nothing,
                              successfully".
    NEEDS_SCHEDULER = False   ELM's 19 columns took 2406 s through SLURM.
                              PFLOTRAN's took 0.3 s each on the login node,
                              measured. Queueing them would cost more in wait
                              than in compute.

DECK GENERATION IS NOT DONE HERE. tools/build_pflotran_cases.py already
generates and runs decks and is verified at 19/19 columns; this manager calls
it. A second deck generator would drift from the one that has been tested.

_to_run_plan EMITS A SPEC, NOT DECKS. The base writes columns.json only after
`_refine_columns`, and the design figure after that — decks written during
materialization would describe a plan that had not been finalised. So the plan
carries what to build and `_build` builds it.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "src")

from core.exp_manager_base import ExperimentManagerBase   # noqa: E402


def _load_tool(name: str):
    """Import a tools/*.py module by name, as the base does."""
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, str(root / "tools" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _PFLOTRANResults:
    """The results object _package consumes: `.results` rows plus `.units`.

    Mirrors ELMResultsAnalyzer's surface so the base needs no backend test.
    """

    def __init__(self, rows, units):
        self.results = rows
        self.units = dict(units)
        self.summary = {"units": dict(units)}


class PFLOTRANExpManager(ExperimentManagerBase):
    """steps 0-5 for standalone subsurface flow over sampled columns."""

    MODEL = "pflotran"

    NEEDS_PREPARE = False       # deck generation is the build; see docstring
    NEEDS_SCHEDULER = False     # 0.3 s per column, measured
    COUPLES_TO = None           # nothing downstream of it yet

    # The key that marks a plan as already executable. DELIBERATELY NOT
    # ELM's CONDITIONS_COUPLERS: if the two backends shared a key, an ELM plan
    # would look already-materialized to this manager and it would build zero
    # experiments without raising — the exact silent mis-dispatch that produced
    # empty PFLOTRAN runs before the managers were split.
    PLAN_KEY = "PFLOTRAN_CASES"

    # Per-column wall limit. Generous for flow, which takes ~1 s; it exists for
    # the reactive subclass, where a deep unsaturated profile can drive the
    # timestep to machine epsilon and grind indefinitely at t≈1 y.
    RUN_TIMEOUT_S = 900

    # What this backend's metrics MEAN, for the Analyzer and for the LLM step
    # 2 hands them to. Every entry is a quantity _extract actually computes;
    # nothing here is inherited. Two of them carry the caveats that a reader
    # comparing PFLOTRAN against ELM most needs and would not otherwise see.
    FIELD_SEMANTICS = {
        "final_water_table_depth_m": {
            "units": "m", "from": ["LIQUID_SATURATION"],
            "note": "positive downward from the surface, at the LAST output "
                    "time. null means NO water table was found in the domain "
                    "— the column never reaches saturation — which is NOT the "
                    "same as a water table at the domain bottom, and must not "
                    "be filled in with domain_depth_m"},
        "water_table_in_domain": {
            "units": "bool", "from": ["LIQUID_SATURATION"],
            "note": "false for columns whose Fan water table lies below the "
                    "domain depth cap; those columns run fully unsaturated "
                    "and their saturation metrics describe drainage, not a "
                    "water table"},
        "saturation_top": {"units": "1", "from": ["LIQUID_SATURATION"],
                           "note": "shallowest cell, last output time"},
        "saturation_bottom": {"units": "1", "from": ["LIQUID_SATURATION"],
                              "note": "deepest cell, last output time"},
        "saturation_mean": {"units": "1", "from": ["LIQUID_SATURATION"],
                            "note": "depth-mean over the column, last output "
                                    "time. UNWEIGHTED by cell thickness"},
        "fan_wtd_m": {
            "units": "m", "from": [],
            "note": "the Fan 2013 water table this column was INITIALISED "
                    "with, not a result. Comparing it to "
                    "final_water_table_depth_m measures drift away from the "
                    "initial condition, not agreement with an observation"},
        "domain_depth_m": {"units": "m", "from": [],
                           "note": "column length, capped by depth_cap"},
        "n_cells": {"units": "1", "from": [], "note": "vertical cells"},
    }

    def _already_executable(self, plan: Dict[str, Any]) -> bool:
        return bool(plan.get(self.PLAN_KEY))

    def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
        """No-op, and that is a statement rather than an omission.

        ELM refines here because a warm start snaps each column to its donor
        gridcell and adopts that cell's soil — the sampled coordinates are not
        the simulated ones. PFLOTRAN runs at the sampled point with the sampled
        soil and the Fan water table already on the column, so the columns the
        sampler produced are the columns that run.
        """
        return {}

    # ─────────────────────────────────────────────────────────
    # PLAN
    # ─────────────────────────────────────────────────────────
    def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
        """Columns → a deck SPEC per column. No files are written here."""
        bottom = config.get("bottom", "fan")
        years = float(config.get("years", 20.0))
        depth_cap = float(config.get("depth_cap", 50.0))
        flux_from = config.get("flux_from")          # an ELM run dir, optional

        cases = []
        for c in columns:
            fan = c.get("fan_wtd_m")
            cases.append({
                "id": c.get("id"),
                "lat": c.get("lat"), "lon": c.get("lon"),
                "elevation_m": c.get("elevation_m"),
                "fan_wtd_m": fan,
                # Recorded per case because it changes what the run means: a
                # column whose water table is below the domain runs fully
                # unsaturated, and on the 2019 Gunnison sample that was 10 of
                # 19 columns. A reader comparing them needs to know which.
                "wt_in_domain": (fan is not None and fan < depth_cap),
                "bottom_bc": bottom,
                "years": years,
            })

        n_capped = sum(1 for c in cases if not c["wt_in_domain"])
        ledger = [
            {"key": "initial_condition",
             "value": "hydrostatic, pinned at the Fan 2013 water table",
             "why": "no measured subsurface state exists for these columns; "
                    "Fan is a modelled equilibrium prior, not an observation"},
            {"key": "bottom_boundary", "value": f"{bottom}",
             "why": "hydrostatic at the Fan water table" if bottom == "fan"
                    else "no-flow"},
            {"key": "domain_depth_cap_m", "value": depth_cap,
             "why": f"{n_capped} of {len(cases)} columns have a Fan water "
                    f"table below the cap and run fully unsaturated"},
            # experiment.json's `period` is the period the USER asked about,
            # written by the base for every backend. It is not what this model
            # simulated, and the two being different is easy to miss: a reader
            # sees period 2019-2019 beside a 20-year relaxation and has no
            # reason to suspect they are different quantities unless it is
            # stated here.
            {"key": "simulated_duration_y", "value": years,
             "why": "a relaxation from the Fan initial condition, NOT a "
                    "simulation of the requested period — experiment.json's "
                    "`period` records what was asked about, not what was run"},
        ]
        if not flux_from:
            ledger.append({
                "key": "top_boundary", "value": "constant nominal recharge",
                "why": "no ELM flux supplied (--flux-from); the top boundary "
                       "is a placeholder, not a derived quantity, so columns "
                       "do not differ in their forcing"})

        return {self.PLAN_KEY: cases,
                "pflotran_settings": {"bottom": bottom, "years": years,
                                      "depth_cap": depth_cap,
                                      "flux_from": flux_from},
                "assumptions_ledger": ledger}

    # ─────────────────────────────────────────────────────────
    # STAGES
    # ─────────────────────────────────────────────────────────
    def _build(self, plan: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Generate the decks by calling the verified standalone tool."""
        cases = plan.get(self.PLAN_KEY) or []
        if not cases:
            raise RuntimeError(
                f"no {self.PLAN_KEY} in the plan — _materialize did not run, "
                f"or _to_run_plan produced nothing")

        settings = plan.get("pflotran_settings") or {}
        bp = _load_tool("build_pflotran_cases")
        columns = self._columns_from_disk()
        out = self.run_dir / "01_inputs" / "pflotran"

        # build_ensemble is the tool's own importable core — the same function
        # its CLI calls — so a run launched from here and one launched from
        # the command line produce byte-identical decks. `run=False`: executing
        # is _run's stage, not _build's.
        built = bp.build_ensemble(
            columns=columns, out_dir=str(out), run=False, quiet=False,
            flux_from=settings.get("flux_from"),
            bottom=settings.get("bottom", "fan"),
            years=settings.get("years", 20.0),
            depth_cap=settings.get("depth_cap", 50.0),
            spin_years=settings.get("spin_years", 10.0))

        # build_ensemble returns {"scenario": ..., "cases": [...]}, NOT a list.
        # Returning it whole made _build report "2 deck(s)" for 19 columns —
        # it was counting the dict's two keys — and handed _run a dict, which
        # iterates as its key STRINGS: 'str' object has no attribute 'get'.
        # Every stage-level test passed because they built `experiments` by
        # hand; only running the stages in sequence reached it.
        self.scenario = (built or {}).get("scenario") or {}
        cases = (built or {}).get("cases") or []

        # A column that build_ensemble skipped is a column the ensemble does
        # not have. Silent shrinkage is how an ensemble comes back smaller than
        # the strategy asked for with nothing on record saying so — and the
        # skip path here (`no ELM flux — skipped`) is reachable whenever
        # flux_from is set.
        if len(cases) != len(plan.get(self.PLAN_KEY) or []):
            print(f"   ⚠️  {len(plan[self.PLAN_KEY]) - len(cases)} of "
                  f"{len(plan[self.PLAN_KEY])} planned column(s) produced no "
                  f"deck and are absent from the ensemble")
        if not cases:
            raise RuntimeError(
                f"build_ensemble produced no decks from "
                f"{len(plan.get(self.PLAN_KEY) or [])} planned case(s)")

        print(f"✓ {len(cases)} deck(s) → {out}")
        return cases

    # Columns run CONCURRENTLY, up to this many. Threads, not processes:
    # each worker only waits on subprocess.run, which releases the GIL, so
    # there is nothing for extra interpreters to do.
    #
    # WHY NOT ProcessPoolExecutor, which is what the reaction MCP's own
    # ensemble_parallel uses: it defaults to fork on Linux, and forking a
    # process that has threads holding locks deadlocks the child before it
    # does any work. Measured — that tool hangs for its full 300 s timeout on
    # three decks that take 1.8 s here, spawning no PFLOTRAN at all, while the
    # same function called outside the server works fine.
    #
    # FOUR, not os.cpu_count(). These run on whatever node the workflow is on,
    # often a shared login node, and a 19-column ensemble at full width is
    # antisocial. Raise it with config['max_parallel'] on a compute node.
    MAX_PARALLEL = 4

    def _run(self, experiments: List[Dict], config: Dict[str, Any]) -> List[Dict]:
        """Execute the decks directly. No scheduler — see NEEDS_SCHEDULER."""
        import os
        from concurrent.futures import ThreadPoolExecutor

        exe = os.environ.get("PFLOTRAN_EXECUTABLE")
        if not exe:
            raise RuntimeError("PFLOTRAN_EXECUTABLE is not set; "
                               "source env_compy.sh")

        limit = config.get("timeout_s", self.RUN_TIMEOUT_S)
        width = max(1, int(config.get("max_parallel", self.MAX_PARALLEL)))
        width = min(width, len(experiments) or 1)
        if width > 1:
            print(f"   running {len(experiments)} column(s), "
                  f"{width} at a time")

        # Results stay in EXPERIMENT ORDER, not completion order: the run
        # record is compared against columns.json by position often enough
        # that a set of rows shuffled by which column happened to finish
        # first would be a needless difference between two identical runs.
        with ThreadPoolExecutor(max_workers=width) as pool:
            results = list(pool.map(
                lambda e: self._run_one(e, exe, limit), experiments))
        return results

    def _run_one(self, e: Dict[str, Any], exe: str, limit: float) -> Dict[str, Any]:
        """One column. Returns its outcome and writes it back onto `e`."""
        import subprocess, time
        case_dir = Path(e.get("case_dir") or "")
        deck = next(case_dir.glob("*.in"), None)
        if deck is None:
            outcome = {"status": "failed",
                       "reason": "no .in deck in the case dir"}
            e.update(outcome)
            return {**e, **outcome}
        t0 = time.time()
        try:
            proc = subprocess.run([exe, "-pflotranin", deck.name],
                                  cwd=str(case_dir), capture_output=True,
                                  text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            # ONE COLUMN, NOT THE ENSEMBLE. subprocess.run RAISES on
            # timeout, and uncaught that discarded every column already
            # computed along with every one still queued. A column whose
            # timestep collapses — the reactive decks do this on deep
            # unsaturated profiles — is a failed column with a reason, and
            # the run continues.
            outcome = {"status": "failed",
                       "runtime_seconds": round(time.time() - t0, 2),
                       "returncode": None,
                       "n_output_files": len(list(case_dir.glob("*.tec"))),
                       "reason": f"exceeded {limit}s — timestep collapse "
                                 f"or a non-converging solve"}
            e.update(outcome)
            print(f"  ✗ {e.get('id')}: TIMEOUT after {limit}s")
            return {**e, **outcome}

        ok = proc.returncode == 0 and any(case_dir.glob("*.tec"))
        outcome = {"status": "completed" if ok else "failed",
                   "runtime_seconds": round(time.time() - t0, 2),
                   "returncode": proc.returncode,
                   "n_output_files": len(list(case_dir.glob("*.tec"))),
                   "reason": None if ok else
                             (proc.stderr or "").strip()[-200:]}
        # Written back onto the experiment too, not only into the returned
        # copy. execute_plan hands _extract the EXPERIMENTS list, never
        # _run's return value, so a timing that lives only in the copy
        # never reaches experiment.json — every column came out with
        # runtime_seconds: null and step 4 reported no compute at all.
        e.update(outcome)
        print(f"  {'✓' if ok else '✗'} {e.get('id')}: "
              f"{outcome['runtime_seconds']}s, "
              f"{outcome['n_output_files']} tec")
        return {**e, **outcome}

    def _extract(self, experiments, plan=None, config=None):
        """.tec depth profiles -> the SAME per-column row shape ELM produces.

        experiment.json is one contract for every backend, so this emits rows
        _package already understands: case_name, status, metrics, variables.
        Nothing downstream needs a PFLOTRAN special case.

        WHAT MAPS CLEANLY AND WHAT DOES NOT. ELM's rows carry per-variable
        DAILY SERIES; PFLOTRAN's native output is a DEPTH PROFILE at a handful
        of output times (here 0, 1, 5, 10, 20 y). Those are different shapes
        and pretending otherwise would be the dishonest move — five yearly
        snapshots are not a daily series, and writing them under a `daily` key
        with invented dates would make step 0 build a frame that looks like a
        hydrograph and is not.

        So:
            metrics    scalars, exactly as ELM does — final water table,
                       saturation at top and bottom, storage. This is what a
                       cross-model comparison can actually use.
            variables  per-variable summary stats, no `daily` block.
            profiles   NEW, and PFLOTRAN-specific: depth, times, and the
                       saturation/pressure fields. Additive, so no existing
                       consumer changes.
        """
        rows, units = [], {"LIQUID_SATURATION": "-",
                           "LIQUID_PRESSURE": "Pa",
                           "WATER_TABLE_DEPTH": "m"}

        for e in experiments or []:
            cid = e.get("id") or e.get("case_name")
            case_dir = Path(e.get("case_dir") or "")

            # A COLUMN _run REJECTED STAYS REJECTED. A timed-out or crashed
            # column still leaves the .tec snapshots it managed to write, and
            # reading those produced a row marked "ok" carrying metrics from a
            # simulation that never reached its final time — while _run's own
            # record said "failed". The partial output is real but it is not
            # the experiment that was asked for, and the ensemble must not
            # average it in as though it were.
            if e.get("status") == "failed":
                rows.append({"case_name": cid, "scenario_name": cid,
                             "status": "failed",
                             "runtime_seconds": e.get("runtime_seconds"),
                             "reason": e.get("reason") or "the run failed",
                             "partial_output_files": len(
                                 list(case_dir.glob("*.tec")))})
                continue

            tecs = sorted(case_dir.glob("*.tec"))
            if not tecs:
                rows.append({"case_name": cid, "scenario_name": cid,
                             "status": "failed",
                             "reason": "no .tec output written"})
                continue

            times, profiles = [], []
            for t in tecs:
                tm, z, sat, pres = self._read_tec(t)
                if z:
                    times.append(tm)
                    profiles.append({"z_m": z, "saturation": sat,
                                     "liquid_pressure_pa": pres})

            if not profiles:
                rows.append({"case_name": cid, "scenario_name": cid,
                             "status": "failed",
                             "reason": ".tec files held no data rows"})
                continue

            final = profiles[-1]
            H = max(final["z_m"])
            depth = [round(H - z, 4) for z in final["z_m"]]
            sat = final["saturation"]
            # Water table = shallowest depth reaching full saturation. None
            # when the column never saturates, which on the 2019 sample was 10
            # of 19 columns — reported as None rather than as the domain
            # bottom, because "no water table in the domain" and "water table
            # at 50 m" are different statements.
            wt = min((d for d, sv in zip(depth, sat) if sv >= 0.999), default=None)

            rows.append({
                "case_name": cid, "scenario_name": cid, "status": "ok",
                "lat": e.get("lat"), "lon": e.get("lon"),
                "runtime_seconds": e.get("runtime_seconds"),
                "metrics": {
                    "final_water_table_depth_m": wt,
                    "water_table_in_domain": wt is not None,
                    "domain_depth_m": round(H, 3),
                    "n_cells": len(sat),
                    "saturation_top": round(sat[-1], 5),
                    "saturation_bottom": round(sat[0], 5),
                    "saturation_mean": round(sum(sat) / len(sat), 5),
                    "fan_wtd_m": e.get("fan_wtd_m"),
                },
                "variables": {
                    "LIQUID_SATURATION": {
                        "units": "-", "min": round(min(sat), 5),
                        "max": round(max(sat), 5),
                        "mean": round(sum(sat) / len(sat), 5)},
                    "LIQUID_PRESSURE": {
                        "units": "Pa",
                        "min": round(min(final["liquid_pressure_pa"]), 1),
                        "max": round(max(final["liquid_pressure_pa"]), 1)},
                },
                # The depth data, kept whole. Step 2's generated scripts read
                # this; the tidy frame has no depth axis and inventing one for
                # ELM's sake would change a shape every existing step relies on.
                "profiles": {
                    "times_y": times,
                    "depth_m": depth,
                    "saturation": [p["saturation"] for p in profiles],
                    "liquid_pressure_pa": [p["liquid_pressure_pa"] for p in profiles],
                },
            })

        n_ok = sum(1 for r in rows if r.get("status") == "ok")
        print(f"✓ extracted {n_ok}/{len(rows)} column(s)")
        return _PFLOTRANResults(rows, units)

    @staticmethod
    def _read_tec(path: Path):
        """(time_y, z, saturation, pressure) from one Tecplot POINT file.

        Columns are X, Y, Z, Liquid Pressure, Liquid Saturation, Material ID;
        the title line carries the output time.
        """
        z, sat, pres = [], [], []
        lines = path.read_text().splitlines()
        tm = None
        if lines and "TITLE" in lines[0]:
            try:
                tm = float(lines[0].split('"')[1].split("[")[0])
            except (IndexError, ValueError):
                tm = None
        for line in lines[3:]:
            f = line.split()
            if len(f) >= 5:
                try:
                    z.append(float(f[2]))
                    pres.append(float(f[3]))
                    sat.append(float(f[4]))
                except ValueError:
                    continue
        return tm, z, sat, pres

    # ─────────────────────────────────────────────────────────
    def _columns_from_disk(self) -> List[Dict[str, Any]]:
        """The columns _materialize just wrote, read back rather than passed.

        Keeps this manager honest about the ordering the base enforces: the
        file is the record, so building from anything else risks building a
        plan that columns.json does not describe.
        """
        for p in (self.input_dir / "columns.json", self.run_dir / "columns.json"):
            if p.exists():
                d = json.loads(p.read_text())
                return d.get("columns", d) if isinstance(d, dict) else d
        raise FileNotFoundError("columns.json not found; _materialize must run first")
