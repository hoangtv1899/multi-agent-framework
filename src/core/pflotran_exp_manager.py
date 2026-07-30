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
        print(f"✓ {len(built) if built else 0} deck(s) → {out}")
        return built or []

    def _run(self, experiments: List[Dict], config: Dict[str, Any]) -> List[Dict]:
        """Execute the decks directly. No scheduler — see NEEDS_SCHEDULER."""
        import subprocess, os, time
        exe = os.environ.get("PFLOTRAN_EXECUTABLE")
        if not exe:
            raise RuntimeError("PFLOTRAN_EXECUTABLE is not set; "
                               "source env_compy.sh")
        results = []
        for e in experiments:
            case_dir = Path(e.get("case_dir") or "")
            deck = next(case_dir.glob("*.in"), None)
            if deck is None:
                results.append({**e, "status": "failed",
                                "reason": "no .in deck in the case dir"})
                continue
            t0 = time.time()
            proc = subprocess.run([exe, "-pflotranin", deck.name],
                                  cwd=str(case_dir), capture_output=True,
                                  text=True, timeout=config.get("timeout_s", 900))
            ok = proc.returncode == 0 and any(case_dir.glob("*.tec"))
            results.append({**e,
                            "status": "completed" if ok else "failed",
                            "runtime_seconds": round(time.time() - t0, 2),
                            "returncode": proc.returncode,
                            "n_output_files": len(list(case_dir.glob("*.tec"))),
                            "reason": None if ok else
                                      (proc.stderr or "").strip()[-200:]})
            print(f"  {'✓' if ok else '✗'} {e.get('id')}: "
                  f"{results[-1]['runtime_seconds']}s, "
                  f"{results[-1]['n_output_files']} tec")
        return results

    def _extract(self, experiments, plan=None, config=None):
        """NOT YET IMPLEMENTED — deliberately raises.

        PFLOTRAN's output is a DEPTH PROFILE per column at a few output times
        (z, pressure, saturation). The Analyzer's context is built on a tidy
        frame keyed date|entity|variable|value — a time series per column, with
        no depth axis. Writing an extractor before that schema question is
        settled would either flatten the profiles into something lossy or bolt
        a second shape onto experiment.json by accident.

        Returning an empty results object instead would be worse: the base
        would package it, the Analyzer would run on nothing, and the run would
        report success with no data. That is the failure this codebase keeps
        rediscovering, so this raises until the shape is decided.
        """
        raise NotImplementedError(
            "PFLOTRAN _extract is pending the profiles-vs-depth-axis decision; "
            "the decks run and their .tec output is on disk under 01_inputs/")

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
