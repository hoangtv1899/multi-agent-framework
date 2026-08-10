#!/usr/bin/env python3
"""Column locations → runnable ELM inputs.

docs/ELM_MCP_PLAN.md §5. This is the half of an ELM study that decides what the
model will actually see: which CONUS gridcell each column is warm-started from,
which soil it therefore runs on, and the thirteen CIME keys that name every file
the case build needs.

EXTRACTED, NOT COPIED. ELMExpManager's methods now delegate here, so there is
one implementation reached two ways — by the manager while it still exists, and
by the MCP tool. A second copy would drift, and the thing that would drift is
the warm start, which is what makes a column credible.

WHY THE ORDER IS FIXED. The warm start SNAPS each column to its donor gridcell,
moving it by ~250-400 m. Everything written before that describes a plan the run
will not follow, which is why the caller must persist columns.json only after
this returns.

    warm_start          finidat per column, coordinates snapped to the donor
    attach_donor_soil   the donor cell's soil — the ONLY soil the run has
    build_column_inputs domain.nc + surface.nc per column
    to_run_plan         columns → CONDITIONS_COUPLERS
    build_case_inputs   → 01_inputs/case_inputs.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from columns_to_plan import columns_to_elm_plan
from elm_experiment_builder import ELMExperimentBuilder

# Keys of an experiment that are plain data and mean something downstream.
# LISTED, not "everything except elm_agent": a new object added upstream would
# otherwise silently become a repr in the case-inputs file.
CASE_KEYS = (
    "scenario_index", "scenario_name", "case_name", "forcing_period",
    "soil_config", "substrate", "forcing_start", "forcing_end", "stop_n",
    "start_date", "description", "lat", "lon", "elevation_m", "band",
    "case_dir",
)

CASE_INPUTS = "case_inputs.json"


# ─────────────────────────────────────────────────────────────────────
# WARM START
# ─────────────────────────────────────────────────────────────────────
# The CONUS restart grid is 1 km, so a point inside a cell is at most a
# half-diagonal (~0.71 km) from its centre. 1.5 km is twice that: comfortably
# past any grid irregularity, and far short of the 4.52 km that says the point
# was never on the land grid.
MAX_SNAP_KM = 1.5


def _donor_topo(surfdata_path):
    """The donor gridcell's surface height, from the surfdata just subset.

    Returns None when the file is unreadable or TOPO is absent or zero. ~19% of
    the CONUS surfdata's TOPO cells are zero (1,353,973 of 1,670,400 are not),
    and a zero silently becoming a column's elevation would be worse than the
    stale value this replaced — so it is a miss, not a number.
    """
    if not surfdata_path:
        return None
    try:
        import netCDF4 as nc
        import numpy as np
        with nc.Dataset(str(surfdata_path)) as d:
            if "TOPO" not in d.variables:
                return None
            v = float(np.asarray(d.variables["TOPO"][:]).ravel()[0])
        return None if (v == 0.0 or not np.isfinite(v)) else v
    except Exception:                                    # noqa: BLE001
        return None


def warm_start(run_dir: Path, columns: List[Dict],
               config: Dict[str, Any]) -> Dict[str, Any]:
    """A finidat per column, straight from the CONUS 1-km restarts.

    No carrier, no prior run, no template library: the CONUS restart already
    holds every variable, so one gridcell is subset out into a standalone
    single-column file. Works on a domain that has never been run.

    config['warm_start'] = True | {'conus_restart': manifest|file|dir}

    Two things that are easy to get wrong:

      * It SNAPS each column to its donor gridcell (~250-400 m). The domain,
        surfdata and finidat must agree on coordinates or ELM aborts at init on
        a surfdata/fatmgrid mismatch, so the donor's lat/lon wins and the shift
        is recorded rather than left silent.
      * It also subsets the CONUS gridcell's own surfdata and passes it on as
        the surface TEMPLATE, so fsurdat and finidat describe the same gridcell
        (ELM's check_weights gate).

    FATAL on failure. This used to return None and let the ensemble cold start,
    which sounds forgiving and is not: the start type is decided PER COLUMN
    downstream, so a partial failure produced a mixed ensemble — some columns
    warm on the donor's CONUS soil, others cold on a different soil dataset —
    inside one run, signalled by nothing but a "(cold)" in a print. Every
    cross-column comparison then spans two experiments.
    """
    run_dir = Path(run_dir)
    ws = config.get("warm_start", True)
    if isinstance(ws, str):
        ws = {"source": ws}
    elif ws is True:
        ws = {}

    try:
        import make_finidat_subset as fs
        import make_warmstart as mw
        spec = ws.get("conus_restart") or mw.DEFAULT_CONUS_MANIFEST
        bands = mw.ConusBandSet(mw.resolve_conus_sources(spec))

        print("\n🌡️  STEP 0b: Warm Start (CONUS subset)")
        print("-" * 40)
        manifest = fs.build_finidats(columns, run_dir / "warmstart", bands)
        if not manifest:
            raise RuntimeError(
                "warm start produced no finidat for any column. The CONUS "
                "restart source is unreadable or the domain lies outside its "
                "coverage.")
        missing = [c.get("id") for c in columns if c.get("id") not in manifest]
        if missing:
            raise RuntimeError(
                f"warm start covered {len(manifest)}/{len(columns)} columns; no "
                f"donor for {', '.join(missing)}. A partial warm start would "
                f"put columns with different initial states and different soil "
                f"datasets in one ensemble.")

        # Snap to the donor cell so domain/surfdata/finidat agree exactly.
        #
        # ELEVATION MOVES WITH THE COORDINATES. Until 2026-08-08 this loop set
        # lat and lon and left elevation_m at the 3DEP value sampled BEFORE the
        # snap, so every warm-started column described the station's elevation at
        # the donor's coordinates — two places in one record. It read as correct
        # and was consumed as correct: step1_compare_swe, _wtd, _streamflow and
        # step2_derive all regress on elevation_m.
        #
        # The two DEMs differ by more than rounding, and in the direction that
        # matters. 3DEP is a ~10 m point query; the donor's TOPO is the mean over
        # a CONUS 1 km gridcell, which in mountains sits tens to hundreds of
        # metres from any point inside it. Measured across 27 pinned columns:
        # |elevation - station| was 4.4 m mean / 15.9 m max against 3DEP, and
        # 41.0 m mean / 155.6 m max against the donor. The model runs the donor
        # cell, so the second pair is the true representativeness gap and the
        # first was flattering nothing.
        #
        # TOPO comes from the per-column surfdata this warm start just wrote, so
        # it is the same file ELM initialises from — no second source to drift.
        excluded = []
        for c in columns:
            m = manifest.get(c.get("id"))
            if not m:
                continue
            # TOO FAR TO BE THE SAME PLACE. The snap is meant to land on the
            # gridcell CONTAINING the sampled point, so on a 1 km grid it cannot
            # honestly exceed a half-diagonal (~0.7 km); measured across the 13
            # chain basins the worst legitimate snap was 0.571 km. A larger
            # distance means the point has no land gridcell at all and the search
            # walked to a different one.
            #
            # centralcoast_1998 col_04 is the case: sampled at 3DEP 0 m on the
            # Big Sur shoreline, where the CONUS land grid simply stops, so the
            # nearest donor was 4.52 km inland and 74 m uphill. Snapping it does
            # not relocate the column, it SUBSTITUTES a different place — and the
            # design still counted it as representing the coast.
            if m.get("dist_km") is not None and m["dist_km"] > MAX_SNAP_KM:
                excluded.append((c.get("id"),
                                 f"nearest CONUS land donor is {m['dist_km']:.2f} km "
                                 f"away (> {MAX_SNAP_KM} km) — the sampled point "
                                 f"is not on the land grid"))
                continue
            c["lat"], c["lon"] = m["donor_lat"], m["donor_lon"]
            topo = _donor_topo(m.get("surface_template"))
            if topo is None:
                excluded.append((c.get("id"), "donor surfdata carries no TOPO"))
                continue
            band = c.get("band_range_m")
            c["elevation_m"] = round(topo, 2)
            c["elevation_source"] = "conus_donor_topo"
            # The BAND is the design stratum and is deliberately not reassigned —
            # the column was chosen to represent it. But the donor can land
            # outside it, so say so rather than leave the record self-inconsistent.
            if band and not (band[0] <= topo <= band[1]):
                c["outside_design_band"] = True
                print(f"  ⚠️  {c.get('id')}: donor elevation {topo:.0f} m is "
                      f"outside its design band {band[0]}-{band[1]} m "
                      f"(kept — the band is what it was sampled to represent)")

        # A column with no donor elevation is FLAGGED AND DROPPED, not run. Its
        # surfdata would carry a zero or missing surface height into the model,
        # and one column integrating at the wrong altitude inside an ensemble is
        # the same failure this function already refuses for a partial warm
        # start: every cross-column comparison would span two experiments.
        if excluded:
            ids = [e[0] for e in excluded]
            for cid, why in excluded:
                print(f"  ⚠️  EXCLUDED {cid}: {why} — it will not be run")
            columns[:] = [c for c in columns if c.get("id") not in ids]
            manifest = {k: v for k, v in manifest.items() if k not in ids}
            manifest["_excluded"] = excluded
            if not columns:
                raise RuntimeError(
                    f"every column was excluded ({excluded}); the CONUS grid "
                    f"has no usable donor over this domain.")

        (run_dir / "warmstart" / "warmstart.json").write_text(
            json.dumps(manifest, indent=2))
        # `_excluded` is a LIST living in a dict of dicts, so anything walking
        # manifest.values() has to skip it. Nothing did, and this line would have
        # raised on the first exclusion — which never happened until the snap
        # limit above started producing them.
        kept = [m for k, m in manifest.items() if k != "_excluded"]
        snap = max((m["dist_km"] for m in kept), default=0.0)
        print(f"✓ {len(kept)}/{len(kept) + len(excluded)} column(s) warm-started "
              f"from CONUS (snapped <= {snap} km) → warmstart/warmstart.json")
        return manifest
    except Exception as e:                                      # noqa: BLE001
        raise RuntimeError(f"warm start failed: {e}") from e


def attach_donor_soil(columns: List[Dict], finidat_map: Dict[str, Any]) -> int:
    """Replace each warm-started column's soil with the donor's own.

    A warm start keeps the donor gridcell's surfdata, so any profile gathered
    at sampling time is characterisation, not what ELM runs on. columns.json
    and the design figure must show the dataset the experiment actually uses,
    or the plan on the page and the run on the machine disagree.

    This is the ONLY soil the run has.
    """
    import make_finidat_subset as fs
    n = 0
    for c in columns:
        entry = finidat_map.get(c.get("id")) or {}
        sd = entry.get("surface_template")
        if not sd or not Path(sd).exists():
            raise RuntimeError(
                f"{c.get('id')}: warm start reported a donor but its surface "
                f"template is missing ({sd}). Skipping would leave this column "
                f"with no soil while its siblings have the donor's.")
        prof = fs.donor_soil_profile(sd)
        if not prof:
            raise RuntimeError(
                f"{c.get('id')}: no soil profile readable from the donor "
                f"surface template {sd}.")
        c["soil_profile"] = prof
        c["soil_layers"] = prof["num_layers"]
        c["soil_top_texture"] = prof["layers"][0]["texture_class"]
        c["soil_source"] = "conus"
        n += 1
    if n:
        print(f"   soil for the design/figure taken from the CONUS donor cells "
              f"({n} column(s)) — the dataset the run uses")
    return n


# ─────────────────────────────────────────────────────────────────────
# SURFACES AND DOMAINS
# ─────────────────────────────────────────────────────────────────────
def build_column_inputs(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """domain.nc + surface.nc per column, before any CIME work.

    The builder generates these itself, per coupler, deep inside _build_one().
    Doing it here first changes nothing about WHAT ELM receives — the
    generators key their output on coordinates plus a content hash, so the
    builder's calls become cache hits on the very files written here.

    What it changes is WHEN you find out. Surface generation reads the CONUS
    donor and can fail for a column; buried in _build_one that surfaces partway
    through case creation, after CIME work has begun. Here it fails before
    anything expensive starts and names the column.

    Non-fatal by design: the builder retains its own generation path, so a
    failure here costs the early warning and the manifest, not the run.
    """
    try:
        import build_column_inputs as bci
        res = bci.build_all(
            Path(run_dir),
            soil_config=config.get("soil_config", "native"),
            substrate=config.get("substrate", "extrapolate"),
            quiet=True,
        )
        n_ok, n_bad = len(res.get("built") or {}), len(res.get("failed") or {})
        print(f"✓ column inputs: {n_ok} built"
              + (f", {n_bad} FAILED" if n_bad else "")
              + ("  (warm)" if res.get("warm_started") else "  (cold)"))
        for cid, why in (res.get("failed") or {}).items():
            print(f"   ⚠️  {cid}: {why}")
        return res
    except Exception as e:                                      # noqa: BLE001
        print(f"   ⚠️  pre-building column inputs failed ({e}) — the builder "
              f"will generate them per column instead")
        return {}


# ─────────────────────────────────────────────────────────────────────
# THE RUN PLAN AND THE CASE LIST
# ─────────────────────────────────────────────────────────────────────
def to_run_plan(columns: List[Dict], config: Dict[str, Any],
                finidat_map: Optional[Dict] = None) -> Dict[str, Any]:
    """Columns → CONDITIONS_COUPLERS.

    FINIDAT is a per-coupler key the builder reads at build time, which is why
    the warm start has to happen before this rather than inside it.
    """
    yr_start = int(config.get("yr_start", 1995))
    yr_end = int(config.get("yr_end", yr_start))
    return columns_to_elm_plan(
        columns,
        yr_start=yr_start,
        yr_end=yr_end,
        soil_config=config.get("soil_config", "native"),
        substrate=config.get("substrate", "extrapolate"),
        finidat_map=finidat_map or {},
        period_source=(config.get("period_source")
                       or ("reception" if config.get("yr_start") else "DEFAULT")),
    )


def build_case_inputs(run_dir: Path, plan: Dict[str, Any],
                      config: Dict[str, Any]) -> Tuple[List[Dict], Any]:
    """The ELMAgentAdapter list, and the builder that made it.

    The builder handle is returned rather than discarded because a caller that
    goes on to build cases needs it for the --keepexe fast path; a caller that
    only wants the case list ignores it.
    """
    run_dir = Path(run_dir)
    build_column_inputs(run_dir, config)

    builder = ELMExperimentBuilder(plan)
    experiments = builder.build_experiments()

    inputs_dir = run_dir / "01_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "experiment_summary.json").write_text(
        json.dumps(builder.get_experiment_summary(), indent=2, default=str))

    print(f"✓ {len(experiments)} experiment(s) built")
    return experiments, builder


def serialise_case_inputs(experiments: List[Dict]) -> List[Dict]:
    """Each case as data, with the adapter replaced by what BUILT it.

    runtime_config is the adapter's whole input — FSURDAT, FINIDAT, the domain
    paths, STOP_N, the forcing years — so the far side can construct an
    identical adapter without the object ever crossing. That config is also
    where the warm start's two products (the subset restart and the donor-soil
    surface) leave as plain paths.
    """
    out = []
    for e in experiments:
        row = {k: e.get(k) for k in CASE_KEYS if k in e}
        rc = getattr(e.get("elm_agent"), "runtime_config", None)
        if isinstance(rc, dict):
            row["runtime_config"] = dict(rc)
        elif isinstance(e.get("runtime_config"), dict):
            row["runtime_config"] = dict(e["runtime_config"])
        out.append(row)
    return out


def write_case_inputs(run_dir: Path, rows: List[Dict]) -> Path:
    """01_inputs/case_inputs.json, with the data provenance alongside it.

    ASSERTED, not hoped: a case with no runtime_config names nothing the build
    needs. Deleting build_elm_cases once took the runtime_config extraction with
    it and a job was submitted against a file that existed, parsed, and was
    empty of every path — so this refuses to write one.
    """
    run_dir = Path(run_dir)
    bad = [r.get("case_name") for r in rows if not (r.get("runtime_config") or {})]
    if bad:
        raise RuntimeError(
            f"these cases have no runtime_config, so nothing could be built "
            f"from them: {bad}")

    inputs_dir = run_dir / "01_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    path = inputs_dir / CASE_INPUTS
    path.write_text(json.dumps(rows, indent=2, default=str))
    return path


# ─────────────────────────────────────────────────────────────────────
# THE WHOLE THING — what the MCP tool calls
# ─────────────────────────────────────────────────────────────────────
def build_from_location(run_dir: Path, columns: List[Dict],
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """Locations → runnable ELM inputs. Returns JSON-able data only.

    The caller persists the returned columns: they are SNAPPED, and the
    pre-snap coordinates describe a run that will not happen.
    """
    run_dir = Path(run_dir)
    finidat_map = warm_start(run_dir, columns, config)
    attach_donor_soil(columns, finidat_map)

    plan = to_run_plan(columns, config, finidat_map)
    experiments, _builder = build_case_inputs(run_dir, plan, config)
    rows = serialise_case_inputs(experiments)
    path = write_case_inputs(run_dir, rows)

    import paths as P
    return {
        "ok": True,
        "n_columns": len(columns),
        "n_cases": len(rows),
        "columns": columns,
        # The plan this call built on its way through. Returned rather than
        # discarded because the FRAMEWORK persists it as run_plan.json and
        # resumes a run from it. Building it here and rebuilding it there would
        # be the same computation twice with two chances to disagree; not
        # returning it at all would delete a resume path as a side effect of
        # moving a boundary, which is not a decision this change gets to make.
        "run_plan": plan,
        "case_inputs_path": str(path),
        "runtime_config_keys": sorted(
            {k for r in rows for k in (r.get("runtime_config") or {})}),
        # Which CONUS restart every column warm-started from. A fact about the
        # science, not a configuration detail: without it a claim cannot be
        # traced to the data it rests on.
        "data_provenance": P.provenance(),
        "assumptions_ledger": plan.get("assumptions_ledger") or [],
    }
