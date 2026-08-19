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
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from columns_to_plan import columns_to_elm_plan
from elm_experiment_builder import ELMExperimentBuilder

# parents[3] is the framework root. keyset imports nothing but the standard
# library; it is here so this list and the framework's obey one rule.
_FW = Path(__file__).resolve().parents[3]
if str(_FW / "src") not in sys.path:
    sys.path.insert(0, str(_FW / "src"))
from core import keyset                                        # noqa: E402
from core.keyset import KeySet                                  # noqa: E402

# Keys of an experiment that are plain data and mean something downstream.
# LISTED, not "everything except elm_agent": a new object added upstream would
# otherwise silently become a repr in the case-inputs file.
#
# And now the list ANSWERS FOR EVERY KEY. The reason this one has stayed
# correct is that it has exactly two producers, both in this repo; that is not
# a property the list had, it is luck it kept having. A key the builder adds
# that nobody classified now stops the build instead of vanishing into it.
_CASE = KeySet(
    "CASE_KEYS",
    keep = ("scenario_index", "scenario_name", "case_name", "forcing_period",
            "soil_config", "substrate", "forcing_start", "forcing_end",
            "stop_n", "start_date", "description", "lat", "lon",
            "elevation_m", "band", "case_dir", "prescribed_weather"),
    drop = {
        "elm_agent": "THE OBJECT THIS LIST EXISTS TO STOP. The adapter is a "
                     "live handle; serialised it becomes a repr string in an "
                     "artifact meant to be re-read. What the far side needs "
                     "from it is runtime_config, taken by name below",
        "runtime_config": "taken explicitly below, off the adapter when the "
                          "adapter is present and off the experiment when it "
                          "is not — two sources for one field, which a "
                          "comprehension cannot express",
    },
    source = "ELMExperimentBuilder.build_experiments() -> experiment dicts",
    where  = "mcp/elm-mcp/src/inputs.py :: CASE_KEYS",
)
CASE_KEYS = _CASE.keep

CASE_INPUTS = "case_inputs.json"


# ─────────────────────────────────────────────────────────────────────
# WARM START
# ─────────────────────────────────────────────────────────────────────
# The CONUS restart grid is 1 km, so a point inside a cell is at most a
# half-diagonal (~0.71 km) from its centre. 1.5 km is twice that: comfortably
# past any grid irregularity, and far short of the 4.52 km that says the point
# was never on the land grid.
MAX_SNAP_KM = 1.5


# ── the forcing grid ─────────────────────────────────────────────────────────
# NLDAS-2 is a regular 1/8 degree grid; its cell edges fall on multiples of
# 0.125 from 25.0 N and -125.0 W. Every column inside one cell is driven by the
# SAME rain, snow and temperature — which is why two columns in one cell that
# partition water differently differ because of soil or terrain, not weather.
#
# STATED, NOT MEASURED. These constants are the published NLDAS-2 geometry, not
# something read out of the forcing files. Checked against 30 columns across
# Naches and Brandywine: grouping by this cell reproduced the grouping by annual
# precipitation EXACTLY, every group, both basins. If the forcing ever changes,
# this is wrong silently, so it is written down rather than inferred.
NLDAS_DEG = 0.125
NLDAS_LAT0, NLDAS_LON0 = 25.0, -125.0


def _nldas_cell(lat, lon):
    """(row, col) of the NLDAS-2 cell containing this point, or None."""
    try:
        return (math.floor((float(lat) - NLDAS_LAT0) / NLDAS_DEG),
                math.floor((float(lon) - NLDAS_LON0) / NLDAS_DEG))
    except (TypeError, ValueError):
        return None


def _soil_summary(prof: Dict[str, Any]) -> Dict[str, Any]:
    """The donor profile in one readable line's worth of numbers.

    THE PROFILE ITSELF STAYS — this is not a replacement for `soil_profile`,
    which keeps every layer. It exists because ten nested layers per column is
    bulky to put in front of a reader or a prompt, where "loam, clay 20-25%, 10
    layers to 380 cm" is one line and says what differs between columns.

    NO KSAT. The old soil attribution wanted `ksat_min_ums` as its second
    predictor and no such field exists anywhere in the surface dataset this
    reads; asking for it is what left that analysis half-configured.

    NOTHING ABOUT FORCING IS IN HERE. Which cell a column is driven by is a
    separate fact with a separate field — see _nldas_cell. Keeping them apart is
    the point: the old `_compute_soil_attribution` summarised soil, binned by
    precipitation and correlated against results in one function, and could not
    be repaired one piece at a time.
    """
    layers = (prof or {}).get("layers") or []
    if not layers:
        return {}

    def rng(key):
        vals = [l.get(key) for l in layers if l.get(key) is not None]
        return [round(min(vals), 1), round(max(vals), 1)] if vals else None

    out = {"texture_top": layers[0].get("texture_class"),
           "n_layers": len(layers)}
    for key, name in (("clay_pct", "clay_pct"), ("sand_pct", "sand_pct"),
                      ("gravel_pct", "gravel_pct")):
        r = rng(key)
        if r:
            out[name] = r
    bot = [l.get("depth_bot_cm") for l in layers
           if l.get("depth_bot_cm") is not None]
    if bot:
        out["depth_cm"] = round(max(bot), 1)
    textures = [l.get("texture_class") for l in layers if l.get("texture_class")]
    if textures:
        # A column whose texture changes with depth is a different object from
        # one that does not, and the top layer alone cannot say which it is.
        out["textures"] = sorted(set(textures))
    return out


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

    # COLD ON PURPOSE, and only when asked in so many words. `False` is the one
    # value that means "do not touch the CONUS restart" — a controlled sweep
    # sets it, because a study that depends on no real place must not be handed
    # a real gridcell's water content.
    #
    # NOT THE SAME AS THE FAILURE PATH described above. That one produced a
    # MIXED ensemble from a partial failure and is still fatal. This returns an
    # empty map for EVERY column, so every column is cold and no comparison
    # spans two kinds of start.
    if ws is False:
        print("\n🧊 STEP 0b: Cold start — no CONUS restart")
        print("-" * 40)
        print(f"   {len(columns)} column(s) start from ELM's own defaults.")
        print("   No donor gridcell, so no snapping, no borrowed soil and no "
              "borrowed water content.")
        print("   COST: a cold column takes years to forget those defaults. A "
              "short run reports")
        print("   the initialisation as much as the soil — recorded as a "
              "caveat, not fixed here.")
        for c in columns:
            # Stated, not left absent. `elevation_m` is already None on a
            # conceptual column, and a reader finding no start type at all
            # cannot tell a cold column from one nobody decided about.
            c["warm_start"] = False
        return {}

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
            # WHICH FORCING CELL THIS COLUMN ENDS UP IN. Recorded HERE, after
            # the snap, because the snap moves the column by up to ~0.6 km and
            # can carry it across a cell edge; the cell computed before the snap
            # would be the cell of a place that is not run.
            c["forcing_cell"] = _nldas_cell(c["lat"], c["lon"])
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

        # ── A COUPLED FOLLOW-UP OVERRIDES THE DONOR'S WATER TABLE ────────
        # A column carrying `initial_water_table_m` (the driving model's
        # solved water table, written by the manager's coupling branch) gets
        # it stamped into its fresh finidat — ZWT and WA kept consistent by
        # ELM's own relation, clamps and the in-soil approximation reported
        # on the column (set_water_table.py). After the subset and before
        # anything reads the file, so the case is BUILT on the edited state.
        import set_water_table as _swt
        for c in columns:
            wt_target = c.get("initial_water_table_m")
            m = manifest.get(c.get("id"))
            if wt_target is None or not m:
                continue
            r = _swt.apply(m["finidat"], float(wt_target))
            c["initial_water_table_written_m"] = r["written_m"]
            c["initial_water_table_note"] = r["note"]
            m["water_table_override"] = {k: r[k] for k in
                                         ("requested_m", "written_m", "regime",
                                          "clamped", "old_zwt_m")}

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
    n = kept = 0
    for c in columns:
        # A PRESCRIBED SOIL IS THE EXPERIMENT, AND MUST SURVIVE THIS.
        #
        # Everything above is right for a site run, where the profile gathered
        # at sampling time is characterisation and the donor's is what ELM
        # runs on. A conceptual sweep inverts that: the soil was CHOSEN, it is
        # the independent variable, and the columns deliberately sit at one
        # lat/lon.
        #
        # So without this guard every column in a soil sweep receives the SAME
        # donor profile, the gradient is erased, every column runs, every check
        # passes, and the analyzer reports a clean n=7 for a sweep with nothing
        # varying in it. A site column never sets this field and can never
        # reach this branch.
        if c.get("soil_source") == "prescribed":
            kept += 1
            continue
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
        # The same profile, small enough to read. See _soil_summary: the layers
        # stay where they are; this is for anything that has to SHOW the soil.
        c["soil_summary"] = _soil_summary(prof)
        n += 1
    if n:
        print(f"   soil for the design/figure taken from the CONUS donor cells "
              f"({n} column(s)) — the dataset the run uses")
    if kept:
        # SAID OUT LOUD, because silence here is indistinguishable from the bug
        # this guard exists to prevent.
        print(f"   soil KEPT AS PRESCRIBED for {kept} column(s) — a controlled "
              f"sweep, so the donor's profile is not applied to them")
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
        # check(), not take(): a key that is ABSENT is omitted here rather than
        # written as null, and write_case_inputs asserts on what is present.
        # The assertion is about keys the builder wrote, not keys it did not.
        _CASE.check(e)
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
    # The build side's own audit. It runs in the SERVER process, so the
    # framework's report at packaging time never sees these lists — a list
    # that asks for a key the builder stopped writing has to say so here or
    # nowhere.
    keyset.report()
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
