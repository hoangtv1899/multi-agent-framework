#!/usr/bin/env python3
"""
CONUS restart -> single-column ELM `finidat`, by SUBSETTING. No carrier file,
no prior run, no template.

A CONUS 1-km ELM restart holds every gridcell in a latitude band. This extracts
ONE gridcell — its landunits, columns and PFTs — into a standalone file the next
ELM case can start from, so a brand-new domain can be warm-started directly.

Why this works (all verified against the real files):

  * The CONUS restarts were written with create_crop_landunit=.false.
    (global attrs cft_lb=17, cft_ub=16), so a land gridcell is
        4 landunits / 16 columns / 32 pfts,  cols1d_ityplun = [1, 7x5, 8x5, 9x5]
    which is EXACTLY what our cases build once elm_wrapper sets the same flag.
  * Each gridcell's landunit / column / pft rows are CONTIGUOUS, so the subset
    is three hyperslabs, not a gather.
  * ELM reads finidat through restFile_read; on a startup run a variable missing
    from the file is logged and skipped rather than fatal (only nsrContinue
    aborts). The one variable our build writes that CONUS lacks is ENDWB_COL,
    which is allocated to NaN and has no consumer.

What ELM *does* hard-check, and what this tool therefore validates before
writing anything:

  * restFile_dimcheck -> check_dim aborts unless gridcell/topounit/landunit/
    column/pft match what ELM builds from fsurdat. Hence the layout assertions.
  * check_weights (subgridRestMod) aborts unless the finidat's natveg PFT
    weights match fsurdat within 5e-3. The finidat's weights OVERWRITE the
    fsurdat-derived ones, so the two files must describe the same vegetation —
    which is why the surface data should be subset from the SAME CONUS gridcell
    (--surfdata), not from a coarser global file.

    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/make_finidat_subset.py --run-dir <dir> [--conus-restart MANIFEST]
    python3 tools/make_finidat_subset.py --lat 46.736 --lon -120.835 \
        --conus-restart <band.nc> --out /tmp/finidat_col_01.nc
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

# Sub-grid dimensions that get sliced; everything else (levgrnd, levsno, ...)
# is copied whole.
SUBGRID_DIMS = ("gridcell", "topounit", "landunit", "column", "pft")

# Index vectors whose VALUES point at another sub-grid level and must be
# renumbered into the 1-gridcell frame. Everything here is 1-BASED on both
# sides — verified: min==1 and max==dimlen for every one of them in the donor,
# and the single-column files ELM writes use the same convention.
#   name -> None            => set to 1 (everything now lives in gridcell 1)
#   name -> level           => remap global 1-based id to local 1-based position
RENUMBER = {
    "grid1d_ixy": None, "grid1d_jxy": None,
    "topo1d_ixy": None, "topo1d_jxy": None, "topo1d_gridcell_index": None,
    "land1d_ixy": None, "land1d_jxy": None,
    "land1d_gridcell_index": None, "land1d_topounit_index": None,
    "cols1d_ixy": None, "cols1d_jxy": None,
    "cols1d_gridcell_index": None, "cols1d_topounit_index": None,
    "cols1d_landunit_index": "landunit",
    "pfts1d_ixy": None, "pfts1d_jxy": None,
    "pfts1d_gridcell_index": None, "pfts1d_topounit_index": None,
    "pfts1d_landunit_index": "landunit",
    "pfts1d_column_index": "column",
}

# Landunit types that make a gridcell unusable as a donor for our 16-column
# layout: lake (5), landice (3/4), wetland (6) each add a landunit+column on
# one side only, and check_dim would abort. ~6% of CONUS gridcells.
EXTRA_LANDUNITS = {3, 4, 5, 6}
EXPECTED_ITYPLUN = [1] + [7] * 5 + [8] * 5 + [9] * 5


class SubsetError(RuntimeError):
    """Raised before anything is written — never leave a half-valid finidat."""


# ─────────────────────────────────────────────────────────────────────────────
# Index-vector cache.
#
# plan_subset() needs six whole index vectors -- grid1d_lat/lon plus the four
# *1d_gridcell_index arrays -- to locate one gridcell. They describe the BAND,
# not the column, so they are identical for every column drawn from the same
# restart, and re-reading them per column is the dominant cost of a warm start:
# hundreds of MB per column off Lustre, for eleven columns, to extract 0.2 MB
# each. Cached per file, they are read once per band instead.
#
# Only ONE file's vectors are held at a time. build_finidats() visits columns
# band by band, so a single slot is enough, and the arrays are large enough
# that keeping every band would matter on a login node.
# ─────────────────────────────────────────────────────────────────────────────
_IDX_CACHE = {"path": None, "vars": {}}

# Columns are subset in parallel; see build_finidats. Four keeps several seeks
# in flight without turning a login node into an I/O storm.
DEFAULT_WORKERS = int(os.environ.get("IDEAS_WARMSTART_WORKERS", "4"))


def _idx(d, var):
    """Whole index vector `var`, read once per restart file."""
    import numpy as np
    path = d.filepath()
    if _IDX_CACHE["path"] != path:
        _IDX_CACHE["path"], _IDX_CACHE["vars"] = path, {}
    hit = _IDX_CACHE["vars"].get(var)
    if hit is None:
        hit = np.asarray(d.variables[var][:])
        _IDX_CACHE["vars"][var] = hit
    return hit


def _locate(d, lat, lon):
    """Nearest gridcell to (lat, lon); returns (g0, dist_km, glat, glon)."""
    import numpy as np
    glat = _idx(d, "grid1d_lat")
    glon = np.where(_idx(d, "grid1d_lon") > 180,
                    _idx(d, "grid1d_lon") - 360, _idx(d, "grid1d_lon"))
    g0 = int(np.argmin((np.asarray(glat) - lat) ** 2 + (glon - lon) ** 2))
    dist = 111.0 * float(np.hypot(glat[g0] - lat,
                                  (glon[g0] - lon) * np.cos(np.radians(lat))))
    return g0, dist, float(glat[g0]), float(glon[g0])


def _rows_for(d, var, g1):
    """Contiguous row range of `var`'s level belonging to 1-based gridcell g1.

    A linear scan, deliberately. Binary search looks like the obvious win on a
    20M-row vector, but it must first prove the vector non-decreasing, and
    `np.all(np.diff(a) >= 0)` allocates a second 20M array: measured at 175 ms
    for a 14-column run against 140 ms for simply scanning. Scanning was never
    the bottleneck — READING the vector was, and _idx now does that once per
    band instead of once per column.
    """
    import numpy as np
    idx = np.where(_idx(d, var) == g1)[0]
    if idx.size == 0:
        raise SubsetError(f"gridcell {g1} has no rows in {var}")
    if not np.all(np.diff(idx) == 1):
        raise SubsetError(
            f"{var} rows for gridcell {g1} are not contiguous — this tool "
            f"hyperslabs and would silently mix gridcells")
    return int(idx[0]), int(idx[-1]) + 1          # [start, stop)


def plan_subset(d, lat, lon):
    """Work out the slices for one gridcell and check it can host our layout."""
    import numpy as np
    g0, dist_km, glat, glon = _locate(d, lat, lon)
    g1 = g0 + 1
    sl = {"gridcell": (g0, g0 + 1)}
    sl["topounit"] = _rows_for(d, "topo1d_gridcell_index", g1) \
        if "topo1d_gridcell_index" in d.variables else (g0, g0 + 1)
    sl["landunit"] = _rows_for(d, "land1d_gridcell_index", g1)
    sl["column"]   = _rows_for(d, "cols1d_gridcell_index", g1)
    sl["pft"]      = _rows_for(d, "pfts1d_gridcell_index", g1)

    ityplun = [int(x) for x in
               np.asarray(d.variables["cols1d_ityplun"][sl["column"][0]:sl["column"][1]])]
    extra = sorted(set(ityplun) & EXTRA_LANDUNITS)
    if extra:
        raise SubsetError(
            f"donor gridcell {g0} carries landunit type(s) {extra} "
            f"(lake/landice/wetland), giving {len(ityplun)} columns. Our cases "
            f"build {len(EXPECTED_ITYPLUN)}; ELM's check_dim would abort. Pick a "
            f"different point or handle the extra landunit explicitly.")
    if ityplun != EXPECTED_ITYPLUN:
        raise SubsetError(
            f"donor gridcell {g0} cols1d_ityplun={ityplun}, expected "
            f"{EXPECTED_ITYPLUN}")
    # ixy/jxy locate this gridcell in the CONUS 2-D mesh. They are overwritten
    # with 1 in the output, so capture them here — they are how the matching
    # surfdata cell is found.
    ixy = int(np.ravel(np.asarray(d.variables["grid1d_ixy"][g0:g0 + 1]))[0])
    jxy = int(np.ravel(np.asarray(d.variables["grid1d_jxy"][g0:g0 + 1]))[0])
    return {"g0": g0, "g1": g1, "slices": sl, "dist_km": round(dist_km, 3),
            "donor_lat": round(glat, 6), "donor_lon": round(glon, 6),
            "donor_ixy": ixy, "donor_jxy": jxy,
            "n_landunit": sl["landunit"][1] - sl["landunit"][0],
            "n_column":   sl["column"][1] - sl["column"][0],
            "n_pft":      sl["pft"][1] - sl["pft"][0]}


def _local_map(d, index_var, level, sl):
    """Global 1-based ids of `level` within this gridcell -> local 1-based."""
    import numpy as np
    lo, hi = sl[level]
    return {gid: i + 1 for i, gid in enumerate(range(lo + 1, hi + 1))}


def write_subset(conus_restart, lat, lon, out_path, quiet=False):
    """Write a single-gridcell finidat; returns a provenance dict."""
    import netCDF4
    import numpy as np
    say = (lambda *a: None) if quiet else print

    d = netCDF4.Dataset(conus_restart)
    try:
        plan = plan_subset(d, lat, lon)
        sl = plan["slices"]
        say(f"    donor gridcell {plan['g0']} @ "
            f"({plan['donor_lat']}, {plan['donor_lon']}) {plan['dist_km']} km; "
            f"{plan['n_landunit']} landunit / {plan['n_column']} column / "
            f"{plan['n_pft']} pft")

        lu_map  = {gid: i + 1 for i, gid in
                   enumerate(range(sl["landunit"][0] + 1, sl["landunit"][1] + 1))}
        col_map = {gid: i + 1 for i, gid in
                   enumerate(range(sl["column"][0] + 1, sl["column"][1] + 1))}
        maps = {"landunit": lu_map, "column": col_map}

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(out, "w", format="NETCDF3_64BIT_OFFSET") as o:
            for name, dim in d.dimensions.items():
                o.createDimension(
                    name, (sl[name][1] - sl[name][0]) if name in sl else len(dim))

            for vname, var in d.variables.items():
                dims = var.dimensions
                # _FillValue must be set at creation time, not after.
                fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                ov = o.createVariable(vname, var.datatype, dims, fill_value=fill)
                for a in var.ncattrs():
                    if a != "_FillValue":
                        ov.setncattr(a, var.getncattr(a))

                # Slice on the FIRST dimension when it is a sub-grid level.
                # Slice by the variable's actual dims, never by its name:
                # pfts1d_topoglc is dimensioned 'column', not 'pft'
                # (subgridRestMod.F90 upstream bug).
                if dims and dims[0] in sl:
                    lo, hi = sl[dims[0]]
                    data = var[lo:hi, ...]
                else:
                    data = var[...]

                if vname in RENUMBER:
                    target = RENUMBER[vname]
                    arr = np.asarray(data).copy()
                    if target is None:
                        arr[...] = 1
                    else:
                        m = maps[target]
                        arr = np.vectorize(lambda v: m[int(v)])(arr)
                    data = arr

                # grid1d_lon is 0..360 in CONUS; single-column files use -180..180
                if vname == "grid1d_lon":
                    data = np.where(np.asarray(data) > 180,
                                    np.asarray(data) - 360, data)
                ov[...] = data

            for a in d.ncattrs():
                o.setncattr(a, d.getncattr(a))
            o.setncattr("finidat_subset_source", str(conus_restart))
            o.setncattr("finidat_subset_gridcell", plan["g0"])
            o.setncattr("finidat_subset_target", f"lat={lat}, lon={lon}")
            o.setncattr("generated_by", "tools/make_finidat_subset.py")
    finally:
        d.close()

    plan["finidat"] = str(out.resolve())
    plan["source"] = "conus-subset"
    return plan


def write_surfdata_subset(conus_surfdata, ixy, jxy, out_path, quiet=False):
    """Subset the matching CONUS 1-km surfdata to the same single gridcell.

    Two ELM gates make this necessary rather than optional:

      surfrdMod.F90:1050-1062 — a surfdata carrying a `cft` dimension with
        create_crop_landunit=.false. is a hard endrun. The CONUS 1-km files are
        the old PFT format (natpft=17, no cft), so they satisfy it; our previous
        0.5 degree template does not.
      subgridRestMod check_weights — the finidat's natveg PFT weights overwrite
        the fsurdat-derived ones and must agree within 5e-3. Taking BOTH files
        from the same gridcell makes them agree by construction, instead of
        needing check_finidat_pct_consistency=.false. and two files that
        describe different vegetation.

    Soil texture is NOT touched here — elm_surface_generator still overwrites
    PCT_SAND/PCT_CLAY/ORGANIC from SSURGO, which is the per-column science.

    ixy/jxy are the 1-based mesh indices from the restart's grid1d_ixy/jxy.
    """
    import netCDF4
    import numpy as np
    say = (lambda *a: None) if quiet else print

    d = netCDF4.Dataset(conus_surfdata)
    try:
        j, i = jxy - 1, ixy - 1                     # -> 0-based (lsmlat, lsmlon)
        nlat, nlon = len(d.dimensions["lsmlat"]), len(d.dimensions["lsmlon"])
        if not (0 <= j < nlat and 0 <= i < nlon):
            raise SubsetError(
                f"ixy/jxy ({ixy},{jxy}) outside this surfdata mesh "
                f"({nlon}x{nlat}) — restart and surfdata are different bands")

        lat0 = float(np.asarray(d.variables["LATIXY"][j, i]))
        lon0 = float(np.asarray(d.variables["LONGXY"][j, i]))
        say(f"    surfdata cell (ixy={ixy}, jxy={jxy}) @ ({lat0:.4f}, {lon0:.4f})")

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(out, "w", format="NETCDF3_64BIT_OFFSET") as o:
            for name, dim in d.dimensions.items():
                # lat/lon are 1-D coordinate vectors over the mesh; they
                # collapse with it. Everything else (natpft, nlevsoi, ...) is
                # structural and copied whole.
                n = 1 if name in ("lsmlat", "lsmlon", "lat", "lon") else len(dim)
                o.createDimension(name, n)

            for vname, var in d.variables.items():
                dims = var.dimensions
                fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                ov = o.createVariable(vname, var.datatype, dims, fill_value=fill)
                for a in var.ncattrs():
                    if a != "_FillValue":
                        ov.setncattr(a, var.getncattr(a))

                if dims[-2:] == ("lsmlat", "lsmlon"):
                    sub = var[..., j:j + 1, i:i + 1]
                    # CONUS stores longitude 0..360; this pipeline's domain
                    # files (elm_domain_generator) and generated surfaces use
                    # -180..180. If they disagree ELM aborts at init on a
                    # surfdata/fatmgrid mismatch.
                    if vname == "LONGXY":
                        sub = np.where(np.asarray(sub) > 180,
                                       np.asarray(sub) - 360, sub)
                    ov[...] = sub
                elif dims == ("lat",):
                    ov[...] = np.array([lat0], dtype=var.datatype)
                elif dims == ("lon",):
                    ov[...] = np.array([lon0 - 360 if lon0 > 180 else lon0],
                                       dtype=var.datatype)
                elif dims == ("lsmlat",):
                    ov[...] = var[j:j + 1]
                elif dims == ("lsmlon",):
                    ov[...] = var[i:i + 1]
                else:
                    ov[...] = var[...]

            for a in d.ncattrs():
                o.setncattr(a, d.getncattr(a))
            o.setncattr("surfdata_subset_source", str(conus_surfdata))
            o.setncattr("surfdata_subset_ixy_jxy", f"{ixy},{jxy}")
            o.setncattr("generated_by", "tools/make_finidat_subset.py")
    finally:
        d.close()
    return {"fsurdat": str(out.resolve()), "lat": lat0, "lon": lon0,
            "ixy": ixy, "jxy": jxy}


def conus_surfdata_for(conus_restart):
    """The CONUS surfdata that pairs with this restart band.

    The restart names its own surfdata in the `surface_dataset` global
    attribute — authoritative, so the band is never guessed from a filename.

    That file is NOT always the one to use, though. The CONUS runs used the
    `..._monthly_2001_2019_LAI_SAI_v2.nc` products, whose MONTHLY_LAI is
    5-D (month, YEAR, pft, lat, lon) — time-varying LAI. Our ELM build expects
    the 4-D (month, pft, lat, lon) form, the same shape as our own SP template.
    So we take the standard-format sibling in the same directory, and verify
    the shape rather than trusting the name.
    """
    import glob
    import re

    import netCDF4
    with netCDF4.Dataset(conus_restart) as d:
        named = d.getncattr("surface_dataset") if "surface_dataset" in d.ncattrs() else None
    if not named:
        raise SubsetError(f"{Path(conus_restart).name} has no surface_dataset attribute")

    named = Path(named)
    m = re.search(r"(lat\d+)", named.name)
    if not m:
        raise SubsetError(f"cannot read band from surfdata name {named.name}")
    band = m.group(1)

    cands = sorted(glob.glob(str(named.parent /
                   f"surfdata_conus_1k_small_{band}_with_fdrain_and_fc_c*.nc")))
    for p in reversed(cands):                       # newest datestamp first
        try:
            with netCDF4.Dataset(p) as s:
                if "MONTHLY_LAI" in s.variables and s["MONTHLY_LAI"].ndim == 4:
                    return p
        except OSError:
            continue
    raise SubsetError(
        f"no standard-format CONUS surfdata for band {band} in {named.parent} "
        f"(need 4-D MONTHLY_LAI; {len(cands)} candidate(s) checked)")


def check_weights_agree(finidat, fsurdat, tol=5e-3):
    """Reproduce ELM's check_weights gate before submitting a job.

    subgridRestMod compares the finidat's natveg PFT weights against the
    fsurdat's PCT_NAT_PFT/100 and calls endrun beyond `tol`. Catching it here
    costs a second; catching it in ELM costs a queue slot.
    """
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(finidat) as f:
        ityplun = np.asarray(f.variables["pfts1d_ityplun"][:])
        w = np.asarray(f.variables["pfts1d_wtlnd"][:])[ityplun == 1]
    with netCDF4.Dataset(fsurdat) as s:
        p = np.ravel(np.asarray(s.variables["PCT_NAT_PFT"][:])) / 100.0
    n = min(len(w), len(p))
    if len(w) != len(p):
        return [f"natveg pft count differs: finidat {len(w)} vs fsurdat {len(p)}"]
    diff = float(np.max(np.abs(w[:n] - p[:n])))
    return [] if diff <= tol else [
        f"check_weights would abort: max |finidat - fsurdat| = {diff:.5f} > {tol}"]


def validate(finidat, expect_columns=16, expect_landunits=4, expect_pfts=32):
    """Re-open a written finidat and assert what ELM will check."""
    import netCDF4
    import numpy as np
    problems = []
    with netCDF4.Dataset(finidat) as d:
        got = {k: len(v) for k, v in d.dimensions.items() if k in SUBGRID_DIMS}
        if got.get("gridcell") != 1:
            problems.append(f"gridcell={got.get('gridcell')} != 1")
        for k, want in (("landunit", expect_landunits), ("column", expect_columns),
                        ("pft", expect_pfts)):
            if got.get(k) != want:
                problems.append(f"{k}={got.get(k)} != {want}")
        ityplun = [int(x) for x in np.asarray(d.variables["cols1d_ityplun"][:])]
        if ityplun != EXPECTED_ITYPLUN:
            problems.append(f"cols1d_ityplun={ityplun}")
        for v, target in RENUMBER.items():
            if v not in d.variables:
                continue
            arr = np.asarray(d.variables[v][:])
            hi = {None: 1, "landunit": expect_landunits,
                  "column": expect_columns}[target]
            if arr.size and (arr.min() < 1 or arr.max() > hi):
                problems.append(f"{v} out of range [1,{hi}]: "
                                f"[{arr.min()},{arr.max()}]")
        lon = float(np.ravel(d.variables["grid1d_lon"][:])[0])
        if lon > 180:
            problems.append(f"grid1d_lon={lon} still 0-360")
    return problems


def main():
    ap = argparse.ArgumentParser(
        description="Subset a CONUS ELM restart into a single-column finidat")
    ap.add_argument("--run-dir", help="run dir with columns.json (batch mode)")
    ap.add_argument("--lat", type=float, help="single-point mode")
    ap.add_argument("--lon", type=float)
    ap.add_argument("--out", help="output path (single-point mode)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--conus-restart", default=None,
                    help="band MANIFEST (default), a single .nc, or a directory")
    args = ap.parse_args()

    mw = _load_make_warmstart()
    spec = args.conus_restart or mw.DEFAULT_CONUS_MANIFEST
    bands = mw.ConusBandSet(mw.resolve_conus_sources(spec))

    if args.lat is not None:
        band, src = bands.for_lat(args.lat)
        if src is None:
            sys.exit(f"no CONUS band covers lat {args.lat}")
        out = args.out or f"finidat_{args.lat:.4f}_{args.lon:.4f}.nc"
        info = write_subset(src.d.filepath(), args.lat, args.lon, out)
        problems = validate(info["finidat"])
        print(json.dumps(info, indent=2))
        sys.exit("VALIDATION FAILED: " + "; ".join(problems) if problems else 0)

    if not args.run_dir:
        sys.exit("need --run-dir or --lat/--lon")
    run_dir = Path(args.run_dir)
    cols = json.loads((run_dir / "columns.json").read_text())
    cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "warmstart"

    manifest = build_finidats(cols, out_dir, bands)
    if not manifest:
        sys.exit("no finidat files produced")
    (out_dir / "warmstart.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(cols)} finidat files -> {out_dir}/")


def build_finidats(columns, out_dir, bands, quiet=False, workers=None):
    """{col_id: entry} for every column a CONUS band can serve.

    Per column, subsets BOTH the restart (-> finidat) and the matching CONUS
    surfdata (-> surface template) at the same gridcell, then re-checks ELM's
    two gates locally before returning.

    Each entry also carries `donor_lat`/`donor_lon`. The caller SNAPS the column
    to those: the donor cell is up to ~700 m from the expander's nominal point,
    and domain / surfdata / finidat must agree on coordinates or ELM aborts at
    init on a surfdata/fatmgrid mismatch.

    Manifest shape matches make_warmstart's, so the existing
    columns_to_plan -> CONDITIONS_COUPLERS -> user_nl_elm path is unchanged.
    """
    say = (lambda *a: None) if quiet else print
    out_dir = Path(out_dir)
    manifest, skipped = {}, []
    surf_cache = {}
    # Resolve every column's band FIRST, then walk band by band. The index
    # vectors are cached one file at a time (see _idx), so grouping turns
    # "read them per column" into "read them per band" — the whole reason a
    # warm start was slow. Order within a band is preserved.
    resolved, order = [], {}
    for c in columns:
        cid, lat, lon = c.get("id"), c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            skipped.append((cid, "no lat/lon")); continue
        band, restart = (bands.path_for_lat(lat)
                         if hasattr(bands, "path_for_lat") else
                         (lambda bs: (bs[0], bs[1].d.filepath() if bs[1] else None))(
                             bands.for_lat(lat)))
        if restart is None:
            skipped.append((cid, f"lat {lat:.3f} in no CONUS band")); continue
        order.setdefault(band, len(order))
        resolved.append((c, cid, lat, lon, band, restart))
    resolved.sort(key=lambda r: order[r[4]])

    # Subset columns CONCURRENTLY. write_subset copies ~216 variables, each a
    # separate seek into a 70 GB file, so a column is dominated by I/O LATENCY
    # rather than bandwidth: 17-27 s each, of which almost none is CPU. Letting
    # several columns have a request outstanding at once overlaps those seeks --
    # measured 53.4 s/column serial against 5.4 s/column with three workers.
    #
    # Separate processes, not threads: netCDF4-python gives no thread-safety
    # guarantee, and each worker wants its own Dataset handle anyway. The file
    # is opened read-only, so concurrent readers are safe.
    n_workers = max(1, min(int(workers or DEFAULT_WORKERS), len(resolved)))
    results = {}
    if n_workers > 1 and len(resolved) > 1:
        from concurrent.futures import ProcessPoolExecutor
        payload = [(cid, lat, lon, band, restart, str(out_dir))
                   for _, cid, lat, lon, band, restart in resolved]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for cid, info, reason in ex.map(_subset_one, payload):
                    results[cid] = (info, reason)
        except Exception as e:
            say(f"  (parallel subset unavailable: {str(e)[:60]} — falling back)")
            results = {}
    if not results:                                   # serial path / fallback
        for _, cid, lat, lon, band, restart in resolved:
            c2, info, reason = _subset_one(
                (cid, lat, lon, band, restart, str(out_dir)))
            results[c2] = (info, reason)

    for _, cid, lat, lon, band, restart in resolved:  # report in stable order
        info, reason = results.get(cid, (None, "not attempted"))
        if info is None:
            skipped.append((cid, reason)); continue
        manifest[cid] = info
        say(f"  ✓ {cid}: {info['n_column']} col / {info['n_pft']} pft, "
            f"band {band}, snapped {info['dist_km']} km to "
            f"({info['donor_lat']}, {info['donor_lon']})")
    for cid, why in skipped:
        say(f"  ! {cid}: {why} — skipped (will COLD start)")
    return manifest


# ELM's 10 soil levels — node depths (m), CLM/ELM standard. Only used to give
# the donor's soil the same depth-labelled shape SSURGO horizons arrive in, so
# the sampling-design figure can draw either without knowing which it got.
ELM_NODE_DEPTH_M = (0.0175, 0.0451, 0.0906, 0.1655, 0.2891,
                    0.4929, 0.8289, 1.3828, 2.2961, 3.8019)


def _texture_class(sand, clay):
    """USDA class from sand/clay percent. Same thresholds the geology MCP uses,
    so a CONUS-soil column and an SSURGO column get comparable labels."""
    silt = max(0.0, 100.0 - sand - clay)
    if clay >= 40:                      return "clay"
    if clay >= 27 and sand <= 20:       return "silty clay"
    if clay >= 27:                      return "clay loam"
    if clay >= 20 and sand <= 45:       return "loam"
    if silt >= 50 and clay < 27:        return "silt loam"
    if sand >= 85:                      return "sand"
    if sand >= 70 and clay < 15:        return "sandy loam"
    return "loam"


def donor_soil_profile(surfdata_path):
    """The donor gridcell's own soil, shaped like an MCP soil_profile.

    This is the soil the run ACTUALLY uses once a warm start keeps the donor's
    surfdata, so the sampling-design figure should show it rather than the
    SSURGO profile that was gathered and then not applied. Same keys the
    geology MCP emits, so the figure needs no special case.

    ORGANIC is already kg/m3 in surfdata; it is reported as-is rather than
    converted back to a percentage, since kg/m3 is the quantity ELM uses.
    """
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(surfdata_path) as d:
        def col(name):
            if name not in d.variables:
                return None
            return np.atleast_1d(np.asarray(d.variables[name][:]).squeeze())
        sand, clay = col("PCT_SAND"), col("PCT_CLAY")
        org, grvl = col("ORGANIC"), col("PCT_GRVL")
    if sand is None or clay is None:
        return None

    layers, prev = [], 0.0
    for i in range(min(len(sand), len(ELM_NODE_DEPTH_M))):
        bot = ELM_NODE_DEPTH_M[i] * 100.0
        layers.append({
            "component": "CONUS 1km",
            "depth_top_cm": round(prev, 1), "depth_bot_cm": round(bot, 1),
            "sand_pct": round(float(sand[i]), 1),
            "clay_pct": round(float(clay[i]), 1),
            "texture_class": _texture_class(float(sand[i]), float(clay[i])),
            "organic_kg_m3": round(float(org[i]), 1) if org is not None else None,
            "gravel_pct": round(float(grvl[i]), 1) if grvl is not None else None,
        })
        prev = bot
    return {"source": "CONUS 1 km surface dataset (donor gridcell)",
            "num_layers": len(layers), "layers": layers}


def _subset_one(args):
    """One column, start to finish. Returns (cid, entry|None, reason).

    Re-imports this module by path rather than relying on the parent's import
    name: ELMExpManager loads it through importlib under a custom name, which a
    pickled reference could not resolve in a fresh worker.
    """
    cid, lat, lon, band, restart, out_dir = args
    import importlib.util as _il
    import sys as _sys
    from pathlib import Path as _P
    mod = _sys.modules.get("_finidat_worker")
    if mod is None:
        _spec = _il.spec_from_file_location("_finidat_worker", __file__)
        mod = _il.module_from_spec(_spec)
        _sys.modules["_finidat_worker"] = mod
        _spec.loader.exec_module(mod)
    out_dir = _P(out_dir)
    try:
        info = mod.write_subset(restart, lat, lon,
                                out_dir / f"finidat_{cid}.nc", quiet=True)
        problems = mod.validate(info["finidat"])
        if problems:
            raise mod.SubsetError("finidat validation: " + "; ".join(problems))

        conus_sd = mod.conus_surfdata_for(restart)
        sd = mod.write_surfdata_subset(conus_sd, info["donor_ixy"],
                                       info["donor_jxy"],
                                       out_dir / f"surfdata_{cid}.nc", quiet=True)
        gate = mod.check_weights_agree(info["finidat"], sd["fsurdat"])
        if gate:
            raise mod.SubsetError("; ".join(gate))
    except mod.SubsetError as e:
        return cid, None, str(e)
    except Exception as e:                       # never take the ensemble down
        return cid, None, f"{type(e).__name__}: {str(e)[:120]}"

    info["conus_band"] = band
    info["surface_template"] = sd["fsurdat"]
    info["conus_surfdata"] = conus_sd
    info["requested_lat"], info["requested_lon"] = lat, lon
    return cid, info, None


def _load_make_warmstart():
    """Reuse ConusSource / band resolution rather than duplicating them."""
    import importlib.util
    if "make_warmstart" in sys.modules:
        return sys.modules["make_warmstart"]
    p = Path(__file__).resolve().parent / "make_warmstart.py"
    spec = importlib.util.spec_from_file_location("make_warmstart", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["make_warmstart"] = m
    spec.loader.exec_module(m)
    return m


if __name__ == "__main__":
    main()
