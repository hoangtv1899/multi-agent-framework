#!/usr/bin/env python3
"""A whole-domain input set, checked before any job is submitted.

The surface data, the domain file and the saved state (finidat) must describe
the same cells, the same subgrid and the same weights. When they do not, ELM
stops at initialisation, an hour into a queue wait, or runs on a quietly
different setup. Every check below is a recorded number with its limit, so a
failed set says which file disagrees with which, and by how much.

What ELM itself stops on (E3SM maint-2.0, components/elm/src), over the
domain's active cells (mask 1, the only cells ELM runs; decompInitMod):

  required data   the surface variables ELM reads and stops without
                  (surfrdMod, SoilStateType, organicFileMod).
  centres         the domain's xc/yc against the surface's LONGXY/LATIXY,
                  longitudes modulo 360: surfrd_get_data stops beyond 0.001 deg.
  percentages     PCT_NAT_PFT / 100 sums to 1 in every cell, and so do the
                  land-unit percentages, within 1e-14 (check_sums_equal_1);
                  lake, wetland, urban and glacier stay under 100 % + 1e-4
                  (surfrd_special); no urban percentage where URBAN_REGION_ID
                  is 0 (CheckUrban).
  subgrid size    the saved state's grid cells, land units, columns and PFTs
                  are those ELM lays out for this surface and domain
                  (restFile_dimcheck); see elm_layout.
  PFT weights     pfts1d_wtlnd of the natural-vegetation PFTs against
                  PCT_NAT_PFT / 100: subgridRestMod stops beyond 5e-3. ELM skips
                  this for transient-PFT, FATES and continued runs; the check
                  here is for a run started from finidat with fixed vegetation,
                  and applies it always.

What this check adds, where ELM would run but not on the setup meant:

  exact centres   the domain carries the surface's own centres, to rounding
                  (1e-12 deg): a domain cut from another source can sit up to
                  0.001 deg off and still run, on cells that are not quite the
                  surface's. domain_from_surface writes such a domain.
  record order    ELM reads the saved state by position and never reads back
                  its grid1d / land1d / cols1d / pfts1d indices, so they must
                  follow ELM's own order record for record, or the weights
                  and the state land on other units.
  land units      land1d_wtxy against the surface's land-unit percentages,
                  within 1e-6: the saved state's weights win at initialisation.
  restart grid    grid1d_lon/lat (the centres of the domain the state was
                  written on) against the surface's, within 1e-4 deg.
  coverage        no crop land unit and no glacier elevation classes, which
                  elm_layout does not lay out; soil fractions finite and not
                  negative.

Ported from the hand scripts of the Naches ELM-BGC experiment
(make_domain_from_surf.py and make_box_inputs.py, 2026-10-12/13) and checked
against the ELM source. The hand scripts' 1e-12 for the centres is the
coupler's land-against-atmosphere tolerance (EPS_AGRID), not ELM's surface
check, which allows 0.001 deg (the run whose saved state the Naches runs
started from had a domain 4.6e-5 deg off its surface). It is kept here as
this framework's own rule, and the reasons say which rule a set breaks.

domain_from_surface writes the CIME domain file whose centres are the
surface's own.
"""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# CIME's Earth radius (shr_const_rearth), for the domain area in km2.
CIME_EARTH_RADIUS_KM = 6371.22

# ELM land-unit types (landunit_varcon.F90).
ISTSOIL, ISTCROP, ISTICE, ISTICE_MEC, ISTDLAK, ISTWET = 1, 2, 3, 4, 5, 6
ISTURB_TBD, ISTURB_HD, ISTURB_MD = 7, 8, 9
MAXPATCH_URB = 5                  # one-PFT columns per urban land unit (elm_varpar)
URBAN_INVALID_REGION = 0          # URBAN_REGION_ID of a cell with no urban data

# ELM's own limits: past one of them the model stops at initialisation.
ELM_COORD_MAX_DIFF_DEG = 1e-3     # surfrd_get_data
ELM_SUM_MAX_DEV = 1e-14           # check_sums_equal_1
ELM_SPECIAL_MAX_PCT = 100.0 + 1e-4  # surfrd_special: lake + wetland + urban + glacier
ELM_PFT_WEIGHT_MAX_DIFF = 5e-3    # subgridRestMod check_weights

# The surface variables ELM reads and stops without.
REQUIRED_SURFACE = ("LONGXY", "LATIXY", "PFTDATA_MASK", "PCT_NATVEG", "PCT_CROP",
                    "PCT_NAT_PFT", "PCT_LAKE", "PCT_WETLAND", "PCT_GLACIER", "PCT_URBAN",
                    "URBAN_REGION_ID", "PCT_SAND", "PCT_CLAY", "ORGANIC")
_SOIL = ("PCT_SAND", "PCT_CLAY", "ORGANIC")

# (record name, land-unit type, surface variable, urban density index)
_LANDUNITS = (("natveg", ISTSOIL, "PCT_NATVEG", None),
              ("urban_tbd", ISTURB_TBD, "PCT_URBAN", 0),
              ("urban_hd", ISTURB_HD, "PCT_URBAN", 1),
              ("urban_md", ISTURB_MD, "PCT_URBAN", 2),
              ("lake", ISTDLAK, "PCT_LAKE", None),
              ("wetland", ISTWET, "PCT_WETLAND", None),
              ("glacier", ISTICE, "PCT_GLACIER", None))
_URBAN = (ISTURB_TBD, ISTURB_HD, ISTURB_MD)


class InputSetError(Exception):
    """The input set cannot be checked or written, and the message says why."""


@dataclass(frozen=True)
class PairingLimits:
    """This check's own limits, where ELM itself would run (see the module text).
    coord_max_diff_deg = ELM_COORD_MAX_DIFF_DEG leaves only ELM's rule."""
    coord_max_diff_deg: float = 1e-12
    landunit_weight_max_diff: float = 1e-6
    restart_grid_max_offset_deg: float = 1e-4


def _raw(var) -> np.ndarray:
    var.set_auto_maskandscale(False)
    return np.asarray(var[:])


def _lon_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| for longitudes, the same cell 360 deg apart counted as 0, as
    surfrd_get_data does. Small differences stay exact."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.abs(np.where(d > 180.0, d - 360.0, np.where(d < -180.0, d + 360.0, d)))


def _verdict(value, limit, past: str, not_a_number: Optional[str]) -> Optional[str]:
    """The reason when value is past limit, or a reason of its own when it is
    not a number: ELM's own tests (x > limit) let a NaN through, so a NaN
    does not stop ELM, it runs on it."""
    if value is None or not np.isfinite(value):
        return not_a_number
    return past if value > limit else None


def _num(x) -> Optional[float]:
    """A float for the record, None when it is not a number."""
    x = float(x)
    return x if np.isfinite(x) else None


def restart_tables(path) -> Dict[str, np.ndarray]:
    """The subgrid tables of a saved state that the checks need."""
    import netCDF4
    names = ("grid1d_lon", "grid1d_lat", "grid1d_ixy", "grid1d_jxy",
             "land1d_ityplun", "land1d_gridcell_index", "land1d_wtxy",
             "cols1d_ityplun", "cols1d_gridcell_index",
             "pfts1d_ityplun", "pfts1d_gridcell_index", "pfts1d_itypveg", "pfts1d_wtlnd")
    with netCDF4.Dataset(path) as r:
        missing = [n for n in names if n not in r.variables]
        if missing:
            raise InputSetError(f"{Path(path).name} lacks {missing}: not an ELM restart")
        t = {n: _raw(r[n]) for n in names}
        t["dims"] = {k: len(r.dimensions[k]) for k in ("gridcell", "landunit", "column", "pft")
                     if k in r.dimensions}
    return t


def elm_layout(active: np.ndarray, urban_valid: np.ndarray, pct_lake: np.ndarray,
               pct_wetland: np.ndarray, pct_glacier: np.ndarray,
               natpft: int) -> Dict[str, np.ndarray]:
    """The subgrid ELM lays out for these cells, in its saved state's order.

    ELM maint-2.0 with one topounit, no crop land unit and no glacier
    elevation classes (subgridMod, initGridCellsMod): every active cell, in
    grid order (longitude fastest), holds the natural-vegetation land unit
    (one column, all natpft PFTs); where URBAN_REGION_ID is not 0, the three
    urban land units (tall building district, high and medium density), each
    of five one-PFT columns; then one land unit, column and PFT each for lake,
    wetland and glacier where that percentage is above 0.

    Returned: the active cells' 0-based (j, i), and per land unit, column and
    PFT its type and 1-based grid cell; per PFT also its vegetation type.
    """
    jj, ii = np.nonzero(active)
    where = ((ISTSOIL, None), (ISTURB_TBD, urban_valid), (ISTURB_HD, urban_valid),
             (ISTURB_MD, urban_valid), (ISTDLAK, pct_lake > 0),
             (ISTWET, pct_wetland > 0), (ISTICE, pct_glacier > 0))
    present = np.stack([np.ones(jj.size, bool) if m is None else np.asarray(m)[jj, ii]
                        for _, m in where], axis=1)
    types = np.broadcast_to(np.array([t for t, _ in where]), present.shape)
    cell = np.broadcast_to(np.arange(1, jj.size + 1)[:, None], present.shape)
    land_type, land_cell = types[present], cell[present]      # cell by cell, in order
    urban = np.isin(land_type, _URBAN)
    ncol = np.where(urban, MAXPATCH_URB, 1)
    npft = np.where(land_type == ISTSOIL, natpft, ncol)
    pft_type = np.repeat(land_type, npft)
    pft_veg = np.zeros(pft_type.size, dtype=int)
    nat = pft_type == ISTSOIL
    pft_veg[nat] = np.tile(np.arange(natpft), int(nat.sum()) // natpft)
    return {"j": jj, "i": ii, "land_type": land_type, "land_cell": land_cell,
            "col_type": np.repeat(land_type, ncol), "col_cell": np.repeat(land_cell, ncol),
            "pft_type": pft_type, "pft_cell": np.repeat(land_cell, npft), "pft_veg": pft_veg}


def _first_difference(t: Dict[str, np.ndarray], want: Dict[str, np.ndarray]) -> Optional[str]:
    """Where the saved state's records first leave ELM's order, or None."""
    pairs = (("grid1d_ixy", want["i"] + 1, "grid cell", "ixy"),
             ("grid1d_jxy", want["j"] + 1, "grid cell", "jxy"),
             ("land1d_gridcell_index", want["land_cell"], "land unit", "grid cell"),
             ("land1d_ityplun", want["land_type"], "land unit", "type"),
             ("cols1d_gridcell_index", want["col_cell"], "column", "grid cell"),
             ("cols1d_ityplun", want["col_type"], "column", "type"),
             ("pfts1d_gridcell_index", want["pft_cell"], "PFT", "grid cell"),
             ("pfts1d_ityplun", want["pft_type"], "PFT", "land-unit type"))
    for name, w, unit, what in pairs:
        bad = np.flatnonzero(t[name].astype(int) != w)
        if bad.size:
            k = int(bad[0])
            return (f"{unit} {k + 1} has {what} {int(t[name][k])} in the saved state, "
                    f"{int(w[k])} in ELM's layout")
    nat = want["pft_type"] == ISTSOIL
    bad = np.flatnonzero(t["pfts1d_itypveg"].astype(int)[nat] != want["pft_veg"][nat])
    if bad.size:
        k = int(np.flatnonzero(nat)[bad[0]])
        return (f"PFT {k + 1} has vegetation type {int(t['pfts1d_itypveg'][k])} in the "
                f"saved state, {int(want['pft_veg'][k])} in ELM's layout")
    return None


def check_pairing(surface, domain, restart,
                  limits: PairingLimits = PairingLimits()) -> Dict:
    """Do these three files make one input set? A JSON-ready record.

    record["pass"] is the verdict and record["reasons"] names every failed
    check, each saying whether ELM would stop or run another setup; the
    numbers behind each check are under record["checks"].
    """
    import netCDF4
    reasons: List[str] = []
    checks: Dict = {}

    def record():
        return {"surface": str(surface), "domain": str(domain), "restart": str(restart),
                "elm_limits": {"coord_max_diff_deg": ELM_COORD_MAX_DIFF_DEG,
                               "sum_max_dev": ELM_SUM_MAX_DEV,
                               "special_landunits_max_pct": ELM_SPECIAL_MAX_PCT,
                               "pft_weight_max_diff": ELM_PFT_WEIGHT_MAX_DIFF},
                "limits": asdict(limits), "checks": checks,
                "pass": not reasons, "reasons": reasons,
                "checked": datetime.datetime.now().isoformat(timespec="seconds")}

    with netCDF4.Dataset(surface) as s:
        missing = [n for n in REQUIRED_SURFACE if n not in s.variables]
        if missing:
            raise InputSetError(f"{Path(surface).name} lacks {missing}, which ELM "
                                f"requires: not ELM surface data, or not all of it")
        if s["LONGXY"].ndim != 2:
            raise InputSetError(f"{Path(surface).name} is an unstructured (1-D) surface: "
                                f"this check reads latitude-longitude grids only")
        lon, lat = _raw(s["LONGXY"]).astype(float), _raw(s["LATIXY"]).astype(float)
        pct = {n: _raw(s[n]).astype(float) for n in ("PCT_NAT_PFT", "PCT_NATVEG", "PCT_CROP",
                                                    "PCT_LAKE", "PCT_WETLAND", "PCT_GLACIER",
                                                    "PCT_URBAN")}
        region = _raw(s["URBAN_REGION_ID"]).astype(int)
        soil = {n: _raw(s[n]).astype(float) for n in _SOIL}
        has_cft = "cft" in s.dimensions
    with netCDF4.Dataset(domain) as d:
        for n in ("xc", "yc", "area", "mask"):
            if n not in d.variables:
                raise InputSetError(f"{Path(domain).name} lacks {n}: not a CIME domain file")
        xc, yc = _raw(d["xc"]).astype(float), _raw(d["yc"]).astype(float)
        area, mask = _raw(d["area"]).astype(float), _raw(d["mask"])
    t = restart_tables(restart)
    nj, ni = lon.shape
    natpft = pct["PCT_NAT_PFT"].shape[0]

    if xc.shape != lon.shape:
        reasons.append(f"ELM stops: the domain is {xc.shape} cells, the surface {lon.shape}")
        return record()
    active = mask == 1
    checks["active_cells"] = int(active.sum())
    checks["box_area_km2"] = float(area.sum() * CIME_EARTH_RADIUS_KM ** 2)
    if not active.any():
        reasons.append("ELM stops: the domain has no active cell (mask 1)")
        return record()

    # ── what ELM stops on ────────────────────────────────────────────────────
    # np.max keeps a NaN; Python's max(x, nan) would return x
    diff = float(np.max([_lon_diff(xc, lon)[active].max(), np.abs(yc - lat)[active].max()]))
    checks["domain_vs_surface_coord_max_diff_deg"] = _num(diff)
    why = _verdict(diff, ELM_COORD_MAX_DIFF_DEG,
                   f"ELM stops: the domain's centres sit {diff:.3g} deg from the surface's "
                   f"(surfrd_get_data allows {ELM_COORD_MAX_DIFF_DEG:g})",
                   "the domain's or the surface's centres include values that are not numbers")
    if why is None:
        why = _verdict(diff, limits.coord_max_diff_deg,
                       f"the domain's centres sit {diff:.3g} deg from the surface's: this "
                       f"check requires the surface's own centres (to "
                       f"{limits.coord_max_diff_deg:g} deg; ELM itself allows "
                       f"{ELM_COORD_MAX_DIFF_DEG:g}), so write the domain with "
                       f"domain_from_surface", None)
    if why:
        reasons.append(why)

    def worst(value, what):
        """The worst active cell of a per-cell number (a NaN counts as worst):
        recorded under checks[what]; returns the value and the cell."""
        j, i = np.unravel_index(int(np.argmax(np.where(active, value, -np.inf))), value.shape)
        checks[what] = _num(value[j, i])
        return float(value[j, i]), (int(j), int(i))

    # ELM's arithmetic: each percentage / 100, then the sum, in its order
    nat_sum = np.zeros((nj, ni))
    for k in range(natpft):
        nat_sum = nat_sum + pct["PCT_NAT_PFT"][k] / 100.0
    v, (j, i) = worst(np.abs(nat_sum - 1.0), "pct_nat_pft_sum_max_dev")
    why = _verdict(v, ELM_SUM_MAX_DEV,
                   f"ELM stops: PCT_NAT_PFT sums to {100.0 * nat_sum[j, i]:.12g} % at cell "
                   f"(j={j}, i={i}) (check_sums_equal_1 allows {ELM_SUM_MAX_DEV:g} off 1 "
                   f"after dividing by 100)",
                   f"PCT_NAT_PFT holds values that are not numbers at cell (j={j}, i={i})")
    if why:
        reasons.append(why)
    lun_sum = np.zeros((nj, ni))
    for part in (pct["PCT_NATVEG"], pct["PCT_CROP"], pct["PCT_GLACIER"], pct["PCT_LAKE"],
                 pct["PCT_WETLAND"], *pct["PCT_URBAN"]):
        lun_sum = lun_sum + part / 100.0
    v, (j, i) = worst(np.abs(lun_sum - 1.0), "landunit_pct_sum_max_dev")
    why = _verdict(v, ELM_SUM_MAX_DEV,
                   f"ELM stops: the land-unit percentages sum to {100.0 * lun_sum[j, i]:.12g} "
                   f"% at cell (j={j}, i={i}) (check_sums_equal_1 allows {ELM_SUM_MAX_DEV:g} "
                   f"off 1 after dividing by 100)",
                   f"the land-unit percentages hold values that are not numbers at cell "
                   f"(j={j}, i={i})")
    if why:
        reasons.append(why)
    special = pct["PCT_LAKE"] + pct["PCT_WETLAND"] + pct["PCT_URBAN"].sum(axis=0) + \
        pct["PCT_GLACIER"]
    v, (j, i) = worst(special, "special_landunit_max_pct")
    why = _verdict(v, ELM_SPECIAL_MAX_PCT,
                   f"ELM stops: lake, wetland, urban and glacier cover {v:.8g} % at cell "
                   f"(j={j}, i={i}) (surfrd_special allows 100 % + 1e-4)", None)
    if why:                         # a NaN here is named by the land-unit sum above
        reasons.append(why)
    urban_valid = region != URBAN_INVALID_REGION
    bad_urban = active & ~urban_valid & (pct["PCT_URBAN"] > 0).any(axis=0)
    checks["urban_pct_where_no_urban_region_cells"] = int(bad_urban.sum())
    if bad_urban.any():
        reasons.append(f"ELM stops: {int(bad_urban.sum())} active cells have an urban "
                       f"percentage but URBAN_REGION_ID {URBAN_INVALID_REGION} (CheckUrban)")

    # ── what the layout does not cover ───────────────────────────────────────
    other = {"PCT_CROP": float(np.nanmax(np.where(active, pct["PCT_CROP"], 0.0)))}
    checks["not_covered_max_pct"] = other
    if other["PCT_CROP"] > 0 or has_cft:
        reasons.append("the surface carries crop (PCT_CROP above 0 or crop types on a cft "
                       "dimension): elm_layout lays out no crop land unit")
    known = {typ for _, typ, _, _ in _LANDUNITS}
    extra_types = sorted(int(v) for v in set(np.unique(t["land1d_ityplun"])) - known)
    checks["restart_landunit_types_not_covered"] = extra_types
    if extra_types:
        reasons.append(f"the saved state has land-unit types {extra_types} (2 crop, 4 "
                       f"glacier elevation classes) that elm_layout does not lay out")
    checks["soil_negative_values"] = {n: int(((a < 0) & active).sum()) for n, a in soil.items()}
    checks["soil_nonfinite_values"] = {n: int((~np.isfinite(a) & active).sum())
                                       for n, a in soil.items()}
    if sum(checks["soil_negative_values"].values()):
        reasons.append(f"negative soil fractions: {checks['soil_negative_values']}")
    if sum(checks["soil_nonfinite_values"].values()):
        reasons.append(f"soil fractions that are not numbers: "
                       f"{checks['soil_nonfinite_values']}")

    # ── the saved state against ELM's layout ─────────────────────────────────
    want = elm_layout(active, urban_valid, pct["PCT_LAKE"], pct["PCT_WETLAND"],
                      pct["PCT_GLACIER"], natpft)
    expected = {"gridcell": int(want["j"].size), "landunit": int(want["land_type"].size),
                "column": int(want["col_type"].size), "pft": int(want["pft_type"].size)}
    checks["expected_dims"], checks["restart_dims"] = expected, t["dims"]
    checks["lake_cells"] = int((active & (pct["PCT_LAKE"] > 0)).sum())
    checks["urban_valid_cells"] = int((active & urban_valid).sum())
    checks["layout_matches"], checks["layout_first_difference"] = False, None
    checks["pft_weight_max_abs_diff"] = None
    checks["landunit_weight_max_abs_diff"] = None
    checks["restart_grid_max_offset_deg"] = None
    if expected != t["dims"]:
        reasons.append(f"ELM stops: the saved state's subgrid {t['dims']} is not the "
                       f"{expected} ELM lays out for this surface and domain "
                       f"(restFile_dimcheck)")
        return record()
    first = _first_difference(t, want)
    checks["layout_first_difference"] = first
    if first:
        reasons.append(f"the saved state is not in ELM's order ({first}); ELM reads it by "
                       f"position, so its weights and state would land on other units")
        return record()
    checks["layout_matches"] = True
    gj, gi = want["j"], want["i"]

    # PFT weights on the natural-vegetation land unit, by position (ELM's check)
    nat = want["pft_type"] == ISTSOIL
    c = want["pft_cell"][nat] - 1
    w_surf = pct["PCT_NAT_PFT"][want["pft_veg"][nat], gj[c], gi[c]] / 100.0
    pdiff = float(np.abs(w_surf - t["pfts1d_wtlnd"][nat]).max())
    checks["pft_weight_max_abs_diff"] = _num(pdiff)
    why = _verdict(pdiff, ELM_PFT_WEIGHT_MAX_DIFF,
                   f"ELM stops: PFT weights differ from the saved state by {pdiff:.3g} "
                   f"(subgridRestMod allows {ELM_PFT_WEIGHT_MAX_DIFF:g})",
                   "the PFT weights of the saved state or the surface include values that "
                   "are not numbers")
    if why:
        reasons.append(why)

    # land-unit weights, by position
    lc = want["land_cell"] - 1
    lu: Dict[str, float] = {}
    for name, typ, var, k in _LANDUNITS:
        m = want["land_type"] == typ
        if not m.any():
            continue
        y = pct[var][k] if k is not None else pct[var]
        lu[name] = float(np.abs(y[gj[lc[m]], gi[lc[m]]] / 100.0 - t["land1d_wtxy"][m]).max())
    checks["landunit_weight_max_abs_diff"] = {n: _num(v) for n, v in lu.items()}
    nan = [n for n, v in lu.items() if not np.isfinite(v)]
    bad = [n for n, v in lu.items() if np.isfinite(v) and v > limits.landunit_weight_max_diff]
    if bad:
        reasons.append(f"land-unit weights differ from the saved state ({', '.join(bad)}: up "
                       f"to {max(lu[n] for n in bad):.3g}; this check allows "
                       f"{limits.landunit_weight_max_diff:g}): ELM would run with the saved "
                       f"state's weights, not the surface's")
    if nan:
        reasons.append(f"land-unit weights include values that are not numbers "
                       f"({', '.join(nan)})")

    # the saved state's own grid (ELM does not compare it)
    off = float(np.max([_lon_diff(t["grid1d_lon"], lon[gj, gi]).max(),
                        np.abs(t["grid1d_lat"] - lat[gj, gi]).max()]))
    checks["restart_grid_max_offset_deg"] = _num(off)
    why = _verdict(off, limits.restart_grid_max_offset_deg,
                   f"the saved state's cells sit {off:.3g} deg from the surface's (this check "
                   f"allows {limits.restart_grid_max_offset_deg:g}; ELM does not compare "
                   f"them): a state from another grid",
                   "the saved state's cell centres include values that are not numbers")
    if why:
        reasons.append(why)
    return record()


def _float32_noise(a: np.ndarray) -> float:
    """How far two neighbouring differences of these coordinates may disagree
    when they are float32 values (stored as such, or copied into doubles): two
    float32 steps at the largest magnitude. 0 for true doubles."""
    if not np.array_equal(a, a.astype(np.float32).astype(np.float64)):
        return 0.0
    return 2.0 * float(np.spacing(np.float32(np.abs(a).max())))


def _spacing(a: np.ndarray, axis: int, what: str, noise: float) -> float:
    """The grid step along one axis, measured over the whole row or column.

    Surface coordinates are often float32 values: at 238 E neighbouring
    centres then differ by 1/120 deg plus or minus 1.5e-5. Taken over 143
    steps the span gives the step to about 1e-7, which is then snapped to
    1/n deg when it is that to within 1e-6, as the cells were defined.
    Neighbouring steps may differ from it by 1 %, or by the float32 noise of
    the coordinates when that is larger. Signed: negative for an axis that
    runs north to south or east to west.
    """
    n_pts = a.shape[axis]
    if n_pts < 2:
        raise InputSetError(f"one {what} only: no spacing to build cell edges from")
    span = np.take(a, -1, axis=axis) - np.take(a, 0, axis=axis)
    step = float(np.mean(span)) / (n_pts - 1)
    if step == 0.0:
        raise InputSetError(f"{what} coordinates do not change")
    n = round(1.0 / step)
    if n and abs(1.0 / n - step) < 1e-6:
        step = 1.0 / n
    d = np.diff(a, axis=axis)
    tol = max(0.01 * abs(step), noise)
    if np.abs(d - step).max() > tol:
        raise InputSetError(f"{what} spacing is not uniform (from {d.min():.8g} to "
                            f"{d.max():.8g} deg against {step:.8g}, beyond {tol:.2g}): "
                            f"not a regular grid")
    return step


def domain_from_surface(surface, out, overwrite: bool = False,
                        mask_from: str = "PFTDATA_MASK") -> Dict:
    """Write the CIME domain file of a regular surface grid. A record.

    Centres are the surface's LONGXY/LATIXY to the last bit. Cell edges lie
    half a spacing either side (the spacing snapped to 1/n degree when it is
    that to within 1e-6; a row may cross 0 or 180 E), corners run
    counterclockwise from the south-west, area is the spherical area in
    radians squared, and frac is the surface's LANDFRAC_PFT, or 1 without it.

    The mask is the set of cells ELM will run: ELM runs exactly the cells
    with mask 1 and reads no other activity flag (PFTDATA_MASK only reaches
    history). mask_from='PFTDATA_MASK' copies the surface's PFTDATA_MASK
    (cells without PFT data stay out), or writes 1 everywhere if the surface
    has none; mask_from='ones' makes every cell active. The record says which.
    """
    import netCDF4
    out = Path(out)
    if mask_from not in ("PFTDATA_MASK", "ones"):
        raise InputSetError(f"mask_from is 'PFTDATA_MASK' or 'ones', not {mask_from!r}")
    if out.exists() and out.resolve() == Path(surface).resolve():
        raise InputSetError(f"{out} is the surface itself")
    if out.exists() and not overwrite:
        raise InputSetError(f"{out} exists; pass overwrite=True to replace it")
    with netCDF4.Dataset(surface) as s:
        for n in ("LONGXY", "LATIXY"):
            if n not in s.variables:
                raise InputSetError(f"{Path(surface).name} lacks {n}: not ELM surface data")
        if s["LONGXY"].ndim != 2:
            raise InputSetError(f"{Path(surface).name} is an unstructured (1-D) surface: "
                                f"a CIME domain here is for latitude-longitude grids only")
        lon, lat = _raw(s["LONGXY"]).astype(float), _raw(s["LATIXY"]).astype(float)
        mask = _raw(s["PFTDATA_MASK"]).astype("i4") \
            if mask_from == "PFTDATA_MASK" and "PFTDATA_MASK" in s.variables else None
        frac = _raw(s["LANDFRAC_PFT"]).astype(float) if "LANDFRAC_PFT" in s.variables else None
    nj, ni = lon.shape
    lon_u = np.unwrap(lon, period=360.0, axis=1)     # a row across 0 or 180 E, made monotonic
    nx, ny = _float32_noise(lon), _float32_noise(lat)
    dx, dy = _spacing(lon_u, 1, "column", nx), _spacing(lat, 0, "row", ny)
    for a, ref, what, tol in ((lon_u, lon_u[:1, :], "longitudes change down a column", nx),
                              (lat, lat[:, :1], "latitudes change along a row", ny)):
        gap = float(np.abs(a - ref).max())
        lim = max(0.01 * min(abs(dx), abs(dy)), tol)
        if gap > lim:
            raise InputSetError(f"{what} by up to {gap:.3g} deg: not a regular "
                                f"latitude-longitude grid")
    hx, hy = abs(dx) / 2.0, abs(dy) / 2.0
    ys, yn = np.deg2rad(lat - hy), np.deg2rad(lat + hy)
    area = np.deg2rad(abs(dx)) * (np.sin(yn) - np.sin(ys))
    rec = {"surface": str(surface), "domain": str(out), "nj": nj, "ni": ni,
           "dx_deg": abs(dx), "dy_deg": abs(dy),
           "mask_from": ("PFTDATA_MASK" if mask is not None else
                         "all ones (asked)" if mask_from == "ones" else
                         "all ones (no PFTDATA_MASK)"),
           "frac_from": "LANDFRAC_PFT" if frac is not None else "all ones (no LANDFRAC_PFT)",
           "box_area_km2": float(area.sum() * CIME_EARTH_RADIUS_KM ** 2),
           "written": datetime.datetime.now().isoformat(timespec="seconds")}
    mask = np.ones((nj, ni), "i4") if mask is None else mask
    frac = np.ones((nj, ni)) if frac is None else frac
    rec["active_cells"] = int((mask == 1).sum())
    spec = {"xc": ("longitude of grid cell center", "degrees_east", lon),
            "yc": ("latitude of grid cell center", "degrees_north", lat),
            "xv": ("longitude of grid cell verticies", "degrees_east",
                   np.stack([lon - hx, lon + hx, lon + hx, lon - hx], -1)),
            "yv": ("latitude of grid cell verticies", "degrees_north",
                   np.stack([lat - hy, lat - hy, lat + hy, lat + hy], -1)),
            "area": ("area of grid cell in radians squared", "radian2", area),
            "frac": ("fraction of grid cell that is active", "unitless", frac),
            "mask": ("domain mask", "unitless", mask)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with netCDF4.Dataset(tmp, "w", format="NETCDF3_64BIT_OFFSET") as d:
        d.createDimension("nj", nj)
        d.createDimension("ni", ni)
        d.createDimension("nv", 4)
        for name, (long_name, units, data) in spec.items():
            dims = ("nj", "ni", "nv") if data.ndim == 3 else ("nj", "ni")
            v = d.createVariable(name, "i4" if name == "mask" else "f8", dims)
            v.long_name, v.units = long_name, units
            v[:] = data
        d.title = f"CIME domain file, {nj} x {ni}, centres from the surface data"
        d.history = (f"{rec['written']} domain_inputs.domain_from_surface; centres copied "
                     f"from {Path(surface).name}")
    tmp.replace(out)
    return rec
