#!/usr/bin/env python3
"""
WTD-informed WARM START — the scalable alternative to per-watershed spin-up.

Takes each column's completed-run restart file (the "carrier" state) and
replaces its deep soil state with a spun-up CONUS one, producing a finidat
file the next run can start from.

WHAT IT DOES (one source, since 2026-08-12):
    A full slow-memory transplant from a spun-up CONUS 1-km ELM restart
             (--conus-restart). For each column, find the nearest gridcell and
             copy its natural-veg soil column's deep state — soil TEMPERATURE
             profile, soil moisture (liq/ice), and aquifer (ZWT/WA/ZWT_PERCH/
             FROST_TABLE/SOILP) — into the carrier's soil columns. This warm-
             starts the multi-year memory a 1-yr carrier and a WTD-only edit
             both miss (esp. deep soil temperature). Snow/canopy/PFT state are
             left as the carrier's (already a real-forcing January state).

    python3 mcp/elm-mcp/src/make_warmstart.py --run-dir <dir> [--cases-file cases_all.json]
                                    [--out-dir <dir>/warmstart]
                                    [--conus-restart /path/MANIFEST_or_file]

Writes Warmstart_<col>.nc per column + warmstart.json (col -> finidat path,
target vs applied WTD). Wire into a new run via columns_to_plan --finidat-map,
or let ELMExpManager do it (config['warm_start'], step 0b).

--conus-restart takes EITHER one band file or a MANIFEST
listing all bands; with a manifest each column is matched to the band whose
latitude range contains it. A watershed straddling a band edge (Naches spans
45-47N and 47-49N) then needs one invocation, not one per band.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

DEFAULT_CONUS_MANIFEST = (
    "/qfs/people/tran289/conus_restart_transfer/MANIFEST_conus_restarts.txt")

# slow-memory, soil-column-level state to transplant from the CONUS restart.
# All are dimensioned [column] or [column, lev*]; snow/canopy/PFT are left as
# the carrier's (already a real-forcing January state for the target year).
CONUS_SLOW_STATE = ["T_SOISNO", "H2OSOI_LIQ", "H2OSOI_ICE", "SOILP",
                    "ZWT", "WA", "ZWT_PERCH", "FROST_TABLE"]
SOIL_LANDUNITS = {1, 2}          # natural-veg + crop columns (weight-bearing)


class ConusSource:
    """Nearest-gridcell state lookup into a big CONUS 1-km ELM restart.

    Opens the restart ONCE and holds the location + sub-grid index vectors in
    memory (a few hundred MB); each per-column query is a nearest-gridcell
    argmin plus single-column slices, so the 71 GiB file is never read whole.
    """
    def __init__(self, path):
        import netCDF4, numpy as np
        self.np = np
        self.d = netCDF4.Dataset(path)
        self.glat = self.d.variables["grid1d_lat"][:]
        glon = self.d.variables["grid1d_lon"][:]
        self.glon = np.where(glon > 180, glon - 360, glon)
        self.cgi = self.d.variables["cols1d_gridcell_index"][:]   # 1-based
        self.clun = self.d.variables["cols1d_ityplun"][:]
        self.lat_range = (float(self.glat.min()), float(self.glat.max()))

    def natveg_column(self, lat, lon):
        """Return (conus_soil_col_index, dist_km) for the nearest gridcell's
        natural-veg (ityplun==1) column, or (None, dist) if it has none."""
        np = self.np
        g = int(np.argmin((self.glat - lat) ** 2 + (self.glon - lon) ** 2))
        dist_km = 111.0 * np.hypot(self.glat[g] - lat,
                                   (self.glon[g] - lon) * np.cos(np.radians(lat)))
        cols = np.where(self.cgi == g + 1)[0]          # cgi is 1-based
        nat = cols[self.clun[cols] == 1]
        return (int(nat[0]) if len(nat) else None), float(dist_km)

    def read_state(self, col):
        """The slow-state arrays at one CONUS column, as {var: ndarray}."""
        return {v: self.np.array(self.d.variables[v][col]) for v in CONUS_SLOW_STATE
                if v in self.d.variables}


def conus_transplant(src, dst, conus, lat, lon):
    """Copy the CONUS nearest-cell natural-veg deep state into the carrier's
    soil columns (nat-veg + crop). Returns a provenance dict."""
    import netCDF4
    ncol, dist = conus.natveg_column(lat, lon)
    if ncol is None:
        return {"ok": False, "reason": f"no natural-veg column at nearest gridcell "
                                       f"({dist:.1f} km away)"}
    state = conus.read_state(ncol)
    shutil.copy2(src, dst)
    with netCDF4.Dataset(dst, "r+") as d:
        lun = d.variables["cols1d_ityplun"][:]
        soil_cols = [i for i, l in enumerate(lun) if int(l) in SOIL_LANDUNITS]
        for v, arr in state.items():
            for i in soil_cols:
                d.variables[v][i] = arr                 # [i] or [i,:] both OK
        zwt = float(state.get("ZWT", [float("nan")])[()]) if "ZWT" in state else float("nan")
        d.setncattr("warmstart", f"CONUS 1-km transplant: slow-memory soil state "
                    f"({', '.join(state)}) from natural-veg column {ncol} "
                    f"({dist:.1f} km from target); snow/canopy/PFT kept from carrier")
    tsoi = state.get("T_SOISNO")
    return {"ok": True, "conus_col": ncol, "dist_km": round(dist, 2),
            "applied_zwt_m": round(float(state["ZWT"]), 3) if "ZWT" in state else None,
            "deep_soiltemp_C": round(float(tsoi[-1]) - 273.15, 2) if tsoi is not None else None,
            "n_soil_columns": len(soil_cols)}


# ─────────────────────────────────────────────────────────────────────
# CONUS BAND RESOLUTION — pick the right restart per column latitude
# ─────────────────────────────────────────────────────────────────────
_MANIFEST_ROW = re.compile(r"^(lat\d+)\s+(\d+)-(\d+)N\s+\S+\s+\S+\s+(/\S+)")


def read_conus_manifest(path):
    """MANIFEST_conus_restarts.txt -> [(band, lat_lo, lat_hi, restart_path)].

    The manifest is the authority on which band covers which latitudes; without
    it you must know that e.g. Naches (45.9-47.1N) needs BOTH lat11 and lat12.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        m = _MANIFEST_ROW.match(line.strip())
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    return rows


def resolve_conus_sources(spec):
    """--conus-restart spec -> [(band, lat_lo, lat_hi, path)].

    Accepts a manifest file, a single restart file, or a directory of them.
    For a bare restart we cannot know its band from the name alone, so lat
    range is left None and ConusSource reports its own (read from the file).
    """
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"--conus-restart not found: {spec}")
    if p.is_dir():
        return [(f.stem[:8], None, None, str(f)) for f in sorted(p.glob("*.nc"))]
    if p.suffix != ".nc":                      # a manifest
        rows = read_conus_manifest(p)
        if not rows:
            raise ValueError(f"no band rows parsed from manifest {spec}")
        return rows
    return [(p.stem[:8], None, None, str(p))]


class ConusBandSet:
    """Lazily-opened set of CONUS band restarts, indexed by latitude.

    Only the bands actually needed get opened — each ConusSource holds a few
    hundred MB of index vectors, and the underlying files are 3-79 GB.
    """
    def __init__(self, sources):
        self.sources = sources              # [(band, lo, hi, path)]
        self._open = {}                     # band -> ConusSource

    def for_lat(self, lat):
        """(band, ConusSource) covering `lat`, or (None, None).

        Declared ranges (from a manifest) are authoritative: if none contains
        `lat`, answer (None, None) WITHOUT opening anything. Probing each file
        instead would open every band — 636 GB across 12 files — to learn what
        the manifest already said.
        """
        declared = [s for s in self.sources if s[1] is not None]
        if declared:
            for band, lo, hi, path in declared:
                if lo <= lat < hi:
                    return band, self._get(band, path)
            return None, None
        # No declared ranges (bare file / directory): the only way to know is
        # to open each source and ask what latitudes it actually holds.
        for band, _, _, path in self.sources:
            src = self._get(band, path)
            if src.lat_range[0] <= lat <= src.lat_range[1]:
                return band, src
        return None, None

    def path_for_lat(self, lat):
        """(band, path) covering `lat` WITHOUT opening the restart.

        ConusSource's constructor reads four whole index vectors so that
        natveg_column() can answer per-column queries. A caller that only wants
        the filename — make_finidat_subset does its own, finer-grained lookup —
        would pay hundreds of MB of I/O per band for nothing.
        """
        for band, lo, hi, path in self.sources:
            if lo is not None and lo <= lat < hi:
                return band, path
        band, src = self.for_lat(lat)          # undeclared ranges: must open
        return band, (src.d.filepath() if src is not None else None)

    def _get(self, band, path):
        if band not in self._open:
            print(f"    opening CONUS band {band}: {Path(path).name}")
            self._open[band] = ConusSource(path)
        return self._open[band]


def build_warmstart(cases, latlon, out_dir,
                    conus_restart=None, quiet=False):
    """Build finidat files for `cases`; return the manifest dict.

    Importable core shared by this CLI and ELMExpManager step 0b, so the
    integrated pipeline and the command line produce identical warm starts.

        cases    list of carrier case dirs (a COMPLETED run's cases.json)
        latlon   {col_id: (lat, lon)} — from the plan's CONDITIONS_COUPLERS
        conus_restart  manifest / restart file / directory

    Columns with no carrier restart, no target, or no covering CONUS band are
    SKIPPED with a reason — never silently cold-started.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = (lambda *a: None) if quiet else print

    if not conus_restart:
        raise ValueError("a CONUS restart is required "
                         "(manifest, file, or directory)")
    bands = ConusBandSet(resolve_conus_sources(conus_restart))
    say(f"  CONUS warm start from {len(bands.sources)} band(s); "
        f"transplanting {', '.join(CONUS_SLOW_STATE)}")

    manifest, skipped = {}, []
    for c in cases:
        name = str(c).split(".")[-1]
        restarts = sorted(Path(c, "run").glob("*.elm.r.*.nc"))
        if not restarts:
            skipped.append((name, "no carrier restart in <case>/run"))
            continue
        dst = out_dir / f"Warmstart_{name}.nc"

        ll = latlon.get(name)
        if ll is None:
            skipped.append((name, "no lat/lon in plan"))
            continue
        band, src = bands.for_lat(ll[0])
        if src is None:
            skipped.append((name, f"lat {ll[0]:.3f} in no available CONUS band"))
            continue
        info = conus_transplant(restarts[-1], dst, src, ll[0], ll[1])
        if not info.get("ok"):
            skipped.append((name, info.get("reason", "transplant failed")))
            continue
        say(f"  \u2713 {name}: ZWT {info['applied_zwt_m']} m, deep Tsoil "
            f"{info['deep_soiltemp_C']} C  (band {band}, "
            f"{info['dist_km']} km, {info['n_soil_columns']} soil cols)")
        manifest[name] = {"finidat": str(dst.resolve()), "source": "conus",
                          "conus_band": band, **info}

    for name, why in skipped:
        say(f"  ! {name}: {why} — skipped (will COLD start)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Build WTD-informed finidat files")
    ap.add_argument("--run-dir", required=True,
                    help="a COMPLETED run dir (its cases' restart files are the carriers)")
    ap.add_argument("--cases-file", default="cases.json")
    ap.add_argument("--plan-file", default="run_plan.json")
    ap.add_argument("--conus-restart", default=DEFAULT_CONUS_MANIFEST,
                    help="CONUS restart: a MANIFEST (default, auto-picks the band "
                         "per column latitude), a single .nc, or a directory")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "warmstart"
    cases = json.loads((run_dir / args.cases_file).read_text())

    plan = json.loads((run_dir / args.plan_file).read_text())
    latlon = {cc["EXPERIMENT"]: (cc["lat"], cc["lon"])
              for cc in plan.get("CONDITIONS_COUPLERS", []) if "lat" in cc}

    manifest = build_warmstart(cases, latlon, out_dir,
                               conus_restart=args.conus_restart)

    if not manifest:
        sys.exit("no warm-start files produced — "
                 "NOT overwriting any existing warmstart.json")
    (out_dir / "warmstart.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(cases)} warm-start files -> {out_dir}/")
    print(f"use: columns_to_plan.py ... --finidat-map {out_dir}/warmstart.json")


if __name__ == "__main__":
    main()
