#!/usr/bin/env python3
"""Phase 1a — can THIS process reach the CONUS data the warm start needs?

docs/ELM_MCP_PLAN.md §9 phase 1a. Under Rule B the warm start moves into the
MCP, and the server currently declares no bulk-data dependency at all --
`_requirements()` checks E3SM, CIME, $PSCRATCH and the scheduler, and its
docstring says outright "no CONUS surfdata, no ELM input-file templates".

The risk is not that the files are missing. It is that an MCP client forwards
only HOME, LOGNAME, PATH, SHELL and USER, so anything resolved through an
environment variable is not there -- the failure mode that made every lambda
tool in the reaction MCP report "No module named 'preprocessing'" under a
standard client while working through ours.

So this checks three things, in the order they can fail:

  1. the paths exist and are readable
  2. netCDF4 can actually OPEN a restart and a surfdata file -- existence is
     not access on a parallel filesystem, and the library has to be importable
     in whatever process the server runs in
  3. the coordinate lookup a warm start actually performs returns a cell

Run it twice: normally, and with `--stripped`, which re-executes it with only
the five variables a client forwards. If (2) or (3) passes normally and fails
stripped, the dependency is real and the launcher must set the paths itself.
"""
import argparse
import os
import re
import sys
from pathlib import Path

FRAMEWORK = Path(os.getenv(
    "IDEAS_FRAMEWORK_DIR", str(Path(__file__).resolve().parents[3])))

CONUS_MANIFEST = Path(
    "/qfs/people/tran289/conus_restart_transfer/MANIFEST_conus_restarts.txt")
CONUS_SURFDATA = Path("/qfs/people/tran289/IDEAS/1d_elm/conus_surfdata")
INPUT_FILES    = Path("/qfs/people/tran289/IDEAS/1d_elm/input_files")

KEEP = ("HOME", "LOGNAME", "PATH", "SHELL", "USER")
_ROW = re.compile(r"^(lat\d+)\s+(\d+)-(\d+)N\s+\S+\s+\S+\s+(/\S+)")

ok = True


def check(label, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stripped", action="store_true",
                    help="re-exec with only the 5 vars an MCP client forwards")
    args = ap.parse_args()

    if args.stripped and os.environ.get("_PROBE_STRIPPED") != "1":
        env = {k: os.environ[k] for k in KEEP if k in os.environ}
        env["_PROBE_STRIPPED"] = "1"
        print(f"re-exec with only {', '.join(sorted(env))}\n")
        os.execve(sys.executable,
                  [sys.executable, str(Path(__file__).resolve())], env)

    mode = "STRIPPED" if os.environ.get("_PROBE_STRIPPED") == "1" else "NORMAL"
    print(f"── CONUS access probe ({mode}) ──")
    print(f"   python  {sys.executable}")
    print(f"   env     {len([k for k in os.environ if k.startswith(('IDEAS','E3SM','PSCRATCH','LD_','MODULE'))])} "
          f"IDEAS/E3SM/PSCRATCH/LD_/MODULE* vars present\n")

    print("1. paths")
    check("CONUS restart manifest", CONUS_MANIFEST.is_file(), str(CONUS_MANIFEST))
    check("CONUS surfdata dir", CONUS_SURFDATA.is_dir(), str(CONUS_SURFDATA))
    check("ELM input_files dir", INPUT_FILES.is_dir(), str(INPUT_FILES))

    print("\n2. open the data")
    try:
        import netCDF4                                          # noqa: F401
        check("import netCDF4", True, netCDF4.__version__)
    except Exception as e:                                      # noqa: BLE001
        check("import netCDF4", False, f"{type(e).__name__}: {e}")
        return 0 if ok else 1

    rows = []
    if CONUS_MANIFEST.is_file():
        for line in CONUS_MANIFEST.read_text().splitlines():
            m = _ROW.match(line.strip())
            if m:
                rows.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    check("manifest parses", bool(rows), f"{len(rows)} band(s)")

    restart = next((Path(r[3]) for r in rows if Path(r[3]).is_file()), None)
    if check("a restart file exists", restart is not None,
             str(restart) if restart else "none of the manifest rows resolve"):
        try:
            d = netCDF4.Dataset(str(restart))
            n = len(d.dimensions.get("gridcell", []) or [])
            check("netCDF4 OPENS the restart", True,
                  f"{n} gridcells, {len(d.variables)} variables")
            d.close()
        except Exception as e:                                  # noqa: BLE001
            check("netCDF4 OPENS the restart", False, f"{type(e).__name__}: {e}")

    surf = sorted(CONUS_SURFDATA.glob("*.nc")) if CONUS_SURFDATA.is_dir() else []
    if check("a CONUS surfdata file exists", bool(surf),
             surf[0].name if surf else "no .nc in conus_surfdata"):
        try:
            d = netCDF4.Dataset(str(surf[0]))
            check("netCDF4 OPENS the surfdata", True,
                  f"{len(d.variables)} variables")
            d.close()
        except Exception as e:                                  # noqa: BLE001
            check("netCDF4 OPENS the surfdata", False, f"{type(e).__name__}: {e}")

    print("\n3. the lookup a warm start performs")
    # Upper Gunnison, the domain of the 19-column fixture run.
    lat, lon = 38.9, -107.0
    band = next((r for r in rows if r[1] <= lat < r[2]), None)
    check("latitude band resolves", band is not None,
          f"{lat}N → {band[0]}" if band else f"no band covers {lat}N")
    if band and Path(band[3]).is_file():
        try:
            d = netCDF4.Dataset(band[3])
            la = d.variables.get("grid1d_lat") or d.variables.get("cols1d_lat")
            lo = d.variables.get("grid1d_lon") or d.variables.get("cols1d_lon")
            if la is None or lo is None:
                check("donor gridcell lookup", False,
                      f"no lat/lon vars; have {sorted(d.variables)[:6]}")
            else:
                import numpy as np
                a, o = np.asarray(la[:]), np.asarray(lo[:])
                o = np.where(o > 180, o - 360, o)
                i = int(np.argmin((a - lat) ** 2 + (o - lon) ** 2))
                check("donor gridcell lookup", True,
                      f"cell {i} at {a[i]:.3f}N {o[i]:.3f}E")
            d.close()
        except Exception as e:                                  # noqa: BLE001
            check("donor gridcell lookup", False, f"{type(e).__name__}: {e}")

    print(f"\n── {'ALL PASS' if ok else 'FAILURES ABOVE'} ({mode}) ──")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
