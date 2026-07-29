#!/usr/bin/env python3
"""
Build the per-column ELM input files
tools/build_column_inputs.py

domain.nc + surface.nc for every column in a run, from the SNAPPED coordinates
in columns.json. A thin CLI over ELMDomainGenerator and ELMSurfaceGenerator —
it decides nothing those two already decide, it only chooses which column to
call them for and where the answer goes.

Why it exists separately from the case build: the two are ordered, and the
order is not obvious. A warm start snaps each column to its donor gridcell and
hands it that cell's surfdata as the template, so the inputs MUST be built
after the warm start and from the snapped coordinates. Built before, ELM aborts
at initialisation on a surfdata/fatmgrid lon/lat mismatch — the coordinates in
the surface file and the coordinates in the domain file disagree, and the model
will not start. Having this as its own step makes the ordering visible and
lets it be re-run without rebuilding a case.

Two rules the generators enforce and this script must not undo:

  veg_source   With a CONUS-subset template the vegetation is already this
               gridcell's own, at 1 km. Re-extracting it from the 0.5 degree
               global file overwrites it with a coarser mixture and breaks the
               finidat/fsurdat weight agreement ELM checks at startup.
  soil_source  A CONUS-subset template means the run is warm-started, and the
               restart's moisture is equilibrated against THAT gridcell's soil.
               Replacing the soil makes the inherited state inconsistent with
               its own hydraulics and spends year one relaxing.

Both reduce to: when a column has a donor template, keep the donor's
vegetation and the donor's soil. The `--soil-config native` path still runs
through the generator even with no soil data of its own, because that is what
rewrites lat/lon to match the per-column domain.

Run:
    python3 tools/build_column_inputs.py --run-dir workflow_outputs/elm_run_X
    python3 tools/build_column_inputs.py --run-dir ... --only col_01,col_07
    python3 tools/build_column_inputs.py --lat 46.8 --lon -121.0
"""
import argparse
import json
import sys
from pathlib import Path
from typing  import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _warmstart_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Per-column donor info written by the warm start, keyed by column id.

    Absent (a cold run), every column builds from the stock template and the
    veg/soil rules below fall back to their non-donor branch.
    """
    f = run_dir / "warmstart" / "warmstart.json"
    if not f.exists():
        return {}
    try:
        raw = json.loads(f.read_text())
    except Exception:
        return {}
    # The warm start writes {col_id: {...}} directly — no wrapper key. Older
    # and list-shaped variants are accepted so this reads any run directory.
    if isinstance(raw, dict):
        inner = raw.get("columns")
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            raw = inner
        else:
            return {k: v for k, v in raw.items() if isinstance(v, dict)}
    return {c.get("id") or c.get("column"): c
            for c in (raw or []) if isinstance(c, dict)}


def build_one(lat:              float,
              lon:              float,
              soil:             Optional[Dict[str, Any]] = None,
              surface_template: Optional[str] = None,
              soil_config:      str = "native",
              substrate:        str = "extrapolate",
              force:            bool = False) -> Dict[str, str]:
    """One column → {'domain': path, 'surface': path}.

    Returns whichever half succeeded; a missing key means that generator
    failed and the caller must not build a case for this column.
    """
    from core.elm_domain_generator  import ELMDomainGenerator
    from core.elm_surface_generator import ELMSurfaceGenerator

    out: Dict[str, str] = {}

    out["domain"] = ELMDomainGenerator().generate(lat, lon, force=force)

    # See the module docstring: a donor template carries the gridcell's own
    # 1 km vegetation and the soil the restart is equilibrated against.
    veg_source  = "template" if surface_template else "conus"
    soil_source = "conus"    if surface_template else "ssurgo"
    gen = (ELMSurfaceGenerator(template_path=surface_template)
           if surface_template else ELMSurfaceGenerator())

    if soil_config == "native":
        # ALWAYS through the generator, even with no soil data of its own —
        # it then writes template soils but CORRECTED lat/lon. Falling
        # through to the raw template gives a surface whose coordinates can
        # never match the per-column domain, and ELM aborts at init.
        out["surface"] = gen.generate_from_mcp(
            lat         = lat,
            lon         = lon,
            mcp_data    = soil or {},
            substrate   = substrate,
            veg_source  = veg_source,
            soil_source = soil_source,
            force       = force,
        )
    else:
        out["surface"] = gen.generate_from_mcp(
            lat         = lat,
            lon         = lon,
            mcp_data    = {"soil_config": soil_config},
            substrate   = substrate,
            veg_source  = veg_source,
            soil_source = soil_source,
            force       = force,
        )
    return out


def build_all(run_dir:     Path,
              only:        Optional[set] = None,
              soil_config: str = "native",
              substrate:   str = "extrapolate",
              force:       bool = False,
              quiet:       bool = False) -> Dict[str, Any]:
    """Every column in the run's columns.json → its input files.

    Reads the SNAPPED coordinates, which is the whole point: columns.json is
    written after the warm start precisely so this step sees where the run
    will actually be, not where the design proposed.
    """
    cf = run_dir / "columns.json"
    if not cf.exists():
        raise FileNotFoundError(
            f"{cf} not found — materialize has not run, so there are no "
            f"columns to build inputs for")
    cols = (json.loads(cf.read_text()) or {}).get("columns") or []
    donors = _warmstart_map(run_dir)

    built:  Dict[str, Dict[str, str]] = {}
    failed: Dict[str, str] = {}
    for c in cols:
        cid = c.get("id")
        if only and cid not in only:
            continue
        donor = donors.get(cid) or {}
        tmpl  = donor.get("surface_template") or donor.get("template")
        try:
            built[cid] = build_one(
                lat              = float(c["lat"]),
                lon              = float(c["lon"]),
                soil             = c.get("soil"),
                surface_template = tmpl,
                soil_config      = soil_config,
                substrate        = substrate,
                force            = force)
            if not quiet:
                print(f"  ✓ {cid}  {'warm' if tmpl else 'cold'}  "
                      f"({c['lat']:.4f}, {c['lon']:.4f})")
        except Exception as e:                                  # noqa: BLE001
            failed[cid] = f"{type(e).__name__}: {e}"
            if not quiet:
                print(f"  ✗ {cid}  {failed[cid]}")

    res = {"run_dir": str(run_dir), "n_columns": len(cols),
           "built": built, "failed": failed,
           "warm_started": bool(donors)}
    (run_dir / "01_inputs" / "column_inputs.json").write_text(
        json.dumps(res, indent=2))
    if not quiet:
        print(f"\n  {len(built)}/{len(built) + len(failed)} column(s) built "
              f"→ 01_inputs/column_inputs.json")
    return res


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2].strip())
    ap.add_argument("--run-dir", help="run directory holding columns.json")
    ap.add_argument("--only", default="", help="comma-separated column ids")
    ap.add_argument("--lat", type=float, help="single column, ad hoc")
    ap.add_argument("--lon", type=float)
    ap.add_argument("--template", help="surface template (donor surfdata)")
    ap.add_argument("--soil-config", default="native",
                    choices=["native", "sandy", "loamy", "clayey"])
    ap.add_argument("--substrate", default="extrapolate")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even when the file already exists")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.lat is not None and a.lon is not None:
        out = build_one(a.lat, a.lon, surface_template=a.template,
                        soil_config=a.soil_config, substrate=a.substrate,
                        force=a.force)
        print(json.dumps(out, indent=2))
        return 0

    if not a.run_dir:
        ap.error("give --run-dir, or --lat and --lon for a single column")

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    res  = build_all(Path(a.run_dir), only=only or None,
                     soil_config=a.soil_config, substrate=a.substrate,
                     force=a.force, quiet=a.quiet)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
