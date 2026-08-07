#!/usr/bin/env python3
"""
Build (clone) ELM cases for every column in an executable plan, sharing one
pre-built executable.

Reads plan.json (CONDITIONS_COUPLERS, from columns_to_plan.py), generates each
column's surface/domain files, then clones one case per column from a reference
case with --keepexe (~15s each, shared exe). Cases land under $PSCRATCH/E3SMv3/.
Writes <out-dir>/cases.json (the cloned case dirs) + exe_path.txt. The
build/clone runs on the login node; the ELM run is a separate salloc step
(tools/run_cases.sh).

    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/build_cases.py --plan <plan.json> --ref <ref_case_dir> --out-dir <run_dir>
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent                    # scripts/
_FRAMEWORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HERE.parent / "src"))              # ELM modules
sys.path.insert(0, str(_FRAMEWORK / "src"))                # framework
from elm_experiment_builder import ELMExperimentBuilder


def main():
    ap = argparse.ArgumentParser(description="Clone an ELM case per column from a reference")
    ap.add_argument("--plan", required=True, help="executable plan.json (columns_to_plan.py)")
    ap.add_argument("--ref", required=True, help="reference case dir to clone from (shared exe)")
    ap.add_argument("--out-dir", required=True, help="dir to write cases.json + exe_path.txt")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    builder = ELMExperimentBuilder(plan)
    exps = builder.build_experiments()                  # generates per-column surfaces
    print(f"cloning {len(exps)} columns from {Path(args.ref).name} ...", flush=True)

    case_dirs = [None] * len(exps)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(e["elm_agent"].prepare_case, ref_case_dir=args.ref): i
                for i, e in enumerate(exps)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                case_dirs[i] = str(f.result())
                print(f"  ✓ {exps[i]['case_name']}", flush=True)
            except Exception as exc:
                print(f"  ✗ {exps[i]['case_name']}: {exc}", flush=True)

    ok = [c for c in case_dirs if c]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(ok, indent=2))
    # Forcing is not chosen here: elm_wrapper's FIXED_XML sets DATM_MODE=
    # CLMMOSARTTEST, which reads the NLDAS-2 streams that are actually
    # populated on Compy. (The old --forcing flag rewrote stream files to
    # atm_forcing.datm7.NLDAS2.0.125d.v1, whose Precip/TPQWL dirs are EMPTY
    # here — it silently produced cases that could not run.)
    (out / "forcing.txt").write_text("NLDAS-2 (0.125°, ~12 km) via DATM_MODE=CLMMOSARTTEST")
    ledger = list(plan.get("assumptions_ledger") or [])
    ledger.append({"parameter": "met forcing",
                   "value": "NLDAS-2 0.125°" if args.forcing == "nldas" else "Qian T62",
                   "source": "DEFAULT" if args.forcing == "nldas" else "user",
                   "note": "NLDAS-2 is the framework default for CONUS"})
    (out / "assumptions.json").write_text(json.dumps(ledger, indent=2))
    exe = str(Path(args.ref) / "build" / "e3sm.exe")
    (out / "exe_path.txt").write_text(exe)
    print(f"\n{len(ok)}/{len(exps)} cloned -> {out / 'cases.json'}\nshared exe: {exe}")


if __name__ == "__main__":
    main()
