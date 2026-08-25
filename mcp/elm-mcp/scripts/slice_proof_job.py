#!/usr/bin/env python3
"""The Phase-2 proof: two 7-day slices must land where 14 straight days land.

    python scripts/slice_proof_job.py <ref_case_dir> <work_dir>

Run INSIDE a SLURM allocation (the model runs via srun). Clones the given
built case twice with --keepexe (no compile): "straight" runs 14 days in one
go; "sliced" runs 7, is pointed at its next window by
elm_wrapper.configure_continuation, and runs 7 more — with NO stamping in
between, because this proof isolates the restart seam itself. The two final
land restarts are then compared variable by variable.

E3SM's own contract is restart exactness (a continued run is bitwise the
run that never stopped), so the expected difference is ZERO — anything
larger means the seam leaks and the walk-through-the-year coupling cannot
be built on it.

A separate FILE, like ensemble_job.py, for the same reasons: runnable by
hand when something fails, readable without shell quoting, syntax-checked
by import rather than after a queue wait.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import elm_wrapper as ew                               # noqa: E402

SLICE_DAYS = 7
N_SLICES = 2


def sh(args, cwd):
    r = subprocess.run(args, cwd=cwd, env=ew._cime_env(),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args))} in {cwd}:\n"
                           f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


def clone(ref: Path, dest: Path, stop_n: int) -> Path:
    cimeroot = sh(["./xmlquery", "CIMEROOT", "--value"], cwd=ref).strip()
    sh([str(Path(cimeroot) / "scripts" / "create_clone"),
        "--case", str(dest), "--clone", str(ref), "--keepexe"], cwd=ref)
    for kv in (f"STOP_N={stop_n}", "STOP_OPTION=ndays",
               f"REST_N={stop_n}", "REST_OPTION=ndays",
               "CONTINUE_RUN=FALSE"):
        sh(["./xmlchange", kv], cwd=dest)
    sh(["./case.setup"], cwd=dest)
    return dest


def main(ref_case, work_dir) -> int:
    ref, work = Path(ref_case), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    straight = clone(ref, work / "straight", stop_n=SLICE_DAYS * N_SLICES)
    sliced = clone(ref, work / "sliced", stop_n=SLICE_DAYS)

    print("== straight: one 14-day run", flush=True)
    assert ew.run_built_case(straight), "straight run failed"

    print("== sliced: 7 days", flush=True)
    assert ew.run_built_case(sliced), "slice 1 failed"
    print("   restart after slice 1:", ew.latest_restart(sliced).name,
          flush=True)
    ew.configure_continuation(sliced, stop_n=SLICE_DAYS)
    print("== sliced: 7 more days (CONTINUE_RUN)", flush=True)
    assert ew.run_built_case(sliced), "slice 2 failed"

    fs, fx = ew.latest_restart(straight), ew.latest_restart(sliced)
    print(f"comparing {fs.name} vs {fx.name}", flush=True)
    # SAME DATE OR NO COMPARISON. A continuation that silently re-runs its
    # first window leaves the pointer on the earlier date, and comparing
    # day 15 against day 8 reports huge "leaks" that are really a slice
    # that never advanced (seen on job 774959).
    ds, dx = fs.name.split(".r.")[-1], fx.name.split(".r.")[-1]
    if ds != dx:
        print(f"VERDICT: SLICE DID NOT ADVANCE — straight ends at {ds}, "
              f"sliced at {dx}; the continuation never took effect")
        return 1

    import numpy as np
    from netCDF4 import Dataset
    worst = {}
    with Dataset(fs) as a, Dataset(fx) as b:
        for var in ("H2OSOI_LIQ", "H2OSOI_ICE", "ZWT", "WA", "T_SOISNO"):
            if var not in a.variables or var not in b.variables:
                worst[var] = "absent"
                continue
            va, vb = a[var][:], b[var][:]
            worst[var] = float(np.nanmax(np.abs(
                np.asarray(va, dtype=float) - np.asarray(vb, dtype=float))))
    print("max |straight - sliced| per variable:")
    for k, v in worst.items():
        print(f"    {k:12s} {v}")
    bad = [k for k, v in worst.items()
           if isinstance(v, float) and v > 1e-9]
    print("VERDICT:", "EXACT — the seam does not leak" if not bad
          else f"LEAKS in {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
