#!/usr/bin/env python3
"""What the prepare_elm_cases batch job actually runs.

D1 put the CIME case build in a SLURM job rather than on a login node. This is
the payload: it rebuilds the experiment list from the run's persisted plan,
builds the reference case, clones the rest with --keepexe, and writes down
where each case landed.

A separate FILE rather than a heredoc inside the sbatch script, for three
reasons: it can be run by hand when a build fails, it can be read without
untangling shell quoting, and a syntax error in it is found by importing it
rather than eight minutes into a queue slot.

    python prepare_job.py <run_dir>

Writes <run_dir>/01_inputs/prepared_cases.json — the case directories, which is
the one thing the caller cannot work out for itself.
"""
import json
import os
import sys
import traceback
from pathlib import Path

FRAMEWORK = Path(os.getenv(
    "IDEAS_FRAMEWORK_DIR", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(FRAMEWORK / "src"))

os.environ.setdefault("LC_ALL", "en_US.utf8")
os.environ.setdefault("LANG",   "en_US.utf8")

RESULT_NAME = "prepared_cases.json"


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    out_path = rd / "01_inputs" / RESULT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plan_file = rd / "run_plan.json"
    if not plan_file.is_file():
        out_path.write_text(json.dumps(
            {"ok": False, "error": f"no run_plan.json in {rd}"}, indent=2))
        print(f"PREPARE_FAILED: no run_plan.json in {rd}")
        return 1

    try:
        from core.elm_experiment_builder import ELMExperimentBuilder
        plan = json.loads(plan_file.read_text())
        builder = ELMExperimentBuilder(plan)
        experiments = builder.build_experiments()
        print(f"rebuilt {len(experiments)} experiment(s) from the plan",
              flush=True)

        # The reference case is compiled from scratch and the rest are cloned
        # with --keepexe. Doing it per case would be a full compile EACH —
        # ~2 h for a 14-column watershed instead of ~12 min.
        case_dirs = builder.prepare_cases(output_dir=str(rd))

        rows = [{"case_name": e.get("case_name"), "case_dir": cd}
                for e, cd in zip(experiments, case_dirs)]
        n_ok = sum(1 for r in rows if r["case_dir"])
        out_path.write_text(json.dumps(
            {"ok": n_ok > 0, "n_ok": n_ok, "n_total": len(rows),
             "cases": rows}, indent=2, default=str))

        for r in rows:
            print(f"  {r['case_name']}: "
                  f"{r['case_dir'] or 'FAILED TO PREPARE'}")
        print(f"PREPARE_DONE {n_ok}/{len(rows)}")
        return 0 if n_ok else 1

    except Exception as e:                                      # noqa: BLE001
        # Written to the result file as well as stdout: the poller reads the
        # file, and a build that died must not look like one that never ran.
        out_path.write_text(json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, indent=2))
        print(f"PREPARE_FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: prepare_job.py <run_dir>")
    sys.exit(main(sys.argv[1]))
