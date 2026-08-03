#!/usr/bin/env python3
"""What the build_elm_cases batch job runs: the CIME case build.

mcp/elm-mcp/build_cases_job.py

    python build_cases_job.py <run_dir>

Driven entirely by the CASE LIST the framework wrote — one entry per column,
carrying its case_name and its runtime_config. That config already names every
file the case needs:

    FSURDAT           the surface data the framework generated
    FINIDAT           the warm-start initial state, subset from the CONUS restart
    LND_DOMAIN_*      the domain files
    ATM_DOMAIN_*
    STOP_N, RUN_STARTDATE, DATM_CLMNCEP_YR_*, REST_*

So this never opens the CONUS surfdata, never generates a surface, and never
needs the plan. Everything the framework's warm start decided crosses the
boundary as data, and this side only compiles cases against it. That is the
whole reason the boundary is here and not one stage earlier.

A separate FILE rather than a heredoc in the sbatch script: it can be run by
hand when a build fails, read without untangling shell quoting, and a syntax
error is found by importing it rather than after a queue wait.

Writes <run_dir>/01_inputs/built_cases.json — where each case landed, which is
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
os.environ.setdefault("LANG", "en_US.utf8")

CASE_INPUTS = "case_inputs.json"
RESULT_NAME = "built_cases.json"


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    out_path = rd / "01_inputs" / RESULT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = rd / "01_inputs" / CASE_INPUTS
    if not src.is_file():
        out_path.write_text(json.dumps(
            {"ok": False, "error": f"no {CASE_INPUTS} in {rd / '01_inputs'} — "
                                   f"the framework builds the inputs and writes "
                                   f"this file"}, indent=2))
        print(f"BUILD_FAILED: no {CASE_INPUTS}")
        return 1

    try:
        from core.elm_input_agent import ELMAgentAdapter
        from core.elm_experiment_builder import ELMExperimentBuilder

        cases = json.loads(src.read_text())
        missing = [c.get("case_name") for c in cases
                   if not (c.get("runtime_config") or {})]
        if missing:
            raise ValueError(
                f"these cases have no runtime_config, so there is nothing to "
                f"build them from: {missing}")

        # The builder owns the expensive part — reference case compiled from
        # scratch, the rest cloned with --keepexe. Doing it per case would be a
        # full compile EACH: ~2 h for a 14-column watershed instead of ~12 min.
        # Its experiment list is normally produced by build_experiments(), which
        # also GENERATES surfaces; here the surfaces already exist, so the list
        # is assembled directly from the case inputs instead.
        builder = ELMExperimentBuilder({})
        builder.experiments = [
            {"case_name": c["case_name"],
             "elm_agent": ELMAgentAdapter(
                 case_name=c["case_name"],
                 runtime_config=dict(c["runtime_config"]))}
            for c in cases
        ]
        print(f"building {len(builder.experiments)} case(s) from "
              f"{CASE_INPUTS}", flush=True)

        case_dirs = builder.prepare_cases(output_dir=str(rd))

        rows = [{"case_name": e["case_name"], "case_dir": cd}
                for e, cd in zip(builder.experiments, case_dirs)]
        n_ok = sum(1 for r in rows if r["case_dir"])
        out_path.write_text(json.dumps(
            {"ok": n_ok > 0, "n_ok": n_ok, "n_total": len(rows),
             "cases": rows}, indent=2, default=str))

        for r in rows:
            print(f"  {r['case_name']}: {r['case_dir'] or 'FAILED TO BUILD'}")
        print(f"BUILD_DONE {n_ok}/{len(rows)}")
        return 0 if n_ok else 1

    except Exception as e:                                      # noqa: BLE001
        # Written to the result file as well as stdout: the poller reads the
        # file, and a build that died must not look like one that never ran.
        out_path.write_text(json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, indent=2))
        print(f"BUILD_FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: build_cases_job.py <run_dir>")
    sys.exit(main(sys.argv[1]))
