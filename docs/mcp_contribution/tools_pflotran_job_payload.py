#!/usr/bin/env python3
"""What a submitted PFLOTRAN ensemble actually runs on the compute node.

tools/pflotran_job_payload.py

    python pflotran_job_payload.py <job_dir>/spec.json

Deliberately thin. It calls run_simulation — the SAME function the inline path
calls — so the batch and inline modes cannot produce different numbers. Anything
it did for itself would be a second implementation to keep in step.

Writes <job_dir>/result.json, which is what job_runner.collect() reads.
A failure is written there TOO: a job that died must not look like one that
never ran, and the poller only ever sees this file.
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(spec_path: str) -> int:
    spec_file = Path(spec_path)
    job_dir = spec_file.parent
    result_file = job_dir / "result.json"

    try:
        spec = json.loads(spec_file.read_text())
    except Exception as e:                                      # noqa: BLE001
        result_file.write_text(json.dumps(
            {"status": "failed", "error": f"unreadable spec: {e}"}, indent=2))
        print(f"ENSEMBLE_FAILED: unreadable spec: {e}")
        return 1

    try:
        from tools.simulation import run_simulation

        decks = spec["input_files"]
        print(f"running {len(decks)} deck(s)", flush=True)
        out = run_simulation(
            input_file=decks,
            executable=spec["executable"],
            num_cores=spec.get("num_cores", 1),
            mpi_command=spec.get("mpi_command", "mpirun"),
            mode="ensemble_parallel" if len(decks) > 1 else "single",
            max_parallel=spec.get("max_parallel"),
            timeout=spec.get("timeout"),
            output_dir=spec.get("output_dir"),
        )

        # results_by_input is the whole point — see job_runner's docstring.
        # Recorded here as an explicit count so a caller can tell "the map is
        # missing" from "the map is empty".
        by_input = (out or {}).get("results_by_input") or {}
        out = dict(out or {})
        out["n_attributed"] = len(by_input)
        out["status"] = out.get("status", "completed")
        result_file.write_text(json.dumps(out, indent=2, default=str))

        n_ok = sum(1 for r in by_input.values()
                   if isinstance(r, dict)
                   and r.get("validation_status") == "success")
        print(f"ENSEMBLE_DONE {n_ok}/{len(decks)} attributed={len(by_input)}")
        return 0

    except Exception as e:                                      # noqa: BLE001
        result_file.write_text(json.dumps(
            {"status": "failed", "error": f"{type(e).__name__}: {e}",
             "traceback": traceback.format_exc()}, indent=2, default=str))
        print(f"ENSEMBLE_FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: pflotran_job_payload.py <job_dir>/spec.json")
    sys.exit(main(sys.argv[1]))
