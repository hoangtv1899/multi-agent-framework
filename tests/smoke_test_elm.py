#!/usr/bin/env python3
"""
ELM Pipeline Smoke Test
tests/smoke_test_elm.py

End-to-end no-LLM test of the ELM pipeline:
    builder → adapter → wrapper → real ELM build → real srun → analyzer

What this verifies:
    1. The simplified wrapper drives create_newcase + xmlchange +
       namelists + case.setup + case.build successfully
    2. The adapter satisfies the ModelAgentBase contract end-to-end
    3. srun produces NetCDF history files on disk
    4. ELMResultsAnalyzer can read the output

What this does NOT exercise:
    - PlannerAgent / ReceptionAgent / AnalysisReportAgent (no LLMs)
    - workflow.py routing
    - Multi-experiment plans (single baseline only)

Requires:
    - An active Perlmutter interactive node allocation
    - Working E3SM source at /global/u2/h/hvtran/E3SM
    - About 15-20 minutes wall time total

Run:
    # If not already in an allocation:
    salloc -N 1 -t 30:00 -q interactive -C cpu -A m3780

    cd ~/RCSFA/multi-agent
    python3 tests/smoke_test_elm.py
"""
import os
import sys
import time
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("smoke_test_elm")


# ────────────────────────────────────────────────────────────────────
# Plan: one short baseline experiment
# ────────────────────────────────────────────────────────────────────
PLAN = {
    "CONDITIONS_COUPLERS": [
        {
            "EXPERIMENT":            "elm_smoke_test",
            "FORCING_PERIOD":        "baseline",
            "STOP_N":                "1",
            "DATM_CLMNCEP_YR_START": "1981",
            "DATM_CLMNCEP_YR_END":   "1981",
            "RUN_STARTDATE":         "1981-01-01",
            "DESCRIPTION":           "Smoke test: 1-year baseline",
        },
    ],
    "ELM_CONFIG": {
        "base_stop_option": "nyears",
        "base_rest_n":      "1",
        "base_rest_option": "nyears",
    },
    "TIME": {
        "forcing_start": 1981,
        "forcing_end":   1981,
    },
}


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
START_TIME = time.time()


def elapsed() -> str:
    s = int(time.time() - START_TIME)
    return f"{s // 60}m{s % 60:02d}s"


def banner(msg: str):
    print()
    print("─" * 72)
    print(f"  [{elapsed()}]  {msg}")
    print("─" * 72)


def fail(msg: str) -> int:
    print()
    print("=" * 72)
    print(f"  ✗ SMOKE TEST FAILED  ({elapsed()})")
    print("=" * 72)
    print(f"  {msg}")
    print()
    return 1


def passed(case_dir: Path) -> int:
    print()
    print("=" * 72)
    print(f"  ✓ SMOKE TEST PASSED  ({elapsed()})")
    print("=" * 72)
    print(f"  Case directory: {case_dir}")
    print("  Inspect output:")
    print(f"    ls -lh {case_dir}/run/*.nc")
    print()
    return 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main() -> int:
    from core.elm_experiment_builder import ELMExperimentBuilder
    from core.elm_results_analyzer   import ELMResultsAnalyzer

    print()
    print("=" * 72)
    print("  ELM PIPELINE SMOKE TEST")
    print("=" * 72)
    print("  Verifies wrapper + adapter + builder + analyzer end-to-end.")
    print("  Estimated time: 10–20 minutes  (build ~10 min, run ~5 min).")
    print()

    # ── Environment check ────────────────────────────────────────
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        print("⚠️  WARNING: SLURM_JOB_ID not set.")
        print("   srun will fail outside of a node allocation.")
        print("   To allocate one:")
        print("     salloc -N 1 -t 30:00 -q interactive -C cpu -A m3780")
        print()
    else:
        print(f"✓ Inside SLURM allocation {slurm_job_id}")

    pscratch = os.environ.get("PSCRATCH")
    if not pscratch:
        return fail("$PSCRATCH not set — required for ELM case directories")
    print(f"✓ PSCRATCH = {pscratch}")

    # ── Step 1: Build experiments from plan ──────────────────────
    banner("Step 1: Build experiments from plan")
    try:
        builder     = ELMExperimentBuilder(PLAN)
        experiments = builder.build_experiments()
    except Exception as e:
        return fail(f"build_experiments raised: {e}")

    if len(experiments) != 1:
        return fail(f"Expected 1 experiment, got {len(experiments)}")
    print(f"✓ {len(experiments)} experiment built: "
          f"{experiments[0]['case_name']}")

    # ── Step 2: Prepare case (slow: ~10 min) ─────────────────────
    banner("Step 2: Prepare case — create_newcase → xmlchange → build")
    print("  This is the slow part (~10 min). Watch the log for progress.")
    try:
        case_dirs = builder.prepare_cases(output_dir="/tmp/elm_smoke_test")
    except Exception as e:
        return fail(f"prepare_cases raised: {e}")

    case_dir = Path(case_dirs[0])
    print(f"✓ Case prepared: {case_dir}")

    # ── Step 3: Verify executable ────────────────────────────────
    banner("Step 3: Verify build artifact")
    exe_path = case_dir / "build" / "e3sm.exe"
    if not exe_path.exists():
        return fail(f"Executable not found at {exe_path}")
    size_mb = exe_path.stat().st_size / 1e6
    print(f"✓ e3sm.exe exists ({size_mb:.1f} MB): {exe_path}")

    # ── Step 4: Run simulation ───────────────────────────────────
    banner("Step 4: Run simulation via srun (~5 min)")
    adapter = experiments[0]["elm_agent"]
    try:
        success = adapter.run_simulation()
    except Exception as e:
        return fail(f"run_simulation raised: {e}")

    if not success:
        return fail("run_simulation returned False — check srun output above")
    print("✓ Simulation completed")

    # ── Step 5: Verify history files ─────────────────────────────
    banner("Step 5: Verify history files on disk")
    summary       = adapter.get_run_summary()
    history_files = summary.get("history_files", [])

    if not history_files:
        run_dir = case_dir / "run"
        print(f"  No *.elm.h0.*.nc files found in {run_dir}")
        if run_dir.exists():
            print(f"  Contents of {run_dir} (first 20 entries):")
            for f in sorted(run_dir.iterdir())[:20]:
                print(f"    {f.name}")
        return fail("No history files produced")

    print(f"✓ Found {len(history_files)} history file(s):")
    for f in history_files:
        size_mb = Path(f).stat().st_size / 1e6
        print(f"    {Path(f).name}  ({size_mb:.1f} MB)")

    # ── Step 6: Run analyzer ─────────────────────────────────────
    banner("Step 6: Run analyzer on the output")
    with tempfile.TemporaryDirectory() as analysis_dir:
        try:
            analyzer = ELMResultsAnalyzer(
                experiments  = experiments,
                analysis_dir = analysis_dir,
            )
            results = analyzer.extract_all()
        except Exception as e:
            return fail(f"analyzer.extract_all raised: {e}")

    result = results.get("elm_smoke_test")
    if result is None:
        return fail("Analyzer returned no result for 'elm_smoke_test'")

    status = result.get("status")
    print(f"✓ Analyzer status: {status}")

    if status != "ok":
        reason = result.get("reason", "(no reason given)")
        return fail(f"Analyzer status is '{status}' — {reason}")

    metrics = result.get("metrics", {})
    if metrics:
        print("  Sample metrics:")
        for k, v in list(metrics.items())[:8]:
            print(f"    {k:.<40} {v}")

    return passed(case_dir)


if __name__ == "__main__":
    sys.exit(main())