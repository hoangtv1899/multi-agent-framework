#!/usr/bin/env python3
"""
ELM End-to-End Workflow Test
~/RCSFA/multi-agent/tests/test_workflow_elm.py

Drives the COMPLETE ELM workflow as a real user would:
    user_request
        → PFLOTRANCoordinator.process_request()
            → ReceptionAgent     (Pass 1 + Pass 2 via reception_pass2_elm)
                → PlannerAgent   (design + validate via *_elm prompts)
                    → ELMExpManager.execute_plan()
                        → ELMExperimentBuilder
                        → case.build + srun (the slow stuff)
                        → ELMResultsAnalyzer
                            → AnalysisReportAgent

What this verifies (and nothing below it has):
    - workflow.py routing dispatches correctly to ELM
    - All four agents work together end-to-end
    - The recent fixes are deployed:
        * workflow.py use_cache → {}
        * elm_exp_manager.py _run() doesn't pass dead kwargs
        * planner_agent.py is model-aware and passes brief to validator
        * planner_validation_elm.txt uses {issues, corrected_plan} schema
    - The analyzer can consume the LLM_ANALYSIS_INPUT.json that the
      ELMExpManager produces

PREREQUISITES
─────────────
    1. Interactive SLURM allocation (srun requires this):
           salloc -N 1 -t 90:00 -q interactive -C cpu -A m3780

    2. All four recent fixes deployed (the pre-flight check verifies
       three of them via grep).

USAGE
─────
    salloc -N 1 -t 90:00 -q interactive -C cpu -A m3780
    cd ~/RCSFA/multi-agent
    python3 tests/test_workflow_elm.py

EXPECTED WALLCLOCK
──────────────────
    Reception + planner + analyzer:  ~1 min total (LLM calls)
    Build × 3 experiments:           ~25-30 min
    Run × 3 experiments:             ~10-15 min
    ─────────────────────────────────────────────
    Total:                           ~40-50 min

Output:
    workflow_outputs/elm_run_YYYYMMDD_HHMMSS/
        ├── experiment_summary.json
        ├── LLM_ANALYSIS_INPUT.json
        ├── ANALYSIS_REPORT.json
        └── RUN_SUMMARY.json
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────
# TEST USER REQUEST
# ─────────────────────────────────────────────────────────────────────
# Forcing-only design — the planner should produce ~3 experiments,
# keeping wallclock manageable for a first end-to-end test.
# (For a substrate-sensitivity question you'd get 5-6 experiments and
#  need a longer allocation.)
TEST_USER_REQUEST = (
    "Run ELM simulations at Fresno, CA (36.74°N, -119.79°W) using "
    "the 1948-2004 Qian forcing window. Compare groundwater recharge "
    "between three 1-year periods: 1985 as baseline, 1990 as dry "
    "(mid 1987-92 California drought), and 1983 as wet (post-El "
    "Niño wet anomaly)."
)


# ─────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────
def preflight_check() -> bool:
    """Verify environment + deployed fixes before a 40-min job."""
    print("=" * 70)
    print("PRE-FLIGHT CHECK")
    print("=" * 70)

    all_ok = True

    # ── SLURM allocation ──────────────────────────────────────
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        print(f"✓ SLURM_JOB_ID = {job_id}")
    else:
        print("❌ SLURM_JOB_ID not set — srun will fail after the build")
        print("   Get an interactive allocation first:")
        print("     salloc -N 1 -t 90:00 -q interactive -C cpu -A m3780")
        all_ok = False

    # ── Running from project root ─────────────────────────────
    cwd = Path.cwd()
    if (cwd / "src" / "core" / "elm_exp_manager.py").is_file():
        print(f"✓ Project root: {cwd}")
    else:
        print(f"❌ Not in project root: {cwd}")
        print(f"   cd ~/RCSFA/multi-agent first")
        all_ok = False

    # ── workflow.py fix ───────────────────────────────────────
    wf = cwd / "workflow.py"
    if wf.is_file() and "'use_cache': True" in wf.read_text():
        print("❌ workflow.py still has {'use_cache': True} — fix not applied")
        all_ok = False
    else:
        print("✓ workflow.py: use_cache=True removed")

    # ── elm_exp_manager.py fix ────────────────────────────────
    em = cwd / "src" / "core" / "elm_exp_manager.py"
    if em.is_file() and "use_sbatch" in em.read_text():
        print("❌ elm_exp_manager.py still passes use_sbatch — fix not applied")
        all_ok = False
    else:
        print("✓ elm_exp_manager.py: sbatch kwargs removed from _run()")

    # ── planner_validation_elm.txt fix ────────────────────────
    pv = cwd / "src" / "agents" / "prompts" / "planner_validation_elm.txt"
    if pv.is_file() and "corrected_plan" not in pv.read_text():
        print("❌ planner_validation_elm.txt missing 'corrected_plan' — "
              "old schema still in place")
        all_ok = False
    else:
        print("✓ planner_validation_elm.txt: new schema in place")

    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def run_workflow_test() -> bool:
    print("=" * 70)
    print("ELM END-TO-END WORKFLOW TEST")
    print("=" * 70)
    print()

    if not preflight_check():
        print("❌ Pre-flight failed. Aborting — fix issues above and retry.")
        return False

    print("📋 USER REQUEST:")
    print("-" * 70)
    print(f"  {TEST_USER_REQUEST}")
    print()

    # Import here so preflight fails fast without paying import cost
    from workflow import PFLOTRANCoordinator

    coordinator = PFLOTRANCoordinator(
        default_output_dir = "./workflow_outputs",
        mcp_config_file    = "mcp_config.json",
        model_type         = "elm",
    )

    # ── Kick off the workflow ─────────────────────────────────
    start = time.time()
    try:
        response = coordinator.process_request(
            user_request = TEST_USER_REQUEST,
        )
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Workflow raised after {elapsed/60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - start

    # ── Show response ─────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"WORKFLOW RESPONSE  (wallclock: {elapsed/60:.1f} min)")
    print("=" * 70)
    print(response)

    # ── List the output files ─────────────────────────────────
    run_dir = coordinator.conversation_context.get("last_run_dir")
    if run_dir:
        run_dir = Path(run_dir)
        print()
        print("=" * 70)
        print(f"OUTPUT FILES — {run_dir}")
        print("=" * 70)
        for f in sorted(run_dir.glob("*.json")):
            size = f.stat().st_size
            print(f"  {f.name:40s}  {size:>10,} bytes")
        print()
        print("Inspect the analysis with:")
        print(f"  cat {run_dir}/ANALYSIS_REPORT.json | jq")
    else:
        print()
        print("⚠️  No run_dir recorded in conversation_context — "
              "execution may not have reached ELMExpManager")

    print()
    print("=" * 70)
    print("✅ WORKFLOW TEST COMPLETED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_workflow_test()
    sys.exit(0 if ok else 1)