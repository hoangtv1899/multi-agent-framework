#!/usr/bin/env python3
"""
ELM Planner Smoke Test
~/RCSFA/multi-agent/tests/test_planner_elm.py

Tests PlannerAgent in isolation:
    handcrafted brief
        → PlannerAgent.create_plan() (2 LLM calls: design + validate)
            → plan JSON
                → Python sanity check (math, bounds, vocabulary)
                    → ELMExperimentBuilder schema check

What this DOES verify:
    - planner_system_elm prompt produces a sensible experiment design
    - planner_validation_elm prompt fixes (or passes) the plan
    - Output schema matches what ELMExperimentBuilder consumes
    - STOP_N / RUN_STARTDATE / vocabulary are internally consistent

What this DOES NOT verify:
    - ELM actually runs (no case build, no srun)
    - Reception → brief synthesis works
    - Analyzer interprets results correctly

Total wallclock: ~30s (just 2 LLM calls + Python validation).
Runs on login node — no SLURM allocation needed.

USAGE
─────
    cd ~/RCSFA/multi-agent

    # Use the built-in test brief:
    python3 tests/test_planner_elm.py

    # Or use your own brief from a JSON file:
    python3 tests/test_planner_elm.py path/to/my_brief.json

Output:
    - Plan printed to stdout (compact summary + full JSON)
    - Plan saved to /tmp/elm_planner_test_plan.json
    - Exit 0 if plan is structurally valid, 1 otherwise
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from agents.planner_agent          import PlannerAgent
from core.elm_experiment_builder   import ELMExperimentBuilder


# ─────────────────────────────────────────────────────────────────────
# HANDCRAFTED BRIEF — what reception_pass2_elm would produce
# ─────────────────────────────────────────────────────────────────────
TEST_BRIEF = {
    "user_request":
        "How does projected groundwater recharge respond to dry vs wet "
        "climate periods at the Hanford Reach site, and how sensitive "
        "is the response to deep-soil substrate properties?",

    "intent":           "design_and_run",
    "experiment_focus":
        "Quantify recharge sensitivity to (a) forcing-period climate "
        "variability and (b) unknown deep-soil substrate. Cross-axis "
        "design to separate the two effects.",

    "region": {
        "location": "Hanford Reach National Monument, WA",
        "lat":      46.50,
        "lon":      -119.30,
    },

    "forcing_data": {
        "available_start_year": 1981,
        "available_end_year":   2020,
        "source":               "GSWP3 v1 / DATM_CLMNCEP",
    },

    "climate_baseline": {
        "precip_mm_yr": 180.0,
        "temp_min_c":   -2.5,
        "temp_max_c":   24.8,
        "period":       "1981-2020",
    },

    "climate_dry": {
        "precip_mm_yr": 125.0,
        "period":       "2001-2005",
        "note":         "Sustained PNW drought, low cool-season precip",
    },

    "climate_wet": {
        "precip_mm_yr": 255.0,
        "period":       "1996-2000",
        "note":         "Wet anomaly, strong winter storm tracks",
    },

    "soil_profile": {
        "num_layers":       4,
        "source":           "SSURGO",
        "depth_coverage_m": 1.8,
        "layers": [
            {
                "depth_top_m":  0.00, "depth_bot_m": 0.30,
                "sand_pct":     65,   "clay_pct":    12,
                "organic_pct":  2.0,  "gravel_pct":  5,
            },
            {
                "depth_top_m":  0.30, "depth_bot_m": 0.80,
                "sand_pct":     60,   "clay_pct":    18,
                "organic_pct":  1.0,  "gravel_pct":  8,
            },
            {
                "depth_top_m":  0.80, "depth_bot_m": 1.30,
                "sand_pct":     55,   "clay_pct":    22,
                "organic_pct":  0.5,  "gravel_pct":  12,
            },
            {
                "depth_top_m":  1.30, "depth_bot_m": 1.80,
                "sand_pct":     50,   "clay_pct":    25,
                "organic_pct":  0.2,  "gravel_pct":  15,
            },
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────
# PYTHON-SIDE SANITY CHECK — catches issues the LLM validator missed
# ─────────────────────────────────────────────────────────────────────
ALLOWED_FORCING_PERIOD = {"baseline", "dry", "wet"}
ALLOWED_SOIL_CONFIG    = {"native", "sandy", "loamy", "clayey"}
ALLOWED_SUBSTRATE      = {"template", "extrapolate", "sandy", "clayey"}


def sanity_check_plan(plan: dict, brief: dict) -> list:
    """Returns list of issues; empty list = plan passes Python checks."""
    issues = []

    couplers = plan.get("CONDITIONS_COUPLERS", [])
    if not couplers:
        issues.append("Plan has no CONDITIONS_COUPLERS")
        return issues

    # Get forcing window
    fdata     = brief.get("forcing_data", {})
    yr_min    = int(fdata.get("available_start_year", 1981))
    yr_max    = int(fdata.get("available_end_year",   2020))

    # Per-experiment checks
    seen_names = set()
    for i, c in enumerate(couplers):
        ctx = f"experiment[{i}] '{c.get('EXPERIMENT', '?')}'"

        # Vocabulary
        fp = c.get("FORCING_PERIOD")
        if fp not in ALLOWED_FORCING_PERIOD:
            issues.append(f"{ctx}: FORCING_PERIOD={fp!r} not in {ALLOWED_FORCING_PERIOD}")

        sc = c.get("SOIL_CONFIG")
        if sc not in ALLOWED_SOIL_CONFIG:
            issues.append(f"{ctx}: SOIL_CONFIG={sc!r} not in {ALLOWED_SOIL_CONFIG}")

        sub = c.get("SUBSTRATE")
        if sc == "native":
            if sub is None:
                issues.append(f"{ctx}: SOIL_CONFIG=native but SUBSTRATE missing")
            elif sub not in ALLOWED_SUBSTRATE:
                issues.append(f"{ctx}: SUBSTRATE={sub!r} not in {ALLOWED_SUBSTRATE}")
        else:
            if sub is not None:
                issues.append(f"{ctx}: SUBSTRATE={sub!r} present but SOIL_CONFIG={sc!r}")

        # Date arithmetic
        try:
            ys     = int(c["DATM_CLMNCEP_YR_START"])
            ye     = int(c["DATM_CLMNCEP_YR_END"])
            stop_n = int(c["STOP_N"])
        except (KeyError, ValueError) as e:
            issues.append(f"{ctx}: cannot parse year/STOP_N fields ({e})")
            continue

        if stop_n != (ye - ys + 1):
            issues.append(
                f"{ctx}: STOP_N={stop_n} but yr_end - yr_start + 1 = {ye-ys+1}"
            )

        rsd      = c.get("RUN_STARTDATE", "")
        expected = f"{ys:04d}-01-01"
        if rsd != expected:
            issues.append(f"{ctx}: RUN_STARTDATE={rsd!r}, expected {expected!r}")

        # Bounds
        if ys < yr_min or ye > yr_max:
            issues.append(
                f"{ctx}: year range {ys}-{ye} outside forcing window {yr_min}-{yr_max}"
            )

        # Uniqueness
        name = c.get("EXPERIMENT")
        if name in seen_names:
            issues.append(f"Duplicate EXPERIMENT name: {name!r}")
        seen_names.add(name)

    # ELM_CONFIG required fields
    elmcfg = plan.get("ELM_CONFIG", {})
    for field in ("base_stop_option", "base_rest_n", "base_rest_option"):
        if field not in elmcfg:
            issues.append(f"ELM_CONFIG missing required field: {field}")

    return issues


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def run_test(brief: dict) -> bool:
    print("=" * 70)
    print("ELM PLANNER SMOKE TEST")
    print("=" * 70)
    print()
    print("📋 INPUT BRIEF:")
    print("-" * 70)
    print(f"  Location: {brief['region']['location']}")
    print(f"  Question: {brief['user_request'][:80]}...")
    fdata = brief.get("forcing_data", {})
    print(f"  Forcing:  {fdata.get('available_start_year')}-"
          f"{fdata.get('available_end_year')}")
    cd, cw = brief.get("climate_dry", {}), brief.get("climate_wet", {})
    print(f"  Dry:      {cd.get('period', 'null')}")
    print(f"  Wet:      {cw.get('period', 'null')}")
    sp = brief.get("soil_profile", {})
    print(f"  SSURGO:   {sp.get('num_layers')} layers, "
          f"{sp.get('depth_coverage_m')} m deep")
    print()

    # Run planner
    print("=" * 70)
    print("RUNNING PLANNER (2 LLM calls: design + validate)")
    print("=" * 70)
    planner = PlannerAgent(model_type="elm")
    plan    = planner.create_plan(brief)
    print()

    # Compact summary
    print("=" * 70)
    print("PLAN SUMMARY:")
    print("=" * 70)
    couplers = plan.get("CONDITIONS_COUPLERS", [])
    print(f"  {len(couplers)} experiments designed")
    print()
    print(f"  {'name':<35} {'forcing':<10} {'years':<11} "
          f"{'soil':<8} {'sub':<11}")
    print(f"  {'-'*35} {'-'*10} {'-'*11} {'-'*8} {'-'*11}")
    for c in couplers:
        years = f"{c.get('DATM_CLMNCEP_YR_START','?')}-" \
                f"{c.get('DATM_CLMNCEP_YR_END','?')}"
        print(f"  {c.get('EXPERIMENT','?'):<35} "
              f"{c.get('FORCING_PERIOD','?'):<10} "
              f"{years:<11} "
              f"{c.get('SOIL_CONFIG','?'):<8} "
              f"{c.get('SUBSTRATE','—'):<11}")
    print()
    ps = plan.get("parameter_space", {})
    print(f"  Primary axis:   {ps.get('primary_axis', '?')}")
    print(f"  Secondary axis: {ps.get('secondary_axis', '?')}")
    print(f"  Question:       {ps.get('scientific_question', '?')}")
    print()

    # Save full plan
    out_path = Path("/tmp/elm_planner_test_plan.json")
    out_path.write_text(json.dumps(plan, indent=2, default=str))
    print(f"💾 Full plan saved to: {out_path}")
    print()

    # Python sanity check
    print("=" * 70)
    print("PYTHON SANITY CHECK")
    print("=" * 70)
    issues = sanity_check_plan(plan, brief)
    if issues:
        print(f"⚠️  Found {len(issues)} issue(s) the LLM validator missed:")
        for issue in issues:
            print(f"   - {issue}")
        sanity_ok = False
    else:
        print("✓ All Python-side checks passed")
        sanity_ok = True
    print()

    # Builder schema check
    print("=" * 70)
    print("BUILDER SCHEMA CHECK")
    print("=" * 70)
    try:
        builder     = ELMExperimentBuilder(plan)
        experiments = builder.build_experiments()
        print(f"✓ Builder accepted plan, created {len(experiments)} "
              f"experiment object(s)")
        builder_ok = True
    except Exception as e:
        print(f"❌ Builder REJECTED plan: {e}")
        builder_ok = False
    print()

    # Verdict
    print("=" * 70)
    if sanity_ok and builder_ok:
        print("✅ PLANNER TEST PASSED")
        print("=" * 70)
        return True
    else:
        print("❌ PLANNER TEST FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    brief = TEST_BRIEF
    if len(sys.argv) > 1:
        brief_path = Path(sys.argv[1])
        print(f"📂 Loading brief from: {brief_path}")
        brief = json.loads(brief_path.read_text())

    ok = run_test(brief)
    sys.exit(0 if ok else 1)