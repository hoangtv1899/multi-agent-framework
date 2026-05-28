#!/usr/bin/env python3
"""
ELM Reception Smoke Test
~/RCSFA/multi-agent/tests/test_reception_elm.py

Tests ReceptionAgent in isolation for the ELM workflow:
    user_request + (best-effort MCP)
        → ReceptionAgent.process()
            → ReceptionResult (intent + parameters)
                → result.to_planner_brief()
                    → brief JSON

What this DOES verify:
    - Intent classification works (design_and_run vs clarification_needed)
    - Location/parameter extraction from natural language
    - reception_pass2_elm prompt produces a well-formed brief
    - Brief contains all top-level keys the PlannerAgent expects

What this DOES NOT verify:
    - The planner produces a good plan from this brief (separate test)
    - ELM actually runs (test_planner_elm + smoke_test_elm cover those)

Total wallclock: 5-30 s depending on MCP availability and number of queries.
Runs on login node — no SLURM allocation needed.

USAGE
─────
    cd ~/RCSFA/multi-agent

    # Built-in test request:
    python3 tests/test_reception_elm.py

    # Or pass your own request as an argument:
    python3 tests/test_reception_elm.py "Run ELM at Niwot Ridge to study..."

Output:
    - Reception result + brief printed to stdout
    - Brief saved to /tmp/elm_reception_test_brief.json
    - Exit 0 if brief shape is valid, 1 otherwise

CHAINING WITH THE PLANNER TEST
─────────────────────────────
    python3 tests/test_reception_elm.py
    python3 tests/test_planner_elm.py /tmp/elm_reception_test_brief.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from agents.reception_agent import ReceptionAgent
from core.mcp_manager       import MCPManager


# ─────────────────────────────────────────────────────────────────────
# TEST USER REQUEST
# ─────────────────────────────────────────────────────────────────────
TEST_USER_REQUEST = (
    "I want to understand how groundwater recharge dynamics at "
    "Hanford Reach (46.50°N, -119.30°W) respond to dry vs wet "
    "climate periods. Can you run a set of ELM simulations comparing "
    "forcing periods, and also explore how sensitive the results are "
    "to deep-soil substrate properties?"
)


# ─────────────────────────────────────────────────────────────────────
# BRIEF SHAPE VALIDATION (what the planner expects)
# ─────────────────────────────────────────────────────────────────────
REQUIRED_TOP_LEVEL_KEYS = {
    "user_request", "intent", "experiment_focus",
    "region", "forcing_data", "climate_baseline",
}
OPTIONAL_TOP_LEVEL_KEYS = {
    "climate_dry", "climate_wet", "soil_profile",
}
REQUIRED_REGION_KEYS  = {"location", "lat", "lon"}
REQUIRED_FORCING_KEYS = {"available_start_year",
                         "available_end_year",
                         "source"}


def validate_brief_shape(brief: dict) -> list:
    """Returns list of shape issues. Empty list = passes."""
    issues = []

    # Top-level required keys
    missing = REQUIRED_TOP_LEVEL_KEYS - set(brief.keys())
    if missing:
        issues.append(f"Missing required top-level keys: {sorted(missing)}")

    # region must have location + coordinates
    region = brief.get("region", {})
    if not isinstance(region, dict):
        issues.append("'region' is not a dict")
    else:
        rmissing = REQUIRED_REGION_KEYS - set(region.keys())
        if rmissing:
            issues.append(f"region missing keys: {sorted(rmissing)}")
        elif region.get("lat") is None or region.get("lon") is None:
            issues.append("region.lat or region.lon is null "
                          "(coords not extracted)")

    # forcing_data must have year range
    fdata = brief.get("forcing_data", {})
    if not isinstance(fdata, dict):
        issues.append("'forcing_data' is not a dict")
    else:
        fmissing = REQUIRED_FORCING_KEYS - set(fdata.keys())
        if fmissing:
            issues.append(f"forcing_data missing keys: {sorted(fmissing)}")

    return issues


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def run_test(user_request: str) -> bool:
    print("=" * 70)
    print("ELM RECEPTION SMOKE TEST")
    print("=" * 70)
    print()
    print("📋 USER REQUEST:")
    print("-" * 70)
    print(f"  {user_request}")
    print()

    # ── Load MCP (best-effort) ────────────────────────────────
    print("=" * 70)
    print("LOADING MCP TOOLS (best-effort)")
    print("=" * 70)
    try:
        mcp_manager = MCPManager("mcp_config.json")
        mcp_clients = mcp_manager.get_all_clients()
        if mcp_clients:
            print(f"✓ Loaded {len(mcp_clients)} MCP server(s):")
            for name in mcp_clients:
                print(f"  - {name}")
        else:
            print("⚠️  No MCP servers loaded — brief will be sparse "
                  "(no climate or soil data)")
    except Exception as e:
        print(f"⚠️  MCP load failed: {e}")
        print("    Test continues without MCP — brief will be sparse")
        mcp_clients = {}
    print()

    # ── Run ReceptionAgent ────────────────────────────────────
    print("=" * 70)
    print("RUNNING RECEPTION (Pass 1: classify, Pass 2: synthesize)")
    print("=" * 70)
    reception = ReceptionAgent(
        mcp_clients = mcp_clients,
        model_type  = "elm",
    )
    try:
        result = reception.process(
            user_request         = user_request,
            conversation_context = {},
        )
    except Exception as e:
        print(f"❌ ReceptionAgent.process() raised: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # ── Show Pass 1 classification ────────────────────────────
    print("=" * 70)
    print("PASS 1 — INTENT CLASSIFICATION:")
    print("=" * 70)
    print(f"  Intent:     {result.intent}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Parameters:")
    print(json.dumps(
        result.parameters, indent=4, default=str
    ))
    print()

    # ── Handle clarification path ─────────────────────────────
    if result.intent == "clarification_needed":
        print("=" * 70)
        print("CLARIFICATION REQUESTED")
        print("=" * 70)
        for q in result.clarification_questions:
            print(f"  - {q}")
        print()
        print("(Test stops here — reception is asking for more info, "
              "so no brief was produced.)")
        return True

    # ── Handle analyze_existing path ──────────────────────────
    if result.intent == "analyze_existing":
        print("=" * 70)
        print("ANALYZE-EXISTING INTENT")
        print("=" * 70)
        print("Reception classified this as analyze_existing, so no "
              "planner brief is produced.")
        print("(Test passes — classification worked.)")
        return True

    # ── Generate planner brief ────────────────────────────────
    print("=" * 70)
    print("PASS 2 — PLANNER BRIEF:")
    print("=" * 70)
    try:
        brief = result.to_planner_brief()
    except Exception as e:
        print(f"❌ result.to_planner_brief() raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save brief for chaining into planner test
    out_path = Path("/tmp/elm_reception_test_brief.json")
    out_path.write_text(json.dumps(brief, indent=2, default=str))
    print(f"💾 Full brief saved to: {out_path}")
    print()

    # Compact summary
    region = brief.get("region", {}) or {}
    fdata  = brief.get("forcing_data", {}) or {}
    cb     = brief.get("climate_baseline", {}) or {}
    cd     = brief.get("climate_dry", {}) or {}
    cw     = brief.get("climate_wet", {}) or {}
    sp     = brief.get("soil_profile", {}) or {}

    ur = brief.get("user_request", "?")
    ef = brief.get("experiment_focus", "?")
    print(f"  user_request:      {ur[:60]}"
          f"{'...' if len(ur) > 60 else ''}")
    print(f"  intent:            {brief.get('intent', '?')}")
    print(f"  experiment_focus:  {ef[:60]}"
          f"{'...' if len(ef) > 60 else ''}")
    print()
    print(f"  region.location:   {region.get('location', '—')}")
    print(f"  region.lat,lon:    "
          f"({region.get('lat')}, {region.get('lon')})")
    print()
    print(f"  forcing_data:      {fdata.get('available_start_year')}-"
          f"{fdata.get('available_end_year')} "
          f"({fdata.get('source', '?')})")
    print()
    print(f"  climate_baseline:  {cb.get('precip_mm_yr')} mm/yr  "
          f"period={cb.get('period')}")
    print(f"  climate_dry:       {cd.get('precip_mm_yr')} mm/yr  "
          f"period={cd.get('period')}")
    print(f"  climate_wet:       {cw.get('precip_mm_yr')} mm/yr  "
          f"period={cw.get('period')}")
    print()
    if sp:
        print(f"  soil_profile:      {sp.get('num_layers')} layers, "
              f"depth={sp.get('depth_coverage_m')} m, "
              f"source={sp.get('source')}")
    else:
        print(f"  soil_profile:      (none)")
    print()

    # ── Validate brief shape ──────────────────────────────────
    print("=" * 70)
    print("BRIEF SHAPE VALIDATION")
    print("=" * 70)
    issues = validate_brief_shape(brief)
    if issues:
        print(f"⚠️  Found {len(issues)} shape issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        shape_ok = False
    else:
        print("✓ All required keys present and well-formed")
        shape_ok = True
    print()

    # ── Verdict ───────────────────────────────────────────────
    print("=" * 70)
    if shape_ok:
        print("✅ RECEPTION TEST PASSED")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"  - Inspect the brief at {out_path}")
        print(f"  - Chain into planner test:")
        print(f"      python3 tests/test_planner_elm.py {out_path}")
        return True
    else:
        print("❌ RECEPTION TEST FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    request = TEST_USER_REQUEST
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])

    ok = run_test(request)
    sys.exit(0 if ok else 1)