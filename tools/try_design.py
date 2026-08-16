#!/usr/bin/env python3
"""
Drive the front half of the pipeline and stop where you say
tools/try_design.py

    python3 tools/try_design.py --through reception   "<question>"
    python3 tools/try_design.py --through plan        "<question>"
    python3 tools/try_design.py --through materialize "<question>"

WHY THIS EXISTS. workflow.py has no dry-run flag: a design request typed there
runs reception, the planner, the strategy gate, materialize, the case build AND
sbatch, in one go. That is the wrong way to find out whether a prompt asks good
questions or whether a sweep produces the columns you meant.

This stops at a stage you name. Nothing here submits a job, ever.

    reception     the conversation only. No files written anywhere.
    plan          + the planner and the strategy gate, into a run directory.
    materialize   + the Experiment Manager's FIRST STAGE — columns.json and,
                  for ELM, the warm start, the donor soil, and the generated
                  surfaces. Real work and several minutes; still no sbatch.

WHAT `materialize` ACTUALLY DOES, so the cost is not a surprise: for ELM it
calls build_elm_inputs_from_location, which subsets the CONUS restart for every
column and generates a surface dataset for each. That reads large files and
writes into the run directory. It does not compile a case and does not queue
anything — the stages that do are execute_plan's, and this tool never calls it.
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from core.mcp_manager import MCPManager                        # noqa: E402
from agents.reception_llm import LLMReceptionAgent             # noqa: E402

DEFAULT_MODEL = "claude-opus-5-project"
STAGES = ("reception", "plan", "materialize")


def _summarise_sweep(brief: dict) -> dict:
    sweep = brief.get("sweep") or {}
    if not sweep:
        return {}
    print("\n--- the sweep reception settled ---")
    print(f"model      : {sweep.get('model')}")
    print(f"  because  : {sweep.get('model_rationale')}")
    print(f"  runnable : {sweep.get('model_availability')}")
    for f in (sweep.get("factors") or []):
        print(f"factor     : {f.get('name')} = {f.get('levels')}"
              f"   [{f.get('settled_by')}]")
    print(f"held fixed : {sweep.get('held_fixed')}")
    print(f"  settled  : {sweep.get('held_fixed_settled_by')}")
    print(f"n_columns  : {sweep.get('n_columns')}")
    print(f"weather    : {sweep.get('forcing_site_rationale')}")
    for u in (sweep.get("unresolved") or []):
        print(f"unresolved : {u}")
    return sweep


def _check_design(clients, sweep: dict) -> None:
    """Ask the model server whether the settled design is buildable.

    Costs nothing and catches the most: reception is PROMPTED to offer only
    what the server declared, and this is where that turns from an instruction
    into a fact.
    """
    if not (clients.get("elm") and sweep.get("factors")):
        return
    design = {"factors": [{"name": f.get("name"), "levels": f.get("levels")}
                          for f in sweep["factors"]],
              "held_fixed": sweep.get("held_fixed") or {}}
    try:
        v = clients["elm"].call_tool_json("check_conceptual_design",
                                          {"design": design}) or {}
    except Exception as e:                                      # noqa: BLE001
        print(f"\n⚠️  could not check the design ({type(e).__name__}: {e})")
        return
    print("\n--- the elm server on that design ---")
    print(f"buildable  : {v.get('buildable')}  ({v.get('n_columns')} columns)")
    for w in (v.get("wont_build") or []):
        print(f"  REFUSE   : {w.get('factor')}: {w.get('why')}")
    for u in (v.get("unusual") or []):
        print(f"  FLAG     : {u.get('factor')}: {u.get('why')}")


def _materialize(run_dir: Path, plan: dict, pkg: dict, clients: dict,
                 backend: str, out: str) -> dict:
    """The Experiment Manager's FIRST STAGE, and only that.

    Calls _materialize directly rather than execute_plan. execute_plan would
    carry on into the case build and the ensemble submission, which is exactly
    what this tool exists not to do.
    """
    from core import backends
    brief = pkg.get("brief") or {}
    Manager = backends.get(backend)
    mgr = Manager(base_output_dir=out, run_dir=str(run_dir))
    cfg = backends.config_for(
        backend,
        {"brief": brief, "reception": pkg, "strategy": plan,
         "mcp_clients": clients, "last_run_dir": None},
        period=(brief.get("run_settings") or {}).get("resolved_period"),
        initialization=(brief.get("run_settings") or {}).get("initialization"))

    print("\n⚙️  Materialize — the manager's FIRST STAGE only")
    print("-" * 50)
    merged = mgr._materialize(plan, cfg)

    cj = run_dir / "columns.json"
    if cj.exists():
        data = json.loads(cj.read_text())
        cols = data.get("columns") or []
        print(f"\n✓ columns.json — {len(cols)} column(s), "
              f"approach={data.get('approach', 'elevation_bands')}")
        for c in cols[:3] + (cols[-1:] if len(cols) > 3 else []):
            bits = [c.get("id"), f"({c.get('lat')}, {c.get('lon')})"]
            if c.get("treatment"):
                bits.append(f"treatment={c['treatment']}")
            if c.get("soil_profile"):
                bits.append(
                    f"clay={c['soil_profile']['layers'][0].get('clay_pct')}%")
            if c.get("soil_source"):
                bits.append(f"soil={c['soil_source']}")
            print("   " + "  ".join(str(b) for b in bits))
        if len(cols) > 4:
            print(f"   … {len(cols) - 4} more")

    print(f"\n✓ run plan: "
          f"{len((merged or {}).get('CONDITIONS_COUPLERS') or [])} coupler(s)")
    for name in ("case_inputs.json", "elm_columns.json"):
        p = run_dir / "01_inputs" / name
        if p.exists():
            print(f"   01_inputs/{name}  ({p.stat().st_size:,} bytes)")

    print(f"\nStopped after MATERIALIZE — step 1 of the Experiment Manager.")
    print(f"NOT run: the case build (CIME compile) and the ensemble (sbatch).")
    print(f"📁 {run_dir}")
    return merged


def _from_strategy(a, clients: dict) -> int:
    """Materialize a hand-written design. No reception, no planner, no cost.

    THE FIXTURE PATH. Testing the column build should not require paying for a
    conversation first, and a design typed by hand is also the only way to
    drive a case the conversation would not produce — a deliberately awkward
    sweep, or one being bisected after a failure.
    """
    from datetime import datetime
    plan = json.loads(Path(a.strategy).read_text())
    if a.reception:
        pkg = json.loads(Path(a.reception).read_text())
    else:
        # A MINIMAL RECEPTION, not a fake one. strategy_check reads the period
        # and the archetype from here; inventing a domain or observations would
        # make a conceptual design look like a site one to the gate.
        samp = plan.get("sampling") or {}
        yrs = (samp.get("held_fixed") or {}).get("years") or [1995]
        # THE QUESTION, from the strategy's own goals. step0_context reads it
        # as reception.user_request, so a fixture run without it produced an
        # analysis whose `question` was null — and step 3 then judged whether
        # the evidence answered a question it had never been told.
        goals = plan.get("goals") or []
        pkg = {"brief": {"design_archetype": plan.get("archetype")
                         or "conceptual",
                         "domain": None,
                         "run_settings": {"resolved_period": {
                             "yr_start": int(yrs[0]),
                             "yr_end": int(yrs[-1])}}},
               "user_request": ("; ".join(str(g) for g in goals)
                                or "(fixture run — no conversation; see "
                                   "strategy.goals)"),
               "observations": {}, "grid": {}}
        print("   (no --reception given — synthesised a minimal one from the "
              "strategy's own years)")

    stamp = datetime.now().strftime("%Y%m%d%H%M")
    run_dir = Path(a.out) / f"{a.backend}_fixture_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reception.json").write_text(json.dumps(pkg, indent=2, default=str))
    (run_dir / "reception_brief.json").write_text(
        json.dumps(pkg.get("brief") or {}, indent=2, default=str))
    (run_dir / "strategy.json").write_text(json.dumps(plan, indent=2, default=str))
    print(f"📁 {run_dir}")
    samp = plan.get("sampling") or {}
    print(f"   approach={samp.get('approach')}  "
          f"n_columns={samp.get('n_columns')}  "
          f"archetype={plan.get('archetype')}")

    if a.through != "materialize":
        print(f"\n--strategy skips reception and the planner, so "
              f"--through {a.through} has nothing left to do. "
              f"strategy.json is written.")
        return 0
    _materialize(run_dir, plan, pkg, clients, a.backend, a.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?",
                    default="How does soil texture split rain between surface "
                            "runoff and drainage?")
    ap.add_argument("--through", choices=STAGES, default="materialize",
                    help="where to stop (default: materialize — the Experiment "
                         "Manager's first stage). Nothing beyond it ever runs.")
    ap.add_argument("--strategy", metavar="FILE",
                    help="skip reception AND the planner: take a hand-written "
                         "strategy.json and go straight to materialize. No "
                         "model call, so this is the cheap way to test the "
                         "column build. Needs --reception too, or a minimal "
                         "one is synthesised.")
    ap.add_argument("--reception", metavar="FILE",
                    help="a reception.json to pair with --strategy")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="elm")
    ap.add_argument("--mcp-config", default="mcp_config.json")
    ap.add_argument("--out", default="workflow_outputs",
                    help="where the run directory is made (plan/materialize)")
    ap.add_argument("--no-ask", action="store_true",
                    help="batch mode — reception asks nothing, so you see what "
                         "it defaults")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print reception's system prompt and exit. No model "
                         "call, no cost — the fastest way to read the menu.")
    a = ap.parse_args()

    clients = {}
    try:
        clients = MCPManager(a.mcp_config).get_all_clients() or {}
    except Exception as e:                                      # noqa: BLE001
        print(f"⚠️  no MCP clients ({type(e).__name__}: {e})")
        print("   the sweep menu will report itself unavailable, which is what "
              "reception is told to say rather than inventing one.\n")

    agent = LLMReceptionAgent(model=a.model, mcp_clients=clients,
                              interactive=not a.no_ask)
    if a.show_prompt:
        print(agent.system)
        return 0

    print("=" * 70)
    print(f"REQUEST : {a.request}")
    print(f"THROUGH : {a.through}   (nothing beyond this is run)")
    print("=" * 70 + "\n")

    # ── a hand-written design skips the two model stages entirely ───────
    if a.strategy:
        return _from_strategy(a, clients)

    # ── reception ───────────────────────────────────────────────────────
    pkg = agent.process(a.request)
    brief = pkg.get("brief") or {}
    route = pkg.get("route") or {}
    print(f"\nroute      : {route.get('action')}")
    print(f"archetype  : {brief.get('design_archetype')}")
    for q in (route.get("questions") or []):
        print(f"  ? {q}")
    sweep = _summarise_sweep(brief)
    _check_design(clients, sweep)

    if a.through == "reception":
        print("\nStopped at reception. Nothing was written, nothing submitted.")
        return 0
    if route.get("action") != "design":
        print(f"\nReception routed to {route.get('action')!r}, not 'design' — "
              f"there is no plan to make. Stopping.")
        return 0

    # ── the run directory, written exactly as the coordinator writes it ──
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    run_dir = Path(a.out) / f"{a.backend}_try_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reception.json").write_text(json.dumps(
        {k: v for k, v in pkg.items() if k not in ("trace", "raw")},
        indent=2, default=str))
    (run_dir / "reception_brief.json").write_text(
        json.dumps(brief, indent=2, default=str))
    print(f"\n📁 {run_dir}")

    # ── planner ─────────────────────────────────────────────────────────
    from agents.planner import Planner
    print("\n📋 Planning")
    print("-" * 50)
    plan = Planner(model=a.model).plan(pkg)
    (run_dir / "strategy.json").write_text(json.dumps(plan, indent=2, default=str))
    samp = plan.get("sampling") or {}
    print(f"✓ approach={samp.get('approach')}  n_columns={samp.get('n_columns')}"
          f"  archetype={plan.get('archetype')}")
    if samp.get("factors"):
        for f in samp["factors"]:
            print(f"   factor: {f.get('name')} = {f.get('levels')}")
    if samp.get("held_fixed"):
        print(f"   held fixed: {samp['held_fixed']}")

    if a.through == "plan":
        print("\nStopped after planning. strategy.json is written; nothing "
              "materialised, nothing submitted.")
        return 0

    # ── materialize — the manager's first stage, and only that ──────────
    _materialize(run_dir, plan, pkg, clients, a.backend, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
