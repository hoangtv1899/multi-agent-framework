#!/usr/bin/env python3
"""
Re-interpret an archived run: step 3 (the reviewer) and step 4 (the report)
again, over the figures and findings the run already holds.

    source /qfs/people/tran289/IDEAS/env_compy.sh        # PNNL_API_KEY
    python tools/reinterpret_run.py RUN_DIR [--model M] [--no-images] [--dry-run]

WHY THIS EXISTS. The figures (step 2) and the comparisons (step 1) are code
and cost nothing to keep; the wording (step 3) is one model call and is the
part that changes when the prompt or the readability gate does. A run that
was analysed before the gate existed can be re-worded without recomputing a
thing, and the audit still checks every number against the findings on disk.

WHAT IT NEVER DOES. It does not touch 01_inputs, 03_results, the figures or
investigation.json. The previous interpretation.json, analysis.json, REPORT.md
and analysis.pptx are moved into 04_analysis/before_<stamp>/ first, so the old
wording stays on the record beside the new.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KEEP = ("interpretation.json", "analysis.json", "REPORT.md", "analysis.pptx")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run_dir")
    ap.add_argument("--model", default=None,
                    help="LLM for the reviewer (default: step 3's own)")
    ap.add_argument("--no-images", action="store_true",
                    help="send the brief without the figures attached")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the brief the reviewer would see and stop")
    ap.add_argument("--render-only", action="store_true",
                    help="rewrite REPORT.md and the deck from the existing "
                         "analysis.json; no model call, no new interpretation")
    a = ap.parse_args()

    from agents.analysis import step0_context, step3_interpret, step4_report

    run_dir = Path(a.run_dir).resolve()
    adir = run_dir / "04_analysis"
    if a.render_only:
        rep_p = adir / "analysis.json"
        if not rep_p.is_file():
            sys.exit(f"{adir} holds no analysis.json to render")
        report = json.loads(rep_p.read_text())
        (adir / step4_report.REPORT_MD).write_text(
            step4_report.render_markdown(report))
        try:
            from agents.analysis import step4_slides
            step4_slides.build(report, adir)
        except Exception as e:                                  # noqa: BLE001
            print(f"deck not rebuilt: {e}")
        print(f"rendered {adir / step4_report.REPORT_MD} and the deck from "
              f"the existing analysis.json")
        return
    inv_p = adir / "investigation.json"
    if not inv_p.is_file():
        sys.exit(f"{adir} holds no investigation.json; there are no findings "
                 f"to interpret (run the Analyzer first)")
    investigation = json.loads(inv_p.read_text())
    cmp_p = adir / "comparison.json"
    comparison = json.loads(cmp_p.read_text()) if cmp_p.is_file() else {}
    ctx = step0_context.load(str(run_dir))

    if a.dry_run:
        print(step3_interpret.review_brief(ctx, comparison, investigation))
        return

    # THE PREVIOUS WORDING STAYS ON THE RECORD, moved aside by timestamp.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    before = adir / f"before_{stamp}"
    prev_report = None
    for name in KEEP:
        src = adir / name
        if src.is_file():
            before.mkdir(exist_ok=True)
            if name == "analysis.json":
                prev_report = json.loads(src.read_text())
            shutil.move(str(src), str(before / name))
    if before.is_dir():
        print(f"previous wording kept in {before.name}/")

    model = a.model or step3_interpret.DEFAULT_MODEL
    interpretation = step3_interpret.interpret(
        ctx, comparison, investigation, adir, model=model,
        with_images=not a.no_images)

    prov = (prev_report or {}).get("provenance") or {}
    steps = dict(prov.get("steps") or {"context": True,
                                      "compare": bool(comparison),
                                      "investigate": True,
                                      "interpret": True})
    steps["interpret"] = True
    rounds = list(prov.get("rounds") or [])
    rounds.append({"round": f"reinterpret {stamp}",
                   "verdict": interpretation.get("verdict"),
                   "n_claims": interpretation["audit"]["n_claims"],
                   "n_struck": interpretation["audit"]["n_struck"]})
    report = step4_report.build(
        ctx, comparison, investigation, interpretation, str(run_dir),
        rounds=rounds, stopped_because=f"reinterpreted {stamp}",
        steps=steps, preflight=prov.get("preflight"))
    path = step4_report.write(report, adir)

    rd = interpretation.get("readability") or {}
    print(f"\nHEADLINE: {interpretation.get('headline')}")
    print(f"\nANSWER:   {interpretation.get('answer')}")
    print(f"\nclaims kept {interpretation['audit']['n_claims'] - interpretation['audit']['n_struck']}"
          f" of {interpretation['audit']['n_claims']}; verdict {interpretation.get('verdict')}")
    print(f"readability: {len(rd.get('problems_before') or [])} problems before, "
          f"rewritten={rd.get('rewritten')}, "
          f"{len(rd.get('problems_after') or [])} after")
    for p in (rd.get("problems_after") or [])[:6]:
        print("   still:", p)
    print(f"\nwritten: {path} and {adir / step4_report.REPORT_MD}")


if __name__ == "__main__":
    main()
