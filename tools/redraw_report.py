#!/usr/bin/env python3
"""
Draw a saved study's report again from its records: step 4 alone, with no
model call and no job.

    source <your environment file>
    python tools/redraw_report.py RUN_DIR

MEANT FOR A COPY OF A STUDY. Copy the study folder first (cp -r) and give
the copy. A study folder is a record of what was run and said, and this tool
replaces its report.

WHAT IT DOES. It rebuilds the report record (analysis.json) from the study's
own records, as step 4 does at the end of a live run: the claim trace and
the second reading of the numbers run again, which read the model output and
need no model. Where the report's figures module is present, it draws the
report's figures and tables too. Then it writes the report's pages and the
deck. The claims, the audit's strikes and the answer are those step 3 saved
(interpretation.json); nothing is worded again.

WHAT IT NEVER DOES. It asks no language model and submits no job. It does
not touch 01_inputs, 03_results, the lookups or step 2's figures. The files
it replaces are first moved into 04_analysis/before_<stamp>/, so the earlier
report stays on the record beside the new one.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# THE FILES STEP 4 WRITES, which a redraw replaces: the record, the four
# pages, the deck, the second reading's record and the report's figures
REPLACED = ("analysis.json", "REPORT.md", "REPORT_DETAILS.md", "SCREEN.md",
            "PLANNED_ANALYSES.md", "analysis.pptx", "evidence_numbers_check.json",
            "report_figures.json")
FIGURES = "figure_over_*.png"


def set_before(adir: Path, stamp: str) -> Path:
    """Move the files a redraw replaces into adir/before_<stamp>/; the
    folder (made only when something moved)."""
    before = adir / f"before_{stamp}"
    names = [n for n in REPLACED if (adir / n).is_file()]
    names += sorted(p.name for p in adir.glob(FIGURES) if p.is_file())
    for name in names:
        before.mkdir(exist_ok=True)
        shutil.move(str(adir / name), str(before / name))
    return before


def put_back(adir: Path, before: Path) -> None:
    """Return what set_before moved, over anything written since."""
    if not before.is_dir():
        return
    for p in before.iterdir():
        shutil.move(str(p), str(adir / p.name))
    before.rmdir()


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def redraw(run_dir, stamp: str = None) -> dict:
    """Step 4 again over a study's records; {"report", "before", "pages"}."""
    from agents.analysis import step0_context, step4_report
    from agents.analysis import report_pages as RP
    run_dir = Path(run_dir).resolve()
    adir = run_dir / "04_analysis"
    interp = _read(adir / "interpretation.json")
    if not isinstance(interp, dict):
        raise SystemExit(f"{adir} holds no interpretation.json: there is no saved step 3 "
                         f"to report on")
    investigation = _read(adir / "investigation.json") or {}
    comparison = _read(adir / "comparison.json") or {}
    prev = _read(adir / "analysis.json") or {}
    ctx = step0_context.load(str(run_dir))
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    before = set_before(adir, stamp)
    try:
        prov = prev.get("provenance") or {}
        report = step4_report.build(
            ctx, comparison, investigation, interp, str(run_dir),
            rounds=list(prov.get("rounds") or []), stopped_because=prov.get("stopped_because"),
            steps=prov.get("steps"), preflight=prov.get("preflight"))
        # THE SPEND OF THE MODEL CALLS IS THE EARLIER PASS'S: this process
        # made none, so its own count would read zero
        if (prev.get("cost") or {}).get("llm"):
            report["cost"]["llm"] = prev["cost"]["llm"]
        report["redrawn"] = {"at": stamp, "by": "tools/redraw_report.py",
                             "earlier": (f"04_analysis/{before.name}/"
                                         if before.is_dir() else None)}
        step4_report.write(report, adir)
    except BaseException:
        # A REDRAW THAT FAILS PUTS THE EARLIER REPORT BACK, so the folder is
        # never left without one (a full disk did this once)
        put_back(adir, before)
        raise
    pages = [n for n in RP.PAGES if (adir / n).is_file()]
    return {"report": report, "before": before if before.is_dir() else None, "pages": pages}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Draw a saved study's report again from its records, with no model call "
                    "and no job. Meant for a copy of a study: the files it replaces are moved "
                    "into 04_analysis/before_<stamp>/.")
    ap.add_argument("run_dir", help="the study folder (a copy of it)")
    a = ap.parse_args(argv)
    out = redraw(a.run_dir)
    rep = out["report"]
    if out["before"]:
        print(f"the earlier report kept in {out['before'].name}/")
    print(f"status {rep.get('status')}; claims kept {len(rep.get('claims') or [])}, "
          f"withheld {len(rep.get('withheld') or [])}"
          + (f"; WITHHELD: {rep['withheld_because']}" if rep.get("withheld_because") else ""))
    print("pages: " + ", ".join(out["pages"]))


if __name__ == "__main__":
    main()
