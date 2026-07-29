#!/usr/bin/env python3
"""
Where things live in a run directory
src/core/run_layout.py

One place that knows the layout, so moving a file is a change here rather than
a grep across nine tools. See docs/RUN_LAYOUT.md for the contract.

The run directory's SURFACE is four files, one per box:

    reception.json   strategy.json   experiment.json   analysis.json

Everything else is working state and lives in 01_inputs/ or 04_analysis/.

Two things this module exists to make safe:

  READING OLD RUNS.  Every past run has the old flat layout, and those runs
  are the only record of experiments that cost hours of compute. resolve()
  returns the canonical location if it exists and the legacy one otherwise, so
  a tool reads any run directory without knowing which era produced it.

  MOVING A FILE.  Because every tool asks here instead of hardcoding a path,
  relocating columns.json into 01_inputs/ is one edit, not nine — and the nine
  cannot drift out of agreement with each other.

Deliberately NOT a schema or a loader with opinions. It answers exactly one
question — where is X — and returns a Path. What is inside the file is the
caller's business.
"""
from pathlib import Path
from typing  import Dict, List, Optional, Union

# ── The four. One per box, at the top level, each written by one producer ──
BOUNDARY = ("reception", "strategy", "experiment", "analysis")

# logical name -> (canonical relative path, *legacy fallbacks)
#
# Order matters: the first entry is where a NEW run writes it, the rest are
# only ever read. A legacy name is never written again.
LAYOUT: Dict[str, tuple] = {
    # ── the boundary files ──────────────────────────────────────────────
    "reception":   ("reception.json",),
    "strategy":    ("strategy.json", "plan.json"),
    "experiment":  ("experiment.json", "LLM_ANALYSIS_INPUT.json"),
    "analysis":    ("analysis.json", "ANALYSIS_REPORT.json"),

    # ── manager working state: 01_inputs/, formerly the top level ────────
    "columns":     ("01_inputs/columns.json",     "columns.json"),
    "cases":       ("01_inputs/cases.json",       "cases.json"),
    "run_plan":    ("01_inputs/run_plan.json",    "run_plan.json"),
    "assumptions": ("01_inputs/assumptions.json", "assumptions.json"),

    # ── extraction + analyzer products: 04_analysis/ ─────────────────────
    "extracted":      ("04_analysis/hydro_summary.json",),
    "validation":     ("04_analysis/validation.json",),
    "interpretation": ("04_analysis/interpretation.md",),

    # ── warm start ───────────────────────────────────────────────────────
    "warmstart":   ("warmstart/warmstart.json",),
}

# Removed outright — the content lives in a boundary file. Named here so the
# error message when someone asks for one can say WHERE it went, instead of
# reporting a missing key.
RETIRED: Dict[str, str] = {
    "reception_brief": "reception",   # was a strict subset of reception.json
    "run_summary":     "experiment",  # counts and timings folded in
    "llm_input":       "experiment",
    "analysis_report": "analysis",
    "plan":            "strategy",
}


def resolve(run_dir: Union[str, Path], name: str,
            must_exist: bool = False) -> Optional[Path]:
    """Where is `name` in this run directory?

    Returns the canonical path when it exists, else the first legacy path that
    does, else the canonical path (which may not exist — callers that care
    pass must_exist=True and get None instead).
    """
    rd = Path(run_dir)
    if name in RETIRED:
        raise KeyError(
            f"{name!r} no longer exists — its content is in "
            f"{RETIRED[name]!r} (see docs/RUN_LAYOUT.md)")
    if name not in LAYOUT:
        raise KeyError(f"unknown run-directory entry {name!r}; "
                       f"known: {', '.join(sorted(LAYOUT))}")

    candidates = [rd / p for p in LAYOUT[name]]
    for p in candidates:
        if p.exists():
            return p
    return None if must_exist else candidates[0]


def write_path(run_dir: Union[str, Path], name: str) -> Path:
    """Where a NEW run writes `name` — always canonical, never a legacy name.

    Creates the parent directory, because half the entries live in a subdir
    and forgetting that is the obvious way to get this wrong.
    """
    rd = Path(run_dir)
    if name in RETIRED:
        raise KeyError(
            f"refusing to write {name!r} — it was retired into "
            f"{RETIRED[name]!r} (see docs/RUN_LAYOUT.md)")
    if name not in LAYOUT:
        raise KeyError(f"unknown run-directory entry {name!r}")
    p = rd / LAYOUT[name][0]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def is_legacy(run_dir: Union[str, Path]) -> bool:
    """True when this directory predates the layout change.

    Told apart by the thing that actually changed: a current run has
    experiment.json at the surface; an old one does not.
    """
    rd = Path(run_dir)
    return not (rd / "experiment.json").exists() and (
        (rd / "columns.json").exists() or (rd / "run_plan.json").exists())


def surface(run_dir: Union[str, Path]) -> Dict[str, bool]:
    """The four boundary files and whether each is present.

    A run missing `experiment` never finished its compute; one missing
    `analysis` finished but was never interpreted. The distinction is worth
    being able to make at a glance.
    """
    return {n: (resolve(run_dir, n, must_exist=True) is not None)
            for n in BOUNDARY}


def strays(run_dir: Union[str, Path]) -> List[str]:
    """Top-level files that are neither a boundary file nor expected clutter.

    The guard against the layout quietly regrowing: if a stage starts writing
    a new file at the surface, this is what notices.
    """
    rd = Path(run_dir)
    allowed = {f"{n}.json" for n in BOUNDARY} | {
        "run.log", "exe_path.txt", "submit_cases.sbatch",
        "sampling_design.png",
    }
    return sorted(p.name for p in rd.iterdir()
                  if p.is_file() and p.name not in allowed)
