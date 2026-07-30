#!/usr/bin/env python3
"""
Analyzer step 1 — the entry point
src/agents/analysis/step1_compare.py

    in   ctx
    out  04_analysis/comparison.json + the five step-1 figures

The three observables each have their own module because each has its own
station matching, its own exclusions and its own caveats. This is the one place
that runs all three, draws the figures, and WRITES THE RESULT DOWN.

Persisting matters for the same reason step 0 is the only step that opens a
file: it makes every later step runnable against an archived study. Until now
the comparison records lived only in memory, so step 3 could not be developed
or re-run without recomputing step 1 — and the caveats step 1 raises, which are
the ones that bind everything downstream, existed nowhere on disk.
"""
import json
from pathlib import Path
from typing import Any, Dict

from agents.analysis import step1_compare_swe as _swe            # noqa: E402
from agents.analysis import step1_compare_streamflow as _flow    # noqa: E402
from agents.analysis import step1_compare_wtd as _wtd            # noqa: E402
from agents.analysis import step1_maps as _maps                  # noqa: E402

FILENAME = "comparison.json"


def compare_all(ctx, out_dir, draw: bool = True) -> Dict[str, Any]:
    """Run the three comparisons, draw the figures, write comparison.json.

    `draw` exists so a caller that only needs the records — step 3 re-reading an
    archived run, a test — does not pay for five renders it will not look at.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    swe = _swe.compare(ctx)
    flow = _flow.compare(ctx)
    wtd = _wtd.compare(ctx)

    figures: Dict[str, str] = {}
    if draw:
        figures = {
            "swe_scatter": _swe.plot_scatter(swe, out_dir / "swe_scatter.png"),
            "swe_timeseries": _swe.plot_timeseries(
                swe, out_dir / "swe_timeseries.png"),
            "streamflow_timeseries": _flow.plot_timeseries(
                flow, out_dir / "streamflow_timeseries.png"),
            "wtd_timeseries": _wtd.plot_timeseries(
                wtd, out_dir / "wtd_timeseries.png"),
            "comparison_spatial_map": _maps.create_comparison_spatial_map(
                ctx, out_dir / "comparison_spatial_map.png",
                swe=swe, streamflow=flow, wtd=wtd),
        }

    # Caveats are hoisted to the top level rather than left inside each
    # observable's record. They are what BINDS later steps, and a step 3 that
    # had to know to look in three different places for them would eventually
    # look in two.
    caveats = []
    for name, rec in (("swe", swe), ("streamflow", flow), ("wtd", wtd)):
        for c in (rec.get("caveats") or []):
            caveats.append(dict(c, observable=name))

    out = {"swe": swe, "streamflow": flow, "wtd": wtd,
           "caveats": caveats, "figures": figures}
    (out_dir / FILENAME).write_text(json.dumps(out, indent=2, default=str))
    return out


def load(out_dir) -> Dict[str, Any]:
    """Read back a previous comparison, or {} if step 1 has not run."""
    p = Path(out_dir) / FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
