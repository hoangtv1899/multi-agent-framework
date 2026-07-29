#!/usr/bin/env python3
"""
The gate between planning and compute
src/core/strategy_check.py

The Experiment Manager's first step. It reads reception.json and strategy.json
together — the only place that holds both — and compares them before a single
column is materialised.

It sits here rather than after the planner because this is where cost begins.
Everything upstream is one LLM call; everything downstream is a CIME build per
column and a queue slot. Checking at the hand-off would have caught the same
problems a minute earlier and cost a second validation point; checking here
costs nothing and there is only one.

Two severities, and the difference is deliberate:

  STOP        the run would produce garbage — no domain, no period, a column
              count that cannot be placed. Better to fail loudly than to spend
              an hour producing numbers nobody can use.
  CORRECTION  something unusable is dropped and the run continues, because the
              rest of the design is sound. A strategy naming one station that
              was never fetched should lose that pin, not the whole experiment.

Corrections are RECORDED, not silent. The planner is purely LLM and nothing
upstream edits its output, so this is the only place a discrepancy becomes
visible — and it belongs in the run record, where someone reading the results
can see what was changed and why.
"""
from typing import Any, Dict, List, Tuple

MAX_COLUMNS = 48          # a hard ceiling: beyond this something is wrong
MIN_COLUMNS = 1


def _station_ids(observations: Dict[str, Any]) -> set:
    """Every station id reception actually fetched, across all three sources."""
    ids = set()
    for kind, key in (("streamflow", "stations"), ("swe", "stations"),
                      ("water_table", "wells")):
        for st in ((observations.get(kind) or {}).get(key) or []):
            sid = st.get("id") or st.get("triplet")
            if sid:
                ids.add(str(sid))
    return ids


def check(reception: Dict[str, Any],
          strategy: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compare the two. Returns (report, corrected_strategy).

    The strategy is never mutated in place — a caller that ignores the return
    value gets the original, which is safer than a silent edit.
    """
    brief = (reception or {}).get("brief") or {}
    obs = (reception or {}).get("observations") or {}
    grid = (reception or {}).get("grid") or {}
    dom = brief.get("domain") or {}
    period = (brief.get("run_settings") or {}).get("resolved_period") or {}

    stop: List[str] = []
    corrections: List[str] = []
    strategy = dict(strategy or {})

    # ── STOP conditions ─────────────────────────────────────────────────
    if not dom.get("bbox"):
        stop.append("reception resolved no bbox — there is nowhere to sample")
    if not (isinstance(period.get("yr_start"), int)
            and isinstance(period.get("yr_end"), int)):
        stop.append("reception resolved no simulation period — the run has no "
                    "years to force")

    sampling = dict(strategy.get("sampling") or {})
    n = sampling.get("n_columns")
    if not isinstance(n, int) or n < MIN_COLUMNS:
        stop.append(f"strategy gives no usable column count (n_columns={n!r})")
    elif n > MAX_COLUMNS:
        stop.append(f"strategy asks for {n} columns, above the {MAX_COLUMNS} "
                    f"ceiling — each is a separate model case")
    else:
        # A column is placed AT a grid point, so more columns than points is
        # not a preference the sampler can satisfy. This check only became
        # possible once reception carried the grid.
        avail = grid.get("n_in_basin")
        if isinstance(avail, int) and avail and n > avail:
            stop.append(f"strategy asks for {n} columns but the basin holds "
                        f"only {avail} grid points")

    # ── CORRECTIONS ─────────────────────────────────────────────────────
    known = _station_ids(obs)
    fixed_validation = []
    for v in (strategy.get("validation") or []):
        ids = [str(s) for s in (v.get("stations") or [])]
        keep = [s for s in ids if s in known]
        if len(keep) != len(ids):
            missing = [s for s in ids if s not in known]
            corrections.append(
                f"{v.get('variable')}: dropped {len(missing)} station id(s) "
                f"reception never fetched ({', '.join(missing[:3])})")
            v = dict(v, stations=keep)
            if not keep:
                v["comparison"] = "unavailable — named stations were not fetched"
        fixed_validation.append(v)
    if fixed_validation:
        strategy["validation"] = fixed_validation

    arche = strategy.get("archetype") or (brief.get("design_archetype"))
    if arche and brief.get("design_archetype") and arche != brief["design_archetype"]:
        corrections.append(f"archetype: strategy says {arche!r}, reception "
                           f"said {brief['design_archetype']!r} — using the "
                           f"strategy's")

    report = {
        "ok": not stop,
        "stop": stop,
        "corrections": corrections,
        "checked": {
            "domain": dom.get("name") or dom.get("huc"),
            "period": f"{period.get('yr_start')}-{period.get('yr_end')}",
            "n_columns": sampling.get("n_columns"),
            "grid_points_in_basin": grid.get("n_in_basin"),
            "stations_fetched": len(known),
            "stations_pinned": sum(len(v.get("stations") or [])
                                   for v in (strategy.get("validation") or [])),
            "observations_ok": {k: (obs.get(k) or {}).get("ok")
                                for k in ("streamflow", "water_table", "swe")
                                if k in obs},
        },
    }
    return report, strategy


def render(report: Dict[str, Any]) -> str:
    """One short block for the run log."""
    c = report.get("checked") or {}
    lines = [f"   domain {c.get('domain')}  period {c.get('period')}  "
             f"columns {c.get('n_columns')} of {c.get('grid_points_in_basin')} grid points",
             f"   observations fetched: {c.get('stations_fetched')} station(s), "
             f"{c.get('stations_pinned')} pinned"]
    for s in report.get("stop") or []:
        lines.append(f"   ✗ STOP: {s}")
    for m in report.get("corrections") or []:
        lines.append(f"   ! corrected: {m}")
    if report.get("ok") and not report.get("corrections"):
        lines.append("   ✓ strategy agrees with reception")
    return "\n".join(lines)
