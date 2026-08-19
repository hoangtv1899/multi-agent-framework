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


APPROACHES = ("elevation_bands", "factor_sweep")
MIN_LEVELS = 2


def _factors(sampling: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [f for f in (sampling.get("factors") or []) if isinstance(f, dict)]


def _sweep_n_columns(sampling: Dict[str, Any]) -> int:
    """The factorial the levels imply. 0 when the design is not expressible.

    Deliberately arithmetic and nothing else. Whether `soil_texture` is a real
    factor, and whether 95% clay can be built, are questions only the model
    server can answer — see the ELM server's check_conceptual_design. This gate
    runs without a server and must not pretend otherwise.
    """
    fs = _factors(sampling)
    if not fs:
        return 0
    total = 1
    for f in fs:
        total *= len(f.get("levels") or ())
    return total


def _sweep_years(sampling: Dict[str, Any]) -> List[Any]:
    """Years the design carries itself — held fixed, or varied as a factor."""
    held = (sampling.get("held_fixed") or {}).get("years")
    if held:
        return list(held) if isinstance(held, (list, tuple)) else [held]
    for f in _factors(sampling):
        if f.get("name") == "forcing_year" and f.get("levels"):
            return list(f["levels"])
    return []


def _sweep_stops(sampling: Dict[str, Any]) -> List[str]:
    """What makes a sweep unrunnable, judged without asking a model server."""
    out: List[str] = []
    approach = str(sampling.get("approach") or "").strip().lower()
    if approach and approach not in APPROACHES:
        out.append(f"sampling.approach is {approach!r}, which is not one of "
                   f"{list(APPROACHES)} — the manager would not know which "
                   f"path to take")

    fs = _factors(sampling)
    if not fs:
        out.append("a factor sweep with no factors varies nothing — there is "
                   "no experiment here, only repeated runs")
        return out

    for f in fs:
        name = f.get("name") or "(unnamed)"
        levels = list(f.get("levels") or ())
        if len(levels) < MIN_LEVELS:
            out.append(f"factor {name!r} has {len(levels)} level(s) — a sweep "
                       f"needs at least {MIN_LEVELS}, or it is one run with a "
                       f"comparison implied and never made")
        if len(set(map(repr, levels))) != len(levels):
            out.append(f"factor {name!r} repeats a level — two identical "
                       f"columns measure the model's determinism, not the "
                       f"factor")

    # NOTHING ABOUT COORDINATES HERE (moved 2026-08-18). This gate used to stop
    # a sweep whose held_fixed named no lat/lon, explaining that "ELM still
    # reads its weather from a grid cell". That is a fact about ELM, judged
    # by the framework, and it refused a PFLOTRAN sweep that has no place by
    # design. Whether a sweep needs a point is the model server's to say:
    # the ELM server's check_conceptual_design refuses a design with no
    # coordinates (with the same two-way wording), and the PFLOTRAN server's
    # does not ask for any. The manager calls that check before compute.
    return out


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

    sampling = dict(strategy.get("sampling") or {})

    # WHICH KIND OF STUDY, decided before any check runs.
    #
    # Half the conditions below are about a BASIN — a bbox to sample, a grid
    # to place columns in. A controlled sweep has neither and is not broken for
    # lacking them. Before this branch existed the first STOP fired on every
    # conceptual run ("there is nowhere to sample"), so nothing conceptual
    # could reach compute at all.
    #
    # Two sources agreeing is not required: the archetype is reception's word
    # and the approach is the planner's, and either alone is enough to know a
    # basin is not expected. They are cross-checked further down.
    arche = (strategy.get("archetype") or brief.get("design_archetype") or "")
    approach = str(sampling.get("approach") or "").strip().lower()
    is_sweep = (str(arche).strip().lower() == "conceptual"
                or approach == "factor_sweep")

    # ── STOP conditions ─────────────────────────────────────────────────
    if not is_sweep and not dom.get("bbox"):
        stop.append("reception resolved no bbox — there is nowhere to sample")

    # A SWEEP STILL NEEDS YEARS, from one of two places: reception resolved a
    # period, or the design varies the year itself. Requiring the first alone
    # would refuse the very study that makes the year a factor.
    has_years = (isinstance(period.get("yr_start"), int)
                 and isinstance(period.get("yr_end"), int))
    if not has_years and is_sweep:
        has_years = bool(_sweep_years(sampling))
    if not has_years:
        stop.append("reception resolved no simulation period — the run has no "
                    "years to force")

    n = sampling.get("n_columns")
    if not isinstance(n, int) or n < MIN_COLUMNS:
        stop.append(f"strategy gives no usable column count (n_columns={n!r})")
    elif n > MAX_COLUMNS:
        stop.append(f"strategy asks for {n} columns, above the {MAX_COLUMNS} "
                    f"ceiling — each is a separate model case")
    elif not is_sweep:
        # A column is placed AT a grid point, so more columns than points is
        # not a preference the sampler can satisfy. This check only became
        # possible once reception carried the grid. A sweep places nothing on
        # a grid — its columns repeat one location on purpose.
        avail = grid.get("n_in_basin")
        if isinstance(avail, int) and avail and n > avail:
            stop.append(f"strategy asks for {n} columns but the basin holds "
                        f"only {avail} grid points")

    if is_sweep:
        stop.extend(_sweep_stops(sampling))

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

    # A SWEEP VALIDATES AGAINST NOTHING, and says so by carrying no entries.
    # The loop above empties the station list of anything reception did not
    # fetch — which on a sweep is everything, since nothing was fetched — and
    # what survives is an entry reading "streamflow: unavailable". That is a
    # true sentence about a site study whose gauges were missing and a false
    # one here: no gauge was sought, because there is no basin. Left in, it
    # becomes a validation the report has to explain away.
    if is_sweep and fixed_validation:
        dropped = sorted({str(v.get("variable")) for v in fixed_validation})
        corrections.append(
            f"validation: dropped {len(fixed_validation)} entr(y/ies) "
            f"({', '.join(dropped)}) — a controlled sweep has no basin and "
            f"fetched no observations, so there was never anything to compare "
            f"against. Absence here is the design, not a failed lookup.")
        fixed_validation = []
        strategy["validation"] = []

    if fixed_validation:
        strategy["validation"] = fixed_validation

    # A STATED COLUMN COUNT IS A CONSTRAINT, not a hint.
    #
    # The checks above ask whether n_columns is PHYSICALLY possible — under
    # MAX_COLUMNS, and no more columns than the basin has grid points. None of
    # them asks whether it is what the user requested, so on 2026-08-06 a
    # request for 3 columns produced a 19-column study and this gate reported
    # "strategy agrees with reception". Six times the compute, approved.
    #
    # Corrected rather than STOPped, matching how the period is handled: the
    # user's number is authoritative, the run continues, and the correction is
    # recorded in the run so the change is visible afterwards.
    want = (brief.get("run_settings") or {}).get("requested_n_columns")
    if is_sweep:
        # THE USER'S NUMBER CANNOT WIN HERE, and this is the one place the two
        # archetypes need opposite treatment. A site design can honour "give me
        # 3 columns" by sampling three points. A sweep's count is the product
        # of its levels — 7 clay values IS 7 columns — so overwriting it would
        # leave a design whose stated size disagrees with the experiment it
        # describes, and the builder would produce a different number again.
        # Correct the count TO the design instead of the design to the count.
        factorial = _sweep_n_columns(sampling)
        if factorial and isinstance(n, int) and n != factorial:
            corrections.append(
                f"n_columns: the strategy says {n}, but the factor levels give "
                f"{factorial} — using {factorial}, which is what the design is")
            sampling["n_columns"] = factorial
            strategy["sampling"] = sampling
            n = factorial
        if isinstance(want, int) and want > 0 and factorial and want != factorial:
            corrections.append(
                f"n_columns: the request asked for {want}, but a sweep's size "
                f"is its levels ({factorial}) — the request cannot be honoured "
                f"without changing the factors, so it was not applied")
    elif isinstance(want, int) and want > 0 and isinstance(n, int) and want != n:
        corrections.append(
            f"n_columns: the request asked for {want}, the strategy designed "
            f"{n} — using the {want} that was asked for")
        sampling["n_columns"] = want
        strategy["sampling"] = sampling
        n = want

    if arche and brief.get("design_archetype") and arche != brief["design_archetype"]:
        corrections.append(f"archetype: strategy says {arche!r}, reception "
                           f"said {brief['design_archetype']!r} — using the "
                           f"strategy's")

    report = {
        "ok": not stop,
        "stop": stop,
        "corrections": corrections,
        "checked": {
            "approach": approach or ("factor_sweep" if is_sweep
                                     else "elevation_bands"),
            "domain": dom.get("name") or dom.get("huc"),
            "period": f"{period.get('yr_start')}-{period.get('yr_end')}",
            "n_columns": sampling.get("n_columns"),
            # None rather than 0 on a sweep: there is no basin to have points
            # in, and a zero would read as an empty one.
            "grid_points_in_basin": None if is_sweep else grid.get("n_in_basin"),
            "factors": [f.get("name") for f in _factors(sampling)] or None,
            # WHOSE WEATHER, in one word, on the line the user actually reads.
            # "no basin and no observations" already tells them what a sweep
            # lacks; without this the one thing that decides whether the result
            # is bounded by a real cell's climate is visible only inside
            # columns.json.
            "weather": (
                "written" if (
                    (sampling.get("held_fixed") or {}).get("weather") is not None
                    or any(f.get("name") == "prescribed_weather"
                           for f in _factors(sampling)))
                else "borrowed from the forcing cell") if is_sweep else None,
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
    if c.get("approach") == "factor_sweep":
        # A SWEEP HAS NO BASIN AND NO STATIONS, so the site line would report
        # None domain, None grid points and 0 pinned — three absences that read
        # as failures rather than as a different kind of study.
        lines = [f"   controlled sweep over {', '.join(c.get('factors') or [])}"
                 f"  period {c.get('period')}  columns {c.get('n_columns')}",
                 f"   no basin and no observations — by design, not by "
                 f"omission",
                 f"   weather: {c.get('weather')}"]
    else:
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
