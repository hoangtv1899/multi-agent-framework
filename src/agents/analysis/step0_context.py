#!/usr/bin/env python3
"""
Analyzer step 0 — load the run, split it three ways
src/agents/analysis/step0_context.py

    in   reception.json, strategy.json, experiment.json
    out  AnalysisContext: {plan, data, caveats} + series() + planned_vs_actual()

The ONLY step that opens a file. Everything downstream takes the context and
returns data, which is what lets any later step run alone against an archived
run.

Reads the three boundary files — one per upstream box — and returns a context
split into PLAN, DATA and CAVEATS.

    reception.json   what was asked: domain, period, and the observations
                     fetched once the period was fixed
    strategy.json    what was chosen: archetype, sampling design, validation
                     targets, feasibility verdict
    experiment.json  what happened: per-column metrics, daily series, the
                     honesty payload

Why split rather than hand over one merged dict: the three answer different
questions and must not be interchangeable.

    plan     the question. Read for context; never a source of results.
    data     the evidence. The ONLY place a number in the interpretation may
             come from.
    caveats  constraints on inference. Neither intent nor evidence — they
             bound what the evidence is allowed to support.

That last bucket is the one that keeps getting lost. The limitations and the
assumptions ledger reached the written report but not the results package, so
an Analyzer reading only the package stated conclusions with none of the
caveats attached. Here they are first-class and, unlike plan and data, they
are UNIFORM records — which is what makes them checkable. A downstream step
can ask "did the interpretation respect caveat warm_start_year_one?" and
verify it. Prose cannot be checked, which is how an unrespected caveat looks
exactly like a respected one.

The three do not share a shape, deliberately. `data` is heterogeneous by
nature: per-column rows with daily series alongside ensemble aggregates that
are not per-column at all, and flattening those together would destroy the
distinction between "this column did X" and "the ensemble shows Y" — which is
the distinction the figures rest on. What they share is conventions: every
number carries units, every block records where it came from.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from core import run_layout as _layout          # noqa: E402

# Severity is about what a caveat DOES to a claim, not how bad it sounds.
BLOCKING = "blocking"       # a claim of this kind must not be made at all
QUALIFY  = "qualify"        # the claim may be made, stated with this attached
CONTEXT  = "context"        # worth knowing; constrains nothing by itself


def _read(path: Optional[Path], default=None):
    if not path or not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return default if default is not None else {}


class AnalysisContext:
    """plan / data / caveats for one finished run, plus the tidy series."""

    def __init__(self, run_dir: str,
                 plan: Dict[str, Any],
                 data: Dict[str, Any],
                 caveats: List[Dict[str, Any]],
                 sources: Dict[str, Optional[str]]):
        self.run_dir = Path(run_dir)
        self.plan    = plan
        self.data    = data
        self.caveats = caveats
        self.sources = sources
        # Filled by series(): the variables it refused to put in the daily
        # frame, and why. Read by step 2's catalog, so the figure planner is
        # told what is absent instead of discovering it as an empty groupby.
        self.series_withheld: Dict[str, Dict[str, Any]] = {}

    # ── the rule the split exists to enforce ────────────────────────────
    @property
    def columns(self) -> List[Dict[str, Any]]:
        return self.data.get("columns") or []

    def blocking(self) -> List[Dict[str, Any]]:
        """Caveats that forbid a class of claim outright."""
        return [c for c in self.caveats if c.get("severity") == BLOCKING]

    def series(self):
        """Model daily series as a TIDY long frame, or None without pandas.

            date | entity | variable | value | units | source

        Long rather than wide because observations join onto it by the same
        keys: a gauge is another `entity`, its discharge another `variable`,
        and `source` tells them apart. Comparison then becomes a groupby, and
        a missing observation stays explicitly missing instead of vanishing —
        which is how "no gauge had records that year" once became a finding
        about hydrology rather than about a failed query.

        ONE VALUE PER DATE, OR THE VARIABLE STAYS OUT (2026-08-13). SOILLIQ and
        H2OSOI are per-layer: 5,280 values against 352 dates on a 15-layer
        Brandywine column, flattened time-by-layer into the same `values` list
        every scalar uses. `zip(dates, vals)` does not fail on that — it stops
        at the shorter one, so the frame got the first 352 numbers, which are
        the first 23 days of the SOIL PROFILE, relabelled with a year of dates.
        Nothing looked wrong: the column existed, the dates were real, the
        units were right. It surfaced only when a reviewer looked at the figure
        and said a total soil-water trace cannot collapse to zero and rebound
        every fortnight — the fortnight being the 15 layers, cycling.

        A frame that silently mislabels is worse than one that is missing the
        variable, so the mismatch is DETECTED AND REPORTED here, and
        `series_withheld` says which variables and why. Reshaping them into a
        depth frame is the right answer and is not this function's to invent:
        a column mean needs layer THICKNESSES, and the packaged block does not
        carry them.
        """
        try:
            import pandas as pd
        except ImportError:
            return None
        recs = []
        withheld: Dict[str, Dict[str, Any]] = {}
        for row in self.columns:
            cid = row.get("case_name")
            for var, blk in (row.get("variables") or {}).items():
                daily = (blk or {}).get("daily") or {}
                vals  = daily.get("values") or []
                dates = daily.get("dates") or list(range(len(vals)))
                units = daily.get("units")
                if len(vals) != len(dates):
                    n_layers = (blk or {}).get("n_layers")
                    if not n_layers and dates and len(vals) % len(dates) == 0:
                        n_layers = len(vals) // len(dates)
                    withheld[var] = {
                        "n_values": len(vals), "n_dates": len(dates),
                        "n_layers": n_layers,
                        "why": (f"{len(vals)} values against {len(dates)} dates"
                                + (f" — one per day per soil layer ({n_layers} "
                                   f"layers), flattened" if n_layers else "")
                                + ". A daily frame holds one value per date; "
                                  "pairing these would relabel the soil "
                                  "profile as a time series."),
                    }
                    continue
                if vals and isinstance(vals[0], list):
                    # Properly layered — one row per day, one entry per layer.
                    # Not a defect and not withheld: it belongs in soil(),
                    # which keeps the depth axis instead of losing it.
                    withheld[var] = {
                        "n_layers": len(vals[0]), "n_dates": len(dates),
                        "why": (f"depth-resolved — {len(dates)} days x "
                                f"{len(vals[0])} soil layers. It is in `soil`, "
                                f"with each layer's thickness and depth."),
                        "in_frame": "soil",
                    }
                    continue
                for d, v in zip(dates, vals):
                    recs.append((d, cid, var, v, units, "model"))
        # Said once, not once per call. series() is rebuilt by every step that
        # wants the frame, and three identical warnings read as three problems.
        if withheld and withheld != self.series_withheld:
            broken = [k for k, v in withheld.items() if not v.get("in_frame")]
            if broken:
                print(f"   ⚠️  malformed, not in any frame: "
                      f"{', '.join(sorted(broken))}")
            moved = [k for k, v in withheld.items() if v.get("in_frame")]
            if moved:
                print(f"   ↳ depth-resolved, in `soil`: "
                      f"{', '.join(sorted(moved))}")
        self.series_withheld = withheld
        if not recs:
            return None
        df = pd.DataFrame(recs, columns=["date", "entity", "variable",
                                         "value", "units", "source"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def soil(self):
        """The soil column through time, as a TIDY long frame, or None.

            entity | date | layer | depth_m | thickness_m
                   | depth_top_m | depth_bottom_m
                   | variable | value | units | source

        THE THIRD FRAME, and it exists because ELM's layered output has an axis
        combination neither of the others has: one value per DAY per LAYER.
        series() is one value per date and profiles() is one value per (output
        time, depth) — a handful of yearly snapshots. Forcing SOILLIQ into
        either would have meant dropping an axis, and dropping the layer axis
        is exactly the bug this frame was built after.

        EVERY LAYER CARRIES ITS OWN GEOMETRY, which is the point. ELM's fifteen
        layers run 1.75 cm at the surface to 13.85 m at the bottom, so the
        column mean of a moisture field is a THICKNESS-WEIGHTED mean —

            (value * thickness_m).sum() / thickness_m.sum()

        — and a plain mean is dominated by the deepest layer, which is mostly
        bedrock. `depth_top_m` and `depth_bottom_m` are the running sum, so a
        probe at 10 cm is the layer whose top and bottom straddle it, and the
        hydrologically active soil is `depth_bottom_m <= 3.8021` — the first
        ten layers. That number is now measured from the file rather than
        asserted, which is what makes it checkable.
        """
        try:
            import pandas as pd
        except ImportError:
            return None
        recs = []
        for row in self.columns:
            cid = row.get("case_name")
            for var, blk in (row.get("variables") or {}).items():
                daily = (blk or {}).get("daily") or {}
                vals = daily.get("values") or []
                if not (vals and isinstance(vals[0], list)):
                    continue                    # not layered; series() has it
                dates = daily.get("dates") or list(range(len(vals)))
                units = daily.get("units")
                thick = daily.get("layer_thickness_m") or []
                depth = daily.get("layer_depth_m") or []
                # Running sum, so a layer knows where it starts and ends. Built
                # once per variable rather than per row: 366 x 15 rows per
                # column, and re-deriving it inside the loop is 5,490 identical
                # accumulations.
                tops, bottoms, run = [], [], 0.0
                for t in thick:
                    tops.append(round(run, 6))
                    run += float(t or 0.0)
                    bottoms.append(round(run, 6))
                for d, layers in zip(dates, vals):
                    for i, v in enumerate(layers):
                        recs.append((
                            cid, d, i + 1,
                            depth[i] if i < len(depth) else None,
                            thick[i] if i < len(thick) else None,
                            tops[i] if i < len(tops) else None,
                            bottoms[i] if i < len(bottoms) else None,
                            var, v, units, "model"))
        if not recs:
            return None
        df = pd.DataFrame(recs, columns=[
            "entity", "date", "layer", "depth_m", "thickness_m",
            "depth_top_m", "depth_bottom_m", "variable", "value", "units",
            "source"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def profiles(self):
        """Depth profiles as a TIDY long frame, or None when the run has none.

            entity | time_y | depth_m | variable | value | units | source

        A SECOND frame rather than more columns on series(), because the two
        have different axes and nothing joins them: series() is one value per
        date, profiles() is one value per (output time, depth). Forcing
        PFLOTRAN's five yearly snapshots into series() would have meant either
        inventing dates for them or leaving `date` null on most of the frame —
        and a null-date row silently drops out of every resample and groupby
        the existing analysis code already does.

        Only PFLOTRAN writes `profiles` today. ELM runs return None here, so a
        caller must handle both being absent — which is the honest state for a
        run whose extraction produced neither.
        """
        try:
            import pandas as pd
        except ImportError:
            return None
        recs = []
        for row in self.columns:
            cid  = row.get("case_name")
            prof = row.get("profiles") or {}
            times = prof.get("times_y") or []
            depth = prof.get("depth_m") or []
            for var, grids in prof.items():
                if var in ("times_y", "depth_m") or not isinstance(grids, list):
                    continue
                units = (self.data.get("variable_units") or {}).get(
                    var.upper(), (self.data.get("variable_units") or {}).get(var))
                for t, layer in zip(times, grids):
                    if not isinstance(layer, list):
                        continue
                    for z, v in zip(depth, layer):
                        recs.append((cid, t, z, var, v, units, "model"))
        if not recs:
            return None
        return pd.DataFrame(recs, columns=["entity", "time_y", "depth_m",
                                           "variable", "value", "units",
                                           "source"])

    def planned_vs_actual(self) -> List[Dict[str, Any]]:
        """Did the run do what was asked? One record per checked claim.

        strategy_check asks this BEFORE compute, where it can still stop the
        run. Asking again afterwards closes the loop, and it is the claim that
        was wrong before: a comparison reporting success while comparing
        against nothing.

        Each claim names the file it was planned in, because the two upstream
        boxes own different halves — the planner never states a period or a
        domain, and reception never states a column count.
        """
        out: List[Dict[str, Any]] = []

        def claim(what, planned, actual, source, ok=None):
            out.append({"claim": what, "planned": planned, "actual": actual,
                        "planned_in": source,
                        "ok": (planned == actual) if ok is None else ok})

        samp = (self.plan.get("sampling") or {})
        claim("n_columns", samp.get("n_columns"),
              self.data.get("columns_total"), "strategy.json")
        claim("columns_succeeded", samp.get("n_columns"),
              self.data.get("columns_succeeded"), "strategy.json")

        planned_bands = samp.get("n_bands")
        actual_bands = len({c.get("band") for c in self.columns
                            if c.get("band") is not None}) or None
        claim("n_bands", planned_bands, actual_bands, "strategy.json")

        period = self.plan.get("period") or {}
        starts = {c.get("forcing_start") for c in self.columns
                  if c.get("forcing_start") is not None}
        claim("yr_start", period.get("yr_start"),
              (starts.pop() if len(starts) == 1 else sorted(starts) or None),
              "reception.json")

        # Validation targets are planned by station id; a target whose station
        # was never fetched cannot be compared, and saying so here is cheaper
        # than discovering it inside a figure.
        known = {s for kind in ("streamflow", "swe", "water_table")
                 for s in _station_ids((self.data.get("observations") or {}), kind)}
        for v in (self.plan.get("validation") or []):
            pinned = [str(s) for s in (v.get("stations") or [])]
            have   = [s for s in pinned if s in known]
            claim(f"validation:{v.get('variable')}", len(pinned), len(have),
                  "strategy.json", ok=(len(have) == len(pinned)))
        return out


def _station_ids(observations: Dict[str, Any], kind: str) -> List[str]:
    blk = observations.get(kind) or {}
    items = blk.get("stations") or blk.get("wells") or []
    return [str(s.get("id") or s.get("triplet")) for s in items
            if isinstance(s, dict) and (s.get("id") or s.get("triplet"))]


# ── caveat construction ─────────────────────────────────────────────────
def _caveats(experiment: Dict[str, Any],
             reception: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every constraint on inference, as uniform records.

    Uniformity is the point. `{id, severity, statement, applies_to, source}`
    can be checked; a paragraph cannot, and an unrespected caveat in prose
    looks exactly like a respected one.
    """
    out: List[Dict[str, Any]] = []

    def add(cid, severity, statement, applies_to, source):
        out.append({"id": cid, "severity": severity,
                    "statement": str(statement).strip(),
                    "applies_to": applies_to, "source": source})

    # limitations is a DICT keyed by kind, each a list of
    # {applies_to, caveat, kind} — not a flat list. Read as a list it yields
    # the kind names as statements, which is how "structural" once appeared
    # in place of a caveat.
    #
    # structural limitations are BLOCKING: "local runoff generation on a 1 m2
    # column, not routed discharge — not comparable to a stream gauge without
    # routing" does not qualify a gauge comparison, it forbids the naive one.
    # configuration limitations qualify: they say how much to trust a number
    # that is still the right number to look at.
    lims = experiment.get("limitations") or {}
    groups = (lims.items() if isinstance(lims, dict)
              else [("limitation", lims if isinstance(lims, list) else [])])
    n = 0
    for kind, items in groups:
        for lim in (items if isinstance(items, list) else [items]):
            n += 1
            if isinstance(lim, dict):
                text = lim.get("caveat") or lim.get("statement") or lim.get("text")
                where = lim.get("applies_to") or "all claims"
                kind_ = lim.get("kind") or kind
            else:
                text, where, kind_ = lim, "all claims", kind
            if not text:
                continue
            add(f"limitation_{kind_}_{n}",
                BLOCKING if str(kind_).startswith("structural") else QUALIFY,
                text, where, "experiment.json:limitations")

    # assumptions_ledger entries are {parameter, value, source, note} — there
    # is no `statement` key, so asking for one yielded None for every entry.
    for i, a in enumerate(experiment.get("assumptions_ledger") or []):
        if isinstance(a, dict):
            head = " = ".join(str(x) for x in (a.get("parameter"), a.get("value"))
                              if x is not None)
            text = " — ".join(x for x in (head, a.get("note")) if x) \
                or a.get("statement") or a.get("text")
            where = a.get("applies_to") or a.get("parameter") or "the run setup"
            src = a.get("source")
        else:
            text, where, src = a, "the run setup", None
        if not text:
            continue
        add(f"assumption_{i+1}", CONTEXT, text, where,
            f"experiment.json:assumptions_ledger"
            + (f" (source: {src})" if src else ""))

    # A bbox sample does not represent the watershed that was asked about, so
    # any basin-level claim over it is a different claim. Blocking, not a
    # footnote.
    sd = experiment.get("sampling_domain") or {}
    if sd.get("caveat"):
        add("bbox_not_watershed", BLOCKING, sd["caveat"],
            "any claim about the watershed as a whole",
            "experiment.json:sampling_domain")

    for i, corr in enumerate((experiment.get("strategy_check") or {})
                             .get("corrections") or []):
        add(f"strategy_correction_{i+1}", CONTEXT, corr,
            "the design as executed vs as planned",
            "experiment.json:strategy_check")

    # An observation source that failed is NOT an absence of observations.
    # Reporting the two the same way is how "no in-domain gauge had records"
    # became a finding about a basin whose gauge reports every year.
    for kind, blk in (reception.get("observations") or {}).items():
        if not isinstance(blk, dict):
            continue
        if blk.get("ok") is False:
            add(f"observations_failed_{kind}", BLOCKING,
                f"the {kind} fetch FAILED ({blk.get('error') or 'no reason recorded'}) "
                f"— this is not evidence that {kind} observations are absent",
                f"any claim about {kind} coverage",
                "reception.json:observations")
    return out


# ── the loader ──────────────────────────────────────────────────────────
def load(run_dir: str) -> AnalysisContext:
    """Read the three boundary files and split them.

    Resolved through run_layout, so a run directory written before the layout
    change still loads — those runs are the only record of experiments that
    cost hours of compute.
    """
    rd = Path(run_dir)
    p_rec = _layout.resolve(rd, "reception",  must_exist=True)
    p_str = _layout.resolve(rd, "strategy",   must_exist=True)
    p_exp = _layout.resolve(rd, "experiment", must_exist=True)

    reception  = _read(p_rec)
    strategy   = _read(p_str)
    experiment = _read(p_exp)
    if not experiment:
        raise FileNotFoundError(
            f"{rd}: no experiment.json — the Experiment Manager has not "
            f"packaged this run, so there is nothing to analyse")

    brief = reception.get("brief") or {}
    rs    = brief.get("run_settings") or {}

    # PLAN — the question. Reception owns domain and period; the planner owns
    # the design. Neither states the other's half, which is why both files
    # are read rather than one.
    plan = {
        "question":       reception.get("user_request"),
        "domain":         experiment.get("domain") or brief.get("domain"),
        "period":         experiment.get("period") or rs.get("resolved_period"),
        "initialization": rs.get("initialization"),
        "archetype":      strategy.get("archetype"),
        "goals":          strategy.get("goals") or experiment.get("goals"),
        "sampling":       strategy.get("sampling"),
        "validation":     strategy.get("validation"),
        "feasibility":    strategy.get("feasibility"),
    }

    # DATA — the evidence, and the only place a reported number may come from.
    data = {
        "model":             experiment.get("model"),
        "columns":           experiment.get("columns") or [],
        "columns_total":     experiment.get("columns_total"),
        "columns_succeeded": experiment.get("columns_succeeded"),
        "variable_units":    experiment.get("variable_units") or {},
        "field_semantics":   experiment.get("field_semantics") or {},
        "boundary":          experiment.get("boundary"),
        "grid":              experiment.get("grid"),
        "bands":             experiment.get("bands"),
        # WHAT THE WARM-START TRIM REMOVED. This dict is a WHITELIST, so a key
        # the manager adds to experiment.json reaches step 4 only if it is
        # named here — and step 4 has read ctx.data["spinup_dropped"] since it
        # was written. The value was wired into the package on 2026-08-13 and
        # still arrived as None, because it was dropped one layer later, here.
        "spinup_dropped":    experiment.get("spinup_dropped"),
        # Observations come from RECEPTION, which fetched them once the period
        # was fixed. The Analyzer does not re-fetch: a second fetch can
        # disagree with the first, and then the run's own record is not what
        # was compared against.
        "observations":      reception.get("observations") or {},
    }

    return AnalysisContext(
        run_dir = str(rd),
        plan    = plan,
        data    = data,
        caveats = _caveats(experiment, reception),
        sources = {"reception":  str(p_rec) if p_rec else None,
                   "strategy":   str(p_str) if p_str else None,
                   "experiment": str(p_exp) if p_exp else None},
    )
