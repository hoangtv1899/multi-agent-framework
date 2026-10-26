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
from core.keyset import KeySet                  # noqa: E402


# EVERY KEY OF experiment.json, SAID OUT LOUD. This is the list that broke:
# `spinup_dropped` was measured by the extractor, folded into the package, and
# read by step 4 — and arrived None for six days, because the dict below was
# eleven names long and that was not one of them. Nothing raised, because a
# whitelist that drops looks exactly like a whitelist that is right.
#
# The package is not copied wholesale into `data` for a reason that has not
# changed: `data` is the ONLY place a reported number may come from, and the
# audit in step 3 strikes any claim citing a number that is not in it. Putting
# the question, the goals and the caveats in there would make them citable as
# evidence. So the split stays — and it now has to be stated.
_EXPERIMENT = KeySet(
    "ctx.data",
    keep = ("model", "columns", "columns_total", "columns_succeeded",
            "variable_units", "field_semantics", "boundary", "grid", "bands",
            "spinup_dropped",
            # WHICH KIND OF STUDY. Kept on the DATA side, not routed to
            # ctx.plan with the rest of the question, because the stages that
            # branch on it are reading the evidence: step1_compare skips the
            # observation comparison on a sweep, and it needs to know that
            # from the artifact it was handed rather than by inferring it from
            # a missing basin — an inference that reads a failed fetch as a
            # design decision. ctx.plan carries its own copy from the
            # strategy; both come from the same word.
            "archetype",
            # ── a whole-domain run (docs/ELM_DOMAIN_EXPERIMENTS.md) ─────────
            # `kind` says which: one case over a whole domain rather than
            # sampled columns. `regional` is its evidence, the numbers by run,
            # region and year the server made from each run's own history;
            # `control_facts` are numbers measured from the control's output;
            # `chain` is the record of what ran (years passed, job times).
            "kind", "regional", "control_facts", "chain"),
    drop = {
        # ── routed to ctx.plan: the QUESTION, not the evidence ──────────
        "domain":     "-> ctx.plan.domain — what was asked about",
        "period":     "-> ctx.plan.period",
        "goals":      "-> ctx.plan.goals",
        "sampling_domain": "-> ctx.plan.sampling — the design, which the "
                           "interpreter reads to know the scope and may not "
                           "cite as a result",
        # ── routed to ctx.caveats: what the run cannot support ──────────
        "limitations": "-> _caveats() — the honesty payload the manager "
                       "folded in from extra_summary. Blocking caveats are "
                       "promoted there, so it must not also be evidence",
        "assumptions_ledger": "-> _caveats()",
        "strategy_check":     "-> _caveats()",
        "experiment": "-> ctx.plan.experiment: the control and the runs that "
                      "differ from it, each change with what its difference "
                      "means and does NOT mean (the design, not a result); "
                      "and _caveats(): its not-meanings and the server's "
                      "check notes",
        # ── a lone whole-domain case (piece 3): no regional numbers, so the
        # Analyzer is not run on it; classified so a reader pointed at one
        # stops on the missing numbers, not on a key ──────────────────────
        "case":         "a lone whole-domain case's recipe record, for the "
                        "design review and a person rerunning it",
        "inputs_check": "a lone whole-domain case's input check, read before "
                        "any compute",
        # ── genuinely not carried ───────────────────────────────────────
        "run_dir":   "the context carries its own run_dir, resolved from the "
                     "directory actually opened rather than from a path "
                     "recorded when the package was written",
        "created":   "when the package was written. The report dates the RUN "
                     "from the scheduler, which is the clock that ran it",
        "artifacts": "the manager's index of files it wrote. A reader wanting "
                     "a figure gets its path from the finding that cites it",
    },
    source = "experiment.json (top level)",
    where  = "src/agents/analysis/step0_context.py :: _EXPERIMENT",
)

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
        # BUILT ONCE PER CONTEXT. Each frame is reconstructed from the packaged
        # rows, and script_runner rebuilds all three for EVERY figure it runs —
        # five figures meant five rebuilds and five pickles of the same ~66k
        # rows. Nothing mutates a frame in place, so one copy serves everyone.
        # A sentinel rather than None, because None is a real answer here: it
        # means this run has no frame of that kind.
        self._frames: Dict[str, Any] = {}

    # ── the frames, built once ──────────────────────────────────────────
    def _frame(self, name: str):
        """Cached frame. `_build_<name>` does the work exactly once."""
        if name not in self._frames:
            try:
                self._frames[name] = getattr(self, f"_build_{name}")()
            except Exception:                                   # noqa: BLE001
                # A frame this run cannot build is None, the same answer a run
                # without that output gives. Raising here would take down every
                # caller of a frame they may not even need.
                self._frames[name] = None
        return self._frames[name]

    def series(self):
        """Model daily series as a tidy long frame, or None. See _build_series."""
        return self._frame("series")

    def soil(self):
        """The soil column through time, or None. See _build_soil."""
        return self._frame("soil")

    def profiles(self):
        """Depth profiles, or None. See _build_profiles."""
        return self._frame("profiles")

    def regional(self):
        """A whole-domain experiment's numbers, or None. See _build_regional."""
        return self._frame("regional")

    @property
    def is_domain(self) -> bool:
        """One case over a whole domain (a control and runs of named changes),
        not sampled columns. Its evidence is `regional()`, and its rows in
        `columns` are the run-years the year gates checked."""
        return str((self.data or {}).get("kind") or "").strip().lower() == "whole_domain"

    # ── the rule the split exists to enforce ────────────────────────────
    @property
    def columns(self) -> List[Dict[str, Any]]:
        return self.data.get("columns") or []

    def blocking(self) -> List[Dict[str, Any]]:
        """Caveats that forbid a class of claim outright."""
        return [c for c in self.caveats if c.get("severity") == BLOCKING]

    def _build_series(self):
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

    def _build_soil(self):
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

    def _build_regional(self):
        """A whole-domain experiment's numbers as a TIDY long frame, or None.

            run | region | variable | year | value | units | how | source

        THE FOURTH FRAME, for the fourth shape of output: one value per run,
        region and model year, the server's reduction of each run's gridded
        history (domain_analysis: rates totalled over the year, states
        averaged, every region restricted to the vegetated cells). Nothing is
        computed here; the numbers are copied from 03_results/regional.json as
        the package carries them.

        `how` is "sum" for a value totalled over the year and "mean" for a
        value averaged over it. An observation the server put beside the
        model's own (a burned area) is rows of its own: `run` "observed" and
        `source` "observation" for the measurement, each run's own value of
        the same quantity under that run with `source` "model". A region or
        year the server could not make is absent, and its reason is in the
        caveats, never a zero here.
        """
        try:
            import pandas as pd
        except ImportError:
            return None
        t = (self.data or {}).get("regional") or {}
        if not isinstance(t, dict) or t.get("error"):
            return None
        recs = []
        for run, r in (t.get("runs") or {}).items():
            fields = r.get("fields") or {}
            for reg, by_var in (r.get("values") or {}).items():
                for var, by_year in (by_var or {}).items():
                    meta = fields.get(var) or {}
                    for y, v in (by_year or {}).items():
                        if v is not None:
                            recs.append((run, reg, var, int(y), v, meta.get("units"),
                                         meta.get("how"), "model"))
        for name, ob in (t.get("observed") or {}).items():
            units = ob.get("units")
            for reg, by_year in (ob.get("values") or {}).items():
                for y, v in (by_year or {}).items():
                    if v is not None:
                        recs.append(("observed", reg, name, int(y), v, units, "sum",
                                     "observation"))
            for run, by_reg in (ob.get("model") or {}).items():
                for reg, by_year in (by_reg or {}).items():
                    for y, v in (by_year or {}).items():
                        if v is not None:
                            recs.append((run, reg, name, int(y), v, units, "sum", "model"))
        if not recs:
            return None
        return pd.DataFrame(recs, columns=["run", "region", "variable", "year", "value",
                                           "units", "how", "source"])

    def _build_profiles(self):
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

    # ── what this run can actually support ──────────────────────────────
    def preflight(self) -> Dict[str, Any]:
        """What is here to analyse, decided before anything is spent on it.

        THE CHEAP HALF OF A FIX THAT WAS ALWAYS THE EXPENSIVE HALF. A run whose
        packaged columns carry no usable series reached step 2, spent a model
        call on five figure specifications, and discovered the absence one
        script at a time inside the runner — five subprocesses to learn
        something knowable here. A Brandywine run spent both rounds and four
        model calls to establish that the field it needed was in no frame.

        Three consumers, one computation:

            the gate    Analyzer.run() skips steps 2-3 when `blocked` is set,
                        rather than paying to be told
            the filter  step 2 refuses a proposed figure whose variables are
                        not in any frame, before running its script
            the brief   the model is shown this, so it proposes what exists

        NOT A VERDICT ON THE SCIENCE. It reports what the packaged run
        contains. Whether that answers the user's question is step 3's to say.
        """
        frames: Dict[str, Any] = {}
        available: set = set()
        for name in ("series", "soil", "profiles", "regional"):
            f = self._frame(name)
            cols = getattr(f, "columns", None)
            if f is None or cols is None or not len(f):
                frames[name] = None
                continue
            vars_ = (sorted({str(v) for v in f["variable"].dropna().unique()})
                     if "variable" in cols else [])
            frames[name] = {"rows": int(len(f)), "variables": vars_}
            available |= set(vars_)

        # WHY A VARIABLE IS NOT THERE, from the frame builder that refused it.
        # "H2OSOI is absent" and "H2OSOI is depth-resolved and lives in `soil`"
        # lead a reader to opposite conclusions.
        withheld = {k: v.get("why") for k, v in
                    (self.series_withheld or {}).items()}

        # THE PLANNER ALREADY SAID WHETHER THIS WAS FEASIBLE. Carried into
        # ctx.plan since step 0 was written and read by nothing until now: the
        # stage whose job is judging whether a study can answer the question
        # published a verdict, and the stage that spends the model calls
        # ignored it.
        feasibility = (self.plan or {}).get("feasibility")
        unmet = [c for c in self.planned_vs_actual() if not c.get("ok")]

        blocked = None
        if self.is_domain:
            # A WHOLE-DOMAIN RUN HAS NO COLUMNS TO MISS: its evidence is the
            # regional table, and that is what is asked for.
            if frames.get("regional") is None:
                why = ((self.data.get("regional") or {}).get("error")
                       if isinstance(self.data.get("regional"), dict) else None)
                blocked = ("this whole-domain run has no regional numbers"
                           + (f" ({why})" if why else
                              " (its plan names no analysis, or none were made)")
                           + ", so there is nothing to investigate")
        elif not self.columns:
            blocked = ("this run packaged no columns, so there is nothing to "
                       "investigate")
        elif not any(frames.values()):
            blocked = ("this run has no daily series, no soil layers and no "
                       "depth profiles — every generated script would fail on "
                       "the same missing frame")

        return {"frames": frames,
                "variables": sorted(available),
                "withheld": withheld,
                "feasibility": feasibility,
                "unmet_plan_targets": unmet,
                "n_columns": len(self.columns),
                "blocked": blocked}

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

        if self.is_domain:
            # NOT COLUMNS, BANDS AND A START YEAR: a whole-domain plan names
            # years, runs, regions and fields, and those are what it is held
            # to. The column claims would all fail on a design that never
            # asked for columns.
            self._domain_claims(claim)
            return out

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

    def _domain_claims(self, claim) -> None:
        """planned_vs_actual for a whole-domain experiment: the years planned
        against the years whose checks passed, the runs planned against the
        runs with numbers, and each planned region and field against the
        table."""
        from agents.analysis.domain_report import region_matches
        de = (self.plan or {}).get("experiment") or {}
        t = (self.data or {}).get("regional") or {}
        ch = (self.data or {}).get("chain") or {}
        years = [int(y) for y in (de.get("years") or [])]
        claim("years", years or None, sorted(int(y) for y in ch.get("years_passed") or [])
              or None, "strategy.json")
        planned_runs = ["control"] + [str(r.get("name") or r.get("run"))
                                      for r in (de.get("planned_runs") or de.get("runs") or [])
                                      if isinstance(r, dict)
                                      and (r.get("name") or r.get("run")) != "control"]
        have_runs = sorted((t.get("runs") or {}).keys())
        claim("runs", sorted(planned_runs), have_runs, "strategy.json")
        analysis = de.get("analysis") or {}
        have = list((t.get("regions") or {}).keys())
        for r in analysis.get("regions") or []:
            name = str(r.get("name") if isinstance(r, dict) else r)
            got = region_matches(name, have)
            claim(f"region:{name}", name, got or None, "strategy.json", ok=bool(got))
        fields = {v for r in (t.get("runs") or {}).values() for v in (r.get("fields") or {})}
        for v in analysis.get("variables") or []:
            claim(f"variable:{v}", v, v if v in fields else None, "strategy.json",
                  ok=v in fields)
        if analysis.get("normal_years"):
            claim("normal_years", [int(y) for y in analysis["normal_years"]],
                  (t.get("event") or {}).get("normal_years"), "strategy.json")


def _station_ids(observations: Dict[str, Any], kind: str) -> List[str]:
    blk = observations.get(kind) or {}
    items = blk.get("stations") or blk.get("wells") or []
    return [str(s.get("id") or s.get("triplet")) for s in items
            if isinstance(s, dict) and (s.get("id") or s.get("triplet"))]


# ── caveat construction ─────────────────────────────────────────────────
def _caveats(experiment: Dict[str, Any],
             reception: Dict[str, Any],
             strategy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
            # AN EXPLICIT SEVERITY WINS. The kind rule below is a good default
            # for the two original families — structural forbids a comparison,
            # configuration qualifies a number — but it cannot speak for a
            # family whose entries differ from each other. The conceptual set
            # is exactly that: "nothing was compared against a measurement"
            # forbids every quantitative claim, while "the soils are synthetic
            # and uniform" qualifies one. Derived from the kind, both came out
            # `qualify` and the blocking one stopped binding.
            sev = str(lim.get("severity") or "").strip().lower() \
                if isinstance(lim, dict) else ""
            add(f"limitation_{kind_}_{n}",
                sev if sev in (BLOCKING, QUALIFY, CONTEXT) else
                (BLOCKING if str(kind_).startswith("structural") else QUALIFY),
                text, where, "experiment.json:limitations")

    # TWO PRODUCERS, TWO SHAPES, AND THE READER HAS TO KNOW BOTH.
    # ELM's ledger is a settings record — {parameter, value, source, note} —
    # and there is no `statement` key, so asking for one yielded None for
    # every entry. PFLOTRAN's is a decision record — {assumption, why, cost,
    # source} — and shares not one of those names, so a ledger of three
    # entries produced a caveat list of zero: the honesty payload was written,
    # packaged and read, and dropped here in silence because the reader knew
    # only the other shape. Both are read now.
    for i, a in enumerate(experiment.get("assumptions_ledger") or []):
        sev = CONTEXT
        if isinstance(a, dict):
            head = " = ".join(str(x) for x in (a.get("parameter"), a.get("value"))
                              if x is not None) or a.get("assumption")
            # `cost` IS A SEVERITY, not a note. It says what the assumption
            # costs the conclusion — "these columns drain far more completely
            # than a real soil would" bounds every claim about drainage — and
            # that is what QUALIFY means. Read off the entry rather than
            # declared, so a ledger that states no cost stays context.
            cost = a.get("cost")
            if cost:
                sev = QUALIFY
            text = " — ".join(x for x in (head, a.get("note"), a.get("why"))
                              if x) or a.get("statement") or a.get("text")
            if cost:
                text = f"{text} (cost: {cost})" if text else str(cost)
            where = (a.get("applies_to") or a.get("parameter")
                     or a.get("assumption") or "the run setup")
            src = a.get("source")
        else:
            text, where, src = a, "the run setup", None
        if not text:
            continue
        add(f"assumption_{i+1}", sev, text, where,
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

    # A TRIM THAT WAS NEEDED AND DID NOT FIT. The extractor drops the start-up
    # transient before any statistic, but it cannot drop more record than the
    # run produced: a one-year cold run wants a year gone and has nothing left
    # over. The series that reaches step 3 is then entirely transient, and the
    # only thing standing between that and an annual mean presented as physics
    # is this caveat.
    drop = experiment.get("spinup_dropped") or {}
    if drop and not drop.get("applied") and drop.get("note"):
        add("spinup_not_trimmed", BLOCKING,
            f"{drop['note']} ({drop.get('reason')})",
            "any mean, total or trend over the run period",
            "experiment.json:spinup_dropped")

    # A ROW'S `warning` IS THE SERVER'S SENTENCE ABOUT THAT COLUMN — "only
    # 0.01 m of unsaturated column", "built steady, no daily rain" — written
    # by the build call, carried onto the row by the package's asserted list.
    # One caveat per distinct sentence, naming the columns that carry it,
    # because "8 of 17 columns cannot show vertical transit" is the single
    # fact most likely to change what a figure may claim, and until
    # 2026-08-18 it reached this reader only as a number (unsaturated_m) that
    # nothing looked at. Model-blind: this knows nothing about saturation,
    # only that a column came with a warning attached.
    by_text: Dict[str, List[str]] = {}
    for row in experiment.get("columns") or []:
        w = row.get("warning") if isinstance(row, dict) else None
        if w:
            by_text.setdefault(str(w).strip(), []).append(
                str(row.get("case_name") or row.get("id") or "?"))
    for i, (text, ids) in enumerate(sorted(by_text.items())):
        add(f"column_warning_{i+1}", QUALIFY,
            f"{len(ids)} column(s) — {', '.join(ids)}: {text}",
            "any claim resting on these columns",
            "experiment.json:columns[*].warning")
    for i, corr in enumerate((experiment.get("strategy_check") or {})
                             .get("corrections") or []):
        add(f"strategy_correction_{i+1}", CONTEXT, corr,
            "the design as executed vs as planned",
            "experiment.json:strategy_check")

    if str(experiment.get("kind") or "").strip().lower() == "whole_domain":
        _domain_caveats(experiment, strategy or {}, add)

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


def _domain_caveats(experiment: Dict[str, Any], strategy: Dict[str, Any], add) -> None:
    """A whole-domain experiment's constraints, each from the record that
    states it. Model-blind: the words are the server's and the planner's.

        not-meaning   what a run's difference from the control must never be
                      called, from the server's change list. BLOCKING: the
                      claim it forbids is wrong, not merely unqualified
        unusual       what the server's check found odd in the experiment
        facts         what the server's check measured that bears on it
        plan          the caveats the planner wrote into the design
        field notes   what a field's label does not say (a rate per second)
        problems      a region, field or year the server could not make
        chain         years planned that did not pass their checks
    """
    ex = experiment.get("experiment") or {}
    for r in ex.get("runs") or []:
        if not isinstance(r, dict) or r.get("reused") or r.get("run") == "control":
            continue
        for c in r.get("changes") or []:
            if isinstance(c, dict) and c.get("not_meaning"):
                add(f"not_meaning_{r.get('run')}_{c.get('id')}", BLOCKING,
                    f"run {r.get('run')} ({c.get('id')}): its difference from the "
                    f"control is NOT {c['not_meaning']}",
                    f"what run {r.get('run')}'s difference from the control is called",
                    "experiment.json:experiment.runs (the model server's change list)")
    check = ex.get("check") or {}
    for i, u in enumerate(check.get("unusual") or [], 1):
        add(f"experiment_unusual_{i}", QUALIFY,
            f"{u.get('where')}: {u.get('what')}" if isinstance(u, dict) else u,
            "the experiment as it was built",
            "experiment.json:experiment.check.unusual (the model server's check)")
    for i, f in enumerate(check.get("facts") or [], 1):
        add(f"experiment_fact_{i}", CONTEXT,
            f"{f.get('where')}: {f.get('what')}" if isinstance(f, dict) else f,
            "the experiment as it was built",
            "experiment.json:experiment.check.facts (the model server's check)")
    for i, c in enumerate(((strategy.get("domain_experiment") or {}).get("caveats")) or [], 1):
        add(f"plan_caveat_{i}", QUALIFY, c, "the experiment's design",
            "strategy.json:domain_experiment.caveats (the planner)")
    t = experiment.get("regional") or {}
    notes: Dict[str, str] = {}
    for run in (t.get("runs") or {}).values():
        for v, meta in ((run or {}).get("fields") or {}).items():
            if (meta or {}).get("note"):
                notes.setdefault(v, meta["note"])
    for v, note in sorted(notes.items()):
        add(f"field_note_{v}", QUALIFY, f"{v}: {note}", v,
            "experiment.json:regional.runs[*].fields (the model server)")
    probs = list(t.get("problems") or [])
    for run, r in (t.get("runs") or {}).items():
        probs += [f"run {run}: {q}" for q in (r or {}).get("problems") or []]
    for i, q in enumerate(probs, 1):
        add(f"regional_problem_{i}", QUALIFY, f"not reported: {q}",
            "completeness of the regional numbers", "experiment.json:regional.problems")
    if isinstance(t, dict) and t.get("error"):
        add("regional_not_made", BLOCKING,
            f"the regional numbers could not be made: {t['error']}",
            "every claim about the experiment's numbers", "experiment.json:regional")
    ch = experiment.get("chain") or {}
    planned, passed = ch.get("years_planned") or [], ch.get("years_passed") or []
    if ch.get("first_unpassed") or sorted(planned) != sorted(passed):
        add("chain_incomplete", BLOCKING,
            f"the chain stopped at {ch.get('first_unpassed') or 'an unrecorded stage'}: "
            f"the years {passed} of {planned} passed their checks",
            "any claim about a year without a passing check",
            "experiment.json:chain")


# ── the loader ──────────────────────────────────────────────────────────
def _experiment_plan(strategy: Dict[str, Any],
                     experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    de = strategy.get("domain_experiment") or {}
    ex = experiment.get("experiment") or {}
    if not (de or ex):
        return None
    return {"control": ex.get("control") or de.get("control"),
            # as built, each change with its meaning and not-meaning (package)
            "runs": ex.get("runs") or de.get("runs") or [],
            # as planned (strategy), for planned_vs_actual
            "planned_runs": de.get("runs") or [],
            "years": de.get("years") or (experiment.get("chain") or {}).get("years_planned"),
            "analysis": de.get("analysis") or {},
            "caveats": de.get("caveats") or [],
            "check": ex.get("check")}


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
        # A WHOLE-DOMAIN EXPERIMENT'S DESIGN: the control, the runs (each
        # change with what its difference means and does not, as the server's
        # list says, from the package), the years and the planner's analysis.
        # The question's half, never evidence. None for any other study.
        "experiment":     _experiment_plan(strategy, experiment),
    }

    # DATA — the evidence, and the only place a reported number may come from.
    #
    # take() raises if the manager wrote a key _EXPERIMENT above neither keeps
    # nor drops. It is the assertion `spinup_dropped` needed: the value existed
    # in experiment.json, the reader existed in step 4, and the only thing
    # missing was a name on the list. That is now a stop, not a null.
    data = _EXPERIMENT.take(experiment)
    data["columns"]         = data["columns"] or []
    data["variable_units"]  = data["variable_units"] or {}
    data["field_semantics"] = data["field_semantics"] or {}
    # Observations come from RECEPTION, which fetched them once the period was
    # fixed, so they are not on the experiment list at all. The Analyzer does
    # not re-fetch: a second fetch can disagree with the first, and then the
    # run's own record is not what was compared against.
    data["observations"]    = reception.get("observations") or {}

    return AnalysisContext(
        run_dir = str(rd),
        plan    = plan,
        data    = data,
        caveats = _caveats(experiment, reception, strategy),
        sources = {"reception":  str(p_rec) if p_rec else None,
                   "strategy":   str(p_str) if p_str else None,
                   "experiment": str(p_exp) if p_exp else None},
    )
