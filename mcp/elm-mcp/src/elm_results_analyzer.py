#!/usr/bin/env python3
"""
ELM Results Analyzer
mcp/elm-mcp/src/elm_results_analyzer.py

Single responsibility: read ELM NetCDF history files
and extract hydrological variables into a standard dict.

Plotting is NOT this class's job, and neither is interpretation. Figures come
from scripts/analyze_run.py; every claim about the ensemble is computed by the
Analyzer's step 2 from the artifact this class writes.
"""
import logging
from pathlib import Path
from typing  import Dict, List, Any

try:
    import xarray as xr
    import netCDF4  # noqa — ensure engine available
    XARRAY_AVAILABLE = True
except ImportError as e:
    XARRAY_AVAILABLE = False
    print(f"⚠️  xarray/netCDF4 not available: {e}")

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
# CANONICAL DEFINITIONS LIVE IN extract.py. Imported rather than repeated so
# the two cannot drift — this module is on its way out, and until it goes both
# it and the extract tool must agree about what a variable is and what it is
# measured in.
from extract import (                                           # noqa: E402
    TARGET_VARIABLES, VARIABLE_UNITS, SPINUP_DAYS,
    extract_column, history_files, write_extracted, _sigfig)    # noqa: F401
from column_metrics import column_metrics                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# ELM RESULTS ANALYZER
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# WARM-START RELAXATION
# ─────────────────────────────────────────────────────────────────────
# A warm start inherits storage from the CONUS spin-up, and that state is not
# in equilibrium with THIS domain's forcing. The column drains hard for the
# first days of the run while it settles, and the transient is enormous:
# measured on the 2019 Upper Gunnison run, column-mean QOVER+QDRAI was
#
#     day 1   45.82 mm/day     256x the Feb-Jun baseline of 0.179
#     day 2    2.90            16x
#     day 6    0.79           4.4x
#     day 14   0.34           1.9x
#
# One day held ~75% of January's total flux. Any annual metric or hydrograph
# built over the whole record is dominated by model settling rather than by
# hydrology, and a monthly plot of it is unreadable.
#
# 14 DAYS, because that is where the decay reaches ~2x the baseline. It is a
# judgement call on a smooth exponential tail with no natural knee — shorter
# leaves a visible transient (day 2 is still 16x), longer costs real record on
# a one-year run. It is one constant, set it to 0 to keep everything.
#
# NOT SILENT. What was dropped is recorded on the analyzer and travels into the
# package, because a run whose first fortnight is missing must say so — a
# reader comparing this to a gauge record needs to know the series does not
# start where the simulation did.
#
# The constant itself now lives in extract.py and is imported above.


def _metric(row: Dict[str, Any], key: str):
    """One metric off a result row, whether it sits at the top or in the budget.

    The flat `annual_runoff_mm_yr` / `annual_recharge_mm_yr` / `recharge_fraction`
    were deleted on 2026-08-13 because they duplicated the water-budget terms at
    a different rounding. Everything that addressed them by name now addresses
    the budget term instead, and this is the one place that knows both shapes.
    """
    m = row.get('metrics') or {}
    if key in m:
        return m[key]
    return (m.get('water_budget') or {}).get(key)


def _keep_last_full_year(data: Dict[str, Any]) -> Dict[str, Any]:
    """The extracted block, narrowed to the last calendar year with a full
    record of days. Returns it unchanged when no year qualifies.

    300 days, not 365: the warm-start trim removes a fortnight from the front
    and a partial day from the back, so a "full" simulated year is never 365
    days on disk. The threshold only has to separate a simulated year from a
    stub, and the years it is choosing between are whole model years.
    """
    dates = (data or {}).get("dates") or []
    if not dates:
        return data
    years: Dict[str, int] = {}
    for d in dates:
        years[d[:4]] = years.get(d[:4], 0) + 1
    full = [y for y, n in years.items() if n >= 300]
    if not full or len(years) < 2:
        return data
    keep_year = max(full)
    idx = [i for i, d in enumerate(dates) if d[:4] == keep_year]
    print(f"   (last-year analysis: {keep_year}, {len(idx)} days)")
    out = dict(data)
    out["dates"] = [dates[i] for i in idx]
    out["variables"] = {}
    for var, blk in (data.get("variables") or {}).items():
        vals = blk.get("values") or []
        out["variables"][var] = dict(blk,
                                     values=[vals[i] for i in idx
                                             if i < len(vals)])
    return out


class ELMResultsAnalyzer:
    """
    Reads ELM NetCDF history files through extract.py and computes each
    column's metrics through column_metrics.py.

    It MEASURES. Every claim about the ensemble — which driver explains the
    spread, what the bands have in common, whether soil separates from forcing
    — belongs to the Analyzer's step 2, which computes it from the artifact
    this class writes.

    Usage:
        analyzer = ELMResultsAnalyzer(experiments, analysis_dir)
        analyzer.extract_all()      # -> results, and 03_results/extracted.json
    """

    def __init__(self,
                 experiments:  List[Dict[str, Any]],
                 analysis_dir: str,
                 last_year_only: bool = False,
                 spinup_days: int = SPINUP_DAYS):
        if not XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray and netCDF4 required.\n"
                "Run: pip install netcdf4"
            )
        self.experiments  = experiments
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        # spin-up runs: analyze only the final full simulated year, so the
        # science year is not averaged together with the equilibration years
        self.last_year_only = last_year_only
        # days of warm-start relaxation trimmed off the front of the record.
        # 0 keeps everything. What was actually dropped lands in
        # spinup_dropped and travels into the package — a series that does not
        # start where the simulation did must say so.
        self.spinup_days = spinup_days
        self.spinup_dropped: Dict[str, Any] = {}
        # extra top-level fields the manager attaches and _package merges into
        # experiment.json (the assumptions ledger + limitations honesty payload)
        self.extra_summary: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        # The raw extracted blocks, kept so extract_all can write the ONE
        # artifact everything downstream reads. Without this the class read the
        # history files and left no record of what it read, and a run ended up
        # with an extracted.json from a previous day beside a package from this
        # one — disagreeing by a day, silently.
        self._blocks: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────
    def extract_all(self) -> Dict[str, Any]:
        """Extract variables from all experiments."""
        print(f"\n{'=' * 60}")
        print(f"ANALYZING ELM RESULTS")
        print(f"{'=' * 60}")

        for exp in self.experiments:
            case_name = exp['case_name']
            case_dir  = exp.get('case_dir')
            print(f"\n📊 {exp['scenario_name']}")

            hist_files = history_files(case_dir) if case_dir else []
            if not hist_files:
                print(f"   ⚠️  No history files found in {case_dir}")
                self.results[case_name] = self._empty_result(
                    exp, "no history files found"
                )
                continue

            print(f"   ✓ {len(hist_files)} history file(s)")
            self.results[case_name] = self._extract_one(
                exp, hist_files
            )

        if self._blocks:
            path = write_extracted(
                self.analysis_dir.parent, self._meta, self._blocks,
                list(TARGET_VARIABLES), int(self.spinup_days or 0))
            print(f"   ✓ {path.name} — the series everything downstream reads")
        # hydro_summary.json IS NOT WRITTEN ANY MORE (2026-08-13). It carried
        # the same rows as experiment.json — verified identical, case for case
        # — at 8 MB per run, and it predated extracted.json, which is now the
        # record of what was read. Two files holding one thing is how they end
        # up disagreeing: this run directory already had an extracted.json a day
        # older than its package, and nothing said so.
        print(f"\n✅ Analysis complete — "
              f"{len(self.results)} experiments")
        return self.results

    # get_llm_analysis_input() IS GONE (2026-08-13), and with it
    # _compute_spatial_summary and _compute_driver_matrix.
    #
    # It packed a prompt for AnalysisReportAgent: the rows, two ensemble
    # correlations, and a `focus_variables` block naming which variables an
    # interpreter should pay attention to. No model was called from here — but
    # deciding what a reader should look at is not an extractor's judgement to
    # make, and this server's rule is that it measures. It also duplicated
    # step2_derive.driver_matrix and .spatial_summary, which the framework
    # already owned and every other caller already used.
    #
    # The agent it fed is deleted; the Analyzer's steps 3 and 4 do that job
    # over the comparison, the caveats and the figures, none of which this
    # payload carried.

    # ─────────────────────────────────────────────────────────
    # PRIVATE — EXTRACTION
    # ─────────────────────────────────────────────────────────
    # FINDING AND OPENING THE FILES IS extract.py's, and always was — this
    # class kept its own copy of both, plus its own _drop_spinup, which is 60
    # lines saying the same thing twice about the same files. `combine`,
    # `coords` and `compat` decide how monthly files concatenate, so a drift
    # between the two copies would not raise; it would quietly change what a
    # year contains. See _extract_one, which now calls the shared ones.

    def _extract_one(self,
                     exp:        Dict[str, Any],
                     hist_files: List[Path]) -> Dict[str, Any]:
        """One column's row: the extracted series, plus what it adds up to.

        READS THROUGH extract.py AND COMPUTES THROUGH column_metrics.py — this
        method now does neither itself. It used to open the dataset, resample
        it, summarise every variable off the RAW 3-hourly array and derive the
        metrics in the same pass, which made every number here impossible to
        check without re-reading NetCDF off purgeable scratch.

        The series is `extract_column`'s, byte for byte the same one
        extracted.json publishes. The metrics are computed FROM that series and
        from nothing else, so anyone holding the run directory can recompute
        them and get the same answer — which is the only arrangement in which a
        wrong number can be traced to where it went wrong.
        """
        try:
            case_dir = exp.get('case_dir')
            data, meta = extract_column(str(case_dir), spinup_days=0
                                        if self.spinup_days is None
                                        else self.spinup_days)
            # Kept BEFORE the last-year filter and before anything derived, so
            # the artifact records what was read rather than what was used.
            name = exp.get('case_name')
            if name:
                self._meta[name] = dict(meta, **{
                    k: exp.get(k) for k in
                    ('lat', 'lon', 'elevation_m', 'band', 'forcing_start',
                     'forcing_end', 'scenario_name', 'pinned', 'station_id',
                     'station_variable') if exp.get(k) is not None})
                if data.get('variables'):
                    self._blocks[name] = data
            if not data:
                return self._empty_result(exp, meta.get('status') or 'no data')

            # LAST-YEAR-ONLY IS NOW A FILTER ON THE SERIES, not a slice of the
            # dataset. It used to select the last calendar year with >=1000
            # 3-hourly steps before anything was resampled; downstream the same
            # question is "which year has a full record of days". Kept because
            # analyze_run.py --last-year is a real entry point, and a spin-up
            # run whose science year is averaged with its equilibration years
            # reports neither.
            if self.last_year_only:
                data = _keep_last_full_year(data)

            n_drop = meta.get('n_timesteps_dropped_spinup') or 0
            dates = data.get('dates') or []
            if n_drop and dates:
                self.spinup_dropped = {
                    "days": int(self.spinup_days),
                    "from": meta.get("record_start"), "to": dates[0],
                    "timesteps_dropped": int(n_drop),
                    "reason": "warm-start relaxation; storage inherited from "
                              "the CONUS spin-up is not in equilibrium with "
                              "this domain's forcing"}
                print(f"   (dropped {self.spinup_days} d of warm-start "
                      f"relaxation: -> {dates[0]})")
            if meta.get('partial_days_dropped'):
                print(f"   (dropped {len(meta['partial_days_dropped'])} "
                      f"partly-sampled day(s): "
                      f"{', '.join(meta['partial_days_dropped'])})")

            derived = column_metrics(data)
            metrics = derived["metrics"]

            # The row's `variables` keeps the shape every consumer already
            # reads — the summary stats at the top, the daily series under
            # `daily` — but both halves now come from the published series.
            variables: Dict[str, Any] = {}
            for var in TARGET_VARIABLES:
                blk = (data.get('variables') or {}).get(var)
                if not blk:
                    print(f"   ⚠️  {var} not in history files")
                    variables[var] = None
                    continue
                entry = dict(derived["stats"].get(var) or {})
                entry["daily"] = {"units": blk.get("units"),
                                  "dates": dates,
                                  "values": blk.get("values")}
                for k in ("n_layers", "layer_depth_m", "layer_thickness_m"):
                    if blk.get(k) is not None:
                        entry["daily"][k] = blk[k]
                variables[var] = entry
                print(f"   ✓ {var}")

            return {
                'case_name':      exp['case_name'],
                'scenario_name':  exp['scenario_name'],
                'forcing_period': exp['forcing_period'],
                'forcing_start':  exp['forcing_start'],
                'forcing_end':    exp['forcing_end'],
                'lat':            exp.get('lat'),
                'lon':            exp.get('lon'),
                'elevation_m':    exp.get('elevation_m'),
                'soil':           exp.get('soil'),
                # PIN PROVENANCE, carried so the comparison can honour the
                # pairing the sampler already decided instead of re-deriving it
                # geometrically. Absent for every column that was not pinned.
                'pinned':            exp.get('pinned'),
                'station_id':        exp.get('station_id'),
                'station_variable':  exp.get('station_variable'),
                'status':         'ok',
                'variables':      variables,
                'metrics':        metrics,
                'history_files':  [str(f) for f in hist_files],
            }

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            print(f"   ✗ Extraction failed: {e}")
            return self._empty_result(exp, str(e))


    # `_compute_comparisons` DELETED 2026-08-13 — it could not return anything.
    #
    # It compared a metric ACROSS FORCING PERIODS, keying the values by
    # `forcing_period` and skipping when fewer than two were distinct. That is a
    # scenario-ensemble question (baseline vs dry vs wet), and this framework
    # builds SPATIAL ensembles: columns_to_plan.build_ledger takes one
    # forcing_period for the whole study and stamps it on every column's
    # coupler, so the dict it built always had exactly one key. Measured across
    # every packaged run on disk: `forcing_period` is 'baseline', everywhere.
    #
    # So it returned [] on every run since the ensemble design changed, and
    # `comparisons: []` travelled into the report payload looking like a
    # finding — "nothing differed" rather than "nothing was compared". If the
    # scenario ensemble comes back, this belongs with whatever builds it.

    @staticmethod
    def _empty_result(exp:    Dict[str, Any],
                      reason: str = 'unknown') -> Dict[str, Any]:
        """Safe fallback when extraction fails."""
        return {
            'case_name':      exp.get('case_name',      'unknown'),
            'scenario_name':  exp.get('scenario_name',  'unknown'),
            'forcing_period': exp.get('forcing_period', 'unknown'),
            'forcing_start':  exp.get('forcing_start',  None),
            'forcing_end':    exp.get('forcing_end',    None),
            'lat':            exp.get('lat'),
            'lon':            exp.get('lon'),
            'elevation_m':    exp.get('elevation_m'),
            'soil':           exp.get('soil'),
            'status':         'failed',
            'reason':         reason,
            'variables':      {v: None for v in TARGET_VARIABLES},
            'metrics':        {},
            'history_files':  [],
        }