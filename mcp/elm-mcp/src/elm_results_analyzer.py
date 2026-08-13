#!/usr/bin/env python3
"""
ELM Results Analyzer
src/core/elm_results_analyzer.py

Single responsibility: read ELM NetCDF history files
and extract hydrological variables into a standard dict.

Plotting is NOT this class's job: figures come from tools/analyze_run.py,
which reads the summaries computed here. They are written straight into
04_analysis/ next to hydro_summary.json.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing  import Dict, List, Any, Optional

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
    Reads ELM NetCDF history files.
    Extracts QCHARGE, QOVER, TWS, SOILLIQ.
    Computes annual metrics and cross-experiment comparisons.

    Usage:
        analyzer = ELMResultsAnalyzer(experiments, analysis_dir)
        analyzer.extract_all()
        llm_input = analyzer.get_llm_analysis_input()
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
        # spinup_dropped and travels into hydro_summary.json — a series that
        # does not start where the simulation did must say so.
        self.spinup_days = spinup_days
        self.spinup_dropped: Dict[str, Any] = {}
        # extra top-level fields merged into hydro_summary.json at save time
        # (e.g. the assumptions ledger + limitations honesty payload)
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
        self._save_hydro_summary()
        print(f"\n✅ Analysis complete — "
              f"{len(self.results)} experiments")
        return self.results

    def get_llm_analysis_input(self) -> Dict[str, Any]:
        """Package results for AnalysisReportAgent."""
        return {
            'model_type':      'elm',
            'experiments':      list(self.results.values()),
            'spatial_summary':  self._compute_spatial_summary(),
            'soil_attribution': self._compute_soil_attribution(),
            'driver_matrix':    self._compute_driver_matrix(),
            'units':           VARIABLE_UNITS,
            'focus_variables': {
                'QCHARGE': 'Primary — aquifer recharge',
                'QOVER':   'Surface runoff',
                'ZWT':     'Water-table depth',
                'TWS':     'Total water storage',
                'SOILLIQ': 'Soil moisture profile',
            },
            'file_locations': {
                'analysis_dir':  str(self.analysis_dir),
                'hydro_summary': str(
                    self.analysis_dir / "hydro_summary.json"
                ),
            }
        }

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
    # `comparisons: []` travelled into LLM_ANALYSIS_INPUT.json looking like a
    # finding — "nothing differed" rather than "nothing was compared". If the
    # scenario ensemble comes back, this belongs with whatever builds it.

    def _compute_spatial_summary(self) -> Dict[str, Any]:
        """Cross-column summary for a SPATIAL ensemble that attributes the response
        to its ACTUAL drivers rather than asserting a clean elevation gradient.

        It surfaces per-column forcing (precip), how well a linear elevation fit
        holds (fit_r2), and whether recharge tracks elevation or precip — so a
        coarse, quantized forcing (which makes 'elevation' a confounded proxy)
        can't masquerade as a smooth elevation effect."""
        ok = [r for r in self.results.values()
              if r.get('status') == 'ok' and r.get('elevation_m') is not None]
        locs = {(r.get('lat'), r.get('lon')) for r in ok}
        if len(ok) < 2 or len(locs) < 2:
            return {}

        ok.sort(key=lambda r: r['elevation_m'])
        rows = [{
            'case_name':           r['case_name'],
            'elevation_m':         round(r['elevation_m'], 1),
            'lat':                 r.get('lat'),
            'lon':                 r.get('lon'),
            'precip_mm_yr':        r['metrics'].get('precip_mm_yr'),
            'recharge_mm_yr':      _metric(r, 'recharge_mm_yr'),
            'runoff_mm_yr':        _metric(r, 'runoff_mm_yr'),
            'recharge_frac_of_P':  _metric(r, 'recharge_frac_of_P'),
            'water_table_depth_m': r['metrics'].get('water_table_depth_m'),
        } for r in ok]

        elevs = np.array([r['elevation_m'] for r in ok], dtype=float)

        def col(metric_key):
            # metrics first, then the water budget — the terms that used to be
            # duplicated at the top level (annual_runoff_mm_yr and friends) now
            # live only in the budget, and a key name should still address them.
            return np.array([_metric(r, metric_key) for r in ok], dtype=float)

        def fit_vs_elev(metric_key):
            """Linear slope per 1000 m AND its r2 (how well a line vs elevation fits)."""
            ys = col(metric_key)
            mask = ~np.isnan(ys)
            if mask.sum() < 2 or np.ptp(elevs[mask]) < 1e-6:
                return None, None
            x, y = elevs[mask], ys[mask]
            a, b = np.polyfit(x, y, 1)
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = (round(1 - float(np.sum((y - (a * x + b)) ** 2)) / ss_tot, 3)
                  if ss_tot > 1e-12 else None)
            return round(float(a) * 1000.0, 4), r2

        def corr(metric_key, xs):
            ys = col(metric_key)
            mask = ~np.isnan(ys) & ~np.isnan(xs)
            # a correlation still needs 3; with 2 the rows carry the comparison
            if mask.sum() < 3 or np.ptp(xs[mask]) < 1e-9 or np.ptp(ys[mask]) < 1e-9:
                return None
            return round(float(np.corrcoef(xs[mask], ys[mask])[0, 1]), 3)

        precip = col('precip_mm_yr')
        finite = precip[~np.isnan(precip)]
        bins = sorted({round(float(p)) for p in finite})        # distinct forcing cells

        slope, r2 = {}, {}
        for key, name in [('recharge_mm_yr', 'recharge_mm_yr'),
                          ('runoff_mm_yr', 'runoff_mm_yr'),
                          ('recharge_frac_of_P', 'recharge_frac_of_P'),
                          ('water_table_depth_m', 'water_table_m')]:
            slope[name], r2[name] = fit_vs_elev(key)

        r_elev = corr('recharge_mm_yr', elevs)
        r_prcp = corr('recharge_mm_yr', precip)

        notes = []
        if bins and len(bins) <= 3 and len(ok) > len(bins):
            notes.append(f"precip is quantized to {len(bins)} value(s) {bins} mm/yr — "
                         f"coarse DATM forcing, NOT elevation-resolved")
        if r_elev is not None and r_prcp is not None and abs(r_prcp) > abs(r_elev):
            notes.append(f"recharge tracks precip (r={r_prcp}) more than elevation "
                         f"(r={r_elev}) — response is forcing/soil-controlled")
        if r2.get('recharge_mm_yr') is not None and r2['recharge_mm_yr'] < 0.5:
            notes.append(f"linear elevation fit is weak (r2={r2['recharge_mm_yr']}) — "
                         f"the elevation 'gradient' is not a reliable summary")

        return {
            'n_columns':         len(ok),
            'elevation_range_m': [round(float(elevs.min()), 1), round(float(elevs.max()), 1)],
            'forcing': {
                'precip_mm_yr_distinct': bins,
                'n_forcing_bins':        len(bins),
                'elevation_resolved':    len(bins) > 3,
            },
            'by_elevation': rows,
            'vs_elevation': {'slope_per_1000m': slope, 'fit_r2': r2},
            'driver_correlation': {'recharge_vs_elevation_r': r_elev,
                                   'recharge_vs_precip_r':    r_prcp},
            'interpretation': notes or ['response varies smoothly with elevation'],
            'note': 'check forcing.n_forcing_bins, vs_elevation.fit_r2 and '
                    'driver_correlation before reading any slope as an elevation effect',
        }

    def _compute_soil_attribution(self) -> Dict[str, Any]:
        """Attribute the partitioning to SOIL, holding forcing constant. Picks the
        largest forcing bin (so precip is fixed) and correlates recharge / runoff
        with per-column soil predictors (max clay %, min Ksat = the drainage
        bottleneck). This is the soil-control answer the spatial run confounds."""
        ok = [r for r in self.results.values()
              if r.get('status') == 'ok' and r.get('soil')]
        if len(ok) < 3:
            return {}

        # hold forcing constant: keep only the most-populated precip bin
        bins: Dict[Any, list] = {}
        for r in ok:
            p = r['metrics'].get('precip_mm_yr')
            bins.setdefault(round(p) if p is not None else None, []).append(r)
        precip_bin, group = max(bins.items(), key=lambda kv: len(kv[1]))
        # 2, not 3. The 12 km forcing quantises precipitation so heavily that a
        # 13-column ensemble rarely puts 3 columns in one bin — so the ONE
        # analysis that holds forcing constant almost never ran. With 2 the
        # correlation is meaningless, but the PAIR is not: two columns under
        # identical forcing that differ in recharge differ because of soil.
        if len(group) < 2:
            return {}

        group.sort(key=lambda r: -(_metric(r, 'recharge_mm_yr') or 0))
        rows = [{
            'case_name':         r['case_name'],
            'texture_top':       r['soil'].get('texture_top'),
            'clay_max_pct':      r['soil'].get('clay_max_pct'),
            'ksat_min_ums':      r['soil'].get('ksat_min_ums'),
            'recharge_mm_yr':    _metric(r, 'recharge_mm_yr'),
            'runoff_mm_yr':      _metric(r, 'runoff_mm_yr'),
            'recharge_frac_of_P': _metric(r, 'recharge_frac_of_P'),
        } for r in group]

        def corr(feat, target):
            xs = np.array([r['soil'].get(feat) for r in group], dtype=float)
            ys = np.array([r['metrics'].get(target) for r in group], dtype=float)
            mask = ~np.isnan(xs) & ~np.isnan(ys)
            # a correlation still needs 3; with 2 the rows carry the comparison
            if mask.sum() < 3 or np.ptp(xs[mask]) < 1e-9 or np.ptp(ys[mask]) < 1e-9:
                return None
            return round(float(np.corrcoef(xs[mask], ys[mask])[0, 1]), 3)

        soil_corr = {
            'recharge_vs_clay_max': corr('clay_max_pct', 'recharge_mm_yr'),
            'recharge_vs_ksat_min': corr('ksat_min_ums', 'recharge_mm_yr'),
            'runoff_vs_clay_max':   corr('clay_max_pct', 'runoff_mm_yr'),
            'runoff_vs_ksat_min':   corr('ksat_min_ums', 'runoff_mm_yr'),
        }
        ranked = sorted(((abs(v), k, v) for k, v in soil_corr.items() if v is not None),
                        reverse=True)
        best = (f"{ranked[0][1]} (r={ranked[0][2]})" if ranked else "no clear predictor")

        return {
            'forcing_held_mm_yr': precip_bin,
            'n_columns':          len(group),
            'by_recharge':        rows,
            'soil_correlation':   soil_corr,
            'strongest_predictor': best,
            'note': f'precip held at {precip_bin} mm/yr across {len(group)} columns, '
                    'so this spread is soil-driven (clay impedes, Ksat permits drainage)',
        }

    # response name -> where its value lives in a result's metrics
    _RESPONSE_GETTERS = {
        'runoff':             lambda m: (m.get('water_budget') or {}).get('runoff_mm_yr'),
        'infiltration':       lambda m: (m.get('water_budget') or {}).get('infiltration_mm_yr'),
        'et':                 lambda m: (m.get('water_budget') or {}).get('et_mm_yr'),
        'recharge':           lambda m: (m.get('water_budget') or {}).get('recharge_mm_yr'),
        'recharge_frac_of_P': lambda m: (m.get('water_budget') or {}).get('recharge_frac_of_P'),
    }
    _DRIVER_GETTERS = {
        'elevation_m':  lambda r: r.get('elevation_m'),
        'precip_mm_yr': lambda r: r['metrics'].get('precip_mm_yr'),
        'clay_max_pct': lambda r: (r.get('soil') or {}).get('clay_max_pct'),
        'ksat_min_ums': lambda r: (r.get('soil') or {}).get('ksat_min_ums'),
    }

    def _compute_driver_matrix(self) -> Dict[str, Any]:
        """Pearson r for EVERY response (runoff, infiltration, ET, recharge,
        recharge fraction) against EVERY driver (elevation, forcing precip,
        soil clay, soil Ksat) across the ok columns — so the interpreter sees
        the full relationship structure, not cherry-picked pairs. Responses
        missing from a run (e.g. infiltration/ET in pre-2026-07 output) are
        simply omitted."""
        ok = [r for r in self.results.values() if r.get('status') == 'ok']
        if len(ok) < 3:
            return {}

        def corr(xs, ys):
            x = np.array(xs, dtype=float)
            y = np.array(ys, dtype=float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 3 or np.ptp(x[mask]) < 1e-9 or np.ptp(y[mask]) < 1e-9:
                return None
            return round(float(np.corrcoef(x[mask], y[mask])[0, 1]), 3)

        nan = float('nan')
        matrix: Dict[str, Any] = {}
        for resp, rget in self._RESPONSE_GETTERS.items():
            ys = [rget(r['metrics']) for r in ok]
            ys = [nan if v is None else v for v in ys]
            if all(np.isnan(v) for v in ys):
                continue                     # response not in this run's output
            row = {}
            for drv, dget in self._DRIVER_GETTERS.items():
                xs = [dget(r) for r in ok]
                xs = [nan if v is None else v for v in xs]
                row[drv] = corr(xs, ys)
            matrix[resp] = row
        if not matrix:
            return {}
        return {'n_columns': len(ok), 'pearson_r': matrix,
                'note': 'correlations across ALL columns — when precip is quantized, '
                        'elevation and precip are confounded; use soil_attribution '
                        '(forcing held) for the clean soil signal'}

    def _save_hydro_summary(self):
        """Save hydro_summary.json."""
        hydro_file = self.analysis_dir / "hydro_summary.json"
        with open(hydro_file, 'w') as f:
            json.dump(
                {
                    'experiments':      list(self.results.values()),
                    # What the record does NOT cover. A reader lining this up
                    # against a gauge series has to know the model series
                    # starts later than the simulation did.
                    'spinup_dropped':   self.spinup_dropped or None,
                    # comparisons and soil_attribution are NOT here either,
                    # for the same reason as the correlations: both are
                    # derived claims, and both filtered on row['soil'], which
                    # extraction never populates. soil_attribution therefore
                    # returned {} on EVERY run and soil_control.png was
                    # silently never drawn. src/agents/drivers.py computes
                    # them from the package rows, where the soil profile is.
                    # spatial_summary and driver_matrix are NOT here any more.
                    # A correlation is a claim about a relationship, which is
                    # interpretation; extraction reads the model's output
                    # format and stops. Frozen here they could also never
                    # answer a driver thought of later. src/agents/drivers.py
                    # computes both from the package rows.
                    #
                    # The concrete damage: this version read
                    # row['soil'].get('clay_max_pct'), and extraction never
                    # populates `soil`, so every soil correlation came out
                    # null on every run — and null in a correlation table
                    # reads as "no relationship", not "not computed".
                    'units':            VARIABLE_UNITS,
                    **self.extra_summary,
                },
                f, indent=2, default=str
            )
        print(f"\n   ✓ hydro_summary.json saved")

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