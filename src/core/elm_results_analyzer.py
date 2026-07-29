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
TARGET_VARIABLES = ['QOVER', 'QCHARGE', 'TWS', 'SOILLIQ', 'ZWT', 'RAIN', 'H2OSNO',
                    'SNOW', 'QINFL', 'QDRAI', 'QSOIL', 'QVEGE', 'QVEGT']

VARIABLE_UNITS = {
    'QOVER':   'mm/s',
    'QCHARGE': 'mm/s',
    'TWS':     'mm',
    'SOILLIQ': 'kg/m2',
    'ZWT':     'm',
    'RAIN':    'mm/s',   # atmospheric forcing (rainfall flux) — surfaces the input
    'H2OSNO':  'mm',     # snow water equivalent (absent in runs before 2026-07;
                         # requested in hist_fincl1 by default since then)
    'SNOW':    'mm/s',   # snowfall forcing
    'QINFL':   'mm/s',   # infiltration into the soil column
    'QDRAI':   'mm/s',   # sub-surface drainage (baseflow)
    # ET is the sum of ground evap + canopy evap + transpiration (this ELM build
    # registers the components, not a single QFLX_EVAP_TOT):
    'QSOIL':   'mm/s',   # ground evaporation
    'QVEGE':   'mm/s',   # canopy evaporation
    'QVEGT':   'mm/s',   # canopy transpiration
}

# treated as annual fluxes (mm/yr) in _summarize
FLUX_VARIABLES = ('QOVER', 'QCHARGE', 'RAIN', 'SNOW', 'QINFL', 'QDRAI',
                  'QSOIL', 'QVEGE', 'QVEGT')

S_TO_YEAR = 86400.0 * 365.25


# ─────────────────────────────────────────────────────────────────────
# ELM RESULTS ANALYZER
# ─────────────────────────────────────────────────────────────────────
def _sigfig(x: float, n: int = 4) -> float:
    """Round to n SIGNIFICANT figures, not n decimal places.

    Decimal places are the wrong instrument when one series is a water-table
    depth near 71.67 m and another is snow water equivalent at 0.00002 mm. Five
    decimals spends characters on the former; three would zero out 4.7% of the
    values in a real run, including small-but-real recharge fluxes — 0.0005
    mm/day is 0.18 mm/yr, which matters on a column that barely recharges.

    Significant figures adapt to magnitude, so nothing real is lost and the
    string stays short. Four is already more than a land-surface model
    justifies; it is chosen to be visibly generous rather than to be argued
    about.
    """
    if x == 0 or not np.isfinite(x):
        return 0.0 if x == 0 else x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


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
                 last_year_only: bool = False):
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
        # extra top-level fields merged into hydro_summary.json at save time
        # (e.g. the assumptions ledger + limitations honesty payload)
        self.extra_summary: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
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

            hist_files = self._find_history_files(case_dir)
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

        self._save_hydro_summary()
        print(f"\n✅ Analysis complete — "
              f"{len(self.results)} experiments")
        return self.results

    def get_llm_analysis_input(self) -> Dict[str, Any]:
        """Package results for AnalysisReportAgent."""
        return {
            'model_type':      'elm',
            'experiments':      list(self.results.values()),
            'comparisons':      self._compute_comparisons(),
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
    def _find_history_files(self,
                             case_dir: Optional[str]
                             ) -> List[Path]:
        """Find *.elm.h0.*.nc files in case run directory."""
        if not case_dir:
            return []
        # Handle literal $PSCRATCH in path
        if '$PSCRATCH' in str(case_dir):
            import os
            case_dir = str(case_dir).replace(
                '$PSCRATCH',
                os.environ.get('PSCRATCH', '')
            )
        run_dir = Path(case_dir) / "run"
        if not run_dir.exists():
            return []
        return sorted(run_dir.glob("*.elm.h0.*.nc"))

    def _open_dataset(self, hist_files: List[Path]):
        """
        Open NetCDF4 files using netCDF4 engine directly.
        ELM produces NetCDF4 format.
        """
        # Explicit current defaults — silences the open_mfdataset FutureWarnings
        # without changing the combine behavior (monthly files, concat on time).
        return xr.open_mfdataset(
            [str(f) for f in hist_files],
            combine      = 'by_coords',
            decode_times = True,
            engine       = 'netcdf4',
            data_vars    = 'all',
            coords       = 'different',
            compat       = 'no_conflicts',
            join         = 'outer',
        )

    def _extract_one(self,
                     exp:        Dict[str, Any],
                     hist_files: List[Path]) -> Dict[str, Any]:
        """Extract variables from one experiment."""
        try:
            ds = self._open_dataset(hist_files)
            if self.last_year_only:
                years = np.asarray(ds['time'].dt.year.values)
                uniq, counts = np.unique(years, return_counts=True)
                full = uniq[counts >= 1000]      # a full 3-hourly year ≈ 2920 steps
                yr = int(full.max()) if len(full) else int(uniq.max())
                ds = ds.isel(time=(years == yr))
                print(f"   (last-year analysis: {yr}, "
                      f"{int(ds.sizes['time'])} timesteps)")
            variables = {}

            for var in TARGET_VARIABLES:
                if var in ds:
                    variables[var] = self._summarize(
                        ds[var], var
                    )
                    print(f"   ✓ {var}")
                else:
                    print(f"   ⚠️  {var} not in history files")
                    variables[var] = None

            ds.close()
            metrics = self._compute_metrics(variables)

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
                'status':         'ok',
                'variables':      variables,
                'metrics':        metrics,
                'history_files':  [str(f) for f in hist_files],
            }

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            print(f"   ✗ Extraction failed: {e}")
            return self._empty_result(exp, str(e))

    def _daily(self, da: "xr.DataArray", var_name: str) -> Dict[str, Any]:
        """The variable as a DAILY series, for the hydrograph and the budget.

        Nothing in the pipeline used to extract this, so validate_run rebuilt
        hydrographs by re-reading the history NetCDFs off $PSCRATCH — which
        left the Analyzer coupled to scratch rather than to the manager's
        package, and meant the series died with the scratch purge while the
        run directory survived.

        DAILY, not raw. ELM writes 3-hourly here (2921 steps for one year),
        and 13 variables x 2921 steps x 19 columns is ~5.8 MB of JSON that
        nothing reads at that resolution: USGS observations are daily, so a
        hydrograph comparison resamples to daily anyway. Aggregating at
        extraction costs one pass and saves 8x.

        Fluxes come out in mm/day — the unit a daily hydrograph is actually
        plotted in, rather than the mm/s ELM stores or the mm/yr the annual
        metrics use. Stating it here is what stops the next reader guessing.
        """
        try:
            if "time" in getattr(da, "dims", ()):
                d = da.squeeze().resample(time="1D").mean()
                vals = np.array(d.values, dtype=float).flatten()
                stamps = [str(t)[:10] for t in np.array(d["time"].values)]
            else:
                vals = np.array(da.squeeze().values, dtype=float).flatten()
                stamps = []
        except Exception:
            return {}

        if vals.size == 0:
            return {}
        if var_name in FLUX_VARIABLES:
            vals = vals * 86400.0          # mm/s -> mm/day
            units = "mm/day"
        else:
            units = VARIABLE_UNITS.get(var_name, "")
        out = {"units": units,
               "values": [None if np.isnan(v) else _sigfig(float(v))
                          for v in vals]}
        if stamps:
            out["dates"] = stamps
        return out

    def _summarize(self,
                   da:       "xr.DataArray",
                   var_name: str) -> Dict[str, Any]:
        """Summary statistics for one variable, plus its daily series.

        The stats and the series answer different questions and are kept
        apart: the stats are what the annual metrics are built from, the
        series is what a hydrograph is drawn from. Attaching the series here
        rather than inside each branch means every variable gets one, and a
        new branch cannot forget.
        """
        stats = self._summarize_stats(da, var_name)
        daily = self._daily(da, var_name)
        if daily:
            stats["daily"] = daily
        return stats

    def _summarize_stats(self,
                         da:       "xr.DataArray",
                         var_name: str) -> Dict[str, Any]:
        """Compute summary statistics for one variable."""
        da  = da.squeeze()
        val = np.array(da.values, dtype=float)

        if var_name in FLUX_VARIABLES:
            val_yr = val * S_TO_YEAR
            return {
                'units_raw':    VARIABLE_UNITS[var_name],
                'units_annual': 'mm/year',
                'annual_mean':  round(float(np.nanmean(val_yr)), 4),
                'annual_std':   round(float(np.nanstd(val_yr)),  4),
                'annual_min':   round(float(np.nanmin(val_yr)),  4),
                'annual_max':   round(float(np.nanmax(val_yr)),  4),
                'n_timesteps':  int(np.size(val)),
            }

        elif var_name == 'TWS':
            val_1d = val.flatten()
            return {
                'units':          VARIABLE_UNITS[var_name],
                'mean_mm':        round(float(np.nanmean(val_1d)), 4),
                'std_mm':         round(float(np.nanstd(val_1d)),  4),
                'min_mm':         round(float(np.nanmin(val_1d)),  4),
                'max_mm':         round(float(np.nanmax(val_1d)),  4),
                'seasonal_range': round(
                    float(np.nanmax(val_1d) -
                          np.nanmin(val_1d)), 4
                ),
                # storage change over the run (last - first) — closes the budget
                'delta_mm':       round(float(val_1d[-1] - val_1d[0]), 4),
                'n_timesteps':    int(len(val_1d)),
            }

        elif var_name == 'SOILLIQ':
            # Shape: (time, levgrnd) or (levgrnd,)
            if val.ndim == 2:
                layer_means = np.nanmean(val, axis=0)
            elif val.ndim == 1:
                layer_means = val
            else:
                layer_means = val.reshape(-1)
            return {
                'units':              VARIABLE_UNITS[var_name],
                'total_column_kg_m2': round(
                    float(np.nansum(layer_means)), 4
                ),
                'layer_means_kg_m2':  [
                    round(float(v), 4) for v in layer_means
                ],
                'n_layers':           int(len(layer_means)),
                'n_timesteps':        int(val.shape[0])
                                      if val.ndim == 2 else 1,
            }

        elif var_name == 'ZWT':
            val_1d = val.flatten()
            return {
                'units':       VARIABLE_UNITS[var_name],
                'mean_m':      round(float(np.nanmean(val_1d)), 4),
                'min_m':       round(float(np.nanmin(val_1d)),  4),
                'max_m':       round(float(np.nanmax(val_1d)),  4),
                # initial vs final expose the cold-start problem: all columns
                # begin at ELM's default (~8.8 m) regardless of the real WTD
                'first_m':     round(float(val_1d[0]),  4),
                'last_m':      round(float(val_1d[-1]), 4),
                'n_timesteps': int(len(val_1d)),
            }

        elif var_name == 'H2OSNO':
            val_1d = val.flatten()
            return {
                'units':       VARIABLE_UNITS[var_name],
                'peak_swe_mm': round(float(np.nanmax(val_1d)),  1),
                'mean_swe_mm': round(float(np.nanmean(val_1d)), 1),
                'n_timesteps': int(len(val_1d)),
            }

        else:
            val_1d = val.flatten()
            return {
                'units': VARIABLE_UNITS.get(var_name, 'unknown'),
                'mean':  round(float(np.nanmean(val_1d)), 6),
                'std':   round(float(np.nanstd(val_1d)),  6),
                'min':   round(float(np.nanmin(val_1d)),  6),
                'max':   round(float(np.nanmax(val_1d)),  6),
            }

    def _compute_metrics(self,
                          variables: Dict) -> Dict[str, Any]:
        """Compute derived per-column metrics."""
        metrics = {}
        qc = variables.get('QCHARGE') or {}
        qo = variables.get('QOVER')   or {}
        tw = variables.get('TWS')     or {}
        zw = variables.get('ZWT')     or {}
        rn = variables.get('RAIN')    or {}

        if qc:
            metrics['annual_recharge_mm_yr'] = qc.get('annual_mean')
        if qo:
            metrics['annual_runoff_mm_yr'] = qo.get('annual_mean')
        sf = variables.get('SNOW') or {}
        if rn or sf:
            # PRECIPITATION IS RAIN + SNOW. This was RAIN alone, which in a
            # snow-dominated basin is not a rounding error: across the 2020
            # Naches columns the two differ by 1.11x in the warm valley and
            # 2.17x at elevation — the ratio IS the snow fraction. Every
            # runoff/P and recharge/P fraction built on it was inflated by
            # exactly that much, and columns appeared to drain more water than
            # fell on them because more than half of what fell was snow.
            # The water budget below already used rain + snow, so the two
            # disagreed inside the same metrics dict.
            _r = rn.get('annual_mean') if rn else None
            _s = sf.get('annual_mean') if sf else None
            if _r is not None or _s is not None:
                metrics['precip_mm_yr'] = round((_r or 0.0) + (_s or 0.0), 1)
                metrics['rainfall_mm_yr'] = round(_r, 1) if _r is not None else None
        if qc and qo:
            qc_m = qc.get('annual_mean', 0) or 0
            qo_m = qo.get('annual_mean', 0) or 0
            if abs(qo_m) > 1e-10:
                metrics['recharge_to_runoff_ratio'] = round(qc_m / qo_m, 4)
            # recharge vs runoff partitioning — the core science question
            total = qc_m + qo_m
            if abs(total) > 1e-9:
                metrics['recharge_fraction'] = round(qc_m / total, 4)
                metrics['runoff_fraction']   = round(qo_m / total, 4)
        if tw:
            metrics['tws_seasonal_range_mm'] = tw.get('seasonal_range')
        if zw:
            metrics['water_table_depth_m'] = zw.get('mean_m')
        sn = variables.get('H2OSNO') or {}
        if sn:
            metrics['peak_swe_mm'] = sn.get('peak_swe_mm')

        # ── per-column water budget (needs the post-2026-07 output fields) ──
        # P = RAIN + SNOW partitions into runoff + infiltration at the surface;
        # infiltration then goes to ET, recharge/drainage, or storage (ΔTWS).
        def mean(key):
            d = variables.get(key) or {}
            return d.get('annual_mean')

        rain, snow = mean('RAIN'), mean('SNOW')
        if rain is not None and snow is not None:
            p = rain + snow
            metrics['precip_total_mm_yr'] = round(p, 1)
            metrics['snowfall_mm_yr'] = round(snow, 1)
            # ET = ground evap + canopy evap + transpiration
            et_parts = [mean(k) for k in ('QSOIL', 'QVEGE', 'QVEGT')]
            et = sum(v for v in et_parts if v is not None) \
                if any(v is not None for v in et_parts) else None
            budget = {}
            for label, v in (('runoff', mean('QOVER')), ('infiltration', mean('QINFL')),
                             ('et', et), ('recharge', mean('QCHARGE')),
                             ('drainage', mean('QDRAI'))):
                if v is not None:
                    budget[f'{label}_mm_yr'] = round(v, 1)
                    if p > 1e-6:
                        budget[f'{label}_frac_of_P'] = round(v / p, 3)
            dtws = (variables.get('TWS') or {}).get('delta_mm')
            if dtws is not None:
                budget['storage_change_mm'] = round(dtws, 1)
                # closure: P - runoff - drainage - ET - ΔS  (QCHARGE is internal
                # to TWS, so it is not an export term here)
                if all(k in budget for k in ('runoff_mm_yr', 'drainage_mm_yr', 'et_mm_yr')):
                    resid = p - budget['runoff_mm_yr'] - budget['drainage_mm_yr'] \
                            - budget['et_mm_yr'] - dtws
                    budget['closure_residual_mm_yr'] = round(resid, 1)
            if budget:
                metrics['water_budget'] = budget
        return metrics

    def _compute_comparisons(self) -> List[Dict[str, Any]]:
        """Compare metrics across experiments."""
        ok = {
            k: v for k, v in self.results.items()
            if v.get('status') == 'ok'
        }
        if len(ok) < 2:
            return []

        comparisons = []
        for metric_key in ['annual_recharge_mm_yr',
                            'annual_runoff_mm_yr']:
            vals = {
                r['forcing_period']: r['metrics'].get(metric_key)
                for r in ok.values()
                if r['metrics'].get(metric_key) is not None
            }
            if len(vals) < 2:
                continue
            high = max(vals, key=vals.get)
            low  = min(vals, key=vals.get)
            comparisons.append({
                'metric':     metric_key,
                'values':     vals,
                'highest':    high,
                'lowest':     low,
                'difference': round(vals[high] - vals[low], 4),
                'units':      'mm/year',
            })
        return comparisons

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
            'recharge_mm_yr':      r['metrics'].get('annual_recharge_mm_yr'),
            'runoff_mm_yr':        r['metrics'].get('annual_runoff_mm_yr'),
            'recharge_fraction':   r['metrics'].get('recharge_fraction'),
            'water_table_depth_m': r['metrics'].get('water_table_depth_m'),
        } for r in ok]

        elevs = np.array([r['elevation_m'] for r in ok], dtype=float)

        def col(metric_key):
            return np.array([r['metrics'].get(metric_key) for r in ok], dtype=float)

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
        for key, name in [('annual_recharge_mm_yr', 'recharge_mm_yr'),
                          ('annual_runoff_mm_yr', 'runoff_mm_yr'),
                          ('recharge_fraction', 'recharge_fraction'),
                          ('water_table_depth_m', 'water_table_m')]:
            slope[name], r2[name] = fit_vs_elev(key)

        r_elev = corr('annual_recharge_mm_yr', elevs)
        r_prcp = corr('annual_recharge_mm_yr', precip)

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

        group.sort(key=lambda r: -(r['metrics'].get('annual_recharge_mm_yr') or 0))
        rows = [{
            'case_name':         r['case_name'],
            'texture_top':       r['soil'].get('texture_top'),
            'clay_max_pct':      r['soil'].get('clay_max_pct'),
            'ksat_min_ums':      r['soil'].get('ksat_min_ums'),
            'recharge_mm_yr':    r['metrics'].get('annual_recharge_mm_yr'),
            'runoff_mm_yr':      r['metrics'].get('annual_runoff_mm_yr'),
            'recharge_fraction': r['metrics'].get('recharge_fraction'),
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
            'recharge_vs_clay_max': corr('clay_max_pct', 'annual_recharge_mm_yr'),
            'recharge_vs_ksat_min': corr('ksat_min_ums', 'annual_recharge_mm_yr'),
            'runoff_vs_clay_max':   corr('clay_max_pct', 'annual_runoff_mm_yr'),
            'runoff_vs_ksat_min':   corr('ksat_min_ums', 'annual_runoff_mm_yr'),
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
        'runoff':            lambda m: m.get('annual_runoff_mm_yr'),
        'infiltration':      lambda m: (m.get('water_budget') or {}).get('infiltration_mm_yr'),
        'et':                lambda m: (m.get('water_budget') or {}).get('et_mm_yr'),
        'recharge':          lambda m: m.get('annual_recharge_mm_yr'),
        'recharge_fraction': lambda m: m.get('recharge_fraction'),
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