#!/usr/bin/env python3
"""
ELM Results Analyzer
src/core/elm_results_analyzer.py

Single responsibility: read ELM NetCDF history files
and extract hydrological variables into a standard dict.

Phase A note:
- plot_all() saves figures directly into analysis_dir (not a "plots/"
  subdir) so they land in 04_analysis/ next to hydro_summary.json
- the cross-experiment plot is named comparison_all_times.png
  to mirror PFLOTRAN's convention (consumed by create_slides.py)
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
TARGET_VARIABLES = ['QOVER', 'QCHARGE', 'TWS', 'SOILLIQ', 'ZWT', 'RAIN']

VARIABLE_UNITS = {
    'QOVER':   'mm/s',
    'QCHARGE': 'mm/s',
    'TWS':     'mm',
    'SOILLIQ': 'kg/m2',
    'ZWT':     'm',
    'RAIN':    'mm/s',   # atmospheric forcing (rainfall flux) — surfaces the input
}

S_TO_YEAR = 86400.0 * 365.25


# ─────────────────────────────────────────────────────────────────────
# ELM RESULTS ANALYZER
# ─────────────────────────────────────────────────────────────────────
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
                 analysis_dir: str):
        if not XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray and netCDF4 required.\n"
                "Run: pip install netcdf4"
            )
        self.experiments  = experiments
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
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
            'experiments':     list(self.results.values()),
            'comparisons':     self._compute_comparisons(),
            'spatial_summary': self._compute_spatial_summary(),
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

    def plot_all(self,
                 skip_plotting: bool = False) -> list:
        """
        Generate plots for all experiments.

        Phase A change: plots go directly into self.analysis_dir
        (no "plots/" subdir), and the cross-experiment plot is
        renamed to comparison_all_times.png so it matches PFLOTRAN's
        convention (consumed by create_slides.py).
        """
        if skip_plotting:
            return []
        try:
            from core.elm_plotting import (
                plot_all_experiment_figures,
                compare_forcing_periods,
            )
        except ImportError:
            print("⚠️  elm_plotting not available — skipping")
            return []

        plot_objects = []

        for case_name, result in self.results.items():
            if result.get('status') != 'ok':
                continue
            try:
                _, plot_obj = plot_all_experiment_figures(
                    result     = result,
                    output_dir = str(self.analysis_dir),
                    prefix     = f"{case_name}_",
                )
                plot_objects.append(plot_obj)
                print(f"   ✓ Plot: {case_name}")
            except Exception as e:
                print(f"   ⚠️  Plot failed {case_name}: {e}")

        if len(plot_objects) > 1:
            try:
                compare_forcing_periods(
                    plot_objects,
                    save_path = str(
                        self.analysis_dir / "comparison_all_times.png"
                    ),
                )
                print(f"   ✓ Comparison plot saved")
            except Exception as e:
                print(f"   ⚠️  Comparison failed: {e}")

        print(f"✓ {len(plot_objects)} ELM plot(s) generated")
        return plot_objects

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
            ds        = self._open_dataset(hist_files)
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
                'status':         'ok',
                'variables':      variables,
                'metrics':        metrics,
                'history_files':  [str(f) for f in hist_files],
            }

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            print(f"   ✗ Extraction failed: {e}")
            return self._empty_result(exp, str(e))

    def _summarize(self,
                   da:       "xr.DataArray",
                   var_name: str) -> Dict[str, Any]:
        """Compute summary statistics for one variable."""
        da  = da.squeeze()
        val = np.array(da.values, dtype=float)

        if var_name in ('QOVER', 'QCHARGE', 'RAIN'):
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
        if rn:
            metrics['precip_mm_yr'] = rn.get('annual_mean')   # forcing input per column
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

    def _save_hydro_summary(self):
        """Save hydro_summary.json."""
        hydro_file = self.analysis_dir / "hydro_summary.json"
        with open(hydro_file, 'w') as f:
            json.dump(
                {
                    'experiments':     list(self.results.values()),
                    'comparisons':     self._compute_comparisons(),
                    'spatial_summary': self._compute_spatial_summary(),
                    'units':           VARIABLE_UNITS,
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
            'status':         'failed',
            'reason':         reason,
            'variables':      {v: None for v in TARGET_VARIABLES},
            'metrics':        {},
            'history_files':  [],
        }