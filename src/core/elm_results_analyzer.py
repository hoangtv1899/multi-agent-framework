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
TARGET_VARIABLES = ['QOVER', 'QCHARGE', 'TWS', 'SOILLIQ']

VARIABLE_UNITS = {
    'QOVER':   'mm/s',
    'QCHARGE': 'mm/s',
    'TWS':     'mm',
    'SOILLIQ': 'kg/m2',
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
            'units':           VARIABLE_UNITS,
            'focus_variables': {
                'QCHARGE': 'Primary — aquifer recharge',
                'QOVER':   'Surface runoff',
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
        return xr.open_mfdataset(
            [str(f) for f in hist_files],
            combine      = 'by_coords',
            decode_times = True,
            engine       = 'netcdf4',
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

        if var_name in ('QOVER', 'QCHARGE'):
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
        """Compute derived metrics."""
        metrics = {}
        qc = variables.get('QCHARGE') or {}
        qo = variables.get('QOVER')   or {}
        tw = variables.get('TWS')     or {}

        if qc:
            metrics['annual_recharge_mm_yr'] = (
                qc.get('annual_mean')
            )
        if qo:
            metrics['annual_runoff_mm_yr'] = (
                qo.get('annual_mean')
            )
        if qc and qo:
            qc_m = qc.get('annual_mean', 0) or 0
            qo_m = qo.get('annual_mean', 0) or 0
            if abs(qo_m) > 1e-10:
                metrics['recharge_to_runoff_ratio'] = round(
                    qc_m / qo_m, 4
                )
        if tw:
            metrics['tws_seasonal_range_mm'] = (
                tw.get('seasonal_range')
            )
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

    def _save_hydro_summary(self):
        """Save hydro_summary.json."""
        hydro_file = self.analysis_dir / "hydro_summary.json"
        with open(hydro_file, 'w') as f:
            json.dump(
                {
                    'experiments': list(self.results.values()),
                    'comparisons': self._compute_comparisons(),
                    'units':       VARIABLE_UNITS,
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
            'status':         'failed',
            'reason':         reason,
            'variables':      {v: None for v in TARGET_VARIABLES},
            'metrics':        {},
            'history_files':  [],
        }