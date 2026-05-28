#!/usr/bin/env python3
"""
ELM Plotting Module
src/core/elm_plotting.py

Mirrors pflotran_plotting.py pattern exactly.
Provides diagnostic figures for ELM standalone experiments.

Main functions:
    ELMPlotter.plot_experiment_overview(result, output_dir)
        → 2-panel: hydro time series + soil moisture profile

    compare_experiments(plot_objects)
        → side-by-side annual metrics across experiments

    compare_forcing_periods(plot_objects)
        → overlay QCHARGE time series across forcing periods

    plot_all_experiment_figures(result, output_dir, prefix)
        → convenience function, generates all plots
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Dict, Any

# mm/s → mm/year
S_TO_YEAR = 86400.0 * 365.25

# ─────────────────────────────────────────────────────────────────────
# DATA CONTAINER — mirrors ExperimentPlot
# ─────────────────────────────────────────────────────────────────────
class ELMExperimentPlot:
    """
    Container for ELM experiment plot data.
    Used for comparison plots across experiments.
    Mirrors ExperimentPlot in pflotran_plotting.py.
    """
    def __init__(self,
                 case_name:      str,
                 forcing_period: str,
                 forcing_start:  int,
                 forcing_end:    int,
                 variables:      Dict[str, Any],
                 metrics:        Dict[str, Any]):
        self.case_name      = case_name
        self.forcing_period = forcing_period
        self.forcing_start  = forcing_start
        self.forcing_end    = forcing_end
        self.variables      = variables   # extracted from NetCDF
        self.metrics        = metrics     # annual totals


# ─────────────────────────────────────────────────────────────────────
# PLOTTER — mirrors PFLOTRANPlotter
# ─────────────────────────────────────────────────────────────────────
class ELMPlotter:
    """
    Plotting utilities for ELM experiments.
    Mirrors PFLOTRANPlotter interface.
    """

    @staticmethod
    def plot_experiment_overview(result:     Dict[str, Any],
                                  output_dir: str = "./",
                                  prefix:     str = "",
                                  save_path:  Optional[str] = None,
                                  show:       bool = False
                                  ) -> "ELMExperimentPlot":
        """
        Plot overview of one ELM experiment.
        2-panel: hydro time series + soil moisture profile.

        Mirrors PFLOTRANPlotter.plot_experiment_overview().

        Args:
            result:     experiment result dict from ELMResultsAnalyzer
            output_dir: directory to save figure
            prefix:     filename prefix
            save_path:  explicit save path (overrides output_dir)
            show:       display plot interactively

        Returns:
            ELMExperimentPlot for comparison plots
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle(
            f"ELM Experiment: {result.get('scenario_name', result.get('case_name', ''))}",
            fontsize = 16,
            fontweight = 'bold'
        )

        # Panel 1 — hydro time series
        ax1 = axes[0]
        ELMPlotter._plot_hydro_timeseries(ax1, result)

        # Panel 2 — soil moisture profile
        ax2 = axes[1]
        ELMPlotter._plot_soil_profile(ax2, result)

        plt.tight_layout()

        # Save
        if not save_path:
            os.makedirs(output_dir, exist_ok=True)
            case = result.get('case_name', 'elm')
            save_path = os.path.join(
                output_dir, f"{prefix}{case}_overview.png"
            )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ELM overview saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return ELMPlotter._create_plot_object(result)

    # ─────────────────────────────────────────────────────────
    # PANEL 1 — Hydro time series
    # mirrors _plot_layer_profile()
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _plot_hydro_timeseries(ax, result: Dict[str, Any]):
        """
        Plot QCHARGE + QOVER annual means as bar chart.
        Mirrors _plot_layer_profile() role in PFLOTRAN.
        """
        variables = result.get('variables', {})
        metrics   = result.get('metrics',   {})

        qcharge = variables.get('QCHARGE') or {}
        qover   = variables.get('QOVER')   or {}

        # Annual means
        qc_mean = qcharge.get('annual_mean', 0) or 0
        qo_mean = qover.get('annual_mean',   0) or 0
        tws_mean = (variables.get('TWS') or {}).get('mean_mm', 0) or 0

        labels = ['QCHARGE\n(Recharge)', 'QOVER\n(Runoff)', 'TWS\n(Storage)']
        values = [qc_mean, qo_mean, tws_mean / 100]  # TWS scaled
        colors = ['steelblue', 'coral', 'lightgreen']
        units  = ['mm/yr', 'mm/yr', 'mm/100']

        bars = ax.bar(
            range(len(labels)), values,
            color      = colors,
            edgecolor  = 'black',
            linewidth  = 1.2,
            alpha      = 0.85,
            width      = 0.5,
        )

        # Value labels on bars
        for i, (bar, val, unit) in enumerate(
            zip(bars, values, units)
        ):
            ax.text(
                i, val + abs(val) * 0.03,
                f"{val:.2f}\n{unit}",
                ha         = 'center',
                va         = 'bottom',
                fontsize   = 9,
                fontweight = 'bold',
            )

        # Recharge/runoff ratio annotation
        ratio = metrics.get('recharge_to_runoff_ratio')
        if ratio is not None:
            ax.text(
                0.98, 0.95,
                f"Recharge/Runoff\nratio = {ratio:.2f}",
                transform  = ax.transAxes,
                ha         = 'right',
                va         = 'top',
                fontsize   = 10,
                fontweight = 'bold',
                bbox       = dict(
                    boxstyle  = 'round',
                    facecolor = 'lightyellow',
                    alpha     = 0.8,
                )
            )

        # Forcing period annotation
        period = result.get('forcing_period', 'N/A')
        yr_s   = result.get('forcing_start',  'N/A')
        yr_e   = result.get('forcing_end',    'N/A')
        ax.text(
            0.02, 0.95,
            f"Period: {period}\n{yr_s}–{yr_e}",
            transform  = ax.transAxes,
            ha         = 'left',
            va         = 'top',
            fontsize   = 9,
            bbox       = dict(
                boxstyle  = 'round',
                facecolor = 'lightblue',
                alpha     = 0.7,
            )
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Annual Mean', fontsize=12, fontweight='bold')
        ax.set_title(
            'Hydrological Variables',
            fontsize   = 13,
            fontweight = 'bold',
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

    # ─────────────────────────────────────────────────────────
    # PANEL 2 — Soil moisture profile
    # mirrors _plot_flux_recharge()
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _plot_soil_profile(ax, result: Dict[str, Any]):
        """
        Plot SOILLIQ profile as horizontal bar chart.
        Mirrors _plot_flux_recharge() role in PFLOTRAN.
        """
        variables = result.get('variables', {})
        soilliq   = variables.get('SOILLIQ') or {}
        layers    = soilliq.get('layer_means_kg_m2', [])

        if not layers:
            ax.text(
                0.5, 0.5,
                'SOILLIQ not available',
                ha        = 'center',
                va        = 'center',
                fontsize  = 12,
                transform = ax.transAxes,
                bbox      = dict(
                    boxstyle  = 'round',
                    facecolor = 'wheat',
                    alpha     = 0.5,
                )
            )
            ax.set_title(
                'Soil Moisture Profile',
                fontsize   = 13,
                fontweight = 'bold',
            )
            ax.axis('off')
            return

        n_layers = len(layers)
        y_pos    = np.arange(n_layers)

        # Color by moisture content
        norm   = plt.Normalize(vmin=min(layers), vmax=max(layers))
        cmap   = plt.cm.get_cmap('Blues')
        colors = [cmap(norm(v)) for v in layers]

        bars = ax.barh(
            y_pos, layers,
            color     = colors,
            edgecolor = 'black',
            linewidth = 0.8,
            alpha     = 0.85,
        )

        # Layer labels
        for i, (bar, val) in enumerate(zip(bars, layers)):
            ax.text(
                val + max(layers) * 0.01,
                i,
                f"{val:.1f}",
                va       = 'center',
                fontsize = 8,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"Layer {i+1}" for i in range(n_layers)],
            fontsize = 8,
        )
        ax.set_xlabel('Mean SOILLIQ (kg/m²)', fontsize=12,
                      fontweight='bold')
        ax.set_title(
            'Soil Moisture Profile',
            fontsize   = 13,
            fontweight = 'bold',
        )
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Total column annotation
        total = soilliq.get('total_column_kg_m2', sum(layers))
        ax.text(
            0.98, 0.02,
            f"Total column:\n{total:.1f} kg/m²\n"
            f"({n_layers} layers)",
            transform  = ax.transAxes,
            ha         = 'right',
            va         = 'bottom',
            fontsize   = 9,
            bbox       = dict(
                boxstyle  = 'round',
                facecolor = 'lightblue',
                alpha     = 0.7,
            )
        )

    # ─────────────────────────────────────────────────────────
    # CREATE PLOT OBJECT — mirrors _create_plot_object()
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _create_plot_object(result: Dict[str, Any]
                             ) -> ELMExperimentPlot:
        """Create ELMExperimentPlot from result dict."""
        return ELMExperimentPlot(
            case_name      = result.get('case_name',      'unknown'),
            forcing_period = result.get('forcing_period', 'unknown'),
            forcing_start  = result.get('forcing_start',  0),
            forcing_end    = result.get('forcing_end',    0),
            variables      = result.get('variables',      {}),
            metrics        = result.get('metrics',        {}),
        )


# ─────────────────────────────────────────────────────────────────────
# COMPARISON — mirrors compare_experiments()
# ─────────────────────────────────────────────────────────────────────
def compare_experiments(plot_objects: List[ELMExperimentPlot],
                         save_path:   Optional[str] = None,
                         show:        bool = False):
    """
    Side-by-side annual metrics comparison.
    Mirrors compare_experiments() in pflotran_plotting.py.

    Args:
        plot_objects: list of ELMExperimentPlot
        save_path:    path to save figure
        show:         display interactively

    Returns:
        fig, axes
    """
    if not plot_objects:
        print("No experiments to compare")
        return None, None

    n_exp   = len(plot_objects)
    n_cols  = min(3, n_exp)
    n_rows  = (n_exp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize = (5 * n_cols, 5 * n_rows)
    )
    fig.suptitle(
        'ELM Experiment Comparison',
        fontsize   = 16,
        fontweight = 'bold',
    )

    # Flatten axes
    if n_exp == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten().tolist()

    for idx, plot_obj in enumerate(plot_objects):
        ax = axes_flat[idx]

        # Bar chart of annual metrics
        qc   = (plot_obj.variables.get('QCHARGE') or {})
        qo   = (plot_obj.variables.get('QOVER')   or {})
        qc_m = qc.get('annual_mean', 0) or 0
        qo_m = qo.get('annual_mean', 0) or 0

        labels = ['QCHARGE', 'QOVER']
        values = [qc_m, qo_m]
        colors = ['steelblue', 'coral']

        bars = ax.bar(
            range(len(labels)), values,
            color     = colors,
            edgecolor = 'black',
            alpha     = 0.85,
        )

        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                i, val + abs(val) * 0.03,
                f"{val:.2f}",
                ha         = 'center',
                va         = 'bottom',
                fontsize   = 9,
                fontweight = 'bold',
            )

        # Period label
        ax.text(
            0.5, 0.95,
            f"{plot_obj.forcing_period}\n"
            f"{plot_obj.forcing_start}–{plot_obj.forcing_end}",
            transform  = ax.transAxes,
            ha         = 'center',
            va         = 'top',
            fontsize   = 9,
            bbox       = dict(
                boxstyle  = 'round',
                facecolor = 'lightyellow',
                alpha     = 0.8,
            )
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('mm/year', fontsize=10)
        ax.set_title(plot_obj.case_name, fontsize=11,
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for idx in range(n_exp, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


# ─────────────────────────────────────────────────────────────────────
# FORCING PERIOD COMPARISON — mirrors compare_flux_conditions()
# ─────────────────────────────────────────────────────────────────────
def compare_forcing_periods(plot_objects: List[ELMExperimentPlot],
                             save_path:   Optional[str] = None,
                             show:        bool = False):
    """
    Compare annual QCHARGE + QOVER across forcing periods.
    Mirrors compare_flux_conditions() in pflotran_plotting.py.

    Args:
        plot_objects: list of ELMExperimentPlot
        save_path:    path to save figure
        show:         display interactively

    Returns:
        fig, ax
    """
    if not plot_objects:
        print("No experiments to compare")
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'ELM Forcing Period Comparison',
        fontsize   = 14,
        fontweight = 'bold',
    )

    periods = [p.forcing_period for p in plot_objects]
    colors  = plt.cm.tab10(np.linspace(0, 1, len(plot_objects)))

    # Panel 1 — QCHARGE
    ax1     = axes[0]
    qc_vals = [
        (p.variables.get('QCHARGE') or {}).get('annual_mean', 0) or 0
        for p in plot_objects
    ]
    bars = ax1.bar(
        range(len(periods)), qc_vals,
        color     = colors,
        edgecolor = 'black',
        alpha     = 0.85,
    )
    for i, (bar, val) in enumerate(zip(bars, qc_vals)):
        ax1.text(
            i, val + abs(val) * 0.03,
            f"{val:.2f}",
            ha         = 'center',
            va         = 'bottom',
            fontsize   = 10,
            fontweight = 'bold',
        )
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, fontsize=11)
    ax1.set_ylabel('mm/year', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Annual Recharge (QCHARGE)',
        fontsize   = 13,
        fontweight = 'bold',
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # Panel 2 — QOVER
    ax2     = axes[1]
    qo_vals = [
        (p.variables.get('QOVER') or {}).get('annual_mean', 0) or 0
        for p in plot_objects
    ]
    bars = ax2.bar(
        range(len(periods)), qo_vals,
        color     = colors,
        edgecolor = 'black',
        alpha     = 0.85,
    )
    for i, (bar, val) in enumerate(zip(bars, qo_vals)):
        ax2.text(
            i, val + abs(val) * 0.03,
            f"{val:.2f}",
            ha         = 'center',
            va         = 'bottom',
            fontsize   = 10,
            fontweight = 'bold',
        )
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, fontsize=11)
    ax2.set_ylabel('mm/year', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Annual Runoff (QOVER)',
        fontsize   = 13,
        fontweight = 'bold',
    )
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Forcing comparison saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE — mirrors plot_all_experiment_figures()
# ─────────────────────────────────────────────────────────────────────
def plot_all_experiment_figures(result:     Dict[str, Any],
                                 output_dir: str = "./",
                                 prefix:     str = ""):
    """
    Generate all plots for one ELM experiment.
    Mirrors plot_all_experiment_figures() in pflotran_plotting.py.

    Args:
        result:     experiment result dict from ELMResultsAnalyzer
        output_dir: directory to save figures
        prefix:     filename prefix

    Returns:
        figures (dict of paths), plot_obj (ELMExperimentPlot)
    """
    os.makedirs(output_dir, exist_ok=True)
    figures  = {}
    case     = result.get('case_name', 'elm')

    # Overview (2 panels)
    overview_path = os.path.join(
        output_dir, f"{prefix}{case}_overview.png"
    )
    plot_obj = ELMPlotter.plot_experiment_overview(
        result     = result,
        save_path  = overview_path,
    )
    figures['overview'] = overview_path

    print(f"✓ Generated ELM overview: {case}")
    return figures, plot_obj


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ELM Plotting Module")
    print("=" * 50)
    print("\nMain functions:")
    print("  ELMPlotter.plot_experiment_overview(result)")
    print("  → 2-panel: hydro metrics + soil profile")
    print("\n  compare_experiments(plot_objects)")
    print("  → side-by-side annual metrics")
    print("\n  compare_forcing_periods(plot_objects)")
    print("  → QCHARGE + QOVER across dry/wet/baseline")
    print("\n  plot_all_experiment_figures(result, output_dir)")
    print("  → all plots for one experiment")