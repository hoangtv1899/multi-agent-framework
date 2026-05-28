#!/usr/bin/env python3
"""
Replot a completed ELM workflow run — no rerun needed.

Self-contained plotting tool. Auto-discovers case dirs on PSCRATCH and
reads fsurdat directly from each case's user_nl_elm, so it works even
when experiment_summary.json is missing those fields.

USAGE
─────
    python3 replot.py <run_dir>                  # both setup + analysis
    python3 replot.py <run_dir> --setup-only
    python3 replot.py <run_dir> --analysis-only
"""
import argparse
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module='xarray.coding.times')
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
PSCRATCH_ROOT = Path(
    os.environ.get('PSCRATCH', '/pscratch/sd/h/hvtran')
) / "E3SMv3"

PERIOD_COLORS = {
    'baseline': '#3D7CB0',
    'dry':      '#C0392B',
    'wet':      '#27AE60',
}

# ELM standard soil layer node depths (m) — from clm_varpar.F90
ELM_LEVEL_NODE_DEPTH_M = np.array([
    0.0175, 0.0451, 0.0906, 0.1656, 0.2891,
    0.4929, 0.8289, 1.3828, 2.2961, 3.4332,
])

CASE_TS_RE   = re.compile(r'1D_ELM\.[^.]+\.(\d{4}-\d{2}-\d{2}-\d{6})\.(.+)$')
RUN_TS_RE    = re.compile(r'(\d{8})_(\d{6})')
FSURDAT_RE   = re.compile(
    r'^\s*fsurdat\s*=\s*[\'"]?([^\'"\s]+)[\'"]?', re.IGNORECASE,
)


# ═════════════════════════════════════════════════════════════════════
#   APPROACH A — AUTO-DISCOVER CASE DIRS FROM PSCRATCH
# ═════════════════════════════════════════════════════════════════════
def _workflow_timestamp(run_dir):
    m = RUN_TS_RE.search(run_dir.name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def _case_timestamp(case_dir_name):
    m = CASE_TS_RE.match(case_dir_name)
    if not m:
        return None, None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H%M%S"), m.group(2)
    except ValueError:
        return None, None


def discover_case_dir(case_name, run_dir, max_delta_seconds=3600):
    if not PSCRATCH_ROOT.exists():
        logger.warning(f"PSCRATCH root not found: {PSCRATCH_ROOT}")
        return None
    run_dt = _workflow_timestamp(run_dir)
    if run_dt is None:
        return None
    candidates = list(PSCRATCH_ROOT.glob(f"1D_ELM.*.{case_name}"))
    if not candidates:
        return None
    best, best_delta = None, None
    for c in candidates:
        case_dt, _ = _case_timestamp(c.name)
        if case_dt is None:
            continue
        delta = abs((case_dt - run_dt).total_seconds())
        if delta <= max_delta_seconds and (best_delta is None or delta < best_delta):
            best, best_delta = c, delta
    return best


def _history_files(case_dir):
    if case_dir is None:
        return []
    run_subdir = case_dir / "run"
    if not run_subdir.exists():
        return []
    files = sorted(run_subdir.glob("*.elm.h0.*.nc"))
    return [f for f in files if not f.name.endswith('-00000.nc')]


# ═════════════════════════════════════════════════════════════════════
#   DATA HELPERS
# ═════════════════════════════════════════════════════════════════════
def _open_history(files):
    """Open ELM history files. Converts cftime → datetime64 for matplotlib."""
    if not files:
        return None
    try:
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            coords='minimal',
            compat='override',
        )
        # ELM uses no-leap calendar → cftime.DatetimeNoLeap, which matplotlib
        # can't plot. Convert to pandas DatetimeIndex (Feb 29 just gets skipped).
        try:
            t_idx = ds.indexes.get('time', None)
            if t_idx is not None and hasattr(t_idx, 'to_datetimeindex'):
                ds = ds.assign_coords(time=t_idx.to_datetimeindex())
        except Exception as e:
            logger.warning(f"Could not convert cftime to datetime: {e}")
        return ds
    except Exception as e:
        logger.warning(f"Cannot open history files: {e}")
        return None


def _get_fsurdat(exp, elm_config):
    """Find the surface file path. Tries 3 sources in order:
    1. runtime_config['FSURDAT']
    2. elm_config['FSURDAT']
    3. Parse from <case_dir>/user_nl_elm
    """
    fsurdat = exp.get('runtime_config', {}).get('FSURDAT')
    if fsurdat:
        return fsurdat
    fsurdat = elm_config.get('FSURDAT')
    if fsurdat:
        return fsurdat
    # Read user_nl_elm directly from the case dir
    case_dir = exp.get('case_dir')
    if case_dir:
        ul_path = Path(case_dir) / 'user_nl_elm'
        if ul_path.exists():
            for line in ul_path.read_text().splitlines():
                m = FSURDAT_RE.match(line)
                if m:
                    return m.group(1)
    return None


def _read_surface_profile(fsurdat_path):
    try:
        ds = xr.open_dataset(fsurdat_path)
        sand = np.asarray(ds['PCT_SAND'].squeeze().values, dtype=float)
        clay = np.asarray(ds['PCT_CLAY'].squeeze().values, dtype=float)
        organic = (
            np.asarray(ds['ORGANIC'].squeeze().values, dtype=float)
            if 'ORGANIC' in ds else None
        )
        ds.close()
        return sand, clay, organic
    except Exception as e:
        logger.warning(f"Cannot read surface file {fsurdat_path}: {e}")
        return None, None, None


def _monthly_precip(ds):
    if ds is None or 'RAIN' not in ds:
        return None
    rain = ds['RAIN'].squeeze()
    daily = (rain * 86400).resample(time='D').mean()
    monthly = daily.resample(time='MS').sum()
    return monthly.time.values, monthly.values


def _flux_to_mm_day(da):
    return da.squeeze() * 86400.0


# ═════════════════════════════════════════════════════════════════════
#   SETUP PLOTS
# ═════════════════════════════════════════════════════════════════════
def plot_domain_configuration(exp, elm_config, history_files, output_path):
    fig, (ax_soil, ax_precip) = plt.subplots(1, 2, figsize=(14, 7))

    case_name = exp.get('case_name', '?')
    period    = str(exp.get('forcing_period', '?'))
    yr_start  = exp.get('forcing_start', '?')
    yr_end    = exp.get('forcing_end', '?')
    lat       = exp.get('latitude') or elm_config.get('LATITUDE', '?')
    lon       = exp.get('longitude') or elm_config.get('LONGITUDE', '?')

    fig.suptitle(
        f"Domain Configuration — {case_name}\n"
        f"({lat}°N, {lon}°E, {period.capitalize()} forcing, "
        f"{yr_start}–{yr_end})",
        fontsize=13, fontweight='bold', y=0.99,
    )

    # ── LEFT: soil profile ──
    fsurdat = _get_fsurdat(exp, elm_config)
    if fsurdat and Path(fsurdat).exists():
        sand, clay, organic = _read_surface_profile(fsurdat)
        if sand is not None:
            n = min(len(sand), len(ELM_LEVEL_NODE_DEPTH_M))
            sand, clay = sand[:n], clay[:n]
            silt = np.clip(100.0 - sand - clay, 0, 100)
            depths = ELM_LEVEL_NODE_DEPTH_M[:n]
            y_pos = np.arange(n)

            ax_soil.barh(y_pos, sand, color='#F4C430', label='Sand',
                         edgecolor='white', linewidth=0.5)
            ax_soil.barh(y_pos, silt, left=sand, color='#C9A66B',
                         label='Silt', edgecolor='white', linewidth=0.5)
            ax_soil.barh(y_pos, clay, left=sand+silt, color='#7B4F2C',
                         label='Clay', edgecolor='white', linewidth=0.5)

            ax_soil.set_yticks(y_pos)
            ax_soil.set_yticklabels(
                [f'L{i+1}  ({d:.2f}m)' for i, d in enumerate(depths)],
                fontsize=9,
            )
            ax_soil.invert_yaxis()
            ax_soil.set_xlabel('Texture fraction (%)')
            ax_soil.set_title('Soil Texture Profile (from surface file)',
                              fontsize=11)
            ax_soil.legend(loc='lower right', fontsize=9, framealpha=0.9)
            ax_soil.set_xlim(0, 100)
            ax_soil.grid(True, alpha=0.3, axis='x')

            for i, s in enumerate(sand):
                if s > 8:
                    ax_soil.text(s/2, i, f'{s:.0f}', va='center',
                                 ha='center', fontsize=8, color='black')
                if clay[i] > 8:
                    ax_soil.text(sand[i]+silt[i]+clay[i]/2, i,
                                 f'{clay[i]:.0f}', va='center',
                                 ha='center', fontsize=8, color='white')

            if organic is not None and len(organic) > 0:
                ax_soil.text(
                    0.02, 0.02,
                    f'Top-layer organic: {organic[0]:.1f} kg/m³',
                    transform=ax_soil.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.85,
                              edgecolor='gray'),
                )
        else:
            ax_soil.text(0.5, 0.5, 'Cannot read surface file',
                         ha='center', va='center',
                         transform=ax_soil.transAxes)
    else:
        path_info = f"\n{fsurdat}" if fsurdat else ""
        ax_soil.text(0.5, 0.5, f'Surface file not available{path_info}',
                     ha='center', va='center', fontsize=10,
                     transform=ax_soil.transAxes)

    # ── RIGHT: monthly precip ──
    ds = _open_history(history_files)
    result = _monthly_precip(ds) if ds is not None else None
    if result is not None:
        times, monthly = result
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        # times are datetime64 after conversion
        labels = []
        for t in times:
            try:
                labels.append(month_labels[int(str(np.datetime64(t, 'M'))[-2:])-1])
            except Exception:
                labels.append('?')
        color = PERIOD_COLORS.get(period.lower(), 'steelblue')

        bars = ax_precip.bar(range(len(labels)), monthly, color=color,
                             edgecolor='white', linewidth=0.8, alpha=0.85)
        ax_precip.set_xticks(range(len(labels)))
        ax_precip.set_xticklabels(labels, fontsize=10)
        ax_precip.set_ylabel('Monthly precipitation (mm)')
        ax_precip.set_title(f'Forcing — Monthly RAIN ({yr_start})',
                            fontsize=11)
        ax_precip.grid(True, alpha=0.3, axis='y')

        annual = float(monthly.sum())
        peak_idx = int(np.argmax(monthly))
        info = (f'Annual total: {annual:.0f} mm\n'
                f'Wettest month: {labels[peak_idx]} ({monthly[peak_idx]:.0f} mm)')
        ax_precip.text(
            0.98, 0.96, info,
            transform=ax_precip.transAxes, ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                      boxstyle='round,pad=0.5'),
        )
        for bar, v in zip(bars, monthly):
            if v > 0.5:
                ax_precip.text(bar.get_x()+bar.get_width()/2, v, f'{v:.0f}',
                               ha='center', va='bottom', fontsize=8)
    else:
        ax_precip.text(0.5, 0.5, 'No RAIN data in history files',
                       ha='center', va='center',
                       transform=ax_precip.transAxes)
    if ds is not None:
        ds.close()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def compare_forcing_conditions(experiment_runs, output_path):
    fig, (ax_monthly, ax_annual) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={'height_ratios': [2, 1]},
    )
    fig.suptitle('Forcing Conditions — Annual Precipitation Cycle',
                 fontsize=14, fontweight='bold', y=0.995)

    annual_totals = {}
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    for exp, history_files in experiment_runs:
        ds = _open_history(history_files)
        result = _monthly_precip(ds) if ds is not None else None
        if result is None:
            if ds is not None:
                ds.close()
            continue
        _, monthly = result
        n = min(len(monthly), 12)
        case_name = exp.get('case_name', '?')
        period = str(exp.get('forcing_period', '?')).lower()
        year   = exp.get('forcing_start', '?')
        color  = PERIOD_COLORS.get(period, 'gray')

        ax_monthly.plot(np.arange(n), monthly[:n],
                        color=color, linewidth=2.5, marker='o', markersize=8,
                        label=f"{case_name}  ({year}, {period})")
        ax_monthly.fill_between(np.arange(n), 0, monthly[:n],
                                color=color, alpha=0.15)
        annual_totals[case_name] = (float(monthly[:n].sum()), color, period, year)
        ds.close()

    ax_monthly.set_xticks(range(12))
    ax_monthly.set_xticklabels(month_labels)
    ax_monthly.set_ylabel('Monthly precipitation (mm)')
    ax_monthly.set_title('Monthly cycle')
    if annual_totals:
        ax_monthly.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_monthly.grid(True, alpha=0.3)

    if annual_totals:
        names  = list(annual_totals.keys())
        totals = [v[0] for v in annual_totals.values()]
        colors = [v[1] for v in annual_totals.values()]
        years  = [v[3] for v in annual_totals.values()]
        x = np.arange(len(names))
        bars = ax_annual.bar(x, totals, color=colors, edgecolor='white',
                             linewidth=0.8, alpha=0.85)
        ax_annual.set_xticks(x)
        ax_annual.set_xticklabels([f"{n}\n({y})" for n, y in zip(names, years)],
                                  fontsize=10)
        ax_annual.set_ylabel('Annual total (mm)')
        ax_annual.set_title('Annual precipitation totals')
        ax_annual.grid(True, alpha=0.3, axis='y')

        mean_total = float(np.mean(totals))
        ax_annual.axhline(mean_total, color='gray', linestyle='--',
                          linewidth=1.5, label=f'Mean: {mean_total:.0f} mm')
        ax_annual.legend(loc='upper right', fontsize=10)
        for bar, t in zip(bars, totals):
            ax_annual.text(bar.get_x()+bar.get_width()/2, t, f'{t:.0f}',
                           ha='center', va='bottom', fontsize=11,
                           fontweight='bold')
    else:
        ax_annual.text(0.5, 0.5, 'No precipitation data available',
                       ha='center', va='center', transform=ax_annual.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


# ═════════════════════════════════════════════════════════════════════
#   ANALYSIS PLOTS
# ═════════════════════════════════════════════════════════════════════
def plot_experiment_timeseries(exp, ds, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    case_name = exp.get('case_name', '?')
    period    = str(exp.get('forcing_period', '?')).capitalize()
    yr        = f"{exp.get('forcing_start','?')}–{exp.get('forcing_end','?')}"
    fig.suptitle(f"{case_name}  —  {period}  ({yr})",
                 fontsize=14, fontweight='bold', y=0.995)

    pcolor = PERIOD_COLORS.get(str(exp.get('forcing_period', '')).lower(),
                                'steelblue')

    # ── (0,0) RAIN bars + QCHARGE line on twin axis ──
    ax = axes[0, 0]
    if 'RAIN' in ds and 'QCHARGE' in ds:
        rain = _flux_to_mm_day(ds['RAIN']).resample(time='D').mean()
        qchg = _flux_to_mm_day(ds['QCHARGE']).resample(time='D').mean()
        # Use width=1.0 day for bar; ax.bar with datetime64 needs explicit float width
        ax.bar(rain.time.values, rain.values, width=1.0,
               color='#7AB8E6', alpha=0.6, edgecolor='none')
        ax.set_ylabel('RAIN (mm/day)', color='#3F7CB0')
        ax.tick_params(axis='y', labelcolor='#3F7CB0')
        ax2 = ax.twinx()
        ax2.plot(qchg.time.values, qchg.values, color=pcolor, linewidth=1.4)
        ax2.set_ylabel('QCHARGE (mm/day)', color=pcolor)
        ax2.tick_params(axis='y', labelcolor=pcolor)
        ax.set_title('Forcing & Recharge Response')
        ax.grid(True, alpha=0.3)

    # ── (0,1) QOVER + QDRAI ──
    ax = axes[0, 1]
    if 'QOVER' in ds:
        q = _flux_to_mm_day(ds['QOVER']).resample(time='D').mean()
        ax.plot(q.time.values, q.values, color='#E67E22', linewidth=1.4,
                label='QOVER (surface runoff)')
        ax.fill_between(q.time.values, 0, q.values, color='#E67E22',
                        alpha=0.25)
    if 'QDRAI' in ds:
        q = _flux_to_mm_day(ds['QDRAI']).resample(time='D').mean()
        ax.plot(q.time.values, q.values, color='#8B4513', linewidth=1.4,
                label='QDRAI (drainage)')
    ax.set_ylabel('Flux (mm/day)')
    ax.set_title('Runoff & Drainage')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # ── (1,0) Cumulative water balance ──
    ax = axes[1, 0]
    if 'RAIN' in ds:
        c = _flux_to_mm_day(ds['RAIN']).resample(time='D').mean().cumsum()
        ax.plot(c.time.values, c.values, color='#3F7CB0', linewidth=2.5,
                label='Cum RAIN (input)')
    if 'QCHARGE' in ds:
        c = _flux_to_mm_day(ds['QCHARGE']).resample(time='D').mean().cumsum()
        ax.plot(c.time.values, c.values, color=pcolor, linewidth=2,
                label='Cum QCHARGE')
    if 'QOVER' in ds:
        c = _flux_to_mm_day(ds['QOVER']).resample(time='D').mean().cumsum()
        ax.plot(c.time.values, c.values, color='#E67E22', linewidth=2,
                label='Cum QOVER')
    if 'QDRAI' in ds:
        c = _flux_to_mm_day(ds['QDRAI']).resample(time='D').mean().cumsum()
        ax.plot(c.time.values, c.values, color='#8B4513', linewidth=2,
                label='Cum QDRAI')
    ax.set_ylabel('Cumulative (mm)')
    ax.set_title('Cumulative Water Balance')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (1,1) TWS + ZWT ──
    ax = axes[1, 1]
    has_tws = 'TWS' in ds
    if has_tws:
        tws = ds['TWS'].squeeze().resample(time='D').mean()
        ax.plot(tws.time.values, tws.values, color='#27AE60',
                linewidth=2, label='TWS')
        ax.set_ylabel('TWS (mm)', color='#27AE60')
        ax.tick_params(axis='y', labelcolor='#27AE60')
    if 'ZWT' in ds:
        zwt = ds['ZWT'].squeeze().resample(time='D').mean()
        ax2 = ax.twinx() if has_tws else ax
        ax2.plot(zwt.time.values, zwt.values, color='#8E44AD',
                 linewidth=1.5, linestyle='--')
        ax2.set_ylabel('ZWT (m below surface)', color='#8E44AD')
        ax2.tick_params(axis='y', labelcolor='#8E44AD')
        ax2.invert_yaxis()
    ax.set_title('Storage & Water Table')
    ax.grid(True, alpha=0.3)

    # ── (2,0) H2OSOI heatmap (time × depth in meters, proper layer cells) ──
    ax = axes[2, 0]
    if 'H2OSOI' in ds:
        h2osoi = ds['H2OSOI'].squeeze()
        daily = h2osoi.resample(time='D').mean()
        if daily.ndim == 2:
            n_layers = min(10, daily.shape[-1])
            data = daily.values[:, :n_layers].T  # (depth, time)
            time_vals = mdates.date2num(daily.time.values)
            node_depths = ELM_LEVEL_NODE_DEPTH_M[:n_layers]

            # Layer interfaces (cell edges in depth) — gives realistic
            # layer thicknesses: thin near surface, thick at depth
            z_int = np.zeros(n_layers + 1)
            z_int[0] = 0.0
            for i in range(1, n_layers):
                z_int[i] = (node_depths[i-1] + node_depths[i]) / 2
            z_int[n_layers] = (
                node_depths[-1] + (node_depths[-1] - z_int[n_layers-1])
            )

            # Time cell edges (daily resolution)
            if len(time_vals) > 1:
                dt = np.diff(time_vals)
                t_int = np.concatenate([
                    [time_vals[0] - dt[0]/2],
                    (time_vals[:-1] + time_vals[1:]) / 2,
                    [time_vals[-1] + dt[-1]/2],
                ])
            else:
                t_int = np.array([time_vals[0] - 0.5, time_vals[0] + 0.5])

            im = ax.pcolormesh(
                t_int, z_int, data,
                cmap='YlGnBu', shading='flat',
                vmin=0, vmax=max(0.5, float(np.nanmax(data) * 1.05)),
            )
            ax.set_ylabel('Depth (m)')
            ax.invert_yaxis()
            ax.set_title('Soil Moisture (H2OSOI, m³/m³) — time × depth')
            ax.xaxis_date()
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label('m³/m³', fontsize=9)

    # ── (2,1) Final SOILLIQ profile (bars at actual depths, thickness shown) ──
    ax = axes[2, 1]
    if 'SOILLIQ' in ds:
        final = ds['SOILLIQ'].isel(time=-1).squeeze().values
        n = min(10, len(final))
        node_depths = ELM_LEVEL_NODE_DEPTH_M[:n]
        vals = final[:n]

        # Layer interfaces → bar heights matching layer thickness
        z_int = np.zeros(n + 1)
        z_int[0] = 0.0
        for i in range(1, n):
            z_int[i] = (node_depths[i-1] + node_depths[i]) / 2
        z_int[n] = node_depths[-1] + (node_depths[-1] - z_int[n-1])
        heights = z_int[1:] - z_int[:-1]

        ax.barh(node_depths, vals, height=heights, color='royalblue',
                alpha=0.75, edgecolor='white', linewidth=0.5,
                align='center')
        ax.set_xlabel('SOILLIQ (kg/m²)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Final SOILLIQ Profile')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        max_v = float(max(vals)) if len(vals) else 1.0
        for d, v in zip(node_depths, vals):
            if v > max_v * 0.02:
                ax.text(v + max_v*0.01, d, f' {v:.0f}',
                        va='center', fontsize=9)
        ax.text(0.98, 0.02, f'Total: {vals.sum():.0f} kg/m²',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10,
                bbox=dict(facecolor='lightblue', alpha=0.85,
                          edgecolor='gray'))

    for ax_i in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]:
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_i.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_cross_experiment_comparison(loaded_runs, output_path):
    """Cross-experiment overlay on common day-of-year axis (not calendar year).

    Each experiment's time series is converted to day-of-year (1-365), so
    a 1983 wet run and a 1990 dry run both span Jan-Dec on the same axis.
    """
    import pandas as pd

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle('Cross-experiment Time Series — Seasonal Overlay',
                 fontsize=14, fontweight='bold', y=0.995)

    def _doy(time_values):
        return pd.to_datetime(time_values).dayofyear

    for r in loaded_runs:
        exp = r['exp']; ds = r['ds']
        period = str(exp.get('forcing_period', '')).lower()
        color = PERIOD_COLORS.get(period, 'gray')
        label = f"{exp['case_name']}  ({exp.get('forcing_start','?')}, {period})"

        if 'QCHARGE' in ds:
            c = _flux_to_mm_day(ds['QCHARGE']).resample(time='D').mean().cumsum()
            axes[0].plot(_doy(c.time.values), c.values,
                         color=color, linewidth=2.2, label=label)
        if 'QOVER' in ds:
            c = _flux_to_mm_day(ds['QOVER']).resample(time='D').mean().cumsum()
            axes[1].plot(_doy(c.time.values), c.values,
                         color=color, linewidth=2.2)
        if 'TWS' in ds:
            tws = ds['TWS'].squeeze().resample(time='D').mean()
            axes[2].plot(_doy(tws.time.values), tws.values,
                         color=color, linewidth=2.2)

    # Month-label x-ticks (DOY at start of each month, non-leap year)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    axes[0].set_ylabel('Cum QCHARGE (mm)')
    axes[0].set_title('Cumulative Recharge')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=10)

    axes[1].set_ylabel('Cum QOVER (mm)')
    axes[1].set_title('Cumulative Runoff')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel('TWS (mm)')
    axes[2].set_xlabel('Month (day-of-year)')
    axes[2].set_title('Total Water Storage')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(month_starts)
    axes[2].set_xticklabels(month_labels)
    axes[2].set_xlim(1, 366)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#   ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════
def _load_run_context(run_dir):
    llm_input_path = run_dir / "LLM_ANALYSIS_INPUT.json"
    summary_path   = run_dir / "01_inputs" / "experiment_summary.json"

    if not llm_input_path.exists() or not summary_path.exists():
        logger.error("Missing required JSON files in run dir")
        return None, None

    llm_input = json.loads(llm_input_path.read_text())
    summary   = json.loads(summary_path.read_text())

    plan = llm_input.get('experiment_plan', {})
    experiments = summary.get('experiments', [])

    for exp in experiments:
        cd = exp.get('case_dir')
        if cd in (None, 'not_prepared', ''):
            discovered = discover_case_dir(exp.get('case_name', ''), run_dir)
            if discovered:
                exp['case_dir'] = str(discovered)
                logger.info(
                    f"   discovered: {exp['case_name']} → {discovered.name}")
            else:
                logger.warning(
                    f"   could not discover case_dir for {exp.get('case_name','?')}")
    return plan, experiments


def regenerate_setup_plots(run_dir):
    plan, experiments = _load_run_context(run_dir)
    if not experiments:
        return 0

    elm_config = plan.get('ELM_CONFIG', {})
    setup_dir  = run_dir / "02_setup_plots"
    setup_dir.mkdir(exist_ok=True)
    n_ok = 0
    experiment_runs = []

    for i, exp in enumerate(experiments, 1):
        case_name = exp.get('case_name', f'exp_{i:03d}')
        case_dir  = Path(exp['case_dir']) if exp.get('case_dir') else None
        h_files   = _history_files(case_dir) if case_dir else []

        exp_subdir = setup_dir / f"exp_{i:03d}_{case_name}"
        exp_subdir.mkdir(exist_ok=True)
        out = exp_subdir / "domain_configuration.png"
        try:
            plot_domain_configuration(exp, elm_config, h_files, out)
            n_ok += 1
            logger.info(f"   ✓ {out.relative_to(run_dir)}")
        except Exception as e:
            logger.warning(f"   ✗ {case_name}: {e}")

        experiment_runs.append((exp, h_files))

    if len(experiment_runs) >= 2:
        out = setup_dir / "comparison_forcing_conditions.png"
        try:
            compare_forcing_conditions(experiment_runs, out)
            n_ok += 1
            logger.info(f"   ✓ {out.relative_to(run_dir)}")
        except Exception as e:
            logger.warning(f"   ✗ comparison_forcing_conditions: {e}")

    return n_ok


def regenerate_analysis_plots(run_dir):
    plan, experiments = _load_run_context(run_dir)
    if not experiments:
        return 0

    analysis_dir = run_dir / "04_analysis"
    analysis_dir.mkdir(exist_ok=True)
    n_ok = 0
    loaded = []

    for exp in experiments:
        case_name = exp.get('case_name', '?')
        case_dir  = Path(exp['case_dir']) if exp.get('case_dir') else None
        h_files   = _history_files(case_dir) if case_dir else []
        if not h_files:
            logger.warning(f"   ✗ {case_name}: no history files")
            continue
        ds = _open_history(h_files)
        if ds is None:
            continue

        out = analysis_dir / f"{case_name}_timeseries.png"
        try:
            plot_experiment_timeseries(exp, ds, out)
            n_ok += 1
            logger.info(f"   ✓ {out.relative_to(run_dir)}")
        except Exception as e:
            logger.warning(f"   ✗ {case_name}: {e}")
        loaded.append({'exp': exp, 'ds': ds})

    if len(loaded) >= 2:
        out = analysis_dir / "comparison_timeseries.png"
        try:
            plot_cross_experiment_comparison(loaded, out)
            n_ok += 1
            logger.info(f"   ✓ {out.relative_to(run_dir)}")
        except Exception as e:
            logger.warning(f"   ✗ comparison: {e}")

    for r in loaded:
        try:
            r['ds'].close()
        except Exception:
            pass
    return n_ok


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", help="Path to a completed workflow run directory")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--setup-only", action="store_true")
    grp.add_argument("--analysis-only", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        logger.error(f"Not a directory: {run_dir}")
        sys.exit(1)

    do_setup    = not args.analysis_only
    do_analysis = not args.setup_only

    print(f"╔{'═'*60}╗")
    print(f"║  REPLOT: {str(run_dir)[-50:]:<50}║")
    print(f"╚{'═'*60}╝")

    total = 0
    if do_setup:
        print("\n📋 Setup plots (02_setup_plots/)")
        print("-" * 50)
        total += regenerate_setup_plots(run_dir)

    if do_analysis:
        print("\n📊 Analysis plots (04_analysis/)")
        print("-" * 50)
        total += regenerate_analysis_plots(run_dir)

    print(f"\n✅ Done — {total} plot(s) generated")
    print(f"   Run dir: {run_dir}")


if __name__ == "__main__":
    main()