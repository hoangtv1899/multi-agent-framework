"""Offline tests for the ELM results analyzer's pure aggregation logic.

Covers the spatial-ensemble fine-tune (no NetCDF, no compute): the
recharge/runoff partitioning metric and the elevation-gradient summary that
kicks in when forcing is uniform but location varies. Results are injected
directly, so no history files are read.

Run under: module load pytorch/2.8.0
"""
import sys

import pytest

pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

sys.path.insert(0, "src")
from core.elm_results_analyzer import ELMResultsAnalyzer   # noqa: E402


def _analyzer(tmp_path):
    # __init__ only needs xarray available + a writable analysis_dir
    return ELMResultsAnalyzer(experiments=[], analysis_dir=str(tmp_path / "an"))


def _col(name, elev, rech, runf, wtd, lat=46.0, lon=-120.0, precip=1000.0, soil=None):
    total = rech + runf
    return {
        "status": "ok", "case_name": name, "scenario_name": name,
        "forcing_period": "baseline", "lat": lat, "lon": lon, "elevation_m": elev,
        "soil": soil,
        "metrics": {
            "annual_recharge_mm_yr": rech, "annual_runoff_mm_yr": runf,
            "recharge_fraction": round(rech / total, 4) if total else None,
            "water_table_depth_m": wtd, "precip_mm_yr": precip,
        },
    }


def _soil(clay, ksat, texture="loam"):
    return {"texture_top": texture, "clay_max_pct": clay, "ksat_min_ums": ksat}


# ── partitioning + WTD metric ────────────────────────────────────────────────
def test_compute_metrics_partitioning_and_wtd(tmp_path):
    az = _analyzer(tmp_path)
    m = az._compute_metrics({
        "QCHARGE": {"annual_mean": 300.0},
        "QOVER":   {"annual_mean": 100.0},
        "ZWT":     {"mean_m": 7.5},
    })
    assert m["annual_recharge_mm_yr"] == 300.0
    assert m["recharge_fraction"] == 0.75 and m["runoff_fraction"] == 0.25
    assert m["recharge_to_runoff_ratio"] == 3.0
    assert m["water_table_depth_m"] == 7.5


def test_compute_metrics_handles_zero_total(tmp_path):
    az = _analyzer(tmp_path)
    m = az._compute_metrics({"QCHARGE": {"annual_mean": 0.0},
                             "QOVER": {"annual_mean": 0.0}})
    assert "recharge_fraction" not in m          # no divide-by-zero blowup


# ── spatial summary + driver attribution ─────────────────────────────────────
def test_spatial_summary_sorts_and_reports_slope(tmp_path):
    az = _analyzer(tmp_path)
    az.results = {
        "c2": _col("c2", 943, 347, 83, 8.71, lat=46.8, lon=-121.1, precip=1373),
        "c1": _col("c1", 750,   0, 53, 8.80, lat=46.7, lon=-120.8, precip=793),
        "c3": _col("c3", 1031, 576, 85, 8.31, lat=46.5, lon=-121.4, precip=1373),
    }
    ss = az._compute_spatial_summary()
    assert ss["n_columns"] == 3
    assert ss["elevation_range_m"] == [750.0, 1031.0]
    assert [r["case_name"] for r in ss["by_elevation"]] == ["c1", "c2", "c3"]  # sorted
    s = ss["vs_elevation"]["slope_per_1000m"]
    assert s["recharge_mm_yr"] > 0          # recharge rises with elevation
    assert s["water_table_m"] < 0           # water table shallows (depth decreases)
    assert "fit_r2" in ss["vs_elevation"]   # fit quality reported alongside the slope


def test_spatial_summary_surfaces_quantized_forcing(tmp_path):
    # two distinct precip values across 3 columns -> forcing confound must be flagged
    az = _analyzer(tmp_path)
    az.results = {
        "c1": _col("c1", 750,   0, 53, 8.80, lat=46.7, lon=-120.8, precip=793),
        "c2": _col("c2", 943, 347, 83, 8.71, lat=46.8, lon=-121.1, precip=1373),
        "c3": _col("c3", 1031, 576, 85, 8.31, lat=46.5, lon=-121.4, precip=1373),
    }
    ss = az._compute_spatial_summary()
    assert ss["forcing"]["n_forcing_bins"] == 2
    assert ss["forcing"]["elevation_resolved"] is False
    assert "recharge_vs_precip_r" in ss["driver_correlation"]
    # the interpretation must call out the coarse, non-elevation-resolved forcing
    assert any("quantized" in n or "forcing" in n for n in ss["interpretation"])


def test_spatial_summary_empty_for_single_location(tmp_path):
    az = _analyzer(tmp_path)
    same = dict(lat=46.0, lon=-120.0)
    az.results = {
        "a": _col("a", 800, 100, 50, 8.5, **same),
        "b": _col("b", 900, 200, 60, 8.0, **same),   # same lat/lon
    }
    assert az._compute_spatial_summary() == {}     # not a spatial ensemble


def test_spatial_summary_needs_two_columns(tmp_path):
    az = _analyzer(tmp_path)
    az.results = {"only": _col("only", 800, 100, 50, 8.5)}
    assert az._compute_spatial_summary() == {}


# ── soil attribution (forcing held constant) ─────────────────────────────────
def test_soil_attribution_holds_forcing_and_blames_clay(tmp_path):
    az = _analyzer(tmp_path)
    # all same precip (one forcing bin); recharge falls as max-clay rises
    az.results = {  # clay monotonic with recharge; Ksat scrambled so clay clearly wins
        "a": _col("a", 900,  648, 80, 8.1, precip=1373, soil=_soil(8, 28)),
        "b": _col("b", 950,  507, 80, 8.3, precip=1373, soil=_soil(15, 5)),
        "c": _col("c", 1000, 309, 80, 8.6, precip=1373, soil=_soil(25, 12)),
    }
    sa = az._compute_soil_attribution()
    assert sa["forcing_held_mm_yr"] == 1373 and sa["n_columns"] == 3
    assert sa["soil_correlation"]["recharge_vs_clay_max"] < -0.9   # clay impedes recharge
    assert "clay" in sa["strongest_predictor"]
    assert [r["case_name"] for r in sa["by_recharge"]] == ["a", "b", "c"]  # recharge desc


def test_soil_attribution_picks_largest_forcing_bin(tmp_path):
    az = _analyzer(tmp_path)
    az.results = {
        "lo":  _col("lo", 700,   0, 50, 8.8, precip=793, soil=_soil(30, 2)),
        "h1":  _col("h1", 950, 600, 85, 8.2, precip=1373, soil=_soil(8, 28)),
        "h2":  _col("h2", 1000, 500, 85, 8.3, precip=1373, soil=_soil(15, 12)),
        "h3":  _col("h3", 1100, 350, 85, 8.5, precip=1373, soil=_soil(24, 4)),
    }
    sa = az._compute_soil_attribution()
    assert sa["forcing_held_mm_yr"] == 1373        # the 3-column bin, not the singleton
    assert sa["n_columns"] == 3


def test_soil_attribution_empty_without_soil(tmp_path):
    az = _analyzer(tmp_path)
    az.results = {n: _col(n, 900, 500, 80, 8.3, precip=1373) for n in ("a", "b", "c")}
    assert az._compute_soil_attribution() == {}     # no soil data -> nothing to attribute
