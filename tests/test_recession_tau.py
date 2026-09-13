#!/usr/bin/env python3
"""The recession timescale, pinned on a synthetic exponential recession.

    pytest tests/test_recession_tau.py -v

A linear reservoir drains as Q = Q0 exp(-t/tau); four injected rises
restart it from a higher level. The segment rule must cut the rises and
their shadow out, and both methods must hand back the tau that built the
series, within 5 percent. The run-level test writes a small reception.json
and experiment.json into tmp_path and goes through compute(), including
the --exclude path for a regulated gauge.
"""
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import recession_tau as rt                             # noqa: E402

TAU = 30.0
RISES = (60, 150, 240, 330)
N = 365
YEAR = 2010


def synthetic_q(tau=TAU, n=N, rises=RISES, q0=5.0, jump=3.0,
                noise=0.0, seed=0):
    q = np.empty(n)
    q[0] = q0
    for i in range(1, n):
        q[i] = q[i - 1] * math.exp(-1.0 / tau) * (jump if i in rises else 1.0)
    if noise:
        q = q * np.exp(np.random.default_rng(seed).normal(0.0, noise, n))
    return q


def dates(n=N, year=YEAR):
    return [date(year, 1, 1) + timedelta(days=i) for i in range(n)]


class TestSegments:
    def test_rises_and_their_shadow_are_cut_out(self):
        q = synthetic_q()
        segs = rt.segments(q)
        assert len(segs) == len(RISES) + 1
        assert segs[0][0] == 1                     # day 0 has no predecessor
        kept = {i for s in segs for i in s}
        for r in RISES:
            assert not kept & set(range(r, r + rt.RISE_SHADOW_DAYS + 1))
            assert r + rt.RISE_SHADOW_DAYS + 1 in kept
        assert sum(len(s) for s in segs) == N - 1 - len(RISES) * 4

    def test_a_dropped_day_splits_a_segment(self):
        q = synthetic_q()
        drop = np.zeros(N, bool)
        drop[100] = True
        segs = rt.segments(q, drop)
        assert len(segs) == len(RISES) + 2
        assert 100 not in {i for s in segs for i in s}

    def test_a_calendar_gap_splits_a_segment(self):
        q = synthetic_q()
        d = dates()
        d = d[:100] + [x + timedelta(days=1) for x in d[100:]]
        assert len(rt.segments(q, None, d)) == len(RISES) + 2

    def test_short_and_flat_runs_are_not_segments(self):
        q = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
        assert rt.segments(q) == []


class TestMethods:
    def test_master_recovers_tau_exactly_on_a_clean_series(self):
        q = synthetic_q()
        m = rt.master_recession(q, rt.segments(q))
        assert m["tau_days"] == pytest.approx(TAU, rel=1e-6)
        assert m["n_segments"] == len(RISES) + 1
        assert m["tau_lo_days"] <= TAU <= m["tau_hi_days"]

    def test_master_within_5_percent_with_noise(self):
        q = synthetic_q(noise=0.005, seed=3)
        m = rt.master_recession(q, rt.segments(q))
        assert m["tau_days"] == pytest.approx(TAU, rel=0.05)

    def test_dqdt_recovers_tau_and_a_unit_exponent(self):
        q = synthetic_q()
        d = rt.dqdt_binned(q, rt.segments(q))
        assert d["n_pairs"] >= rt.MIN_PAIRS
        assert d["n_bins"] >= rt.MIN_BINS
        assert d["tau_days"] == pytest.approx(TAU, rel=0.05)
        assert d["b_free"] == pytest.approx(1.0, abs=0.02)
        assert d["tau_pointwise_median_days"] == pytest.approx(TAU, rel=0.05)

    def test_dqdt_within_5_percent_with_noise(self):
        q = synthetic_q(noise=0.005, seed=3)
        d = rt.dqdt_binned(q, rt.segments(q))
        assert d["tau_days"] == pytest.approx(TAU, rel=0.05)

    def test_too_few_pairs_is_reported_not_fitted(self):
        d = rt.dqdt_binned(synthetic_q(), [list(range(1, 6))])
        assert d["tau_days"] is None
        assert d["n_pairs"] == 4


class TestDayFlags:
    def test_wet_today_or_yesterday_and_unknown_days(self):
        d = dates(10)
        forcing = {"dates": d[2:], "mm_day": np.zeros(8), "n_columns": 1}
        forcing["mm_day"][3] = 2.5                # day index 5 is wet
        f = rt.day_flags(d, forcing, None)
        assert list(f["known"]) == [False, False] + [True] * 8
        assert list(np.nonzero(f["wet"])[0]) == [5, 6]
        assert list(np.nonzero(f["drop"])[0]) == [0, 1, 5, 6]

    def test_melt_from_any_station_and_nothing_without_forcing(self):
        d = dates(6)
        swe = {"A": (d, np.array([100.0, 99.0, 90.0, 90.0, 90.0, 90.0])),
               "B": (d, np.array([50.0, 50.0, 50.0, 47.0, 47.0, 47.0]))}
        f = rt.day_flags(d, None, swe)
        assert list(np.nonzero(f["melt"])[0]) == [2, 3]
        assert list(np.nonzero(f["drop"])[0]) == [2, 3]
        assert not f["known"].any()


class TestLoadForcing:
    def test_basin_mean_over_columns_that_ran(self):
        d = [str(x) for x in dates(3)]
        col = lambda r, s, status="ok": {                    # noqa: E731
            "case_name": "c", "status": status,
            "variables": {"RAIN": {"daily": {"dates": d, "values": r}},
                          "SNOW": {"daily": {"dates": d, "values": s}}}}
        f = rt.load_forcing({"columns": [col([1, 0, 0], [0, 0, 2]),
                                          col([3, 0, 0], [0, 4, 0]),
                                          col([9, 9, 9], [9, 9, 9], "failed")]})
        assert f["n_columns"] == 2
        assert list(f["mm_day"]) == [2.0, 2.0, 1.0]
        assert rt.load_forcing({"columns": []}) is None


def write_run(tmp_path, regulated_tau=10.0, wet_day=None):
    d = dates()
    keys = [str(x) for x in d]

    def station(gid, name, q, in_basin=True, area=100.0):
        return {"id": gid, "name": name, "in_basin": in_basin,
                "drainage_area_km2": area, "altitude_m": 10.0,
                "mm_day": dict(zip(keys, map(float, q)))}
    stations = [station("USGS-A", "free", synthetic_q(TAU), area=200.0),
                station("USGS-B", "regulated", synthetic_q(regulated_tau)),
                station("USGS-Z", "elsewhere", synthetic_q(3.0), in_basin=False)]
    reception = {"brief": {"domain": {"name": "Synthetic"}},
                 "observations": {"streamflow": {"stations": stations},
                                  "swe": {"stations": []}}}
    rain = [0.0] * N
    if wet_day is not None:
        rain[wet_day] = 5.0
    experiment = {"columns": [{"case_name": "col_01", "status": "ok",
                               "variables": {
                                   "RAIN": {"daily": {"dates": keys, "values": rain}},
                                   "SNOW": {"daily": {"dates": keys,
                                                      "values": [0.0] * N}}}}]}
    run = tmp_path / "run"
    run.mkdir()
    (run / "reception.json").write_text(json.dumps(reception))
    (run / "experiment.json").write_text(json.dumps(experiment))
    return run


class TestCompute:
    def test_excluded_gauge_is_reported_but_not_pooled(self, tmp_path):
        run = write_run(tmp_path, wet_day=100)
        rep = rt.compute(run, exclude=["USGS-B"])
        assert [g["id"] for g in rep["gauges"]] == ["USGS-A", "USGS-B"]
        a, b = rep["gauges"]
        assert not a["excluded"] and b["excluded"]
        assert b["master"]["tau_days"] == pytest.approx(10.0, rel=0.05)
        assert rep["pooled"]["gauges"] == ["USGS-A"]
        assert rep["pooled"]["tau_days"] == pytest.approx(TAU, rel=0.05)
        assert rep["sink_tau_days"] == rep["pooled"]["tau_days"]
        assert "USGS-B" in rep["sink_tau_source"]
        # the wet day (and the one after) split one segment in two
        assert a["n_wet_days"] == 2
        assert a["n_segments"] == len(RISES) + 2
        assert rep["forcing"]["available"] is True
        assert rep["swe"]["available"] is False

    def test_without_exclusion_the_regulated_gauge_pulls_the_pool(self, tmp_path):
        rep = rt.compute(write_run(tmp_path))
        assert rep["pooled"]["n_gauges"] == 2
        assert 10.0 < rep["pooled"]["tau_days"] < TAU

    def test_an_unknown_exclusion_is_refused_by_name(self, tmp_path):
        with pytest.raises(ValueError, match="USGS-X"):
            rt.compute(write_run(tmp_path), exclude=["USGS-X"])

    def test_out_inside_the_run_is_refused(self, tmp_path):
        run = write_run(tmp_path)
        with pytest.raises(ValueError, match="inside the run"):
            rt.outside_run(run / "tau.json", run)
        assert rt.outside_run(tmp_path / "tau.json", run) == \
            (tmp_path / "tau.json").resolve()
