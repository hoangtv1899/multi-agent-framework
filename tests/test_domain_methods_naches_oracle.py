"""The method catalog on the Naches control: its suite and its daily history.

    pytest tests/test_domain_methods_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the reference table ("reference", the
Demo 1 regional table of 2015-2023), the control's run directory ("run_dir")
and its built suite ("suite_dir").

  * anomaly, from the suite's series, reproduces the anomalies of the
    regional table (regions.anomalies on its year values; the mean of the
    normal years' same month or season, the table's own normal rule, for its
    months and seasons), to 1e-9 relative, and the values the plan quotes;
  * every timing measure gives a date or a reason for every vegetated
    watershed cell and every year 2015-2023, and each reason (and each
    default date) falls exactly where its own rule says, cell by cell, as
    worked out here by separate per-cell code. How the reasons fall by
    vegetation type is reported, not required;
  * the screen's table of 2022 against 2016-2020 holds every catalog field
    and period with its variable type and support, and its GPP scores are
    the table's months set against their normals.

The daily history is read once per year (9 passes); opening the stream reads
every file's time stamps, so the index cache points at a temporary directory
for the session, as the suite's oracle test does.
"""
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_analysis as DA
import domain_methods as M
import domain_suite as S
import gridded as G
import regions as R

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
REGION = "watershed, vegetated"
NORMAL = [2016, 2017, 2018, 2019, 2020]
YEARS = list(range(2015, 2024))
FIELDS = ("GPP", "TOTECOSYSC", "H2OSNO", "EFLX_LH_TOT", "QRUNOFF")


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    if not all(c.get(k) for k in ("reference", "run_dir", "suite_dir")) or \
            not Path(c["run_dir"]).is_dir() or not (Path(c["suite_dir"]) / "suite.json").is_file():
        pytest.skip("no naches_suite block with reference, run_dir and a built suite_dir")
    ref = json.loads(Path(c["reference"]).read_text())
    return {"run_dir": c["run_dir"], "suite": M.Suite(c["suite_dir"]),
            "table": ref["runs"]["control"]}


@pytest.fixture(scope="module")
def index_cache(tmp_path_factory):
    old = os.environ.get(G.INDEX_CACHE_ENV)
    os.environ[G.INDEX_CACHE_ENV] = str(tmp_path_factory.mktemp("stream_index"))
    yield
    if old is None:
        os.environ.pop(G.INDEX_CACHE_ENV, None)
    else:
        os.environ[G.INDEX_CACHE_ENV] = old


# ── anomaly ──────────────────────────────────────────────────────────────────

def test_anomaly_reproduces_the_regional_table(cfg):
    s, t = cfg["suite"], cfg["table"]
    for f in FIELDS:
        want = R.anomalies({int(y): v for y, v in t["values"][REGION][f].items()},
                           NORMAL, YEARS)
        for y in YEARS:
            r = M.anomaly(s, f, str(y), NORMAL, region=REGION)
            w = want["years"][y]
            assert r["anomaly"] == pytest.approx(w["anomaly"], rel=1e-9, abs=1e-9 * abs(w["value"]))
            if r["percent"] is not None:
                assert r["percent"] == pytest.approx(w["anomaly_pct"], rel=1e-9), (f, y)
        for kind, key, parts in (("month", "monthly", range(1, 13)),
                                 ("season", "seasonal", DA.SEASONS)):
            for y in YEARS:
                pn = DA.period_normals(t, NORMAL, target_years=[y])
                normals = pn["monthly_normal" if kind == "month" else "seasonal_normal"][REGION][f]
                for p in parts:
                    p = p[0] if kind == "season" else p
                    lab = M.period_label(kind, y, p)
                    v = t[key][REGION][f].get(lab)
                    r = M.anomaly(s, f, lab, NORMAL, region=REGION)
                    if v is None:
                        assert r["anomaly"] is None, (f, lab)
                        continue
                    n = normals[f"{p:02d}" if kind == "month" else p]
                    assert r["normal"] == pytest.approx(n, rel=1e-9), (f, lab)
                    assert r["anomaly"] == pytest.approx(v - n, rel=1e-9, abs=1e-9 * abs(v)), (f, lab)
                    if r["percent"] is not None:
                        assert r["percent"] == pytest.approx(100 * (v - n) / n, rel=1e-9), (f, lab)
                    else:
                        # the field notes' percent policy, apart from the scale (review
                        # round 1, 2026-11-05)
                        assert not M.percent_policy(M._notes(None)[f])[0] or n == 0 or \
                            abs(n) < 0.01 * r["reference_range"], (f, lab)
    quoted = {("GPP", "2015"): 1.1, ("GPP", "2021"): -17.8, ("GPP", "2022"): -5.9,
              ("GPP", "2015-07"): -18.7, ("GPP", "2021-07"): -26.8, ("GPP", "2022-05"): -22.5,
              ("H2OSNO", "2015-02"): -94.7, ("QRUNOFF", "2015-03"): -75.4}
    for (f, lab), pct in quoted.items():
        assert round(M.anomaly(s, f, lab, NORMAL, region=REGION)["percent"], 1) == pct, (f, lab)


# ── the timing measures ──────────────────────────────────────────────────────
C = M.CODE


def _runs(x, thr, days):
    return np.convolve((x < thr).astype(np.int64), np.ones(days, np.int64), "valid") == days


def _missing(x, held):
    """A day no cell holds is a missing time record, for every cell; a value
    the cell alone lacks on a day the records hold is a missing cell value
    (review round 1, 2026-11-05)."""
    if not held.all():
        return C["records_missing"]
    return C["cell_values_missing"] if not np.isfinite(x).all() else 0


def expect(mid, x, st, p, held):
    """(reason, date) by the measure's own rule, worked out for one cell;
    held says, per day, whether any cell has a record."""
    r0, fd = p.record_start, p.first_day
    if mid == "snow_peak":
        if r0:
            return C["before_record"], None
        if _missing(x, held):
            return _missing(x, held), None
        if x.max() < st["threshold_mm"]:
            return C["no_snow"], None
        return 0, int(np.argmax(x)) + fd
    if mid == "snow_gone":
        x, held = x[r0:], held[r0:]
        if _missing(x, held):
            return _missing(x, held), None
        if x.max() < st["threshold_mm"]:
            return C["no_snow"], None
        pk = int(np.argmax(x))
        js = np.flatnonzero(_runs(x, st["threshold_mm"], st["days"])[pk:])
        if js.size:
            return 0, pk + int(js[0]) + r0 + fd
        # days below the threshold that reach the water year's end too soon to
        # make the run: insufficient follow-up, apart from snow that persists
        above = np.flatnonzero(~(x < st["threshold_mm"]))
        tail = x.size - (int(above[-1]) + 1)
        if 0 < tail < st["days"] and x.size - tail > pk:
            return C["insufficient_follow_up"], None
        return C["snow_persists"], None
    if mid == "half_runoff":
        if r0:
            return C["before_record"], None
        if _missing(x, held):
            return _missing(x, held), None
        c = np.cumsum(x)
        if not c[-1] > 0:
            return C["no_runoff"], None
        return 0, int(np.flatnonzero(c >= st["fraction"] * c[-1])[0]) + fd
    if _missing(x, held):
        return _missing(x, held), None
    if mid == "leaf_green_up":
        pk = int(np.argmax(x))
        if not x[pk] > 0:
            return C["no_leaves"], None
        low_day = int(np.argmin(x[:pk + 1]))
        rise = x[pk] - x[low_day]
        if rise < st["rise_mark"] * x[pk]:
            return C["little_leaf_change"], None
        mark = x[low_day] + st["fraction"] * rise
        return 0, low_day + 1 + int(np.flatnonzero(x[low_day + 1:] >= mark)[0]) + fd
    w = st["window"]
    m = np.convolve(x, np.ones(w) / w, "valid")
    pk = int(np.argmax(m))
    if not m[pk] > 0:
        return C["no_photosynthesis"], None
    if mid == "growth_peak":
        return 0, pk + w // 2 + fd
    lo = int(np.argmin(m[:pk + 1]))
    mark = st["fraction"] * m[pk]
    if not m[lo] < mark:
        return C["no_dormant_season"], None
    return 0, lo + 1 + int(np.flatnonzero(m[lo + 1:] >= mark)[0]) + w // 2 + fd


@pytest.fixture(scope="module")
def timing(cfg, index_cache):
    """Every measure, every year, every setting, from one pass per year file;
    the watershed cells' results and every disagreement with expect()."""
    cells = np.flatnonzero(cfg["suite"].mask(REGION).ravel())
    s = G.open_stream(cfg["run_dir"], "h1")
    out = {m: {} for m in M.TIMING_IDS}
    wrong = []
    for p in M.daily_periods(s, YEARS):
        for mid in M.TIMING_IDS:
            if M.TIMING_KIND[mid] != p.kind:
                continue
            res = M.METHODS[mid].run(p)
            x = p.data[M.TIMING_FIELD[mid]]
            held = np.isfinite(x).any(axis=1)
            keep = {"variants": {}, "record_start": p.record_start}
            for lab, v in res["variants"].items():
                keep["variants"][lab] = {k: v[k][cells] for k in ("date", "reason", "flags")}
                for i, c in zip(range(cells.size), cells):
                    reason, date = expect(mid, x[:, c], v["settings"], p, held)
                    got_r, got_d = int(v["reason"][c]), v["date"][c]
                    if got_r != reason or (lab == "default" and date is not None and got_d != date):
                        wrong.append((mid, p.year, lab, int(c), got_r, reason, got_d, date))
            out[mid][p.year] = keep
    return {"by": out, "wrong": wrong, "cells": cells}


def test_every_timing_measure_gives_a_date_or_a_reason(timing):
    n = timing["cells"].size
    assert n == 4127
    for mid in M.TIMING_IDS:
        assert sorted(timing["by"][mid]) == YEARS, mid
        for y, keep in timing["by"][mid].items():
            for lab, v in keep["variants"].items():
                assert v["reason"].size == n
                assert set(np.unique(v["reason"]).tolist()) <= set(M.REASON_OF_CODE), (mid, y)
                assert (np.isfinite(v["date"]) == (v["reason"] == 0)).all(), (mid, y, lab)


def test_every_timing_reason_follows_its_rule(timing):
    w = timing["wrong"]
    assert not w, f"{len(w)} cells disagree with their rule; first: {w[:5]}"


def test_a_water_year_without_its_start(timing):
    sp = timing["by"]["snow_peak"][2015]["variants"]["default"]
    assert (sp["reason"] == C["before_record"]).all()
    assert timing["by"]["snow_peak"][2015]["record_start"] > 0
    sg = timing["by"]["snow_gone"][2015]["variants"]["default"]
    assert ((sg["flags"] & M.FLAG["partial_record"]) > 0).all()
    assert (timing["by"]["half_runoff"][2015]["variants"]["default"]["reason"]
            == C["before_record"]).all()
    for y in YEARS[1:]:
        f = timing["by"]["snow_gone"][y]["variants"]["default"]["flags"]
        assert not ((f & M.FLAG["partial_record"]) > 0).any(), y


def test_the_suites_timing_file_holds_the_same(cfg, timing):
    """timing.nc, made by the suite's own pass, against the measures above
    (which follow their rules cell by cell): every setting, value for value."""
    p = cfg["suite"].dir / S.TIMING_FILE
    if not p.is_file():
        pytest.skip("the control's suite has no timing.nc yet")
    with netCDF4.Dataset(p) as d:
        d.set_auto_mask(False)
        ty = [int(y) for y in np.asarray(d["year"][:])]
        assert ty == YEARS
        for mid in M.TIMING_IDS:
            for y in YEARS:
                for lab, v in timing["by"][mid][y]["variants"].items():
                    base = S._timing_name(mid, lab)
                    for part in ("date", "reason", "flags"):
                        a = np.asarray(d[f"{base}__{part}"][..., ty.index(y)]).ravel()
                        a = a[timing["cells"]]
                        if part == "date":
                            np.testing.assert_array_equal(a, v[part].astype(np.float32),
                                                          err_msg=f"{mid} {y} {lab}")
                        else:
                            assert (a == v[part]).all(), (mid, y, lab, part)


# ── the screen ───────────────────────────────────────────────────────────────

def test_screen_2022_holds_every_field_and_period(cfg, tmp_path):
    s, t = cfg["suite"], cfg["table"]
    notes = M._notes(None)
    fields = M.catalog_fields(notes)
    res = M.screen(s, REGION, 2022, NORMAL)
    rows = res["rows"]
    periods = M.screen_periods(2022)
    assert [(r["field"], r["period"]) for r in rows] == [(f, p) for f in fields for p in periods]
    assert len(rows) == len(fields) * 18 == 52 * 18
    for r in rows:
        assert r["type"] == notes[r["field"]]["type"] and r["group"] == notes[r["field"]]["group"]
        assert r["status"] not in ("not in the suite", "period absent"), r
        assert r["n_normal"] == 5 and r["normal_years_used"] == "2016 2017 2018 2019 2020", r
        assert r["normal_sd"] is not None and r["reference_range"] is not None
        if r["type"] in ("store", "bounded"):
            assert r["status"] in ("not scored yet", "no change")
        if r["status"] == "scored":
            assert r["type"] in ("flux", "state") and np.isfinite(r["score"])
    gm = t["monthly"][REGION]["GPP"]
    normals = [np.mean([gm[f"{y}-{m:02d}"] for y in NORMAL]) for m in range(1, 13)]
    rng = max(normals) - min(normals)
    for m in range(1, 13):
        lab = f"2022-{m:02d}"
        r = next(x for x in rows if x["field"] == "GPP" and x["period"] == lab)
        nv = np.array([gm[f"{y}-{m:02d}"] for y in NORMAL])
        assert r["reference_range"] == pytest.approx(rng, rel=1e-9)
        if nv.std(ddof=1) <= 0.01 * rng:          # winter photosynthesis: flagged, never scored
            assert r["status"] == "near-constant normal" and r["score"] is None, lab
            continue
        assert r["status"] == "scored", lab
        assert r["score"] == pytest.approx((gm[lab] - nv.mean()) / nv.std(ddof=1), rel=1e-9)
    p = M.write_screen_csv(res, tmp_path / "screen_2022.csv")
    assert len(p.read_text().splitlines()) == len(rows) + 1
    shown = res["display"]
    assert all(len(v) <= 3 and len({r["field"] for r in v}) == len(v)
               for v in shown["groups"].values())
