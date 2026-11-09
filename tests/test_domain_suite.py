"""The suite (domain_suite): one run's maps and region series, on small fake runs.

    pytest tests/test_domain_suite.py -v

The fake run has a monthly stream (h0: GPP, TOTECOSYSC, the layered H2OSOI
with its layer thicknesses, the monthly-only QCHARGE) and a daily one (h1:
GPP, TOTECOSYSC, H2OSOI and the daily-only COL_FIRE_CLOSS), each year in its
own files, so a year can be added by adding files. Every expected number is
made here from the arrays written, by the rule the suite states, not by the
suite's code; the month, season and year series are held to the regional
table's own numbers (domain_analysis.run_values, read without the suite).
"""
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_analysis as A
import domain_suite as S
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import gridded as G
import regions as R

Y0 = 2015
DT = fx.DT
MD = fx.MONTH_DAYS
DZ = (0.3, 0.5, 0.6)          # the top metre holds 0.3, 0.5 and 0.2 m of these layers
UNITS = {"GPP": "gC/m^2/s", "TOTECOSYSC": "gC/m^2", "H2OSOI": "mm3/mm3", "QCHARGE": "mm/s",
         "COL_FIRE_CLOSS": "gC/m^2/s"}
SCALE = {"GPP": 3e-5, "TOTECOSYSC": 2000.0, "H2OSOI": 0.3, "QCHARGE": 1e-5,
         "COL_FIRE_CLOSS": 1e-7}
H0 = ("GPP", "TOTECOSYSC", "H2OSOI", "QCHARGE")
H1 = ("GPP", "TOTECOSYSC", "H2OSOI", "COL_FIRE_CLOSS")
FIELDS = sorted(set(H0) | set(H1))
TABLE = ["COL_FIRE_CLOSS", "GPP", "QCHARGE", "TOTECOSYSC"]       # the table makes these
HOLE = (2, 2)                 # a vegetated cell: TOTECOSYSC has no value there once
HOLE_DAY = (2016, 40)         # (year, record of the year): 10 February 2016
HOLE_MONTH = (2016, 3)        # March 2016


def _add(path, data, dz=False):
    with netCDF4.Dataset(path, "a") as d:
        for name, a in data.items():
            dims = ("time", "levgrnd", "lat", "lon") if a.ndim == 4 else ("time", "lat", "lon")
            v = d.createVariable(name, "f4", dims, fill_value=fx.FILL)
            v.units = UNITS[name]
            v.long_name = name
            v.cell_methods = "time: mean"
            v[:] = a
        if dz:
            nlat, nlon = len(d.dimensions["lat"]), len(d.dimensions["lon"])
            v = d.createVariable("DZSOI", "f4", ("levgrnd", "lat", "lon"), fill_value=fx.FILL)
            v.units = "m"
            v[:] = np.broadcast_to(np.array(DZ, dtype=np.float32)[:, None, None],
                                   (fx.NLEV, nlat, nlon))


def _values(rng, field, shape):
    return (rng.uniform(0.5, 2.0, shape) * SCALE[field]).astype(np.float32)


def write_years(run, years):
    """Each year's monthly and daily files; the same numbers every time a
    year is written (seeded by year), so a run written in two goes equals one
    written at once."""
    g = fx.grid()
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    store = {}
    for y in years:
        off = 365.0 * (y - Y0)
        # daily, the year's own 365 days (the run's start step first in its first file)
        tb = ([(-DT, 0.0)] if y == Y0 else []) + [(off + k, off + k + 1.0) for k in range(365)]
        p = run / f"synth.elm.h1.{y:04d}-01-01-00000.nc"
        fx._write(p, g, Y0, tb, [None] * len(tb), [])
        rng = np.random.default_rng(2000 + y)
        lead = 1 if y == Y0 else 0
        data = {f: _values(rng, f, (len(tb), fx.NLEV) + g["shape"] if f == "H2OSOI"
                           else (len(tb),) + g["shape"]) for f in H1}
        if y == HOLE_DAY[0]:
            data["TOTECOSYSC"][lead + HOLE_DAY[1], HOLE[0], HOLE[1]] = fx.FILL
        _add(p, data)
        store[("h1", y)] = {f: a[lead:].astype(np.float64) for f, a in data.items()}
        # monthly, January folding in the run's start step
        tb, cum = [], 0
        for m in range(12):
            a0 = off + cum - (DT if (y == Y0 and m == 0) else 0.0)
            cum += MD[m]
            tb.append((a0, off + cum))
        p = run / f"synth.elm.h0.{y:04d}-02-01-00000.nc"
        fx._write(p, g, Y0, tb, [None] * 12, [])
        rng = np.random.default_rng(1000 + y)
        data = {f: _values(rng, f, (12, fx.NLEV) + g["shape"] if f == "H2OSOI"
                           else (12,) + g["shape"]) for f in H0}
        if y == HOLE_MONTH[0]:
            data["TOTECOSYSC"][HOLE_MONTH[1] - 1, HOLE[0], HOLE[1]] = fx.FILL
        _add(p, data, dz=True)
        store[("h0", y)] = {f: a.astype(np.float64) for f, a in data.items()}
    for v in store.values():
        for f, a in v.items():
            a[np.abs(a) > 1e30] = np.nan
    return store


def make_control(root, years):
    root = Path(root)
    g = fx.grid()
    case_dir, run_dir = root / "case", root / "run"
    (case_dir / "CaseDocs").mkdir(parents=True)
    surf = cf.write_surface(root / "surfdata.nc", g)
    (case_dir / "CaseDocs" / "lnd_in").write_text(
        cf.LND_IN.format(restart=root / "finidat.nc", surface=surf))
    lat, lon = g["lat"], g["lon"] - 360.0
    d = (lat[1] - lat[0]) / 2
    ring = [[float(lon[0] - d), float(lat[0] - d)], [float(lon[-1] + d), float(lat[0] - d)],
            [float(lon[-1] + d), float(lat[1] + d)], [float(lon[0] - d), float(lat[1] + d)],
            [float(lon[0] - d), float(lat[0] - d)]]
    box = root / "box.geojson"
    box.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    store = write_years(run_dir, years)
    return {"name": "synth", "case_dir": str(case_dir), "run_dir": str(run_dir),
            "regions": {"box": {"polygon": str(box), "what": "the first two rows"}}}, store


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    control, store = make_control(root, [2015, 2016])
    out = root / "suite"
    res = S.build(control, out, fields=FIELDS)
    return {"root": root, "control": control, "store": store, "out": out, "res": res,
            "man": S.load(out)}


def _nc(path, var):
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        return np.asarray(d[var][:])


def _labels(path, kind):
    with netCDF4.Dataset(path) as d:
        return [str(x) for x in np.asarray(d[kind][:], dtype=object)]


def _grid_weights(run):
    """area x landfrac as the files hold them (float32), the weights of every
    region number."""
    return G.open_stream(run, "h0").grid.weight_km2


def _days(y, m):
    return float(G.days_in_month(y, m, "noleap"))


# The thicknesses as the files hold them (float32), so the top metre holds
# 0.3, 0.5 and 0.2 m of the layers to about 1e-8, not exactly.
DZ32 = np.array(DZ, dtype=np.float32).astype(np.float64)
TOPW = np.array([DZ32[0], DZ32[1], 1.0 - (DZ32[0] + DZ32[1])])
TOPW = TOPW / TOPW.sum()


def _top(a):
    """The top-1 m mean by hand: the first two layers whole, the third by its
    part above 1 m."""
    return TOPW[0] * a[:, 0] + TOPW[1] * a[:, 1] + TOPW[2] * a[:, 2]


def _month_cells(store, field, y, m):
    """The per-cell value of one month, by the rule: a rate's total (x 86,400 s
    x the month's days), a state's mean; from the stream the suite reads."""
    if field == "COL_FIRE_CLOSS":
        a = store[("h1", y)][field]
        d0 = sum(MD[:m - 1])
        return a[d0:d0 + MD[m - 1]].sum(axis=0) * 86400.0
    a = store[("h0", y)][field]
    x = _top(a)[m - 1] if field == "H2OSOI" else a[m - 1]
    return x * 86400.0 * _days(y, m) if field in ("GPP", "QCHARGE") else x


def _period_cells(store, field, keys):
    xs = [_month_cells(store, field, y, m) for y, m in keys]
    if field in ("GPP", "QCHARGE", "COL_FIRE_CLOSS"):
        return sum(xs)
    d = [_days(y, m) for y, m in keys]
    return sum(x * w for x, w in zip(xs, d)) / sum(d)


# ── the products, their precision and their periods ─────────────────────────

def test_the_products_their_precision_and_the_manifest(built):
    out, man = built["out"], built["man"]
    assert built["res"]["action"] == "built"
    for p in S.PRODUCTS:
        assert (out / f"{p}.nc").is_file()
    assert (out / S.MASKS_FILE).is_file()
    with netCDF4.Dataset(out / "maps_year.nc") as d:
        assert d["GPP"].dtype == np.float32 and d["GPP"].dimensions == ("year", "lat", "lon")
    with netCDF4.Dataset(out / "maps_season.nc") as d:
        assert d["GPP"].dtype == np.float32
        assert d["GPP"].dimensions == ("year", "season", "lat", "lon")
    for k in S.SERIES_KINDS:
        with netCDF4.Dataset(out / f"series_{k}.nc") as d:
            assert d["TOTECOSYSC"].dtype == np.float64
            assert d["TOTECOSYSC"].dimensions == ("region", k)
            assert d["TOTECOSYSC" + S.COVERED].dtype == np.float64
    # the keys the catalog reads, and what the products depend on
    assert man["SUITE_VERSION"] == S.SUITE_VERSION
    assert man["fields_left_out"] == {}
    assert man["fields"]["GPP"]["products"] == list(S.PRODUCTS)
    assert "series_day" not in man["fields"]["QCHARGE"]["products"]
    assert man["fields"]["QCHARGE"]["no_series_day"] == "no daily stream holds it"
    assert man["fields"]["COL_FIRE_CLOSS"]["stream"] == "h1"
    assert man["fields"]["GPP"]["stream"] == "h0" and man["fields"]["GPP"]["day_stream"] == "h1"
    assert man["fields"]["H2OSOI"]["levels"]["layer_weights"] == [0.3, 0.5, 0.2]
    assert sorted(man["history"]) == ["h0", "h1"]
    assert all({"name", "bytes", "mtime_ns"} == set(f) for f in man["history"]["h0"])
    assert len(man["masks_sha256"]) == 64 and man["band_edges_m"] == [1000.0, 1500.0]
    names = [r["name"] for r in man["regions"]]
    assert names[:2] == [A.VEGETATED, "box, vegetated"]
    assert "1000-1500 m" in names and "box, vegetated, below 1000 m" in names
    assert {"git_commit", "uncommitted_changes"} == set(man["code"])
    # periods: the first year's DJF and water year absent, with the reasons
    assert man["periods"]["year"] == ["2015", "2016"]
    assert man["periods"]["wateryear"] == ["WY2016"]
    assert "2015-DJF" not in man["periods"]["season"] and "2016-DJF" in man["periods"]["season"]
    assert len(man["periods"]["day"]) == 730 and len(man["periods"]["month"]) == 24
    assert "December 2014" in man["periods_absent"]["season"]["2015-DJF"]
    assert "its output starts in 2015" in man["periods_absent"]["wateryear"]["WY2015"]
    assert _labels(out / "series_wateryear.nc", "wateryear") == ["2016"] or \
        list(_nc(out / "series_wateryear.nc", "wateryear")) == [2016]
    assert np.all(np.isnan(_nc(out / "maps_season.nc", "GPP")[0, 0]))       # DJF 2015


def test_maps_are_rate_totals_and_state_means_cell_by_cell(built):
    out, st = built["out"], built["store"]
    for f in FIELDS:
        got = _nc(out / "maps_year.nc", f)
        for i, y in enumerate((2015, 2016)):
            want = _period_cells(st, f, [(y, m) for m in range(1, 13)])
            np.testing.assert_allclose(got[i], want.astype(np.float32), rtol=2e-7, err_msg=f)
    # the cell without a value one month has no year, the others do
    tec = _nc(out / "maps_year.nc", "TOTECOSYSC")
    assert np.isnan(tec[1][HOLE]) and np.isfinite(tec[0][HOLE])
    assert np.isfinite(np.delete(tec[1].ravel(), HOLE[0] * 5 + HOLE[1])).all()
    share = built["man"]["fields"]["TOTECOSYSC"]["maps_min_covered_share"]
    assert 0.9 < share < 1.0
    assert built["man"]["fields"]["GPP"]["maps_min_covered_share"] == 1.0


def test_season_and_water_year_maps_hold_the_months_before(built):
    out, st = built["out"], built["store"]
    for f in ("GPP", "TOTECOSYSC", "H2OSOI", "COL_FIRE_CLOSS"):
        sea = _nc(out / "maps_season.nc", f)
        djf = _period_cells(st, f, [(2015, 12), (2016, 1), (2016, 2)])
        np.testing.assert_allclose(sea[1, 0], djf.astype(np.float32), rtol=2e-7)
        jja = _period_cells(st, f, [(2015, 6), (2015, 7), (2015, 8)])
        np.testing.assert_allclose(sea[0, 2], jja.astype(np.float32), rtol=2e-7)
        wy = _nc(out / "maps_wateryear.nc", f)
        assert wy.shape[0] == 1
        want = _period_cells(st, f, [(2015, 10), (2015, 11), (2015, 12)]
                             + [(2016, m) for m in range(1, 10)])
        np.testing.assert_allclose(wy[0], want.astype(np.float32), rtol=2e-7)


def test_the_top_metre_weights_each_layer_by_its_thickness_inside_it():
    w = S.top_metre_weights(np.array(DZ)[:, None])
    np.testing.assert_allclose(w[:, 0], [0.3, 0.5, 0.2], rtol=0, atol=1e-15)
    w = S.top_metre_weights(np.array([[0.4], [0.6], [0.5]]))      # an interface at 1 m
    np.testing.assert_allclose(w[:, 0], [0.4, 0.6, 0.0], rtol=0, atol=1e-15)
    w = S.top_metre_weights(np.array([[0.2, np.nan], [0.3, 0.3]]))  # shallow; no thickness
    np.testing.assert_allclose(w[:, 0], [0.4, 0.6], rtol=0, atol=1e-15)
    assert np.all(np.isnan(w[:, 1]))


def test_month_season_and_year_series_are_the_tables_numbers(built):
    out, control = built["out"], built["control"]
    run = control["run_dir"]
    ms = R.MaskSet.from_netcdf(out / S.MASKS_FILE, G.open_stream(run, "h0").grid)
    table = A.run_values(run, ms, TABLE, [2015, 2016])        # read without the suite
    regions = _labels(out / "series_year.nc", "region")
    assert regions == ms.names()
    for kind, key, labels in (("month", "monthly", None), ("season", "seasonal", None),
                              ("year", "values", None)):
        path = out / f"series_{kind}.nc"
        labs = [str(x) for x in (_nc(path, kind) if kind == "year" else _labels(path, kind))]
        for f in TABLE:
            got = _nc(path, f)
            for j, reg in enumerate(regions):
                want = [table[key][reg][f][lab] for lab in labs]
                np.testing.assert_allclose(got[j], np.array(want, dtype=float), rtol=1e-12,
                                           atol=0, err_msg=f"{kind} {f} {reg}")
    assert set(_labels(out / "series_season.nc", "season")) == \
        set(table["seasonal"][A.VEGETATED]["GPP"])


def test_the_top_metre_series_follow_the_tables_rule(built):
    out, st = built["out"], built["store"]
    w = _grid_weights(built["control"]["run_dir"]).ravel()
    regions = _labels(out / "series_month.nc", "region")
    ms = R.MaskSet.from_netcdf(out / S.MASKS_FILE,
                               G.open_stream(built["control"]["run_dir"], "h0").grid)
    got_m = _nc(out / "series_month.nc", "H2OSOI")
    got_y = _nc(out / "series_year.nc", "H2OSOI")
    for j, reg in enumerate(regions):
        cells = ms[reg].ravel()
        months = []
        for y in (2015, 2016):
            for m in range(1, 13):
                x = _month_cells(st, "H2OSOI", y, m).ravel()
                months.append(float(np.sum(w[cells] * x[cells]) / np.sum(w[cells])))
        np.testing.assert_allclose(got_m[j], months, rtol=1e-12)
        for i, y in enumerate((2015, 2016)):
            d = [_days(y, m) for m in range(1, 13)]
            want = sum(x * dd for x, dd in zip(months[12 * i:12 * i + 12], d)) / sum(d)
            np.testing.assert_allclose(got_y[j, i], want, rtol=1e-12)


def test_water_year_series_are_the_months_totals_or_day_weighted_means(built):
    out = built["out"]
    labs = _labels(out / "series_month.nc", "month")
    keys = [f"2015-{m:02d}" for m in (10, 11, 12)] + [f"2016-{m:02d}" for m in range(1, 10)]
    days = [_days(int(k[:4]), int(k[5:])) for k in keys]
    for f in FIELDS:
        mon = _nc(out / "series_month.nc", f)
        wy = _nc(out / "series_wateryear.nc", f)
        cols = [labs.index(k) for k in keys]
        if f in ("GPP", "QCHARGE", "COL_FIRE_CLOSS"):
            want = sum(mon[:, c] for c in cols)
        else:
            want = sum(mon[:, c] * d for c, d in zip(cols, days)) / sum(days)
        np.testing.assert_allclose(wy[:, 0], want, rtol=1e-12, err_msg=f)


def test_day_series_are_area_weighted_day_totals_and_means(built):
    out, st = built["out"], built["store"]
    w = _grid_weights(built["control"]["run_dir"]).ravel()
    regions = _labels(out / "series_day.nc", "region")
    ms = R.MaskSet.from_netcdf(out / S.MASKS_FILE,
                               G.open_stream(built["control"]["run_dir"], "h0").grid)
    labs = _labels(out / "series_day.nc", "day")
    assert labs[0] == "2015-01-01" and labs[-1] == "2016-12-31" and len(labs) == 730
    for f in ("GPP", "TOTECOSYSC", "H2OSOI", "COL_FIRE_CLOSS"):
        got = _nc(out / "series_day.nc", f)
        cov = _nc(out / "series_day.nc", f + S.COVERED)
        x = np.concatenate([st[("h1", y)][f] for y in (2015, 2016)])
        x = _top(x) if f == "H2OSOI" else x
        x = x.reshape(730, -1) * (86400.0 if f in ("GPP", "COL_FIRE_CLOSS") else 1.0)
        for j, reg in enumerate(regions):
            c = ms[reg].ravel()
            ok = np.isfinite(x[:, c])
            want = np.where(ok, x[:, c], 0.0) @ w[c] / (ok @ w[c])
            np.testing.assert_allclose(got[j], want, rtol=1e-12, err_msg=f"{f} {reg}")
            np.testing.assert_allclose(cov[j], ok @ w[c] / w[c].sum(), rtol=1e-12)
    # the day without a value in one cell: covered below 1 there, 1 elsewhere
    cov = _nc(out / "series_day.nc", "TOTECOSYSC" + S.COVERED)[0]
    hole = 365 + HOLE_DAY[1]
    assert cov[hole] < 1.0 and np.all(np.delete(cov, hole) == 1.0)
    assert "COL_FIRE_CLOSS" in _nc_vars(out / "series_day.nc")
    assert "QCHARGE" not in _nc_vars(out / "series_day.nc")


def _nc_vars(path):
    with netCDF4.Dataset(path) as d:
        return list(d.variables)


def test_each_year_file_of_each_stream_is_read_in_one_pass(tmp_path, monkeypatch):
    """The table's passes feed the suite (domain_suite._Tee): a build reads
    each (stream, year) once, the daily fields riding on the pass the table
    makes for COL_FIRE_CLOSS."""
    control, _ = make_control(tmp_path, [2015, 2016])
    real = G.Stream.read_block_pass
    calls = []

    def counted(self, vars, year, *a, **k):
        calls.append((self.name, int(year), tuple(sorted(vars))))
        return real(self, vars, year, *a, **k)
    monkeypatch.setattr(G.Stream, "read_block_pass", counted)
    S.build(control, tmp_path / "suite", fields=FIELDS)
    assert sorted((s, y) for s, y, _ in calls) == [("h0", 2015), ("h0", 2016),
                                                    ("h1", 2015), ("h1", 2016)]
    for s, y, vs in calls:
        assert set(vs) == (set(H0) if s == "h0" else set(H1)), (s, y, vs)


# ── versioning ──────────────────────────────────────────────────────────────

def _mtimes(folder):
    return {p.name: p.stat().st_mtime_ns for p in Path(folder).iterdir()}


def test_a_second_build_is_a_no_op(built):
    before = _mtimes(built["out"])
    res = S.build(built["control"], built["out"], fields=FIELDS)
    assert res["action"] == "reused" and res["made"] == {}
    assert _mtimes(built["out"]) == before


def _slices(out):
    """Every product's values, by file and variable."""
    got = {}
    for p in sorted(Path(out).glob("*.nc")):
        if p.name == S.MASKS_FILE:
            continue
        with netCDF4.Dataset(p) as d:
            d.set_auto_mask(False)
            for v in d.variables:
                a = np.asarray(d[v][:])
                got[(p.name, v)] = a
    return got


def test_adding_a_year_makes_exactly_the_products_that_cross_into_it(tmp_path):
    control, _ = make_control(tmp_path / "a", [2015, 2016])
    out = tmp_path / "a" / "suite"
    assert S.build(control, out, fields=FIELDS)["action"] == "built"
    old = _slices(out)
    write_years(control["run_dir"], [2017])
    res = S.build(control, out, fields=FIELDS)
    assert res["action"] == "added"
    made = res["made"]
    assert made["year"] == ["2017"] and made["wateryear"] == ["WY2017"]
    assert made["season"] == ["2017-DJF", "2017-MAM", "2017-JJA", "2017-SON"]
    assert made["month"] == [f"2017-{m:02d}" for m in range(1, 13)]
    assert len(made["day"]) == 365 and made["day"][0] == "2017-01-01"
    new = _slices(out)
    # nothing that was there changed
    for (f, v), a in old.items():
        b = new[(f, v)]
        fixed = ("lat", "lon", "weight_km2", "region") + (("season",) if f.startswith("maps_")
                                                           else ())
        if a.ndim and v not in fixed:
            axis = 0 if f.startswith("maps_") else (a.ndim - 1)
            b = np.take(b, range(a.shape[axis]), axis=axis)
        if a.dtype.kind in "fc":
            np.testing.assert_array_equal(a, b, err_msg=f"{f} {v}")
        else:
            assert a.tolist() == b.tolist(), (f, v)
    # and what was added is what a build of the three years makes
    full, _ = make_control(tmp_path / "b", [2015, 2016, 2017])
    out2 = tmp_path / "b" / "suite"
    S.build(full, out2, fields=FIELDS)
    ref = _slices(out2)
    for key, a in ref.items():
        b = new[key]
        assert a.shape == b.shape, key
        if a.dtype.kind == "f" and key[0].startswith("series_"):
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=0, err_msg=str(key))
        elif a.dtype.kind == "f":
            np.testing.assert_array_equal(b, a, err_msg=str(key))
        else:
            assert a.tolist() == b.tolist(), key
    m1, m2 = S.load(out), S.load(out2)
    for k in ("periods", "periods_absent", "fields", "regions", "masks_sha256", "missing",
              "problems", "band_edges_m"):          # not "files": HDF5 lays out appends otherwise
        assert m1[k] == m2[k], k
    assert m1["last_build"]["action"] == "added"
    # and a third build is a no-op again
    before = _mtimes(out)
    assert S.build(control, out, fields=FIELDS)["action"] == "reused"
    assert _mtimes(out) == before


def test_a_changed_history_file_rebuilds(tmp_path):
    control, _ = make_control(tmp_path, [2015, 2016])
    out = tmp_path / "suite"
    S.build(control, out, fields=FIELDS)
    p = sorted(Path(control["run_dir"]).glob("*.h0.2015*"))[0]
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    res = S.build(control, out, fields=FIELDS)
    assert res["action"] == "built" and "changed" in res["why"]
    assert S.load(out)["last_build"]["action"] == "built"
    # other fields asked for also rebuild
    res = S.build(control, out, fields=["GPP"])
    assert res["action"] == "built" and "fields_asked" in res["why"]
    assert list(S.load(out)["fields"]) == ["GPP"]


def test_the_controls_view_gives_the_suite_in_one_line(built):
    import domain_controls as DC
    line = DC.suite_line({"suite_dir": str(built["out"])})
    assert line.startswith(f"version {S.SUITE_VERSION}, built ") and " UTC, " in line
    assert "GB: 5 fields, 2015-2016; maps by year, water year and season" in line
    assert len(line) <= 200
    assert DC.suite_line({}) == "not built (the registry names no suite_dir)"
    assert DC.suite_line({"suite_dir": str(built["root"] / "nowhere")}) == "not built"
    entry = {"name": "synth", "suite_dir": str(built["out"])}
    assert DC.summary(entry, None)["suite"] == line


# ── the elevation bands (THE STUDY'S ELEVATION BANDS, domain_analysis) ──────

def test_band_edges_are_the_controls_or_every_500_m_of_its_own_range(tmp_path):
    control, _ = make_control(tmp_path, [2015, 2016])
    out = tmp_path / "suite"
    S.build(control, out, fields=["GPP"])
    man = S.load(out)
    # the fake's vegetated cells span 550-1600 m: edges at 1000 and 1500 m
    assert man["band_edges_m"] == [1000.0, 1500.0]
    assert man["band_edges_from"].startswith("every 500 m across the vegetated cells")
    names = [r["name"] for r in man["regions"]]
    assert "1500 m and above" in names and "1500-2000 m" not in names
    assert "box, vegetated, below 1000 m" in names
    nc = {p.name: p.stat().st_mtime_ns for p in out.glob("*.nc")}
    # the same edges named in the registry: the products are reused, the
    # manifest says where the edges now come from
    control["elevation_band_edges"] = [1000, 1500]
    res = S.build(control, out, fields=["GPP"])
    assert res["action"] == "reused" and "band_edges_from" in res["why"]
    man = S.load(out)
    assert man["band_edges_from"] == "the control's registry entry (elevation_band_edges)"
    assert man["manifest_refreshed"]["keys"] == ["band_edges_from"]
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.nc")} == nc
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    assert S.build(control, out, fields=["GPP"])["action"] == "reused"
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before
    # other edges are other regions: everything is built again
    control["elevation_band_edges"] = [1000, 1500, 2000]
    res = S.build(control, out, fields=["GPP"])
    assert res["action"] == "built"
    names = [r["name"] for r in S.load(out)["regions"]]
    assert "1500-2000 m" in names and "1500 m and above" not in names
