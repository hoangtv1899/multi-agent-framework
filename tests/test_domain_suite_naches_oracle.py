"""The Naches control's suite against the regional table it must equal.

    pytest tests/test_domain_suite_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the reference table ("reference", the
Demo 1 regional table of 2015-2023), the control's run directory ("run_dir")
and its built suite ("suite_dir"). The month, season and year series are the
table's own numbers (domain_analysis.run_values) and must equal them to 1e-12
relative; a water year is its months' total or day-weighted mean, from the
table's months, to the same; a region's mean of a year map, made from float32
cells, equals the year series within the numbers check's tolerance (RTOL 1e-5,
ATOL 1e-9). A second build reads no year file and writes nothing. Opening the
streams reads every file's time stamps, so the stream index cache points at a
temporary directory for the session, as the catalog's oracle test does.
"""
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_suite as S
import gridded as G
from agents.analysis import numbers_check as N

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
FIELDS = ("GPP", "TOTECOSYSC", "H2OSNO", "EFLX_LH_TOT", "QRUNOFF")
REGIONS = ("vegetated cells", "watershed, vegetated")


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    need = ("reference", "run_dir", "suite_dir")
    if not all(c.get(k) for k in need) or not Path(c["run_dir"]).is_dir() or \
            not (Path(c["suite_dir"]) / S.MANIFEST).is_file():
        pytest.skip("no naches_suite block with reference, run_dir and a built suite_dir")
    ref = json.loads(Path(c["reference"]).read_text())
    return {"suite": Path(c["suite_dir"]), "run_dir": c["run_dir"], "ref": ref,
            "table": ref["runs"]["control"], "man": S.load(c["suite_dir"])}


def _read(path, var):
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        return np.asarray(d[var][:])


def _labels(path, kind):
    with netCDF4.Dataset(path) as d:
        return [str(x) for x in np.asarray(d[kind][:], dtype=object)]


def test_the_suite_is_the_controls(cfg):
    man = cfg["man"]
    assert man["SUITE_VERSION"] == S.SUITE_VERSION
    assert Path(man["run_dir"]).resolve() == Path(cfg["run_dir"]).resolve()
    assert man["periods"]["year"] == [str(y) for y in range(2015, 2024)]
    assert man["periods"]["wateryear"] == [S.wy_label(y) for y in range(2016, 2024)]
    assert len(man["periods"]["season"]) == 35 and len(man["periods"]["month"]) == 108
    assert len(man["periods"]["day"]) == 9 * 365
    assert set(man["periods_absent"]) == {"season", "wateryear"}
    assert list(man["periods_absent"]["season"]) == ["2015-DJF"]
    assert list(man["periods_absent"]["wateryear"]) == ["WY2015"]
    assert len(man["fields"]) == 52 and man["fields_left_out"] == {}
    for reg in REGIONS:
        r = next(x for x in man["regions"] if x["name"] == reg)
        assert r["cells"] == cfg["ref"]["regions"][reg]["cells"]


def test_month_season_and_year_series_equal_the_table(cfg):
    t = cfg["table"]
    for kind, key in (("month", "monthly"), ("season", "seasonal"), ("year", "values")):
        path = cfg["suite"] / f"series_{kind}.nc"
        regs = _labels(path, "region")
        labs = ([str(x) for x in _read(path, kind)] if kind == "year"
                else _labels(path, kind))
        for f in FIELDS:
            a = _read(path, f)
            for reg in REGIONS:
                want = t[key][reg][f]
                assert sorted(want) == sorted(labs), (kind, f, reg)
                got = a[regs.index(reg)]
                np.testing.assert_allclose(got, [want[lab] for lab in labs], rtol=1e-12,
                                           atol=0, err_msg=f"{kind} {f} {reg}")


def test_water_year_series_are_the_tables_months(cfg):
    t = cfg["table"]
    path = cfg["suite"] / "series_wateryear.nc"
    regs = _labels(path, "region")
    wys = [int(x) for x in _read(path, "wateryear")]
    assert wys == list(range(2016, 2024))
    for f in FIELDS:
        rate = t["fields"][f]["how"] == "sum"
        a = _read(path, f)
        for reg in REGIONS:
            mon = t["monthly"][reg][f]
            for i, y in enumerate(wys):
                keys = [(y - 1, m) for m in (10, 11, 12)] + [(y, m) for m in range(1, 10)]
                xs = [mon[f"{yy}-{m:02d}"] for yy, m in keys]
                if rate:
                    want = sum(xs)
                else:
                    d = [float(G.days_in_month(yy, m, "noleap")) for yy, m in keys]
                    want = sum(x * w for x, w in zip(xs, d)) / sum(d)
                got = float(a[regs.index(reg), i])
                assert abs(got - want) <= 1e-12 * abs(want), (f, reg, y, got, want)


def test_a_year_maps_region_means_are_the_year_series(cfg):
    suite = cfg["suite"]
    with netCDF4.Dataset(suite / "maps_year.nc") as d:
        w = np.asarray(d["weight_km2"][:]).ravel()
        grid_shape = d["weight_km2"].shape
    masks = {}                # read straight from masks.nc: no stream need be opened
    with netCDF4.Dataset(suite / S.MASKS_FILE) as d:
        for v in d.variables.values():
            if "region" in v.ncattrs():
                masks[v.region] = np.asarray(v[:]).astype(bool)
    regs = _labels(suite / "series_year.nc", "region")
    assert regs == list(masks) and all(m.shape == tuple(grid_shape) for m in masks.values())
    worst, bad = 0.0, []
    for f in sorted(cfg["man"]["fields"]):
        maps = _read(suite / "maps_year.nc", f).astype(np.float64)
        ser = _read(suite / "series_year.nc", f)
        for j, reg in enumerate(regs):
            cells = masks[reg].ravel()
            for i in range(maps.shape[0]):
                x = maps[i].ravel()[cells]
                ok = np.isfinite(x)
                mean = float(np.sum(w[cells][ok] * x[ok]) / np.sum(w[cells][ok]))
                s = float(ser[j, i])
                d = abs(mean - s)
                worst = max(worst, d / abs(s) if s else d)
                if not d <= N.RTOL * abs(s) + N.ATOL:
                    bad.append((f, reg, 2015 + i, mean, s))
    assert not bad, bad[:5]
    print(f"largest relative difference, map means against series: {worst:.2e}")


@pytest.fixture(scope="module")
def index_cache(tmp_path_factory):
    old = os.environ.get(G.INDEX_CACHE_ENV)
    os.environ[G.INDEX_CACHE_ENV] = str(tmp_path_factory.mktemp("stream_index"))
    yield
    if old is None:
        os.environ.pop(G.INDEX_CACHE_ENV, None)
    else:
        os.environ[G.INDEX_CACHE_ENV] = old


def test_a_second_build_is_a_no_op(cfg, index_cache):
    import domain_controls
    try:
        e = domain_controls.get("naches")
    except domain_controls.ControlError as err:
        pytest.skip(f"the registry has no usable naches control: {err}")
    before = {p.name: p.stat().st_mtime_ns for p in cfg["suite"].iterdir()}
    res = S.build(e, cfg["suite"])
    assert res["action"] == "reused", res
    assert {p.name: p.stat().st_mtime_ns for p in cfg["suite"].iterdir()} == before
