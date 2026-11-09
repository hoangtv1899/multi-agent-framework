"""The method catalog on small fake grids with known answers.

    pytest tests/test_domain_methods.py -v

THE METHODS' CONTRACT TESTS (2026-11-05). Every case of the plan's synthetic
list: a seasonal flux, a rare-event series, a drifting store, a bounded index,
a cell with no snow, a cell with weak leaf seasonality, a cell without a
dormant season, a cell with two growth peaks, missing records, negative daily
runoff, tied peaks, snow that returns after the melt-out; and the screen's
rules (zero change, a near-constant normal in kelvin and in Celsius, an
experiment's score over the control's spread), a water year without its
October-December, and every alternative setting beside the default. Each
answer is worked out here by hand or by separate code, never by calling the
method twice. A suite is written to a temporary folder in the layout
domain_suite.build writes (suite.json, masks.nc, series_*.nc, maps_*.nc).
"""
import json
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

import domain_methods as M
import gridded as G

YEARS = list(range(2015, 2022))
NORMAL = [2016, 2017, 2018, 2019, 2020]
SEASONS = ("DJF", "MAM", "JJA", "SON")
ABSENT = {"season": {"2015-DJF": "DJF 2015 needs December 2014"},
          "wateryear": {"WY2015": "WY2015 needs October-December 2014"}}
SHAPE = (2, 3)
W = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
REGIONS = {"all": np.ones(SHAPE, bool),
           "left": np.array([[1, 1, 0], [1, 0, 0]], bool),
           "below 1000 m": np.array([[1, 0, 0], [1, 1, 0]], bool),
           "1000 m and above": np.array([[0, 1, 1], [0, 0, 1]], bool)}


def labels(kind):
    if kind == "year":
        return [str(y) for y in YEARS]
    if kind == "wateryear":
        return [f"WY{y}" for y in YEARS if f"WY{y}" not in ABSENT["wateryear"]]
    if kind == "season":
        return [f"{y}-{s}" for y in YEARS for s in SEASONS if f"{y}-{s}" not in ABSENT["season"]]
    return [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]


def write_suite(d, series, maps=None, units=None, covered=None):
    """series[field](kind, label, region) -> value; maps[field](label) -> array."""
    d.mkdir(parents=True, exist_ok=True)
    fields = sorted(series)
    units = units or {}
    man = {"SUITE_VERSION": 1, "band_edges_m": [1000.0],
           "fields": {f: {"how": "total", "products": [],
                          "units": {k: units.get(f, "u") for k in M.KINDS}} for f in fields},
           "periods": {k: labels(k) for k in M.KINDS}, "periods_absent": ABSENT}
    (d / "suite.json").write_text(json.dumps(man))
    with netCDF4.Dataset(d / "masks.nc", "w") as n:
        n.createDimension("lat", SHAPE[0])
        n.createDimension("lon", SHAPE[1])
        n.createVariable("weight_km2", "f8", ("lat", "lon"))[:] = W
        for i, (name, m) in enumerate(REGIONS.items()):
            v = n.createVariable(f"mask_{i}", "i1", ("lat", "lon"))
            v[:] = m.astype("i1")
            v.region = name
            v.rule = "test"
            v.sources = "[]"
    regs = list(REGIONS)
    for kind in M.KINDS:
        labs = labels(kind)
        with netCDF4.Dataset(d / f"series_{kind}.nc", "w") as n:
            n.createDimension("region", len(regs))
            n.createDimension(kind, len(labs))
            rv = n.createVariable("region", str, ("region",))
            for i, r in enumerate(regs):
                rv[i] = r
            if kind in ("year", "wateryear"):
                n.createVariable(kind, "i4", (kind,))[:] = [int(x.replace("WY", "")) for x in labs]
            else:
                kv = n.createVariable(kind, str, (kind,))
                for i, x in enumerate(labs):
                    kv[i] = x
            for f in fields:
                a = np.array([[series[f](kind, lab, r) for lab in labs] for r in regs], float)
                n.createVariable(f, "f8", ("region", kind), fill_value=np.nan)[:] = a
                c = np.array([[(covered or (lambda *a: 1.0))(f, lab, r) for lab in labs]
                              for r in regs], float)
                n.createVariable(f + "_covered", "f8", ("region", kind), fill_value=np.nan)[:] = c
    for kind in ("year", "wateryear", "season"):
        with netCDF4.Dataset(d / f"maps_{kind}.nc", "w") as n:
            n.createDimension("lat", SHAPE[0])
            n.createDimension("lon", SHAPE[1])
            if kind == "season":
                n.createDimension("year", len(YEARS))
                n.createDimension("season", 4)
                n.createVariable("year", "i4", ("year",))[:] = YEARS
                sv = n.createVariable("season", str, ("season",))
                for i, s in enumerate(SEASONS):
                    sv[i] = s
                dims = ("year", "season", "lat", "lon")
            else:
                ys = [int(x.replace("WY", "")) for x in labels(kind)]
                n.createDimension(kind, len(ys))
                n.createVariable(kind, "i4", (kind,))[:] = ys
                dims = (kind, "lat", "lon")
            for f, fn in (maps or {}).items():
                v = n.createVariable(f, "f4", dims, fill_value=np.nan)
                if kind == "season":
                    for i, y in enumerate(YEARS):
                        for j, s in enumerate(SEASONS):
                            lab = f"{y}-{s}"
                            v[i, j] = np.nan if lab in ABSENT["season"] else fn(lab)
                else:
                    for i, y in enumerate(ys):
                        v[i] = fn(M.period_label(kind, y))
    return M.Suite(d)


def parts(label):
    return M.parse_period(label)


def seasonal(kind, label, region, amp=1.0, base=100.0, bump=None):
    """A seasonal flux: a fixed pattern by month or season, a little noise by
    year, and the region scaling it."""
    k, y, p = parts(label)
    shape = {"month": lambda: 10.0 + 8.0 * np.sin(np.pi * (p - 1) / 11.0),
             "season": lambda: {"DJF": 5.0, "MAM": 40.0, "JJA": 60.0, "SON": 20.0}[p],
             "year": lambda: base, "wateryear": lambda: base * 0.98}[k]()
    noise = {2015: 3.0, 2016: -1.0, 2017: 2.0, 2018: -2.5, 2019: 1.5, 2020: 0.0, 2021: -4.0}[y]
    out = (shape + amp * noise) * (1.0 if region == "all" else 0.5)
    return out + (bump(k, y, p) if bump else 0.0)


def _note(group, typ, scale, percent):
    """A test's field note, with its percent policy (review round 1)."""
    return {"group": group, "type": typ, "scale": scale,
            "percent": {"allowed": percent, "why": "the test's own policy"}}


NOTES = {"FLUX": _note("carbon in and out", "flux", "ratio", True),
         "FLUX2": _note("carbon in and out", "flux", "ratio", True),
         "FLUX3": _note("carbon in and out", "flux", "ratio", True),
         "FLUX4": _note("carbon in and out", "flux", "ratio", True),
         "TK": _note("energy", "state", "interval", False),
         "TC": _note("energy", "state", "interval", False),
         "FAREA_BURNED": _note("fire", "event", "ratio", False),
         "COL_FIRE_CLOSS": _note("fire", "event", "ratio", False),
         "STORE": _note("carbon stores", "store", "ratio", True),
         "BIDX": _note("water to the air", "bounded", "ratio", False),
         "NET": _note("carbon in and out", "flux", "interval", False)}


def temp_k(kind, label, region):
    """A temperature whose July normal barely moves: every normal July the same
    to 1e-4 K, while the months between them span 20 K."""
    k, y, p = parts(label)
    if k == "month":
        if p == 7:
            return 295.0 + (1e-4 if y in (2017, 2019) else 0.0) + (2.0 if y == 2021 else 0.0)
        return 270.0 + 20.0 * np.sin(np.pi * (p - 1) / 11.0) + 0.5 * (y - 2018)
    return 280.0 + 0.3 * (y - 2018) + {"DJF": -8.0, "MAM": 0.0, "JJA": 10.0, "SON": 1.0}.get(p, 0.0)


def fire(kind, label, region):
    """A rare event: burned fraction in a few periods only."""
    k, y, p = parts(label)
    if k == "month":
        return {(2017, 8): 0.002, (2019, 8): 0.001, (2021, 8): 0.005, (2021, 9): 0.0005}.get((y, p), 0.0)
    if k == "season":
        return {(2017, "JJA"): 0.002, (2019, "JJA"): 0.001, (2021, "JJA"): 0.005,
                (2021, "SON"): 0.0005}.get((y, p), 0.0)
    return {2017: 0.002, 2019: 0.001, 2021: 0.0055}.get(y, 0.0)


def store(kind, label, region):
    k, y, p = parts(label)
    return 1000.0 + 10.0 * (y - 2015) + (0.1 * p if k == "month" else 0.0)


def bidx(kind, label, region):
    k, y, p = parts(label)
    return 0.6 + 0.05 * np.sin(y) + (0.01 * p if k == "month" else 0.0)


def closs(kind, label, region):
    return 100.0 * fire(kind, label, region)


def flux_map(label):
    k, y, p = parts(label)
    base = {"year": 100.0, "wateryear": 98.0, "season": {"DJF": 5.0, "MAM": 40.0, "JJA": 60.0,
                                                          "SON": 20.0}.get(p)}[k]
    a = base + np.arange(6, dtype=float).reshape(SHAPE) + (y - 2018) * np.array(
        [[1.0, -1.0, 0.5], [2.0, 0.0, -0.5]])
    if y == 2017:
        a[1, 2] = np.nan            # one cell without a value in one normal year
    return a


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    d = tmp_path_factory.mktemp("suite")
    series = {"FLUX": seasonal, "TK": temp_k,
              "TC": lambda k, lab, r: temp_k(k, lab, r) - 273.15,
              "FAREA_BURNED": fire, "COL_FIRE_CLOSS": closs, "STORE": store, "BIDX": bidx,
              "NET": lambda k, lab, r: seasonal(k, lab, r) - 50.0,
              "FLUX2": lambda k, lab, r: 3.0 * seasonal(k, lab, r, amp=2.0),
              "FLUX3": lambda k, lab, r: seasonal(k, lab, r, amp=0.5),
              "FLUX4": lambda k, lab, r: seasonal(k, lab, r, amp=0.2)}
    return write_suite(d, series, maps={"FLUX": flux_map})


def by_hand(suite, field, region, label, normal=NORMAL, skip=()):
    k, y, p = parts(label)
    v = suite.value(field, region, label)[0]
    nv = [suite.value(field, region, M.period_label(k, z, p))[0] for z in normal if z not in skip]
    nv = [x for x in nv if x is not None]
    return v, np.array(nv)


# ── anomaly and difference ───────────────────────────────────────────────────

def test_anomaly_series_and_percent_rules(suite):
    r = M.anomaly(suite, "FLUX", "2021-07", NORMAL, region="all", notes=NOTES)
    v, nv = by_hand(suite, "FLUX", "all", "2021-07")
    assert r["value"] == v and r["normal"] == pytest.approx(nv.mean(), rel=1e-15)
    assert r["anomaly"] == pytest.approx(v - nv.mean(), rel=1e-12)
    assert r["percent"] == pytest.approx(100 * (v - nv.mean()) / nv.mean(), rel=1e-12)
    assert r["support"]["n"] == 5 and r["support"]["sd"] == pytest.approx(nv.std(ddof=1))
    assert r["method"] == "anomaly" and r["version"] == 2 and r["settings"] == {"near_zero_share": 0.01}
    # the regional table's rule: a normal winter holding an event year's December
    # is left out, and a water year holding its October-December likewise
    r = M.anomaly(suite, "FLUX", "2015-JJA", NORMAL, region="all", notes=NOTES)
    assert r["support"]["n"] == 5 and not r["normal_left_out"]
    r = M.anomaly(suite, "FLUX", "2021-DJF", NORMAL, region="all", notes=NOTES,
                  event_years=[2015, 2021])
    assert sorted(r["normal_left_out"]) == [2016] and r["support"]["n"] == 4
    v, nv = by_hand(suite, "FLUX", "all", "2021-DJF", skip=(2016,))
    assert r["normal"] == pytest.approx(nv.mean(), rel=1e-15)
    r = M.anomaly(suite, "FLUX", "2015", NORMAL, region="all", notes=NOTES)
    assert not r["normal_left_out"]
    r = M.anomaly(suite, "FLUX", "WY2021", NORMAL, region="all", notes=NOTES, event_years=[2015])
    assert sorted(r["normal_left_out"]) == [2016]
    # a period the run lacks, and a normal that does not exist
    r = M.anomaly(suite, "FLUX", "2015-DJF", NORMAL, region="all", notes=NOTES)
    assert r["anomaly"] is None and "absent" in r["why"]
    r = M.anomaly(suite, "FLUX", "2021-07", [2010, 2011], region="all", notes=NOTES)
    assert r["anomaly"] is None and r["why"].startswith("not estimable")


def test_anomaly_no_percent_for_interval_event_bounded_or_near_zero(suite):
    for f in ("TK", "TC", "NET", "FAREA_BURNED", "BIDX"):
        r = M.anomaly(suite, f, "2021-08", NORMAL, region="all", notes=NOTES)
        assert r["anomaly"] is not None and r["percent"] is None and r["percent_why_not"], f
    # a ratio-scale field whose normal is near zero beside the range of its normals
    notes = dict(NOTES, COL_FIRE_CLOSS=_note("fire", "flux", "ratio", True))
    r = M.anomaly(suite, "COL_FIRE_CLOSS", "2021-01", NORMAL, region="all", notes=notes)
    assert r["normal"] == 0.0 and r["percent"] is None and "near zero" in r["percent_why_not"]
    r = M.anomaly(suite, "COL_FIRE_CLOSS", "2021-08", NORMAL, region="all", notes=notes)
    rr = max(np.mean([fire("month", f"{y}-{m:02d}", "all") for y in NORMAL]) for m in range(1, 13))
    assert r["reference_range"] == pytest.approx(100 * rr) and r["percent"] is not None
    # the store is ratio-scale: it has a percent
    r = M.anomaly(suite, "STORE", "2021", NORMAL, region="all", notes=NOTES)
    assert r["percent"] == pytest.approx(100 * r["anomaly"] / r["normal"])


def test_anomaly_map_per_cell(suite):
    r = M.anomaly(suite, "FLUX", "2021-JJA", NORMAL, notes=NOTES)
    stack = np.stack([flux_map(f"{y}-JJA") for y in NORMAL]).astype(np.float32).astype(float)
    normal = stack.mean(axis=0)
    want = flux_map("2021-JJA").astype(np.float32).astype(float) - normal
    assert np.isnan(r["anomaly"][1, 2]) and np.isnan(r["normal"][1, 2])
    ok = np.isfinite(want)
    assert np.allclose(r["anomaly"][ok], want[ok], rtol=1e-12, atol=0)
    assert np.allclose(r["percent"][ok], 100 * want[ok] / normal[ok], rtol=1e-12)
    assert r["support"]["n"] == 5
    with pytest.raises(M.MethodError):
        M.anomaly(suite, "FLUX", "2021-07", NORMAL, notes=NOTES)       # no month maps


def test_difference_run_minus_control(suite, tmp_path):
    run = write_suite(tmp_path / "run", {"FLUX": lambda k, lab, r: 1.1 * seasonal(k, lab, r),
                                         "TK": lambda k, lab, r: temp_k(k, lab, r) + 2.0},
                      maps={"FLUX": lambda lab: 1.1 * flux_map(lab)})
    r = M.difference(run, suite, "FLUX", "2021-07", region="all", notes=NOTES)
    c = suite.value("FLUX", "all", "2021-07")[0]
    assert r["difference"] == pytest.approx(0.1 * c, rel=1e-12)
    assert r["percent"] == pytest.approx(10.0, rel=1e-12)
    r = M.difference(run, suite, "TK", "2021-07", region="all", notes=NOTES)
    assert r["difference"] == pytest.approx(2.0, rel=1e-9) and r["percent"] is None
    r = M.difference(run, suite, "FLUX", "2021", notes=NOTES)
    ok = np.isfinite(r["difference"])
    assert np.allclose(r["percent"][ok], 10.0, rtol=1e-5)


# ── the timing measures ──────────────────────────────────────────────────────

def wy(h2osno=None, qrunoff=None, record_start=0):
    n = 365
    data = {"H2OSNO": np.zeros((n, 1)) if h2osno is None else np.asarray(h2osno, float),
            "QRUNOFF": np.ones((n, 1)) if qrunoff is None else np.asarray(qrunoff, float)}
    return M.DailyPeriod("wateryear", 2016, -91, data, record_start=record_start)


def yr(tlai=None, gpp=None):
    n = 365
    return M.DailyPeriod("year", 2016, 1,
                         {"TLAI": np.zeros((n, 1)) if tlai is None else np.asarray(tlai, float),
                          "GPP": np.zeros((n, 1)) if gpp is None else np.asarray(gpp, float)})


def cols(*cs):
    return np.stack(cs, axis=1)


def snowpack(peak_day, melt_day, peak=300.0, start=20):
    s = np.zeros(365)
    s[start:peak_day + 1] = np.linspace(2.0, peak, peak_day + 1 - start)
    s[peak_day + 1:melt_day] = np.linspace(peak, 1.5, melt_day - peak_day - 1)
    return s                     # 0 from melt_day on: below 1 mm from melt_day


def d(res, key="date", v="default"):
    return res["variants"][v][key]


def test_snow_peak_dates_ties_and_no_snow():
    s1 = snowpack(150, 230)
    s2 = np.zeros(365)
    s2[100:200] = 50.0                    # a flat peak: ties go to the earliest day
    s3 = np.full(365, 0.9)                # never 1 mm: no snow
    r = M.snow_peak(wy(cols(s1, s2, s3)))
    assert d(r).tolist()[:2] == [150 - 91, 100 - 91] and np.isnan(d(r)[2])
    assert d(r, "reason").tolist() == [0, 0, M.CODE["no_snow"]]
    assert list(r["variants"]) == ["default"]          # snow_peak has no alternative


def test_snow_gone_searched_from_the_peak():
    s = snowpack(170, 240)
    s[30:60] = 40.0                       # an early snowpack that melts mid-winter...
    s[60:80] = 0.0                        # ...and stays off for 20 days before the main one
    r = M.snow_gone(wy(cols(s)))
    assert d(r)[0] == 240 - 91            # the melt-out after the peak, not day 60
    assert d(r, "flags")[0] == 0


def test_snow_that_returns_is_flagged():
    s = snowpack(150, 230)
    s[280:300] = 30.0                     # back above 1 mm for 20 days
    s2 = snowpack(150, 230)
    s2[280:290] = 30.0                    # back for 10 days only: not a return
    r = M.snow_gone(wy(cols(s, s2)))
    assert d(r).tolist() == [230 - 91, 230 - 91]
    assert d(r, "flags").tolist() == [M.FLAG["snow_returns"], 0]
    assert d(r, "flags", "days=7").tolist() == [M.FLAG["snow_returns"], M.FLAG["snow_returns"]]


def test_cell_with_no_snow():
    s = np.full(365, 0.5)
    s2 = np.full(365, 3.0)                # below 5 mm all year: no snow at the 5 mm setting
    s2[100:] = 0.0
    r = M.snow_gone(wy(cols(s, s2)))
    assert d(r, "reason").tolist() == [M.CODE["no_snow"], 0]
    assert d(r, "reason", "threshold_mm=5").tolist() == [M.CODE["no_snow"], M.CODE["no_snow"]]
    s3 = np.full(365, 10.0)               # never below: snow persists
    r = M.snow_gone(wy(cols(s3)))
    assert d(r, "reason").tolist() == [M.CODE["snow_persists"]]


def test_water_year_without_its_start():
    s = snowpack(150, 230)
    s[:92] = np.nan                       # October-December before the run's records
    q = np.ones(365)
    q[:92] = np.nan
    p = wy(cols(s), cols(q), record_start=92)
    assert p.starts_before_record
    r = M.snow_peak(p)
    assert np.isnan(d(r)[0]) and d(r, "reason")[0] == M.CODE["before_record"]
    r = M.snow_gone(p)
    assert d(r)[0] == 230 - 91 and d(r, "flags")[0] & M.FLAG["partial_record"]
    r = M.half_runoff(p)
    assert d(r, "reason")[0] == M.CODE["before_record"]
    # a partial-record date never enters a normal
    full = M.snow_gone(wy(cols(snowpack(150, 230))))
    by = {2015: r_ for r_ in [M.snow_gone(p)]}
    by.update({y: full for y in (2016, 2017)})
    nd = M.normal_dates(by, [2016, 2017])
    assert nd[0] == 230 - 91
    assert np.isnan(M.normal_dates(by, [2015, 2016])[0])
    summ = M.timing_summary("snow_gone", by, 2015, np.ones(1, bool), np.ones(1), [2016, 2017])
    assert summ["variants"]["default"]["shift"]["comparability"] == "restricted comparability"


def test_missing_records_give_records_missing():
    s = snowpack(150, 230)
    s[200] = np.nan
    t = 1.0 + 2.0 * np.exp(-((np.arange(365) - 180) / 40.0) ** 2)
    t[10] = np.nan
    p, q = wy(cols(s), cols(np.where(np.isnan(s), np.nan, 1.0))), yr(cols(t), cols(t))
    for f in (M.snow_peak, M.snow_gone, M.half_runoff):
        assert d(f(p), "reason")[0] == M.CODE["records_missing"], f.__name__
    for f in (M.leaf_green_up, M.photosynthesis_onset, M.growth_peak):
        r = f(q)
        for v in r["variants"].values():
            assert v["reason"][0] == M.CODE["records_missing"] and np.isnan(v["date"][0])


def test_alternatives_are_computed_beside_the_default():
    want = {"snow_peak": ["default"], "half_runoff": ["default"],
            "snow_gone": ["default", "threshold_mm=5", "days=7"],
            "leaf_green_up": ["default", "fraction=0.1", "fraction=0.5", "rise_mark=0.25",
                              "rise_mark=0.75"],
            "photosynthesis_onset": ["default", "fraction=0.1", "fraction=0.3"],
            "growth_peak": ["default", "window=15"]}
    days = np.arange(365)
    s = snowpack(150, 230)
    s[230:240] = 3.0                      # below 5 mm from 230, below 1 mm from 240
    s[250:258] = 2.0                      # a short return: 7 days below come first
    t = 0.2 + 3.0 / (1.0 + np.exp(-(days - 150) / 6.0))
    g = np.maximum(0.0, 10.0 * np.sin(np.pi * (days - 60) / 240.0))
    p, q = wy(cols(s)), yr(cols(t), cols(g))
    res = {m: M.METHODS[m].run(p if M.TIMING_KIND[m] == "wateryear" else q) for m in want}
    for m, labs in want.items():
        assert list(res[m]["variants"]) == labs, m
        for lab, v in res[m]["variants"].items():
            assert v["settings"] == dict(M.METHODS[m].variants())[lab]
    sg = res["snow_gone"]["variants"]
    assert sg["default"]["date"][0] == 258 - 91         # 14 days below 1 mm start after the return
    first5 = min(j for j in range(150, 352) if (s[j:j + 14] < 5.0).all())
    assert first5 < 230 and sg["threshold_mm=5"]["date"][0] == first5 - 91   # the return stays below 5 mm
    assert sg["days=7"]["date"][0] == 240 - 91          # 7 days fit before the return
    lg = res["leaf_green_up"]["variants"]
    assert lg["fraction=0.1"]["date"][0] < lg["default"]["date"][0] < lg["fraction=0.5"]["date"][0]
    po = res["photosynthesis_onset"]["variants"]
    assert po["fraction=0.1"]["date"][0] < po["default"]["date"][0] < po["fraction=0.3"]["date"][0]
    # a plan's own setting is a further variant beside the default, never in its place
    r = M.snow_gone(p, settings={"threshold_mm": 5.0})
    assert "default" in r["variants"] and r["variants"]["plan"]["settings"]["threshold_mm"] == 5.0


def test_settings_refused_unless_named():
    with pytest.raises(M.MethodError):
        M.settings_for("snow_gone", {"threshold_mm": 3.0})
    assert M.settings_for("snow_gone", {"threshold_mm": 3.0}, named={"threshold_mm": [3.0]}
                          )["threshold_mm"] == 3.0
    with pytest.raises(M.MethodError):
        M.settings_for("snow_gone", {"no_such": 1})


def test_leaf_green_up_rules():
    days = np.arange(365)
    t = np.full(365, 2.0)                 # high in January (last year's leaves)...
    t[40:100] = 0.4                       # ...a low before the peak...
    t[100:200] = np.linspace(0.4, 4.4, 100)
    t[200:] = 4.4                         # ...and the year's peak, reached on day 199
    r = M.leaf_green_up(yr(cols(t, np.zeros(365))))
    mark = 0.4 + 0.2 * 4.0
    first = int(np.argmax((t >= mark) & (days > 40)))
    assert d(r)[0] == first + 1           # dates are days of the year, 1 = 1 January
    assert d(r, "rise_share")[0] == pytest.approx(4.0 / 4.4)
    assert d(r, "reason")[1] == M.CODE["no_leaves"]


def test_cell_with_weak_leaf_seasonality():
    days = np.arange(365)
    t = 3.0 + 0.5 * np.sin(2 * np.pi * (days - 100) / 365.0)    # an evergreen canopy
    t2 = 1.0 + 1.0 * np.sin(2 * np.pi * (days - 100) / 365.0)   # rise 2.0 of a peak 2.0
    r = M.leaf_green_up(yr(cols(t, t2)))
    assert d(r, "reason").tolist() == [M.CODE["little_leaf_change"], 0]
    assert np.isnan(d(r)[0]) and d(r, "rise_share")[0] < 0.5 <= d(r, "rise_share")[1]


def test_photosynthesis_onset_rules():
    g = np.zeros(365)
    g[:20] = 3.0                          # a warm spell in January, before the low
    g[90:250] = np.linspace(0.0, 10.0, 160)
    g[250:] = 0.5
    r = M.photosynthesis_onset(yr(gpp=cols(g, np.zeros(365))))
    m7 = np.array([g[j:j + 7].mean() for j in range(359)])
    pk = int(np.argmax(m7))
    lowj = int(np.argmin(m7[:pk + 1]))
    j = lowj + 1 + int(np.argmax(m7[lowj + 1:] >= 0.2 * m7.max()))
    assert d(r)[0] == j + 3 + 1           # dated by the window's middle day
    assert d(r, "reason")[1] == M.CODE["no_photosynthesis"]


def test_cell_without_a_dormant_season():
    days = np.arange(365)
    g = 5.0 + 1.0 * np.sin(2 * np.pi * (days - 90) / 365.0)       # never below 20 % of its peak
    r = M.photosynthesis_onset(yr(gpp=cols(g)))
    assert d(r, "reason")[0] == M.CODE["no_dormant_season"] and np.isnan(d(r)[0])


def test_seasonal_flux_dates():
    days = np.arange(365)
    g = np.maximum(0.0, np.sin(np.pi * (days - 59) / 244.0)) * 8.0   # zero to day 59, peak at 181
    r = M.photosynthesis_onset(yr(gpp=cols(g)))
    m7 = np.array([g[j:j + 7].mean() for j in range(359)])
    first = int(np.argmax(m7 >= 0.2 * m7.max()))
    assert d(r)[0] == first + 3 + 1
    r = M.growth_peak(yr(gpp=cols(g)))
    assert d(r)[0] == int(np.argmax(m7)) + 3 + 1 == 182 and d(r, "flags")[0] == 0


def test_growth_peak_two_peaks_and_ties():
    days = np.arange(365)
    two = 6.0 * np.exp(-((days - 120) / 15.0) ** 2) + 5.8 * np.exp(-((days - 220) / 15.0) ** 2)
    shoulder = 6.0 * np.exp(-((days - 120) / 15.0) ** 2) + 2.0 * np.exp(-((days - 220) / 15.0) ** 2)
    tied = np.zeros(365)
    tied[100:107] = 4.0
    tied[200:207] = 4.0                   # two equal weeks: the earlier is the date
    r = M.growth_peak(yr(gpp=cols(two, shoulder, tied)))
    assert d(r, "flags").tolist() == [M.FLAG["two_peaks"], 0, M.FLAG["two_peaks"]]
    assert d(r)[2] == 100 + 3 + 1
    assert d(r)[0] == 120 + 1


def test_half_runoff_negative_days():
    q = np.ones(365)
    q2 = np.ones(365)
    q2[50:60] = -5.0                      # the curve falls: flagged, first crossing kept
    q3 = np.zeros(365)
    q3[10] = -1.0
    r = M.half_runoff(wy(qrunoff=cols(q, q2, q3)))
    c = np.cumsum(q2)
    assert d(r)[0] == int(np.argmax(np.cumsum(q) >= 182.5)) - 91
    assert d(r)[1] == int(np.argmax(c >= 0.5 * c[-1])) - 91
    assert d(r, "flags").tolist()[:2] == [0, M.FLAG["runoff_falls"]]
    assert d(r, "reason")[2] == M.CODE["no_runoff"]


class FakeStream:
    """A daily stream of two cells: H2OSNO = year * 1000 + day of year, GPP a
    per-second rate of 1 / 86,400."""
    name, cadence, calendar = "h1", "daily", "noleap"

    def __init__(self, years):
        self._years = list(years)
        self.grid = SimpleNamespace(shape=(1, 2))
        self.reads = []

    def years(self):
        return list(self._years)

    def read_block_pass(self, vars, year, reduce=None, require_complete=True, **kw):
        self.reads.append(year)
        keys = [(year, m, dd, 0, 0, 0) for m in range(1, 13)
                for dd in range(1, G.days_in_month(year, m, "noleap") + 1)]
        doy = np.arange(1, len(keys) + 1, dtype=float)
        base = (year * 1000.0 + doy)[:, None, None] * np.ones((1, 1, 2))
        data = {v: (base if v != "GPP" else np.full_like(base, 1.0 / 86400.0)) for v in vars}
        reduce(SimpleNamespace(keys=keys, data=data,
                               per_second={v: v in ("GPP", "QRUNOFF") for v in vars}))
        return {}


def test_daily_periods_read_each_year_once_and_assemble_water_years():
    s = FakeStream([2015, 2016])
    got = {(p.kind, p.year): p for p in M.daily_periods(s, [2015, 2016])}
    assert s.reads == [2015, 2016]
    p = got[("wateryear", 2016)]
    assert p.first_day == -91 and p.record_start == 0
    h = p.data["H2OSNO"][:, 0]
    assert h[0] == 2015 * 1000 + 274 and h[91] == 2015 * 1000 + 365 and h[92] == 2016001
    assert h[-1] == 2016 * 1000 + 273 and h.size == 365
    p = got[("wateryear", 2015)]
    assert p.record_start == 92 and np.isnan(p.data["H2OSNO"][:92]).all()
    assert p.data["H2OSNO"][92, 0] == 2015001
    y = got[("year", 2016)]
    assert y.first_day == 1 and y.data["TLAI"][0, 0] == 2016001
    assert np.allclose(y.data["GPP"], 1.0)                 # a rate becomes the day's total
    # a registry naming other months moves both starts
    s = FakeStream([2015, 2016])
    got = {(p.kind, p.year): p for p in
           M.daily_periods(s, [2016], {"water_year_start_month": 9, "year_start_month": 7})}
    p = got[("wateryear", 2016)]
    assert p.first_day == 1 - 122 and p.data["H2OSNO"][0, 0] == 2015 * 1000 + 244
    y = got[("year", 2016)]
    assert y.data["TLAI"][0, 0] == 2016 * 1000 + 182 and np.isnan(y.data["TLAI"][-1, 0])


def test_timing_summary_distribution_shift_and_sensitivity():
    wts = np.array([1.0, 1.0, 2.0, 4.0])
    mask = np.array([True, True, True, False])
    by = {}
    for y, melt in ((2016, 200), (2017, 210), (2018, 220), (2021, 190)):
        cells = [snowpack(150, melt), snowpack(150, melt + 10), np.full(365, 0.2),
                 snowpack(150, melt)]
        by[y] = M.snow_gone(wy(cols(*cells)))
    s = M.timing_summary("snow_gone", by, 2021, mask, wts, normal_years=[2016, 2017, 2018])
    dist = s["variants"]["default"]["distribution"]
    assert dist["cells"] == 3 and dist["dates"] == 2 and dist["reasons"] == {"no_snow": 1}
    assert dist["median"] == pytest.approx((190 - 91 + 200 - 91) / 2)   # equal areas: midpoint
    sh = s["variants"]["default"]["shift"]
    assert sh["median"] == pytest.approx(-20.0) and sh["cells"] == 2
    rec = {x["variant"]: x for x in s["sensitivity"]}
    assert set(rec) == {"threshold_mm=5", "days=7"}
    assert rec["days=7"]["default_median_date"] == dist["median"]
    assert "median_cell_change" in rec["days=7"]


def test_wquantile_equal_weights_is_the_median():
    x = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
    for q in (0.25, 0.5, 0.75):
        assert M.wquantile(x, np.ones(6), q) == np.quantile(x, q, method="averaged_inverted_cdf")
    assert M.wquantile(x, np.ones(6), 0.5) == pytest.approx(np.median(x))
    assert M.wquantile([1.0, 2.0, 3.0], [1.0, 1.0, 10.0], 0.5) == 3.0


# ── the screen ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def table(suite):
    return M.screen(suite, "all", 2021, NORMAL, notes=NOTES, fields=list(NOTES))


def rows_of(t, field):
    return {r["period"]: r for r in t["rows"] if r["field"] == field}


def test_screen_scores_by_type(suite, table):
    assert len(table["rows"]) == len(NOTES) * 18
    r = rows_of(table, "FLUX")["2021-07"]
    v, nv = by_hand(suite, "FLUX", "all", "2021-07")
    assert r["status"] == "scored" and r["n_normal"] == 5
    assert r["score"] == pytest.approx((v - nv.mean()) / nv.std(ddof=1), rel=1e-12)
    assert r["percent"] == pytest.approx(100 * (v - nv.mean()) / nv.mean(), rel=1e-12)
    assert r["type"] == "flux" and r["group"] == "carbon in and out"
    r = rows_of(table, "TK")["2021-08"]
    assert r["status"] == "scored" and r["percent"] is None
    assert rows_of(table, "NET")["2021-08"]["percent"] is None      # interval: no percent


def test_screen_insufficient_support(suite, tmp_path):
    t = M.screen(suite, "all", 2021, [2016], notes=NOTES, fields=["FLUX"])
    assert {r["status"] for r in t["rows"]} == {"insufficient support"}
    part = write_suite(tmp_path / "part", {"FLUX": seasonal},
                       covered=lambda f, lab, r: 0.9 if lab == "2018-07" else 1.0)
    t = M.screen(part, "all", 2021, NORMAL, notes=NOTES, fields=["FLUX"])
    r = rows_of(t, "FLUX")
    assert r["2021-07"]["status"] == "insufficient support" and r["2021-08"]["status"] == "scored"
    # a missing normal year lowers the count shown and still scores
    t = M.screen(suite, "all", 2021, [2014, 2016, 2017, 2018], notes=NOTES, fields=["FLUX"])
    r = rows_of(t, "FLUX")["2021-07"]
    assert r["n_normal"] == 3 and r["status"] == "scored" and "2014" in r["missing_time_records"]


def test_screen_rare_event_series(suite, table):
    fa = rows_of(table, "FAREA_BURNED")
    area = W.sum()
    r = fa["2021-08"]
    assert r["units"] == "km2" and r["value"] == pytest.approx(0.005 * area)
    assert r["mark"] == "above_normal_maximum" and r["normal_largest"] == pytest.approx(0.002 * area)
    assert r["normal_years_with_fire"] == 2 and r["reference"] == pytest.approx(0.0006 * area)
    assert r["score"] is None and r["percent"] is None
    r = fa["2021-09"]
    assert r["mark"] == "above_normal_maximum"            # no normal fire in September
    assert fa["2021-07"]["status"] == "no change"
    t = M.screen(suite, "all", 2019, NORMAL, notes=NOTES, fields=["FAREA_BURNED"])
    r = rows_of(t, "FAREA_BURNED")["2019-08"]
    assert r["status"] == "within the normal years' range" and r.get("mark") is None
    assert rows_of(table, "COL_FIRE_CLOSS")["2021-08"]["value"] == pytest.approx(0.5)


def test_screen_drifting_store_and_bounded_index_not_scored(table):
    for f in ("STORE", "BIDX"):
        for r in rows_of(table, f).values():
            assert r["status"] == "not scored yet" and r["score"] is None and r["change"] != 0, f
    r = rows_of(table, "STORE")["2021"]
    assert r["change"] == pytest.approx(1060.0 - 1030.0) and r["percent"] is not None
    assert rows_of(table, "BIDX")["2021"]["percent"] is None
    ns = {x["field"]: x for x in table["display"]["not_scored"]}
    assert ns["STORE"]["label"] == "not scored" and set(ns) == {"STORE", "BIDX"}


def test_screen_zero_change_is_no_change(suite, tmp_path):
    t = M.screen(suite, "all", 2021, NORMAL, control=suite, notes=NOTES, fields=list(NOTES))
    assert len(t["rows"]) == len(NOTES) * 18
    assert {r["status"] for r in t["rows"]} == {"no change"}
    assert all(r.get("score") is None for r in t["rows"])
    assert t["display"]["groups"] == {} and t["display"]["events"] == []
    # a control row whose value equals its normal exactly
    flat = write_suite(tmp_path / "flat", {"FLUX": lambda k, lab, r: 7.0 if lab.endswith("-05")
                                           else seasonal(k, lab, r)})
    r = rows_of(M.screen(flat, "all", 2021, NORMAL, notes=NOTES, fields=["FLUX"]), "FLUX")["2021-05"]
    assert r["status"] == "no change" and r["score"] is None and r["change"] == 0.0


def test_near_constant_normal_kelvin_and_celsius(table):
    k, c = rows_of(table, "TK")["2021-07"], rows_of(table, "TC")["2021-07"]
    for r in (k, c):
        assert r["normal_sd"] > 0 and r["near_constant"] is True
        assert r["status"] == "near-constant normal" and r["score"] is None
    assert k["reference_range"] == pytest.approx(c["reference_range"], rel=1e-9)
    assert rows_of(table, "TK")["2021-08"]["near_constant"] is False


def test_experiment_score_uses_the_controls_spread(suite, tmp_path):
    run = write_suite(tmp_path / "exp", {"FLUX": lambda k, lab, r: seasonal(k, lab, r) +
                                         (5.0 if lab == "2021-07" else 0.0)})
    t = M.screen(run, "all", 2021, NORMAL, control=suite, notes=NOTES, fields=["FLUX"])
    r = rows_of(t, "FLUX")["2021-07"]
    v, nv = by_hand(suite, "FLUX", "all", "2021-07")
    assert r["reference"] == v and r["change"] == pytest.approx(5.0, rel=1e-12)
    assert r["score"] == pytest.approx(5.0 / nv.std(ddof=1), rel=1e-12)
    assert r["percent"] == pytest.approx(500.0 / v, rel=1e-12)
    assert rows_of(t, "FLUX")["2021-08"]["status"] == "no change"


def test_display_rule_one_row_per_field(table):
    shown = table["display"]
    assert shown["rule"] == M.DISPLAY_RULE == table["display_rule"]
    rows = shown["groups"]["carbon in and out"]
    assert len(rows) == 3 and len({r["field"] for r in rows}) == 3
    scored = [r for r in table["rows"] if r["status"] == "scored" and r["group"] == "carbon in and out"]
    best = {}
    for r in scored:
        if r["field"] not in best or abs(r["score"]) > abs(best[r["field"]]["score"]):
            best[r["field"]] = r
    want = sorted(best.values(), key=lambda r: -abs(r["score"]))[:3]
    assert [(r["field"], r["period"]) for r in rows] == [(r["field"], r["period"]) for r in want]
    assert all(r["n_normal"] for r in rows)
    assert {(r["field"], r["period"]) for r in shown["events"]} == {
        (r["field"], r["period"]) for r in table["rows"] if r.get("mark")}


def test_screen_csv_holds_the_whole_table(table, tmp_path):
    import csv
    p = M.write_screen_csv(table, tmp_path / "screen.csv")
    with p.open() as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == len(table["rows"]) and list(got[0]) == list(M.SCREEN_COLUMNS)
    r = next(x for x in got if x["field"] == "FLUX" and x["period"] == "2021-07")
    assert float(r["score"]) == rows_of(table, "FLUX")["2021-07"]["score"]


# ── overlap and elevation split ──────────────────────────────────────────────

def test_overlap_quarters_and_sign_share():
    w = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    b = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    a = np.array([-2.0, -1.0, 1.0, 2.0, -3.0, 3.0])
    r = M.overlap(a, b, np.ones(6, bool), w)
    xm = (w * a).sum() / w.sum()
    ym = (w * b).sum() / w.sum()
    want = (w * (a - xm) * (b - ym)).sum() / np.sqrt((w * (a - xm) ** 2).sum() * (w * (b - ym) ** 2).sum())
    assert r["correlation"] == pytest.approx(want, rel=1e-12)
    q = r["quarters_by_second_map"]
    assert [x["area_km2"] for x in q] == [2.0, 2.0, 2.0, 2.0]       # equal areas: cell 1 split
    assert [x["first_map_mean"] for x in q] == pytest.approx([-1.5, 0.0, -0.5, 3.0])
    assert r["same_sign_share"] == pytest.approx(6.0 / 8.0)
    r = M.overlap(np.array([-1.0, -2.0, -3.0, -1.0]), np.array([1.0, -1.0, 2.0, 3.0]),
                  np.ones(4, bool), np.ones(4))
    assert r["same_sign_share"] is None and "more than 80" in r["same_sign_why_not"]
    r = M.overlap(a, b, np.ones(6, bool), w, bands={"low": np.array([1, 1, 1, 0, 0, 0], bool)})
    assert r["by_band"]["low"]["cells"] == 3
    assert M.overlap(a[:2], b[:2], np.ones(2, bool), w[:2])["correlation"] is None


def test_elevation_split_contributions_and_concentration():
    x = np.array([4.0, -1.0, 0.0, 1.0, np.nan])
    w = np.array([1.0, 1.0, 2.0, 4.0, 1.0])
    bands = {"low": np.array([1, 1, 0, 0, 1], bool), "high": np.array([0, 0, 1, 1, 0], bool)}
    r = M.elevation_split(x, np.ones(5, bool), w, bands,
                          others={"burned": np.array([1, 0, 0, 0, 0], bool),
                                  "not burned": np.array([0, 1, 1, 1, 1], bool)})
    assert r["area_km2"] == 8.0 and r["covered_share"] == pytest.approx(8.0 / 9.0)
    lo, hi = r["parts"]["low"], r["parts"]["high"]
    assert lo["positive_x_area"] == 4.0 and lo["negative_x_area"] == -1.0
    assert lo["abs_change_share"] == pytest.approx(5.0 / 9.0) and lo["area_share"] == 0.25
    assert hi["mean"] == pytest.approx(4.0 / 6.0)
    # 4.5 of the 9 needed: cell 0 (4, area 1), then half a unit of area at |x| = 1
    assert r["concentration"] == pytest.approx(1.5 / 8.0)
    assert r["other_regions"]["burned"]["cells"] == 1


def test_suite_bands_and_other_regions(suite):
    b = suite.band_masks("left")
    assert list(b) == ["below 1000 m", "1000 m and above"]
    assert (b["below 1000 m"] == (REGIONS["left"] & REGIONS["below 1000 m"])).all()


# ── review round 1 (2026-11-05): the behaviour changes ───────────────────────
# Each test below failed on the code before the change it covers and passes
# after it; the method's behaviour version was raised by one and recorded in
# its status (domain_methods.METHOD_STATUS).

def test_percent_follows_the_notes_policy_not_the_scale(suite):
    # a ratio-scale field whose policy forbids a percent gets none, with the
    # policy's reason; an interval-scale one whose policy allows it gets one
    pol = lambda ok, why: {"allowed": ok, "why": why}                  # noqa: E731
    notes = dict(NOTES,
                 FLUX={"group": "carbon in and out", "type": "flux", "scale": "ratio",
                       "percent": pol(False, "values can cross zero")},
                 NET={"group": "carbon in and out", "type": "flux", "scale": "interval",
                      "percent": pol(True, "a study asked for it")})
    r = M.anomaly(suite, "FLUX", "2021-07", NORMAL, region="all", notes=notes)
    assert r["percent"] is None and "values can cross zero" in r["percent_why_not"]
    r = M.anomaly(suite, "NET", "2021-07", NORMAL, region="all", notes=notes)
    assert r["percent"] == pytest.approx(100 * r["anomaly"] / r["normal"], rel=1e-12)
    r = M.difference(suite, suite, "FLUX", "2021-07", region="all", notes=notes)
    assert r["percent"] is None and "values can cross zero" in r["percent_why_not"]
    t = M.screen(suite, "all", 2021, NORMAL, notes=notes, fields=["FLUX", "NET"])
    assert rows_of(t, "FLUX")["2021-07"]["percent"] is None
    assert rows_of(t, "NET")["2021-07"]["percent"] is not None
    # a note without a policy gives no percent, whatever its scale
    bare = {"FLUX": {"group": "carbon in and out", "type": "flux", "scale": "ratio"}}
    r = M.anomaly(suite, "FLUX", "2021-07", NORMAL, region="all", notes=bare)
    assert r["percent"] is None and "no percent policy" in r["percent_why_not"]


def test_a_field_kept_in_percent_changes_in_percentage_points(tmp_path):
    # RH2M, with the field notes' own policy: its change is in percentage
    # points and no relative percent is made by default
    rh = write_suite(tmp_path / "rh", {"RH2M": lambda k, lab, r: 50.0 + seasonal(k, lab, r) / 10.0},
                     units={"RH2M": "%"})
    notes = {"RH2M": M.DC.load_notes()["RH2M"]}
    r = M.anomaly(rh, "RH2M", "2021-07", NORMAL, region="all", notes=notes)
    assert r["units"] == "%" and r["change_units"] == "percentage points"
    assert r["anomaly"] is not None and r["percent"] is None
    r = M.difference(rh, rh, "RH2M", "2021-07", region="all", notes=notes)
    assert r["change_units"] == "percentage points" and r["percent"] is None
    row = rows_of(M.screen(rh, "all", 2021, NORMAL, notes=notes, fields=["RH2M"]),
                  "RH2M")["2021-07"]
    assert row["change_units"] == "percentage points" and row["percent"] is None


def test_snow_gone_insufficient_follow_up_is_not_snow_persists():
    late = snowpack(150, 360)             # below 1 mm for the water year's last 5 days only
    never = np.full(365, 10.0)            # never below the threshold: snow persists
    brief = snowpack(150, 300)
    brief[310:] = 5.0                     # 10 days below, then above to the end
    r = M.snow_gone(wy(cols(late, never, brief)))
    assert d(r, "reason").tolist() == [M.CODE["insufficient_follow_up"], M.CODE["snow_persists"],
                                       M.CODE["snow_persists"]]
    assert np.isnan(d(r)).all()
    late9 = snowpack(150, 356)            # 9 days to the end: 7 fit, 14 do not
    r = M.snow_gone(wy(cols(late9)))
    assert d(r, "reason")[0] == M.CODE["insufficient_follow_up"]
    assert d(r, "date", "days=7")[0] == 356 - 91
    assert M.REASONS["insufficient_follow_up"][1] != M.REASONS["snow_persists"][1]


def test_timing_shift_reports_eligibility_change():
    wts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, True, True, True, False])

    def year(melts):                      # None: a cell without snow, so without a date
        return M.snow_gone(wy(cols(*[snowpack(150, x) if x else np.full(365, 0.2)
                                     for x in melts])))
    by = {2016: year([200, 210, None, None, 200]), 2017: year([200, 210, None, None, 200]),
          2021: year([190, None, 220, None, 190])}
    want = {"both": (1, 0.1), "only_this": (1, 0.3), "only_reference": (1, 0.2),
            "neither": (1, 0.4)}
    for kw in ({"normal_years": [2016, 2017]}, {"control": {2021: by[2016]}}):
        sh = M.timing_summary("snow_gone", by, 2021, mask, wts, **kw)["variants"]["default"]["shift"]
        assert sh["cells"] == 1 and sh["median"] == pytest.approx(-10.0)
        assert sh["area_share"] == pytest.approx(0.1) and sh["area_km2"] == 1.0
        el = sh["eligible"]
        assert {k: (el[k]["cells"], round(el[k]["area_share"], 12)) for k in want} == want, kw
        assert el["region_area_km2"] == 10.0
    # between settings: a 3 mm snowpack has a date at 1 mm and none at 5 mm
    small = np.zeros(365)
    small[20:100] = 3.0
    by2 = {2021: M.snow_gone(wy(cols(snowpack(150, 200), small)))}
    s = M.timing_summary("snow_gone", by2, 2021, np.ones(2, bool), np.array([1.0, 3.0]))
    rec = {x["variant"]: x for x in s["sensitivity"]}["threshold_mm=5"]
    el = rec["eligible"]
    assert (el["both"]["cells"], el["only_default"]["cells"], el["only_alternative"]["cells"],
            el["neither"]["cells"]) == (1, 1, 0, 0)
    assert el["only_default"]["area_share"] == pytest.approx(0.75)
    assert rec["cell_change_area_share"] == pytest.approx(0.25)


ONSET_SETTINGS = {"default": {"fraction": 0.2, "window": 7},
                  "fraction=0.1": {"fraction": 0.1, "window": 7},
                  "fraction=0.3": {"fraction": 0.3, "window": 7}}


def _onset_year(y, dates):
    """A stored photosynthesis_onset year, {setting: [each cell's date, None
    for no date]}, shaped as timing.nc gives it."""
    v = {}
    for lab, ds in dates.items():
        x = np.array([np.nan if t is None else float(t) for t in ds])
        v[lab] = {"settings": ONSET_SETTINGS[lab], "date": x,
                  "reason": np.where(np.isfinite(x), M.CODE["date"],
                                     M.CODE["no_dormant_season"]).astype(np.int64),
                  "flags": np.zeros(x.size, dtype=np.int64)}
    return {"method": "photosynthesis_onset",
            "version": M.METHODS["photosynthesis_onset"].version, "period": str(y),
            "kind": "year", "year": y, "variants": v}


def test_timing_range_across_settings_on_comparable_cells():
    """THE RANGE ACROSS SETTINGS (the user's decisions on G part 1,
    2026-11-08): each setting's median shift on the cells eligible under
    every setting, in the year and in its baseline; that set's cells and
    share beside each setting's own eligibility; the default, the smallest
    and the largest with their settings; whether their spread exceeds 7 days
    and whether the settings reverse the direction of the shift."""
    wts = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    mask = np.array([True, True, True, True, False])
    normal = {"default": [100] * 5, "fraction=0.1": [90] * 5, "fraction=0.3": [110] * 5}
    gap = dict(normal, **{"fraction=0.3": [110, 110, 110, None, 110]})   # cell 3: no normal
    by = {2016: _onset_year(2016, normal), 2017: _onset_year(2017, gap),
          # cell 2 has no date at 0.1 in the year
          2021: _onset_year(2021, {"default": [98, 96, 92, 80, 50],
                                   "fraction=0.1": [95, 93, None, 70, 50],
                                   "fraction=0.3": [120, 125, 140, 150, 50]})}
    s = M.timing_summary("photosynthesis_onset", by, 2021, mask, wts, normal_years=[2016, 2017])
    a = s["across_settings"]
    # the comparable cells: 0 and 1 (cell 2 lacks a date at 0.1, cell 3 a normal at 0.3)
    assert a["comparable"] == {"cells": 2, "area_km2": 3.0, "area_share": pytest.approx(0.3)}
    assert [(e["variant"], e["median_shift"]) for e in a["evaluated"]] == \
        [("default", -4.0), ("fraction=0.1", 3.0), ("fraction=0.3", 15.0)]
    assert a["evaluated"][2]["settings"] == ONSET_SETTINGS["fraction=0.3"]
    # each setting's own shift and eligibility stay as they were
    own = {lab: v["shift"] for lab, v in s["variants"].items()}
    assert (own["default"]["cells"], own["default"]["median"]) == (4, -8.0)
    assert (own["fraction=0.1"]["cells"], own["fraction=0.3"]["cells"]) == (3, 3)
    assert {x["variant"] for x in s["sensitivity"]} == {"fraction=0.1", "fraction=0.3"}
    assert a["default"] == {"variant": "default", "settings": ONSET_SETTINGS["default"],
                            "median_shift": -4.0}
    assert (a["minimum"]["variant"], a["minimum"]["median_shift"]) == ("default", -4.0)
    assert (a["maximum"]["variant"], a["maximum"]["median_shift"]) == ("fraction=0.3", 15.0)
    assert a["spread_days"] == 19.0 and a["limit_days"] == 7.0
    assert a["range_required"] is True and a["direction_reversal"] is True
    # a spread under 7 days, all earlier: no range required, no reversal; a
    # tie keeps the setting listed first; then a spread of exactly 7 days
    by[2021] = _onset_year(2021, {"default": [99, 99, 99, 99, 50],
                                  "fraction=0.1": [83, 83, 83, 83, 50],
                                  "fraction=0.3": [109, 109, 109, 109, 50]})
    a = M.timing_summary("photosynthesis_onset", by, 2021, mask, wts,
                         normal_years=[2016, 2017])["across_settings"]
    assert [e["median_shift"] for e in a["evaluated"]] == [-1.0, -7.0, -1.0]
    assert (a["minimum"]["variant"], a["maximum"]["variant"]) == ("fraction=0.1", "default")
    assert a["spread_days"] == 6.0 and a["range_required"] is False
    assert a["direction_reversal"] is False
    by[2021] = _onset_year(2021, {"default": [100, 100, 100, 100, 50],
                                  "fraction=0.1": [83, 83, 83, 83, 50],
                                  "fraction=0.3": [110, 110, 110, 110, 50]})
    a = M.timing_summary("photosynthesis_onset", by, 2021, mask, wts,
                         normal_years=[2016, 2017])["across_settings"]
    assert a["spread_days"] == 7.0 and a["range_required"] is False
    assert a["direction_reversal"] is False                  # a median of 0 is not later
    # against the control's same year, the same rules
    ctl = {2021: _onset_year(2021, {"default": [100, 100, 100, 100, 50],
                                    "fraction=0.1": [90, 90, 90, 90, 50],
                                    "fraction=0.3": [110, 110, None, 110, 50]})}
    run = {2021: _onset_year(2021, {"default": [95, 95, 95, 95, 50],
                                    "fraction=0.1": [99, 99, 99, 99, 50],
                                    "fraction=0.3": [106, 106, 106, 106, 50]})}
    a = M.timing_summary("photosynthesis_onset", run, 2021, mask, wts,
                         control=ctl)["across_settings"]
    assert a["comparable"]["cells"] == 3 and a["comparable"]["area_share"] == pytest.approx(0.7)
    assert [e["median_shift"] for e in a["evaluated"]] == [-5.0, 9.0, -4.0]
    assert a["spread_days"] == 14.0 and a["range_required"] and a["direction_reversal"]
    # no cell eligible under every setting: no range, and the reason
    run[2021]["variants"]["fraction=0.1"]["date"][:] = np.nan
    a = M.timing_summary("photosynthesis_onset", run, 2021, mask, wts,
                         control=ctl)["across_settings"]
    assert a["comparable"]["cells"] == 0 and a["minimum"] is None and a["spread_days"] is None
    assert a["range_required"] is False and "no cell is eligible" in a["why"]


def test_leaf_green_up_rise_mark_alternatives_and_eligibility_change():
    def canopy(low, peak):                # a low until day 60, then a rise to the peak on day 199
        t = np.full(365, low)
        t[60:200] = np.linspace(low, peak, 140)
        t[200:] = peak
        return t
    # rises of 0.95, 0.30, 0.55 and 0.10 of the peak
    r = M.leaf_green_up(yr(cols(canopy(0.1, 2.0), canopy(1.4, 2.0), canopy(0.9, 2.0),
                                canopy(1.8, 2.0))))
    v = r["variants"]
    assert list(v) == ["default", "fraction=0.1", "fraction=0.5", "rise_mark=0.25",
                       "rise_mark=0.75"]
    has = lambda lab: np.isfinite(v[lab]["date"]).tolist()               # noqa: E731
    assert has("default") == [True, False, True, False]
    assert has("rise_mark=0.25") == [True, True, True, False]
    assert has("rise_mark=0.75") == [True, False, False, False]
    s = M.timing_summary("leaf_green_up", {2016: r}, 2016, np.ones(4, bool), np.ones(4))
    rec = {x["variant"]: x for x in s["sensitivity"]}
    cnt = lambda e: (e["both"]["cells"], e["only_default"]["cells"],     # noqa: E731
                     e["only_alternative"]["cells"], e["neither"]["cells"])
    assert cnt(rec["rise_mark=0.25"]["eligible"]) == (2, 0, 1, 1)
    assert cnt(rec["rise_mark=0.75"]["eligible"]) == (1, 1, 0, 2)
    assert rec["rise_mark=0.75"]["eligible"]["only_default"]["area_share"] == 0.25


def test_elevation_split_flags_empty_and_small_bands_within_the_region():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    w = np.array([50.0, 49.5, 0.5, 10.0, 10.0, 10.0])
    region = np.array([1, 1, 1, 0, 0, 0], bool)
    bands = {"low": np.array([1, 1, 0, 1, 0, 0], bool),      # whole-domain masks, not cut
             "middle": np.array([0, 0, 1, 0, 1, 0], bool),
             "high": np.array([0, 0, 0, 0, 0, 1], bool)}
    r = M.elevation_split(x, region, w, bands, region="r")
    p = r["parts"]
    assert r["region"] == "r" and r["region_area_km2"] == 100.0
    assert p["low"]["region_cells"] == 2 and p["low"]["area_km2"] == 99.5   # cell 3 left out
    assert not p["low"]["empty"] and not p["low"]["small"]
    assert p["middle"]["region_cells"] == 1 and p["middle"]["small"] and not p["middle"]["empty"]
    assert p["middle"]["region_area_share"] == pytest.approx(0.005)
    assert p["high"]["region_cells"] == 0 and p["high"]["empty"] and not p["high"]["small"]
    assert r["flagged"] == {"middle": "small", "high": "empty"}


def test_missing_cell_values_apart_from_missing_records():
    s = snowpack(150, 230)
    gap = s.copy()
    gap[200] = np.nan                     # this cell lacks a value on a day the records hold
    q = np.ones(365)
    qgap = q.copy()
    qgap[200] = np.nan
    p = wy(cols(s, gap), cols(q, qgap))
    for f in (M.snow_peak, M.snow_gone, M.half_runoff):
        assert d(f(p), "reason").tolist() == [0, M.CODE["cell_values_missing"]], f.__name__
    both = np.stack([s, s], axis=1)
    both[200, :] = np.nan                 # a day no cell holds: a missing time record
    qq = np.ones((365, 2))
    qq[200, :] = np.nan
    for f in (M.snow_peak, M.snow_gone, M.half_runoff):
        assert d(f(wy(both, qq)), "reason").tolist() == [M.CODE["records_missing"]] * 2
    g = 2.0 * np.exp(-((np.arange(365) - 180) / 40.0) ** 2)
    g2 = g.copy()
    g2[10] = np.nan
    for f in (M.leaf_green_up, M.photosynthesis_onset, M.growth_peak):
        for v in f(yr(cols(g, g2), cols(g, g2)))["variants"].values():
            assert v["reason"][1] == M.CODE["cell_values_missing"], f.__name__
            assert v["reason"][0] not in (M.CODE["records_missing"],
                                          M.CODE["cell_values_missing"]), f.__name__
    dist = M.distribution(M.snow_gone(p)["variants"]["default"], np.ones(2, bool),
                          np.array([1.0, 3.0]))
    assert dist["inputs_area_share"] == pytest.approx(0.25)
    assert dist["missing_cell_values"] == {"cells": 1, "area_share": 0.75}
    assert dist["missing_time_records"] == {"cells": 0, "area_share": 0.0}


def test_every_method_reports_its_coverage(suite, tmp_path):
    part = write_suite(tmp_path / "part", {"FLUX": seasonal},
                       covered=lambda f, lab, r: 0.9 if lab in ("2021-07", "2018-07") else 1.0)
    r = M.anomaly(part, "FLUX", "2021-07", [2014, 2016, 2018], region="all", notes=NOTES)
    cov = r["coverage"]
    assert cov["area_share"]["2021-07"] == 0.9 and cov["least_area_share"] == 0.9
    assert cov["missing_cell_values"] == ["2021-07", "2018-07"]
    assert cov["missing_time_records"] == ["2014-07"]
    r = M.difference(part, suite, "FLUX", "2021-07", region="all", notes=NOTES)
    assert r["coverage"]["area_share"] == {"run 2021-07": 0.9, "control 2021-07": 1.0}
    assert r["coverage"]["missing_cell_values"] == ["run 2021-07"]
    r = M.anomaly(suite, "FLUX", "2021-JJA", NORMAL, notes=NOTES)     # the 2017 map lacks a cell
    cov = r["coverage"]
    assert cov["missing_cell_values"] == ["2017-JJA"] and cov["missing_time_records"] == []
    assert cov["result"]["missing_cells"] == 1
    assert cov["result"]["area_share"] == pytest.approx(1.0 - 2.0 / W.sum())
    t = M.screen(part, "all", 2021, [2014, 2016, 2018], notes=NOTES, fields=["FLUX"])
    row = rows_of(t, "FLUX")["2021-07"]
    assert "2018" in row["missing_cell_values"] and "2021-07" in row["missing_cell_values"]
    assert "2014" in row["missing_time_records"] and "2014" not in row["missing_cell_values"]
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 5.0])
    r = M.overlap(a, b, np.ones(4, bool), np.array([1.0, 1.0, 2.0, 4.0]))
    assert r["missing_cell_values"]["first"] == {"cells": 1, "area_share": 0.125}
    assert r["missing_cell_values"]["second"] == {"cells": 1, "area_share": 0.25}
    r = M.elevation_split(a, np.ones(4, bool), np.ones(4), {"all": np.ones(4, bool)})
    assert r["missing_cell_values"] == {"cells": 1, "area_share": 0.25}
    assert r["parts"]["all"]["missing_cell_values"] == {"cells": 1, "area_share": 0.25}


def test_growth_peak_flag_area_share_and_denominator():
    days = np.arange(365)
    two = 6.0 * np.exp(-((days - 120) / 15.0) ** 2) + 5.8 * np.exp(-((days - 220) / 15.0) ** 2)
    one = 6.0 * np.exp(-((days - 120) / 15.0) ** 2)
    r = M.growth_peak(yr(gpp=cols(two, one, np.zeros(365))))
    dist = M.distribution(r["variants"]["default"], np.ones(3, bool), np.array([1.0, 3.0, 4.0]))
    f = dist["flag_area"]["two_peaks"]
    assert f["cells"] == 1 and f["area_km2"] == 1.0
    assert f["dated_area_km2"] == 4.0 and f["share_of_dated_area"] == 0.25
    assert f["region_area_km2"] == 8.0 and f["share_of_region_area"] == 0.125
