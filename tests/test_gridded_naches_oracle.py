"""The gridded reader and regions against the finished Naches ELM-BGC run.

    pytest tests/test_gridded_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the run is on disk. Paths come from
tests/local_oracle_paths.json (not tracked; copy the .example beside it).

Every expected number is read from the hand analysis's own result files
(analysis/results/*.json, written 2026-10-14/15 and checked there by two
independent recomputations), so a pass means the framework code reproduces the
numbers the report was written from: the same cells, the same days, the same
totals. A handful of headline values (13,394 vegetated cells, 4,143 and 4,127
watershed cells, 2,930 compared cells) are also written out here, so a changed
result file cannot quietly move the target.

One reduction over 2015-2021 feeds most tests; cold, it reads about 9 GB of
daily history and takes several minutes.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pytest

import gridded as G
import regions as R

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
YEARS = list(range(2015, 2022))
NORMAL = [2016, 2017, 2018, 2019, 2020]
WS = "Naches watershed (USGS 12494000)"
BAND_NAMES = ["below 1000 m", "1000-1500 m", "1500-2000 m", "2000 m and above"]
ALL = "all vegetated cells"
RIVER = "river mask (simplified polygon, all cells)"


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}; copy {CONFIG.name}.example and fill it in")
    c = json.loads(CONFIG.read_text()).get("naches_bgc") or {}
    missing = [k for k, v in c.items() if not Path(v).exists()]
    if missing or not c:
        pytest.skip(f"local paths missing: {missing or 'naches_bgc block'}")
    return {k: Path(v) for k, v in c.items()}


def _json(cfg, name):
    return json.loads((cfg["results_dir"] / name).read_text())


@pytest.fixture(scope="module")
def h1(cfg):
    return G.open_stream(cfg["run_dir"], "h1")


@pytest.fixture(scope="module")
def grid(h1):
    return h1.grid


@pytest.fixture(scope="module")
def surf(cfg, grid):
    return R.read_surface(cfg["surface"], grid, ("TOPO", "PCT_NATVEG"))


@pytest.fixture(scope="module")
def static(cfg, grid, surf):
    usgs = cfg["obs_dir"] / "usgs"
    veg = R.vegetated(surf["PCT_NATVEG"])
    bands = R.elevation_bands(surf["TOPO"])
    simp, simp_audit = R.polygon_mask(grid, usgs / "nldi_basin_USGS-12494000.geojson")
    uns, uns_audit = R.polygon_mask(
        grid, usgs / "nldi_basin_USGS-12494000_unsimplified.geojson")
    modis = cfg["obs_dir"] / "modis"
    burned = R.mask_from_field(modis / "MCD64A1_naches_elmgrid.nc",
                               "burned_fraction_year", grid)
    footprint = R.mask_from_field(cfg["fire_footprint"], "BURNED_AREA", grid)
    gpp_ok = R.mask_from_field(modis / "MOD17A2HGF_naches_elmgrid.nc", "gpp_n", grid,
                               over="all")
    lai_ok = R.mask_from_field(modis / "MCD15A3H_naches_elmgrid.nc", "lai_n", grid,
                               over="all")
    return dict(veg=veg, bands=bands, simp=simp, simp_audit=simp_audit, uns=uns,
                uns_audit=uns_audit, burned=burned, footprint=footprint,
                gpp_ok=gpp_ok, lai_ok=lai_ok)


@pytest.fixture(scope="module")
def stations(cfg, grid, surf):
    import netCDF4
    with netCDF4.Dataset(cfg["domain"]) as d:
        d.set_auto_mask(False)
        corners = (np.asarray(d["xv"][:], float), np.asarray(d["yv"][:], float))
    rows = list(csv.DictReader(open(cfg["obs_dir"] / "snotel" / "snotel_stations.csv")))
    sts = [{"id": r["triplet"], "name": r["name"], "lat": float(r["lat"]),
            "lon": float(r["lon"]), "elevation_m": float(r["elevation_m"])} for r in rows]
    return R.station_cells(grid, sts, topo=surf["TOPO"], radius_km=3.0, corners=corners)


@pytest.fixture(scope="module")
def masks(grid, static, stations):
    s = static
    ms = R.MaskSet(grid)
    ms.add(ALL, s["veg"], "PCT_NATVEG >= 100", ["surface"])
    for b in BAND_NAMES:
        ms.add(b, s["veg"] & s["bands"][b], f"vegetated, TOPO {b}", ["surface"])
    ws = s["veg"] & s["uns"]
    ms.add(WS, ws, "vegetated, centre inside the unsimplified NLDI basin", ["nldi"])
    for b in BAND_NAMES:
        ms.add(f"{WS}, {b}", ws & s["bands"][b], f"watershed, {b}", ["nldi"])
    ms.add(RIVER, s["simp"], "every cell, centre inside the simplified basin", ["nldi"])
    comp = ws & ~s["burned"] & ~s["footprint"] & s["gpp_ok"] & s["lai_ok"]
    ms.add("comp all", comp, "watershed, unburned, full MODIS record", ["modis"])
    for b in BAND_NAMES:
        ms.add(f"comp {b}", comp & s["bands"][b], f"compared, {b}", ["modis"])
    for st in stations:
        if st["containing"] is not None:
            ms.add(f"station {st['id']}", R.cell_mask(grid, st["containing"]["flat"]),
                   "the cell containing the pillow", ["snotel"])
    return ms


@pytest.fixture(scope="module")
def reduced(h1, masks):
    fields = [R.Field("GPP"), R.Field("NPP"), R.Field("ET", ("QVEGT", "QVEGE", "QSOIL")),
              R.Field("QRUNOFF"), R.Field("PRECIP", ("RAIN", "SNOW")), R.Field("TLAI"),
              R.Field("H2OSNO"), R.Field("TSA"), R.Field("BTRAN"),
              R.Field("SOILWATER_10CM")]
    return R.reduce_regions(h1, fields, masks, YEARS)


# ── streams ──────────────────────────────────────────────────────────────────
class TestStreams:

    def test_daily_stream(self, h1):
        assert (h1.cadence, h1.calendar, len(h1.records)) == ("daily", "noleap", 2555)
        assert h1.labels()[0] == "2015-01-01" and h1.labels()[-1] == "2021-12-31"
        assert h1.complete_years() == YEARS
        assert all(len(h1.labels(y)) == 365 for y in YEARS)
        assert not any(lab.endswith("-02-29") for lab in h1.labels())
        assert len(h1.dropped) == 1 and h1.dropped[0]["record"] == 0
        assert abs(h1.dropped[0]["length_days"] - 1 / 48) < 1e-6

    def test_monthly_stream_keeps_january_2015(self, cfg):
        h0 = G.open_stream(cfg["run_dir"], "h0")
        assert (h0.cadence, len(h0.records)) == ("monthly", 84)
        assert h0.labels(2015) == [f"2015-{m:02d}" for m in range(1, 13)]
        assert h0.dropped == [] and [e["month"] for e in h0.extended] == ["2015-01"]

    def test_three_hourly_stream(self, cfg):
        h2 = G.open_stream(cfg["run_dir"], "h2")
        assert (h2.cadence, h2.steps_per_day, len(h2.records)) == ("subdaily", 8, 20440)
        assert h2.complete_years() == YEARS
        assert h2.labels(2021)[-1] == "2021-12-31 21:00"


# ── static masks ─────────────────────────────────────────────────────────────
class TestMasks:

    def test_the_three_files_describe_the_same_cells(self, cfg, grid):
        import netCDF4
        with netCDF4.Dataset(cfg["domain"]) as d:
            got = grid.check_coordinates(np.asarray(d["yc"][:]), np.asarray(d["xc"][:]),
                                         "domain")
        assert got["max_abs_lat_diff_deg"] < 1e-6 and got["max_abs_lon_diff_deg"] < 1e-6

    def test_vegetated_cells_and_bands(self, cfg, grid, static):
        meta = _json(cfg, "model_anomalies.json")["meta"]["regions"]
        assert int(static["veg"].sum()) == 13394 == meta[ALL]["cells"]
        assert grid.weight_km2[static["veg"]].sum() == pytest.approx(
            meta[ALL]["area_km2"], rel=1e-5)
        for b in BAND_NAMES:
            m = static["veg"] & static["bands"][b]
            assert int(m.sum()) == meta[b]["cells"], b
            assert grid.weight_km2[m].sum() == pytest.approx(meta[b]["area_km2"],
                                                             rel=1e-5), b

    def test_watershed_masks(self, cfg, grid, static):
        import netCDF4
        usgs = _json(cfg, "usgs_comparison.json")["basin_mask_naches"]
        assert static["simp_audit"]["n_cells"] == 4143 == usgs["n_cells"]
        assert static["simp_audit"]["mask_area_km2"] == pytest.approx(
            usgs["mask_area_km2"], abs=0.005)
        with netCDF4.Dataset(cfg["results_dir"] / "usgs_comparison_basin_masks.nc") as d:
            saved = np.asarray(d["mask_naches_usgs_12494000"][:]) > 0
        assert np.array_equal(static["simp"], saved), "differs from the shapely mask"
        v = _json(cfg, "verify_model_modis.json")["watershed"]
        uns = static["uns"]
        assert int(uns.sum()) == v["cells_all"] == 4142
        assert grid.weight_km2[uns].sum() == pytest.approx(v["area_all_km2"], abs=0.005)
        veg_ws = uns & static["veg"]
        assert int(veg_ws.sum()) == v["cells_vegetated"] == 4127
        assert grid.weight_km2[veg_ws].sum() == pytest.approx(v["area_vegetated_km2"],
                                                              abs=0.005)
        assert int((uns ^ static["simp"]).sum()) == \
            v["cells_that_differ_from_usgs_comparison_mask"] == 13
        # The spherical polygon area is an audit within 0.3 % of the geodesic one.
        geo = usgs["polygon_area_km2_wgs84_geodesic"]
        assert abs(static["simp_audit"]["polygon_area_km2_spherical"] / geo - 1) < 0.003
        assert static["simp_audit"]["polygon_inside_grid"]

    def test_compared_cells(self, cfg, masks):
        counts = json.loads(cfg["two_run_comparison"].read_text())["counts"]
        assert masks.mask("comp all").n_cells == 2930 == counts["comp all"]
        for b in BAND_NAMES:
            assert masks.mask(f"comp {b}").n_cells == counts[f"comp {b}"], b

    def test_station_cells(self, cfg, stations):
        ref = {s["triplet"]: s["pairing"]
               for s in _json(cfg, "snotel_comparison.json")["stations"]}
        assert len(stations) == len(ref) == 16
        assert sum(s["inside_grid"] for s in stations) == 9
        for s in stations:
            p = ref[s["id"]]
            assert s["inside_grid"] == p["inside_model_grid"], s["id"]
            if p["containing_cell"]:
                c = p["containing_cell"]
                assert (s["containing"]["row"], s["containing"]["col"]) == \
                    (c["row_j_south_to_north"], c["col_i_west_to_east"]), s["id"]
                assert s["containing"]["distance_km"] == pytest.approx(
                    c["distance_km"], abs=0.006)
            assert round(s["nearest_centre_km"], 2) == p["nearest_cell_centre_km"]
            assert s["n_within_radius"] == p["n_cells_within_3km"], s["id"]
            e = p["elevation_matched_cell"]
            if e is None:
                assert s["elevation_matched"] is None
            else:
                assert (s["elevation_matched"]["row"], s["elevation_matched"]["col"]) == \
                    (e["row_j_south_to_north"], e["col_i_west_to_east"]), s["id"]

    def test_the_mask_record_round_trips_on_the_real_grid(self, masks, grid, tmp_path):
        back = R.MaskSet.from_netcdf(masks.to_netcdf(tmp_path / "masks.nc"), grid)
        assert back.names() == masks.names()
        for n in masks.names():
            assert np.array_equal(back[n], masks[n]), n


# ── regional numbers ─────────────────────────────────────────────────────────
# metric in model_anomalies.json -> (field, how, months)
METRICS = {
    "gpp": ("GPP", "sum", None), "npp": ("NPP", "sum", None),
    "et": ("ET", "sum", None), "runoff_qrunoff": ("QRUNOFF", "sum", None),
    "precip_rain_plus_snow": ("PRECIP", "sum", None),
    "tlai_annual_mean": ("TLAI", "mean", None),
    "tlai_may_sep_mean": ("TLAI", "mean", [5, 6, 7, 8, 9]),
    "swe_mar_apr_mean": ("H2OSNO", "mean", [3, 4]),
    "swe_peak_jan_jun": ("H2OSNO", "max", [1, 2, 3, 4, 5, 6]),
    "btran_may_sep_mean": ("BTRAN", "mean", [5, 6, 7, 8, 9]),
    "soilwater_10cm_jun_sep_mean": ("SOILWATER_10CM", "mean", [6, 7, 8, 9]),
    "tsa_annual_mean": ("TSA", "mean", None),
    "tsa_jun_aug_mean": ("TSA", "mean", [6, 7, 8]),
}


def _sf(x, n):
    """Half a unit in the n-th significant figure, the rounding of a recorded
    value (4 s.f.: 5.861 -> 0.0005, -18.61 -> 0.005; 6 s.f.: 14.1553 ->
    0.00005), with 2 % to spare."""
    import math
    e = math.floor(math.log10(abs(x))) if x else 0
    return 0.5 * 10.0 ** (e - n + 1) * 1.02 + 1e-12


class TestRegionalNumbers:

    @pytest.mark.parametrize("region", BAND_NAMES + [ALL])
    @pytest.mark.parametrize("metric", list(METRICS))
    def test_model_anomalies_by_year(self, cfg, reduced, region, metric):
        ref = _json(cfg, "model_anomalies.json")["annual"][region][metric]
        fld, how, months = METRICS[metric]
        got, why = reduced.by_year(region, fld, YEARS, how=how, months=months)
        assert why == {}
        for y in YEARS:
            assert got[y] == pytest.approx(ref["by_year"][str(y)], rel=1e-5), y
        a = R.anomalies(got, NORMAL, [2015, 2021], percent=not metric.startswith("tsa"))
        assert a["normal"] == pytest.approx(ref["normal_2016_2020"], rel=1e-5)
        for y in (2015, 2021):
            assert a["years"][y]["rank_lowest_first"] == \
                ref[str(y)]["rank_of_7_lowest_first"]
            want = ref[str(y)]["anomaly_pct"]
            if want is not None:
                assert a["years"][y]["anomaly_pct"] == pytest.approx(
                    want, abs=_sf(want, 4)), y

    @pytest.mark.parametrize("band", [""] + [f", {b}" for b in BAND_NAMES])
    def test_watershed_numbers(self, cfg, reduced, masks, band):
        region = WS + band
        ref = _json(cfg, "verify_model_modis.json")["method_sensitivity"][
            "watershed_only"]["model_anomalies_regions"][region]
        assert masks.mask(region).n_cells == ref["cells"]
        assert masks.mask(region).area_km2 == pytest.approx(ref["area_km2"], rel=1e-5)
        items = {"GPP": ("GPP", "sum", None), "NPP": ("NPP", "sum", None),
                 "ET": ("ET", "sum", None), "QRUNOFF": ("QRUNOFF", "sum", None),
                 "PRECIP": ("PRECIP", "sum", None),
                 "H2OSNO_mar_apr": ("H2OSNO", "mean", [3, 4]),
                 "TSA_jja": ("TSA", "mean", [6, 7, 8]),
                 "BTRAN_may_sep": ("BTRAN", "mean", [5, 6, 7, 8, 9]),
                 "SOILWATER_10CM_jun_sep": ("SOILWATER_10CM", "mean", [6, 7, 8, 9])}
        for key, (fld, how, months) in items.items():
            got, _ = reduced.by_year(region, fld, YEARS, how=how, months=months)
            a = R.anomalies(got, NORMAL, [2015, 2021], percent=(key != "TSA_jja"))
            for y in (2015, 2021):
                r = ref[f"{key} {y}"]
                assert got[y] == pytest.approx(r["value"], rel=1e-5), (key, y)
                assert a["normal"] == pytest.approx(r["normal"], rel=1e-5), key
                assert a["years"][y]["anomaly"] == pytest.approx(
                    r["anomaly"], rel=1e-4, abs=1e-6), (key, y)
                if "anomaly_pct" in r:
                    assert a["years"][y]["anomaly_pct"] == pytest.approx(
                        r["anomaly_pct"], abs=_sf(r["anomaly_pct"], 6)), (key, y)
        gpp, _ = reduced.by_year(region, "GPP", YEARS)
        rank = R.anomalies(gpp, NORMAL, [2021])["years"][2021]["rank_lowest_first"]
        assert rank == ref["GPP rank of 2021 among 2015-2021 (1 = lowest)"]

    def test_river_flow_matches_the_gauge_comparison(self, cfg, reduced):
        u = _json(cfg, "usgs_comparison.json")["naches"]
        cal, _ = reduced.by_year(RIVER, "QRUNOFF", YEARS)
        aj, _ = reduced.by_year(RIVER, "QRUNOFF", YEARS, months=[4, 5, 6, 7])
        js, _ = reduced.by_year(RIVER, "QRUNOFF", YEARS, months=range(1, 10))
        assert R.anomalies(cal, NORMAL, [])["normal"] == pytest.approx(
            u["calendar_year"]["normal_2016_2020"]["model_q"]["depth_mm"], rel=1e-5)
        for blk, by, y in ((u["apr_jul"], aj, 2015), (u["calendar_year"], cal, 2021),
                           (u["calendar_year"], cal, 2015), (u["jan_sep"], js, 2015)):
            pct = R.anomalies(by, NORMAL, [y])["years"][y]["anomaly_pct"]
            assert pct == pytest.approx(blk["anomaly"][str(y)]["model_q"]["depth_pct"],
                                        abs=1e-4), y
        p, _ = reduced.by_year(RIVER, "PRECIP", YEARS)
        assert R.anomalies(p, NORMAL, [])["normal"] == pytest.approx(
            u["calendar_year"]["normal_2016_2020"]["model_p"]["depth_mm"], rel=1e-5)
        cot = u["centre_of_timing"]["jan_sep"]["model_q"]
        doys = {}
        for y in YEARS:
            c = R.centre_of_timing(reduced, RIVER, "QRUNOFF", y, range(1, 10))
            doys[y] = c["day_of_year"]
            assert c["label"][5:] == cot[str(y)]["date"], y
        assert np.mean([doys[y] for y in NORMAL]) == pytest.approx(cot["normal_2016_2020"])
        assert doys[2015] - np.mean([doys[y] for y in NORMAL]) == pytest.approx(
            cot["2015"]["anomaly_days"])

    def test_compared_cells_match_the_two_run_comparison(self, cfg, reduced):
        ours = json.loads(cfg["two_run_comparison"].read_text())["runs"]["ours"]
        for region, ref in ours["compared_gpp"].items():
            got, _ = reduced.by_year(region, "GPP", YEARS)
            a = R.anomalies(got, NORMAL, [2015, 2021])
            assert a["normal"] == pytest.approx(ref["gpp_normal"], abs=0.051), region
            assert a["years"][2015]["anomaly_pct"] == pytest.approx(
                ref["gpp_2015_pct"], abs=0.0051), region
            assert a["years"][2021]["anomaly_pct"] == pytest.approx(
                ref["gpp_2021_pct"], abs=0.0051), region
        riv = ours["river"]
        cal, _ = reduced.by_year(RIVER, "QRUNOFF", YEARS)
        assert R.anomalies(cal, NORMAL, [])["normal"] == pytest.approx(
            riv["normal_mm"], abs=0.051)

    def test_station_cell_snow_peaks(self, cfg, reduced, stations):
        rows = [r for r in csv.DictReader(
            open(cfg["results_dir"] / "snotel_comparison_station_years.csv"))
            if r["source"] == "model_containing" and int(r["water_year"]) >= 2016]
        inside = {s["id"] for s in stations if s["containing"] is not None}
        assert {r["triplet"] for r in rows} == inside and len(rows) == 54
        labels = reduced.labels
        for r in rows:
            region, wy = f"station {r['triplet']}", int(r["water_year"])
            got, why = reduced.by_year(region, "H2OSNO", [wy], how="max", water_year=True)
            assert got[wy] == pytest.approx(float(r["peak_swe_mm"]), abs=0.051), \
                (r["triplet"], wy)
            v = reduced.series(region, "H2OSNO")
            idx = [k for k, lab in enumerate(labels)
                   if lab >= f"{wy - 1}-10-01" and lab <= f"{wy}-09-30"]
            assert labels[idx[int(np.argmax(v[idx]))]] == r["peak_date"], \
                (r["triplet"], wy)

    def test_the_2021_heat_wave_peak(self, cfg, reduced):
        ref = _json(cfg, "model_anomalies.json")["late_june_2021"][ALL]["summary"]
        labels = reduced.labels
        tsa, gpp = reduced.series(ALL, "TSA"), reduced.series(ALL, "GPP")
        days = [f"{m:02d}-{d:02d}" for m, n in ((6, 30), (7, 31)) for d in range(1, n + 1)
                if (m, d) >= (6, 15)]
        pos = {lab: k for k, lab in enumerate(labels)}
        an_t, an_g = [], []
        for md in days:
            norm_t = np.mean([tsa[pos[f"{y}-{md}"]] for y in NORMAL])
            norm_g = np.mean([gpp[pos[f"{y}-{md}"]] for y in NORMAL])
            an_t.append(tsa[pos[f"2021-{md}"]] - norm_t)
            an_g.append(100 * (gpp[pos[f"2021-{md}"]] - norm_g) / norm_g)
        k = int(np.argmax(an_t))
        assert f"2021-{days[k]}" == ref["tsa_anomaly_max_date"] == "2021-06-29"
        assert an_t[k] == pytest.approx(ref["tsa_anomaly_max_K"],
                                        abs=_sf(ref["tsa_anomaly_max_K"], 6))
        assert an_g[k] == pytest.approx(ref["gpp_anomaly_min_pct"],
                                        abs=_sf(ref["gpp_anomaly_min_pct"], 4))

    def test_the_runs_own_replication_record(self, cfg, reduced):
        rec = sorted(glob.glob(str(cfg["checks_dir"] /
                                   "replication_naches_bgc_fix_2015_*.json")))[-1]
        want = json.loads(Path(rec).read_text())["annual_gpp_gC_m2"]["ours"]
        got, _ = reduced.by_year(ALL, "GPP", [2015])
        assert got[2015] == pytest.approx(want, abs=0.051)
