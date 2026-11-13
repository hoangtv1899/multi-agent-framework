"""The places a study's numbers are reported for (domain_places), on made-up stores.

    pytest tests/test_domain_places.py -v

Each store here is a masks.nc written by the store's own writer
(regions.MaskSet) on a small made-up grid of 3 rows and 4 columns, so that a
map laid out column by row would show. Every region, cell and area is made
up. One store is built whole by domain_suite on test_domain_suite's fake run,
to hold the areas to the ones the catalog gives.
"""
import json

import netCDF4
import numpy as np
import pytest

import domain_catalog as DCAT
import domain_places as DP
import domain_suite as S
import gridded as G
import regions as R
from test_domain_suite import FIELDS, make_control

LAT = [40.1, 40.2, 40.3]
LON = [250.5, 250.6, 250.7, 250.8]          # east of Greenwich, as a history writes it
WEST = [-109.5, -109.4, -109.3, -109.2]
# every cell its own area, so that a region's area tells which cells it has
AREA = np.arange(1.0, 13.0).reshape(3, 4)


def _grid():
    lat_axis = np.asarray(LAT, float)
    lon_axis = G.to_pm180(np.asarray(LON, float))
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    return G.Grid(lat=lat, lon=lon, area_km2=AREA.copy(), landfrac=np.ones(lat.shape),
                  dims=("lat", "lon"), lat_axis=lat_axis, lon_axis=lon_axis)


def _cells(*where):
    m = np.zeros((3, 4), bool)
    for j, i in where:
        m[j, i] = True
    return m


# made-up regions: a valley of eight cells, two parts of it, a ridge that
# crosses the valley's edge, and a patch of no cell
VALLEY = _cells((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1))
LOW = _cells((0, 0), (0, 1), (1, 0))
HIGH = _cells((1, 2), (2, 0), (2, 1))
RIDGE = _cells((0, 2), (0, 3), (1, 3))
TWIN = LOW.copy()
NONE = _cells()


def _area(m):
    return float(AREA[m].sum())


def _store(tmp_path, regions):
    """A results store holding only masks.nc, written by the store's writer."""
    ms = R.MaskSet(_grid())
    for name, m in regions.items():
        ms.add(name, m, rule=f"made up: {name}")
    sd = tmp_path / "store"
    ms.to_netcdf(sd / S.MASKS_FILE)
    return {"name": "made_up_valley", "suite_dir": str(sd)}, ms


ALL = {"valley": VALLEY, "valley low": LOW, "valley high": HIGH, "ridge": RIDGE,
       "low twin": TWIN, "nowhere": NONE}


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path, ALL)


def _classes(rec):
    return np.array(rec["classes"])


# ── a region inside another ──────────────────────────────────────────────────
class TestARegionInsideAnother:
    def test_the_outer_is_an_outline_and_the_inner_ones_are_filled(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley", "valley low", "valley high"])
        assert rec["problems"] == []
        assert [o["label"] for o in rec["outlines"]] == ["valley"]
        assert np.array_equal(np.array(rec["outlines"][0]["mask"]), VALLEY.astype(int))
        assert [f["label"] for f in rec["filled"]] == ["valley low", "valley high"]
        want = np.zeros((3, 4), int)
        want[LOW], want[HIGH] = 1, 2
        assert np.array_equal(_classes(rec), want)

    def test_cells_and_areas_are_the_stores_own(self, store):
        entry, ms = store
        rec = DP.page_record(entry, ["valley", "valley low", "valley high"])
        got = {r["label"]: (r["cells"], r["area_km2"]) for r in rec["filled"] + rec["outlines"]}
        assert got == {n: (ms.summary()[n]["n_cells"], ms.summary()[n]["area_km2"])
                       for n in ("valley", "valley low", "valley high")}
        assert got["valley"] == (8, _area(VALLEY)) and got["valley low"] == (3, _area(LOW))

    def test_the_order_asked_is_kept(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley high", "valley", "valley low"])
        assert rec["regions"] == ["valley high", "valley", "valley low"]
        assert [f["label"] for f in rec["filled"]] == ["valley high", "valley low"]
        assert _classes(rec)[2, 0] == 1 and _classes(rec)[0, 0] == 2

    def test_a_region_named_alone_is_filled(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley"])
        assert rec["outlines"] == [] and [f["label"] for f in rec["filled"]] == ["valley"]
        assert np.array_equal(_classes(rec), VALLEY.astype(int))

    def test_a_region_inside_one_inside_another(self, tmp_path):
        """Each region that holds another is an outline, however deep."""
        entry, _ = _store(tmp_path, {"valley": VALLEY, "valley low": LOW,
                                     "one cell": _cells((0, 0))})
        rec = DP.page_record(entry, ["valley", "valley low", "one cell"])
        assert [o["label"] for o in rec["outlines"]] == ["valley", "valley low"]
        assert [f["label"] for f in rec["filled"]] == ["one cell"]
        assert int(_classes(rec).sum()) == 1 and _classes(rec)[0, 0] == 1

    def test_a_region_crossing_the_outline_is_filled_and_does_not_make_it_one(self, store):
        """The ridge holds no other region and is not held by the valley."""
        entry, _ = store
        rec = DP.page_record(entry, ["ridge", "valley"])
        assert rec["outlines"] == [] and {f["label"] for f in rec["filled"]} == {"ridge",
                                                                                   "valley"}


# ── two regions that overlap ─────────────────────────────────────────────────
class TestTwoRegionsThatOverlap:
    def test_a_shared_cell_goes_to_the_smaller_and_the_overlap_is_named(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley", "ridge"])
        assert _area(RIDGE) < _area(VALLEY)
        c = _classes(rec)
        assert c[0, 2] == 2                       # the shared cell: the ridge's
        assert c[0, 0] == 1 and c[1, 3] == 2 and c[2, 3] == 0
        assert rec["problems"] == ["the regions 'valley' and 'ridge' overlap in 1 cell, "
                                   "which are drawn as 'ridge', the smaller"]

    def test_the_smaller_wins_whichever_is_asked_first(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["ridge", "valley"])
        assert _classes(rec)[0, 2] == 1
        assert "drawn as 'ridge', the smaller" in rec["problems"][0]

    def test_two_regions_of_the_same_cells_are_both_filled_and_named(self, store):
        """Neither holds the other, so neither is an outline; of two of one
        area the one asked for first is drawn."""
        entry, _ = store
        rec = DP.page_record(entry, ["low twin", "valley low"])
        assert rec["outlines"] == [] and len(rec["filled"]) == 2
        assert np.array_equal(_classes(rec), LOW.astype(int))
        assert rec["problems"] == ["the regions 'low twin' and 'valley low' overlap in 3 "
                                   "cells, which are drawn as 'low twin', the smaller"]


# ── what is left out ─────────────────────────────────────────────────────────
class TestWhatIsLeftOut:
    def test_a_label_the_store_does_not_know_is_named_and_left_out(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley low", "the moon"])
        assert rec["regions"] == ["valley low", "the moon"]
        assert [f["label"] for f in rec["filled"]] == ["valley low"]
        assert len(rec["problems"]) == 1
        assert rec["problems"][0].startswith("the results store has no region 'the moon', "
                                             "so it is left out (it has ")
        assert "valley high" in rec["problems"][0]

    def test_a_region_with_no_cell_is_left_out_and_makes_no_outline(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley", "nowhere"])
        assert rec["outlines"] == [] and [f["label"] for f in rec["filled"]] == ["valley"]
        assert rec["problems"] == ["the region 'nowhere' holds no cell, so it is left out"]

    def test_a_label_that_is_not_words_is_left_out(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley low", 7, "  "])
        assert [f["label"] for f in rec["filled"]] == ["valley low"]
        assert rec["problems"] == ["7 is not a region's name, so it is left out",
                                   "'  ' is not a region's name, so it is left out"]

    def test_a_label_asked_twice_is_drawn_once(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley low", "valley low"])
        assert len(rec["filled"]) == 1 and rec["problems"] == []

    def test_one_label_given_as_words_is_one_region(self, store):
        entry, _ = store
        rec = DP.page_record(entry, "valley low")
        assert rec["regions"] == ["valley low"] and len(rec["filled"]) == 1

    def test_no_region_asked_is_said(self, store):
        entry, _ = store
        for asked in ([], None):
            rec = DP.page_record(entry, asked)
            assert rec["filled"] == [] and rec["outlines"] == []
            assert rec["problems"] == ["no region was asked for"]


# ── a control with no store ──────────────────────────────────────────────────
class TestNoStore:
    def test_a_control_with_no_results_store_has_none(self, tmp_path):
        assert DP.page_record({"name": "made_up_valley"}, ["valley"]) is None
        assert DP.page_record({"name": "made_up_valley", "suite_dir": None}, ["valley"]) is None
        assert DP.page_record({"name": "v", "suite_dir": str(tmp_path / "gone")}, ["v"]) is None
        # a folder with no masks in it is no store either
        assert DP.page_record({"name": "v", "suite_dir": str(tmp_path)}, ["v"]) is None
        assert DP.page_record(None, ["v"]) is None

    def test_masks_that_cannot_be_read_are_a_problem_never_an_error(self, tmp_path):
        (tmp_path / S.MASKS_FILE).write_text("not a netCDF file")
        rec = DP.page_record({"name": "v", "suite_dir": str(tmp_path)}, ["valley"])
        assert rec["filled"] == [] and rec["outlines"] == [] and rec["classes"] == []
        assert rec["problems"][0].startswith("the results store's regions cannot be read")


# ── the map's grid ───────────────────────────────────────────────────────────
class TestTheGrid:
    def test_lat_and_lon_are_the_cell_centres_and_the_arrays_lat_by_lon(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley", "valley low"])
        assert rec["lat"] == LAT and rec["lon"] == pytest.approx(WEST, abs=1e-9)
        assert len(rec["classes"]) == 3 and all(len(r) == 4 for r in rec["classes"])
        mask = rec["outlines"][0]["mask"]
        assert len(mask) == 3 and all(len(r) == 4 for r in mask)

    def test_longitudes_written_east_of_greenwich_come_back_west(self, tmp_path):
        """A store whose file holds 0-360 longitudes, written by hand."""
        sd = tmp_path / "store"
        sd.mkdir()
        with netCDF4.Dataset(sd / S.MASKS_FILE, "w") as d:
            d.createDimension("lat", 3)
            d.createDimension("lon", 4)
            d.createVariable("lat", "f8", ("lat",))[:] = LAT
            d.createVariable("lon", "f8", ("lon",))[:] = LON
            d.createVariable("weight_km2", "f8", ("lat", "lon"))[:] = AREA
            v = d.createVariable("mask_valley", "i1", ("lat", "lon"))
            v[:] = VALLEY.astype(np.int8)
            v.region = "valley"
        rec = DP.page_record({"name": "v", "suite_dir": str(sd)}, ["valley"])
        assert rec["lon"] == pytest.approx(WEST, abs=1e-9)
        assert all(-180 <= x <= 180 for x in rec["lon"])
        # a mask written without its count and area is counted from the file
        assert rec["filled"] == [{"label": "valley", "cells": 8, "area_km2": _area(VALLEY)}]

    def test_the_cell_centres_are_kept_to_six_decimals(self, tmp_path):
        """A store's axes may carry more digits than a map needs; the record
        keeps DIGITS of them, so it stays small."""
        lat = [40.12345678912, 40.22222229, 40.3]
        lon = [-109.51234567891, -109.4, -109.3, -109.20000004]
        sd = tmp_path / "store"
        sd.mkdir()
        with netCDF4.Dataset(sd / S.MASKS_FILE, "w") as d:
            d.createDimension("lat", 3)
            d.createDimension("lon", 4)
            d.createVariable("lat", "f8", ("lat",))[:] = lat
            d.createVariable("lon", "f8", ("lon",))[:] = lon
            v = d.createVariable("mask_valley", "i1", ("lat", "lon"))
            v[:] = VALLEY.astype(np.int8)
        rec = DP.page_record({"name": "v", "suite_dir": str(sd)}, ["valley"])
        assert DP.DIGITS == 6
        assert rec["lat"] == [40.123457, 40.222222, 40.3]
        assert rec["lon"] == [-109.512346, -109.4, -109.3, -109.2]

    def test_the_stores_own_count_of_cells_is_kept(self, tmp_path):
        """The store writes each mask's count beside it, and the catalog gives
        the same count; a page never shows a second one counted here."""
        sd = tmp_path / "store"
        sd.mkdir()
        with netCDF4.Dataset(sd / S.MASKS_FILE, "w") as d:
            d.createDimension("lat", 3)
            d.createDimension("lon", 4)
            d.createVariable("lat", "f8", ("lat",))[:] = LAT
            d.createVariable("lon", "f8", ("lon",))[:] = WEST
            v = d.createVariable("mask_valley", "i1", ("lat", "lon"))
            v[:] = VALLEY.astype(np.int8)
            v.region = "valley"
            v.n_cells = 11                      # not the 8 cells of the mask, on purpose
            v.area_km2 = 99.5
        rec = DP.page_record({"name": "v", "suite_dir": str(sd)}, ["valley"])
        assert int(VALLEY.sum()) == 8
        assert rec["filled"] == [{"label": "valley", "cells": 11, "area_km2": 99.5}]

    def test_the_record_is_plain_json_and_small(self, store):
        entry, _ = store
        rec = DP.page_record(entry, ["valley", "valley low", "valley high", "ridge"])
        assert json.loads(json.dumps(rec)) == rec
        for grid in [rec["classes"]] + [o["mask"] for o in rec["outlines"]]:
            assert all(type(x) is int for row in grid for x in row)
        assert {x for o in rec["outlines"] for row in o["mask"] for x in row} <= {0, 1}


# ── the areas the catalog gives ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("places")
    control, _ = make_control(root, [2015, 2016])
    out = root / "suite"
    S.build(control, out, fields=FIELDS)
    return dict(control, suite_dir=str(out)), S.load(out)


class TestTheCatalogsAreas:
    def test_the_areas_are_the_ones_the_catalog_gives(self, built):
        control, manifest = built
        rows = DCAT._regions(control, None, manifest)["regions"]
        names = [r["name"] for r in rows]
        rec = DP.page_record(control, names)
        got = {r["label"]: (r["cells"], r["area_km2"]) for r in rec["filled"] + rec["outlines"]}
        assert got == {r["name"]: (r["cells"], r["area_km2"]) for r in rows if r["cells"]}
        assert rec["regions"] == names
