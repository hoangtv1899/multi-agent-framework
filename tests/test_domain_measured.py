"""A control run beside its measurements (domain_measured), on a small fake run.

    pytest tests/test_domain_measured.py -v

The fake run and its store are test_domain_suite's (a 4 x 5 grid, two years,
monthly and daily history). Every measured value here is made up. The model's
side is held to numbers read from the store's files and from the arrays the
history was written from, never to the module's own reading.
"""
import json

import netCDF4
import numpy as np
import pytest

import domain_measured as DM
import domain_methods as M
import domain_suite as S
import elm_gridded_fixtures as fx
from test_domain_suite import FIELDS, HOLE, HOLE_MONTH, make_control

YEARS = [2015, 2016]
MONTHS = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
STATE = "TOTECOSYSC"          # a field the store holds as a mean
FLUX = "GPP"                  # a field the store holds as a total


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("measured")
    control, store = make_control(root, YEARS)
    out = root / "suite"
    S.build(control, out, fields=FIELDS)
    suite = M.Suite(out)
    assert suite.how(STATE) == "mean" and suite.how(FLUX) == "total"
    g = fx.grid()
    return {"root": root, "control": dict(control, suite_dir=str(out)), "store": store,
            "out": out, "suite": suite, "grid": g, "region": sorted(suite.masks)[0]}


def _table(root, name, values, header="month,value"):
    p = root / name
    p.write_text(header + "\n" + "".join(f"{m},{'' if v is None else v}\n"
                                         for m, v in values.items()))
    return str(p)


def _point(g, j, i, name=None):
    return {"id": name or f"p{j}{i}", "lat": float(g["lat"][j]), "lon": float(g["lon"][i] - 360.0)}


def _entry(b, measured, **more):
    return dict(b["control"], measured=measured, **more)


def _series_month(b, field, region):
    with netCDF4.Dataset(b["out"] / "series_month.nc") as d:
        d.set_auto_mask(False)
        regions = [str(x) for x in np.asarray(d["region"][:], dtype=object)]
        return [float(x) for x in np.asarray(d[field][regions.index(region)])]


# ── the table ────────────────────────────────────────────────────────────────
class TestTheTable:
    def test_a_month_with_no_value_is_none(self, tmp_path):
        p = _table(tmp_path, "t.csv", {"2015-01": 3.5, "2015-02": None, "2015-03": 0})
        assert DM.read_table(p) == {"2015-01": 3.5, "2015-02": None, "2015-03": 0.0}

    @pytest.mark.parametrize("text, why", [
        ("when,what\n2015-01,1\n", "month,value"),
        ("month,value\n2015-1,1\n", "YYYY-MM"),
        ("month,value\n2015-13,1\n", "YYYY-MM"),
        ("month,value\n2015-01,1\n2015-01,2\n", "twice"),
        ("month,value\n2015-01,much\n", "no number"),
        ("month,value\n", "no month"),
    ])
    def test_a_table_that_cannot_be_read_says_why(self, tmp_path, text, why):
        p = tmp_path / "bad.csv"
        p.write_text(text)
        with pytest.raises(DM.MeasuredError, match=why):
            DM.read_table(p)

    def test_a_missing_table_says_so(self, tmp_path):
        with pytest.raises(DM.MeasuredError, match="no table"):
            DM.read_table(tmp_path / "none.csv")


# ── a region ─────────────────────────────────────────────────────────────────
class TestARegion:
    def test_the_models_side_is_the_stores_series_and_the_measured_side_the_table(self, built):
        units = built["suite"].units(FLUX, "month")
        made_up = {m: round(40.0 + 3 * k, 1) for k, m in enumerate(MONTHS)}
        made_up["2016-05"] = None
        rec = DM.build(_entry(built, [{"id": "growth", "field": FLUX, "region": built["region"],
                                       "table": _table(built["root"], "g.csv", made_up),
                                       "units": units, "title": "Growth", "label": "a sensor",
                                       "model_words": "Growth", "measured_words": "seen",
                                       "source": "a made-up sensor"}]), write=False)
        assert rec["problems"] == [] and rec["labels"] == MONTHS and rec["step"] == "month"
        s = rec["series"][0]
        assert s["model"] == pytest.approx(_series_month(built, FLUX, built["region"]))
        assert s["measured"] == [made_up[m] for m in MONTHS]
        assert s["months_both"] == 23 and s["how"] == "total" and s["units"] == units
        assert s["where"] == built["region"]
        assert s["cells"] == int(built["suite"].masks[built["region"]].sum())
        assert (s["title"], s["label"], s["source"]) == ("Growth", "a sensor", "a made-up sensor")

    def test_months_outside_the_stores_are_left_out(self, built):
        units = built["suite"].units(FLUX, "month")
        t = _table(built["root"], "wide.csv", {"2014-12": 9.0, "2015-01": 1.0, "2017-01": 9.0})
        rec = DM.build(_entry(built, [{"id": "a", "field": FLUX, "region": built["region"],
                                       "table": t, "units": units}]), write=False)
        s = rec["series"][0]
        assert s["measured"] == [1.0] + [None] * 23 and s["months_both"] == 1


# ── points ───────────────────────────────────────────────────────────────────
class TestPoints:
    def test_the_model_is_read_at_the_cells_that_hold_the_points_and_averaged(self, built):
        g, units = built["grid"], built["suite"].units(STATE, "month")
        cells = [(1, 1), (0, 3)]
        want = []
        for y in YEARS:
            a = built["store"][("h0", y)][STATE]
            want += [float(np.mean([a[m, j, i] for j, i in cells])) for m in range(12)]
        t = _table(built["root"], "state.csv", {m: 1000.0 + k for k, m in enumerate(MONTHS)})
        rec = DM.build(_entry(built, [{"id": "store", "field": STATE, "table": t,
                                       "units": units,
                                       "points": [_point(g, j, i) for j, i in cells]}]),
                       write=False)
        assert rec["problems"] == []
        s = rec["series"][0]
        assert s["model"] == pytest.approx(want, rel=1e-6)
        assert s["where"] == "the 2 cells that hold the points" and s["cells"] == 2

    def test_a_month_in_which_one_cell_has_no_value_has_none(self, built):
        g, units = built["grid"], built["suite"].units(STATE, "month")
        t = _table(built["root"], "hole.csv", {m: 5.0 for m in MONTHS})
        rec = DM.build(_entry(built, [{"id": "h", "field": STATE, "table": t, "units": units,
                                       "points": [_point(g, 1, 1), _point(g, *HOLE)]}]),
                       write=False)
        s = rec["series"][0]
        hole = f"{HOLE_MONTH[0]}-{HOLE_MONTH[1]:02d}"
        assert s["model"][MONTHS.index(hole)] is None
        assert sum(v is None for v in s["model"]) == 1 and s["months_both"] == 23

    def test_a_point_outside_the_grid_is_refused_by_name(self, built):
        g, units = built["grid"], built["suite"].units(STATE, "month")
        t = _table(built["root"], "out.csv", {m: 5.0 for m in MONTHS})
        far = {"id": "far_station", "lat": 10.0, "lon": 20.0}
        rec = DM.build(_entry(built, [{"id": "o", "field": STATE, "table": t, "units": units,
                                       "points": [_point(g, 1, 1), far]}]), write=False)
        assert rec["series"] == []
        assert "o: outside the model's grid: far_station" in rec["problems"][0]

    def test_a_monthly_total_is_not_read_at_points(self, built):
        g = built["grid"]
        t = _table(built["root"], "tot.csv", {m: 5.0 for m in MONTHS})
        rec = DM.build(_entry(built, [{"id": "t", "field": FLUX, "table": t,
                                       "units": built["suite"].units(FLUX, "month"),
                                       "points": [_point(g, 1, 1)]}]), write=False)
        assert rec["series"] == [] and "holds as a mean" in rec["problems"][0]


# ── what is refused, and what is shared ──────────────────────────────────────
class TestTheRules:
    def _one(self, built, **over):
        m = {"id": "x", "field": FLUX, "region": built["region"],
             "units": built["suite"].units(FLUX, "month"),
             "table": _table(built["root"], "x.csv", {m: 2.0 for m in MONTHS})}
        m.update(over)
        return DM.build(_entry(built, [m]), write=False)

    def test_units_that_differ_from_the_stores_are_refused_never_converted(self, built):
        rec = self._one(built, units="furlongs")
        assert rec["series"] == []
        assert "the table is in furlongs" in rec["problems"][0] and "convert" in rec["problems"][0]

    def test_an_unknown_region_names_the_ones_there_are(self, built):
        rec = self._one(built, region="nowhere")
        assert "no region 'nowhere'" in rec["problems"][0]
        assert built["region"] in rec["problems"][0]

    def test_an_unknown_field_is_refused(self, built):
        assert "holds no ZZZ" in self._one(built, field="ZZZ")["problems"][0]

    def test_a_region_and_points_together_are_refused(self, built):
        rec = self._one(built, points=[_point(built["grid"], 1, 1)])
        assert "one of the two" in rec["problems"][0]

    def test_no_month_in_common_is_refused(self, built):
        t = _table(built["root"], "early.csv", {"2001-01": 2.0})
        assert "no month has both" in self._one(built, table=t)["problems"][0]

    def test_one_bad_measurement_leaves_the_others_in(self, built):
        units = built["suite"].units(FLUX, "month")
        t = _table(built["root"], "ok.csv", {m: 2.0 for m in MONTHS})
        good = {"id": "good", "field": FLUX, "region": built["region"], "units": units,
                "table": t}
        rec = DM.build(_entry(built, [dict(good, id="bad", units="wrong"), good,
                                       dict(good)]), write=False)
        assert [s["id"] for s in rec["series"]] == ["good"]
        assert len(rec["problems"]) == 2 and "used twice" in rec["problems"][1]

    def test_two_measurements_of_one_thing_share_the_models_series(self, built):
        units = built["suite"].units(FLUX, "month")
        a = {"id": "a", "field": FLUX, "region": built["region"], "units": units,
             "table": _table(built["root"], "a.csv", {m: 2.0 for m in MONTHS})}
        b = dict(a, id="b", table=_table(built["root"], "b.csv", {m: 3.0 for m in MONTHS}))
        rec = DM.build(_entry(built, [a, b]), write=False)
        assert rec["series"][0]["model"] == rec["series"][1]["model"]
        assert DM.where_key(a) == DM.where_key(b)
        assert DM.where_key(a) != DM.where_key(dict(a, region="other"))


# ── the file in the store ────────────────────────────────────────────────────
class TestTheFile:
    def test_it_is_written_into_the_store_and_read_back(self, built):
        units = built["suite"].units(FLUX, "month")
        t = _table(built["root"], "file.csv", {m: 2.0 for m in MONTHS})
        e = _entry(built, [{"id": "f", "field": FLUX, "region": built["region"],
                            "units": units, "table": t}])
        before = S.load(built["out"])
        rec = DM.build(e)
        assert (built["out"] / DM.FILE).is_file()
        assert DM.load(built["out"]) == json.loads(json.dumps(rec))
        assert rec["series"][0]["table"]["sha256"] == DM._sha256(t)
        assert S.load(built["out"]) == before          # the store's own manifest is untouched

    def test_another_version_or_no_file_is_none(self, built, tmp_path):
        assert DM.load(tmp_path) is None and DM.load(None) is None
        (tmp_path / DM.FILE).write_text(json.dumps({"version": DM.VERSION + 1}))
        assert DM.load(tmp_path) is None

    def test_no_store_is_an_error_that_names_the_control(self, tmp_path):
        with pytest.raises(DM.MeasuredError, match="valley: no results store"):
            DM.build({"name": "valley", "suite_dir": str(tmp_path / "none")})


# ── the maps ─────────────────────────────────────────────────────────────────
def _year_maps(b, field):
    with netCDF4.Dataset(b["out"] / "maps_year.nc") as d:
        d.set_auto_mask(False)
        return np.asarray(d[field][:], dtype=np.float64)


class TestTheMaps:
    def test_a_map_is_the_mean_of_the_year_maps(self, built):
        got = DM.mean_maps(built["suite"], [{"words": "Growth", "fields": [FLUX]}])
        assert got["problems"] == []
        p = got["panels"][0]
        want = np.nanmean(_year_maps(built, FLUX), axis=0)
        have = np.array([[np.nan if x is None else x for x in row] for row in p["values"]])
        assert np.allclose(have, want, rtol=1e-3, equal_nan=True)
        assert (p["words"], p["years"], p["how"]) == ("Growth", YEARS, "total")
        assert p["units"] == built["suite"].units(FLUX, "year")
        assert len(got["lat"]) == 4 and len(got["lon"]) == 5

    def test_fields_of_one_unit_are_summed_before_the_mean(self, built):
        other = next(f for f in FIELDS if f != FLUX
                     and built["suite"].units(f, "year") == built["suite"].units(FLUX, "year")
                     and built["suite"].map(f, "2015") is not None)
        got = DM.mean_maps(built["suite"], [{"words": "Both", "fields": [FLUX, other]}])
        want = np.nanmean(_year_maps(built, FLUX) + _year_maps(built, other), axis=0)
        have = np.array([[np.nan if x is None else x for x in row]
                         for row in got["panels"][0]["values"]])
        assert np.allclose(have, want, rtol=1e-3, equal_nan=True)

    def test_fields_of_different_units_are_refused(self, built):
        got = DM.mean_maps(built["suite"], [{"words": "Mixed", "fields": [FLUX, STATE]}])
        assert got["panels"] == [] and "different units" in got["problems"][0]

    def test_the_outline_is_the_regions_mask(self, built):
        got = DM.mean_maps(built["suite"], [{"words": "G", "fields": [FLUX]}],
                           outline=built["region"])
        assert np.array_equal(np.array(got["outline"]["mask"], bool),
                              built["suite"].masks[built["region"]])
        assert "no region 'nowhere'" in DM.mean_maps(built["suite"], [], "nowhere")["problems"][0]


# ── what the pages are given ─────────────────────────────────────────────────
class TestThePageRecord:
    def test_it_holds_the_pairs_and_the_maps(self, built):
        units = built["suite"].units(FLUX, "month")
        t = _table(built["root"], "page.csv", {m: 2.0 for m in MONTHS})
        e = _entry(built, [{"id": "p", "field": FLUX, "region": built["region"],
                            "units": units, "table": t}],
                   page_figures={"maps": [{"words": "Growth", "fields": [FLUX]}],
                                 "outline": built["region"]})
        DM.build(e)
        rec = DM.page_record(e)
        assert rec["problems"] == [] and rec["control"] == "synth"
        assert [s["id"] for s in rec["measured"]["series"]] == ["p"]
        assert rec["maps"]["panels"][0]["words"] == "Growth"
        assert rec["maps"]["outline"]["region"] == built["region"]

    def test_a_control_with_no_store_says_so(self):
        rec = DM.page_record({"name": "valley"})
        assert rec["measured"] is None and rec["maps"] is None
        assert rec["problems"] == ["the control has no results store"]

    def test_measurements_registered_but_not_built_name_the_command(self, built, tmp_path):
        import shutil
        bare = tmp_path / "suite"
        shutil.copytree(built["out"], bare)
        (bare / DM.FILE).unlink(missing_ok=True)
        rec = DM.page_record(dict(built["control"], suite_dir=str(bare),
                                  measured=[{"id": "x"}]))
        assert "python domain_measured.py synth builds it" in rec["problems"][0]
