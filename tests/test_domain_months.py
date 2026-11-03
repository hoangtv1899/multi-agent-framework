"""Months and seasons of a whole-domain experiment, and where its numbers come from.

    pytest tests/test_domain_months.py -v

The synthetic case of test_domain_analysis (the run-gate fixture's grid, a
monthly stream holding GPP, TSA and FAREA_BURNED for 2015 and 2016), each
month's value computed here from the fixture's own formulas, never by the
reader under test.
"""
import numpy as np
import pytest

import domain_analysis as A
import elm_gridded_fixtures as fx
from test_domain_analysis import _case, _grid, _polygon, _veg, _weights

MONTHS = fx.MONTH_DAYS


def _expected_month(var, year, month, mask, c, rate=True, scale=1.0):
    """One month's value over a region from the formulas: the record's area
    x landfrac weighted mean, times the month's seconds for a rate."""
    w = _weights(c)
    nlat, nlon = c["grid"]["shape"]
    doy = sum(MONTHS[:month - 1]) + 15
    v = np.array([[fx.value(var, year, doy, j, i, 2015, nlon) for i in range(nlon)]
                  for j in range(nlat)]) * scale
    mean = float(np.sum(w[mask] * v[mask]) / np.sum(w[mask]))
    return mean * 86400.0 * MONTHS[month - 1] if rate else mean


@pytest.fixture
def one(tmp_path):
    c = _case(tmp_path)
    ms = A.masks(_grid(c), c["surface"], [A.VEGETATED])["masks"]
    return c, A.run_values(c["run_dir"], ms, ["GPP", "TSA"], [2015, 2016])


class TestOneRunsMonths:

    def test_each_month_from_the_formulas(self, one):
        c, out = one
        veg, _ = _veg(c)
        gpp = out["monthly"][A.VEGETATED]["GPP"]
        tsa = out["monthly"][A.VEGETATED]["TSA"]
        assert list(gpp) == [A.month_key(y, m) for y in (2015, 2016) for m in range(1, 13)]
        for y in (2015, 2016):
            for m in range(1, 13):
                k = A.month_key(y, m)
                assert gpp[k] == pytest.approx(_expected_month("GPP", y, m, veg, c),
                                               rel=1e-6), k
                assert tsa[k] == pytest.approx(
                    _expected_month("TSA", y, m, veg, c, rate=False), rel=1e-6), k

    def test_a_rates_months_add_up_to_its_year(self, one):
        _, out = one
        for y in ("2015", "2016"):
            months = [x for k, x in out["monthly"][A.VEGETATED]["GPP"].items()
                      if k.startswith(y)]
            assert sum(months) == pytest.approx(out["values"][A.VEGETATED]["GPP"][y],
                                                rel=1e-9)

    def test_units_per_month_and_season(self, one):
        _, out = one
        assert out["period_units"]["GPP"] == {"month": "gC/m^2/month",
                                              "season": "gC/m^2/season"}
        assert out["period_units"]["TSA"] == {"month": "K", "season": "K"}
        # the yearly record keeps its exact shape
        assert out["fields"]["GPP"] == {"stream": "h0", "how": "sum", "units": "gC/m^2/yr",
                                        "note": None}

    def test_seasons_a_total_and_a_day_weighted_mean(self, one):
        _, out = one
        g = out["monthly"][A.VEGETATED]["GPP"]
        t = out["monthly"][A.VEGETATED]["TSA"]
        sg = out["seasonal"][A.VEGETATED]["GPP"]
        st = out["seasonal"][A.VEGETATED]["TSA"]
        assert sg["2016-JJA"] == pytest.approx(g["2016-06"] + g["2016-07"] + g["2016-08"],
                                               rel=1e-12)
        assert st["2015-SON"] == pytest.approx(
            (t["2015-09"] * 30 + t["2015-10"] * 31 + t["2015-11"] * 30) / 91, rel=1e-12)
        # DJF 2016: December 2015 with January and February 2016
        assert sg["2016-DJF"] == pytest.approx(g["2015-12"] + g["2016-01"] + g["2016-02"],
                                               rel=1e-12)
        assert st["2016-DJF"] == pytest.approx(
            (t["2015-12"] * 31 + t["2016-01"] * 31 + t["2016-02"] * 28) / 90, rel=1e-12)

    def test_a_winter_without_its_december_is_left_out_with_the_reason(self, one):
        _, out = one
        assert "2015-DJF" not in out["seasonal"][A.VEGETATED]["GPP"]
        why = out["seasons_left_out"]["2015-DJF"]
        assert "December 2014" in why and "h0" in why
        assert set(out["seasons_left_out"]) == {"2015-DJF"}

    def test_the_december_before_the_first_year_asked_is_read(self, tmp_path):
        """Asked for 2016 alone, the run's 2015 December is read for DJF 2016,
        and 2015 is reported nowhere else."""
        c = _case(tmp_path)
        ms = A.masks(_grid(c), c["surface"], [A.VEGETATED])["masks"]
        both = A.run_values(c["run_dir"], ms, ["GPP"], [2015, 2016])
        out = A.run_values(c["run_dir"], ms, ["GPP"], [2016])
        assert out["seasonal"][A.VEGETATED]["GPP"]["2016-DJF"] == pytest.approx(
            both["seasonal"][A.VEGETATED]["GPP"]["2016-DJF"], rel=1e-12)
        assert set(out["values"][A.VEGETATED]["GPP"]) == {"2016"}
        assert all(k.startswith("2016") for k in out["monthly"][A.VEGETATED]["GPP"])
        assert "seasons_left_out" not in out
        h0 = out["provenance"]["streams"]["h0"]
        assert h0["years_read"] == [2015, 2016] and h0["records_per_year"] == {
            "2015": 12, "2016": 12}

    def test_where_one_runs_numbers_come_from(self, one):
        c, out = one
        p = out["provenance"]
        assert p["run_dir"] == str(c["run_dir"])
        h0 = p["streams"]["h0"]
        assert h0["fields"] == ["GPP", "TSA"] and h0["cadence"] == "monthly"
        assert h0["records_per_year"] == {"2015": 12, "2016": 12}
        names = [f["name"] for f in h0["files"]]
        assert names == ["synth.elm.h0.2015-02-01-00000.nc", "synth.elm.h0.2016-02-01-00000.nc"]
        assert all(f["bytes"] > 0 and f["modified"].endswith("+00:00") for f in h0["files"])


class TestAnExperimentsMonths:

    def _table(self, tmp_path, normal=()):
        ctl = _case(tmp_path, name="ctl")
        twin = _case(tmp_path, gpp_scale=1.1, name="twin")
        poly = _polygon(tmp_path, ctl)
        control = {"case_dir": str(ctl["case_dir"]), "regions": {"box": {"polygon": str(poly)}}}
        t = A.experiment(control, {"control": ctl["run_dir"], "twin": twin["run_dir"]},
                         {"regions": [A.VEGETATED, "box"], "variables": ["GPP", "TSA"],
                          "normal_years": list(normal)}, [2015, 2016])
        return t, ctl, twin, poly

    def test_monthly_and_seasonal_differences(self, tmp_path):
        t, *_ = self._table(tmp_path)
        c = t["runs"]["control"]["monthly"][A.VEGETATED]["GPP"]
        d = t["monthly_differences"]["twin"][A.VEGETATED]["GPP"]
        assert set(d) == set(c)
        for k in c:
            assert d[k] == pytest.approx(0.1 * c[k], rel=1e-5), k
        assert t["monthly_differences"]["twin"][A.VEGETATED]["TSA"]["2015-07"] == \
            pytest.approx(0.0, abs=1e-9)
        sd = t["seasonal_differences"]["twin"][A.VEGETATED]["GPP"]
        assert set(sd) == {"2015-MAM", "2015-JJA", "2015-SON", "2016-DJF", "2016-MAM",
                           "2016-JJA", "2016-SON"}
        assert "control" not in t["monthly_differences"]

    def test_the_normal_years_monthly_mean(self, tmp_path):
        t, *_ = self._table(tmp_path, normal=[2016])
        ev = t["event"]
        c = t["runs"]["control"]["monthly"][A.VEGETATED]["GPP"]
        mn = ev["monthly_normal"][A.VEGETATED]["GPP"]
        assert list(mn) == [f"{m:02d}" for m in range(1, 13)]
        for m in range(1, 13):
            assert mn[f"{m:02d}"] == pytest.approx(c[A.month_key(2016, m)], rel=1e-12)
        sn = ev["seasonal_normal"][A.VEGETATED]["GPP"]
        assert sn["DJF"] == pytest.approx(
            t["runs"]["control"]["seasonal"][A.VEGETATED]["GPP"]["2016-DJF"])
        assert ev["normal_counts"] == {}

    def test_a_normal_short_of_a_year_says_so(self):
        run = {"monthly": {"r": {"GPP": {"2016-01": 1.0, "2017-01": None, "2016-02": None,
                                         "2017-02": None}}},
               "seasonal": {"r": {"GPP": {"2017-DJF": 4.0}}}}
        out = A.period_normals(run, [2016, 2017])
        assert out["monthly_normal"]["r"]["GPP"]["01"] == 1.0
        assert out["monthly_normal"]["r"]["GPP"]["02"] is None
        assert out["normal_counts"]["r"]["GPP"]["01"] == 1
        assert out["normal_counts"]["r"]["GPP"]["02"] == 0
        assert out["seasonal_normal"]["r"]["GPP"]["DJF"] == 4.0

    def test_the_provenance_block(self, tmp_path):
        t, ctl, twin, poly = self._table(tmp_path)
        p = t["provenance"]
        assert set(p["runs"]) == {"control", "twin"}
        assert p["runs"]["twin"]["run_dir"] == str(twin["run_dir"])
        assert p["runs"]["control"]["streams"]["h0"]["fields"] == ["GPP", "TSA"]
        assert "provenance" not in t["runs"]["control"]
        assert p["surface_file"] == str(ctl["surface"])
        assert p["registered_files"] == {"box": str(poly)}
        r = p["regions"]["box, vegetated"]
        assert r["cells"] == t["regions"]["box, vegetated"]["cells"]
        assert r["sources"] == ["box.geojson", ctl["surface"].name if hasattr(
            ctl["surface"], "name") else str(ctl["surface"]).rsplit("/", 1)[-1]]
