"""One whole-domain experiment's numbers by region: every run, every year.

    pytest tests/test_domain_analysis.py -v

A synthetic case (the run-gate fixture's grid, a monthly stream holding GPP,
TSA and FAREA_BURNED), its numbers computed here from the fixture's own
formulas, never by the reader under test.
"""
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_analysis as A
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import regions as R

MONTHS = fx.MONTH_DAYS


def _case(tmp_path, gpp_scale=1.0, name="x"):
    c = cf.write_case(tmp_path / name)
    fx.write_monthly(c["run_dir"], [2015, 2016], case="synth",
                     variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    if gpp_scale != 1.0:
        for p in Path(c["run_dir"]).glob("synth.elm.h0.*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["GPP"][:] = d["GPP"][:] * gpp_scale
    return c


def _veg(c):
    s = R.read_surface(c["surface"], _grid(c), ("PCT_NATVEG", "TOPO"))
    return R.vegetated(s["PCT_NATVEG"]), s["TOPO"]


def _grid(c):
    import gridded as G
    return G.open_stream(c["run_dir"], "h0").grid


def _weights(c):
    return np.asarray(c["grid"]["area"]) * np.asarray(c["grid"]["landfrac"])


def _expected_total(var, year, mask, c, scale=1.0):
    """The year's total of a per-second rate over a region, from the formulas:
    each month's value (labelled mid-month) times its seconds in the year. The
    run's first 30-minute step, folded into 2015's January record, is 2014's."""
    w = _weights(c)
    nlat, nlon = c["grid"]["shape"]
    tot = 0.0
    doy0 = 0
    for m, days in enumerate(MONTHS, start=1):
        width = days
        v = np.array([[fx.value(var, year, doy0 + 15, j, i, 2015, nlon)
                       for i in range(nlon)] for j in range(nlat)]) * scale
        tot += float(np.sum(w[mask] * v[mask]) / np.sum(w[mask])) * 86400.0 * width
        doy0 += days
    return tot


def _polygon(tmp_path, c, rows=2):
    lat, lon = c["grid"]["lat"], c["grid"]["lon"] - 360.0
    d = (lat[1] - lat[0]) / 2
    ring = [[float(lon[0] - d), float(lat[0] - d)], [float(lon[-1] + d), float(lat[0] - d)],
            [float(lon[-1] + d), float(lat[rows - 1] + d)],
            [float(lon[0] - d), float(lat[rows - 1] + d)], [float(lon[0] - d), float(lat[0] - d)]]
    p = tmp_path / "box.geojson"
    p.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    return p


def _burn_obs(tmp_path, c):
    g = c["grid"]
    burn = np.zeros(g["shape"], dtype=np.float32)
    burn[0, 0], burn[1, 2] = 0.5, 0.25
    p = tmp_path / "burn.nc"
    with netCDF4.Dataset(p, "w") as d:
        d.createDimension("lat", len(g["lat"]))
        d.createDimension("lon", len(g["lon"]))
        d.createDimension("year", 2)
        d.createVariable("lat", "f8", ("lat",))[:] = g["lat"]
        d.createVariable("lon", "f8", ("lon",))[:] = g["lon"] - 360.0
        d.createVariable("year", "i4", ("year",))[:] = [2015, 2016]
        d.createVariable("burned_fraction_year", "f4", ("year", "lat", "lon"))[:] = \
            np.stack([np.zeros_like(burn), burn])
    return p, burn


class TestTheRegions:

    def test_standard_registered_and_unknown(self, tmp_path):
        c = _case(tmp_path)
        poly = _polygon(tmp_path, c)
        obs, burn = _burn_obs(tmp_path, c)
        reg = {"box": {"polygon": str(poly)},
               "burned": {"field": str(obs), "var": "burned_fraction_year", "op": ">",
                          "value": 0.0, "over": "any"}}
        out = A.masks(_grid(c), c["surface"], [A.VEGETATED, A.BANDS, "box", "burned",
                                               "nowhere"], reg)
        ms = out["masks"]
        veg, topo = _veg(c)
        assert np.array_equal(ms[A.VEGETATED], veg)
        assert "box, vegetated" in ms.names() and "burned, vegetated" in ms.names()
        box = np.zeros_like(veg)
        box[:2, :] = True
        assert np.array_equal(ms["box, vegetated"], box & veg)
        assert np.array_equal(ms["burned, vegetated"], (burn > 0) & veg)
        for b, m in R.elevation_bands(topo, A.BAND_EDGES).items():
            if (veg & m).any():
                assert np.array_equal(ms[b], veg & m)
        assert len(out["problems"]) == 1 and "'nowhere'" in out["problems"][0]


class TestOneRunsValues:

    def test_totals_and_means_from_the_formulas(self, tmp_path):
        c = _case(tmp_path)
        ms = A.masks(_grid(c), c["surface"], [A.VEGETATED])["masks"]
        out = A.run_values(c["run_dir"], ms, ["GPP", "TSA", "FAREA_BURNED"], [2015, 2016])
        veg, _ = _veg(c)
        for y in (2015, 2016):
            got = out["values"][A.VEGETATED]["GPP"][str(y)]
            assert got == pytest.approx(_expected_total("GPP", y, veg, c), rel=1e-6)
            fire = out["values"][A.VEGETATED]["FAREA_BURNED"][str(y)]
            assert fire == pytest.approx(_expected_total("FAREA_BURNED", y, veg, c), rel=1e-6)
        assert out["fields"]["GPP"] == {"stream": "h0", "how": "sum", "units": "gC/m^2/yr",
                                        "note": None}
        assert out["fields"]["TSA"]["how"] == "mean"
        assert "PER SECOND" in out["fields"]["FAREA_BURNED"]["note"]
        assert out["problems"] == []

    def test_what_cannot_be_made_is_named(self, tmp_path):
        c = _case(tmp_path)
        ms = A.masks(_grid(c), c["surface"], [A.VEGETATED])["masks"]
        out = A.run_values(c["run_dir"], ms, ["GPP", "NOPE"], [2015, 2016, 2017])
        assert any("NOPE is in none of the run's streams" in p for p in out["problems"])
        assert any("has no complete 2017" in p for p in out["problems"])
        assert set(out["values"][A.VEGETATED]["GPP"]) == {"2015", "2016"}


class TestAnExperiment:

    def test_the_whole_table(self, tmp_path):
        ctl = _case(tmp_path, name="ctl")
        twin = _case(tmp_path, gpp_scale=1.1, name="twin")
        poly = _polygon(tmp_path, ctl)
        obs, burn = _burn_obs(tmp_path, ctl)
        control = {"case_dir": str(ctl["case_dir"]),
                   "regions": {"box": {"polygon": str(poly)}},
                   "observations": {"burn": {"path": str(obs), "var": "burned_fraction_year",
                                             "what": "observed burned fraction"}}}
        t = A.experiment(control, {"control": ctl["run_dir"], "twin": twin["run_dir"]},
                         {"regions": [{"name": A.VEGETATED}, {"name": "box"}],
                          "variables": ["GPP"], "normal_years": [2016]}, [2015, 2016])
        assert set(t["regions"]) == {A.VEGETATED, "box, vegetated"}
        veg, _ = _veg(ctl)
        assert t["regions"][A.VEGETATED]["cells"] == int(veg.sum())
        c15 = t["runs"]["control"]["values"][A.VEGETATED]["GPP"]["2015"]
        d15 = t["differences"]["twin"][A.VEGETATED]["GPP"]["2015"]
        assert d15 == pytest.approx(0.1 * c15, rel=1e-5)
        ev = t["event"]
        assert ev["normal_years"] == [2016] and ev["target_years"] == [2015]
        a = ev["anomalies"][A.VEGETATED]["GPP"]
        assert a["normal"] == pytest.approx(
            t["runs"]["control"]["values"][A.VEGETATED]["GPP"]["2016"])
        # THE MODEL'S OUTPUT ONLY: an observation registered for the control
        # is not read into the table (2026-10-27)
        assert "observed" not in t
        # no fire field asked for, no burned area made
        assert "burned_area_km2" not in t["runs"]["control"]["fields"]
        assert t["problems"] == []

    def test_each_runs_own_burned_area_is_a_field_of_its_values(self, tmp_path):
        """Asked for FAREA_BURNED, each run gets its burned area in km2 beside
        its other fields: the year's burned fraction times the region's area,
        with its difference from the control like any field."""
        ctl = _case(tmp_path, name="ctl")
        twin = _case(tmp_path, name="twin")
        obs, _ = _burn_obs(tmp_path, ctl)
        control = {"case_dir": str(ctl["case_dir"]),
                   "observations": {"burn": {"path": str(obs),
                                             "var": "burned_fraction_year"}}}
        t = A.experiment(control, {"control": ctl["run_dir"], "twin": twin["run_dir"]},
                         {"regions": [A.VEGETATED], "variables": ["GPP", "FAREA_BURNED"]},
                         [2015, 2016])
        assert "observed" not in t
        veg, _ = _veg(ctl)
        area = t["regions"][A.VEGETATED]["area_km2"]
        for run, case in (("control", ctl), ("twin", twin)):
            km2 = t["runs"][run]["values"][A.VEGETATED]["burned_area_km2"]
            frac = t["runs"][run]["values"][A.VEGETATED]["FAREA_BURNED"]
            for y in ("2015", "2016"):
                assert km2[y] == pytest.approx(frac[y] * area, rel=1e-12)
                assert frac[y] == pytest.approx(
                    _expected_total("FAREA_BURNED", int(y), veg, case), rel=1e-6)
        meta = t["runs"]["control"]["fields"]["burned_area_km2"]
        assert meta["how"] == "sum" and meta["units"] == "km2"
        assert "the model's own burned area" in meta["note"]
        assert t["differences"]["twin"][A.VEGETATED]["burned_area_km2"]["2016"] == \
            pytest.approx(0.0, abs=1e-9)
        assert t["problems"] == []

    def test_a_run_without_the_field_is_a_problem_not_a_zero(self, tmp_path):
        c = cf.write_case(tmp_path / "nofield")
        for q in list(Path(c["run_dir"]).glob("*.elm.h[12].*.nc")):
            q.unlink()                     # the daily streams hold the field too
        fx.write_monthly(c["run_dir"], [2015, 2016], case="synth",
                         variables=("GPP", "TSA", "CELLID"))
        t = A.experiment({"case_dir": str(c["case_dir"])}, {"control": c["run_dir"]},
                         {"regions": [A.VEGETATED], "variables": ["GPP", "FAREA_BURNED"]},
                         [2015, 2016])
        assert "burned_area_km2" not in t["runs"]["control"]["fields"]
        assert any("FAREA_BURNED is in none of the run's streams" in q
                   for q in t["runs"]["control"]["problems"])

    def test_the_limitations_are_in_the_shape_step_0_reads(self):
        for kind, items in A.LIMITATIONS.items():
            for lim in items:
                assert lim["kind"] == kind and lim["caveat"] and lim["applies_to"]
                assert lim["severity"] in ("blocking", "qualify", "context")

    def test_a_plan_that_names_no_variable(self, tmp_path):
        c = _case(tmp_path)
        with pytest.raises(A.AnalysisError, match="names no variables"):
            A.experiment({"case_dir": str(c["case_dir"])}, {"control": c["run_dir"]},
                         {"regions": []}, [2015])


class TestAPercentForKilogramsNotKelvin:
    """KELVIN, NOT KILOGRAMS (2026-11-03): an event's percent was refused for
    any unit starting with K, so soil water in kg/m2 had none, as if it were a
    temperature."""

    def test_which_units_are_kelvin(self):
        assert A.is_kelvin("K") and A.is_kelvin(" K ") and A.is_kelvin("K/s")
        assert not A.is_kelvin("kg/m2") and not A.is_kelvin("KG/M2")
        assert not A.is_kelvin("gC/m^2/yr") and not A.is_kelvin(None)

    def test_soil_water_gets_a_percent_and_temperature_does_not(self):
        run = {"values": {"r": {"SOILWATER_10CM": {"2015": 20.0, "2016": 25.0},
                                "TSA": {"2015": 282.0, "2016": 281.0}}},
               "fields": {"SOILWATER_10CM": {"units": "kg/m2"}, "TSA": {"units": "K"}}}
        ev = A.event(run, [2016], [2015])["anomalies"]["r"]
        assert ev["SOILWATER_10CM"]["years"][2015]["anomaly_pct"] == pytest.approx(-20.0)
        assert ev["TSA"]["years"][2015]["anomaly_pct"] is None
