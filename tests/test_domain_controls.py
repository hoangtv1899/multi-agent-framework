"""The prepared whole-domain controls, and the facts read from each one's output.

    pytest tests/test_domain_controls.py -v

A synthetic control (the run-gate fixture's case, with a monthly stream that
carries FAREA_BURNED) whose every fact is known in advance: the burned area
is the fixture's per-second rate times each month's seconds, weighted by
area x landfrac, and nothing in it is a number copied from a real run.
"""
import json
import os
import time
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_controls as DCT
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import paths


def _obs(path, g, values):
    """A burned fraction per cell and year on the model grid, as the hand
    experiment's regridded MODIS MCD64A1 holds it (longitudes west negative)."""
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("lat", len(g["lat"]))
        d.createDimension("lon", len(g["lon"]))
        d.createDimension("year", len(values))
        d.createVariable("lat", "f8", ("lat",))[:] = g["lat"]
        d.createVariable("lon", "f8", ("lon",))[:] = g["lon"] - 360.0
        d.createVariable("year", "i4", ("year",))[:] = sorted(values)
        v = d.createVariable("burned_fraction_year", "f4", ("year", "lat", "lon"),
                             fill_value=np.float32(np.nan))
        v[:] = np.stack([values[y] for y in sorted(values)])
    return path


def _control(tmp_path, with_obs=True):
    c = cf.write_case(tmp_path / "x")
    fx.write_monthly(c["run_dir"], [2015, 2016], case="synth",
                     variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    (c["case_dir"] / "env_run.xml").write_text("".join(
        f'<entry id="{k}" value="{v}">\n' for k, v in (
            ("CCSM_CO2_PPMV", "367.0"), ("ELM_CO2_TYPE", "constant"),
            ("DATM_MODE", "CLMMOSARTTEST"), ("DATM_CLMNCEP_YR_ALIGN", "2013"),
            ("DATM_CLMNCEP_YR_START", "2013"), ("DATM_CLMNCEP_YR_END", "2023"),
            ("RUN_STARTDATE", "2015-01-01"))))
    rec = tmp_path / "proof"
    rec.mkdir()
    (rec / "STATUS.log").write_text(
        "2026-10-23 13:17:16 job=1 stage=y2015 CHECK PASS hours_per_sim_year=0.796 2015: x\n"
        "2026-10-23 14:06:11 job=2 stage=y2016 CHECK PASS hours_per_sim_year=0.783 2016: x\n")
    burn = np.zeros(c["grid"]["shape"], dtype=np.float32)
    burn[0, 0], burn[1, 2] = 0.5, 0.25
    entry = {"description": "a synthetic control", "case_dir": str(c["case_dir"]),
             "run_dir": str(c["run_dir"]),
             "reproduced": {"record_dir": str(rec), "years": [2015, 2016], "how": "bit for bit"},
             "regions": {}, "observations": {}}
    if with_obs:
        o = _obs(tmp_path / "burn.nc", c["grid"], {2015: np.zeros_like(burn), 2016: burn})
        entry["observations"]["burned"] = {"path": str(o), "var": "burned_fraction_year",
                                           "what": "observed burned fraction"}
    reg = tmp_path / "domain_controls.json"
    reg.write_text(json.dumps({"controls": {"synth": entry}}))
    return c, reg, burn


def _weights(g):
    return np.asarray(g["area"]) * np.asarray(g["landfrac"])


class TestTheRegistry:

    def test_no_registry_is_a_machine_with_no_control(self, tmp_path):
        reg = DCT.load(str(tmp_path / "none.json"))
        assert reg["controls"] == {} and "has no prepared whole-domain control" in \
            reg["problems"][0]
        with pytest.raises(DCT.ControlError, match="no prepared control 'naches'"):
            DCT.get("naches", str(tmp_path / "none.json"))

    def test_an_entry_with_a_missing_path_is_unusable(self, tmp_path):
        c, reg, _ = _control(tmp_path)
        d = json.loads(reg.read_text())
        d["controls"]["synth"]["observations"]["burned"]["path"] = str(tmp_path / "gone.nc")
        d["controls"]["broken"] = {"run_dir": str(tmp_path)}
        reg.write_text(json.dumps(d))
        out = DCT.load(str(reg))
        assert out["controls"]["synth"]["usable"] is False
        assert any("observations burned" in p and "is missing" in p for p in out["problems"])
        assert "broken: no case_dir" in out["problems"]
        with pytest.raises(DCT.ControlError, match="cannot be used"):
            DCT.get("synth", str(reg))

    def test_an_unreadable_registry_says_so(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text("{not json")
        assert "cannot be read" in DCT.load(str(p))["problems"][0]

    def test_where_the_registry_is_found(self, tmp_path, monkeypatch):
        default, src = paths.resolve("domain_controls")
        assert default.name == "domain_controls.json" and default.parent.name == "elm-mcp"
        monkeypatch.setenv("IDEAS_DOMAIN_CONTROLS", str(tmp_path / "r.json"))
        assert paths.resolve("domain_controls") == (tmp_path / "r.json",
                                                    "env:IDEAS_DOMAIN_CONTROLS")


class TestTheFacts:

    def test_every_fact_from_the_controls_own_output(self, tmp_path):
        c, reg, burn = _control(tmp_path)
        f = DCT.compute_facts("synth", str(reg))
        assert (f["start_year"], f["forcing_years"], f["output_years"]) == (
            2015, [2013, 2023], [2015, 2016])
        s = f["settings"]
        assert (s["CCSM_CO2_PPMV"], s["ELM_CO2_TYPE"], s["stream_year_first_ndep"]) == (
            "367.0", "constant", 2000.0)
        assert s["requested_fields"]["h0"] == ["GPP", "TSA", "CELLID"]
        # the per-second rate times each month's seconds in the year (the run's
        # first 30-minute step, folded into 2015's January record, is 2014's)
        cell = np.arange(20).reshape(4, 5)
        w = _weights(c["grid"])
        for y in (2015, 2016):
            want = float(np.sum(1e-10 * (1.0 + cell) * 365.0 * 86400.0 * w))
            assert f["burned_area_km2"]["model"][str(y)] == pytest.approx(round(want, 1),
                                                                          abs=0.051)
        exact = DCT.burned_area_km2(c["run_dir"], 2015)["km2"]
        assert exact == pytest.approx(float(np.sum(1e-10 * (1.0 + cell) * 365.0 * 86400.0
                                                   * w)), rel=1e-6)
        assert f["burned_area_km2"]["observed"] == {
            "2015": 0.0, "2016": round(float(0.5 * w[0, 0] + 0.25 * w[1, 2]), 1)}
        assert f["burned_area_km2"]["observed_source"] == "observed burned fraction"
        assert f["per_year"]["2015"]["gpp_gC_m2_yr"] is not None
        assert f["hours_per_model_year"] == pytest.approx((0.796 + 0.783) / 2, abs=1e-3)
        assert set(f["streams"]) == {"h0", "h1", "h2"}
        assert "FAREA_BURNED" in f["streams"]["h0"]["fields"]
        assert set(f["field_notes"]) == {"FAREA_BURNED"}
        assert DCT.facts("synth", str(reg)) == json.loads(json.dumps(f, default=str))

    def test_facts_older_than_the_output_are_not_used(self, tmp_path):
        c, reg, _ = _control(tmp_path, with_obs=False)
        DCT.compute_facts("synth", str(reg))
        assert DCT.facts("synth", str(reg)) is not None
        h = sorted(Path(c["run_dir"]).glob("*.elm.h0.*.nc"))[0]
        later = time.time() + 60
        os.utime(h, (later, later))
        assert DCT.facts("synth", str(reg)) is None

    def test_what_reception_is_shown_is_small(self, tmp_path):
        c, reg, _ = _control(tmp_path)
        before = DCT.describe(str(reg))
        assert before["controls"][0]["facts"] is None
        assert "refresh_domain_control_facts" in before["controls"][0]["facts_note"]
        DCT.compute_facts("synth", str(reg))
        d = DCT.describe(str(reg))
        row = d["controls"][0]
        # GPP, TSA, CELLID, FAREA_BURNED and the fixture's own DOYCODE
        assert row["facts"]["streams"]["h0"] == {"n_fields": 5, "years": [2015, 2016]}
        assert "model_cells" not in row["facts"]["burned_area_km2"]
        assert len(json.dumps(d)) < 12000
        # whole: a run's difference from the control is computed from them
        assert row["facts"]["per_year"] == DCT.facts("synth", str(reg))["per_year"]
