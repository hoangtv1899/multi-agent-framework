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


# ── what the details page is given: the burned area as a table ───────────────
def _fire(model, observed, source="a made-up satellite"):
    fire = {"model": model, "observed": observed, "pattern": "spread over 1-2 cells"}
    if source is not None:
        fire["observed_source"] = source
    return {"burned_area_km2": fire}


def _sentence_pairs(f):
    """The year, model and observed numbers not_in_it's sentence states."""
    import re
    return [(y, int(a), int(b)) for y, a, b in re.findall(r"(\d{4}) (\d+) against (\d+) km2",
                                                        DCT.not_in_it(f)[0])]


class TestTheFireTable:

    def test_the_years_both_give_with_the_control_run_first(self):
        f = _fire({"2041": 12.4, "2042": 7.5, "2043": 30.0},
                  {"2040": 1.0, "2041": 3.2, "2042": 40.6})
        t = DCT.fire_tables(f)
        assert t == [{"about": "fires", "lead": DCT.FIRE_TABLE_LEAD,
                      "corner": "Burned area (km2)", "columns": ["2041", "2042"],
                      "rows": [{"label": "The control run", "values": [12, 8]},
                               {"label": "a made-up satellite", "values": [3, 41]}]}]

    def test_it_says_what_the_sentence_says(self):
        """The same years and the same rounding to whole km2, a half rounded
        as the sentence rounds it."""
        f = _fire({"2041": 0.5, "2042": 1.5, "2043": 2.5, "2044": 99.49},
                  {"2041": 10.5, "2042": 0.4, "2043": 3.5, "2044": 0.0})
        t = DCT.fire_tables(f)[0]
        got = list(zip(t["columns"], t["rows"][0]["values"], t["rows"][1]["values"]))
        assert got == _sentence_pairs(f) == [("2041", 0, 10), ("2042", 2, 0), ("2043", 2, 4),
                                             ("2044", 99, 0)]

    def test_the_lead_is_plain(self):
        lead = DCT.FIRE_TABLE_LEAD
        assert lead.startswith("The control run's fire is the model's own")
        assert ":" not in lead and "\u2014" not in lead and "\u2013" not in lead
        assert "control" in lead and "the control " not in lead.replace("the control run", "")

    @pytest.mark.parametrize("f", [
        None, {}, {"burned_area_km2": {}},
        _fire({"2041": 12.4}, {}),                      # no observed row
        _fire({}, {"2041": 3.2}),                       # no row of the control run
        {"burned_area_km2": {"model": {"2041": 12.4}}},
        {"burned_area_km2": {"observed": {"2041": 3.2}, "observed_source": "x"}},
    ])
    def test_no_table_when_the_facts_lack_a_row(self, f):
        assert DCT.fire_tables(f) == []

    def test_no_table_when_no_year_is_in_both(self):
        assert DCT.fire_tables(_fire({"2041": 12.4, "2042": 1.0}, {"2043": 3.2})) == []

    def test_no_table_of_zeros_for_a_run_with_no_fire(self):
        f = _fire({"2041": 0.0, "2042": 0.0}, {"2041": 3.2, "2042": 40.6})
        assert DCT.fire_tables(f) == []
        assert DCT.not_in_it(f)[0].startswith(DCT._FIRE_NONE)
        # a fire of the control run in one year is a table
        assert DCT.fire_tables(_fire({"2041": 0.0, "2042": 0.3}, {"2041": 3.2, "2042": 40.6}))

    def test_a_source_the_facts_do_not_name(self):
        for src in (None, "", "  "):
            t = DCT.fire_tables(_fire({"2041": 12.4}, {"2041": 3.2}, source=src))
            assert t[0]["rows"][1]["label"] == "The observation"

    def test_a_year_without_a_number_is_left_out(self):
        t = DCT.fire_tables(_fire({"2041": 12.4, "2042": None, "2043": 5.0},
                                  {"2041": 3.2, "2042": 1.0, "2043": float("nan")}))
        assert t[0]["columns"] == ["2041"]

    def test_from_the_facts_of_a_made_up_control(self, tmp_path):
        c, reg, _ = _control(tmp_path)
        f = DCT.compute_facts("synth", str(reg))
        t = DCT.fire_tables(f)[0]
        assert t["columns"] == ["2015", "2016"]
        assert [r["label"] for r in t["rows"]] == ["The control run", "observed burned fraction"]
        assert t["rows"][0]["values"] == [int(f"{f['burned_area_km2']['model'][y]:.0f}")
                                          for y in ("2015", "2016")]
        assert list(zip(t["columns"], *[r["values"] for r in t["rows"]])) == _sentence_pairs(f)

    def test_the_columns_are_strings_whatever_the_facts_key_them_by(self):
        """Facts read from JSON key the years as strings, but facts handed in
        whole may key them as numbers, and a page's table needs words."""
        t = DCT.fire_tables(_fire({2041: 12.4, 2042: 7.5}, {2041: 3.2, 2042: 40.6}))[0]
        assert t["columns"] == ["2041", "2042"]
        assert all(type(c) is str for c in t["columns"])


# ── the observed row's short name, from the registry ─────────────────────────
def _entry(**obs):
    """A registry entry whose burned-area observation is the one the facts of
    _fire() were made from, with the keys given."""
    return {"observations": {"burned": dict({"path": "/made/up.nc", "var": "burned_fraction_year",
                                             "what": "a made-up satellite"}, **obs)}}


class TestTheObservedRowsLabel:
    F = _fire({"2041": 12.4, "2042": 7.5}, {"2041": 3.2, "2042": 40.6})

    def test_a_label_names_the_row(self):
        t = DCT.fire_tables(self.F, _entry(label="Made-up record"))[0]
        assert [r["label"] for r in t["rows"]] == ["The control run", "Made-up record"]
        assert DCT.observed_label(_entry(label=" Made-up record "), self.F) == (
            "Made-up record", [])

    def test_without_a_label_or_an_entry_the_row_keeps_the_source(self):
        for entry in (None, {}, _entry(), {"observations": None}):
            t = DCT.fire_tables(self.F, entry)[0]
            assert t["rows"][1]["label"] == "a made-up satellite"
            assert DCT.observed_label(entry, self.F) == (None, [])

    def test_only_the_label_of_the_observation_the_facts_were_made_from(self):
        """Another observation's label, or one of another variable, names
        nothing in this table."""
        other = {"observations": {
            "burned": {"var": "burned_fraction_year", "what": "another made-up satellite",
                       "label": "Wrong record"},
            "snow": {"var": "snow_cover", "what": "a made-up satellite", "label": "Snow"}}}
        assert DCT.fire_tables(self.F, other)[0]["rows"][1]["label"] == "a made-up satellite"
        # an observation with no `what` is named by its key, as compute_facts names it
        by_key = {"observations": {"a made-up satellite": {
            "var": "burned_fraction_year", "label": "Made-up record"}}}
        assert DCT.fire_tables(self.F, by_key)[0]["rows"][1]["label"] == "Made-up record"

    @pytest.mark.parametrize("label", ["", "  ", 7, None, ["Made-up record"],
                                       {"text": "Made-up record"}])
    def test_a_label_of_another_form_is_left_out_and_named(self, label):
        t = DCT.fire_tables(self.F, _entry(label=label))[0]
        assert t["rows"][1]["label"] == "a made-up satellite"
        assert DCT.observed_label(_entry(label=label), self.F) == (None, [
            "the label of the observation 'burned' is not a name written out, so it is left "
            "out and the burned-area table names the observation by its source"])

    def test_from_the_facts_and_the_registry_of_a_made_up_control(self, tmp_path):
        c, reg, _ = _control(tmp_path)
        d = json.loads(reg.read_text())
        d["controls"]["synth"]["observations"]["burned"]["label"] = "Made-up record"
        reg.write_text(json.dumps(d))
        f = DCT.compute_facts("synth", str(reg))
        t = DCT.fire_tables(f, DCT.get("synth", str(reg)))[0]
        assert [r["label"] for r in t["rows"]] == ["The control run", "Made-up record"]


# ── what the details page is given: the set-up rows ──────────────────────────
ROWS = [{"label": "Area", "words": "50 km2 of made-up land"},
        {"label": "Grid", "words": "10 by  8 cells"},
        {"label": "Fire", "words": "none"}]


class TestTheSetUpRows:

    def test_rows_come_through_as_entered(self):
        rows, bad = DCT.page_setup({"page_setup": json.loads(json.dumps(ROWS))})
        assert rows == ROWS and bad == []

    def test_a_control_without_them_has_none_and_no_problem(self):
        assert DCT.page_setup({}) == ([], []) and DCT.page_setup(None) == ([], [])
        assert DCT.page_setup({"page_setup": []}) == ([], [])

    @pytest.mark.parametrize("value", ["Area: 50 km2", {"label": "Area", "words": "x"}, 7])
    def test_rows_that_are_not_a_list_give_none(self, value):
        rows, bad = DCT.page_setup({"page_setup": value})
        assert rows == [] and bad == ["page_setup is not a list of rows with a label and "
                                      "words, so the control run's set-up is not given"]

    @pytest.mark.parametrize("row", [
        {"label": "Area"}, {"words": "50 km2"}, {"label": "Area", "words": ""},
        {"label": " ", "words": "50 km2"}, {"label": "Area", "words": 50},
        {"label": None, "words": "50 km2"},
        {"label": "Area", "words": "50 km2", "source": "a person"},
        {"Label": "Area", "words": "50 km2"}, ["Area", "50 km2"], "Area, 50 km2", None])
    def test_a_row_of_another_form_is_left_out_by_its_place_and_the_others_kept(self, row):
        rows, bad = DCT.page_setup({"page_setup": [ROWS[0], row, ROWS[2]]})
        assert rows == [ROWS[0], ROWS[2]]
        assert bad == ["page_setup row 2 is not a label and words, both written out, so it "
                       "is left out"]

    def test_a_label_given_twice_keeps_the_first(self):
        again = {"label": "Area ", "words": "another area"}
        rows, bad = DCT.page_setup({"page_setup": ROWS + [again]})
        assert rows == ROWS
        assert bad == ["page_setup row 4 repeats the label 'Area', so it is left out"]

    def test_reception_and_the_planner_are_shown_neither_rows_nor_table(self, tmp_path):
        """What Reception and the Planner read of a control (summary, through
        describe) holds no set-up row, and no number of the table beyond the
        facts it already gave."""
        c, reg, _ = _control(tmp_path)
        d = json.loads(reg.read_text())
        d["controls"]["synth"]["page_setup"] = [{"label": "Made-up row",
                                                 "words": "words only a page prints"}]
        reg.write_text(json.dumps(d))
        DCT.compute_facts("synth", str(reg))
        text = json.dumps(DCT.describe(str(reg)))
        assert "page_setup" not in text and "words only a page prints" not in text
        assert DCT.FIRE_TABLE_LEAD not in text and '"tables"' not in text
