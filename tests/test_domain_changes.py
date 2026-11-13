"""The changes a whole-domain experiment may make to its control run.

    pytest tests/test_domain_changes.py -v

Every run of a whole-domain experiment is the control plus changes from the
server's list (docs/ELM_DOMAIN_EXPERIMENTS.md). These tests hold the list to
what it declares: each change's recipe edit, its checks, and the check of a
proposed experiment against the list and the control's facts.
"""
import json
import re
from pathlib import Path

import pytest

import domain_case as DC
import domain_changes as CH

ROOT = Path(__file__).resolve().parents[1]


def _control(tmp_path, **kw):
    base = dict(case="ctl", case_root=str(tmp_path), srcroot="/src/E3SM", compset="C",
                xml=[("NTASKS", "40"), ("RUN_STARTDATE", "2015-01-01")],
                user_nl={"elm": ["finidat = '/in/f.nc'", "fsurdat = '/in/s.nc'",
                                 "use_nofire = .false.", "suplnitro = 'NONE'",
                                 "suplphos = 'ALL'", "stream_year_first_ndep = 2000",
                                 "stream_year_last_ndep = 2000"]},
                start="2015-01-01", copied_from="/cases/the_control")
    base.update(kw)
    return DC.Recipe(**base)


FACTS = {"start_year": 2015, "forcing_years": [2013, 2023], "output_years": [2015, 2021],
         "settings": {"use_nofire": False, "co2_ppmv": 367.0, "ELM_CO2_TYPE": "constant",
                      "suplnitro": "NONE", "suplphos": "ALL",
                      "stream_year_first_ndep": 2000.0, "stream_year_last_ndep": 2000.0,
                      "ndep_stream_years": [1849, 2106]},
         "burned_area_km2": {"model": {"2015": 115.2, "2021": 132.6},
                             "pattern": "spread over 2903-5296 cells",
                             "observed": {"2015": 3.7, "2021": 344.2},
                             "observed_source": "MODIS MCD64A1"},
         "hours_per_model_year": 0.79}
CONTROL = {"name": "naches", "facts": FACTS,
           "reproduced": {"years": [2015, 2016], "how": "bit for bit",
                          "fingerprint": "503a94c5d1a0362c54e9"}}


def _exp(*runs, years=(2015, 2016), reuse=True):
    return {"control": {"name": "naches", "reuse": reuse}, "years": list(years),
            "runs": list(runs)}


# ── the list ─────────────────────────────────────────────────────────────────
class TestTheList:

    def test_every_change_says_what_it_does_and_does_not_mean(self):
        d = CH.declare()
        ids = [c["id"] for c in d["changes"]]
        assert len(ids) == len(set(ids))
        for c in d["changes"]:
            assert c["summary"] and c["meaning"] and c["not_meaning"] and c["cost"], c["id"]
            if c["available"]:
                # a change that writes files states its checks in words; the
                # writer makes them when the run is built
                assert c["alters"] and (c["checks"]["settings"]
                                        or c["checks"].get("by_writer")), c["id"]
            else:
                assert c["requires"] and not c["alters"], c["id"]

    def test_the_design_note_lists_every_change(self):
        """The planner reads the server; a person reads the note. They must agree."""
        note = (ROOT / "docs" / "ELM_DOMAIN_EXPERIMENTS.md").read_text()
        rows = dict(re.findall(r"^\| `(\w+)` \| .* \| (yes|no)", note, re.M))
        assert rows == {c["id"]: ("yes" if c["available"] else "no")
                        for c in CH.declare()["changes"]}

    def test_it_fits_a_tool_result(self):
        """As Reception receives it, wrapper and all, under the cap the loop
        gives a framework tool's answer, with room left: at the data tools'
        12,000 the list had 2 characters to spare (2026-11-10)."""
        from agents import tool_loop
        from core import model_servers as MS

        class Server:
            def list_tools_detailed(self):
                return [{"name": MS.DOMAIN_MENUS["changes"]}]

            def list_tools(self):
                return [MS.DOMAIN_MENUS["changes"]]

            def call_tool_json(self, tool, args):
                return CH.declare()
        got = MS.describe_domain({"elm": Server()}, "elm", "changes")
        assert got["offers_whole_domain_experiments"]
        assert len(json.dumps(got, default=str)) < tool_loop._MAX_LOCAL_RESULT_CHARS - 6000

    @pytest.mark.parametrize("item,problem", [
        ({"id": "nope"}, "no change 'nope'"),
        ({"id": "burn_footprint"}, "is not available here: it needs a burn tool"),
        ({"id": "co2_ppmv"}, "co2_ppmv needs ppmv"),
        ({"id": "co2_ppmv", "params": {"ppmv": "500"}}, "is not a number"),
        ({"id": "co2_ppmv", "params": {"ppmv": 5000}}, "outside 180.0-1200.0 ppmv"),
        ({"id": "co2_ppmv", "params": {"ppmv": 500, "x": 1}}, "has no parameter 'x'"),
        ({"id": "ndep_year", "params": {"year": 1850.5}}, "is not a whole number"),
        ({"id": "nutrient_supplement", "params": {"nitrogen": "SOME", "phosphorus": "ALL"}},
         "is not one of ['NONE', 'ALL']"),
        ("fire_off", "a change is"),
    ])
    def test_what_does_not_resolve(self, item, problem):
        with pytest.raises(CH.ChangeError, match=re.escape(problem)):
            CH.resolve(item)

    def test_a_resolved_change_has_its_values_filled(self):
        c = CH.resolve({"id": "co2_ppmv", "params": {"ppmv": 560}})
        assert c["alters"] == [{"xml": "CCSM_CO2_PPMV", "value": "560"}]
        assert c["checks"]["settings"] == [{"file": "lnd_in", "key": "co2_ppmv", "value": "560"}]
        assert "560 ppmv" in c["meaning"]
        y = CH.resolve({"id": "ndep_year", "params": {"year": 1850.0}})
        assert y["params"] == {"year": 1850} and "1850 level" in y["meaning"]


# ── a run's recipe ───────────────────────────────────────────────────────────
class TestARunsRecipe:

    def test_fire_off(self, tmp_path):
        ctl = _control(tmp_path)
        r = CH.run_recipe(ctl, {"name": "nofire", "changes": [{"id": "fire_off"}]})
        assert r.case == "ctl_nofire" and r.case_root == ctl.case_root
        assert r.user_nl["elm"] == [l.replace("use_nofire = .false.", "use_nofire = .true.")
                                    for l in ctl.user_nl["elm"]]
        assert r.copied_from == "/cases/the_control" and r.source == {}
        assert r.notes[-1] == "run nofire of an experiment on ctl: fire_off"
        assert ctl.user_nl["elm"][2] == "use_nofire = .false." and ctl.case == "ctl"
        assert r.check() == []

    def test_a_key_the_control_does_not_set_is_added(self, tmp_path):
        ctl = _control(tmp_path, user_nl={"elm": ["finidat = '/in/f.nc'"]})
        r = CH.run_recipe(ctl, {"name": "nofire", "changes": [{"id": "fire_off"}]})
        assert r.user_nl["elm"] == ["finidat = '/in/f.nc'", "use_nofire = .true."]

    def test_xml_and_several_keys(self, tmp_path):
        ctl = _control(tmp_path)
        r = CH.run_recipe(ctl, {"name": "co2_ndep", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560}},
            {"id": "ndep_year", "params": {"year": 1850}}]})
        assert ("CCSM_CO2_PPMV", "560") in r.xml and r.xml[:2] == ctl.xml
        assert "stream_year_first_ndep = 1850" in r.user_nl["elm"]
        assert "stream_year_last_ndep = 1850" in r.user_nl["elm"]
        ctl2 = _control(tmp_path, xml=[("CCSM_CO2_PPMV", "367.0", {"subgroup": "x"}),
                                       ("RUN_STARTDATE", "2015-01-01")])
        r2 = CH.run_recipe(ctl2, {"name": "co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 400}}]})
        assert r2.xml[0] == ("CCSM_CO2_PPMV", "400", {"subgroup": "x"}) and len(r2.xml) == 2

    @pytest.mark.parametrize("run,problem", [
        ({"name": "Fire", "changes": []}, "run name 'Fire'"),
        ({"name": "control", "changes": []}, "not 'control'"),
        ({"name": "twice", "changes": [{"id": "fire_off"}, {"id": "fire_off"}]},
         "fire_off and fire_off both alter user_nl_elm:use_nofire"),
        ({"name": "n", "changes": [{"id": "burn_footprint"}]}, "is not available here"),
        ({"name": "n", "changes": [{"id": "weather_scale"}]}, "weather_scale changes nothing"),
    ])
    def test_what_is_refused(self, tmp_path, run, problem):
        with pytest.raises(CH.ChangeError, match=re.escape(problem)):
            CH.run_recipe(_control(tmp_path), run)

    def test_the_checks_the_run_carries(self):
        e = CH.expectations({"name": "nofire", "changes": [{"id": "fire_off"}]}, "/c/ctl")
        assert e["settings"] == {"lnd_in": {"use_nofire": True}}
        assert e["output"] == [{"stream": "h0", "field": "FAREA_BURNED", "rule": "all_zero",
                                "why": "no fire can burn with the fire model off",
                                "change": "fire_off"}]
        assert e["control_case_dir"] == "/c/ctl" and e["changes"][0]["id"] == "fire_off"


# ── the settings check ───────────────────────────────────────────────────────
LND = """&elm_inparm
 co2_ppmv = 367.0
 fsurdat = '/in/s.nc'
 hist_fincl1 = 'GPP','TSA'
 suplnitro = 'NONE'
 use_nofire = .false.
 finidat = '/scratch/{case}/run/f.nc'
/
"""
DATM = "&shr_strdata_nml\n presaero = .true.\n/\n"


def _docs(root, case, **edit):
    cd = root / case
    (cd / "CaseDocs").mkdir(parents=True)
    text = LND.format(case=case)
    for k, v in edit.items():
        text = re.sub(rf"^ {k} = .*$", f" {k} = {v}", text, flags=re.M)
    (cd / "CaseDocs" / "lnd_in").write_text(text)
    (cd / "CaseDocs" / "datm_in").write_text(DATM)
    return cd


class TestTheSettingsCheck:

    def test_a_run_that_differs_only_by_its_change(self, tmp_path):
        c = _docs(tmp_path, "ctl")
        r = _docs(tmp_path, "ctl_nofire", use_nofire=".true.")
        out = CH.settings_differences(r, c, {"lnd_in": {"use_nofire": True}},
                                      "ctl_nofire", "ctl")
        assert out["pass"], out["reasons"]
        assert out["files"]["lnd_in"]["differences"] == {
            "use_nofire": {"control": [False], "run": [True]}}

    def test_a_difference_no_change_explains_fails(self, tmp_path):
        c = _docs(tmp_path, "ctl")
        r = _docs(tmp_path, "ctl_nofire", use_nofire=".true.", co2_ppmv="400.0")
        out = CH.settings_differences(r, c, {"lnd_in": {"use_nofire": True}},
                                      "ctl_nofire", "ctl")
        assert not out["pass"]
        assert out["reasons"] == ["lnd_in: co2_ppmv differs from the control ([367.0] -> "
                                  "[400.0]), and no change of this run alters it"]

    def test_a_change_that_did_not_take(self, tmp_path):
        c = _docs(tmp_path, "ctl")
        r = _docs(tmp_path, "ctl_co2", co2_ppmv="500.0")
        out = CH.settings_differences(r, c, {"lnd_in": {"co2_ppmv": "560"}}, "ctl_co2", "ctl")
        assert out["reasons"] == ["lnd_in: co2_ppmv is [500.0], the change sets 560"]

    def test_a_change_to_the_controls_own_value_is_not_an_error(self, tmp_path):
        c = _docs(tmp_path, "ctl")
        r = _docs(tmp_path, "ctl_same")
        out = CH.settings_differences(r, c, {"lnd_in": {"suplnitro": "NONE"}},
                                      "ctl_same", "ctl")
        assert out["pass"] and out["files"]["lnd_in"]["no_op"] == ["suplnitro"]

    def test_a_missing_namelist_is_not_a_pass(self, tmp_path):
        c = _docs(tmp_path, "ctl")
        r = _docs(tmp_path, "ctl_nofire", use_nofire=".true.")
        (c / "CaseDocs" / "datm_in").unlink()
        out = CH.settings_differences(r, c, {"lnd_in": {"use_nofire": True}},
                                      "ctl_nofire", "ctl")
        assert not out["pass"] and out["reasons"][0].startswith("datm_in: cannot compare")


# ── the check of an experiment ───────────────────────────────────────────────
def _whats(items):
    return [i["what"] for i in items]


class TestTheCheck:

    def test_a_fire_off_twin_of_a_reused_control(self):
        out = CH.check(_exp({"name": "nofire", "changes": [{"id": "fire_off"}]}), CONTROL)
        assert out["ok"] and out["wont_build"] == [] and out["unusual"] == []
        facts = _whats(out["facts"])
        assert facts[0].startswith("meaning: the difference from the control is what the "
                                   "model's own fire does")
        assert "Not: the effect of an observed fire" in facts[0]
        # the model's own fire only: these facts reach the Analyzer (2026-10-27)
        assert facts[1] == ("the control's own fire burns 115-133 km2 a year (spread over "
                            "2903-5296 cells); it is the model's fire, and the fires "
                            "observed in those years are not in the run")
        assert facts[2].startswith("reused; reproduced bit for bit (2015-2016), source "
                                   "fingerprint 503a94c5d1a0")
        assert out["cost"] == {"runs_executed": 1, "model_years": 2,
                               "hours_per_model_year": 0.79, "node_hours": 1.6,
                               "note": "the control is reused: it spends nothing"}

    def test_a_question_needing_a_tool_the_server_lacks(self):
        out = CH.check(_exp({"name": "moved", "changes": [{"id": "burn_footprint"}]}), CONTROL)
        assert not out["ok"]
        w = out["wont_build"][0]
        assert w["where"] == "run moved, change burn_footprint"
        assert w["requires"][0].startswith("a burn tool")

    @pytest.mark.parametrize("exp,problem", [
        ({"control": {"name": "naches", "reuse": True}, "years": [2015], "runs": "x"},
         "runs must be a list"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]},
              {"name": "a", "changes": [{"id": "fire_off"}]}), "two runs have this name"),
        (_exp({"name": "A b", "changes": [{"id": "fire_off"}]}), "a run's name is"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]}, years=(2015, 2017)),
         "are not consecutive"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]}, years=range(2015, 2025)),
         "outside the forcing the control reads (2013-2023)"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]}, years=range(2015, 2023)),
         "outside the reused control's output (2015-2021)"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]}, years=(2016, 2017)),
         "the run's first year is one of those, not 2016"),
        (_exp({"name": "a", "changes": [{"id": "ndep_year", "params": {"year": 1800}}]}),
         "1800 is outside the control's deposition dataset (1849-2106)"),
        (_exp({"name": "a", "changes": [{"id": "fire_off"}]}, years=[]),
         "years must be a non-empty list"),
    ])
    def test_what_will_not_build(self, exp, problem):
        exp["years"] = list(exp["years"])
        out = CH.check(exp, CONTROL)
        assert not out["ok"] and any(problem in w for w in _whats(out["wont_build"])), \
            out["wont_build"]

    def test_co2_needs_a_constant_co2_control(self):
        facts = dict(FACTS, settings=dict(FACTS["settings"], ELM_CO2_TYPE="diagnostic"))
        out = CH.check(_exp({"name": "co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560}}]}), dict(CONTROL, facts=facts))
        assert _whats(out["wont_build"]) == [
            "the control's ELM_CO2_TYPE is diagnostic: a CO2 level can be held constant "
            "only when the control's CO2 is"]

    def test_what_is_unusual(self):
        out = CH.check(_exp({"name": "same", "changes": [
            {"id": "nutrient_supplement", "params": {"nitrogen": "NONE", "phosphorus": "ALL"}}]},
            {"name": "repeat", "changes": []}), CONTROL)
        assert out["ok"]
        u = _whats(out["unusual"])
        assert "the control already has suplnitro = NONE: this change changes nothing" in u
        assert any(x.startswith("this run has no changes: it repeats the control") for x in u)

    def test_a_control_without_facts_or_reproduction(self):
        out = CH.check(_exp({"name": "nofire", "changes": [{"id": "fire_off"}]}),
                       {"name": "x"})
        u = _whats(out["unusual"])
        assert any("could not be confirmed" in x for x in u)
        assert any(x.startswith("reused without a record that it reproduces") for x in u)
        assert out["cost"]["hours_per_model_year"] == CH.DEFAULT_HOURS_PER_YEAR

    def test_the_control_alone(self):
        out = CH.check(_exp(), CONTROL)
        assert out["ok"] and out["runs"] == [] and out["cost"]["node_hours"] == 0.0
        assert _whats(out["facts"])[0] == ("no run differs from the control: the experiment "
                                           "reads the reused control alone and spends nothing")

    def test_an_executed_control_is_counted(self):
        out = CH.check(_exp({"name": "nofire", "changes": [{"id": "fire_off"}]}, reuse=False),
                       CONTROL)
        assert out["cost"]["runs_executed"] == 2 and out["cost"]["node_hours"] == 3.2


# ── what a run does not cover, in short ──────────────────────────────────────
# The short limits as they were written for the design page (2026-11-13),
# copied here word for word, so that a change to one in the server's list
# fails a test and is made on purpose.
CUTS = ("The run cuts rain and snow only. Air temperature, humidity and sunshine "
        "stay as recorded, so it is not a full drought.")
ADDS = ("The run adds rain and snow only. Air temperature, humidity and sunshine "
        "stay as recorded, so it is not a full wet spell.")
WARMS = ("The run warms the air only. Snow does not turn into rain, and the air's "
         "moisture and sunshine stay as recorded. Warmer air is then also drier air "
         "to the model, and its leaves and its fire respond to that.")
COOLS = ("The run cools the air only. Rain does not turn into snow, and the air's "
         "moisture and sunshine stay as recorded. The cooled air can hold less "
         "moisture than was recorded, so the model forms dew and frost that the "
         "recorded weather would not.")
BOTH = ("The run changes air temperature, rain and snow together. Rain and snow "
        "keep their recorded split, and the air's moisture and sunshine stay as "
        "recorded, so it is not a full change of climate.")
REPEAT = ("Each repeated year is a year of the record, with its storms on the same "
          "dates. The run is not a forecast or a climate scenario, and it holds no "
          "drought worse than the record's.")
FIRE = ("The run removes the model's own fire, which is unrelated in place and year "
        "to the fires that happened. It does not show the effect of an observed fire.")
CO2 = ("The run changes CO2 for the plants only, through photosynthesis and stomata. "
       "CO2 does not change the weather here, so it is not a climate-change "
       "experiment.")
NDEP_1777 = ("The run holds nitrogen deposition at its level of 1777, as the control "
             "run's own dataset gives it. No other input follows that year.")
NUTRIENT = ("The run supplies whatever plants demand of the nutrient, without limit. "
            "It is not a fertiliser experiment with real amounts.")
WEATHER = (CUTS, ADDS, WARMS, COOLS, BOTH)


def _ws(**params):
    return {"id": "weather_scale", "params": params}


class TestNotCovered:

    @pytest.mark.parametrize("params,text", [
        ({"precip_factor": 0.6, "temperature_K": 0.0}, CUTS),
        ({"precip_factor": 1.4, "temperature_K": 0.0}, ADDS),
        ({"temperature_K": 1.5, "precip_factor": 1.0}, WARMS),
        ({"temperature_K": -2.5, "precip_factor": 1.0}, COOLS),
        ({"temperature_K": 1.5, "precip_factor": 0.6}, BOTH),
        ({"temperature_K": -2.5, "precip_factor": 1.4}, BOTH),
    ])
    def test_each_weather_case_has_its_one_text(self, params, text):
        assert CH.not_covered(_ws(**params)) == [text]

    def test_the_weather_as_recorded_has_none(self):
        assert CH.not_covered(_ws()) == []
        assert CH.not_covered(_ws(temperature_K=0.0, precip_factor=1.0, months=[4, 5])) == []

    @pytest.mark.parametrize("params,text", [
        ({"precip_factor": 0.6}, CUTS), ({"precip_factor": 1.4}, ADDS),
        ({"temperature_K": 1.5}, WARMS), ({"temperature_K": -2.5}, COOLS)])
    def test_a_setting_left_out_takes_its_default(self, params, text):
        assert CH.not_covered(_ws(months=[6, 7], **params)) == [text]

    def test_one_text_at_most_over_a_grid_of_settings(self):
        """The expected text is worked out here from the signs of the two
        settings, never from the list's own conditions: a setting at its
        bound is neither above nor below it."""
        for t in (-3.0, -0.5, 0.0, 0.5, 3.0):
            for f in (0.0, 0.5, 1.0, 1.5, 4.0):
                if t == 0 and f == 1:
                    want = []
                elif t == 0:
                    want = [CUTS if f < 1 else ADDS]
                elif f == 1:
                    want = [WARMS if t > 0 else COOLS]
                else:
                    want = [BOTH]
                assert CH.not_covered(_ws(temperature_K=t, precip_factor=f)) == want, (t, f)

    def test_a_setting_within_rounding_of_its_bound_is_the_bound(self):
        """As the meaning's words treat it (_text): a factor of 1 plus a
        rounding error multiplies nothing."""
        assert CH.not_covered(_ws(temperature_K=1.5, precip_factor=1.0 + 1e-13)) == [WARMS]
        assert CH.not_covered(_ws(temperature_K=-1e-13, precip_factor=0.6)) == [CUTS]
        assert CH.not_covered(_ws(temperature_K=1e-13, precip_factor=1.0 - 1e-13)) == []

    @pytest.mark.parametrize("change,texts", [
        ({"id": "fire_off"}, [FIRE]),
        ({"id": "fire_off", "params": {}}, [FIRE]),
        ({"id": "co2_ppmv", "params": {"ppmv": 432}}, [CO2]),
        ({"id": "weather_repeat_year", "params": {"map": {"2031": 2029}}}, [REPEAT]),
        ({"id": "ndep_year", "params": {"year": 1777}}, [NDEP_1777]),
        ({"id": "ndep_year", "params": {"year": 1777.0}}, [NDEP_1777]),
        ({"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "NONE"}},
         [NUTRIENT]),
        ({"id": "nutrient_supplement", "params": {"nitrogen": "NONE", "phosphorus": "ALL"}},
         [NUTRIENT]),
        ({"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "ALL"}},
         [NUTRIENT]),
        ({"id": "nutrient_supplement", "params": {"nitrogen": "NONE", "phosphorus": "NONE"}},
         []),
        ({"id": "burn_footprint"}, []),
        ({"id": "parameter_value"}, []),
        ({"id": "no_such_change", "params": {"year": 1777}}, []),
    ])
    def test_the_other_changes(self, change, texts):
        assert CH.not_covered(change) == texts

    @pytest.mark.parametrize("change,texts", [
        (None, []), ("fire_off", []), ([], []), ({"params": {}}, []),
        ({"id": ["fire_off"]}, []),
        ({"id": "fire_off", "params": "nothing"}, [FIRE]),
        # a year it cannot fill: the text is left out, never shown with {year}
        ({"id": "ndep_year"}, []),
        ({"id": "ndep_year", "params": {}}, []),
        # a setting of another kind is neither above nor below a number
        ({"id": "weather_scale", "params": {"precip_factor": "0.6"}}, []),
        # a choice is matched as written
        ({"id": "nutrient_supplement", "params": {"nitrogen": "all", "phosphorus": "NONE"}},
         []),
        # a setting not given, with no default, holds no condition on it
        ({"id": "nutrient_supplement", "params": {}}, []),
    ])
    def test_a_change_of_another_form_never_raises(self, change, texts):
        assert CH.not_covered(change) == texts

    def test_every_condition_must_hold_and_the_order_is_the_lists(self, monkeypatch):
        """On a made-up change: an entry holds when all its conditions do,
        one with none always holds, the texts come in the list's order, and
        a text two entries give comes once."""
        made_up = {"id": "made_up",
                   "params": {"a": {"kind": "number", "default": 2.5},
                              "b": {"kind": "choice", "choices": ["X", "Y"]}},
                   "not_covered": [
                       {"when": [{"param": "a", "above": 1.0}, {"param": "b", "is": "X"}],
                        "say": "First {b}."},
                       {"when": [], "say": "Second."},
                       {"when": [{"param": "a", "below": 1.0}], "say": "Low."},
                       {"when": [{"param": "b", "is_not": "Y"}], "say": "First X."},
                       {"say": "Third."},
                       {"when": [{"param": "a", "above": 1.0, "below": 9.0}],
                        "say": "Two tests in one condition."},
                       {"when": [{"param": "a", "near": 2.5}], "say": "An unknown test."},
                       {"when": [{"param": "c", "is_not": "X"}], "say": "No such setting."},
                       {"when": [{"param": "a", "is": 1}], "say": "A true is not one."}]}
        monkeypatch.setitem(CH._BY_ID, "made_up", made_up)
        assert CH.not_covered({"id": "made_up", "params": {"b": "X"}}) == [
            "First X.", "Second.", "Third."]
        assert CH.not_covered({"id": "made_up", "params": {"a": 0.5, "b": "Y"}}) == [
            "Second.", "Low.", "Third."]
        assert CH.not_covered({"id": "made_up", "params": {"a": 0.5, "b": "X"}}) == [
            "Second.", "Low.", "First X.", "Third."]
        # true is not a number here, so it is neither 1 nor above 1
        assert CH.not_covered({"id": "made_up", "params": {"a": True, "b": "Y"}}) == [
            "Second.", "Third."]

    def test_resolve_carries_them(self):
        assert CH.resolve(_ws(precip_factor=0.6))["not_covered"] == [CUTS]
        assert CH.resolve(_ws(temperature_K=-2.5, precip_factor=1.4))["not_covered"] == [BOTH]
        assert CH.resolve({"id": "ndep_year", "params": {"year": 1777.0}})["not_covered"] == [
            NDEP_1777]
        assert CH.resolve({"id": "fire_off"})["not_covered"] == [FIRE]
        # the key is there when no limit holds
        assert CH.resolve({"id": "nutrient_supplement", "params": {
            "nitrogen": "NONE", "phosphorus": "NONE"}})["not_covered"] == []

    def test_every_run_the_check_resolves_carries_them(self):
        out = CH.check(_exp({"name": "fewer_fires", "changes": [
                                {"id": "fire_off"}, {"id": "co2_ppmv", "params": {"ppmv": 432}}]},
                            {"name": "warmer", "changes": [_ws(temperature_K=1.5)]}), CONTROL)
        assert [[c["not_covered"] for c in r["changes"]] for r in out["runs"]] == [
            [[FIRE], [CO2]], [[WARMS]]]

    def test_the_published_list_does_not_carry_them(self):
        """The list goes to the agents as a tool result with a size limit it
        already nearly fills (test_it_fits_a_tool_result); the design page
        reads the short limits from the resolved runs instead."""
        d = CH.declare()
        assert all("not_covered" not in c for c in d["changes"])
        assert all("not_covered" not in CH._published(c) for c in CH._CHANGES)
        text = json.dumps(d)
        for t in WEATHER + (REPEAT, FIRE, CO2, NUTRIENT, NDEP_1777[:40]):
            assert t not in text

    def test_which_changes_have_them(self):
        assert {c["id"] for c in CH._CHANGES if c.get("not_covered")} == {
            "fire_off", "co2_ppmv", "ndep_year", "nutrient_supplement",
            "weather_repeat_year", "weather_scale"}
        assert all(c["available"] for c in CH._CHANGES if c.get("not_covered"))

    def test_the_lists_own_form(self):
        """Each condition names one setting of its change and makes one test
        of four; each text fills only settings of its change."""
        for c in CH._CHANGES:
            for e in c.get("not_covered") or []:
                assert set(e) <= {"when", "say"}, (c["id"], e)
                for w in e.get("when") or []:
                    assert w["param"] in c["params"], (c["id"], w)
                    assert len(w) == 2 and (set(w) - {"param"}) <= {
                        "is", "is_not", "above", "below"}, (c["id"], w)
                for name in re.findall(r"\{(\w*)\}", e["say"]):
                    assert name in c["params"], (c["id"], name)

    def test_every_text_is_short_plain_and_whole(self):
        said = [e["say"] for c in CH._CHANGES for e in c.get("not_covered") or []]
        said += [NDEP_1777]
        assert len(said) >= 11
        for t in said:
            assert t.endswith("."), t
            assert "\u2014" not in t and "\u2013" not in t, t
            assert ":" not in t, t
            assert len(t) <= 300, t
            assert t == " ".join(t.split()), t

    def test_the_long_texts_stay_beside_them(self):
        """The short limits are a key of their own: a resolved change keeps
        its long text, for the details page."""
        c = CH.resolve(_ws(precip_factor=0.6))
        assert set(c) == {"id", "params", "summary", "alters", "meaning", "not_meaning",
                          "checks", "cost", "not_covered"}
        assert c["not_meaning"].startswith("a consistent change of climate: the data "
                                           "atmosphere splits rain from snow")
