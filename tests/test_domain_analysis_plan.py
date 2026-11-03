"""What a whole-domain plan's analysis may ask of the numbers, and the check
that holds it to that before any job.

    pytest tests/test_domain_analysis_plan.py -v

Two defects a real study met (2026-10-27, the warm dry summer over the
Naches watershed): the plan named H2OSOI, a field with soil levels, and the
reduction refused it for every field of every run, so job B would have made
no number at all; and it named "elevation bands" for a change applied only
inside the watershed, which the standard bands would have averaged over the
whole box, changed and unchanged cells together. Each is now refused before
any job, and bands can be made inside a registered region.
"""
import json

import numpy as np

import domain_analysis as A
import domain_changes as CH
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import regions as R
from agents import design_review as DR
from agents.analysis.domain_report import region_keys, region_label, region_matches

from test_domain_analysis import _grid, _polygon, _veg
from test_domain_experiment import lab  # noqa: F401
from test_domain_manager import mgr, srv  # noqa: F401


def _case(tmp_path, variables=("GPP", "TSA", "CELLID")):
    c = cf.write_case(tmp_path / "ctl")
    fx.write_monthly(c["run_dir"], [2015, 2016], case="synth", variables=variables)
    return c


class TestBandsWithinARegion:

    def test_each_band_is_the_region_and_the_band_and_nothing_else(self, tmp_path):
        c = _case(tmp_path)
        reg = {"box": {"polygon": str(_polygon(tmp_path, c))}}
        out = A.masks(_grid(c), c["surface"],
                      [{"name": A.BANDS, "within": "box"}], reg)
        ms = out["masks"]
        assert out["problems"] == []
        veg, topo = _veg(c)
        box = np.zeros_like(veg)
        box[:2, :] = True
        made = 0
        for b, m in R.elevation_bands(topo, A.BAND_EDGES).items():
            name = A.band_within("box", b)
            if (veg & box & m).any():
                assert np.array_equal(ms[name], veg & box & m)
                made += 1
            else:
                assert name not in ms.names()
            assert b not in ms.names()          # no band over the whole domain
        assert made >= 1

    def test_what_cannot_be_made_within_is_named(self, tmp_path):
        c = _case(tmp_path)
        reg = {"box": {"polygon": str(_polygon(tmp_path, c))}}
        out = A.masks(_grid(c), c["surface"],
                      [{"name": A.BANDS, "within": "nowhere"},
                       {"name": "box", "within": "box"}], reg)
        assert len(out["problems"]) == 2
        assert "'nowhere'" in out["problems"][0] and "within it" in out["problems"][0]
        assert "within applies to elevation bands only" in out["problems"][1]

    def test_a_region_named_twice_is_made_once(self, tmp_path):
        c = _case(tmp_path)
        reg = {"box": {"polygon": str(_polygon(tmp_path, c))}}
        out = A.masks(_grid(c), c["surface"], [A.BANDS, A.BANDS, "box", "box",
                                               {"name": A.BANDS, "within": "box"},
                                               {"name": A.BANDS, "within": "box"}], reg)
        assert out["problems"] == []

    def test_the_whole_table_carries_them(self, tmp_path):
        c = _case(tmp_path)
        reg = {"box": {"polygon": str(_polygon(tmp_path, c))}}
        t = A.experiment({"case_dir": c["case_dir"], "regions": reg},
                         {"control": c["run_dir"]},
                         {"regions": [{"name": "box"}, {"name": A.BANDS, "within": "box"}],
                          "variables": ["GPP"]}, [2015, 2016])
        assert t["problems"] == []
        within = [k for k in t["regions"] if k.startswith("box, vegetated, ")]
        assert within and set(within) <= set(t["runs"]["control"]["values"])


class TestAFieldWithLevels:

    def test_it_is_a_problem_of_its_own_and_the_others_are_made(self, tmp_path):
        c = _case(tmp_path, variables=("GPP", "TSA", "CELLID", "SOILLIQ"))
        ms = A.masks(_grid(c), c["surface"], [A.VEGETATED])["masks"]
        out = A.run_values(c["run_dir"], ms, ["GPP", "SOILLIQ", "TSA"], [2015, 2016])
        assert set(out["values"][A.VEGETATED]) == {"GPP", "TSA"}
        assert len(out["problems"]) == 1
        p = out["problems"][0]
        assert p.startswith("SOILLIQ has a value at each of its levels (levgrnd)")
        assert "a regional number is made from one value per cell" in p
        assert "is one value per cell" not in p and "SOILWATER_10CM" in p


def _check(tmp_path, analysis, region="box", variables=("GPP", "TSA", "CELLID", "SOILLIQ")):
    c = _case(tmp_path, variables=variables)
    params = {"temperature_K": 2.0}
    if region:
        params["region"] = region
    exp = {"control": {"name": "naches", "reuse": True}, "years": [2015, 2016],
           "runs": [{"name": "warm", "changes": [{"id": "weather_scale", "params": params}]}],
           "analysis": analysis}
    facts = {"start_year": 2015, "forcing_years": [2006, 2300], "output_years": [2006, 2300],
             "streams": {"h1": {"fields": ["RAIN", "SNOW", "TBOT"],
                                "years": list(range(2006, 2301))}}}
    return CH.check(exp, {"name": "naches", "facts": facts, "run_dir": str(c["run_dir"]),
                          "regions": {"box": {"polygon": "/p.geojson"},
                                      "other": {"polygon": "/q.geojson"}}})


def _analysis_wont(out):
    return [w for w in out["wont_build"] if w["where"].startswith("analysis")]


class TestTheCheckBeforeAnyJob:

    def test_what_the_numbers_can_give_passes(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "box"},
                                            {"name": A.BANDS, "within": "box"}],
                                "variables": ["GPP", "TSA"]})
        assert out["ok"], out["wont_build"]

    def test_bands_over_the_domain_for_a_change_in_one_region_will_not_build(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "box"}, {"name": A.BANDS}],
                                "variables": ["GPP"]})
        w = _analysis_wont(out)
        assert not out["ok"] and len(w) == 1
        assert w[0]["where"] == "analysis, region elevation bands"
        assert "applies only in box" in w[0]["what"]
        assert "average changed and unchanged cells" in w[0]["what"]
        assert '{"name": "elevation bands", "within": "box"}' in w[0]["what"]

    def test_another_region_or_the_whole_domain_will_not_build_either(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "other"}, {"name": A.VEGETATED},
                                            {"name": A.BANDS, "within": "other"}],
                                "variables": ["GPP"]})
        assert [w["where"] for w in _analysis_wont(out)] == [
            "analysis, region other", "analysis, region vegetated cells",
            "analysis, region elevation bands within other"]

    def test_a_change_over_the_whole_domain_may_report_any_region(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "other"}, {"name": A.VEGETATED},
                                            {"name": A.BANDS}],
                                "variables": ["GPP"]}, region=None)
        assert out["ok"], out["wont_build"]

    def test_a_field_with_levels_or_none_will_not_build(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "box"}],
                                "variables": ["GPP", "SOILLIQ", "NOPE"]})
        w = {x["where"]: x["what"] for x in _analysis_wont(out)}
        assert set(w) == {"analysis, variable SOILLIQ", "analysis, variable NOPE"}
        assert w["analysis, variable SOILLIQ"].startswith(
            "SOILLIQ has a value at each of its levels (levgrnd)")
        assert "in none of the control's history streams" in w["analysis, variable NOPE"]

    def test_a_region_that_is_not_one_will_not_build(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "nowhere"},
                                            {"name": A.BANDS, "within": "nowhere"},
                                            {"name": "box", "within": "box"}],
                                "variables": ["GPP"]}, region=None)
        w = [x["what"] for x in _analysis_wont(out)]
        assert len(w) == 3
        assert "neither a standard region" in w[0]
        assert "'nowhere' is not registered" in w[1]
        assert "within applies to elevation bands only" in w[2]

    def test_an_experiment_without_an_analysis_is_checked_as_before(self, tmp_path):
        out = _check(tmp_path, {}, region="box")
        assert out["ok"], out["wont_build"]

    def test_no_regions_named_is_checked_as_the_default_regions(self, tmp_path):
        """The analysis reports vegetated cells and elevation bands when the
        plan names no regions, so a run changed in one region is refused for
        them as it would be had the plan named them (2026-11-03)."""
        assert A.plan_regions({"variables": ["GPP"]}) == [A.VEGETATED, A.BANDS]
        assert A.plan_regions({"regions": ["box"]}) == ["box"]
        out = _check(tmp_path, {"variables": ["GPP"]})
        assert not out["ok"]
        assert [w["where"] for w in _analysis_wont(out)] == [
            "analysis, region vegetated cells", "analysis, region elevation bands"]
        out = _check(tmp_path / "whole", {"variables": ["GPP"]}, region=None)
        assert out["ok"], out["wont_build"]

    def test_burned_area_is_refused_with_the_field_it_is_made_from(self, tmp_path):
        out = _check(tmp_path, {"regions": [{"name": "box"}],
                                "variables": ["GPP", A.BURNED_AREA]})
        w = {x["where"]: x["what"] for x in _analysis_wont(out)}
        assert set(w) == {"analysis, variable burned_area_km2"}
        assert w["analysis, variable burned_area_km2"] == (
            "burned_area_km2 is made from FAREA_BURNED: name FAREA_BURNED and the "
            "burned area in km2 comes with it")

    def test_an_unreadable_history_keeps_the_region_refusals(self, tmp_path):
        """A control whose run directory is gone: the region and confinement
        refusals stand, and the unreadable history is a refusal of its own,
        not an unusual line beside a pass (2026-11-03)."""
        exp = {"control": {"name": "naches", "reuse": True}, "years": [2015, 2016],
               "runs": [{"name": "warm", "changes": [{"id": "weather_scale", "params": {
                   "temperature_K": 2.0, "region": "box"}}]}],
               "analysis": {"regions": [{"name": "nowhere"}, {"name": A.VEGETATED}],
                            "variables": ["GPP"]}}
        facts = {"start_year": 2015, "forcing_years": [2006, 2300],
                 "output_years": [2006, 2300],
                 "streams": {"h1": {"fields": ["RAIN", "SNOW", "TBOT"],
                                    "years": list(range(2006, 2301))}}}
        out = CH.check(exp, {"name": "naches", "facts": facts,
                             "run_dir": str(tmp_path / "purged"),
                             "regions": {"box": {"polygon": "/p.geojson"}}})
        assert not out["ok"]
        w = _analysis_wont(out)
        assert [x["where"] for x in w] == [
            "analysis, region nowhere", "analysis, region nowhere",
            "analysis, region vegetated cells", "analysis, variables"]
        assert w[-1]["what"].startswith("the control's history cannot be read, so the "
                                        "fields cannot be checked: GriddedReadError")
        assert not [u for u in out["unusual"] if u["where"] == "analysis"]

    def test_the_rules_are_published_for_the_planner(self):
        rules = " ".join(CH.declare()["rules"])
        assert "single-level history fields" in rules and "SOILWATER_10CM" in rules
        assert '"within"' in rules and "average changed and unchanged cells" in rules
        assert "with no regions named the analysis reports vegetated cells" in rules
        assert "burned_area_km2 is not a history field: name FAREA_BURNED" in rules


class TestTheReportAndThePageReadWithin:

    HAVE = ["vegetated cells", "below 1000 m", "1000-1500 m", "watershed, vegetated",
            "watershed, vegetated, below 1000 m", "watershed, vegetated, 1000-1500 m"]

    def test_bands_within_match_only_their_region(self):
        assert region_matches({"name": "elevation bands", "within": "watershed"},
                              self.HAVE) == ["watershed, vegetated, below 1000 m",
                                             "watershed, vegetated, 1000-1500 m"]
        assert region_matches("elevation bands", self.HAVE) == ["below 1000 m",
                                                                "1000-1500 m"]
        assert region_matches({"name": "elevation bands"}, self.HAVE) == [
            "below 1000 m", "1000-1500 m"]
        assert region_matches({"name": "watershed"}, self.HAVE) == ["watershed, vegetated"]
        assert region_matches({"name": "elevation bands", "within": "burned"},
                              self.HAVE) == []

    def test_the_report_keeps_the_plans_order(self):
        t = {"regions": {k: {} for k in self.HAVE}}
        keys = region_keys(t, {"regions": [{"name": "watershed"},
                                           {"name": "elevation bands",
                                            "within": "watershed"}]})
        assert keys == ["watershed, vegetated", "watershed, vegetated, below 1000 m",
                        "watershed, vegetated, 1000-1500 m"]

    def test_a_figure_does_not_repeat_vegetated_on_every_label(self):
        from agents.analysis.domain_report import _label
        assert _label("watershed, vegetated, 1000-1500 m") == "watershed, 1000-1500 m"
        assert _label("watershed, vegetated") == "watershed"
        assert _label("vegetated cells") == "vegetated cells"
        assert _label("1000-1500 m") == "1000-1500 m"

    def test_the_label(self):
        assert region_label({"name": "elevation bands", "within": "watershed"}) == \
            "elevation bands within watershed"
        assert region_label("watershed") == "watershed"

    def test_the_design_page_says_within_and_one_year_plainly(self):
        de = {"control": {"name": "naches", "reuse": True}, "years": [2015],
              "runs": [{"name": "warm", "changes": [{"id": "weather_scale",
                                                     "params": {"temperature_K": 2}}]}],
              "analysis": {"regions": [{"name": "watershed", "why": "the change is there"},
                                       {"name": "elevation bands", "within": "watershed",
                                        "why": "by elevation"}, "vegetated cells"]}}
        lines = "\n".join(DR._whole_domain_lines({}, {"domain_experiment": de}))
        assert "elevation bands within watershed (by elevation)" in lines
        assert "; vegetated cells" in lines
        assert "1 model year(s), 2015, for 1 run(s)" in lines


class TestARunNameAnEarlierStudyUsed:

    def test_it_is_refused_before_any_job_in_plain_words(self, lab):
        from test_domain_experiment import NOFIRE
        exp = {"control": {"name": "synth", "reuse": True}, "runs": NOFIRE,
               "years": [2015, 2016]}
        assert json.loads(lab["srv"].check_domain_experiment(exp))["ok"]
        earlier = lab["control_dir"].parent / "src_case_nofire"
        earlier.mkdir()
        out = json.loads(lab["srv"].check_domain_experiment(exp))
        assert out["ok"] is False
        w = [x for x in out["wont_build"] if x["where"] == "run nofire"]
        assert len(w) == 1 and "a case named src_case_nofire exists already" in w[0]["what"]
        assert "Give the run another name (for example nofire_2)" in w[0]["what"]

    def test_a_control_run_again_is_not_asked(self, lab):
        exp = {"control": {"name": "synth", "reuse": False}, "runs": [], "years": [2015]}
        assert CH.case_collisions(exp, {"case_dir": str(lab["control_dir"])}) == []


class TestTheShareOfARegionAChangeReached:

    def test_it_is_measured_for_every_region_of_the_table(self, tmp_path):
        c = _case(tmp_path)
        reg = {"box": {"polygon": str(_polygon(tmp_path, c))}}
        t = A.experiment({"case_dir": c["case_dir"], "regions": reg},
                         {"control": c["run_dir"], "warm": c["run_dir"]},
                         {"regions": [{"name": "box"}, {"name": A.BANDS, "within": "box"}],
                          "variables": ["GPP"]}, [2015, 2016], confined={"warm": "box"})
        cs = t["changed_share"]["warm"]
        assert cs["region"] == "box"
        veg, _ = _veg(c)
        box = np.zeros_like(veg)
        box[:2, :] = True
        w = np.asarray(_grid(c).weight_km2)
        want = round(float(w[veg & box].sum() / w[veg].sum()), 4)
        assert 0 < want < 1 and cs["share"][A.VEGETATED] == want
        assert cs["share"]["box, vegetated"] == 1.0
        assert all(v == 1.0 for k, v in cs["share"].items() if k.startswith("box, vegetated, "))

    def test_a_region_it_cannot_make_is_a_problem(self, tmp_path):
        c = _case(tmp_path)
        t = A.experiment({"case_dir": c["case_dir"], "regions": {}},
                         {"control": c["run_dir"]}, {"variables": ["GPP"]}, [2015],
                         confined={"warm": "nowhere"})
        assert t["changed_share"] == {}
        assert any("its change's region 'nowhere' is not registered" in p
                   for p in t["problems"])

    def test_no_confined_run_no_share(self, tmp_path):
        c = _case(tmp_path)
        t = A.experiment({"case_dir": c["case_dir"], "regions": {}},
                         {"control": c["run_dir"]}, {"variables": ["GPP"]}, [2015])
        assert "changed_share" not in t


class TestTheManagerNamesEachRunsRegion:

    def test_from_the_experiment_as_checked(self, mgr):
        mgr.input_dir.mkdir(parents=True, exist_ok=True)
        (mgr.input_dir / mgr.DOMAIN_EXPERIMENT).write_text(json.dumps({"runs": [
            {"name": "warm", "changes": [{"id": "weather_scale", "params": {
                "temperature_K": 2.0, "region": "watershed"}}]},
            {"name": "hot", "changes": [{"id": "weather_scale",
                                         "params": {"temperature_K": 4.0}}]},
            {"name": "nofire", "changes": [{"id": "fire_off"}]},
            {"name": "gone", "changes": [{"id": "weather_scale", "params": {
                "temperature_K": 2.0, "region": "watershed"}}]}]}))
        got = mgr._confined({"control": "/c", "warm": "/w", "hot": "/h", "nofire": "/n"})
        assert got == {"warm": "watershed"}


class TestTheAnalyzerIsBoundByTheShare:

    def _caveats(self, share):
        from agents.analysis.step0_context import _domain_caveats
        out = []
        _domain_caveats({"regional": {"changed_share": {"warm": {
            "region": "watershed", "share": share}}}}, {},
            lambda cid, sev, st, ap, src: out.append((cid, sev, st)))
        return [c for c in out if c[0].startswith("changed_share_")]

    def test_a_region_partly_reached_is_a_blocking_caveat(self):
        got = self._caveats({"vegetated cells": 0.3012, "watershed, vegetated": 1.0,
                             "below 1000 m": 0.0})
        assert [(c[0], c[1]) for c in got] == [("changed_share_1", "blocking"),
                                               ("changed_share_2", "blocking")]
        assert got[0][2] == ("run warm's change applies only in watershed and does not "
                             "reach below 1000 m: its difference from the control there is "
                             "no response to the change")
        assert got[1][2] == ("run warm's change applies only in watershed, which covers 30% "
                             "of the area of vegetated cells: its difference from the control "
                             "there averages changed and unchanged cells, so it is not the "
                             "response of vegetated cells")

    def test_a_region_wholly_reached_is_not(self):
        assert self._caveats({"watershed, vegetated": 1.0,
                              "watershed, vegetated, 1000-1500 m": 1.0}) == []


class TestARunSummaryWithAReusedControl:
    """Demo 2's job B, 2026-10-27: the control read as it is and one run that
    passed every check were summarised as "ELM COMPLETE: 1/2", the control
    recorded as failed."""

    ROWS = [{"case_name": "ctl", "status": "reused"},
            {"case_name": "ctl_warm", "status": "completed"},
            {"case_name": "ctl_cold", "status": "failed"}]

    def test_a_reused_control_is_a_success(self, mgr):
        assert mgr._outcome_map(self.ROWS) == {"ctl": True, "ctl_warm": True,
                                               "ctl_cold": False}

    def test_the_summary_says_reused_and_counts_it(self, mgr):
        from datetime import datetime
        t = datetime(2026, 10, 27)
        s = mgr._create_run_summary({}, self.ROWS[:2], self.ROWS[:2], t, t)
        assert (s["experiments_total"], s["experiments_success"],
                s["experiments_failed"]) == (2, 2, 0)
        assert [e["status"] for e in s["experiments"]] == ["reused", "completed"]


class TestTheModelsOutputOnly:
    """The Analyzer reads the model's output alone (decided 2026-10-27): Demo
    A's answer argued from MODIS burned area ("340 km2, so the carbon drop is
    not a fire response") and stated a cause no field showed."""

    ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

    def test_the_servers_fire_fact_carries_no_observation(self):
        exp = {"control": {"name": "naches", "reuse": True}, "years": [2015, 2016],
               "runs": [{"name": "nofire", "changes": [{"id": "fire_off"}]}]}
        facts = {"start_year": 2015, "forcing_years": [2006, 2300],
                 "output_years": [2006, 2300],
                 "burned_area_km2": {"model": {"2015": 115.0, "2016": 60.3}, "pattern": "thin",
                                     "observed": {"2015": 3.7, "2016": 2.6},
                                     "observed_source": "MODIS"}}
        out = CH.check(exp, {"name": "naches", "facts": facts})
        fire = [f["what"] for f in out["facts"] if "own fire burns" in f["what"]]
        assert fire == ["the control's own fire burns 60-115 km2 a year (thin); it is the "
                        "model's fire, and the fires observed in those years are not in "
                        "the run"]

    def test_the_planner_offers_no_observation_to_the_analysis(self):
        text = (self.ROOT / "src" / "agents" / "prompts" / "planner.txt").read_text()
        wd = text[text.index("THE WHOLE-DOMAIN PATH"):text.index("THE COLUMN BUDGET")]
        assert "controls[].observations" not in wd and "observations   " not in wd
        assert "The answer is read from the model's output alone" in wd
        assert "No caveat carries an observation's numbers" in wd
        assert '"observations": ["<a name from controls[].observations>"]' not in text

    def test_the_analyzer_is_told_both_rules(self):
        from agents.analysis import step3_interpret as S3
        src = (self.ROOT / "src" / "agents" / "analysis" / "step3_interpret.py").read_text()
        assert "CAUSES REST ON NUMBERS" in src and "why as far as the numbers show it" in src
        assert "why (the mechanism in plain words)" not in src

        class Ctx:
            is_domain = True
            data = {"regional": {"runs": {"control": {"fields": {
                "GPP": {"how": "sum"}, "TLAI": {"how": "mean"}}}}}}
        lines = "\n".join(S3.domain_reading(Ctx()))
        assert "THE MODEL'S OUTPUT ONLY" in lines
        assert "Never use an observation to explain, support or rule out a result" in lines


class TestThePlanSaysMonthsAndSeasons:
    """MONTHS AND SEASONS ON THE PAGE (2026-11-03): the server's analysis plan
    says how each field's month and season are made, in what units, and over
    which months each season runs, from the definitions the table uses; the
    design page quotes it."""

    def test_each_field_by_month_and_season(self, tmp_path):
        c = _case(tmp_path, variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
        out = A.analysis_plan({"case_dir": str(c["case_dir"]), "run_dir": str(c["run_dir"]),
                               "regions": {}},
                              {"regions": [A.VEGETATED], "variables": ["GPP", "TSA",
                                                                       "FAREA_BURNED"]})
        f = {x["name"]: x for x in out["fields"]}
        assert (f["GPP"]["how"], f["GPP"]["by_month"], f["GPP"]["by_season"]) == \
            ("total over the year", "total over the month", "total over the season")
        assert (f["GPP"]["units"], f["GPP"]["month_units"], f["GPP"]["season_units"]) == \
            ("gC/m^2/yr", "gC/m^2/month", "gC/m^2/season")
        assert (f["TSA"]["by_month"], f["TSA"]["month_units"]) == ("mean over the month", "K")
        assert (f[A.BURNED_AREA]["by_season"], f[A.BURNED_AREA]["season_units"]) == \
            ("total over the season", "km2")
        seasons = {s["name"]: s["months"] for s in out["periods"]["seasons"]}
        assert seasons == {"DJF": ["December of the year before", "January", "February"],
                           "MAM": ["March", "April", "May"],
                           "JJA": ["June", "July", "August"],
                           "SON": ["September", "October", "November"]}
        # the periods are the table's own: its seasons, and the reason it gives
        # for a winter it leaves out
        assert [n for n, _ in A.SEASONS] == list(seasons)
        assert A.SEASON_LEFT_OUT in out["periods"]["season_left_out"]

    def test_the_design_page_quotes_them(self, tmp_path):
        preview = {"analysis": {"fields": [
            {"name": "GPP", "stream": "h0", "how": "total over the year", "units": "gC/m^2/yr",
             "by_month": "total over the month", "month_units": "gC/m^2/month",
             "by_season": "total over the season", "season_units": "gC/m^2/season"}],
            "periods": A.periods_plan()}}
        text = "\n".join(DR._appendix(preview, tmp_path, {},
                                      {"domain_experiment": {"years": [2015]}}))
        assert ("| GPP | h0 | total over the year | gC/m^2/yr | total over the month "
                "(gC/m^2/month) | total over the season (gC/m^2/season) |") in text
        assert ("each season: DJF (December of the year before, January, February); MAM "
                "(March, April, May); JJA (June, July, August); SON (September, October, "
                "November). A season whose December the run's output does not hold is left "
                "out, not made from two months.") in text
        assert DR.MONTHS_AND_SEASONS_LINE not in text
