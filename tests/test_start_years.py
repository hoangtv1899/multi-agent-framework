"""A run starts from the year asked about, where the control saved its state.

    pytest tests/test_start_years.py -v

Every new run was held to the control's own first year, so a question about
two late years simulated every year before them. The control saves its state
on every 1 January, and a run can start from any of them. The model server
now lists the years it has a saved state for, and nothing else decides where
a run starts (domain_controls.start_years, domain_changes.start_from).

    the list        read from the control's run folder: its own first year
                    and each later 1 January it saved, inside its output.
    the check       a run's first year is one of them; any other is refused
                    with the list. A later start is said to be untested until
                    a run started that way has been completed here.
    the build       the run's recipe starts on 1 January of its first year,
                    from the control's saved state of that day, and its case
                    job expects that one more difference from the control.
    the cost        only the years run are counted.
    the page        says where each new run starts.

The manager and the model server run together in process on a made-up
control, with the scheduler that records every submission, as in
tests/test_domain_experiment.py. Every number is made up.
"""
import json
import re
import shutil
from pathlib import Path

import pytest

import domain_case as DC
import domain_changes as CH
import domain_controls as DCT
from test_design_preview import _preview
from test_domain_experiment import NOFIRE, study  # noqa: F401
from test_domain_manager import mgr, srv  # noqa: F401

from core import approved_spending as SP


def _first_state(study):
    """The state the made-up control itself started from."""
    nl = (study["control_dir"] / "user_nl_elm").read_text()
    return Path(re.search(r"finidat = '([^']+)'", nl).group(1))


def _save_state(study, year, name=None):
    """The control's saved state of 1 January `year`, as ELM names it."""
    p = study["control_run"] / (name or f"src_case.elm.r.{year}-01-01-00000.nc")
    shutil.copy(_first_state(study), p)
    return p


def _entry():
    return dict(DCT.get("synth"), facts=DCT.facts("synth"))


def _experiment(years):
    return {"control": {"name": "synth", "reuse": True}, "runs": NOFIRE, "years": years}


def _with_years(study, years):
    st = json.loads(json.dumps(study["cfg"]["strategy"]))
    st["domain_experiment"]["years"] = years
    study["cfg"]["strategy"] = st
    return st


# ── the list ─────────────────────────────────────────────────────────────────
class TestTheServerListsTheYears:

    def test_with_no_saved_state_only_the_controls_own_first_year(self, study):
        assert DCT.saved_states(_entry()) == {}
        assert DCT.start_years(_entry()) == [2015]

    def test_each_first_of_january_it_saved_is_a_year_to_start_from(self, study):
        p = _save_state(study, 2016)
        assert DCT.saved_states(_entry()) == {2016: str(p)}
        assert DCT.start_years(_entry()) == [2015, 2016]
        row = DCT.describe()["controls"][0]
        assert row["facts"]["start_years"] == [2015, 2016]
        assert row["facts"]["start_year"] == 2015             # its own, as before

    def test_a_state_past_its_output_or_of_another_day_is_no_start(self, study):
        _save_state(study, 2017)                              # the output ends in 2016
        _save_state(study, 2016, "src_case.elm.r.2016-06-01-00000.nc")
        _save_state(study, 2016, "src_case.elm.rh0.2016-01-01-00000.nc")
        _save_state(study, 2016, "src_case.cpl.r.2016-01-01-00000.nc")
        assert sorted(DCT.saved_states(_entry())) == [2017]
        assert DCT.start_years(_entry()) == [2015]

    def test_a_control_with_no_run_folder_or_no_facts(self, tmp_path):
        assert DCT.saved_states(None) == {} and DCT.saved_states({}) == {}
        assert DCT.saved_states({"run_dir": str(tmp_path / "gone")}) == {}
        assert DCT.start_years({"run_dir": str(tmp_path)}, {}) == []
        assert DCT.start_years(None, {"start_year": 2031, "output_years": [2031, 2033]}) == [2031]


# ── the check ────────────────────────────────────────────────────────────────
class TestARunsFirstYearIsOneOfThem:

    def test_a_later_year_with_a_saved_state_builds(self, study):
        _save_state(study, 2016)
        chk = CH.check(_experiment([2016]), _entry())
        assert chk["ok"] and chk["wont_build"] == [], chk["wont_build"]
        said = [u["what"] for u in chk["unusual"] if u["where"] == "years"]
        assert len(said) == 1 and said[0].startswith(
            "not demonstrated: the run starts from the control's saved state of 1 January "
            "2016")
        assert "its start is untested" in said[0]

    def test_the_controls_own_first_year_is_as_it_was(self, study):
        _save_state(study, 2016)
        chk = CH.check(_experiment([2015, 2016]), _entry())
        assert chk["ok"] and not [u for u in chk["unusual"] if u["where"] == "years"]

    def test_a_year_with_no_saved_state_is_refused_with_the_list(self, study):
        chk = CH.check(_experiment([2016]), _entry())
        assert not chk["ok"]
        assert chk["wont_build"] == [{"where": "years", "what": (
            "a run starts from a saved state of the control, which has one for 1 January "
            "of 2015: the run's first year is one of those, not 2016")}]

    def test_the_list_is_said_as_a_span_or_one_by_one(self):
        facts = {"start_year": 2031, "output_years": [2031, 2039], "forcing_years": [2031, 2039]}
        bad, _ = CH._years_problems([2033], facts, True, start_years=[2031, 2032, 2034])
        assert "1 January of 2031, 2032, 2034: the run's first year" in bad[0]
        bad, _ = CH._years_problems([2030], dict(facts, output_years=[2030, 2039],
                                                forcing_years=[2030, 2039]), True,
                                    start_years=[2031, 2032, 2033, 2034])
        assert "1 January of 2031 to 2034: the run's first year" in bad[0]

    def test_a_control_read_alone_starts_nowhere(self, study):
        """No job runs, so no saved state is needed for any year of its output."""
        chk = CH.check({"control": {"name": "synth", "reuse": True}, "runs": [],
                        "years": [2016]}, _entry())
        assert chk["ok"] and not [u for u in chk["unusual"] if u["where"] == "years"]

    def test_once_a_later_start_has_completed_here_it_is_no_longer_said(self, study,
                                                                        monkeypatch):
        _save_state(study, 2016)
        monkeypatch.setattr(CH, "LATER_START_DEMONSTRATED", True)
        chk = CH.check(_experiment([2016]), _entry())
        assert chk["ok"] and not [u for u in chk["unusual"] if u["where"] == "years"]

    def test_only_the_years_run_are_counted_in_the_cost(self, study):
        _save_state(study, 2016)
        both = CH.check(_experiment([2015, 2016]), _entry())["cost"]
        one = CH.check(_experiment([2016]), _entry())["cost"]
        assert (both["model_years"], one["model_years"]) == (2, 1)
        assert one["node_hours"] == pytest.approx(both["node_hours"] / 2, abs=0.05)


# ── the build ────────────────────────────────────────────────────────────────
class TestTheRunStartsFromThatState:

    def test_its_recipe_starts_that_day_from_the_saved_state(self, study):
        state = _save_state(study, 2016)
        _with_years(study, [2016])
        m, cfg, st, p = _preview(study)
        assert p["refused"] == [] and p["error"] is None, p
        inp = m.input_dir / "runs" / "nofire"
        recipe = DC.Recipe.from_json((inp / "domain_recipe.json").read_text())
        assert recipe.start == "2016-01-01" and recipe.first_year == 2016
        assert recipe.setting("RUN_STARTDATE") == "2016-01-01"
        assert recipe.inputs()["finidat"] == str(state)
        assert recipe.check() == []
        assert any(n == f"starts on 2016-01-01 from the control's saved state of that day: "
                        f"{state}" for n in recipe.notes)
        # the control's own recipe is as it was
        ctl = DC.recipe_from_case(study["control_dir"])
        assert ctl.start == "2015-01-01" and ctl.inputs()["finidat"] == str(_first_state(study))
        # the case job expects that one more difference from the control's settings
        expect = json.loads((inp / "domain_expectations.json").read_text())
        assert expect["settings"]["lnd_in"]["finidat"] == str(state)
        # one model year, and the stage script starts the model that day
        run = p["runs"][0]
        assert [j["stage"] for j in run["jobs"]] == ["case", "smoke", "y2016", "report"]
        assert "RUN_STARTDATE=2016-01-01" in (m.run_dir / run["stage_script"]).read_text()
        assert SP.estimate(p) == pytest.approx(0.8, abs=0.05)

    def test_after_the_yes_its_chain_is_submitted_for_that_year_alone(self, study):
        from elm_exp_manager import ELMExpManager
        _save_state(study, 2016)
        _with_years(study, [2016])
        m, cfg, st, p = _preview(study)
        again = ELMExpManager(base_output_dir=str(m.run_dir.parent), run_dir=str(m.run_dir))
        out = again.execute_plan(dict(st), dict(cfg, strategy=st))
        assert out.get("status") == "pending", out
        assert [(j["run"], j["stage"]) for j in out["chain"]] == [
            ("nofire", "case"), ("nofire", "smoke"), ("nofire", "y2016"),
            ("nofire", "report")]

    def test_a_run_from_the_controls_own_first_year_is_untouched(self, study):
        _save_state(study, 2016)
        m, cfg, st, p = _preview(study)                       # years 2015 and 2016
        inp = m.input_dir / "runs" / "nofire"
        recipe = DC.Recipe.from_json((inp / "domain_recipe.json").read_text())
        assert recipe.start == "2015-01-01"
        assert recipe.inputs()["finidat"] == str(_first_state(study))
        expect = json.loads((inp / "domain_expectations.json").read_text())
        assert "finidat" not in (expect["settings"].get("lnd_in") or {})
        assert [j["stage"] for j in p["runs"][0]["jobs"]] == ["case", "smoke", "y2015",
                                                              "y2016", "report"]

    def test_a_state_that_is_gone_when_the_run_is_built_is_an_error(self, study):
        ctl = DC.recipe_from_case(study["control_dir"])
        with pytest.raises(CH.ChangeError, match="no saved state for 1 January 2016"):
            CH.build_run(ctl, NOFIRE[0], study["mgr"].input_dir, [2016],
                         control_case_dir=str(study["control_dir"]), control_entry=_entry())

    def test_a_control_that_runs_with_its_runs_has_no_saved_state_to_start_from(self, study):
        ctl = DC.recipe_from_case(study["control_dir"])
        recipe, expect = CH.build_run(ctl, NOFIRE[0], study["mgr"].input_dir, [2016],
                                      control_case_dir=str(study["control_dir"]),
                                      control_entry=None)
        assert recipe.start == "2015-01-01"
        assert "finidat" not in (expect["settings"].get("lnd_in") or {})


# ── the page ─────────────────────────────────────────────────────────────────
class TestThePageSaysWhereARunStarts:

    def _page(self, years, preview):
        from agents import design_page as DP
        strategy = {"feasibility": {"verdict": "full"},
                    "domain_experiment": {"control": {"name": "synth", "reuse": True},
                                          "runs": NOFIRE, "years": years}}
        return " ".join(DP.render({"user_request": "a question", "brief": {}}, strategy,
                                  preview=preview).split())

    def test_it_names_the_year_the_run_starts_from(self, study):
        _save_state(study, 2016)
        _with_years(study, [2016])
        m, cfg, st, p = _preview(study)
        page = self._page([2016], p)
        assert "Each new run starts from the control run's saved state at the beginning " \
               "of 2016 and simulates 2016." in page
        assert "The model server has a note on years." in page
        assert "ot demonstrated: the run starts from the control's saved state of " \
               "1 January 2016" in page
        assert "It allows the 0.8 node-hours estimated above, and no more." in page

    def test_from_the_controls_first_year_it_says_that_year(self, study):
        m, cfg, st, p = _preview(study)
        page = self._page([2015, 2016], p)
        assert "Each new run starts from the control run's saved state at the beginning " \
               "of 2015 and simulates 2015 to 2016." in page
        assert "not demonstrated: the run starts" not in page
