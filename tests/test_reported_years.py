"""The years a study reports are not the years it simulates (2026-11-10).

    pytest tests/test_reported_years.py -v

Three things a breadth test of the front half showed, and what holds now:

  * a question about later years of the control alone was refused until its
    years began at the control's start year, although nothing was run: a rule
    about runs is applied only when a run executes;
  * a what-if question about one year got the start year its run needs in its
    numbers: the years reported are the years asked about and their normal
    years (core/analysis_years), and the start year is simulated only;
  * the page called a widened span "you stated it": each part of the When
    line is labelled by whose choice it is.

Also here: a change with no completed run is labelled as not demonstrated, by
the server's list and on the page, and a live report says which kinds of
statement code did not check. Every year, name and number is made up.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

from core import analysis_years as AY                       # noqa: E402
from agents import design_review as dr                      # noqa: E402
import domain_changes as CH                                 # noqa: E402

START = 2031
FACTS = {"start_year": START, "forcing_years": [2029, 2039], "output_years": [2031, 2036],
         "settings": {"ELM_CO2_TYPE": "constant", "co2_ppmv": 400.0},
         "hours_per_model_year": 0.5}
CONTROL = {"name": "c", "facts": FACTS}
FIRE_OFF = [{"name": "nofire", "changes": [{"id": "fire_off"}]}]


def _brief(asked, resolved, runs=None, years=None, source="user"):
    return {"design_archetype": "whole_domain", "model": "elm", "domain": {"name": "c"},
            "run_settings": {
                "requested_period": ({"yr_start": asked[0], "yr_end": asked[-1]}
                                     if asked else None),
                "resolved_period": {"yr_start": resolved[0], "yr_end": resolved[-1],
                                    "source": source}},
            "domain_experiment": {"control": {"name": "c", "reuse": True},
                                  "runs": runs or [], "years": years or list(resolved)}}


def _strategy(years, runs=None, normal=None, report=None):
    an = {"variables": ["A_FIELD"], "normal_years": normal or []}
    if report is not None:
        an["years"] = report
    return {"archetype": "whole_domain",
            "domain_experiment": {"control": {"name": "c", "reuse": True},
                                  "runs": runs or [], "years": list(years), "analysis": an},
            "domain_rulebook": {"controls": {"controls": [dict(CONTROL)]},
                                "changes": CH.declare()}}


def _years(a, b):
    return list(range(a, b + 1))


# ── which years are reported ─────────────────────────────────────────────────
class TestTheYearsReported:

    def test_a_runs_start_year_is_simulated_and_not_reported(self):
        """Asked about one year; the run must start a year earlier."""
        years, how = AY.reported(_brief([2032], [2031, 2032], FIRE_OFF),
                                 _strategy([2031, 2032], FIRE_OFF))
        assert years == [2032]
        assert how["left_out"] == [2031] and how["source"] == "request"

    def test_the_normal_years_are_reported_with_the_years_asked_about(self):
        years, how = AY.reported(_brief([2036], [2032, 2036]),
                                 _strategy(_years(2032, 2036), normal=_years(2032, 2035)))
        assert years == _years(2032, 2036) and how["left_out"] == []

    def test_a_start_year_added_out_of_habit_is_left_out_too(self):
        """The control alone, asked about 2033 to 2035, in a design that still
        begins at the start year: the numbers hold the years asked about."""
        years, how = AY.reported(_brief(_years(2033, 2035), [2031, 2035]),
                                 _strategy(_years(2031, 2035)))
        assert years == _years(2033, 2035) and how["left_out"] == [2031, 2032]

    def test_the_plans_own_years_win(self):
        years, how = AY.reported(_brief([2032], [2031, 2033], FIRE_OFF),
                                 _strategy(_years(2031, 2033), FIRE_OFF, report=[2032, 2033]))
        assert years == [2032, 2033] and how["source"] == "plan"

    @pytest.mark.parametrize("asked", [None, [2040]])
    def test_with_nothing_to_narrow_them_every_simulated_year_is_reported(self, asked):
        years, how = AY.reported(_brief(asked, [2031, 2032]), _strategy([2031, 2032]))
        assert years == [2031, 2032] and how["source"] == "all" and how["left_out"] == []

    def test_plan_years_outside_the_experiment_are_not_used(self):
        years, how = AY.reported(_brief([2032], [2031, 2032]),
                                 _strategy([2031, 2032], report=[2032, 2050]))
        assert years == [2032] and how["source"] == "request"


# ── the server's check ───────────────────────────────────────────────────────
def _wont(exp):
    return [w["what"] for w in CH.check(exp, CONTROL)["wont_build"]]


class TestTheServersRuleIsAboutRuns:

    def test_the_control_alone_may_start_after_its_saved_state(self):
        exp = {"control": {"name": "c", "reuse": True}, "runs": [],
               "years": _years(2033, 2035)}
        assert _wont(exp) == []

    def test_a_run_still_starts_at_the_saved_state(self):
        exp = {"control": {"name": "c", "reuse": True}, "runs": FIRE_OFF,
               "years": _years(2033, 2035)}
        assert any(f"a run starts from the control's saved state at {START}" in w
                   for w in _wont(exp))

    def test_the_control_alone_stays_inside_its_output(self):
        exp = {"control": {"name": "c", "reuse": True}, "runs": [],
               "years": _years(2035, 2037)}
        assert any("outside the reused control's output" in w for w in _wont(exp))

    def test_the_years_reported_are_among_the_experiments(self):
        exp = {"control": {"name": "c", "reuse": True}, "runs": FIRE_OFF,
               "years": [2031, 2032], "analysis": {"years": [2032]}}
        assert _wont(exp) == []
        exp["analysis"]["years"] = [2032, 2034]
        assert any("[2034] are not among the experiment's years" in w for w in _wont(exp))
        exp["analysis"]["years"] = []
        assert any("non-empty list of whole years" in w for w in _wont(exp))


# ── the page ─────────────────────────────────────────────────────────────────
def _page(brief, strategy):
    return dr.render({"user_request": "a question", "brief": brief}, strategy, "x")


def _when(page):
    return next(ln for ln in page.splitlines() if ln.startswith("- **When:**"))


class TestThePageSaysWhoseChoiceEachYearIs:

    def test_a_widened_span_is_not_called_the_persons(self):
        page = _page(_brief([2032], [2031, 2032], FIRE_OFF), _strategy([2031, 2032], FIRE_OFF))
        when = _when(page)
        assert "you stated it" not in page
        assert "2032 (the years you asked about)" in when
        assert (f"Also simulated and not reported: 2031 (required: a run starts from the "
                f"control's saved state at {START})") in when
        assert "- simulated: 2 model year(s), 2031–2032, for 1 run(s) to execute" in page
        assert ("- reported: the numbers are read for 2032; 2031 is simulated and not "
                "reported") in page

    def test_a_change_with_no_year_selector_is_said_to_cover_every_year(self):
        """Asked about two summers; the run's rain is scaled in every summer
        it simulates. The page says so by code, not by a caveat alone."""
        runs = [{"name": "dry", "changes": [{"id": "weather_scale",
                                             "params": {"precip_factor": 0.7}}]}]
        page = _page(_brief([2034, 2035], [2031, 2035], runs),
                     _strategy(_years(2031, 2035), runs))
        assert ("- in effect in EVERY simulated year, 2031–2035, not only the years "
                "reported: weather_scale. A reported year carries the change's effect "
                "from 2031 on") in page

    def test_a_change_that_sets_its_own_years_is_not(self):
        runs = [{"name": "again", "changes": [{"id": "weather_repeat_year",
                                               "params": {"map": {"2032": 2031}}}]}]
        page = _page(_brief([2032], [2031, 2032], runs), _strategy([2031, 2032], runs))
        assert "in effect in EVERY simulated year" not in page

    def test_with_every_year_reported_the_line_is_not_needed(self):
        page = _page(_brief([2031, 2032], [2031, 2032], FIRE_OFF),
                     _strategy([2031, 2032], FIRE_OFF))
        assert "in effect in EVERY simulated year" not in page

    def test_the_normal_years_are_the_designs(self):
        when = _when(_page(_brief([2036], [2032, 2036]),
                           _strategy(_years(2032, 2036), normal=_years(2032, 2035))))
        assert when == ("- **When:** 2036 (the years you asked about); compared with "
                        "2032–2035 (the normal years, chosen by the design)")

    def test_the_control_alone_simulates_nothing(self):
        page = _page(_brief(_years(2033, 2035), [2033, 2035]), _strategy(_years(2033, 2035)))
        assert _when(page) == "- **When:** 2033–2035 (the years you asked about)"
        assert "- simulated:" not in page
        assert ("- reported: the numbers are read for 2033–2035; nothing is simulated, "
                "the control's existing output is read") in page

    def test_years_the_request_never_named_are_the_designs(self):
        when = _when(_page(_brief(None, [2031, 2032]), _strategy([2031, 2032])))
        assert when == ("- **When:** 2031–2032 (the design's years; the request named none)")

    def test_a_single_column_study_whose_span_was_widened(self):
        brief = {"design_archetype": "site", "domain": {"name": "Testville"},
                 "run_settings": {"requested_period": {"yr_start": 2032, "yr_end": 2032},
                                  "resolved_period": {"yr_start": 2031, "yr_end": 2032,
                                                      "source": "user"}}}
        line = dr._period_line(brief, {"archetype": "site"})
        assert line == "2031–2032 (you asked about 2032; the design reads the wider span)"

    def test_what_the_person_stated_is_still_called_theirs(self):
        brief = {"run_settings": {"requested_period": {"yr_start": 2032, "yr_end": 2033},
                                  "resolved_period": {"yr_start": 2032, "yr_end": 2033,
                                                      "source": "user"}}}
        assert dr._period_line(brief) == "2032–2033 (you stated it)"


# ── every caveat, and what Reception noted ───────────────────────────────────
class TestThePageShowsEveryCaveat:

    def _with_caveats(self, n, text="a caveat about the design"):
        st = _strategy([2031, 2032], FIRE_OFF)
        st["domain_experiment"]["caveats"] = [f"{text} number {k}" for k in range(1, n + 1)]
        return _page(_brief([2031, 2032], [2031, 2032], FIRE_OFF), st)

    def test_a_fourth_caveat_is_on_the_page(self):
        page = self._with_caveats(5)
        assert all(f"- caveat: a caveat about the design number {k}" in page
                   for k in range(1, 6))

    def test_beyond_the_limit_they_are_counted_not_dropped(self):
        page = self._with_caveats(dr.CAVEATS_SHOWN + 3)
        assert f"number {dr.CAVEATS_SHOWN}" in page
        assert f"number {dr.CAVEATS_SHOWN + 1}" not in page
        assert "- and 3 more caveat(s), in `strategy.json`" in page

    def test_a_long_caveat_keeps_its_second_sentence(self):
        long = ("The change has no setting for this. " + "x" * 190
                + " The second sentence says which years.")
        st = _strategy([2031, 2032], FIRE_OFF)
        st["domain_experiment"]["caveats"] = [long]
        assert "The second sentence says which years." in \
            _page(_brief([2031, 2032], [2031, 2032], FIRE_OFF), st)

    def test_receptions_notes_are_not_called_data_gaps(self):
        brief = _brief([2031, 2032], [2031, 2032], FIRE_OFF)
        brief["known_data_gaps"] = ["the change is demonstrated in this framework"]
        page = _page(brief, _strategy([2031, 2032], FIRE_OFF))
        assert "- noted by Reception: the change is demonstrated" in page
        assert "data gap:" not in page

    def test_a_single_column_study_keeps_the_old_label(self):
        brief = {"design_archetype": "site", "domain": {"name": "Testville"},
                 "known_data_gaps": ["no tower in the basin"],
                 "run_settings": {"resolved_period": {"yr_start": 2031, "yr_end": 2031,
                                                      "source": "user"}}}
        page = dr.render({"user_request": "q", "brief": brief}, {"archetype": "site"}, "x")
        assert "- data gap: no tower in the basin" in page

    def test_a_change_that_names_its_years_is_not_said_to_cover_every_year(self):
        runs = [{"name": "dry", "changes": [{"id": "weather_scale", "params": {
            "precip_factor": 0.7, "months": [6, 7, 8], "years": [2034, 2035]}}]}]
        page = _page(_brief([2034, 2035], [2031, 2035], runs),
                     _strategy(_years(2031, 2035), runs))
        assert "in effect in EVERY simulated year" not in page
        assert "- reported: the numbers are read for 2034–2035" in page


# ── a run's change in the run's own words ────────────────────────────────────
class TestTheRunsOwnWords:

    RUNS = [{"name": "dry", "changes": [{"id": "weather_scale", "params": {
        "precip_factor": 0.7, "months": [6, 7, 8], "years": [2032]}}]}]

    def _render(self, tmp_path, with_check):
        if with_check:
            (tmp_path / "01_inputs").mkdir()
            resolved = [dict(r, changes=[CH.resolve(c) for c in r["changes"]])
                        for r in self.RUNS]
            (tmp_path / dr.RESOLVED_FILE).write_text(json.dumps(
                {"check": {"runs": resolved}}, default=str))
        return dr.render({"user_request": "q", "brief": _brief([2032], [2031, 2032], self.RUNS)},
                         _strategy([2031, 2032], self.RUNS), "x", run_dir=tmp_path)

    def test_with_the_servers_check_on_disk_the_page_says_the_runs_words(self, tmp_path):
        page = self._render(tmp_path, True)
        assert ("  - weather_scale means the difference from the control is the response "
                "to rain and snow multiplied by 0.7, in June, July, August of 2032 only") \
            in page
        assert "the chosen months of the chosen years" not in page
        assert "  - it is NOT a consistent change of climate" in page

    def test_without_it_the_lists_general_words_stand(self, tmp_path):
        page = self._render(tmp_path, False)
        assert "the chosen months of the chosen years" in page


# ── run names already taken ──────────────────────────────────────────────────
class TestRunNamesAlreadyTaken:

    def test_the_cases_beside_the_controls_are_listed(self, tmp_path):
        import domain_controls as DCT
        for d in ("ctl", "ctl_warm", "ctl_fire_off", "ctl_Odd-Name", "other_dry"):
            (tmp_path / d).mkdir()
        (tmp_path / "ctl_a_file").write_text("not a case")
        assert DCT.run_names_taken({"case_dir": str(tmp_path / "ctl")}) == ["fire_off",
                                                                            "warm"]

    def test_a_control_whose_case_cannot_be_read_has_none(self, tmp_path):
        import domain_controls as DCT
        assert DCT.run_names_taken({"case_dir": str(tmp_path / "gone" / "ctl")}) == []
        assert DCT.run_names_taken({}) == []


# ── the table ────────────────────────────────────────────────────────────────
class TestTheTableHoldsTheYearsReported:

    def _years(self, tmp_path, config):
        import elm_exp_manager as M
        stub = SimpleNamespace(run_dir=tmp_path)
        return M.ELMExpManager._reported_years(stub, [{"years": [2031, 2032]}], config)

    def test_from_the_studys_brief_and_strategy(self, tmp_path):
        years, how = self._years(tmp_path, {
            "brief": _brief([2032], [2031, 2032], FIRE_OFF),
            "strategy": _strategy([2031, 2032], FIRE_OFF)})
        assert years == [2032] and how["left_out"] == [2031]

    def test_from_the_files_beside_the_run_in_a_finalize_job(self, tmp_path):
        (tmp_path / "reception.json").write_text(json.dumps(
            {"brief": _brief([2032], [2031, 2032], FIRE_OFF)}))
        (tmp_path / "strategy.json").write_text(json.dumps(_strategy([2031, 2032], FIRE_OFF)))
        years, _ = self._years(tmp_path, {})
        assert years == [2032]

    def test_every_simulated_year_when_nothing_is_recorded(self, tmp_path):
        years, how = self._years(tmp_path, {})
        assert years == [2031, 2032] and how["source"] == "all"


# ── a change with no completed run ───────────────────────────────────────────
class TestAChangeThatHasNeverRunSaysSo:

    def test_the_list_marks_every_available_change(self):
        menu = {c["id"]: c for c in CH.declare()["changes"]}
        for cid, c in menu.items():
            if c["available"]:
                assert isinstance(c["demonstrated"], bool), cid
            else:
                assert "demonstrated" not in c, cid
        assert menu["fire_off"]["demonstrated"] is True
        assert menu["co2_ppmv"]["demonstrated"] is False
        # the words are said once, for the whole list, so it fits a tool result
        assert "untested" in CH.declare()["demonstrated_false"]

    def test_the_check_calls_it_unusual_and_still_builds(self):
        exp = {"control": {"name": "c", "reuse": True}, "years": [2031],
               "runs": [{"name": "high", "changes": [{"id": "co2_ppmv",
                                                      "params": {"ppmv": 500.0}}]}]}
        out = CH.check(exp, CONTROL)
        assert out["wont_build"] == []
        assert [u for u in out["unusual"] if u["where"] == "run high, change co2_ppmv"
                and u["what"].startswith("not demonstrated: no run with this change")]

    def test_a_demonstrated_change_is_not_called_unusual(self):
        exp = {"control": {"name": "c", "reuse": True}, "years": [2031], "runs": FIRE_OFF}
        assert not [u for u in CH.check(exp, CONTROL)["unusual"]
                    if "not demonstrated" in u["what"]]

    def test_the_page_says_it_under_the_run(self):
        runs = [{"name": "high", "changes": [{"id": "co2_ppmv", "params": {"ppmv": 500.0}}]}]
        page = _page(_brief([2031], [2031, 2031], runs), _strategy([2031], runs))
        assert "**co2_ppmv is NOT DEMONSTRATED.** No run with this change has been completed" \
            in page
        page = _page(_brief([2031], [2031, 2031], FIRE_OFF), _strategy([2031], FIRE_OFF))
        assert "NOT DEMONSTRATED" not in page


# ── the live report says what code did not check ─────────────────────────────
class TestTheLiveReportSaysWhatCodeDidNotCheck:

    def _report(self, **kw):
        return dict({"kind": "whole_domain",
                     "claims": [{"claim": "the highest band lost most"},
                                {"claim": "the field fell by 3 units"}],
                     "withheld": [{"claim": "x", "struck_by": "r", "struck_because": "b"}],
                     "evidence_numbers": {"n_compared": 1200, "pass": True}}, **kw)

    def test_the_section_names_the_claims_with_ranking_words(self):
        from agents.analysis import step4_report as S4
        text = "\n".join(S4.verification_limits(self._report()))
        assert text.startswith(S4.LIMITS_HEADING)
        assert "1,200 numbers made again" in text and "kept 2 and withheld 1" in text
        assert "**Ranking words.**" in text and "Claim 1 above uses such a word." in text
        assert "**Figures and plotted series.**" in text and "**Interpretation.**" in text

    def test_no_ranking_word_is_said_too(self):
        from agents.analysis import step4_report as S4
        text = "\n".join(S4.verification_limits(self._report(
            claims=[{"claim": "the field fell by 3 units"}])))
        assert "No kept claim above uses such a word." in text

    def test_it_is_on_the_page_a_live_run_writes(self):
        from agents.analysis import step4_report as S4
        page = S4.render_markdown(self._report(question="q", answer="a", headline="h"))
        assert S4.LIMITS_HEADING in page
        assert page.index(S4.LIMITS_HEADING) < page.index("## Provenance")

    def test_another_kind_of_report_is_left_as_it_was(self):
        from agents.analysis import step4_report as S4
        assert S4.verification_limits(self._report(kind="site")) == []

    def test_the_preview_pages_read_the_same_words(self):
        from agents.analysis import step4_report as S4, report_views as RV
        assert RV.ranking_claims is S4.ranking_claims and RV._RANKING is S4.RANKING
