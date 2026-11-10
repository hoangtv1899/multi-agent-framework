#!/usr/bin/env python3
"""
DESIGN_REVIEW.md — a renderer, not a reasoner, and a real pause.

What is pinned: the page quotes the records verbatim (request, the
planner's justification and feasibility reason, the period's SOURCE);
each archetype gets its own middle; the page stays one readable screen;
the decision line records what actually happened; and the coordinator
STOPS on it with a person present, continues without one, and treats
no-consent as no.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents import design_review as dr                     # noqa: E402


def _site():
    reception = {
        "user_request": "How does runoff vary with elevation in Testville?",
        "brief": {
            "design_archetype": "site", "model": "elm",
            "model_rationale": "the land surface is the question",
            "domain": {"name": "Testville", "huc": "01234567",
                       "area_km2": 100.4, "source": "USGS WBD"},
            "run_settings": {
                "resolved_period": {"yr_start": 1988, "yr_end": 1988,
                                    "source": "default"},
                "conflicts": ["period unstated; 1988 chosen as default"]},
            "known_data_gaps": ["no ET tower in basin"],
        },
    }
    strategy = {
        "archetype": "site",
        "goals": ["Quantify runoff against elevation."],
        "feasibility": {"verdict": "partial",
                        "why": "no routed discharge in a 1-D column"},
        "sampling": {"approach": "stratified by elevation band",
                     "n_columns": 7,
                     "justification": "relief 900 m -> 3 bands x 2 + 1 pin"},
        "validation": [{"variable": "swe", "stations": ["642:WA:SNTL"],
                        "comparison": "daily"}],
    }
    return reception, strategy


class TestTheRendererQuotesTheRecords:

    def test_site_page_carries_the_records_verbatim(self):
        reception, strategy = _site()
        page = dr.render(reception, strategy, "elm_run_x")
        assert "How does runoff vary with elevation in Testville?" in page
        assert "relief 900 m -> 3 bands x 2 + 1 pin" in page
        assert "no routed discharge in a 1-D column" in page
        assert "partial" in page and "642:WA:SNTL" in page

    def test_a_defaulted_period_is_called_a_default(self):
        reception, strategy = _site()
        page = dr.render(reception, strategy, "x")
        assert "DEFAULT" in page
        reception["brief"]["run_settings"]["resolved_period"]["source"] = \
            "carried"
        assert "carried from the prior run" in dr.render(reception, strategy, "x")

    def test_conceptual_middle_names_factors_and_held_fixed(self):
        reception = {"user_request": "sweep the water table",
                     "brief": {"design_archetype": "conceptual",
                               "model": "pflotran",
                               "sweep": {"factors": [{"name": "water_table_m",
                                                      "levels": [2, 20],
                                                      "settled_by": "user"}],
                                         "held_fixed": {"soil": "loam"},
                                         "held_fixed_settled_by":
                                             {"soil": "user"}}}}
        strategy = {"archetype": "conceptual", "goals": [],
                    "feasibility": {"verdict": "full", "why": "its core output"},
                    "sampling": {"n_columns": 2}}
        page = dr.render(reception, strategy, "x")
        assert "water_table_m" in page and "[2, 20]" in page
        assert "loam" in page and "no basin" in page

    def test_coupling_middle_names_the_prior_and_what_travels(self):
        reception = {"user_request": "re-run ELM from the prior PFLOTRAN run",
                     "brief": {"design_archetype": "coupling", "model": "elm",
                               "coupling": {"prior_experiment": "pf_run_1",
                                            "n_columns_prior": 3,
                                            "from_model": "pflotran",
                                            "to_model": "elm",
                                            "coupling_variable":
                                                "solved water table"}}}
        strategy = {"archetype": "coupling", "goals": [],
                    "feasibility": {"verdict": "full", "why": "carried"},
                    "coupling": {"prior_experiment": "pf_run_1"}}
        page = dr.render(reception, strategy, "x")
        assert "pf_run_1" in page and "3 of them" in page
        assert "pflotran → elm" in page and "nothing is fetched" in page.lower()

    def test_the_page_stays_one_readable_screen(self):
        reception, strategy = _site()
        page = dr.render(reception, strategy, "x")
        assert len(page.splitlines()) <= 60

    def test_write_and_decide_on_a_run_dir(self, tmp_path):
        reception, strategy = _site()
        (tmp_path / "reception.json").write_text(json.dumps(reception))
        (tmp_path / "strategy.json").write_text(json.dumps(strategy))
        p = dr.write_review(tmp_path)
        assert p.name == "DESIGN_REVIEW.md" and "pending" in p.read_text()
        dr.record_decision(tmp_path, "accepted at the terminal")
        text = p.read_text()
        assert "accepted at the terminal at 20" in text
        assert "pending" not in text


# ── the pause in the coordinator ─────────────────────────────────────
def _coordinator(tmp_path, interactive):
    import workflow as wf
    co = object.__new__(wf.WorkflowCoordinator)
    co.conversation_context = {}
    co.interactive_reception = interactive
    reception, strategy = _site()
    co._plan = lambda result, run_dir: strategy
    return co, reception


class TestThePauseIsReal:

    def test_unattended_runs_record_and_continue(self, tmp_path, monkeypatch):
        co, reception = _coordinator(tmp_path, interactive=False)
        co._execute = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("SENTINEL-EXECUTED"))
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(
            "unattended run must never ask"))
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "SENTINEL-EXECUTED" in out
        assert "auto-continued" in (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_a_person_can_decline_and_nothing_runs(self, tmp_path, monkeypatch):
        co, reception = _coordinator(tmp_path, interactive=True)
        co._execute = lambda **kw: pytest.fail("declined design must not run")
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "declined" in out.lower() and "nothing was run" in out
        assert "DECLINED" in (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_a_person_can_accept_and_it_runs(self, tmp_path, monkeypatch):
        co, reception = _coordinator(tmp_path, interactive=True)
        co._execute = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("SENTINEL-EXECUTED"))
        monkeypatch.setattr("builtins.input", lambda *a: "yes")
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "SENTINEL-EXECUTED" in out
        assert "accepted at the terminal" in \
            (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_a_refusal_before_any_job_is_an_answer_not_a_crash(self, tmp_path,
                                                                monkeypatch):
        from core.exp_manager_base import RefusedBeforeAnyJob
        co, reception = _coordinator(tmp_path, interactive=False)
        co._execute = lambda **kw: (_ for _ in ()).throw(RefusedBeforeAnyJob(
            "the experiment will not build: run warm: a case named c_warm exists "
            "already"))
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert out.startswith("🛑 The model server's check refused this design, so "
                              "nothing was submitted.")
        assert "a case named c_warm exists already" in out and str(tmp_path) in out
        assert "Traceback" not in out and "Pipeline failed" not in out

    def test_no_consent_is_a_no(self, tmp_path, monkeypatch):
        co, reception = _coordinator(tmp_path, interactive=True)
        co._execute = lambda **kw: pytest.fail("EOF at the prompt must not run")
        def eof(*a):
            raise EOFError
        monkeypatch.setattr("builtins.input", eof)
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "declined" in out.lower()


# ── a change asked for at the review keeps what was settled ─────────
# A live revision (2026-10-27) asked only to add elevation bands to the
# reporting; the re-run reception saw just the request and the change, and
# dropped the region the user had agreed, warming the whole domain instead of
# the watershed. The design under review and the user's answers now go back.
def _warm_run():
    return {
        "user_request": "How would a 2 K warmer summer change the watershed in 2015?",
        "brief": {
            "design_archetype": "whole_domain", "model": "elm",
            "domain": {"name": "naches"},
            "domain_experiment": {
                "control": {"name": "naches", "reuse": True}, "years": [2015],
                "runs": [{"name": "warm", "changes": [{"id": "weather_scale", "params": {
                    "temperature_K": 2.0, "precip_factor": 0.8,
                    "months": [6, 7, 8], "region": "watershed"}}]}]},
            "run_settings": {"resolved_period": {"yr_start": 2015, "yr_end": 2015},
                             "conflicts": ["not a decision"]},
            "scientific_framing": {"goals": ["re-derived, not carried"]},
        },
        "trace": [
            {"tool": "describe_domain_changes", "args": {}, "result": {"changes": {}}},
            {"tool": "ask_user", "args": {"question": "Whole domain or the watershed?"},
             "result": {"answer": "Over the watershed only, please."}},
        ],
    }


class TestARevisionKeepsWhatWasSettled:

    def test_the_settled_design_and_the_answers_go_back(self):
        import workflow as wf
        rev = wf._revision_of(_warm_run(), "also report the elevation bands")
        exp = rev["settled"]["domain_experiment"]
        assert exp["runs"][0]["changes"][0]["params"]["region"] == "watershed"
        assert rev["settled"]["run_settings"] == {
            "resolved_period": {"yr_start": 2015, "yr_end": 2015}}
        assert "scientific_framing" not in rev["settled"]
        assert rev["user_answers"] == [{"question": "Whole domain or the watershed?",
                                        "answer": "Over the watershed only, please."}]
        assert rev["changes"] == ["also report the elevation bands"]

    def test_a_second_revision_keeps_the_first_answers(self):
        import workflow as wf
        first = wf._revision_of(_warm_run(), "also report the elevation bands")
        again = _warm_run()
        again["trace"] = []                       # the revision asked nothing
        again["revision_of"] = first
        rev = wf._revision_of(again, "name the run warm_jja")
        assert rev["user_answers"] == first["user_answers"]
        assert rev["changes"] == ["also report the elevation bands", "name the run warm_jja"]

    def test_the_change_at_the_prompt_goes_back_with_the_design(self, tmp_path,
                                                                monkeypatch):
        co, _ = _coordinator(tmp_path, interactive=True)
        reception = _warm_run()
        co._execute = lambda **kw: pytest.fail("a change request must not run")
        seen = {}

        def again(request, output_dir=None, revision=None):
            seen.update(request=request, revision=revision)
            return "REVISED"
        co.process_request = again
        monkeypatch.setattr("builtins.input", lambda *a: "also report the elevation bands")
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert out == "REVISED"
        assert seen["request"].endswith(
            "Revision requested at the design review: also report the elevation bands")
        params = seen["revision"]["settled"]["domain_experiment"]["runs"][0][
            "changes"][0]["params"]
        assert params["region"] == "watershed"
        assert seen["revision"]["user_answers"][0]["answer"] == "Over the watershed only, please."
        assert "revision requested" in (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_reception_reads_the_revision_in_its_context(self, tmp_path):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.conversation_context = {}
        co.default_output_dir = str(tmp_path)
        seen = {}

        class Reception:
            def process(self, user_request, context=None, run_dir=None):
                seen["context"] = context
                return {"route": {"action": "clarify", "questions": ["?"]}, "brief": {}}
        co.reception = Reception()
        co._workflow_clarification = lambda result: result
        rev = wf._revision_of(_warm_run(), "also report the elevation bands")
        result = co.process_request("q", output_dir=str(tmp_path), revision=rev)
        assert seen["context"]["revision_of"] == rev
        assert result["revision_of"] == rev
        co.process_request("q", output_dir=str(tmp_path))
        assert not (seen["context"] or {}).get("revision_of")

    def test_reception_is_told_to_keep_the_settled_decisions(self):
        text = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
        assert "revision_of" in text
        assert "Keep every decision already made with the user" in text
        assert "Apply the changes, and only them" in text
        assert "Do not ask again about anything already answered" in text


class TestThePageShowsEachChange:

    def test_the_question_and_each_change_are_shown_apart(self):
        reception, strategy = _site()
        reception["user_request"] = (
            "How does runoff vary with elevation in Testville?"
            + dr.REVISION_MARK + "report by elevation band"
            + dr.REVISION_MARK + "use the watershed only")
        page = dr.render(reception, strategy, "x")
        assert "> How does runoff vary with elevation in Testville?\n" in page
        assert "1. report by elevation band" in page
        assert "2. use the watershed only" in page

    def test_the_workflow_appends_changes_with_the_same_mark(self):
        src = (ROOT / "workflow.py").read_text()
        assert 'f"{original}{REVISION_MARK}{ans}"' in src

    def test_a_clause_cut_at_a_semicolon_is_marked_shortened(self):
        text = "a" * 150 + "; " + "b" * 100
        out = dr._trim(text, 200)
        assert out.endswith("…") and not out.endswith(";…") and ";" not in out[-2:]

    def test_a_sentence_cut_at_a_full_stop_keeps_it(self):
        out = dr._trim("a" * 150 + ". " + "b" * 100, 200)
        assert out.endswith(".")

    def test_the_control_alone_says_nothing_runs(self):
        reception = {"user_request": "q", "brief": {
            "design_archetype": "whole_domain", "model": "elm",
            "domain_experiment": {"control": {"name": "naches", "reuse": True},
                                  "runs": [], "years": [2015]}}}
        strategy = {"archetype": "whole_domain", "goals": ["g"],
                    "feasibility": {"verdict": "full", "why": "w"},
                    "domain_experiment": {"runs": []}}
        page = dr.render(reception, strategy, "x")
        assert "Nothing is run" in page and "chain of one-year jobs" not in page

    def test_reception_reads_the_place_in_the_question(self):
        text = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
        assert "A change that takes a region applies where the QUESTION says" in text


class TestThePageSaysMonthsAndSeasons:
    """MONTHS AND SEASONS (2026-11-03): the page a scientist approves said the
    numbers are by region and year only, while the server also reports each
    field by month and by season."""

    def test_the_field_table_is_followed_by_the_months_line(self, tmp_path):
        preview = {"analysis": {"fields": [{"name": "GPP", "stream": "h0",
                                            "how": "total over the year",
                                            "units": "gC/m^2/yr"}]}}
        L = dr._appendix(preview, tmp_path, {}, {"domain_experiment": {"years": [2015]}})
        text = "\n".join(L)
        assert "| GPP | h0 | total over the year | gC/m^2/yr |" in text
        assert dr.MONTHS_AND_SEASONS_LINE in text
        assert "needs the December before it" in dr.MONTHS_AND_SEASONS_LINE

    def test_the_control_alone_says_months_and_seasons(self):
        reception = {"user_request": "q", "brief": {
            "design_archetype": "whole_domain", "model": "elm",
            "domain_experiment": {"control": {"name": "naches", "reuse": True},
                                  "runs": [], "years": [2015]}}}
        strategy = {"archetype": "whole_domain", "goals": ["g"],
                    "feasibility": {"verdict": "full", "why": "w"},
                    "domain_experiment": {"runs": []}}
        page = dr.render(reception, strategy, "x")
        assert "by region, year, month and season" in page
        assert "by region and year" not in page

    def test_the_analyzers_task_says_months_and_seasons(self):
        from agents.analysis import step2_investigate as S2
        assert "reported by region, year, month and\nseason" in S2.DOMAIN_TASK
        assert "write the code that draws them" in S2.DOMAIN_TASK
