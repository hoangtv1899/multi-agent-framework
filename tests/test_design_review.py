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
        monkeypatch.setattr("builtins.input", lambda *a: "")
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "SENTINEL-EXECUTED" in out
        assert "accepted at the terminal" in \
            (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_no_consent_is_a_no(self, tmp_path, monkeypatch):
        co, reception = _coordinator(tmp_path, interactive=True)
        co._execute = lambda **kw: pytest.fail("EOF at the prompt must not run")
        def eof(*a):
            raise EOFError
        monkeypatch.setattr("builtins.input", eof)
        out = co._workflow_design_and_run(reception, str(tmp_path), tmp_path)
        assert "declined" in out.lower()
