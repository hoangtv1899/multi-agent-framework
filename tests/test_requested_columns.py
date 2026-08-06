"""A stated column count is a constraint, not a suggestion.

2026-08-06, live end to end: the request said "a small ELM study of 3 columns",
reception kept that text in brief.user_request, the planner designed 19, and
the strategy gate reported "strategy agrees with reception" — because every
column check it had asked whether 19 was PHYSICALLY possible (under
MAX_COLUMNS, fewer than the basin's 68 grid points), never whether it was what
was asked for. Nineteen columns built and ran: six times the compute, from a
number the user had stated outright.

The number now has to become DATA at reception
(run_settings.requested_n_columns, the treatment the period already got) so
that something downstream is able to enforce it.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from core.strategy_check import check

ROOT = Path(__file__).resolve().parents[1]


def _reception(n=None):
    rs = {"resolved_period": {"yr_start": 2020, "yr_end": 2020, "source": "user"}}
    if n is not None:
        rs["requested_n_columns"] = n
    return {"brief": {"domain": {"name": "Upper Gunnison",
                                 "bbox": [-107.5, 38.4, -106.5, 39.0]},
                      "run_settings": rs}}


def _strategy(n):
    return {"sampling": {"n_columns": n, "n_bands": 4},
            "grid": {"n_in_basin": 68}}


class TestAStatedCountWins:

    def test_the_strategy_is_corrected_to_what_was_asked(self):
        report, fixed = check(_reception(3), _strategy(19))
        assert report["ok"]
        assert fixed["sampling"]["n_columns"] == 3, \
            "the user asked for 3; 19 is six times the compute"

    def test_the_correction_is_recorded_not_silent(self):
        report, _ = check(_reception(3), _strategy(19))
        assert any("n_columns" in c for c in report["corrections"])
        assert any("3" in c and "19" in c for c in report["corrections"])

    def test_agreement_produces_no_correction(self):
        report, _ = check(_reception(5), _strategy(5))
        assert not [c for c in report["corrections"] if "n_columns" in c]

    def test_no_stated_count_leaves_the_planner_alone(self):
        """"A small study" is not a number. Absent a stated count the planner
        designs, which is its job."""
        report, fixed = check(_reception(None), _strategy(19))
        assert fixed["sampling"]["n_columns"] == 19
        assert not [c for c in report["corrections"] if "n_columns" in c]

    def test_the_physical_limits_still_apply(self):
        """Honouring the request must not let it exceed the basin."""
        report, _ = check(_reception(3), _strategy(9999))
        assert not report["ok"] and report["stop"]


class TestReceptionIsToldToExtractIt:

    def test_the_prompt_carries_the_field(self):
        """The gate can only enforce a number reception actually produced."""
        p = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
        assert "requested_n_columns" in p
        assert "run_settings.requested_n_columns" in p

    def test_the_prompt_forbids_inventing_one(self):
        p = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
        assert "Do NOT invent one" in p
