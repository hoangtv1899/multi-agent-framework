"""What this version answers is said before the first question (2026-11-10).

    pytest tests/test_scope_banner.py -v

The start screen listed the data servers and nothing about what could be
asked; the prepared run was introduced only in Reception's first reply. The
banner says it first, and every word about a run or a change in it is the
model server's own record. Every name, year and sentence here is made up. No
test reaches a language model or a real server.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core import scope_banner as SB                         # noqa: E402

CONTROL = {"name": "valley", "description": "the Valley control: one case over a made-up "
                                            "box, weather from a made-up record",
           "facts": {"output_years": [2031, 2036], "start_year": 2031},
           "not_in_it_words": ["the observed floods: the run makes its own"]}
CHANGES = [
    {"id": "pump_off", "summary": "switch the pump off", "available": True,
     "demonstrated": True},
    {"id": "more_rain", "summary": "multiply the rain", "available": True,
     "demonstrated": False},
    {"id": "new_dam", "summary": "build a dam", "available": False},
]


def _menus(controls=(CONTROL,), changes=CHANGES):
    return {"elm": {"controls": list(controls), "changes": list(changes),
                    "undemonstrated": "words"}}


class TestTheBanner:

    def test_it_says_the_run_its_years_and_its_changes(self):
        text = "\n".join(SB.lines(_menus()))
        assert SB.TITLE in text
        assert "Questions about this prepared run, and what-if runs made from it:" in text
        assert "valley (elm): the Valley control: one case over a made-up box" in text
        assert "output for 2031–2036" in text
        assert "A what-if run can: switch the pump off; multiply the rain*." in text
        assert "Not answered: another place or another model" in text
        assert "Not in valley: the observed floods: the run makes its own." in text

    def test_it_says_where_the_comparison_with_measurements_is_shown(self):
        """The banner says what a run is; its comparison with measurements is
        on the page a design is approved from, and only a record that holds
        one is said to have one."""
        where = "with measurements is on the design page, before you are asked to approve"
        assert where not in " ".join("\n".join(SB.lines(_menus())).split())
        held = dict(CONTROL, evaluation={"summary": "made-up words", "source": "s"})
        flat = " ".join("\n".join(SB.lines(_menus(controls=(held,)))).split())
        assert "How it compares " + where in flat and "made-up words" not in flat
        two = " ".join("\n".join(SB.lines(_menus(controls=(
            held, dict(CONTROL, name="ridge"))))).split())
        assert "How each compares " + where in two

    def test_the_runs_it_described_are_named(self):
        other = dict(CONTROL, name="ridge")
        assert SB.run_names(_menus(controls=(CONTROL, other))) == ["valley", "ridge"]
        assert SB.run_names(_menus(controls=())) == []
        assert SB.run_names({"": {"problems": ["elm: no answer"]}}) == []

    def test_a_change_that_cannot_be_used_is_not_offered(self):
        assert "build a dam" not in "\n".join(SB.lines(_menus()))

    def test_a_change_with_no_completed_run_is_marked_and_explained(self):
        text = "\n".join(SB.lines(_menus()))
        assert "multiply the rain*" in text and "switch the pump off*" not in text
        assert "* no run with this change has been completed here yet" in text

    def test_with_every_change_demonstrated_there_is_no_footnote(self):
        text = "\n".join(SB.lines(_menus(changes=CHANGES[:1])))
        assert "*" not in text

    def test_no_prepared_run_is_said_plainly(self):
        text = " ".join(" ".join(SB.lines(_menus(controls=()))).split())   # wrapped
        assert "No prepared run is registered on this machine" in text
        assert "A what-if run can" not in text

    def test_a_server_that_could_not_be_read_is_named(self):
        text = "\n".join(SB.lines({"": {"problems": ["elm: TimeoutError: no answer"]}}))
        assert "could not be read: elm: TimeoutError: no answer" in text

    def test_two_runs_are_both_listed(self):
        other = dict(CONTROL, name="ridge", description="the Ridge control")
        text = "\n".join(SB.lines(_menus(controls=(CONTROL, other))))
        assert "these prepared runs" in text and "ridge (elm): the Ridge control" in text

    def test_the_developers_switch_is_said(self):
        assert "developer's switch is set" in "\n".join(SB.lines(_menus(), developer=True))
        assert "developer's switch" not in "\n".join(SB.lines(_menus()))

    def test_no_line_is_wider_than_the_screen(self):
        long = dict(CONTROL, description="a control " + "described at length " * 30)
        assert max(len(ln) for ln in SB.lines(_menus(controls=(long,)))) <= SB.WIDTH

    def test_the_module_names_no_model_place_or_change(self):
        """Its words about a run or a change come from the server's records."""
        src = (ROOT / "src" / "core" / "scope_banner.py").read_text().lower()
        for word in ("naches", "elm-bgc", "pflotran", "fire_off", "co2", "weather_scale"):
            assert word not in src, word


class _Client:
    """A model server's two menus, as its tools return them."""
    def __init__(self, controls, changes, fail=False):
        self.controls, self.changes, self.fail = controls, changes, fail

    def list_tools_detailed(self):
        return [{"name": "describe_domain_controls"}, {"name": "describe_domain_changes"}]

    def list_tools(self):
        return ["describe_domain_controls", "describe_domain_changes"]

    def call_tool_json(self, tool, args):
        if self.fail:
            raise TimeoutError("no answer")
        if tool == "describe_domain_controls":
            return {"controls": self.controls, "registry": "r"}
        return {"changes": self.changes, "demonstrated_false": "words"}


class TestReadFromTheServer:

    def test_it_reads_the_two_menus_of_a_model_server(self):
        menus = SB.read({"elm": _Client([CONTROL], CHANGES), "terrain": object()})
        assert list(menus) == ["elm"]
        assert menus["elm"]["controls"] == [CONTROL] and menus["elm"]["changes"] == CHANGES

    def test_a_server_that_does_not_answer_is_a_problem_not_a_crash(self):
        menus = SB.read({"elm": _Client([CONTROL], CHANGES, fail=True)})
        assert "elm" not in menus and "TimeoutError" in menus[""]["problems"][0]
        assert "could not be read" in "\n".join(SB.lines(menus))

    def test_no_server_at_all(self):
        assert SB.read({}) == {}
        assert "No prepared run" in " ".join(SB.text({}, developer=False).split())


class TestTheCoordinatorSaysItOnce:

    def _coordinator(self, clients):
        import workflow as wf
        co = object.__new__(wf.WorkflowCoordinator)
        co.mcp_clients = clients
        return co

    def test_before_the_first_question_and_not_again(self, capsys):
        co = self._coordinator({"elm": _Client([CONTROL], CHANGES)})
        co._show_scope()
        first = capsys.readouterr().out
        assert SB.TITLE in first and "valley (elm)" in first
        co._show_scope()
        assert capsys.readouterr().out == ""

    def test_reception_is_told_which_runs_the_person_has_read_about(self, capsys):
        """The run is described once: Reception shows no card for a run the
        banner has just described."""
        class Front:
            introduced = set()
        co = self._coordinator({"elm": _Client([CONTROL], CHANGES)})
        co.reception = Front()
        co._show_scope()
        assert co.reception.introduced == {"valley"}
        # a banner that could not be printed introduces nothing
        co = self._coordinator({"elm": _Client([CONTROL], CHANGES, fail=True)})
        co.reception = Front()
        co._show_scope()
        assert co.reception.introduced == set()

    def test_a_request_shows_it_before_reception_starts(self, capsys, tmp_path):
        import workflow as wf
        co = self._coordinator({"elm": _Client([CONTROL], CHANGES)})
        co.default_output_dir = str(tmp_path)
        co._reception_context = lambda: None

        class Front:
            def process(self, user_request, context=None, run_dir=None):
                print("RECEPTION STARTS")
                return {"route": {"action": "clarify", "questions": ["which year?"]},
                        "brief": {}}
        co.reception = Front()
        wf.WorkflowCoordinator.process_request(co, "a question", output_dir=str(tmp_path))
        out = capsys.readouterr().out
        assert out.index(SB.TITLE) < out.index("RECEPTION STARTS")

    def test_a_banner_that_cannot_be_read_does_not_stop_the_request(self, capsys,
                                                                    monkeypatch):
        co = self._coordinator({"elm": _Client([CONTROL], CHANGES)})
        monkeypatch.setattr(SB, "text", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")))
        co._show_scope()                                    # must not raise
        assert "could not be read: RuntimeError: boom" in capsys.readouterr().out
