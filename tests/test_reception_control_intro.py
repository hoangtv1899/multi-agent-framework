"""Reception introduces the control before it settles anything.

    pytest tests/test_reception_control_intro.py -v

Demo A (2026-10-27) went from the question straight to the design page: the
scientist was never told what the run they were questioning was (its own fire
on, CO2 held fixed, a third of the measured snow). The control's entry now
carries how well it matches observations (`evaluation`, measured outside the
server and entered with its source), Reception's first message introduces the
control from its entry, and the design page shows the same note.
"""
import json
from pathlib import Path

from agents import design_review as DR

ROOT = Path(__file__).resolve().parents[1]
EVAL = {"summary": "It reproduces the 2015 drop in river flow. It holds a third of the "
                   "measured snow.",
        "source": "the hand analysis of 20 October 2026"}


def test_the_summary_reception_reads_carries_the_evaluation(tmp_path, monkeypatch):
    import domain_controls as D
    case, run = tmp_path / "case", tmp_path / "run"
    case.mkdir()
    run.mkdir()
    reg = tmp_path / "domain_controls.json"
    reg.write_text(json.dumps({"controls": {"synth": {
        "description": "a synthetic control", "case_dir": str(case), "run_dir": str(run),
        "evaluation": EVAL}}}))
    monkeypatch.setenv("IDEAS_DOMAIN_CONTROLS", str(reg))
    row = [r for r in D.describe()["controls"] if r["name"] == "synth"][0]
    assert row["evaluation"] == EVAL


def test_a_control_without_one_says_none():
    import domain_controls as D
    row = D.summary({"name": "x", "description": "d", "usable": True}, None)
    assert row["evaluation"] is None


def _page(row):
    de = {"control": {"name": "synth", "reuse": True}, "years": [2015], "runs": []}
    return "\n".join(DR._whole_domain_lines(
        {}, {"domain_experiment": de, "domain_rulebook": {"controls": {"controls": [row]}}}))


def test_the_design_page_shows_it_with_its_source():
    page = _page({"name": "synth", "description": "a synthetic control",
                  "evaluation": EVAL, "facts": {"output_years": [2015, 2021],
                                                "start_year": 2015}})
    assert ("- against observations: It reproduces the 2015 drop in river flow. It holds a "
            "third of the measured snow. (the hand analysis of 20 October 2026)") in page


def test_no_evaluation_no_line():
    assert "against observations" not in _page({"name": "synth", "description": "d"})


def _block():
    text = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
    return text[text.index("THEN ASK BEFORE YOU SETTLE ANYTHING"):
                text.index("TURN THE QUESTION INTO RUNS")]


def test_reception_asks_before_settling_and_does_not_repeat_the_card():
    block = _block()
    assert "When you fetch the controls" in block and "Do not repeat the card" in block
    assert "first `ask_user` message" in block
    assert "even when the question needs no run" in block
    assert "batch mode" in block and "the design page shows the card" in block


def test_what_the_control_does_not_contain_is_said_from_its_facts():
    import domain_controls as D
    fire = {"model": {"2020": 115.8, "2021": 132.4}, "observed": {"2020": 285.7, "2021": 343.8},
            "observed_source": "MODIS MCD64A1"}
    got = D.not_in_it({"burned_area_km2": fire})
    assert got == ["the observed fires: the run's fire is the model's own, lit and spread in "
                   "its own places and years, so a fire that happened is not in it; burned "
                   "area over the grid, the model against MODIS MCD64A1: 2020 116 against "
                   "286 km2; 2021 132 against 344 km2"]
    none = D.not_in_it({"burned_area_km2": dict(fire, model={"2020": 0.0, "2021": 0.0})})
    assert none[0].startswith("the observed fires: the run has no fire")
    assert D.not_in_it({"burned_area_km2": {"model": {"2021": 1.0}}}) == []
    assert D.not_in_it(None) == []
    row = D.summary({"name": "x", "description": "d", "usable": True},
                    {"burned_area_km2": fire, "settings": {}})
    assert row["not_in_it"] == got


def test_reception_names_what_the_run_does_not_contain():
    block = _block()
    assert "`not_in_it`" in block and "`evaluation`" in block
    assert "does not contain: its" in block and "effect cannot be read from the run" in block


def test_the_design_page_says_what_is_not_in_the_run():
    page = _page({"name": "synth", "description": "d",
                  "not_in_it": ["the observed fires: the run's fire is the model's own"]})
    assert "- not in this run: the observed fires: the run's fire is the model's own" in page


ROW = {"name": "synth", "usable": True, "description": "a synthetic control, fire model on",
       "evaluation": EVAL, "not_in_it": ["the observed fires: the run's fire is the model's own"]}


def test_the_card_quotes_the_servers_words_and_adds_nothing():
    from agents.reception_llm import control_card
    card = control_card([ROW])
    assert card.splitlines()[1:] == [
        "  ── The prepared run, in the model server's own words ──",
        "  synth: a synthetic control, fire model on",
        "  Against observations: " + EVAL["summary"],
        "  Not in this run: the observed fires: the run's fire is the model's own"]


def _agent(revision=False):
    from agents.reception_llm import LLMReceptionAgent
    a = object.__new__(LLMReceptionAgent)
    a._card_shown = bool(revision)
    return a


TRACE = [{"tool": "list_servers", "result": {}},
         {"tool": "describe_domain_controls",
          "result": {"server": "elm", "controls": {"controls": [ROW, dict(ROW, name="broken",
                                                                          usable=False)]}}}]


def test_the_card_is_shown_once_after_a_control_is_described():
    a = _agent()
    assert a._control_card(TRACE[:1]) is None          # nothing described yet
    card = a._control_card(TRACE)
    assert "synth: a synthetic control" in card and "broken" not in card
    assert a._control_card(TRACE) is None              # once only


def test_a_revision_does_not_show_it_again():
    assert _agent(revision=True)._control_card(TRACE) is None


def test_reading_the_controls_prints_the_card_once(monkeypatch, capsys):
    """The card appears when Reception reads the controls, whether or not it
    then asks a question: a live try asked nothing and the card never showed
    while it waited for a question."""
    import agents.reception_llm as R

    class FakeLoop:
        def __init__(self, **kw):
            self.local, self.tools = kw["local_tools"], []

    rows = {"server": "elm", "controls": {"controls": [ROW]}}
    monkeypatch.setattr(R, "ToolLoopAgent", FakeLoop)
    monkeypatch.setattr(R, "model_tools", lambda clients: {
        "describe_domain_controls": {"schema": {}, "fn": lambda **kw: rows}})
    a = R.LLMReceptionAgent(model="x", mcp_clients={})
    read = a.loop.local["describe_domain_controls"]["fn"]
    assert read(server="elm") == rows and read(server="elm") == rows
    out = capsys.readouterr().out
    assert out.count("The prepared run, in the model server's own words") == 1
    assert "Not in this run: the observed fires" in out


def test_a_revision_is_not_asked_to_confirm_again():
    """A REVIEW FINDING (2026-11-03): the revision block said do not re-ask,
    and STEP 2c said always ask, with nothing saying which wins."""
    block = " ".join(_block().split())
    assert ("In a revision (`revision_of` in the context), the card is not shown again "
            "and the user already confirmed the design: skip this question unless the "
            "new change itself is unclear.") in block


def test_process_shows_the_card_for_a_question_and_not_for_a_revision(monkeypatch, capsys):
    """A REVIEW FINDING (2026-11-03): the revision test set the flag by hand,
    so the reset in process() was never exercised."""
    import agents.reception_llm as R

    class FakeLoop:
        def __init__(self, **kw):
            self.local, self.tools = kw["local_tools"], []

        def run(self, system, msg):
            self.local["describe_domain_controls"]["fn"](server="elm")
            return {"content": '{"intent":"clarification_needed","questions":["q"]}',
                    "trace": [], "rounds": 1, "messages": []}

    rows = {"server": "elm", "controls": {"controls": [ROW]}}
    monkeypatch.setattr(R, "ToolLoopAgent", FakeLoop)
    monkeypatch.setattr(R, "model_tools", lambda clients: {
        "describe_domain_controls": {"schema": {}, "fn": lambda **kw: rows}})
    a = R.LLMReceptionAgent(model="x", mcp_clients={})
    heading = "The prepared run, in the model server's own words"

    a.process("q")
    assert capsys.readouterr().out.count(heading) == 1
    a.process("q", context={"revision_of": {"changes": ["x"]}})
    assert heading not in capsys.readouterr().out
    a.process("q")                                      # a new question shows it again
    assert capsys.readouterr().out.count(heading) == 1
