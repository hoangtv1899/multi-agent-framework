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
    return text[text.index("ASK BEFORE YOU SETTLE ANYTHING"):
                text.index("OUTPUT: when done")]


def test_reception_asks_before_settling_and_does_not_describe_the_run_again():
    block = " ".join(_block().split())
    assert "The framework has told them what the prepared run is" in block
    assert "Do not describe the run again" in block
    assert "Send ONE `ask_user` message before the brief" in block
    assert "even when the question needs no run" in block
    assert "batch mode" in block and "the design page shows the control's whole record" in block


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
    block = " ".join(_block().split())
    # the comparison with measurements is not shown to the model any more, so
    # its prompt does not name the field (controls_for_the_llm)
    assert "`not_in_it`" in block and "`evaluation`" not in block
    assert ("that the run does not contain (`not_in_it`): its effect cannot be read from "
            "the run") in block
    # the card no longer says it, so the prompt says where the person reads it
    assert "is printed on the design page, before they approve anything" in block


def test_the_design_page_says_what_is_not_in_the_run():
    page = _page({"name": "synth", "description": "d",
                  "not_in_it": ["the observed fires: the run's fire is the model's own"]})
    assert "- not in this run: the observed fires: the run's fire is the model's own" in page


ROW = {"name": "synth", "usable": True, "description": "a synthetic control, fire model on",
       "evaluation": EVAL, "not_in_it": ["the observed fires: the run's fire is the model's own"]}


def test_the_card_says_what_the_run_is_and_where_the_rest_is_shown():
    """IDENTITY ONLY (the user's decision of 2026-11-11): the comparison with
    measurements, the baseline years and what the run lacks were a paragraph
    of numbers ahead of any answer. They are on the design page, before the
    yes, and the card says so, naming only what the record holds."""
    from agents.reception_llm import control_card
    card = control_card([dict(ROW, normal_years=[2016, 2017])])
    assert card.splitlines()[1:3] == ["  ── The prepared run ──",
                                      "  synth: a synthetic control, fire model on"]
    flat = " ".join(card.split())
    assert flat.endswith("How it compares with measurements and what it does not contain are "
                         "on the design page, before you are asked to approve anything.")
    for gone in ("Against observations", EVAL["summary"], "Not in this run", "observed fires",
                 "normal years", "2016", "Fire:", "Grid:"):
        assert gone not in card, gone
    assert max(len(x) for x in card.splitlines()) <= 98
    # only what the record holds is promised
    only_eval = " ".join(control_card([dict(ROW, not_in_it=[])]).split())
    assert only_eval.endswith("How it compares with measurements is on the design page, "
                              "before you are asked to approve anything.")
    bare = control_card([{"name": "synth", "usable": True, "description": "a control"}])
    assert bare.splitlines()[1:] == ["  ── The prepared run ──", "  synth: a control"]


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
    for _ in range(2):
        got = read(server="elm")["controls"]["controls"][0]
        assert got["name"] == "synth" and got["not_in_it"] == ROW["not_in_it"]
        # the model reads its own copy, without the comparison with measurements
        assert got["evaluation"] == R.SHOWN_ON_THE_DESIGN_PAGE
    assert rows["controls"]["controls"][0]["evaluation"] == EVAL    # the record is whole
    out = capsys.readouterr().out
    assert out.count("── The prepared run ──") == 1
    assert "on the design page" in out and "Not in this run" not in out


def test_a_run_the_start_banner_introduced_gets_no_card(monkeypatch, capsys):
    """THE RUN IS DESCRIBED ONCE (2026-11-11): the coordinator's banner says
    what each prepared run is before the first question, and the card under
    it said the same a few lines later."""
    a = _agent()
    a.introduced = {"synth"}
    assert a._control_card(TRACE) is None
    a.introduced = {"another"}                         # a run the banner did not name
    assert "synth: a synthetic control" in a._control_card(TRACE)

    import agents.reception_llm as R

    class FakeLoop:
        def __init__(self, **kw):
            self.local, self.tools = kw["local_tools"], []

    rows = {"server": "elm", "controls": {"controls": [ROW]}}
    monkeypatch.setattr(R, "ToolLoopAgent", FakeLoop)
    monkeypatch.setattr(R, "model_tools", lambda clients: {
        "describe_domain_controls": {"schema": {}, "fn": lambda **kw: rows}})
    b = R.LLMReceptionAgent(model="x", mcp_clients={})
    assert b.introduced == set()                       # alone, Reception shows the card
    b.introduced = {"synth"}
    b.loop.local["describe_domain_controls"]["fn"](server="elm")
    assert "The prepared run" not in capsys.readouterr().out


def test_the_model_is_not_shown_the_runs_results_or_its_comparison():
    """Handed the comparison with measurements and the numbers by year,
    Reception quoted them and then judged them, against its prompt. Its copy
    of the record has neither; a note says where the person sees them."""
    from agents.reception_llm import (controls_for_the_llm, REPORTED_BY_THE_ANALYSIS,
                                      SHOWN_ON_THE_DESIGN_PAGE as NOTE)
    full = {"name": "synth", "usable": True, "description": "a control", "evaluation": EVAL,
            "normal_years": [2031, 2032], "not_in_it": ["the observed floods"],
            "run_names_taken": ["wet"],
            "facts": {"start_year": 2031, "output_years": [2031, 2036],
                      "settings": {"co2_ppmv": 400.0},
                      "per_year": {"2031": {"runoff_mm": 123.4}},
                      "burned_area_km2": {"model": {"2031": 5.0}}}}
    answer = {"server": "elm", "offers_whole_domain_experiments": True,
              "controls": {"registry": "r", "controls": [full, {"name": "bare",
                                                                "evaluation": None,
                                                                "facts": None}]}}
    before = json.dumps(answer, sort_keys=True)
    got = controls_for_the_llm(answer)
    assert json.dumps(answer, sort_keys=True) == before          # a copy
    row, bare = got["controls"]["controls"]
    assert row["evaluation"] == NOTE
    assert row["facts"]["per_year"] == REPORTED_BY_THE_ANALYSIS
    text = json.dumps(got)
    for gone in (EVAL["summary"], "123.4", "runoff_mm"):
        assert gone not in text, gone
    # everything a design is settled from is still there
    for key in ("name", "description", "usable", "normal_years", "not_in_it",
                "run_names_taken"):
        assert row[key] == full[key], key
    for key in ("start_year", "output_years", "settings", "burned_area_km2"):
        assert row["facts"][key] == full["facts"][key], key
    assert got["server"] == "elm" and got["controls"]["registry"] == "r"
    # a record that holds none is not said to hold one
    assert bare == {"name": "bare", "evaluation": None, "facts": None}
    # an answer of another shape (a server that could not be read) is passed on
    for other in ({"server": "elm", "offers_whole_domain_experiments": False, "why": "x"},
                  {"error": "no server"}, None, "text"):
        assert controls_for_the_llm(other) == other
    # each note says who does report what was left out, and it is true: the
    # design page prints the comparison with measurements and not the numbers
    # by year, which are the analysis's to report
    assert "design page" in NOTE and "Say nothing about it" in NOTE
    assert "design page" not in REPORTED_BY_THE_ANALYSIS
    assert "the analysis reports results" in REPORTED_BY_THE_ANALYSIS
    from agents import design_review
    page_code = Path(design_review.__file__).read_text()
    assert '"evaluation"' in page_code and "per_year" not in page_code


def test_a_revision_is_not_asked_to_confirm_again():
    """A REVIEW FINDING (2026-11-03): the revision block said do not re-ask,
    and STEP 2c said always ask, with nothing saying which wins."""
    block = " ".join(_block().split())
    assert ("In a revision (`revision_of` in the context) the design was already "
            "confirmed: skip the message unless the new change itself is unclear.") in block


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
    heading = "── The prepared run ──"

    a.process("q")
    assert capsys.readouterr().out.count(heading) == 1
    a.process("q", context={"revision_of": {"changes": ["x"]}})
    assert heading not in capsys.readouterr().out
    a.process("q")                                      # a new question shows it again
    assert capsys.readouterr().out.count(heading) == 1


# THE CONTROL'S OWN NORMAL YEARS (2026-11-10): entered in the registry as the
# evaluation is, carried by the server's summary and said on the design page,
# so a study's baseline does not change from one asking to the next. The card
# no longer prints them (2026-11-11): read before any design, "normal years"
# was taken for a statement about the climate.

def test_the_summary_carries_the_controls_own_normal_years():
    import domain_controls as D
    row = D.summary({"name": "x", "description": "d", "usable": True,
                     "normal_years": [2016, 2017], "normal_years_why": "its evaluation's"}, None)
    assert row["normal_years"] == [2016, 2017] and row["normal_years_why"] == "its evaluation's"
    assert D.summary({"name": "x", "description": "d"}, None)["normal_years"] is None


def test_the_design_page_says_whose_baseline_the_design_uses():
    from agents import design_review as DR

    def lines(design_years, own):
        strategy = {"domain_experiment": {
            "control": {"name": "synth", "reuse": True}, "runs": [], "years": [2015, 2016, 2017],
            "analysis": {"normal_years": design_years}},
            "domain_rulebook": {"controls": {"controls": [dict(ROW, normal_years=own)]}}}
        return "\n".join(DR._whole_domain_lines({}, strategy))
    same, other, none = lines([2016, 2017], [2016, 2017]), lines([2016], [2016, 2017]), \
        lines([2016], None)
    assert "normal years [2016, 2017] (the control's own)" in same
    assert "normal years [2016]; the control's own normal years are [2016, 2017]" in other
    assert "normal years [2016]\n" in none + "\n"
