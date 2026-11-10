"""The Analyzer's access to the suite, its recorded analysis plan and the
lookups log (package E), on a synthetic suite, with no model call.

    pytest tests/test_domain_access.py -v

THE THREE SAFEGUARDS, EACH HELD HERE (2026-11-05). The method contracts: a
script reaches the methods only through `suite`, and a setting neither the
contract nor the plan (nor a declared amendment) names is refused. The
analysis plan: step 2a's inputs hold no outcome-bearing table, step 3's review
is recorded once and tied to the plan's hash, and an amendment is appended,
time-stamped and never planned. The lookups log: the script runner refuses a
lookup before the review, binds no outcome table before it, and logs every
lookup after it with a hash of what came back. Every model call (step 2a, step
3's review, the Planner) is a scripted fake.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import domain_access as DAC
import domain_methods as M
import domain_suite as S
from test_domain_suite import FIELDS, make_control

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import script_runner as SR          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2015, 2016]
MARK = 987654.321                       # an outcome no plan input may carry
REGION = "vegetated cells"
PLAN_REPLY = {
    "comparisons": [{"id": "C1", "what": "2016 against the normal year", "run": "control",
                     "against": "normal years", "years": [2016]}],
    "baseline": {"normal_years": [2015], "why": "the Planner's"},
    "analyses": [{"id": "P1", "question": "GPP in 2016", "fields": ["GPP"],
                  "regions": [REGION], "periods": ["2016"], "methods": ["anomaly"],
                  "settings": {}, "why": "what was asked",
                  # its required outputs, apart from its scope (2026-11-08)
                  "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP",
                               "period": "2016", "region": REGION}]}],
    "expectations": [],
    "competing": [{"text": "soil water", "fields": ["H2OSOI"], "analysis": "P1"}],
    "screen": {"when": "first", "region": REGION, "year": 2016, "why": "exploratory"},
    "question_type": "hypothesis"}       # the model's word on the type is ignored
# A PLAN THAT DECLARES EVERY CALL THE TESTS BELOW MAKE (closing package E): the
# runner now refuses any lookup the reviewed plan does not declare.
PLAN_ALL = dict(PLAN_REPLY, analyses=[
    {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
     "periods": ["2016"], "steps": ["month"], "methods": ["anomaly", "difference"],
     "settings": {}, "why": "what was asked",
     "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                  "region": REGION}]},
    {"id": "P2", "question": "the maps", "fields": ["GPP", "TOTECOSYSC"], "regions": [],
     "periods": ["2016", "normal"], "methods": [], "settings": {}, "why": "to draw them",
     "outputs": [{"id": "O1", "kind": "map", "field": "GPP", "period": "2016"},
                 {"id": "O2", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
    {"id": "P3", "question": "how the maps go together", "fields": [], "regions": [REGION],
     "periods": [], "methods": ["overlap", "elevation_split"], "settings": {}, "why": "where",
     "outputs": [{"id": "O1", "kind": "overlap", "region": REGION,
                  "inputs": ["P2.O1", "P2.O2"]},
                 {"id": "O2", "kind": "elevation_split", "region": REGION,
                  "inputs": ["P2.O1"]}]},
    {"id": "P4", "question": "the season's dates", "fields": [], "regions": [REGION],
     "periods": ["2016"], "methods": ["photosynthesis_onset", "growth_peak"], "settings": {},
     "why": "when",
     "outputs": [{"id": "O1", "kind": "photosynthesis_onset", "region": REGION,
                  "period": "2016"},
                 {"id": "O2", "kind": "growth_peak", "region": REGION, "period": "2016"}]}])


class Fake:
    """A scripted model: one reply, every prompt kept."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def ask(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


class Ctx:
    """The parts of an AnalysisContext the Analyzer's steps read."""
    is_domain = True

    def __init__(self, run_dir, plan, frames=None, strict=False):
        self.run_dir, self.plan = Path(run_dir), plan
        self.caveats, self.columns = [], []
        self.data = {"kind": "whole_domain", "variable_units": {},
                     "regional": {"years": YEARS, "outcome": MARK}}
        self._frames, self._strict, self.read = frames or {}, strict, []

    def series(self):
        return None

    def profiles(self):
        return None

    def soil(self):
        return None

    def _f(self, name):
        self.read.append(name)
        if self._strict:
            raise AssertionError(f"{name} was read")
        return self._frames.get(name)

    def regional(self):
        return self._f("regional")

    def regional_monthly(self):
        return self._f("regional_monthly")

    def regional_seasonal(self):
        return self._f("regional_seasonal")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("access")
    control, _ = make_control(root / "ctl", YEARS)
    suite = root / "suite"
    S.build(control, suite, fields=FIELDS)
    return {"control": control, "suite": suite}


@pytest.fixture()
def study(built, tmp_path, monkeypatch):
    entry = dict(built["control"], suite_dir=str(built["suite"]))
    monkeypatch.setattr(AP, "_registry_entry", lambda name: entry if name == "synth" else {})
    rd = tmp_path / "study"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": built["control"]["run_dir"]}}))
    plan = {"question": "What happened to GPP in 2016?", "goals": ["GPP in 2016"],
            "experiment": {"control": {"name": "synth", "reuse": True}, "runs": [],
                           "years": YEARS,
                           "analysis": {"normal_years": [2015], "question_type": "exploratory",
                                        "regions": [{"name": REGION}], "variables": ["GPP"]}}}
    frames = {"regional": pd.DataFrame({"run": ["control"] * 3, "value": [MARK] * 3})}
    out = rd / "04_analysis"
    out.mkdir()
    # a code-only report with an outcome, as a whole-domain study has: never step 2a's
    (out / "DOMAIN_REPORT.md").write_text(f"GPP fell by {MARK} gC/m^2\n")
    return {"ctx": Ctx(rd, plan, frames), "rd": rd, "out": out, "suite": built["suite"],
            "plan": plan}


def _approve(study, follows=True, reply=PLAN_REPLY):
    AP.write_plan(study["ctx"], study["out"], client=Fake(reply))
    return S3.rule_plan_follows_question(study["ctx"], study["out"],
                                         client=Fake({"follows": follows, "why": "it asks "
                                                      "what was asked", "problems": []}))


def _log(study):
    return AP.lookups(study["out"])


# ── which studies ────────────────────────────────────────────────────────────

def test_only_a_suite_bearing_study_takes_the_new_path(study, monkeypatch):
    ctx = study["ctx"]
    assert AP.suite_dirs(ctx) == {"control": str(study["suite"])}
    # a suite of other output is not this study's: the run directories differ
    (study["rd"] / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": "/elsewhere/run"}}))
    assert AP.suite_dirs(ctx) == {} and AP.access_spec(ctx) is None
    assert "suite_spec" not in SR._payload(ctx)
    assert S2.task_for(ctx) == S2.DOMAIN_TASK
    # not a whole-domain study at all: the column path, untouched
    col = Ctx(study["rd"], study["plan"])
    col.is_domain = False
    assert AP.suite_dirs(col) == {} and S2.task_for(col) == S2.TASK


# ── safeguard 2: the plan ────────────────────────────────────────────────────

def test_step_2a_payload_holds_no_outcome_bearing_table(study):
    ctx = Ctx(study["rd"], study["plan"], strict=True)       # reading a table fails
    inputs = AP.plan_inputs(ctx)
    # "recipe" (2026-11-09): guidance read from the question's words alone,
    # which holds no outcome; none for this question
    assert set(inputs) == {"question", "question_type", "question_type_from",
                           "configuration", "catalog", "methods", "recipe"}
    assert inputs["recipe"] == ""
    blob = json.dumps(inputs, default=str)
    assert str(MARK) not in blob and ctx.read == []
    for name in AP.OUTCOME_TABLES + ("DOMAIN_REPORT", "comparison", "observed"):
        assert name not in inputs and name not in inputs["configuration"], name
    fake = Fake(PLAN_REPLY)
    plan = AP.write_plan(ctx, study["out"], client=fake)
    assert ctx.read == [] and len(fake.prompts) == 1 and str(MARK) not in fake.prompts[0]
    assert "THE DATA CATALOG" in fake.prompts[0] and "THE METHODS" in fake.prompts[0]
    # the type is the Planner's, never the model's; the record says what was seen and withheld
    assert plan["question_type"] == "exploratory" and plan["question_type_from"] == "the Planner"
    assert plan["withheld"] == list(AP.WITHHELD) and plan["seen"] == list(AP.SEEN)
    assert plan["written_at"].endswith("+00:00") and plan["problems"] == []
    saved = json.loads((study["out"] / AP.PLAN_FILE).read_text())
    assert saved == plan
    # written once: a second call returns it and asks nothing
    again = Fake(PLAN_REPLY)
    assert AP.write_plan(ctx, study["out"], client=again) == plan and again.prompts == []


def test_the_plan_review_is_recorded_once_and_tied_to_the_plan(study):
    assert AP.plan_passed(study["out"])[0] is False
    rv = _approve(study)
    assert rv["passed"] is True and rv["rule"] == AP.RULE == "rule_plan_follows_question"
    assert rv["plan_sha256"] == AP.plan_sha256(study["out"])
    assert AP.plan_passed(study["out"]) == (True, "the analysis plan passed step 3's review")
    later = Fake({"follows": False})
    assert S3.rule_plan_follows_question(study["ctx"], study["out"], client=later) == rv
    assert later.prompts == []
    # a plan changed after its review is no longer the plan that passed
    p = study["out"] / AP.PLAN_FILE
    p.write_text(p.read_text().replace("GPP in 2016", "GPP in 2015"))
    ok, why = AP.plan_passed(study["out"])
    assert not ok and "changed after its review" in why


def test_code_problems_fail_the_review_whatever_the_model_says(study):
    bad = dict(PLAN_REPLY, analyses=[dict(PLAN_REPLY["analyses"][0], fields=["NOT_A_FIELD"],
                                          methods=["guess"])])
    plan = AP.write_plan(study["ctx"], study["out"], client=Fake(bad))
    assert any("NOT_A_FIELD" in p for p in plan["problems"])
    assert any("no method 'guess'" in p for p in plan["problems"])
    rv = S3.rule_plan_follows_question(study["ctx"], study["out"],
                                       client=Fake({"follows": True, "why": "fine"}))
    assert rv["passed"] is False and any(p.startswith("code: ") for p in rv["problems"])


def test_an_amendment_is_appended_time_stamped_and_never_planned(study):
    with pytest.raises(AP.PlanNotApproved):
        AP.append_amendment(study["out"], "too early", {"fields": ["GPP"]})
    _approve(study)
    spec = AP.access_spec(study["ctx"])
    acc = AP.make_access(spec, script="fig_a")
    acc.screen(2016, REGION)                         # an exploratory question: the screen first
    acc.anomaly("GPP", "2016", region=REGION)                        # planned
    a1 = AP.append_amendment(study["out"], "seen a dip; check the stock",
                             {"fields": ["TOTECOSYSC"], "methods": ["anomaly"],
                              "settings": {"anomaly": {"near_zero_share": 0.05}}})
    first = (study["out"] / AP.AMENDMENTS_FILE).read_text()
    a2 = AP.append_amendment(study["out"], "and the soil water", {"fields": ["H2OSOI"]})
    text = (study["out"] / AP.AMENDMENTS_FILE).read_text()
    assert text.startswith(first) and len(text.splitlines()) == 2      # appended, not rewritten
    assert (a1["id"], a2["id"]) == ("A1", "A2") and a1["planned"] is a2["planned"] is False
    assert a1["at"].endswith("+00:00") and a1["after_lookups"] == 2
    am = AP.make_access(spec, script="fig_b", amendment="A1")
    am.anomaly("TOTECOSYSC", "2016", region=REGION, settings={"near_zero_share": 0.05})
    led = AP.lookups_of(study["out"], "A1")
    assert len(led) == 1 and led[0]["planned"] is False and led[0]["script"] == "fig_b"
    assert led[0]["settings"] == {"near_zero_share": repr(0.05)}
    log = _log(study)
    assert log[0]["plan_items"] == ["screen"] and log[0]["label"] == AP.EXPLORATORY_LABEL
    assert log[1]["planned"] is True and log[1]["plan_items"] == ["P1"]


# ── safeguard 1: the contracts ───────────────────────────────────────────────

def test_a_setting_neither_the_contract_nor_the_plan_names_is_refused(study):
    _approve(study, reply=PLAN_ALL)
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="fig")
    acc.screen(2016, REGION)
    with pytest.raises(DAC.AccessRefused, match="setting anomaly.near_zero_share=0.05"):
        acc.anomaly("GPP", "2016", region=REGION, settings={"near_zero_share": 0.05})
    with pytest.raises(DAC.AccessRefused, match="window=9 is not a setting timing.nc holds"):
        acc.growth_peak(2016, REGION, settings={"window": 9})
    # the default and a contract's alternative run
    assert acc.anomaly("GPP", "2016", region=REGION, settings={"near_zero_share": 0.01})
    assert acc.growth_peak(2016, REGION, settings={"window": 15})["method"] == "growth_peak"
    log = _log(study)[1:]
    assert [("refused" in r) for r in log] == [True, True, False, False]
    # settings_for stays the second line: a value the plan names is accepted, no other
    named = AP.named_settings({"analyses": [{"settings": {"anomaly": {"near_zero_share": 0.05}}}]})
    assert M.settings_for("anomaly", {"near_zero_share": 0.05}, named["anomaly"])
    with pytest.raises(M.MethodError, match="neither its default"):
        M.settings_for("anomaly", {"near_zero_share": 0.07}, named["anomaly"])


def test_every_method_is_a_gated_logged_call(study):
    _approve(study, reply=PLAN_ALL)
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="fig")
    scr = acc.screen(2016, REGION)                   # an exploratory question: the screen first
    x = acc.get_map("control", "GPP", "2016")
    y = acc.get_map("control", "TOTECOSYSC", "2016")
    n = acc.get_map("control", "GPP", "normal")
    got = {"anomaly": acc.anomaly("GPP", "2016", region=REGION),
           "difference": acc.difference("control", "GPP", "2016", region=REGION),
           "screen": scr,
           "overlap": acc.overlap(x - n, y, REGION),
           "elevation_split": acc.elevation_split(x - n, REGION),
           "photosynthesis_onset": acc.photosynthesis_onset(2016, REGION),
           "growth_peak": acc.growth_peak(2016, REGION, against="normal")}
    for mid, r in got.items():
        assert r["method"] == mid and r["version"] == M.METHODS[mid].version, mid
    assert set(got) == {m for m, ts in M.ANALYZER_ACCESS.items()
                        if any(t.startswith("tests/test_domain_access.py::") for t in ts)}
    ser = acc.get_series("control", "GPP", REGION, "month")
    assert len(ser["labels"]) == 24 and ser["values"].shape == (24,)
    tim = acc.get_timing("control", "growth_peak", 2016)
    assert "default" in tim and tim["default"]["date"].shape == x.ravel().shape
    assert acc.region_mask(REGION).shape == x.shape and acc.cell_area().shape == x.shape
    assert any(f["name"] == "GPP" and f["in_suite"] for f in acc.find_fields("gpp"))
    # a month's normal: the plan cannot declare it (no month maps), so the gate refuses it
    with pytest.raises(DAC.AccessRefused, match="period 'normal-07'"):
        acc.get_map("control", "GPP", "normal-07")
    log = _log(study)
    answered = [r for r in log if "sha256" in r]
    assert len(log) == 16 and len(answered) == 15
    # the hash is what came back: recomputed apart from the access path
    first = answered[1]
    assert first["call"] == "get_map" and first["keys"] == {"run": "control", "field": "GPP",
                                                            "period": "2016"}
    assert first["sha256"] == DAC.fingerprint(M.Suite(study["suite"]).map("GPP", "2016"))
    for r in log:
        assert r["at"] and r["script"] == "fig" and "call" in r
    assert {r["call"] for r in log if r.get("support")} == {"region_mask", "cell_area",
                                                           "find_fields"}


# ── safeguard 3: the runner ──────────────────────────────────────────────────

CODE = '''
suite.screen(2016, "vegetated cells")         # an exploratory question: the screen first
x = suite.get_map("control", "GPP", "2016")
draw_map(x, out_path=out_path, units="gC/m^2/yr")
result = {"n": int(np.isfinite(x).sum())}
'''


def test_the_runner_refuses_a_lookup_before_the_review_and_logs_after(study):
    ctx, out = study["ctx"], study["out"]
    AP.write_plan(ctx, out, client=Fake(PLAN_REPLY))
    pay = SR._payload(ctx)
    assert pay["suite_spec"]["suites"] == {"control": str(study["suite"])}
    assert not any(k in pay for k in AP.OUTCOME_TABLES)          # nothing bound before
    before = SR.run(CODE, ctx, out / "fig.png", script_path=out / "fig.py",
                    lookup_tag={"script": "fig", "amendment": None})
    assert before["ok"] is False and "step 3's review" in before["error"]
    assert not (out / "fig.png").exists()
    log = _log(study)
    assert len(log) == 1 and "refused" in log[0] and "sha256" not in log[0]
    S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "yes"}))
    # nor after (package F1 closed the bypass): only the suite's logged calls
    assert not any(k in SR._payload(ctx) for k in AP.OUTCOME_TABLES)
    after = SR.run(CODE, ctx, out / "fig.png", script_path=out / "fig.py",
                   lookup_tag={"script": "fig", "amendment": None})
    assert after["ok"] is True, after.get("error")
    assert after["result"]["n"] >= 3
    log = _log(study)[1:]
    assert [r["call"] for r in log] == ["screen", "get_map", "lat_lon"]
    assert log[1]["sha256"] == DAC.fingerprint(M.Suite(study["suite"]).map("GPP", "2016"))
    assert log[1]["planned"] is True and log[1]["script"] == "fig"


def test_step_2_proposes_nothing_before_the_plan_passes(study):
    fake = Fake({"figures": []})
    inv = S2.investigate(study["ctx"], study["out"], client=fake)
    assert fake.prompts == [] and inv["n_proposed"] == 0
    assert inv["caveats"][0]["id"] == "analysis_plan_not_approved"
    _approve(study, follows=False)
    inv = S2.investigate(study["ctx"], study["out"], client=fake)
    assert fake.prompts == [] and "did not pass" in inv["caveats"][0]["statement"]


def test_step_2b_reads_the_catalog_the_compact_methods_and_the_plan(study):
    _approve(study)
    text = S2.context_brief(study["ctx"])
    assert "THE RECORDED ANALYSIS PLAN" in text and "## 6. How to load" in text
    assert "# Analysis methods: the compact view" in text
    assert S2.task_for(study["ctx"]) == S2.SUITE_TASK
    assert "VARIABLES IN `reg`" not in text            # domain_brief's table is replaced


# ── the question type ────────────────────────────────────────────────────────

def test_the_question_type_is_the_planners_and_its_rule_is_written(study, monkeypatch):
    rule = (ROOT / "src" / "agents" / "prompts" / "planner.txt").read_text()
    assert '"hypothesis" only when the question proposes a testable' in rule
    assert 'Naming a\n                 field alone is not a hypothesis: "what happened to GPP?" is\n' \
           '                 exploratory.' in rule
    assert '"question_type": "hypothesis" | "exploratory"' in rule
    reception = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
    assert "Both kinds of question are answerable" in reception
    # the Planner, scripted: its question_type reaches the analysis the Analyzer reads
    import agents.planner as P
    reply = {"domain_experiment": {"analysis": {"normal_years": [2015],
                                                "question_type": "hypothesis"}}}
    monkeypatch.setattr(P, "SimpleLLMClient", lambda model=None: Fake(reply))
    out = P.Planner().plan({"brief": {}})
    assert out["domain_experiment"]["analysis"]["question_type"] == "hypothesis"
    ctx = study["ctx"]
    assert AP.question_type(ctx) == {"type": "exploratory", "from": "the Planner"}
    ctx.plan["experiment"]["analysis"]["question_type"] = "hypothesis"
    assert AP.question_type(ctx)["type"] == "hypothesis"
    del ctx.plan["experiment"]["analysis"]["question_type"]
    qt = AP.question_type(ctx)
    assert qt["type"] == "exploratory" and "recorded none" in qt["from"]


def test_the_design_review_shows_the_question_type():
    from agents import design_review as DR
    strat = {"domain_experiment": {"control": {"name": "synth", "reuse": True},
                                   "analysis": {"normal_years": [2015],
                                                "question_type": "exploratory"}}}
    lines = DR._whole_domain_lines({}, strat)
    assert any(x.startswith("- question type: exploratory (the question asks what changed")
               for x in lines)
    assert DR.question_type_line({"question_type": "hypothesis"}).startswith(
        "question type: hypothesis (the question proposes an expectation to test")
    assert "read it as exploratory" in DR.question_type_line({"question_type": "maybe"})
    assert not any("question type" in x for x in DR._whole_domain_lines(
        {}, {"domain_experiment": {"control": {"name": "synth"}, "analysis": {}}}))


def test_steps_2a_and_the_review_run_once_before_the_rounds(study, monkeypatch):
    calls = []
    monkeypatch.setattr(AP, "write_plan", lambda ctx, out, model=None: calls.append("2a"))

    def review(ctx, out, model=None):
        calls.append("review")
        return {"passed": False}
    monkeypatch.setattr(S3, "rule_plan_follows_question", review)
    monkeypatch.setattr(S2, "investigate", lambda *a, **k: calls.append("2b") or
                        {"findings": [], "n_succeeded": 0, "n_proposed": 0})
    monkeypatch.setattr(S3, "interpret", lambda *a, **k: calls.append("3") or
                        {"verdict": "insufficient", "feedback": "more",
                         "audit": {"n_claims": 0, "n_struck": 0}})
    loop = S3.investigate_and_interpret(study["ctx"], study["out"], max_rounds=2)
    # a plan that did not pass makes one round, and step 2 says why; step 3 is
    # not asked (the user's decision on G part 2, 2026-11-08: nothing was
    # looked up, so there is nothing to interpret and the call is saved)
    assert calls == ["2a", "review", "2b"] and loop["n_rounds"] == 1
    assert loop["plan_review"] == {"passed": False}
    assert loop["interpretation"]["skipped"] is True
    assert loop["interpretation"]["skipped_because"].startswith("there is no analysis plan")


# ── package F1, step 1: no whole outcome table for a study with a suite ─────
#
# Before F1 the runner bound reg, reg_monthly and reg_seasonal whole once the
# plan had passed its review, so a script could read any value past the
# scoped, logged suite calls. Now a study with a suite never gets them, before
# or after the review; a study without one keeps its payload byte for byte.

READS_REG = '''
x = reg[reg["variable"] == "GPP"]["value"]
result = {"n": 3, "x": float(x.sum())}
'''


def _old_payload(ctx):
    """The payload a study without a suite got before package F1, built
    here from the same objects in the same order, apart from the runner."""
    t = (getattr(ctx, "data", None) or {}).get("regional") or {}
    exp = (getattr(ctx, "plan", None) or {}).get("experiment") or {}
    df, conv = SR._to_daily_rates(ctx.series(), (ctx.data or {}).get("variable_units"))
    return {"df": df, "prof": ctx.profiles(), "soil": ctx.soil(),
            "columns": [{k: v for k, v in c.items() if k != "variables"}
                        for c in ctx.columns],
            "caveats": list(ctx.caveats or []), "reg": ctx.regional(),
            "reg_monthly": ctx.regional_monthly(), "reg_seasonal": ctx.regional_seasonal(),
            "regions": dict(t.get("regions") or {}) if isinstance(t, dict) else {},
            "runs": [{k: r.get(k) for k in ("run", "reused", "status", "changes")}
                     for r in (exp.get("runs") or []) if isinstance(r, dict)],
            "converted_to_daily": conv}


def test_a_study_without_a_suite_keeps_its_payload_byte_for_byte(study):
    import pickle
    ctx = study["ctx"]
    ctx._frames.update(regional_monthly=pd.DataFrame({"m": [1.0]}),
                       regional_seasonal=pd.DataFrame({"s": [2.0]}))
    ctx.data["regional"]["regions"] = {REGION: {"cells": 3}}
    ctx.columns = [{"name": "c1", "variables": {"GPP": [1.0]}, "metrics": {"a": 1}}]
    # a whole-domain study whose suite is of other output: no suite
    (study["rd"] / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": "/elsewhere/run"}}))
    assert AP.access_spec(ctx) is None
    assert pickle.dumps(SR._payload(ctx)) == pickle.dumps(_old_payload(ctx))
    # a column study: the same
    col = Ctx(study["rd"], study["plan"], dict(ctx._frames))
    col.is_domain = False
    assert pickle.dumps(SR._payload(col)) == pickle.dumps(_old_payload(col))
    assert SR._payload(col)["reg"] is ctx._frames["regional"]


def test_a_suite_study_never_gets_a_whole_outcome_table(study):
    ctx, out = study["ctx"], study["out"]
    ctx._frames.update(regional_monthly=pd.DataFrame({"m": [MARK]}),
                       regional_seasonal=pd.DataFrame({"s": [MARK]}))
    _approve(study)
    assert AP.plan_passed(out)[0]
    ctx.read.clear()
    pay = SR._payload(ctx)
    assert not any(k in pay for k in AP.OUTCOME_TABLES)
    assert pay["df"] is None and pay["prof"] is None and pay["soil"] is None
    assert str(MARK) not in repr(pay)
    assert ctx.read == []                     # the tables are not even read
    # a script that reads reg fails, naming the suite call to use instead
    res = SR.run(READS_REG, ctx, out / "r.png", script_path=out / "r.py",
                 lookup_tag={"script": "r", "amendment": None})
    assert res["ok"] is False
    assert 'suite.get_series(run, field, region, "year")' in res["error"]
    for name, step in (("reg_monthly", "month"), ("reg_seasonal", "season")):
        res = SR.run(f'result = {{"n": 3, "k": list({name}.columns)}}', ctx,
                     out / "r.png", script_path=out / "r.py",
                     lookup_tag={"script": "r", "amendment": None})
        assert res["ok"] is False and name in res["error"]
        assert f'suite.get_series(run, field, region, "{step}")' in res["error"]
    # testing for the name does not fail: only reading it does
    res = SR.run('plt.plot([1, 2]); plt.savefig(out_path)\n'
                 'result = {"n": 3, "bound": reg is not None}', ctx, out / "r.png",
                 script_path=out / "r.py", lookup_tag={"script": "r", "amendment": None})
    assert res["ok"] is True and res["result"]["bound"] is True
