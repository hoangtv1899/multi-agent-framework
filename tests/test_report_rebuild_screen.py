"""The screen only where it is needed (the report rebuild, package A10), on
small fake runs, with no model call.

    pytest tests/test_report_rebuild_screen.py -v

A study with a new run screens that run against the control run in the
screen's year; with several new runs, the run of the plan's first comparison
whose run is not the control run (analysis_plan.screen_run). A study of the
control run alone screens the year against the normal years, as before. The
framework's own call and a script's suite.screen(year, region) both screen
the chosen run; in a study with a new run the gate refuses a screen of the
control run, with a plain reason; the screen's record names the run it
screened; and the second reading of the numbers makes a new run's screen
again against the control run.
"""
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_access as DAC
import domain_suite as S
from test_domain_access import PLAN_REPLY, REGION, YEARS, Ctx, Fake, built  # noqa: F401
from test_domain_suite import FIELDS, make_control

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import suite_check as SC            # noqa: E402

NEW = "wet"


def _cmp(run, cid="C1"):
    return {"id": cid, "what": "w", "run": run,
            "against": "control" if run != "control" else "normal years", "years": [2016]}


# ── which run ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("comparisons,run", [
    ([_cmp("wet")], "wet"),
    ([_cmp("control"), _cmp("wet", "C2"), _cmp("dry", "C3")], "wet"),
    ([_cmp("dry"), _cmp("wet", "C2")], "dry"),
    ([_cmp("control")], "control"),
    ([], "control"),
], ids=["one new run", "two new runs after the control", "two new runs", "the control alone",
        "no comparison"])
def test_the_screen_looks_at_the_first_comparisons_new_run(comparisons, run):
    assert AP.screen_run({"comparisons": comparisons}) == run
    assert AP.screen_run(None) == "control"


# ── a study with a new run ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def wet_suite(built, tmp_path_factory):
    root = tmp_path_factory.mktemp("wet")
    wet, _ = make_control(root / "wet", YEARS)
    for p in sorted(Path(wet["run_dir"]).glob("*.elm.h[01].*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            if "GPP" in d.variables:
                d["GPP"][:] = (np.asarray(d["GPP"][:]) * 1.2).astype(np.float32)
    out = root / "suite_wet"
    S.build(built["control"], out, run_dir=wet["run_dir"], fields=FIELDS)
    return out


def _study(built, wet_suite, tmp_path, monkeypatch, comparisons):
    entry = dict(built["control"], suite_dir=str(built["suite"]))
    monkeypatch.setattr(AP, "_registry_entry", lambda name: entry if name == "synth" else {})
    rd = tmp_path / "study"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": built["control"]["run_dir"]}}))
    shutil.copytree(wet_suite, rd / "03_results" / "suite" / NEW)
    plan = {"question": "What did the wet run change in 2016?", "goals": ["GPP in 2016"],
            "experiment": {"control": {"name": "synth", "reuse": True},
                           "runs": [{"run": NEW, "changes": []}, {"run": "dry", "changes": []}],
                           "years": YEARS,
                           "analysis": {"normal_years": [2015], "question_type": "exploratory",
                                        "regions": [{"name": REGION}], "variables": ["GPP"]}}}
    out = rd / "04_analysis"
    out.mkdir()
    reply = dict(PLAN_REPLY, comparisons=comparisons, analyses=[
        {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2016"], "methods": ["difference"], "settings": {}, "why": "asked",
         "outputs": [{"id": "O1", "kind": "difference", "run": NEW, "field": "GPP",
                      "period": "2016", "region": REGION}]}])
    ctx = Ctx(rd, plan)
    AP.write_plan(ctx, out, client=Fake(reply))
    rv = S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "w",
                                                              "problems": []}))
    assert rv["passed"] is True, rv
    assert set(AP.suite_dirs(ctx)) == {"control", NEW}
    return {"ctx": ctx, "rd": rd, "out": out}


@pytest.fixture()
def wet_study(built, wet_suite, tmp_path, monkeypatch):
    return _study(built, wet_suite, tmp_path, monkeypatch, [_cmp("control"), _cmp(NEW, "C2")])


def test_the_frameworks_screen_sets_the_new_run_against_the_control(wet_study):
    AP.produce_required(wet_study["ctx"], wet_study["out"])
    log = AP.lookups(wet_study["out"])
    first = log[0]
    assert (first["call"], first["script"]) == ("screen", AP.REQUIRED_SCRIPT)
    assert first["keys"]["run"] == NEW and first["plan_items"] == ["screen"]
    assert first["planned"] is True                      # the plan's screen, of an exploratory study
    rec = json.loads((wet_study["out"] / AP.SCREEN_FILE).read_text())
    assert rec["scope"]["run"] == NEW and rec["scope"]["against"] == "the control"
    assert rec["keys"]["run"] == NEW
    # what came after it was allowed: the screen ran
    assert all("refused" not in x for x in log)


def test_a_scripts_screen_defaults_to_the_studys_run(wet_study):
    acc = AP.make_access(AP.access_spec(wet_study["ctx"]), script="fig_a")
    got = acc.screen(2016, REGION)
    assert got["against"] == "the control"
    line = AP.lookups(wet_study["out"])[-1]
    assert line["keys"]["run"] == NEW and line["script"] == "fig_a"
    assert line["plan_items"] == ["screen"]


def test_the_gate_refuses_the_control_runs_screen_in_a_study_with_a_new_run(wet_study):
    acc = AP.make_access(AP.access_spec(wet_study["ctx"]), script="fig_a")
    with pytest.raises(DAC.AccessRefused, match="this study has a new run, wet") as e:
        acc.screen(2016, REGION, run="control")
    assert "says nothing about the change" in str(e.value)
    line = AP.lookups(wet_study["out"])[-1]
    assert line["keys"]["run"] == "control" and line["refused"].startswith("the gate: ")
    assert not (wet_study["out"] / AP.SCREEN_FILE).exists()


def test_a_second_new_runs_screen_is_not_the_plans(built, wet_suite, tmp_path, monkeypatch):
    st = _study(built, wet_suite, tmp_path, monkeypatch, [_cmp("dry"), _cmp(NEW, "C2")])
    assert AP.screen_run(AP.load_plan(st["out"])) == "dry"
    with pytest.raises(AP.PlanNotApproved, match="declares no"):
        AP.check_lookup(st["out"], {"call": "screen", "method": "screen", "settings": {},
                                    "keys": {"run": NEW, "year": 2016, "region": REGION}})
    marks = AP.check_lookup(st["out"], {"call": "screen", "method": "screen", "settings": {},
                                        "keys": {"run": "dry", "year": 2016, "region": REGION}})
    assert marks["plan_items"] == ["screen"]


def test_screen_study_makes_the_new_screen_and_sets_the_old_one_aside(wet_study, monkeypatch):
    # a study saved before the rule: its screen looked at the control run
    plain = AP.screen_run
    monkeypatch.setattr(AP, "screen_run", lambda plan: "control")
    acc = AP.make_access(AP.access_spec(wet_study["ctx"]), script=AP.REQUIRED_SCRIPT)
    acc.screen(2016, REGION)
    old = json.loads((wet_study["out"] / AP.SCREEN_FILE).read_text())
    assert old["scope"]["run"] == "control" and old["scope"]["against"] == "the normal years"
    monkeypatch.setattr(AP, "screen_run", plain)
    import agents.analysis.step0_context as S0
    monkeypatch.setattr(S0, "load", lambda d: wet_study["ctx"])
    rec = AP.screen_study(wet_study["rd"])
    assert rec["scope"]["run"] == NEW and rec["scope"]["against"] == "the control"
    log = AP.lookups(wet_study["out"])
    assert len(log) == 2 and log[-1]["keys"]["run"] == NEW
    moved = list((wet_study["out"] / "superseded").glob(f"*/{AP.SCREEN_FILE}"))
    assert len(moved) == 1 and json.loads(moved[0].read_text()) == old
    assert (wet_study["out"] / "screen_L2.csv").is_file()
    # a second call keeps the screen the study now has and makes none
    assert AP.screen_study(wet_study["rd"]) == rec and len(AP.lookups(wet_study["out"])) == 2


# ── a study of the control run alone ────────────────────────────────────────

def test_the_control_alone_still_screens_its_year_against_the_normal_years(built, wet_suite,
                                                                           tmp_path,
                                                                           monkeypatch):
    st = _study(built, wet_suite, tmp_path, monkeypatch, [_cmp("control")])
    AP.produce_required(st["ctx"], st["out"])
    rec = json.loads((st["out"] / AP.SCREEN_FILE).read_text())
    assert rec["scope"]["run"] == "control" and rec["scope"]["against"] == "the normal years"
    acc = AP.make_access(AP.access_spec(st["ctx"]), script="fig_b")
    acc.screen(2016, REGION)
    assert AP.lookups(st["out"])[-1]["keys"]["run"] == "control"


# ── the second reading follows the run ──────────────────────────────────────

@pytest.mark.parametrize("run,against", [("control", None), (NEW, "control")],
                         ids=["the control against its normal years",
                              "a new run against the control"])
def test_the_second_reading_makes_a_screen_again_against_what_it_was_set(monkeypatch, run,
                                                                          against):
    seen = []

    def row(ctx, r, control, field, period, region, ny, frac):
        seen.append((r, control))
        return dict.fromkeys(SC.SCREEN_COLS + ("_sd",))

    monkeypatch.setattr(SC, "screen_row", row)

    class C:
        lines = [{"call": "screen", "keys": {"run": run, "year": 2016, "region": REGION,
                                             "normal_years": None},
                  "result": {"rows.0.field": "GPP", "rows.0.period": "2016",
                             "rows.0.change": 1.5}}]
        _runs = {}

        def normal_years(self, keys):
            return [2015]
    SC.recompute(C(), 1, ["rows.0.change"])
    assert seen == [(run, against)]
