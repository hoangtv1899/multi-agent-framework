"""The recorded replies of G part 2 (2026-11-08), fed back through the fixed
Analyzer with no model call: questions 1, 2 and 3 on copies of their studies.

    pytest tests/test_analyzer_g2_replay_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the g2_replay block of
tests/local_oracle_paths.json names the folder of the recorded studies
(IDEAS/elm_bgc_naches/oracles/g2_replay_2026-11-08/, read only). Each test
copies a study to its own temporary folder and writes only there; the
control's suite on /compyfs is read, never written.

A REPLAY, NOT A RUN. Each model step gets the reply recorded for it, by its
client's label (g2_replay_support.Replies). Question 1's scripts are NOT run
again: its recorded lookups log (lookups.jsonl) and investigation are used as
they are, which is what the claim trace and the brief read. Question 1's
wording rewrite was not saved by the code of 2026-11-08 (only the first step
3 exchange was); its reply is rebuilt from interpretation.json, which holds the
rewrite's wording.

What each question shows:
  1. step 3's brief lists every record of every finding with its warnings and
     names question 1's BTRAN "20.2 percent" as a number no record holds; its
     template asks for every part the trace reads; the recorded claims still
     fail the trace ("no record"); step 3's audit strikes the BTRAN percent;
     the plan review's two dashes stay out of its record; the close computed
     from the recorded lines gives every analysis a status and the acceptance
     blockers; the spend counts five calls; the study record knows package F
     and says the claims are withheld.
  2. the recorded plan gives exactly its recorded problem (the screen as an
     analysis), the task states that rule, and the plan replayed twice fails
     twice and stops with the code check's reason on the caveat and the page;
     step 2b and step 3 are never asked.
  3. the recorded plan gives exactly its recorded problem (by_band as a
     setting), the task lists overlap's settings, a rewrite with by_band
     removed passes the same check and review, and the plan given twice
     fails twice and stops with the code check's reason, as question 2's.
"""
import copy
import fnmatch
import json
import shutil
from pathlib import Path

import pytest

from g2_replay_support import Replies, forbid_model_clients, record_strings

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).with_name("local_oracle_paths.json")
EM = "\u2014"
REGION = "watershed, vegetated"
# What step 1 and the numbers report wrote before the Analyzer: kept in a copy
# whose Analyzer outputs are taken out, so the replay writes the rest again.
BEFORE_ANALYZER = ("DOMAIN_REPORT.md", "domain_report.json", "numbers_check.json",
                   "comparison.json", "domain_fig_*")
Q1_BRIEF_BEFORE = 35269          # step3_round1_prompt.txt of question 1: brief and SUITE_CLAIMS


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


@pytest.fixture(scope="module")
def replay():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    d = (json.loads(CONFIG.read_text()).get("g2_replay") or {}).get("dir")
    if not d or not (Path(d) / "q1" / "04_analysis" / "lookups.jsonl").is_file():
        pytest.skip("no g2_replay block naming the recorded studies")
    return Path(d)


def _copy(replay, q, tmp_path, analysis=True):
    """The recorded study `q`, copied; with analysis=False its Analyzer
    outputs (everything in 04_analysis but what came before the Analyzer) are
    left out, so the replay writes them again."""
    rd = tmp_path / q
    shutil.copytree(replay / q, rd)
    if not analysis:
        for p in (rd / "04_analysis").iterdir():
            if not any(fnmatch.fnmatch(p.name, k) for k in BEFORE_ANALYZER):
                p.unlink()
    return rd


def _said(replay, q, name):
    return (replay / q / "04_analysis" / name).read_text()


def _recorded(replay, q, name):
    return json.loads(_said(replay, q, name))


def _investigation(rd):
    inv = json.loads((rd / "04_analysis" / "investigation.json").read_text())
    inv["plan_md"] = str(rd / "04_analysis" / "investigation_plan.md")   # the copy's
    return inv


def _comparison(rd):
    p = rd / "04_analysis" / "comparison.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _no_outputs(plan):
    """What the contract of required outputs (2026-11-08) adds to a plan
    written before it: one problem per analysis, in the check's order."""
    return [f"{a['id']}: names no required output (\"outputs\"): each analysis names the "
            f"outputs it must produce, apart from its scope" for a in plan["analyses"]]


# WHAT THE TIMING RULE ADDS (the user's decision after the third acceptance
# batch, 2026-11-08) to these plans, written before the required outputs: a
# question about timing, and each expectation of "earlier" or "later", find no
# timing measure among outputs the plan does not name. After every analysis's
# own problems, in the check's order.
NO_TIMING_OUTPUT = {
    "q1": ["timing: the question's 'early' is about snow loss, but no analysis has snow_gone "
           "among its outputs for 2015",
           "timing: E2 expects 'earlier' for H2OSNO, but its analysis P2 has no snow_peak or "
           "snow_gone among its outputs",
           "timing: E4 expects 'earlier' for GPP, but its analysis P4 has no "
           "photosynthesis_onset or growth_peak among its outputs"],
    "q2": [],
    "q3": ["timing: E5 expects 'earlier' for H2OSNO, but its analysis P6 has no snow_peak or "
           "snow_gone among its outputs"]}


def _without_outputs_rule(problems):
    return [p for p in problems if "names no required output" not in p]


def _one_output(a):
    """One output inside an analysis's own scope, as a plan following the
    contract would name it: an anomaly, else a timing measure, else a map."""
    import domain_methods as M
    periods = [p for p in a.get("periods") or [] if not str(p).startswith("normal")]
    if "anomaly" in (a.get("methods") or []) and a.get("fields") and periods:
        return {"id": "O1", "kind": "anomaly", "field": a["fields"][0], "period": periods[0],
                "region": (a.get("regions") or [None])[-1]}
    tm = [m for m in a.get("methods") or [] if m in M.TIMING_IDS]
    if tm:
        return {"id": "O1", "kind": tm[0], "region": a["regions"][0], "period": periods[0]}
    return {"id": "O1", "kind": "map", "field": a["fields"][0], "period": periods[0]}


def _step3_claims(replay):
    return S3._parse(_said(replay, "q1", "step3_round1_reply.txt"))


# ── question 1 ───────────────────────────────────────────────────────────────

def test_q1_the_plan_and_its_review_replay_as_recorded_without_the_reviews_dashes(replay,
                                                                                   tmp_path):
    rd = _copy(replay, "q1", tmp_path, analysis=False)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    r = Replies({"step2a_plan": [_said(replay, "q1", "step2a_round1_reply.txt")],
                 "step3_plan_review": [_said(replay, "q1",
                                             "step3_plan_review_round1_reply.txt")]})
    plan = AP.write_plan(ctx, out, client=r.clients()["step2a_plan"])
    rec = _recorded(replay, "q1", "analysis_plan.json")
    # the plan as recorded, part for part; a plan written now has also seen the
    # recipe its question calls for (analysis_recipes, 2026-11-09), and says so
    left = ("written_at", "problems", "seen")
    assert {k: v for k, v in plan.items() if k not in left} == \
        {k: v for k, v in rec.items() if k not in left}
    assert plan["seen"] == rec["seen"] + [AP.SEEN_RECIPE]
    # written before the contract of required outputs (2026-11-08): it names none,
    # so the check now finds one problem per analysis and, since the timing rule,
    # the timing problems that follow from it, and only those
    assert rec["problems"] == [] and \
        plan["problems"] == _no_outputs(rec) + NO_TIMING_OUTPUT["q1"]
    rv = S3.rule_plan_follows_question(ctx, out, client=r.clients()["step3_plan_review"])
    assert rv["passed"] is False and rv["follows"] is True
    assert rv["code_problems"] == _no_outputs(rec) + NO_TIMING_OUTPUT["q1"] and \
        AP.can_rewrite(out) == (True, "")
    said = json.loads(_said(replay, "q1", "step3_plan_review_round1_reply.txt"))["why"]
    assert said.count(EM) == 2                            # the model's text, as recorded
    assert not any(EM in x for x in record_strings(out / AP.REVIEW_FILE))
    assert rv["why"] == said.replace(f" {EM} ", ", ").replace(EM, ", ")
    for name in (AP.PLAN_FILE, AP.REVIEW_FILE):
        assert not any(EM in x for x in record_strings(out / name)), name
    assert (out / "step3_plan_review_round1_reply.txt").read_text().count(EM) == 2


def test_q1_step_3_sees_every_record_and_the_recorded_claims_still_fail(replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    inv = _investigation(rd)
    brief = S3.review_brief(ctx, _comparison(rd), inv)
    task = S3.suite_task()
    # the brief before the fix, for the record: the recorded prompt
    assert len(_said(replay, "q1", "step3_round1_prompt.txt")) == Q1_BRIEF_BEFORE
    (tmp_path / "q1_brief_sizes.json").write_text(json.dumps(
        {"before": Q1_BRIEF_BEFORE, "after_brief": len(brief), "after_task": len(task),
         "after_saved": len(brief) + 1 + len(task)}))
    recs = CT.evidence_records(out)
    n = 0
    for f in inv["findings"]:
        block = brief[brief.index(f"    id: {f['id']}   "):]
        nxt = block.find("\n    id: ", 10)
        block = block if nxt < 0 else block[:nxt]
        for x in f["lookups"]:
            assert f"          {x['id']}: " in block, (f["id"], x["id"])
            for w in recs[x["id"]]["warnings"]:
                assert f"{w['code']} (\"{w['label']}\")" in block, (x["id"], w["code"])
            n += 1
        if f["id"] == "seasonal_annual_anomalies_2015":
            # question 1's "BTRAN 20.2 percent lower": the script's own arithmetic
            made = block[block.index("numbers the script made itself, a value of no record"):]
            assert "values.BTRAN|2015-JJA.relative_pct_plotted = -20.185445399047374" in made
            # the BTRAN anomaly itself is L24's value, listed under L24
            l24 = block[block.index("          L24: "):block.index("          L25: ")]
            assert "its values in the finding's result: values.BTRAN|2015-JJA.anomaly = " \
                   "-0.052659106640679965" in l24
            assert "percent: none (a bounded index: its change is given within its bounds, " \
                   "never as a percent)" in l24
    assert n == 70                                     # every lookup of every finding
    # (since 2026-11-08 a study with a suite ranks its claims for the headline
    # in "headline_rank" and writes no headline of its own)
    for part in ("evidence", "evidence_class", "expectation", "warnings", "headline_rank"):
        assert f'"{part}": ' in task, part
    # the recorded claims, traced: all six still fail, each for want of a record
    spec = _step3_claims(replay)
    t = CT.trace_evidence(copy.deepcopy(spec["claims"]), spec["headline"], out,
                          study_dir=rd, regions=S4.suite_regions(ctx))
    assert t["claims_failed"] == [1, 2, 3, 4, 5, 6]
    for i in range(1, 7):
        assert any(p.startswith("no record: every empirical claim names the logged lookup")
                   for p in t["problems"][i]), i


def test_q1_the_btran_percent_is_struck_by_step_3s_audit(replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    inv = _investigation(rd)
    records = CT.evidence_records(out)
    claims = _step3_claims(replay)["claims"]
    a = S3.audit(claims, inv, list(ctx.caveats or []), facts=S3.run_facts(ctx), records=records)
    assert [s["claim"] for s in a["struck"]] == [claims[5]["claim"]]
    s = a["struck"][0]
    assert s["struck_by"] == "percent"
    assert "a percent of BTRAN, whose notes give none: a bounded index: its change is given " \
           "within its bounds, never as a percent" in s["struck_because"]
    assert len(a["kept"]) == 5                         # the other rules keep what they kept
    # without the records (the audit before the fix) it was kept
    assert len(S3.audit(claims, inv, list(ctx.caveats or []), facts=S3.run_facts(ctx))["kept"]) \
        == 6
    # the trace fails it too: for want of a record, and by the same rule once it names L24
    t = CT.check_claim(dict(claims[5], evidence=["L24"], evidence_class="observation",
                            warnings=[w["code"] for w in records["L24"]["warnings"]]),
                       records, AP.load_plan(out), {}, S4.suite_regions(ctx))
    assert any(p.startswith("a percent of BTRAN, whose notes give none") for p in t["problems"])


def test_q1_the_close_from_the_recorded_lines(replay, tmp_path):
    """The close of the planned phase on question 1's own lookups, judged by
    required outputs the test names on its recorded plan, as a plan
    following the contract (2026-11-08) would. Positive: outputs its lookups
    match exactly, an overlap by the maps it was given and an elevation
    split by another analysis's output. Not estimable, apart: the 2015 snow
    peak (2015 is the run's first year). Negative: P1's 2015 annual snow
    anomaly, never looked up although both figures naming P1 ran."""
    rd = _copy(replay, "q1", tmp_path)
    out = rd / "04_analysis"
    lines = (out / AP.LOOKUPS_FILE).read_text().splitlines()
    assert len(lines) == 75 and json.loads(lines[74])["call"] == "screen"
    # the lines made before the screen's figure, as the fixed flow closes there
    (out / AP.LOOKUPS_FILE).write_text("\n".join(lines[:74]) + "\n")
    W, B = REGION, f"{REGION}, 1000-1500 m"

    def anom(oid, field, period, region=W):
        return {"id": oid, "kind": "anomaly", "field": field, "period": period, "region": region}
    outputs = {
        "P1": [anom("O1", "H2OSNO", "2015-MAM"), anom("O2", "H2OSNO", "2015")],
        "P2": [{"id": "O1", "kind": "snow_peak", "region": W, "period": "2015"},
               {"id": "O2", "kind": "snow_gone", "region": W, "period": "2015"}],
        "P3": [anom("O1", "GPP", "2015-JJA")],
        "P4": [{"id": "O1", "kind": "photosynthesis_onset", "region": W, "period": "2015"},
               {"id": "O2", "kind": "growth_peak", "region": W, "period": "2015"}],
        "P5": [anom("O1", "BTRAN", "2015-JJA")],
        "P6": [anom("O1", "TOTECOSYSC", "2015")],
        "P7": [anom("O1", "H2OSNO", "2015-MAM", None), anom("O2", "GPP", "2015-JJA", None),
               {"id": "O3", "kind": "overlap", "region": W, "inputs": ["O2", "O1"]}],
        "P8": [{"id": "O1", "kind": "elevation_split", "region": W, "inputs": ["P7.O1"]},
               anom("O2", "GPP", "2015-JJA", B)],
        "P9": [anom("O1", "TBOT", "2015-JJA")]}
    plan = json.loads((out / AP.PLAN_FILE).read_text())
    for an in plan["analyses"]:
        an["outputs"] = outputs[an["id"]]
    (out / AP.PLAN_FILE).write_text(json.dumps(plan, indent=1))
    assert AP.plan_problems(plan, *AP._catalog_names(S0.load(str(rd))), [2016, 2017, 2018,
                                                                          2019, 2020],
                            "hypothesis") == []
    (out / AP.REVIEW_FILE).unlink()                    # the plan the test names, reviewed
    AP.record_review(out, True, "the test's plan", reviewer="the test")
    inv = _investigation(rd)
    figures = [{"id": f["id"], "plan_items": f["plan_items"], "ok": True, "error": None}
               for f in inv["findings"]]
    ph = AP.close_planned_phase(out, when="before the screen", by="the replay", figures=figures)
    rows = {r["id"]: r for r in ph["analyses"]}
    assert {k: r["status"] for k, r in rows.items()} == {
        "P1": "failed", "P2": "completed", "P3": "completed", "P4": "completed",
        "P5": "completed", "P6": "completed", "P7": "completed", "P8": "completed",
        "P9": "completed"}
    out_of = {k: {o["id"]: (o["outcome"], o["record"]) for o in r["outputs"]}
              for k, r in rows.items()}
    assert out_of["P1"] == {"O1": ("produced", "L3"), "O2": ("unmet", None)}
    assert rows["P1"]["reason"].startswith("O2 unmet: no planned lookup matched it (the "
                                           "figures 'daily_snow_and_gpp_2015_vs_normal', "
                                           "'seasonal_annual_anomalies_2015' naming this "
                                           "analysis ran without it)")
    assert out_of["P2"] == {"O1": ("not estimable", "L70"), "O2": ("produced", "L71")}
    assert rows["P2"]["reason"] == "1 required output produced, 1 not estimable"
    assert out_of["P7"] == {"O1": ("produced", "L54"), "O2": ("produced", "L55"),
                            "O3": ("produced", "L57")}
    assert out_of["P8"]["O1"] == ("produced", "L58")
    assert {k for k, r in rows.items() if r["required"]} == {"P1", "P2", "P3", "P4", "P5",
                                                            "P7", "P8"}
    assert ph["lookups_before"] == 74 and ph["counts"] == {
        "completed": 8, "completed after an earlier failure": 0, "failed": 1, "skipped": 0}
    # the screen is allowed after the close, labelled as before
    marks = AP.check_lookup(out, {"call": "screen", "method": "screen", "settings": {},
                                  "keys": {"run": "control", "year": 2015, "region": REGION,
                                           "normal_years": None}})
    assert (marks["planned"], marks["label"], marks["after_close"]) == \
        (False, AP.SCREENING_LABEL, True)
    acc = AP.acceptance(ph)
    assert acc["ok"] is False
    assert [(b["analysis"], b["status"], b["unmet"]) for b in acc["blockers"]] == [
        ("P1", "failed", ["O2"])]
    assert [(x["analysis"], x["output"], x["record"]) for x in acc["not_estimable"]] == [
        ("P2", "O1", "L70")]
    # E2 (the earlier melt-out and peak) cannot be tested on the 2015 peak: listed
    # apart, its task complete, its melt-out still produced (the open points, 2026-11-08)
    assert [(u["expectation"], u["analysis"], [o["output"] for o in u["outputs"]],
             u["none_estimable"]) for u in acc["untestable"]] == [("E2", "P2", ["O1"], False)]
    (tmp_path / "q1_close.json").write_text(json.dumps(
        {k: {x: r[x] for x in ("status", "required", "produced", "not_estimable", "unmet",
                               "reason")} for k, r in rows.items()}, indent=1))


def test_q1_the_spend_counts_all_five_calls(replay, monkeypatch):
    import agents.llm_agent as LA
    calls = json.loads((replay / "q1" / "calls.json").read_text())
    monkeypatch.setattr(LA, "USAGE_LOG", list(calls["usage_log"]))
    llm = S4._llm_accounting(expect_calls=True)
    assert llm["analyzer"]["calls"] == 5 == calls["usage_totals"]["calls"]
    assert (llm["analyzer"]["prompt_tokens"], llm["analyzer"]["completion_tokens"]) == \
        (77570, 23960)
    assert "llm      : 5 calls, 101530 tokens" in S4.summary(
        {"cost": {"llm": llm}, "provenance": {}, "claims": [], "withheld": []})


def test_q1_step_3_and_step_4_replayed_the_record_knows_package_f(replay, tmp_path):
    """Step 3 through the recorded replies (the claims, then the wording
    rewrite rebuilt from interpretation.json), step 4 and the study record:
    no record holds the model's dashes, the study record names every package
    F file and says the report and its claims are withheld."""
    from agents import study_record as SRD
    rd = _copy(replay, "q1", tmp_path)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    inv = _investigation(rd)
    old = _recorded(replay, "q1", "interpretation.json")
    rewrite = {"headline": old["headline"], "answer": old["answer"],
               "claims": [c["claim"] for c in old["claims"]]}
    r = Replies({"step3_interpret": [_said(replay, "q1", "step3_round1_reply.txt"), rewrite]})
    interp = S3.interpret(ctx, _comparison(rd), inv, out, client=r.clients()["step3_interpret"],
                          with_images=False)
    assert r.labels_asked() == ["step3_interpret", "step3_interpret"]
    assert interp["audit"] == {"n_claims": 6, "n_struck": 1}
    assert interp["struck"][0]["struck_by"] == "percent"
    rep = S4.build(ctx, _comparison(rd), inv, interp, rd, rounds=[], stopped_because="replay",
                   steps={"investigate": True, "interpret": True})
    assert rep["evidence_trace"]["claims_failed"] == [1, 2, 3, 4, 5]
    S4.write(rep, out, slides=False)
    page = SRD.write_record(rd).read_text()
    for name in (AP.PLAN_FILE, AP.REVIEW_FILE, AP.LOOKUPS_FILE, "evidence_numbers_check.json"):
        row = [x for x in page.splitlines() if x.startswith(f"| `04_analysis/{name}` |")]
        assert row and "no role on record" not in row[0], name
    sec = page[page.index("## The Analyzer"):page.index("## Every file in this folder")]
    assert "is **withheld**: 5 of 5 claims fail the evidence trace" in sec
    assert "Its report" not in sec and "kept by the audit" not in sec
    assert "- claims withheld with the report: 5; struck by step 3's audit: 1" in sec
    assert sec.count("**(withheld)**") == 5
    assert sec.count("evidence class: none; records: none; the evidence trace: fails (no "
                     "record") == 5
    assert "  - struck (percent): In the same summer" in sec
    assert "- the claims' evidence classes: none 5" in sec
    # the model's dashes reach no record the replay wrote
    said = [w for name in ("step3_plan_review_round1_reply.txt", "step3_round1_reply.txt",
                           "step2_round1_reply.txt", "step2a_round1_reply.txt")
            for w in _said(replay, "q1", name).split('"') if EM in w]
    assert said                                       # the review's two
    for p in (out / "interpretation.json", out / "analysis.json", out / "REPORT.md",
              rd / "STUDY_RECORD.md"):
        strings = record_strings(p)
        for w in said:
            assert not any(w in x for x in strings), (p.name, w)
    assert not any(EM in x for x in record_strings(out / "interpretation.json"))


# ── question 2 ───────────────────────────────────────────────────────────────

def test_q2_the_recorded_plan_gives_its_recorded_problem_the_task_states_the_rule(replay,
                                                                                tmp_path):
    rd = _copy(replay, "q2", tmp_path, analysis=False)
    ctx = S0.load(str(rd))
    r = Replies({"step2a_plan": [_said(replay, "q2", "step2a_round1_reply.txt")]})
    plan = AP.write_plan(ctx, rd / "04_analysis", client=r.clients()["step2a_plan"])
    rec = _recorded(replay, "q2", "analysis_plan.json")
    assert _without_outputs_rule(plan["problems"]) == rec["problems"] == \
        ["P1: the screen is declared in the plan's screen entry, not as an analysis"]
    assert [p for p in plan["problems"] if p not in rec["problems"]] == _no_outputs(rec)
    task = " ".join(r.prompts("step2a_plan")[0].split())
    assert "The screen is never an analysis's method: it is declared once, in the plan's own " \
           "\"screen\" entry, and an analysis that lists \"screen\" fails the check" in task


CODE_PROBLEM = {"q2": "P1: the screen is declared in the plan's screen entry, not as an analysis",
                "q3": "P5: settings for no method 'by_band'"}


@pytest.mark.parametrize("q", ["q2", "q3"])
def test_replayed_twice_fails_twice_and_stops_with_the_real_reason(replay, tmp_path, q):
    """Questions 2 and 3, their recorded plan given twice (the rewrite with no
    fix): both versions fail on the code check, step 2b and step 3 are never
    asked, and the caveat, analysis.json, the page and the study record give
    the code check's problem first."""
    rd = _copy(replay, q, tmp_path, analysis=False)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    reply = _said(replay, q, "step2a_round1_reply.txt")
    review = _said(replay, q, "step3_plan_review_round1_reply.txt")
    r = Replies({"step2a_plan": [reply], "step3_plan_review": [review],
                 "step2a_plan_rewrite": [reply], "step3_plan_review_rewrite": [review]})
    loop = S3.investigate_and_interpret(ctx, out, comparison=_comparison(rd),
                                        with_images=False, clients=r.clients())
    # step 2b and step 3 never asked: nothing was looked up
    assert r.labels_asked() == ["step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                                "step3_plan_review_rewrite"]
    assert loop["interpretation"]["skipped"] is True and not AP.lookups(out)
    code = CODE_PROBLEM[q]
    ok, why = AP.plan_passed(out)
    n = 1 + len(_recorded(replay, q, "analysis_plan.json")["analyses"]) + \
        len(NO_TIMING_OUTPUT[q])                     # and no outputs, and the timing rule
    assert not ok and why.startswith(f"the analysis plan did not pass step 3's review: the "
                                     f"code check found {n} problems: ")
    assert all(t in why for t in NO_TIMING_OUTPUT[q])
    assert code in why and why.index(code) < why.index("the model review said the plan "
                                                       "follows from the question")
    cav = loop["investigation"]["caveats"][0]
    assert cav["id"] == "analysis_plan_not_approved" and code in cav["statement"]
    assert cav["statement"].index(code) < cav["statement"].index("the model review")
    rep = S4.build(ctx, _comparison(rd), loop["investigation"], loop["interpretation"], rd,
                   rounds=loop["rounds"], stopped_because=loop["stopped_because"],
                   steps={"investigate": True, "interpret": "skipped"})
    S4.write(rep, out, slides=False)
    page = (out / "REPORT.md").read_text()
    assert page.startswith("# WITHHELD: the analysis plan did not pass its review")
    assert page.count("- the code check: ") == 2 and page.count(code) >= 2
    assert "No lookup was made" in page and "No results were produced" not in page
    a = json.loads((out / "analysis.json").read_text())
    assert code in a["withheld_because"] and a["status"] == S4.INSUFFICIENT
    assert a["provenance"]["skipped_steps"]["interpret"].startswith("the analysis plan did not")
    assert [v["passed"] for v in a["analysis_plan"]["versions"]] == [False, False]
    # the study record: drawn without the interpretation step 3 never wrote,
    # with the code check's reason first and both versions
    from agents import study_record as SRD
    page = SRD.write_record(rd).read_text()
    sec = page[page.index("## The Analyzer"):page.index("## Every file in this folder")]
    assert f"is **withheld**: the analysis plan did not pass step 3's review: the code check " \
           f"found {n} problems: " in sec and code in sec
    assert "the first version, `04_analysis/analysis_plan.v1.json`, did not pass its review" in sec
    assert "its review, `04_analysis/analysis_plan_review.json`: **did not pass**; the code " \
           "check: " in sec and "; the model review: follows from the question" in sec
    assert "- the lookups: none" in sec and "- acceptance: **not met**: the analysis plan did " \
           "not pass its review" in sec
    for name in (AP.PLAN_FILE, AP.PLAN_V1_FILE, AP.REVIEW_FILE, AP.REVIEW_V1_FILE):
        row = [x for x in page.splitlines() if x.startswith(f"| `04_analysis/{name}` |")]
        assert row and "no role on record" not in row[0], name


# ── question 3 ───────────────────────────────────────────────────────────────

def test_q3_by_band_is_its_recorded_problem_and_a_rewrite_without_it_passes(replay, tmp_path):
    rd = _copy(replay, "q3", tmp_path, analysis=False)
    ctx = S0.load(str(rd))
    out = rd / "04_analysis"
    reply = _said(replay, "q3", "step2a_round1_reply.txt")
    fixed = S3._parse(reply)
    p5 = [a for a in fixed["analyses"] if a["id"] == "P5"][0]
    assert p5["settings"] == {"by_band": True}
    p5["settings"] = {}                                # the rewrite the test makes
    for an in fixed["analyses"]:                       # and its required outputs
        an["outputs"] = [_one_output(an)]
    review = _said(replay, "q3", "step3_plan_review_round1_reply.txt")
    r = Replies({"step2a_plan": [reply], "step3_plan_review": [review],
                 "step2a_plan_rewrite": [fixed], "step3_plan_review_rewrite": [review]})
    c = r.clients()
    plan = AP.write_plan(ctx, out, client=c["step2a_plan"])
    rec = _recorded(replay, "q3", "analysis_plan.json")
    assert rec["problems"] == ["P5: settings for no method 'by_band'"]
    assert _without_outputs_rule(plan["problems"]) == rec["problems"] + NO_TIMING_OUTPUT["q3"]
    assert [p for p in plan["problems"] if p not in rec["problems"]] == \
        _no_outputs(rec) + NO_TIMING_OUTPUT["q3"]
    task = " ".join(r.prompts("step2a_plan")[0].split())
    assert "- overlap: sign_share_limit, default 0.8, no alternative (share of the area)." in task
    assert "a call's own arguments (overlap's by_band, a timing measure's against) are not " \
           "settings" in task
    rv = S3.rule_plan_follows_question(ctx, out, client=c["step3_plan_review"])
    assert rv["passed"] is False and rv["code_problems"] == plan["problems"]
    assert AP.can_rewrite(out) == (True, "")
    v2 = AP.rewrite_plan(ctx, out, client=c["step2a_plan_rewrite"])
    assert v2["problems"] == [] and v2["attempt"] == 2
    rv2 = S3.rule_plan_follows_question(ctx, out, client=c["step3_plan_review_rewrite"],
                                        label=AP.REWRITE_REVIEW_LABEL)
    assert rv2["passed"] is True and AP.plan_passed(out) == (True, "the analysis plan passed "
                                                                  "step 3's review")
    assert r.prompts("step3_plan_review_rewrite")[0].endswith(S3.PLAN_REVIEW_ASK)
    assert (out / AP.PLAN_V1_FILE).is_file() and (out / AP.REVIEW_V1_FILE).is_file()
    # a lookup the rewrite declares now passes the gate, planned
    marks = AP.check_lookup(out, {"call": "overlap", "method": "overlap", "settings": {},
                                  "keys": {"region": REGION, "by_band": True}})
    assert marks["planned"] is True and "P5" in marks["plan_items"]
