"""Closing package E (2026-11-05): what the reviewed analysis plan lets a
script look up, in what order, the full method contracts on demand, the timing
settings timing.nc holds, and the line between the automated review of the
analysis plan and the user's authorization of a simulation. On the synthetic
suite of tests/test_domain_access.py; every model call is a scripted fake.

    pytest tests/test_analysis_plan_scope.py -v

1. A lookup (get_map, get_series, get_timing, a method call) the reviewed plan
   does not declare is refused, unless an amendment appended before it does;
   an amended lookup is logged, linked to its amendment and marked not
   planned. An exploratory plan may declare the screen alone.
2. A hypothesis question runs its screen only after the framework has closed
   the planned-analysis phase (planned_phase.json; the user's decision on G
   part 2, 2026-11-08: until then it waited for every declared lookup), what
   the screen turns up is "found by screening, not tested", and every lookup
   after the close is exploratory; an exploratory question runs the screen
   first and records its scope, scoring rule and display rule.
3. suite.method_contract(method_id) returns a method's section of
   docs/ANALYSIS_METHODS.md before any review, logged as a contract read.
4. A timing setting timing.nc does not hold is refused, naming the setting
   and the settings that exist; nothing is computed on the fly.
6. The plan and lookup path imports nothing that builds or submits a run, and
   the runner starts nothing but the Python interpreter on a vetted script.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import domain_access as DAC
import domain_methods as M
from test_domain_access import (PLAN_ALL, PLAN_REPLY, REGION, _approve, _log,  # noqa: F401
                                built, study)

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import script_runner as SR          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS = dict(
    PLAN_REPLY,
    analyses=[{"id": "P1", "question": "lower GPP in 2016", "fields": ["GPP"],
               "regions": [REGION], "periods": ["2016"], "methods": ["anomaly"],
               "settings": {}, "why": "the expectation",
               "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                            "region": REGION}]},
              {"id": "P2", "question": "an earlier peak", "fields": [], "regions": [REGION],
               "periods": ["2016"], "methods": ["growth_peak"], "settings": {},
               "why": "the timing",
               "outputs": [{"id": "O1", "kind": "growth_peak", "region": REGION,
                            "period": "2016"}]}],
    expectations=[{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "lower",
                   "text": "GPP is lower in 2016 than in the normal year"}],
    screen={"when": "last", "region": REGION, "year": 2016, "why": "a hypothesis: last"})
SCREEN_ONLY = dict(PLAN_REPLY, analyses=[], competing=[])
# modules that build, submit or drive a model run
JOB_MODULES = ("elm_exp_manager", "elm_wrapper", "domain_case", "exp_manager_base",
               "resumable", "workflow")


def _hypothesis(study):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "hypothesis"
    return _approve(study, reply=HYPOTHESIS)


def _acc(study, script="fig", amendment=None):
    return AP.make_access(AP.access_spec(study["ctx"]), script=script, amendment=amendment)


# ── 1. the lookup scope ─────────────────────────────────────────────────────

def test_a_lookup_the_reviewed_plan_does_not_declare_is_refused(study):
    _approve(study)                      # P1: anomaly of GPP over REGION in 2016; screen first
    acc = _acc(study)
    acc.screen(2016, REGION)
    refusals = {
        "field 'TOTECOSYSC'": lambda: acc.get_map("control", "TOTECOSYSC", "2016"),
        "period '2015'": lambda: acc.get_map("control", "GPP", "2015"),
        "region 'elsewhere'": lambda: acc.anomaly("GPP", "2016", region="elsewhere"),
        "method 'difference'": lambda: acc.difference("control", "GPP", "2016", region=REGION),
        "series step 'day'": lambda: acc.get_series("control", "GPP", REGION, "day"),
        "method 'growth_peak'": lambda: acc.get_timing("control", "growth_peak", 2016),
        "setting anomaly.near_zero_share=0.05":
            lambda: acc.anomaly("GPP", "2016", region=REGION,
                                settings={"near_zero_share": 0.05}),
    }
    for why, call in refusals.items():
        with pytest.raises(DAC.AccessRefused, match="the reviewed plan declares no") as e:
            call()
        assert why in str(e.value) and "declare it first in an amendment" in str(e.value), why
    # what the plan declares runs, marked planned under its analysis
    acc.get_map("control", "GPP", "2016")
    acc.anomaly("GPP", "2016", region=REGION)
    acc.get_series("control", "GPP", REGION, "year")     # the step of a declared period's kind
    log = _log(study)
    refused = [r for r in log if "refused" in r]
    assert len(refused) == len(refusals)
    assert all(r["refused"].startswith("the gate: ") and "sha256" not in r and
               "planned" not in r for r in refused)
    ok = [(r["call"], r["planned"], r["plan_items"]) for r in log if "sha256" in r][1:]
    assert ok == [("get_map", True, ["P1"]), ("anomaly", True, ["P1"]),
                  ("get_series", True, ["P1"])]


def test_an_amendment_declares_a_lookup_first_and_its_lookups_are_not_planned(study):
    _approve(study)
    acc = _acc(study, script="early")
    acc.screen(2016, REGION)
    with pytest.raises(DAC.AccessRefused, match="field 'TOTECOSYSC'"):
        acc.anomaly("TOTECOSYSC", "2016", region=REGION)      # no amendment declares it yet
    a1 = AP.append_amendment(study["out"], "the screen points at the carbon stock",
                             {"fields": ["TOTECOSYSC"]})
    assert a1["at"].endswith("+00:00") and a1["after_lookups"] == 2 and a1["planned"] is False
    later = _acc(study, script="late", amendment="A1")
    assert later.anomaly("TOTECOSYSC", "2016", region=REGION)["method"] == "anomaly"
    later.anomaly("GPP", "2016", region=REGION)        # the plan's own stays planned
    # an amendment declares exactly what it names: another period is still refused
    with pytest.raises(DAC.AccessRefused, match="A1 declares no period '2015'"):
        later.anomaly("TOTECOSYSC", "2015", region=REGION)
    # a script that names no amendment reaches A1's lookup too, linked to A1
    _acc(study, script="other").get_map("control", "TOTECOSYSC", "2016")
    log = _log(study)
    marks = [(r["script"], r["call"], r["planned"], r.get("amendment"), r["plan_items"])
             for r in log if "sha256" in r]
    assert marks == [("early", "screen", True, None, ["screen"]),
                     ("late", "anomaly", False, "A1", []),
                     ("late", "anomaly", True, None, ["P1"]),
                     ("other", "get_map", False, "A1", [])]
    assert [r["keys"]["field"] for r in AP.lookups_of(study["out"], "A1")] == \
        ["TOTECOSYSC", "TOTECOSYSC"]
    assert sum("refused" in r for r in log) == 2


def test_an_exploratory_plan_may_declare_the_screen_alone(study):
    _approve(study, reply=SCREEN_ONLY)
    assert AP.load_plan(study["out"])["problems"] == []
    acc = _acc(study)
    rows = acc.screen(2016, REGION)["rows"]
    assert {"GPP", "TOTECOSYSC"} <= {r["field"] for r in rows}     # its own scope: every field
    with pytest.raises(DAC.AccessRefused, match="the plan declares no analysis"):
        acc.get_map("control", "GPP", "2016")
    AP.append_amendment(study["out"], "GPP stood out", {"fields": ["GPP"],
                                                        "methods": ["anomaly"]})
    # the amendment takes the screen's region and the periods of its year
    _acc(study, amendment="A1").anomaly("GPP", "2016-JJA", region=REGION)
    last = _log(study)[-1]
    assert (last["planned"], last["amendment"], last["plan_items"]) == (False, "A1", [])


# ── 2. the order the question type sets ─────────────────────────────────────

def test_an_exploratory_question_runs_the_screen_first_and_records_it(study):
    _approve(study)
    acc = _acc(study)
    with pytest.raises(DAC.AccessRefused, match="an exploratory question runs the screen first"):
        acc.anomaly("GPP", "2016", region=REGION)               # declared, but before the screen
    assert acc.region_mask(REGION).any()                        # reads no outcome: not held back
    assert not (study["out"] / AP.SCREEN_FILE).exists()
    scr = acc.screen(2016, REGION)
    rec = json.loads((study["out"] / AP.SCREEN_FILE).read_text())
    assert set(rec) == {"at", "script", "question_type", "label", "keys", "sha256", "scope",
                        "scoring_rule", "display_rule", "method", "version", "settings"}
    assert (rec["scope"]["region"], rec["scope"]["year"]) == (REGION, 2016)
    assert rec["scope"]["fields"] == list(dict.fromkeys(r["field"] for r in scr["rows"]))
    assert set(rec["scope"]["periods"]) == set(M.screen_periods(2016))
    assert rec["display_rule"] == M.DISPLAY_RULE
    assert "standard deviation" in rec["scoring_rule"] and "near-constant" in rec["scoring_rule"]
    assert rec["label"] == AP.EXPLORATORY_LABEL and rec["question_type"] == "exploratory"
    line = _log(study)[-1]
    assert line["screen_record"] == AP.SCREEN_FILE and line["sha256"] == rec["sha256"]
    acc.anomaly("GPP", "2016", region=REGION)                   # now it runs
    assert _log(study)[-1]["planned"] is True


def test_a_hypothesis_question_runs_the_screen_last_and_labels_what_it_turns_up(study):
    _hypothesis(study)
    plan = AP.load_plan(study["out"])
    assert plan["question_type"] == "hypothesis" and plan["problems"] == []
    acc = _acc(study)
    # THE SCREEN WAITS FOR THE CLOSE OF THE PLANNED PHASE (the user's decision
    # on G part 2, 2026-11-08), no longer for every declared lookup
    with pytest.raises(DAC.AccessRefused, match="runs the screen last") as e:
        acc.screen(2016, REGION)
    assert "the planned-analysis phase is not closed yet" in str(e.value)
    acc.anomaly("GPP", "2016", region=REGION)                   # P1
    acc.growth_peak(2016, REGION)                               # P2
    with pytest.raises(DAC.AccessRefused, match="planned_phase.json"):
        acc.screen(2016, REGION)          # every declared lookup made: it waits for the close
    AP.close_planned_phase(study["out"], when="before the screen", by="the test")
    acc.screen(2016, REGION)
    done = [r for r in _log(study) if "sha256" in r]
    assert [(r["call"], r["planned"]) for r in done] == [("anomaly", True), ("growth_peak", True),
                                                          ("screen", False)]
    assert done[-1]["label"] == AP.SCREENING_LABEL == "found by screening, not tested"
    assert done[-1]["after_close"] is True and "after_screen" not in done[-1]
    assert json.loads((study["out"] / AP.SCREEN_FILE).read_text())["label"] == \
        AP.SCREENING_LABEL
    # what the screen turns up is reached by an amendment: labelled exploratory
    # (no longer "found by screening, not tested"), never planned, and marked
    # as after the screen and after the close
    a1 = AP.append_amendment(study["out"], "the screen showed the carbon stock",
                             {"fields": ["TOTECOSYSC"], "methods": ["anomaly"]})
    assert a1["after_screen"] is True and a1["label"] == AP.EXPLORATORY_LABEL
    assert a1["after_close"] is True
    _acc(study, amendment="A1").anomaly("TOTECOSYSC", "2016", region=REGION)
    last = _log(study)[-1]
    assert (last["planned"], last["amendment"], last["label"]) == (False, "A1",
                                                                   AP.EXPLORATORY_LABEL)
    assert last["after_close"] is True and last["after_screen"] is True


def test_the_plan_check_holds_the_screen_to_the_question_type():
    def probs(plan, qt):
        return AP.plan_problems(plan, ["GPP"], [REGION], [2015], qt)
    assert probs(PLAN_REPLY, "exploratory") == []
    late = dict(PLAN_REPLY, screen=dict(PLAN_REPLY["screen"], when="last"))
    assert any("runs it first" in p for p in probs(late, "exploratory"))
    assert any("runs it last" in p for p in probs(PLAN_REPLY, "hypothesis"))
    assert any(p.startswith("screen: not declared") for p in probs(dict(PLAN_REPLY, screen={}),
                                                                   "hypothesis"))
    as_analysis = dict(PLAN_REPLY, analyses=[dict(PLAN_REPLY["analyses"][0],
                                                  methods=["screen"], steps=["hour"])])
    got = probs(as_analysis, "exploratory")
    assert any("declared in the plan's screen entry" in p for p in got)
    assert any("no series step 'hour'" in p for p in got)


# ── 3. a method's full contract, on demand ──────────────────────────────────

def test_a_methods_full_contract_is_read_before_the_review_and_is_not_a_lookup(study):
    acc = _acc(study, script="planning")               # no plan yet, let alone a review
    with pytest.raises(DAC.AccessRefused):
        acc.get_map("control", "GPP", "2016")          # every lookup is held back
    doc = (ROOT / "docs" / "ANALYSIS_METHODS.md").read_text()
    for i, mid in enumerate(M.METHODS, 1):
        c = acc.method_contract(mid)
        start = doc.index(f"## {i}. `{mid}` (version")
        assert doc[start:doc.index("\n## ", start)] == c["text"], mid
        m = M.METHODS[mid]
        assert (c["version"], c["text_revision"]) == (m.version, m.text_revision)
        assert c["status"]["review"] == json.loads(json.dumps(m.status["review"]))
        assert [s["name"] for s in c["settings"]] == [s.name for s in m.settings]
        for part in ("**Numerical contract.**", "**Limitations.**", "**Validation evidence**",
                     "**Status** (each kept apart):", "review: "):
            assert part in c["text"], (mid, part)
    with pytest.raises(DAC.AccessError, match="no method 'guess'"):
        acc.method_contract("guess")
    # logged as contract reads, never among the lookups
    reads = AP.contract_reads(study["out"])
    assert [r["keys"]["method"] for r in reads] == list(M.METHODS) + ["guess"]
    assert all(r["contract_read"] is True and r["script"] == "planning" for r in reads)
    assert [r["call"] for r in _log(study)] == ["get_map"]          # the one refused lookup
    # and it counts for nothing a lookup counts for: a hypothesis screen still
    # waits (for the close of the planned phase, since 2026-11-08)
    _hypothesis(study)
    acc = _acc(study)
    acc.method_contract("anomaly")
    with pytest.raises(DAC.AccessRefused, match="the planned-analysis phase is not closed"):
        acc.screen(2016, REGION)
    assert AP.pending_lookups(study["out"]) == AP.declared_lookups(AP.load_plan(study["out"]))


# ── 4. timing settings: only what timing.nc holds ───────────────────────────

def test_a_timing_setting_timing_nc_does_not_hold_is_refused_not_computed(study):
    _approve(study, reply=PLAN_ALL)
    acc = _acc(study)
    acc.screen(2016, REGION)
    # no amendment: the gate refuses it and names what timing.nc holds
    with pytest.raises(DAC.AccessRefused,
                       match=r"window=9 is not a setting timing.nc holds \(it holds the "
                             r"default 7 and the contract's alternatives \[15\]\)"):
        acc.growth_peak(2016, REGION, settings={"window": 9})
    # declared by an amendment it is still refused, by what timing.nc holds
    AP.append_amendment(study["out"], "a wider window",
                        {"methods": ["growth_peak"], "settings": {"growth_peak": {"window": 9}}})
    am = _acc(study, amendment="A1")
    for call in (lambda: am.get_timing("control", "growth_peak", 2016, settings={"window": 9}),
                 lambda: am.growth_peak(2016, REGION, settings={"window": 9})):
        with pytest.raises(DAC.AccessError) as e:
            call()
        msg = str(e.value)
        assert not isinstance(e.value, DAC.AccessRefused)        # the data said no, not the gate
        assert "timing.nc of control holds no setting window=9; it holds default (window=7" \
            in msg and "window=15 (window=15" in msg and "never computed on the fly" in msg
    # a stored setting is answered from timing.nc, alone when it is named
    one = am.get_timing("control", "growth_peak", 2016, settings={"window": 15})
    assert list(one) == ["window=15"] and one["window=15"]["settings"]["window"] == 15
    assert set(am.get_timing("control", "growth_peak", 2016)) == {"default", "window=15"}
    refused = [r for r in _log(study) if "refused" in r]
    assert len(refused) == 3 and not any("sha256" in r for r in refused)
    # a plan that names it fails the code check, so it cannot pass the review
    bad = dict(PLAN_ALL, analyses=[dict(PLAN_ALL["analyses"][3],
                                        settings={"growth_peak": {"window": 9}})])
    assert any("growth_peak window=9 is not a setting timing.nc holds" in p
               for p in AP.plan_problems(bad, ["GPP"], [REGION], [2015], "exploratory"))


# ── 6. the analysis review never reaches a simulation ───────────────────────

def test_the_plan_and_lookup_path_cannot_reach_a_simulation(study, monkeypatch):
    rv = _approve(study, reply=PLAN_ALL)
    assert rv["approval"] == AP.APPROVES
    for words in ("approves this analysis plan only", "never authorizes a simulation",
                  "user's own authorization", "design review's yes", "SLURM"):
        assert words in rv["approval"], words
    # the two modules neither import nor name anything that builds or submits a run
    for rel in ("src/agents/analysis/analysis_plan.py", "mcp/elm-mcp/src/domain_access.py"):
        text = (ROOT / rel).read_text()
        mods = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
        assert not {m.split(".")[-1] for m in mods} & set(JOB_MODULES + ("subprocess",)), rel
        assert "sbatch" not in text and "os.system" not in text, rel
    # a fresh interpreter that loads the whole lookup path loads no such module
    src = ROOT / "src"
    probe = ("import sys; sys.path[:0] = [%r, %r, %r]\n"
             "from agents.analysis import analysis_plan, script_runner, step2_investigate\n"
             "import domain_access, domain_methods, domain_suite, domain_catalog, "
             "domain_controls, make_methods_doc\n"
             "print(sorted(m for m in sys.modules if m.split('.')[-1] in %r))"
             % (str(src), str(ROOT / "mcp" / "elm-mcp" / "src"), str(ROOT / "tools"),
                JOB_MODULES))
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         timeout=300)
    assert got.returncode == 0, got.stderr[-2000:]
    assert got.stdout.strip() == "[]", got.stdout
    # the runner starts nothing but the interpreter on the saved script ...
    started = []
    real = SR.subprocess.run

    def spy(cmd, *a, **k):
        started.append(list(cmd))
        return real(cmd, *a, **k)
    monkeypatch.setattr(SR.subprocess, "run", spy)
    code = ('suite.screen(2016, "vegetated cells")\nx = suite.get_map("control", "GPP", '
            '"2016")\ndraw_map(x, out_path=out_path)\nresult = {"n": 3}\n')
    ctx, out = study["ctx"], study["out"]
    ran = SR.run(code, ctx, out / "f.png", script_path=out / "f.py",
                 lookup_tag={"script": "f", "amendment": None})
    assert ran["ok"] is True, ran.get("error")
    assert started == [[sys.executable, str((out / "f.py").resolve())]]
    # ... and a script that would start anything else is refused before it runs
    for bad in ('import subprocess\nsubprocess.run(["sbatch", "job.sh"])\n',
                'import os\nos.system("sbatch job.sh")\n'):
        r = SR.run(bad, ctx, out / "g.png", script_path=out / "g.py",
                   lookup_tag={"script": "g", "amendment": None})
        assert r["ok"] is False and r["error"].startswith("refused before running")
    assert len(started) == 1
