"""One definition per model-facing schema (the user's decision on G part 2,
2026-11-08): what the model is told to write and what the code checks come
from one definition, so the two cannot drift apart.

    pytest tests/test_analyzer_g2_schemas.py -v

FOUR DEFINITIONS, EACH HELD BOTH WAYS. The analysis plan (analysis_plan.
PLAN_PARTS and PLAN_RULES; step 2a's task and plan_problems), the plan
review's reply (REVIEW_REPLY; PLAN_REVIEW_ASK and the review), step 2b's
reply (step2_investigate's parts; SUITE_TASK and investigate) and step 3's
claims with the warning registry (claim_trace's CLAIM_PARTS, REPLY_PARTS and
WARNINGS; suite_task and the trace). For each: every field and value its
validator reads is in the rendered prompt, an example built from the
definition passes the validator's shape checks, and a field added to the
definition shows up in the prompt (no hand-written copy). The step 3 prompt
of a study without a suite is held byte for byte.

No model is called: SimpleLLMClient raises if anything constructs one.
"""
import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

import domain_methods as M
from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No model call of any kind: constructing the real client raises, where
    every step imports it."""
    import agents.llm_agent as LA

    class NoModel:
        def __init__(self, *a, **k):
            raise AssertionError("a real model client was constructed in a test")
    monkeypatch.setattr(LA, "SimpleLLMClient", NoModel)
    for mod in ("agents.planner", "agents.tool_loop", "agents.analyzer_agent"):
        try:
            monkeypatch.setattr(f"{mod}.SimpleLLMClient", NoModel)
        except (ImportError, AttributeError):
            pass


def _keys_read(fn, names=("a", "plan", "sc", "c", "claim", "f", "spec", "am", "got", "x",
                          "oc")) -> set:
    """The keys a function reads with name.get("key") or name["key"], from its
    source: what its validator reads of a reply."""
    src = inspect.getsource(fn)
    out = set()
    for n in names:
        out |= set(re.findall(rf"\b{n}\.get\(\"(\w+)\"", src))
        out |= set(re.findall(rf"\b{n}\[\"(\w+)\"\]", src))
    return out


# ── the analysis plan ────────────────────────────────────────────────────────

def test_the_plan_task_states_every_part_value_rule_and_setting():
    task = AP.plan_task()
    assert AP.PLAN_TASK == task                                     # rendered, not written
    flat = " ".join(task.split())
    for key, d in AP.PLAN_PARTS.items():
        assert f'"{key}":' in task and f"  - {key}: " in task, key
        for p in d["item"]:
            assert f'"{p.name}":' in task, (key, p.name)            # the JSON block
            assert f"{p.name} (" in flat, (key, p.name)             # what it holds
            for v in p.choices:
                shown = json.dumps(v) if p.holds == "choice" else str(v)
                assert shown in task, (key, p.name, v)
    for r in AP.PLAN_RULES:
        assert " ".join(r.text().split()) in flat, r.id      # rendered where it reads a module
    for v in AP.SERIES_STEPS + AP.SIGNS + AP.AGAINST + tuple(AP.SCREEN_WHEN.values()):
        assert v in task, v
    # each method's settings, read from domain_methods, under its own id
    for mid, m in M.METHODS.items():
        assert f"  - {mid}" in task, mid
        for s in m.settings:
            words = f"{s.name}, default {s.default!r}"
            assert words in flat, (mid, words)
            for a in s.alternatives:
                assert repr(a) in flat, (mid, s.name, a)
            if s.status == M.PROVISIONAL:
                assert f"{words}, no alternative ({s.units}), PROVISIONAL" in flat, (mid, s.name)
    assert "a timing measure (only these values)" in task
    for mid in M.TIMING_IDS:
        assert f"{mid}, a timing measure (only these values)" in flat, mid


def test_plan_problems_runs_the_rules_the_task_states(monkeypatch):
    # a plan breaking every rule gets one problem from each, in the rules' order
    bad = {"analyses": [{"id": "P1", "fields": ["NOPE"], "regions": ["nowhere"],
                         "methods": ["screen", "guess"], "steps": ["hour"],
                         "settings": {"by_band": True, "overlap": {"by_band": True},
                                      "growth_peak": {"window": 9}},
                         "periods": ["2015-07", "normal-07"]}],
           "baseline": {"normal_years": [2001]}, "screen": {}}
    got = AP.plan_problems(bad, ["GPP"], ["R"], [2016], "hypothesis")
    for words in ("no catalog field 'NOPE'", "no region 'nowhere'",
                  "the screen is declared in the plan's screen entry, not as an analysis",
                  "no method 'guess'", "no series step 'hour'", "settings for no method 'by_band'",
                  "overlap has no setting 'by_band'",
                  "growth_peak window=9 is not a setting timing.nc holds",
                  "normal-07: month maps are not in the suite",
                  "normal years [2001] are not the Planner's", "screen: not declared"):
        assert any(words in p for p in got), (words, got)
    # a rule added to the definition is checked and stated, with no other copy
    extra = AP.PlanRule("extra", "analysis", "extra: every analysis says why",
                        lambda a, env: [] if a.get("why") else [f"{a.get('id')}: says no why"])
    monkeypatch.setattr(AP, "PLAN_RULES", AP.PLAN_RULES + (extra,))
    assert "P1: says no why" in AP.plan_problems(bad, ["GPP"], ["R"], [2016], "hypothesis")
    assert "extra: every analysis says why." in AP.plan_task()


def test_every_key_the_plan_check_reads_is_a_part_of_the_plan():
    parts = {p.name for d in AP.PLAN_PARTS.values() for p in d["item"]} | set(AP.PLAN_KEYS)
    read = set()
    for r in AP.PLAN_RULES:
        read |= _keys_read(r.check)
    read |= _keys_read(AP.plan_problems)
    assert read, "the scan found nothing"
    # beside the model's parts, only what the framework writes into the plan
    assert read <= parts | set(AP.PLAN_RECORD_KEYS), read - parts - set(AP.PLAN_RECORD_KEYS)
    assert read - parts <= {"question_type"}


def _example_plan(fields, regions, years, qt):
    """A plan built from PLAN_PARTS alone, with real names: what a model
    following the task would write."""
    def value(p, key):
        if p.holds == "choice":
            return {"when": AP.SCREEN_WHEN[qt]}.get(p.name, p.choices[0])
        if p.name in ("fields",):
            return list(fields[:1])
        if p.name in ("field",):
            return fields[0]
        if p.name in ("regions",):
            return list(regions[:1])
        if p.name in ("region",):
            return regions[0]
        if p.name == "periods":
            return [str(years[0])]
        if p.name == "normal_years":
            return list(years)
        if p.holds in ("years",):
            return [years[0]]
        if p.holds == "year":
            return years[0]
        if p.holds == "settings":
            return {}
        if p.holds == "outputs":
            # one output of the example's own scope (plan_contract)
            return [{"id": "O1", "kind": "anomaly", "field": fields[0],
                     "period": str(years[0]), "region": regions[0]}]
        if p.name == "methods":
            return ["anomaly"]
        if p.name == "analysis":
            return "P1"
        if p.holds == "list":
            return []
        if p.holds in ("name", "words"):
            return p.example if p.holds == "name" else "words"
        raise AssertionError(f"a part this example does not know: {p}")
    out = {}
    for key, d in AP.PLAN_PARTS.items():
        item = {p.name: value(p, key) for p in d["item"]}
        out[key] = item if d["shape"] == "object" else \
            ([] if key == "expectations" and qt != "hypothesis" else [item])
    return out


@pytest.mark.parametrize("qt", AP.QUESTION_TYPES)
def test_an_example_plan_built_from_the_parts_passes_the_check_and_keeps_its_shape(qt):
    fields, regions, years = ["GPP", "H2OSNO"], ["vegetated cells"], [2016]
    ex = _example_plan(fields, regions, years, qt)
    assert AP.plan_problems(ex, fields, regions, years, qt) == []
    for key, d in AP.PLAN_PARTS.items():
        assert isinstance(ex[key], type(AP._shape_default(key))), key
    # a key of the wrong shape is replaced by an empty one, which the check names
    broken = dict(ex, screen=[ex["screen"]])
    assert not isinstance(broken["screen"], type(AP._shape_default("screen")))


def test_a_part_added_to_the_plan_shows_up_in_the_task(monkeypatch):
    parts = {k: dict(v) for k, v in AP.PLAN_PARTS.items()}
    parts["analyses"]["item"] = parts["analyses"]["item"] + (
        AP.Part("audience", "words", "who reads the answer"),)
    monkeypatch.setattr(AP, "PLAN_PARTS", parts)
    t = AP.plan_task()
    assert '"audience": "..."' in t and "audience (who reads the answer)" in " ".join(t.split())


def test_the_plan_text_is_kept_without_dashes_names_untouched():
    plan = {"comparisons": [{"id": "C1", "what": "2015 \u2014 against the normal"}],
            "analyses": [{"id": "P1", "question": "snow \u2013 and GPP", "why": "x\u2014y",
                          "regions": ["watershed, vegetated, 1000-1500 m"]}],
            "baseline": {"why": "the Planner\u2014s"}, "screen": {"why": "last \u2014 always"},
            "expectations": [{"text": "lower \u2014 E1"}], "competing": [{"text": "0.78–0.84"}]}
    out = AP._plain_parts(json.loads(json.dumps(plan)))
    assert "\u2014" not in json.dumps(out, ensure_ascii=False)
    assert out["competing"][0]["text"] == "0.78–0.84"           # a range keeps its dash
    assert out["analyses"][0]["regions"] == ["watershed, vegetated, 1000-1500 m"]


# ── the plan review's reply ─────────────────────────────────────────────────

OLD_PLAN_REVIEW_ASK_SHA = "faa8bbfacdfa67fb4dd9aeb3f50cc2fc839270259fb8840a7c29477217f06c8a"


def test_the_review_prompt_is_rendered_from_the_reply_the_review_reads(monkeypatch):
    # byte for byte the prompt question 1 to 3 were reviewed with
    assert hashlib.sha256(S3.PLAN_REVIEW_ASK.encode()).hexdigest() == OLD_PLAN_REVIEW_ASK_SHA
    for p in AP.REVIEW_REPLY:
        assert f'"{p.name}": ' in S3.PLAN_REVIEW_ASK, p.name
    assert '"follows": true | false' in S3.PLAN_REVIEW_ASK
    # the review reads exactly these names
    assert _keys_read(S3.read_review_reply) <= {p.name for p in AP.REVIEW_REPLY}
    r = S3.read_review_reply({"follows": True, "why": "w", "problems": "one problem"})
    assert r == {"follows": True, "why": "w", "problems": ["one problem"]}   # never its letters
    assert S3.read_review_reply({"follows": "yes"})["follows"] is False
    # a part added to the definition shows up in the prompt
    monkeypatch.setattr(AP, "REVIEW_REPLY", AP.REVIEW_REPLY + (
        AP.Part("confidence", "name", "how sure", "low"),))
    assert '"confidence": "low"' in S3._review_json_line()


# ── step 2b's reply ─────────────────────────────────────────────────────────

def test_step_2b_task_holds_every_part_the_code_reads(monkeypatch):
    task = S2.SUITE_TASK
    for p in S2.SUITE_REPLY_PARTS + S2.SUITE_FIGURE_PARTS:
        assert f'"{p.name}": ' in task, p.name
        for v in p.choices:
            assert json.dumps(v) in task, (p.name, v)
    for k in S2.AMENDMENT_KEYS + S2.SKIPPED_KEYS:
        assert f'"{k}": ' in task, k
    assert S2.AMENDMENT_KEYS == ("why",) + AP.SCOPE_KEYS + ("settings",)
    # what append_amendment keeps of an amendment is the definition's
    src = inspect.getsource(AP.append_amendment)
    assert "SCOPE_KEYS + (\"settings\",)" in src
    # every key investigate and propose read of a figure, a reply or a skip
    read = _keys_read(S2.investigate) | _keys_read(S2.propose) | _keys_read(S2._runs_screen)
    named = {p.name for p in S2.SUITE_REPLY_PARTS + S2.SUITE_FIGURE_PARTS} | \
        set(S2.AMENDMENT_KEYS) | set(S2.SKIPPED_KEYS) | {"prompt", "raw"}
    model_keys = read - {"figure", "script", "error", "result", "ok", "script_path", "n",
                         "blocked", "variables_missing", "withheld",
                         "finish_reason"}           # the client's word on the reply, not the model's
    assert model_keys <= named, model_keys - named
    # the ordering rule says the close
    flat = " ".join(task.split())
    for words in ("closes the planned phase (planned_phase.json", "the screen is refused before "
                  "that close", "is exploratory and never planned", "goes in `skipped`",
                  '["screen"] for the one figure that runs the plan\'s screen'):
        assert words in flat, words
    # and what carrying out an analysis means, in the contract's own words
    # (the user's decision after G part 2's fixes, 2026-11-08)
    from agents.analysis import plan_contract as PC
    assert " ".join(PC.STEP2_WORDS.split()) in flat
    # a part added shows up in the template
    monkeypatch.setattr(S2, "SUITE_FIGURE_PARTS", S2.SUITE_FIGURE_PARTS + (
        AP.Part("caption", "words", "the figure's caption", "<caption>"),))
    assert '"caption": "<caption>"' in "\n".join(S2._suite_reply_lines())


# ── step 3's claims and the warning registry ────────────────────────────────

def test_every_key_the_trace_and_the_audit_read_of_a_claim_is_a_part():
    parts = {p.name for p in CT.CLAIM_PARTS} | {p.name for p in CT.REPLY_PARTS} | \
        set(CT.OPEN_CAUSE_PARTS)
    read = set()
    for fn in (CT.check_claim, CT._headline_claims, CT.check_proposals, S3.rule_cites,
               S3.rule_declared_values, S3.rule_respects, S3.rule_percent):
        read |= _keys_read(fn, names=("c", "claim", "oc"))
    read |= _keys_read(S3.interpret, names=("spec",))
    produced = {"evidence_trace", "trace", "derivations", "traced", "verdict_asked_again",
                "struck_because", "struck_by", "finding"}
    # NOT A PART OF A SUITE STUDY'S REPLY (2026-11-08): its headline is a
    # claim's own text, chosen from "headline_rank"; "headline" is the
    # sentence a study without a suite writes (TASK_REPLY), and the per-claim
    # flag a reply recorded before the change carried, read to replay it
    other = {"headline"}
    assert read, "the scan found nothing"
    assert read - produced - other <= parts, (read - produced - other) - parts
    assert "headline" not in {p.name for p in CT.CLAIM_PARTS + CT.REPLY_PARTS}
    for k in ("evidence", "evidence_class", "expectation", "warnings", "headline_rank",
              "values", "claim", "finding_id", "caveats"):
        assert k in read | parts, k


def test_the_suite_task_holds_every_part_class_and_warning_code(monkeypatch):
    task = S3.suite_task()
    assert task.startswith(S3.TASK_BODY)
    assert S3.TASK_REPLY not in task                   # the generic template is replaced
    for p in CT.CLAIM_PARTS + CT.REPLY_PARTS:
        assert f'"{p.name}": ' in task, p.name
    for c in CT.EVIDENCE_CLASSES:
        assert f'"{c}"' in task, c
    for code, (label, _) in CT.WARNINGS.items():
        assert f"{code} (\"{label}\")" in task, code
    for kind in CT.PARAMETERISED_WARNINGS:
        assert f"{kind}:<x>" in task
        for x in CT.warning_vocabulary(kind):
            assert x in task, (kind, x)
    for words in ("open_causes", '"proposed_run"', '"changes"', '"params"'):
        assert words in task, words
    # a part added to the definition shows up in the template
    monkeypatch.setattr(CT, "CLAIM_PARTS", CT.CLAIM_PARTS + (
        CT.ReplyPart("confidence", '"low" | "high"', "how sure"),))
    assert '"confidence": "low" | "high"' in S3.suite_task()


def test_an_example_claim_built_from_the_parts_passes_the_trace_shape(tmp_path):
    """A record logged as the access logs one, and a claim built from
    CLAIM_PARTS that names it: the trace finds no shape problem (no missing
    record, class or warning)."""
    out = tmp_path / "04_analysis"
    out.mkdir()
    AP.LookupLog(out, script="fig")({
        "call": "anomaly", "keys": {"run": "control", "field": "GPP", "period": "2016",
                                    "region": "vegetated cells"},
        "method": "anomaly", "settings": None, "sha256": "x" * 64, "planned": True,
        "plan_items": ["P1"],
        "result": {"method": "anomaly", "field": "GPP", "period": "2016",
                   "region": "vegetated cells", "anomaly": 12.5, "covered": 1.0,
                   "normal_years": [2015]}})
    recs = CT.evidence_records(out)
    example = {"claim": "Over the vegetated cells, GPP in 2016 was 12.5 gC/m^2/yr above its "
                        "2015 normal.",
               "finding_id": "fig", "values": [12.5], "caveats": [], "evidence": ["L1"],
               "evidence_class": "observation", "expectation": None,
               "warnings": sorted(w["code"] for w in recs["L1"]["warnings"])}
    assert set(example) == {p.name for p in CT.CLAIM_PARTS}
    t = CT.check_claim(example, recs, {"question_type": "hypothesis"}, {}, ["vegetated cells"])
    assert t["problems"] == [], t["problems"]


def _record(**kw):
    rec = {"id": "L1", "call": kw.pop("call", "anomaly"), "method": kw.pop("method", None),
           "measure": None, "field": kw.pop("field", None), "label": kw.pop("label", None),
           "planned": kw.pop("planned", True), "amendment": kw.pop("amendment", None),
           "after_close": kw.pop("after_close", False)}
    rec.update(kw)
    return rec


def test_every_warning_code_the_records_can_carry_is_registered_and_in_the_template():
    """Each branch of record_warnings, row_warnings and _sensitivity_rows
    exercised with a synthetic record: every code it makes is in WARNINGS
    (or a parameterised code of the registry) and in step 3's template."""
    made = set()
    res = {"covered": 0.5, "missing_cell_values": [2016],
           "normal_left_out.2017": "absent",
           "variants.default.shift.eligible.both.cells": 3,
           "variants.default.shift.eligible.only_year.cells": 2,
           "variants.default.shift.eligible.only_year.area_share": 0.1,
           "variants.default.shift.area_share": 0.5,
           "variants.default.comparability": "restricted comparability: a partial-record date",
           "variants.default.distribution.reasons.no_snow": 4,
           "variants.default.distribution.flags.partial_record": 2,
           "near_constant": True, "status": "insufficient support",
           "across_settings.direction_reversal": True,
           "across_settings.minimum.variant": "default",
           "across_settings.minimum.median_shift": -3.0,
           "across_settings.maximum.variant": "days=7",
           "across_settings.maximum.median_shift": 4.0}
    for kw in ({}, {"label": AP.SCREENING_LABEL}, {"label": AP.EXPLORATORY_LABEL},
               {"after_close": True, "label": AP.EXPLORATORY_LABEL},
               {"amendment": "A1", "planned": False}, {"planned": False},
               {"field": "FPG"}, {"method": "growth_peak"}):
        made |= {w["code"] for w in CT.record_warnings(_record(**kw), res)}
    made |= {w["code"] for w in CT.record_warnings(_record(), dict(res, status="not scored yet"))}
    row = {"field": "FPG", "period": "2016", "status": "near-constant normal",
           "near_constant": True, "label": "not scored", "mark": "differs_from_control"}
    made |= {w["code"] for w in CT.row_warnings(_record(call="screen"), row, "rows.0")}
    made |= {w["code"] for w in CT.row_warnings(_record(call="screen"),
                                                dict(row, status="insufficient support"),
                                                "rows.1")}
    sens = {"sensitivity.0.eligible.both.cells": 3, "sensitivity.0.eligible.only_a.cells": 1}
    made |= {w["code"] for ws in CT._sensitivity_rows(_record(), sens).values() for w in ws}
    fixed = set(CT.WARNINGS)
    params = {c for c in made if ":" in c}
    assert made - params <= fixed, (made - params) - fixed
    assert fixed <= made, fixed - made                     # every registered code is reachable
    assert {"reason:no_snow", "flag:partial_record"} <= params
    task = S3.suite_task()
    for code in made:
        kind, _, x = code.partition(":")
        assert (f"{kind}:<x>" in task and x in task) if x else f"{code} (\"" in task, code
    # a code the registry does not hold cannot be made
    with pytest.raises(KeyError):
        CT._warn({"id": "L1"}, "invented_code", "why")
    with pytest.raises(KeyError):
        CT.warning_label("reason:not_a_reason")
    # the labels are the ones the claims carried before the registry
    assert {c: CT.warning_label(c) for c in ("coverage", "not_planned", "event_mark",
                                             "reason:no_snow", "flag:two_peaks")} == {
        "coverage": "area share covered", "not_planned": "not planned", "event_mark": "marked",
        "reason:no_snow": "no snow", "flag:two_peaks": "two peaks"}
    # every literal code in the source is registered
    src = inspect.getsource(CT.record_warnings) + inspect.getsource(CT.row_warnings) + \
        inspect.getsource(CT._sensitivity_rows)
    lits = set(re.findall(r'_warn\(rec, "(\w+)"', src)) | {v[0] for v in CT._SCREEN_STATUS.values()}
    assert lits <= fixed, lits - fixed


# ── a study without a suite keeps its step 3 prompt byte for byte ───────────

OLD_TASK_SHA = "27b226f277eb9cdb76cb5f9f1d018d27ed6dbc38b768d06943b4a1c5d8c206b6"
# The briefs below as the code before 2026-11-08's changes rendered them
# (from the snapshot of that code, the same inputs): a column study and a
# whole-domain study without a suite.
NON_SUITE = {
    "plan": {"question": "Where does recharge \u2014 the water below the roots \u2014 rise?",
             "goals": ["recharge by elevation"], "period": {"yr_start": 2001, "yr_end": 2003}},
    "caveats": [{"id": "wtd", "severity": "blocking", "applies_to": "water table (ZWT)",
                 "statement": "ELM diagnoses the water table from an aquifer store."},
                {"id": "ctx1", "severity": "context", "applies_to": "runoff",
                 "statement": "Runoff is not routed."}],
    "comparison": {"findings": [{"id": "compare_swe", "n": 4, "scale": "places",
                                 "question": "SWE against SNOTEL", "result": {"bias": -12.5}}],
                   "caveats": []},
    "investigation": {"notes": "Two figures \u2014 one per band.",
                      "findings": [{"id": "recharge_by_band", "n": 19, "scale": "overall",
                                    "question": "How does recharge vary with elevation?",
                                    "result": {"median": 31.4, "range": [12.0, 95.0], "n": 19}}],
                      "caveats": [{"id": "figure_failed:x", "statement": "x failed \u2014 no n"}]},
    "regional": {"runs": {"control": {"fields": {"GPP": {"how": "sum", "units": "gC/m^2/yr"},
                                                 "TOTVEGC": {"how": "mean", "units": "gC/m^2"}},
                                      "monthly": {"x": 1}}}},
}
NON_SUITE_SHA = {"columns": "26e97ea443ed374b1da4097989eec02d59a7a5f4f18f0efc8a6834c2660f053b",
                 "domain": "7a1837d6c555deb8579f1d074ad2177c26ab9afae72620f2cf310d4fd31fd516"}


class _NonSuiteCtx:
    def __init__(self, domain):
        d = json.loads(json.dumps(NON_SUITE))
        self.plan = d["plan"]
        if domain:
            self.plan = dict(self.plan, experiment={"control": {"name": "synth"}, "runs": []})
        self.caveats = d["caveats"]
        self.columns = []
        self.is_domain = domain
        self.run_dir = Path("/nonexistent/study")
        self.data = {"regional": d["regional"]} if domain else {}


def non_suite_prompt(domain: bool) -> str:
    d = json.loads(json.dumps(NON_SUITE))
    ctx = _NonSuiteCtx(domain)
    brief = S3.review_brief(ctx, d["comparison"], d["investigation"])
    return S3._content(brief, [], False)


def test_a_study_without_a_suite_keeps_its_step_3_prompt_byte_for_byte():
    assert hashlib.sha256(S3.TASK.encode()).hexdigest() == OLD_TASK_SHA
    assert S3.TASK == S3.TASK_BODY + S3.TASK_REPLY
    for kind, domain in (("columns", False), ("domain", True)):
        assert S3.suite_records(_NonSuiteCtx(domain)) is None, kind
        got = non_suite_prompt(domain)
        assert got.endswith("\n" + S3.TASK), kind
        assert hashlib.sha256(got.encode()).hexdigest() == NON_SUITE_SHA[kind], kind
