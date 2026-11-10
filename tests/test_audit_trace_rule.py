"""A claim that fails the evidence trace is struck alone (2026-11-10),
synthetic, with no model call.

    pytest tests/test_audit_trace_rule.py -v

Step 3's audit struck a claim under four rules; the trace of a claim's
periods, regions and numbers ran only at step 4, where one failing claim
withheld the whole report. The audit now applies the trace as its last rule:
the claim that fails is struck with the trace's own reasons, and the claims
that pass are reported. A summary sentence that rested on a struck claim is
left out of what the page prints when it gives a number no kept claim's
record gives.
"""
from agents.analysis import step3_interpret as S3
from agents.analysis import step4_report as S4

INV = {"findings": [{"id": "f1", "n": 9, "result": {"mean": 31.4}}]}
GOOD = {"claim": "The mean is 31.4", "finding_id": "f1", "values": [31.4]}
LOOSE = {"claim": "It ran above normal from April to June", "finding_id": "f1", "values": []}


def _tracer(claim):
    return ["it names April, which none of its records names",
            "it names June, which none of its records names"] if "April" in claim["claim"] \
        else []


def test_the_rule_strikes_with_the_traces_own_reasons_and_needs_a_tracer():
    assert S3.rule_trace(GOOD, _tracer) is None
    assert S3.rule_trace(LOOSE, None) is None             # no suite: nothing to trace to
    assert S3.rule_trace(LOOSE, _tracer) == (
        "it fails the evidence trace: it names April, which none of its records names; "
        "it names June, which none of its records names")

    def broken(claim):
        raise KeyError("L9")
    assert S3.rule_trace(GOOD, broken) == \
        "it fails the evidence trace: it could not be traced (KeyError: 'L9')"


def test_the_audit_strikes_the_failing_claim_alone_and_keeps_the_rest():
    r = S3.audit([GOOD, LOOSE], INV, [], tracer=_tracer)
    assert r["kept"] == [GOOD] and (r["n_claims"], r["n_struck"]) == (2, 1)
    s, = r["struck"]
    assert s["claim"] == LOOSE["claim"] and s["struck_by"] == "trace"
    assert s["struck_because"].startswith("it fails the evidence trace: it names April")
    # without a tracer the audit is as it was: both pass its four other rules
    assert S3.audit([GOOD, LOOSE], INV, [])["n_struck"] == 0
    # the trace is the LAST rule: an earlier rule's reason wins
    r = S3.audit([dict(LOOSE, finding_id="nope")], INV, [], tracer=_tracer)
    assert r["struck"][0]["struck_by"] == "cites"


def test_the_page_prints_the_summary_without_the_sentences_that_were_left_out():
    base = {"question": "q", "status": "supported", "generated_at": "2026-11-10",
            "claims": [], "answer": "A kept sentence. A sentence on a struck claim, 7.3."}
    page = S4.render_markdown(dict(base))
    assert "A sentence on a struck claim, 7.3." in page
    kept = dict(base, answer_kept="A kept sentence.",
                answer_left_out=[{"sentence": "A sentence on a struck claim, 7.3.",
                                  "unsupported": ["7.3"]}])
    page = S4.render_markdown(kept)
    assert "A kept sentence." in page and "A sentence on a struck claim" not in page
    assert "*1 sentence of the Analyzer's summary is left out here:" in page
    none = S4.render_markdown(dict(base, answer_kept="", answer_left_out=[{}, {}]))
    assert "(no sentence of the summary rests on the kept claims alone)" in none
    assert "*2 sentences of the Analyzer's summary are left out here:" in none


# ── a what-if study's baseline, and an amendment that names no setting ──────
# (2026-11-10, from the first live what-if study through the rebuilt Analyzer)

def test_an_amendment_whose_settings_name_no_method_is_refused_with_what_they_should_be(
        tmp_path, monkeypatch):
    import pytest
    from agents.analysis import analysis_plan as AP
    monkeypatch.setattr(AP, "check_gate", lambda out: None)
    with pytest.raises(ValueError, match=r"settings are \{method: \{setting: value\}\}, and "
                                         r"\['normal_years'\] is not a method"):
        AP.append_amendment(tmp_path, "the screen needs normal years",
                            {"methods": ["screen"], "settings": {"normal_years": [2016, 2017]}})
    ok = AP.append_amendment(tmp_path, "a wider window",
                             {"methods": ["growth_peak"], "settings": {"growth_peak": {"window": 15}}})
    assert ok["adds"]["settings"] == {"growth_peak": {"window": 15}}
    # one recorded before the refusal existed names no setting and stops nothing
    old = {"adds": {"settings": {"normal_years": [2016, 2017]}}}
    assert AP.named_settings({"analyses": []}, amendments=[old, ok]) == {
        "growth_peak": {"window": [15]}}


def test_the_baseline_is_the_planners_or_else_the_controls_own(monkeypatch):
    from agents.analysis import analysis_plan as AP

    class Ctx:
        def __init__(self, years):
            self.plan = {"experiment": {"control": {"name": "synth"},
                                        "analysis": {"normal_years": years}}}
    monkeypatch.setattr(AP, "_registry_entry", lambda name: {"normal_years": [2016, 2017]}
                        if name == "synth" else {})
    assert AP.planner_normal_years(Ctx([2018, 2019])) == [2018, 2019]
    assert AP.normal_years_from(Ctx([2018, 2019])) == "the Planner"
    assert AP.planner_normal_years(Ctx([])) == [2016, 2017]
    assert AP.normal_years_from(Ctx([])) == "the control's own (the Planner named none)"
    monkeypatch.setattr(AP, "_registry_entry", lambda name: {})
    assert AP.planner_normal_years(Ctx([])) == [] and AP.normal_years_from(Ctx([])) == "none named"
