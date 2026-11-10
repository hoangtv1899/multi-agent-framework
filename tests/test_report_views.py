"""The scientific report and its audit appendix (the user's decision of
2026-11-09), synthetic, with no model call.

    pytest tests/test_report_views.py -v

R   The report's narrative uses only what the audit kept: a summary sentence
    whose number no kept claim's record gives is left out of it, each kept
    sentence names its claims, and the limitations its claims carry stand
    beside it. The report and the appendix link to each other, and the
    figures by relative names.
A   The appendix holds the whole page the report used to be, its headings
    one level down and a few documented tidyings only.
P   The percent rule still judges a claim whole; its reason now names the
    field whose value the percent is.
"""
import re

import pytest

from g2_replay_support import forbid_model_clients

from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import report_views as RV           # noqa: E402

CT._elm_modules()


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _rec(rid, values, field="GPP", region="watershed, vegetated", call="anomaly",
         years=("2015",), percent_allowed=True, **kw):
    return {"id": rid, "values": dict(values), "field": field, "region": region,
            "call": call, "names": {"years": list(years), "periods": []},
            "field_labels": [], "keys": {}, "settings": {}, "comparison": {},
            "percent_allowed": percent_allowed,
            "percent_why_not": None if percent_allowed else "a temperature: its zero is set "
                                                            "by its unit", **kw}


# made-up numbers, in the shape a study's records hold them
RECS = {
    "L1": _rec("L1", {"variants.default.shift.median": -48.3}, field="H2OSNO",
               call="snow_gone", years=("2015", "2016", "2020")),
    "L5": _rec("L5", {"anomaly": -37.46271, "percent": -10.91634},
               years=("2015", "2016", "2017", "2018", "2019", "2020")),
    "L12": _rec("L12", {"anomaly": 1.7263}, field="TSA", percent_allowed=False),
    "L16": _rec("L16", {"anomaly": -0.0061,
                        "percent": -0.5231}, field="TLAI"),
}
CLAIMS = [{"claim": "Over the watershed the snow went 48.3 days earlier in 2015.",
           "evidence": ["L1"], "values": [-48.3],
           "evidence_class": "tested expectation",
           "warnings": ["restricted_comparability", "flag:partial_record"]},
          {"claim": "Summer GPP was 37.5 gC/m2 below normal, 10.9 percent lower.",
           "evidence": ["L5"], "values": [-37.46271, -10.91634],
           "evidence_class": "tested expectation", "warnings": []}]
ANSWER = ("In the model the snow went about 48.3 days earlier in 2015 than in the 2016 to 2020 "
          "normal years. Summer uptake of carbon by plants was 37.5 gC/m2, or 10.9 percent, "
          "below the normal-year summer. Warmer summer air, 1.73 K, was also present.")
COMBINED = ("Summer air was 1.73 K warmer than the normal years while summer leaf area was "
            "0.0061 lower, a change of 0.52 percent.")


# ── R: the summary holds to the kept claims ─────────────────────────────────

def test_a_summary_sentence_is_kept_only_when_a_kept_claim_gives_its_numbers():
    rows = CT.summary_support(ANSWER, CLAIMS, RECS, {}, {})
    assert [r["claims"] for r in rows] == [[1], [2], []]
    assert [r["unsupported"] for r in rows] == [[], [], ["1.73 K"]]
    # years and decimals are no sentence ends and no unsupported numbers
    assert rows[0]["sentence"].endswith("normal years.") and "2016" not in rows[0]["unsupported"]


def test_a_number_a_withheld_claim_gave_is_not_kept_even_from_its_record():
    # the withheld claim's own record (L12) is no kept claim's record
    assert CT.summary_support("Warmer summer air, 1.73 K, was also present.", CLAIMS, RECS,
                              {}, {})[0]["unsupported"] == ["1.73 K"]
    kept = CLAIMS + [{"claim": "Summer air was 1.73 K warmer.", "evidence": ["L12"],
                      "values": [1.7263], "evidence_class": "observation"}]
    assert CT.summary_support("Warmer summer air, 1.73 K, was also present.", kept, RECS,
                              {}, {})[0]["claims"] == [3]


# ── P: whose percent it is ───────────────────────────────────────────────────

def test_the_percent_rule_still_strikes_a_combined_claim_but_names_whose_percent_it_is():
    recs = [RECS["L12"], RECS["L16"]]
    assert CT.percent_attribution(COMBINED, recs) == {"0.52 percent": ["TLAI"]}
    why = CT.percent_problems(COMBINED, recs)
    assert len(why) == 1                                             # still struck
    assert "its percent is of TLAI, whose notes allow one" in why[0]
    assert "gives TSA, whose notes give none" in why[0]
    assert "give each quantity a claim of its own" in why[0]


def test_separate_claims_for_separate_quantities_pass_the_percent_rule():
    assert CT.percent_problems("Summer air was 1.73 K warmer than the normal years.",
                               [RECS["L12"]]) == []
    assert CT.percent_problems("Summer leaf area was 0.0061 lower, a change of 0.52 percent.",
                               [RECS["L16"]]) == []


def test_a_percent_of_temperature_is_still_struck_as_a_percent_of_temperature():
    why = CT.percent_problems("Summer air was 1.73 K warmer, 0.7 percent higher.",
                              [RECS["L12"]])
    assert why == ["a percent of TSA, whose notes give none: a temperature: its zero is set "
                   "by its unit"]


# ── the eligibility wording ─────────────────────────────────────────────────

def test_the_eligibility_wording_lists_cell_groups_only():
    res = {"shift.eligible.both.cells": 3050, "shift.eligible.both.area_share": 0.99,
           "shift.eligible.only_this.cells": 4, "shift.eligible.only_this.area_share": 0.001,
           "shift.eligible.region_area_km2": 1830.5, "shift.area_share": 1.0,
           "sensitivity.0.eligible.both.cells": 3050,
           "sensitivity.0.eligible.only_default.cells": 4,
           "sensitivity.0.eligible.region_area_km2": None}
    rec = {"id": "L1", "call": "snow_gone", "field": "H2OSNO"}
    words = " ".join(w["reason"] for w in CT.record_warnings(rec, res))
    words += " ".join(w["reason"] for ws in CT._sensitivity_rows(rec, res).values() for w in ws)
    assert "None cells" not in words and "region area" not in words
    assert "only this 4 cells" in words and "only default 4 cells" in words


# ── the proposed runs, in words ──────────────────────────────────────────────

def test_a_proposed_run_says_what_it_tests_and_what_it_cannot_isolate():
    cool = RV.experiment_sentence({"id": "weather_scale", "params": {
        "temperature_K": -2.0, "precip_factor": 1.0, "months": [1, 2, 3, 4, 5], "region": None}})
    assert cool.startswith("A run with the air 2 K cooler in January to May, over the whole "
                           "domain")
    assert "how the results respond to cooler air in those months" in cool
    assert "cannot isolate any one of them, snow included" in cool
    wet = RV.experiment_sentence({"id": "weather_scale", "params": {
        "temperature_K": 0, "precip_factor": 1.5, "months": [1, 2, 3, 11, 12]}})
    assert "rain and snow times 1.5 in November to March" in wet
    assert "air temperature" not in wet and "more rain and snow" in wet


# ── R and A: the two views of one record ─────────────────────────────────────

def _report(tmp_path):
    fig = tmp_path / "rec" / "snow_fig.png"
    fig.parent.mkdir()
    fig.write_bytes(b"\x89PNG\r\n")
    return {"question": "Did an early snow loss accompany the summer GPP decline?",
            "status": "supported", "generated_at": "2026-11-09", "model": "elm",
            "answer": ANSWER, "claims": [dict(c, finding_id="snow_fig") for c in CLAIMS],
            "headline": CLAIMS[0]["claim"],
            "headline_selection": {"chosen": 1, "order": [1, 2]},
            "withheld": [{"claim": COMBINED, "finding_id": "x", "struck_by": "percent",
                          "struck_because": "gives a percent of TSA, whose notes give none"}],
            "caveats": [{"id": "plan_caveat_1", "severity": "qualify",
                         "statement": "A limitation that names 2015."},
                        {"id": "plan_caveat_2", "severity": "qualify",
                         "statement": "A limitation that names 2021."}],
            "open_causes": [{"cause": "Whether the early snow loss, rather than the warmer "
                                      "summer, set the shortfall.",
                             "proposed_run": {"changes": [{"id": "weather_scale", "params": {
                                 "temperature_K": -2.0, "months": [1, 2, 3, 4, 5]}}]}}],
            "figures": {"investigation": {"snow_fig": str(fig)}},
            "numbers_check": {"pass": True, "n_compared": 2042},
            "evidence_numbers": {"pass": True, "n_compared": 484}}


def test_the_report_uses_only_kept_results_and_links_its_appendix(tmp_path):
    rep = _report(tmp_path)
    out = tmp_path / "views"
    RV.write_views(rep, out, RECS, {}, {}, analysis_dir=tmp_path / "rec")
    main, audit = (out / RV.MAIN_MD).read_text(), (out / RV.AUDIT_MD).read_text()
    # the narrative: the kept sentences, each with its claims, and nothing withheld
    assert "earlier in 2015 than in the 2016 to 2020 normal years (claim 1)." in main
    assert "or 10.9 percent, below the normal-year summer (claim 2)." in main
    assert "1.73" not in main and "0.52" not in main and "leaf area" not in main
    # the limitations its claims carry, beside the answer, in the registry's words
    answer = main.split("## The answer")[1].split("## Model and data")[0]
    assert "**restricted comparability** (claim 1)" in answer
    assert "**partial record** (claim 1)" in answer
    assert "plan_caveat_1" in answer and "plan_caveat_2" not in answer      # 2015 only
    assert "**snow_gone** (claim 1): The first day of the first sustained melt-out" in answer
    assert "No run in this study isolates a cause" in answer
    # the open cause leans on a withheld result: in the appendix, not the report
    assert "warmer summer" not in main
    assert "Proposed sensitivity experiment (not run):** A run with the air 2 K cooler in " \
           "January to May" in main
    # links: each way, and the figure by a relative name that is there
    assert f"[audit appendix]({RV.AUDIT_MD})" in main and f"[the report]({RV.MAIN_MD})" in audit
    figs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", main)
    assert figs == ["snow_fig.png"] and (out / "snow_fig.png").is_file()
    assert "/tmp" not in main and str(tmp_path) not in main
    # one short verification statement, no undefined n
    assert main.count("made again from the model output") == 1 and "n = " not in main
    # the appendix: the left-out sentence and the withheld claim, with why
    assert "left out of the report: \"Warmer summer air, 1.73 K, was also present.\" It " \
           "prints 1.73 K" in audit
    assert COMBINED in audit and "struck by `percent`" in audit
    order = [main.index(h) for h in ("## The answer", "## Model and data", "## Key findings",
                                     "## Figures", "## Remaining uncertainty and proposed "
                                                   "sensitivity experiments",
                                     "## Verification")]
    assert order == sorted(order)


def test_the_appendix_holds_the_whole_record_page(tmp_path):
    from agents.analysis import step4_report as S4
    rep = _report(tmp_path)
    page = S4.render_markdown(rep).splitlines()
    held = RV.record_page(rep).splitlines()
    assert len(held) == len(page)
    for a, b in zip(page, held):
        if a != b:                       # only the documented tidyings
            assert b == "#" + a or str(tmp_path) in a or "n = " in a or "None cells" in a, a
    audit = RV.render_audit(rep, CT.summary_support(ANSWER, rep["claims"], RECS, {}, {}), RECS)
    assert "\n".join(held) in audit


# ── the three-part page (2026-11-09) ─────────────────────────────────────────

Q3P = "Did an early snow loss in 2015 accompany the summer GPP decline?"
PLAN3 = {"expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "lower"}],
         "competing": [{"text": "Warmer or drier air lowers photosynthesis directly.",
                        "fields": ["TSA", "RH2M"], "analysis": "P3"},
                       {"text": "A change in leaf area lowers GPP.", "fields": ["TLAI"],
                        "analysis": "P3"}],
         "analyses": [
             {"id": "P1", "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP",
                                       "period": "2015-JJA", "region": "watershed, vegetated"}]},
             {"id": "P2", "outputs": [{"id": f"O{m}", "kind": "anomaly", "field": "GPP",
                                       "period": f"2015-{m:02d}",
                                       "region": "watershed, vegetated"} for m in (6, 7)]
              + [{"id": "O9", "kind": "anomaly", "field": "GPP", "period": "2015-JJA",
                  "region": None},
                 {"id": "O10", "kind": "elevation_split", "region": "watershed, vegetated",
                  "inputs": ["O9"]}]}]}
RECS3 = {
    "L5": _rec("L5", {"anomaly": -37.46271, "percent": -10.91634},
               period="2015-JJA", plan_items=["P1"]),
    "L6": _rec("L6", {"anomaly": -6.41}, period="2015-06", plan_items=["P2"]),
    "L7": _rec("L7", {"anomaly": -19.3}, period="2015-07", plan_items=["P2"]),
    "L9": _rec("L9", {"parts.below 1000 m.mean": -24.6}, call="elevation_split",
               plan_items=["P2"]),
    "L12": _rec("L12", {"anomaly": 1.7263}, field="TSA", percent_allowed=False,
                period="2015-JJA", plan_items=["P3"]),
}
CLAIMS3 = [
    {"claim": "Summer GPP was 37.5 gC/m2 below normal, 10.9 percent lower.", "evidence": ["L5"],
     "values": [-37.46271, -10.91634], "evidence_class": "tested expectation",
     "warnings": [], "finding_id": "f"},
    {"claim": "Month by month, GPP was 6.41 gC/m2 below normal in June and 19.3 in July.",
     "evidence": ["L6", "L7"], "values": [-6.41, -19.3], "evidence_class": "observation",
     "warnings": [], "finding_id": "f"},
    {"claim": "Below 1000 m the summer GPP shortfall averaged 24.6 gC/m2.", "evidence": ["L9"],
     "values": [-24.6], "evidence_class": "observation", "warnings": [], "finding_id": "f"},
    {"claim": "Summer air was 1.73 K warmer than normal.", "evidence": ["L12"],
     "values": [1.7263], "evidence_class": "observation", "warnings": [],
     "finding_id": "f"}]
ANSWER3 = ("Yes, in the model. Summer GPP was 10.9 percent below normal. The shortfall was "
           "largest in July. The shortfall sat in every elevation band. Summer air was also "
           "1.73 K warmer. Either may have contributed.")


def _report3():
    return {"question": Q3P, "status": "supported", "generated_at": "2026-11-09",
            "answer": ANSWER3, "claims": CLAIMS3, "headline": CLAIMS3[0]["claim"],
            "headline_selection": {"chosen": 1, "order": [1, 2, 3, 4]},
            "withheld": [{"claim": "Summer relative humidity (RH2M) averaged 43.9 percent.",
                          "struck_by": "percent", "struck_because": "x"}],
            "caveats": [], "figures": {"investigation": {}},
            "open_causes": [{"cause": "Whether snow or warm air set it.", "proposed_run": {}},
                            {"cause": "Whether more snow would restore it.",
                             "proposed_run": {}}]}


def test_each_claim_and_each_summary_sentence_has_its_part_of_the_page():
    rep = _report3()
    parts = RV.claim_parts(rep, RECS3, PLAN3)
    assert parts == {1: RV.TESTED, 2: RV.YEAR, 3: RV.SPACE, 4: RV.MECHANISMS}
    sup = CT.summary_support(ANSWER3, CLAIMS3, RECS3, {}, PLAN3)
    got = [(r["part"], r["claims"]) for r in RV.sentence_parts(sup, parts)]
    assert got == [(RV.TESTED, []), (RV.TESTED, [1]), (RV.YEAR, []), (RV.SPACE, []),
                   (RV.MECHANISMS, [4]), (RV.MECHANISMS, [])]


def test_the_three_part_page_gives_interpretation_then_the_verified_claims_as_evidence():
    rep = _report3()
    sup = CT.summary_support(ANSWER3, CLAIMS3, RECS3, {}, PLAN3)
    main = RV.render_main(rep, sup, RECS3, plan=PLAN3, note="Re-rendered, no model call.")
    heads = ["## The answer"] + ["## " + RV.PART_TITLES[k] for k in (RV.YEAR, RV.SPACE,
                                                                    RV.MECHANISMS)] + [
        "## Remaining uncertainty and proposed sensitivity experiments", "## Model and data",
        "## Verification"]
    order = [main.index(h) for h in heads]
    assert order == sorted(order) and "## Key findings" not in main
    assert "*Re-rendered, no model call.*" in main and RV.HOW_TO_READ in main

    def part(k):
        a = main.index("## " + RV.PART_TITLES[k])
        return main[a:min(x for x in order + [len(main)] if x > a)]
    lead = main.split("## The answer")[1].split("## " + RV.PART_TITLES[RV.YEAR])[0]
    assert "Yes, in the model. Summer GPP was 10.9 percent below normal (claim 1)." in lead
    assert "largest in July" not in lead and "1.73" not in lead
    year = part(RV.YEAR)
    # the label's own words, in every part: interpretation, and no cause
    for k in (RV.YEAR, RV.SPACE, RV.MECHANISMS):
        assert "**Interpretation** (the Analyzer's own summary sentences; no cause is " \
               "stated):" in part(k), k
    assert "The shortfall was largest in July." in year
    assert year.index("largest in July") < year.index("**Verified evidence**") \
        < year.index("(claim 2; observation; records L6, L7)")
    space = part(RV.SPACE)
    assert "The shortfall sat in every elevation band." in space
    assert "*(claim 3; observation; record L9)*" in space
    mech = part(RV.MECHANISMS)
    assert "Summer air was also 1.73 K warmer (claim 4). Either may have contributed." in mech
    assert RV.NO_RULING in mech
    assert ("- \"Warmer or drier air lowers photosynthesis directly.\" Its fields: TSA (claim "
            "4), RH2M (no kept claim; the audit struck one, see the appendix).") in mech
    assert "Its fields: TLAI (no kept claim)." in mech
    # every kept claim is on the page once, as evidence, and none says a cause
    for c in CLAIMS3:
        assert main.count(c["claim"]) == 1, c["claim"]
    assert "caused" not in main.replace("what caused what", "").replace(
        "whether one of these results caused another", "")
    # two open causes behind a withheld claim: the general line once
    assert main.count("**Open:** whether one of these results caused another.") == 1
    assert "statements of its 2 open causes are in the audit appendix" in main


def test_a_part_the_summary_and_the_claims_are_silent_on_says_so():
    rep = dict(_report3(), claims=[CLAIMS3[0], CLAIMS3[2]],
               answer="Summer GPP was 10.9 percent below normal.")
    sup = CT.summary_support(rep["answer"], rep["claims"], RECS3, {}, PLAN3)
    main = RV.render_main(rep, sup, RECS3, plan=PLAN3)
    year = main.split("## " + RV.PART_TITLES[RV.YEAR])[1].split("## " + RV.PART_TITLES[RV.SPACE])[0]
    silent = "*The Analyzer's saved summary holds no sentence on this.*"
    assert silent in year and "**Verified evidence:** none." in year
    space = main.split("## " + RV.PART_TITLES[RV.SPACE])[1].split(
        "## " + RV.PART_TITLES[RV.MECHANISMS])[0]
    assert silent in space and "(claim 2; observation; record L9)" in space


def test_a_plan_that_carried_out_no_part_of_the_recipe_keeps_the_key_findings_page():
    rep = _report3()
    sup = CT.summary_support(ANSWER3, CLAIMS3, RECS3, {}, PLAN3)
    for plan in (None, {}, {"analyses": [PLAN3["analyses"][0]], "expectations": []}):
        main = RV.render_main(rep, sup, RECS3, plan=plan)
        assert "## Key findings" in main and RV.PART_TITLES[RV.YEAR] not in main


def test_a_percent_ties_its_sentence_to_the_claim_whose_percent_it_is():
    recs = {"A": _rec("A", {"value": 3.27, "anomaly": -70.4, "percent": -95.6}),
            "B": _rec("B", {"value": 512.0, "percent": 3.268})}
    claims = [{"claim": "a", "evidence": ["A"], "values": [3.27, -95.6]},
              {"claim": "b", "evidence": ["B"], "values": [512.0, 3.268]}]
    rows = CT.summary_support("Over the year it was 3.3 percent above normal. It averaged "
                              "3.27 mm.", claims, recs, {}, {})
    assert [r["claims"] for r in rows] == [[2], [1]]


# ── the editorial synthesis (2026-11-09) ────────────────────────────────────
# E   A page's synthesis may be an editorial revision, written after the run
#     from the claims the audit kept. Every sentence of it names kept claims
#     and is read against their records as a claim is, so it is no way back
#     in for a claim the audit withheld; the page says whose words it is and
#     keeps the Analyzer's claims, unchanged, as the evidence.

def _whole(recs):
    return {i: dict({"warnings": [], "row_warnings": {}, "plan_items": []}, **r)
            for i, r in recs.items()}


RECSE = _whole(dict(RECS3, L12=_rec(
    "L12", {"anomaly": 1.7263}, field="TBOT", percent_allowed=False,
    period="2015-JJA", plan_items=["P3"], years=("2015", "2016", "2020"))))
for _r in RECSE.values():                    # each names its own period, as a real record does
    _r["names"] = {"years": ["2015", "2016", "2020"],
                   "periods": [x for x in [CT.period_of(_r.get("period"))] if x]}
PLANE = dict(PLAN3, question_type="hypothesis",
             competing=[dict(PLAN3["competing"][0], fields=["TBOT", "RH2M"]),
                        PLAN3["competing"][1]])
HELD = ("Summer 2015 relative humidity (RH2M) averaged 43.9 percent, below the driest normal "
        "year.")


def _report_e():
    return dict(_report3(), withheld=[{"claim": HELD, "struck_by": "percent",
                                       "struck_because": "x"}])


def _editorial(**change):
    ed = {"by": "an editor", "date": "2026-11-09",
          "answer": [{"text": "Summer GPP was 10.9 percent below normal.", "claims": [1]}],
          "caveats": [{"text": "These are model results.", "claims": []}],
          "sections": [
              {"title": "Where and when",
               "paragraphs": [[{"text": "The shortfall was 6.41 gC m-2 in June and 19.3 gC "
                                        "m-2 in July.", "claims": [2]},
                               {"text": "Below 1000 m it averaged 24.6 gC m-2.",
                                "claims": [3]}]],
               "evidence": [1, 2, 3], "figures": []},
              {"title": "Which explanations the evidence addresses",
               "paragraphs": [[{"text": "The summer air was 1.73 K warmer than normal.",
                                "claims": [4]}]],
               "explanations": [
                   {"status": "not distinguished",
                    "reading": [{"text": "A direct effect of the warmer air fits the "
                                         "evidence.", "claims": [4]}]},
                   {"status": "not assessed", "reading": []}],
               "evidence": [4], "figures": []}],
          "unresolved": [{"text": "The run cannot separate the two.", "claims": [1, 4]}]}
    ed.update(change)
    return ed


def _why(text, claims, free=False, report=None):
    return RV.sentence_problems({"text": text, "claims": claims, "free": free},
                                report or _report_e(), RECSE, PLANE, {})


def test_an_editorial_sentence_rests_on_kept_claims_and_their_records():
    assert _why("Summer GPP was 10.9 percent below normal.", [1]) == []
    assert _why("Summer growth was below normal.", []) == [
        "it names no claim: every sentence of the synthesis rests on kept claims"]
    assert "it names claim 5, which is no kept claim" in _why("Summer GPP was low.", [5])
    # a number its own claims' records do not give, though another claim's does
    assert any("the number 1.73 k matches no value of its records (L5)" in p
               for p in _why("Summer GPP fell and the air was 1.73 K warmer.", [1]))
    assert any("names DJF" in p for p in _why("Winter GPP was 10.9 percent below normal.", [1]))
    assert any("the causal word \"drove\"" in p
               for p in _why("The warm air drove the summer GPP shortfall.", [1, 4]))
    # a claim referred to in words is one it names, and is no number of its own
    assert _why("Like claim 2, it reads June and July.", [2]) == []
    assert _why("Like claim 3, it reads June and July.", [2]) == [
        "it refers to claim 3 in its words and does not name it"]


def test_the_synthesis_is_no_way_back_in_for_a_withheld_claim():
    assert sorted(RV.withheld_words(_report_e()).values()) == [
        "RH2M", "driest", "humidity", "relative"]
    # the Analyzer's own sentence, alone and hung on the temperature claim
    for claims in ([], [4]):
        assert any("it uses \"drier\": a word of a claim the audit withheld (\"driest\")" in p
                   for p in _why("The summer air was also drier than in any normal year.",
                                 claims))
    assert any("it names RH2M, which none of its claims' records reads" in p
               for p in _why("Relative humidity was low and the air 1.73 K warmer.", [4]))
    assert any("the number 43.9 percent matches no value" in p
               for p in _why("The air held 43.9 percent of its water.", [4]))
    # with nothing withheld the same word is no one's, and a kept claim's word is free
    assert _why("The summer air was 1.73 K warmer, a dry heat.", [4], report=_report3()) \
        == _why("The summer air was 1.73 K warmer.", [4]) == []


def test_a_sentence_names_only_the_fields_its_claims_records_read():
    assert _why("Summer rain was low.", [1]) == [
        "it names RAIN, which none of its claims' records reads"]
    recs = dict(RECSE, L20=dict(RECSE["L5"], id="L20", field="QRUNOFF"))
    rep = dict(_report_e(), claims=CLAIMS3 + [dict(CLAIMS3[0], evidence=["L20"])])
    # a plain word that is part of the name of the field its records read
    assert RV.sentence_problems({"text": "Summer runoff was 10.9 percent below normal.",
                                 "claims": [5]}, rep, recs, PLANE, {}) == []
    assert RV.sentence_problems({"text": "Summer runoff was 10.9 percent below normal.",
                                 "claims": [1]}, rep, recs, PLANE, {}) != []


def test_a_sentence_with_no_claim_states_a_limitation_and_nothing_else():
    assert _why("These are model results.", [], free=True) == []
    assert _why("The air was 1.73 K warmer.", [], free=True) == [
        "it prints a number and names no claim"]
    assert any("the causal word" in p for p in _why("Warm air caused it.", [], free=True))
    # only a caveat or a sentence on what remains uncertain may be one
    ed = _editorial()
    ed["sections"][0]["paragraphs"][0].append({"text": "Growth fell everywhere.", "claims": []})
    assert any("\"Where and when\": \"Growth fell everywhere.\": it names no claim" in p
               for p in RV.editorial_problems(ed, _report_e(), RECSE, PLANE))


def test_the_caveats_cover_every_warning_the_answers_claims_carry():
    rep = _report_e()
    rep["claims"] = [dict(CLAIMS3[0], warnings=["restricted_comparability"])] + CLAIMS3[1:]
    assert RV.editorial_problems(_editorial(), rep, RECSE, PLANE) == [
        "the answer names claim 1, which carries the warning \"restricted comparability\" "
        "(restricted_comparability), and no caveat covers it"]
    ok = _editorial(caveats=[{"text": "Its comparison is restricted.", "claims": [1],
                              "covers": ["restricted_comparability"]}])
    assert RV.editorial_problems(ok, rep, RECSE, PLANE) == []
    bad = _editorial(caveats=[{"text": "Its comparison is restricted.", "claims": [1],
                               "covers": ["restricted_comparability", "made_up"]}])
    assert any("covers \"made_up\"" in p for p in RV.editorial_problems(bad, rep, RECSE, PLANE))


def test_each_competing_explanation_has_a_reading_or_is_not_assessed():
    assert RV.editorial_problems(_editorial(), _report_e(), RECSE, PLANE) == []

    def with_explanations(exs):
        ed = _editorial()
        ed["sections"][1]["explanations"] = exs
        return RV.editorial_problems(ed, _report_e(), RECSE, PLANE)
    good = _editorial()["sections"][1]["explanations"]
    assert any("reads 1 explanations and the plan named 2" in p
               for p in with_explanations(good[:1]))
    # no kept claim reads leaf area: it cannot be given a verdict
    assert any("explanation 2: no kept claim reads its fields" in p
               for p in with_explanations([good[0], {"status": "not supported",
                                                     "reading": good[0]["reading"]}]))
    assert any("explanation 1: a status other than \"not assessed\" needs a reading" in p
               for p in with_explanations([{"status": "consistent", "reading": []}, good[1]]))
    assert any("explanation 1: its status is one of" in p
               for p in with_explanations([{"status": "proven", "reading": []}, good[1]]))


def test_every_kept_claim_and_listed_figure_is_shown_once():
    ed = _editorial()
    ed["sections"][1]["evidence"] = []
    assert RV.editorial_problems(ed, _report_e(), RECSE, PLANE) == [
        "claim 4 is listed as evidence 0 times: once each"]
    rep = dict(_report_e(), figures={"investigation": {"f1": "/x/f1.png"}})
    assert RV.editorial_problems(_editorial(), rep, RECSE, PLANE) == [
        "the figure f1 is shown 0 times: once each"]


def test_the_editorial_page_says_whose_words_it_is_and_keeps_the_claims_as_evidence():
    rep, ed = _report_e(), _editorial()
    main = RV.render_editorial(rep, RECSE, PLANE, {}, ed)
    assert "Editorial revision (an editor, 2026-11-09)." in main
    assert "It is not the Analyzer's output" in main
    heads = ["## The answer", "## Where and when",
             "## Which explanations the evidence addresses",
             "## Remaining uncertainty and proposed sensitivity experiments",
             "## Model and data", "## Verification"]
    order = [main.index(h) for h in heads]
    assert order == sorted(order)
    # the label under the answer, each section and the remaining uncertainty
    assert main.count("*Editorial synthesis.*") == 4
    assert "Summer GPP was 10.9 percent below normal (claim 1)." in main
    # the Analyzer's own summary is not on the page; its claims are, once each
    assert "Either may have contributed" not in main
    for c in CLAIMS3:
        assert main.count(c["claim"]) == 1, c["claim"]
    assert main.count("**Verified evidence** (the Analyzer's claims, as the audit kept "
                      "them)") == 2
    # what the audit withheld is said by code, and the synthesis is silent on it
    assert ("- The audit withheld 1 claim, on relative humidity (RH2M). This report makes no "
            "statement on it;") in main
    assert HELD not in main and "43.9" not in main
    mech = main.split("## Which explanations")[1].split("## Remaining")[0]
    assert "The plan's words: \"Warmer or drier air lowers photosynthesis directly.\"" in mech
    assert "**Not distinguished by the evidence.** A direct effect of the warmer air fits " \
           "the evidence (claim 4)." in mech
    assert "No accepted claim reads RH2M (relative humidity at 2 m); the audit withheld the " \
           "claim on it." in mech
    assert "**Not assessed.**" in mech and "No accepted claim reads TLAI (" in mech
    assert f"- {RV.QUESTION_TYPE_LINES['hypothesis']}" in main and f"- {RV.NO_CAUSE}" in main
    assert "4 of its 6 sentences name kept claims" not in main      # counts every sentence
    assert "6 of its 7 sentences name kept claims" in main
    # a synthesis with a problem is not rendered at all
    bad = _editorial(answer=[{"text": "The summer air was also drier.", "claims": [4]}])
    with pytest.raises(ValueError, match="the editorial synthesis is not rendered"):
        RV.render_editorial(rep, RECSE, PLANE, {}, bad)
    # the appendix holds the synthesis sentence by sentence, and the Analyzer's summary
    audit = RV.render_audit(rep, CT.summary_support(ANSWER3, CLAIMS3, RECSE, {}, PLANE),
                            RECSE, editorial=ed)
    assert audit.index("## The editorial synthesis, sentence by sentence") \
        < audit.index("## The answer's limitations, in the registry's words") \
        < audit.index("## The summary, sentence by sentence")
    assert ("- the answer: \"Summer GPP was 10.9 percent below normal.\" Rests on claim 1; "
            "checked against record L5.") in audit
    assert "The Analyzer's own summary, which the editorial synthesis replaces" in audit
    assert "kept: \"Either may have contributed.\"" in audit


def test_one_reference_covers_sentences_in_a_row_on_the_same_claims():
    para = [{"text": "A.", "claims": [1, 2]}, {"text": "B.", "claims": [2, 1]},
            {"text": "C.", "claims": [3]}, {"text": "D.", "claims": []}]
    assert RV._paragraph(para) == "A. B (claims 1 and 2). C (claim 3). D."


# 2e. A LATER REVISION of an editorial page (2026-11-09, after an outside
#     review of the first). A kept claim a later reading found wrong carries a
#     CORRECTION: the claim stays as the audit kept it, marked, with the
#     corrected wording and its reason under it; the corrected wording is
#     traced as a claim is, and a record it adds must have been made again by
#     a numbers check. The page says of each figure what it draws beyond the
#     claims' records and which check made that again, and its verification
#     statement claims no more than that.

RECSC = dict(RECSE, **_whole({
    "L20": _rec("L20", {"anomaly": -8.6}, period="2015-08"),
    "L21": _rec("L21", {}, call="get_series"),
    "L23": _rec("L23", {"anomaly": -4.2}, field="RH2M", period="2015-JJA",
                percent_allowed=False)}))
for _i in ("L20", "L21", "L23"):
    RECSC[_i]["names"] = {"years": ["2015", "2016", "2020"],
                          "periods": [x for x in [CT.period_of(RECSC[_i].get("period"))] if x]}
RUN_CHECK = {"n_compared": 125, "pass": True,
             "records": {i: {"call": "anomaly", "compared": 25, "mismatched": 0}
                         for i in ("L5", "L6", "L7", "L9", "L12")}}
REVIEW = {"checked_at": "2026-11-09T20:00:00", "n_compared": 50, "n_mismatched": 0,
          "mismatches": [], "note": "The same check, on lookups no claim cites.",
          "asked": {"L20": ["cited by the correction of claim 2"], "L21": ["drawn by f1"],
                    "L23": ["drawn by f1"]},
          "records": {"L20": {"call": "anomaly", "compared": 25, "mismatched": 0},
                      "L21": {"call": "get_series", "compared": 0,
                              "note": "it logs no number"},
                      "L23": {"call": "anomaly", "compared": 25, "mismatched": 0}}}
FIXED = ("Month by month, GPP was 6.41 gC/m2 below normal in June, 19.3 in July and 8.6 in "
         "August.")


def _report_c():
    return dict(_report_e(), evidence_numbers=RUN_CHECK,
                numbers_check={"n_compared": 2042, "pass": True})


def _corrected(**change):
    a = {"claim": 2, "date": "2026-11-09", "text": FIXED, "records": ["L6", "L7", "L20"],
         "reason": [{"text": "The claim left out August, when the shortfall was 8.6 gC m-2."}]}
    a.update(change)
    return _editorial(revision=2, amendments=[a])


def _correction_problems(review=REVIEW, **change):
    return RV.editorial_problems(_corrected(**change), _report_c(), RECSC, PLANE, review=review)


def test_a_correction_is_traced_as_a_claim_and_its_added_record_was_made_again():
    assert _correction_problems() == []
    # the record it adds is in no check, or failed the one it is in
    none = "the correction of claim 2 cites L20, which no numbers check has made again"
    assert none in _correction_problems(review=None)
    failed = dict(REVIEW, records=dict(REVIEW["records"], L20={"call": "anomaly",
                                                               "compared": 25, "mismatched": 1}))
    assert none in _correction_problems(review=failed)
    # its wording is held to its records as a claim's is
    assert any("the correction of claim 2: the number 99.9" in p for p in _correction_problems(
        text=FIXED.replace("8.6", "99.9")))
    assert any("the correction of claim 2 uses \"drier\"" in p for p in _correction_problems(
        text=FIXED + " The air was drier."))
    assert "the correction of claim 2 changes no word of it" in _correction_problems(
        text=CLAIMS3[1]["claim"])
    assert "the correction of claim 2 gives no reason" in _correction_problems(reason=[])
    assert "the correction of claim 2 cites L99, which is no record of the study" in \
        _correction_problems(records=["L6", "L7", "L99"])
    assert "a correction names claim 9, which is no kept claim" in _correction_problems(claim=9)
    # a sentence that names the claim is read on the record the correction adds, and only then
    s = {"text": "August was 8.6 gC m-2 below normal.", "claims": [2]}
    assert RV.sentence_problems(s, _report_c(), RECSC, PLANE, {}, {2: ["L20"]}) == []
    assert RV.sentence_problems(s, _report_c(), RECSC, PLANE, {}) != []
    # the editor's account of a revision is no way back in for a withheld claim either
    hist = _editorial(revision=2, history=[{"revision": 2, "changes": ["The air was drier."]}])
    assert any("the revision history" in p and "\"drier\"" in p
               for p in RV.editorial_problems(hist, _report_c(), RECSC, PLANE))


def test_the_page_keeps_the_wording_the_audit_kept_beside_its_correction():
    rep, ed = _report_c(), _corrected()
    main = RV.render_editorial(rep, RECSC, PLANE, {}, ed, review=REVIEW)
    kept = CLAIMS3[1]["claim"]
    assert f"- {kept} *(claim 2, as the audit kept it; corrected below; observation; " in main
    assert (f"  - **Claim 2, corrected in review** (2026-11-09; an editorial correction, not "
            f"the Analyzer's wording): {FIXED} *(records L6, L7, L20)*") in main
    assert "  - **Why it was corrected:** The claim left out August" in main
    assert "- Claim 2 is shown with a correction made in review:" in main
    assert "claim 2 carries a correction made in review, shown under the original wording" in main
    assert ("Claim 2 carries a correction made in review. The corrected wording was traced to "
            "its records as a claim is, and record L20, which it adds, was made again by the "
            "review check. The run's own record is unchanged.") in main
    assert rep["claims"][1]["claim"] == kept                 # the record itself is not touched
    audit = RV.render_audit(rep, CT.summary_support(ANSWER3, CLAIMS3, RECSC, {}, PLANE),
                            RECSC, editorial=ed, review=REVIEW)
    assert audit.index("## Corrections to claims the audit kept") \
        < audit.index("## The review check") \
        < audit.index("## The editorial synthesis, sentence by sentence")
    assert f"- **Claim 2, as the audit kept it:** \"{kept}\" (records L6, L7)" in audit
    assert "Record L20 was made again by the review check (25 numbers, no mismatch)." in audit
    assert ("Records L6 and L7 were made again by the numbers check of the run (50 numbers, no "
            "mismatch).") in audit
    assert "- L21 (get_series, GPP, watershed, vegetated); drawn by f1: not made again" in audit
    # a correction that fails is not shown: no page is made
    with pytest.raises(ValueError, match="no numbers check has made again"):
        RV.render_editorial(rep, RECSC, PLANE, {}, ed)


def test_a_figure_says_what_it_draws_beyond_the_claims_and_which_check_made_it_again():
    rep = _report_c()
    checks = RV.record_checks(rep, REVIEW)
    assert checks["L5"]["by"] == "run" and checks["L20"]["by"] == "review"
    assert checks["L20"]["ok"] and not checks["L21"]["ok"] and "L22" not in checks
    assert RV.figure_check_line(rep, RECSC, ["L5", "L6"], checks) is None
    assert RV.figure_check_line(rep, RECSC, ["L5", "L20"], checks) == (
        "Beyond the claims' records, this figure draws record L20, which no accepted claim "
        "cites. The review check made its 25 numbers again from the model output, with no "
        "mismatch.")
    line = RV.figure_check_line(rep, RECSC, ["L5", "L20", "L21", "L22"], checks)
    assert "draws records L20 to L22, which no accepted claim cites." in line   # a run of ids
    assert "The review check made the 25 numbers of record L20 again" in line
    assert ("Record L21 holds a series of which the log keeps no number, so no check made it "
            "again.") in line
    assert "Record L22 was made again by no check." in line
    # the record of a withheld claim: what the rule struck, and that no sentence reads it
    assert RV.figure_check_line(rep, RECSC, ["L23"], checks).endswith(
        "Record L23 is relative humidity at 2 m (RH2M), the record of the claim the audit "
        "withheld: the percent rule struck that claim for how it words a percent, not for this "
        "record's values. No sentence on this page reads it.")
    # without a review check nothing beyond the claims was made again, and the line says so
    assert RV.figure_check_line(rep, RECSC, ["L20"], RV.record_checks(rep)).endswith(
        "It was made again by no check.")


def test_a_later_revision_says_what_each_check_covered_and_names_the_question_type():
    rep = dict(_report_c(), figures={"investigation": {"f1": "/x/f1.png"}})
    ed = _editorial(revision=2)
    ed["sections"][0]["figures"] = ["f1"]
    figs = {"f1": ["L5", "L20", "L21"]}
    main = RV.render_editorial(rep, RECSC, PLANE, {}, ed, fig_records=figs, review=REVIEW)
    assert ("*Question with expectations stated before the results were seen. Status "
            "**supported** (the analysis ran and its claims passed the audit; it is not a "
            "verdict on any explanation). Generated") in main
    assert "Editorial revision 2 (an editor, 2026-11-09)." in main
    assert RV.STATUS_NOTE in main
    ver = main.split("## Verification")[1]
    assert ver.index("**Checked by code**") < ver.index("**Not checked by code**")
    assert ("- **The numbers in the accepted claims.** Before this report was written, an "
            "independent reading made them again from the model output (the 2,042 numbers of "
            "the regional table and the 125 numbers of the 5 lookups the claims cite), with no "
            "mismatch.") in ver
    assert ("- **The other records the figures draw.** The figures draw 2 lookups that no "
            "accepted claim cites, which the reading above did not cover. In the review of "
            "2026-11-09, the same reading made 1 of them again (25 numbers, no mismatch). A "
            "note under each figure says which.") in ver
    # what code did not check is said as plainly as what it did
    gaps = ver.split("**Not checked by code**")[1]
    assert ("- **Plotted series.** 1 of the lookups the figures draw holds a series of which "
            "the log keeps no number, so no check made it again (Figure 1).") in gaps
    assert "- **Ranking words.**" not in gaps                # no kept claim uses one here
    assert ("- **Interpretation.** Code checks a sentence's numbers, periods, regions, fields "
            "and wording. It cannot tell whether the sentence follows from the claims it "
            "names; 3 of the synthesis's 7 sentences print no number at all. That reading is "
            "a person's, and no code stands in for it.") in gaps
    assert "Every number above" not in main
    assert "*Beyond the claims' records, this figure draws records L20 and L21" in main
    # the first revision's page is as it was: no type on the status line, the old statement
    first = dict(ed)
    del first["revision"]
    old = RV.render_editorial(rep, RECSC, PLANE, {}, first, fig_records=figs)
    assert "*Status **supported**. Generated" in old and "**Checked by code**" not in old
    # its one statement claims no more than the run's check covered
    assert ("Every number in the accepted claims was made again" in old
            and "a figure may also draw other records, a series among them, which it does not "
                "cover, and a ranking word is not held to records a claim leaves out." in old)
    assert RV.STATUS_NOTE not in old and "Beyond the claims' records" not in old
    explore = RV.render_editorial(rep, RECSC, dict(PLANE, question_type="exploratory"), {}, ed,
                                  fig_records=figs, review=REVIEW)
    assert "*Exploratory question: no expectation was stated before the results were seen. " \
           "Status **supported**" in explore


def test_a_ranking_word_is_labelled_as_unchecked_with_what_a_person_read():
    rep = _report_c()
    rep["claims"] = CLAIMS3[:2] + [dict(CLAIMS3[2], claim="Below 1000 m the summer GPP shortfall "
                                        "was smallest, 24.6 gC/m2.")] + CLAIMS3[3:]
    assert RV.ranking_claims(rep) == [3] and RV.ranking_claims(_report_c()) == []
    ed = _editorial(revision=2, read_by_hand=[
        {"text": "In review claim 3 was read by hand against every elevation band.",
         "claims": [3]}])
    ver = RV.render_editorial(rep, RECSC, PLANE, {}, ed, review=REVIEW).split("## Verification")[1]
    assert ("- **Ranking words.** The claim trace holds a claim's numbers to the records it "
            "cites. It does not hold a ranking, or a statement about every member of a set "
            "(\"most\", \"least\", \"largest\", \"every band\"), to records the claim leaves "
            "out. Claim 3 uses such words. *Read by hand in review:* In review claim 3 was read "
            "by hand against every elevation band.") in ver
    # what was read by hand is held to the claims it names like any other sentence
    bad = _editorial(revision=2, read_by_hand=[{"text": "Claim 3 was read by hand.",
                                                "claims": []}])
    assert any("the verification, read by hand" in p
               for p in RV.editorial_problems(bad, rep, RECSC, PLANE, review=REVIEW))
