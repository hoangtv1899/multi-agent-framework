"""The readability gate and the reader's rendering (2026-09-12).

The audit proves a claim's numbers; these tests prove the WORDING is checked
too, mechanically, and that what reaches a reader is shown to three
significant figures with the exact value still on record.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.analysis import step3_interpret as s3          # noqa: E402
from agents.analysis import step4_report as s4             # noqa: E402

BAD = {
    "headline": "",
    "answer": ("Locally generated surface runoff stays a small share of "
               "precipitation everywhere while sub-surface drainage is bimodal "
               "because where the diagnosed water table is shallow enough to "
               "intersect the drainage parameterisation QDRAI/P reaches 0.78 "
               "but col_11 recharges 804.5238 mm of QCHARGE with zero QDRAI "
               "and the water table sits below the soil in all columns."),
    # the values also back the figures of the rewrite the stub returns
    "claims": [{"claim": "QOVER/P is 0.367648204485183 at col_04.",
                "finding_id": "f1",
                "values": [0.367648204485183, 0.781, 0.843, 3.8]}],
}
GOOD = {
    "headline": "Drainage, not runoff, moves most water: 78 to 84 percent of "
                "precipitation drains where the water table is shallow.",
    "answer": ("Surface runoff (QOVER) stays a small share everywhere, 37 "
               "percent at the lowest column. Sub-surface drainage (QDRAI) "
               "splits the columns in two groups. Ten of 19 columns drain "
               "almost nothing because their water table sits below the "
               "3.8 m of active soil. The split itself cannot be compared "
               "with observations."),
    "claims": [{"claim": "Runoff is 37 percent of precipitation at the lowest "
                         "column, at 471 m.",
                "finding_id": "f1", "values": [0.367648204485183, 471.04]},
               {"claim": "Drainage takes 78 to 84 percent of precipitation "
                         "where the water table is shallow.",
                "finding_id": "f2", "values": [0.781, 0.843]},
               {"claim": "The water table sits below the 3.8 m of active "
                         "soil in 10 of 19 columns.",
                "finding_id": "f3", "values": [3.8, 10, 19]}],
}


def test_the_gate_names_every_failure_mode_on_the_real_shape():
    probs = s3.check_readability(BAD)
    joined = " | ".join(probs)
    assert "headline: empty" in joined
    assert "word sentence" in joined                    # too long
    assert "QDRAI appears without" in joined            # bare code
    assert "col_11 by id" in joined                     # column id
    assert "parameterisation" in joined                 # jargon
    assert "804.5238" in joined or "0.367648204485183" in joined  # long number


def test_plain_wording_with_glosses_passes():
    assert s3.check_readability(GOOD) == []


def test_a_gloss_anywhere_covers_the_code_everywhere():
    spec = dict(GOOD)
    # the one claim backs every figure the headline and answer carry
    spec["claims"] = [{"claim": "QDRAI reaches 84 percent at 1913 m.",
                       "finding_id": "f1",
                       "values": [0.843, 1913, 0.781, 0.368, 3.8]}]
    assert s3.check_readability(spec) == []             # glossed in the answer


def test_dashes_become_commas_but_ranges_keep_theirs():
    assert s3._strip_dashes("wet — dry") == "wet, dry"
    assert s3._strip_dashes("0.78–0.84") == "0.78–0.84"
    assert s3._strip_dashes("one – two") == "one, two"


class _Client:
    """A reviewer that returns the plain wording when asked to rewrite."""
    def __init__(self):
        self.calls = 0

    def ask(self, messages):
        self.calls += 1
        return json.dumps({"headline": GOOD["headline"],
                           "answer": GOOD["answer"],
                           "claims": [c["claim"] for c in GOOD["claims"]]})


def test_gate_rewrites_once_and_leaves_the_numbers_alone():
    client = _Client()
    spec, record = s3.gate_wording(BAD, client)
    assert client.calls == 1
    assert record["rewritten"] is True
    assert record["problems_before"] and record["problems_after"] == []
    assert spec["claims"][0]["values"] == BAD["claims"][0]["values"]  # untouched
    assert spec["claims"][0]["finding_id"] == "f1"
    assert spec["headline"] == GOOD["headline"]


def test_gate_asks_nothing_when_the_wording_is_already_plain():
    client = _Client()
    spec, record = s3.gate_wording(GOOD, client)
    assert client.calls == 0 and record["rewritten"] is False
    assert spec["answer"] == GOOD["answer"]


def test_values_are_shown_to_three_significant_figures():
    f = s4.format_value
    assert f(0.367648204485183) == "0.368"
    assert f(804.5238) == "805"
    assert f(1913.31) == "1913"
    assert f(12345.6) == "12,346"
    assert f(19) == "19"
    assert f(0.0044) == "0.0044"
    assert f(-85.8531) == "-85.9"
    assert f(296.3) == "296"
    assert f("n/a") == "n/a"


def test_report_md_carries_headline_claims_caveats_and_no_dashes():
    report = {
        "question": "How deep does water move?", "model": "pflotran",
        "status": "supported", "generated_at": "2026-09-12T00:00:00+00:00",
        "run_dir": "/x/pflotran_run_1", "headline": GOOD["headline"],
        "answer": GOOD["answer"], "readability": {"problems_after": []},
        "claims": [dict(GOOD["claims"][0], n=19, caveats=["c1"])],
        "caveats": [{"id": "c1", "severity": "blocking",
                     "statement": "no well records a daily series"}],
        "withheld": [{"claim": "x", "struck_by": "declared_values",
                      "struck_because": "not in the finding"}],
        "provenance": {"rounds": [1], "stopped_because": "sufficient",
                       "audit": {"n_claims": 2, "n_struck": 1}},
        "cost": {"compute": {"columns_succeeded": 17, "columns_total": 17},
                 "llm": {"analyzer": {"calls": 2}}},
    }
    md = s4.render_markdown(report)
    assert md.startswith("# How deep does water move?")
    assert "**" + GOOD["headline"] + "**" in md
    assert "values 0.368, 471" in md and "finding `f1`" in md
    assert "Caveats that bind this answer" in md and "c1" in md
    assert "withheld (1)" in md
    assert "—" not in md


def test_the_headline_and_answer_are_held_to_the_scope_of_a_claim():
    """2026-09-13: a Gunnison headline said water reached the given water
    table, while its own claim 3 said four of 17 columns did. The gate
    cannot see a qualitative overreach, so the rule is in the prompt that
    writes the wording and in the one that rewrites it."""
    for prompt in (s3.TASK, s3.REWRITE):
        assert "no more than one" in prompt and "same count of columns" in prompt
    assert "paste the headline and the answer onto a slide unread" in s3.TASK


def test_a_percent_or_decimal_figure_must_be_one_of_the_claims_values():
    """The live case of 2026-09-13: "96 to 100 percent" over values of
    0.991 to 0.999 passed the audit, which reads `values`, not the text."""
    bad = number_problems = s3.number_problems
    got = bad("in four columns it reached 96 to 100 percent of the thickness",
              [0.991, 0.996, 0.999, 0.998], "claim 3")
    assert len(got) == 1 and '"96 percent"' in got[0] and "0.991" in got[0]
    assert bad("37 percent at the lowest column, 471 m, and 15 percent at "
               "1913 m", [471, 1913, 0.368, 0.147], "claim 1") == []
    assert bad("the deepest front was 296.3 m below the surface", [296],
               "c") == []
    assert bad("it falls by 0.029 m and 0.011 m", [-0.0289, -0.0105], "c") == []
    assert bad("wetness rose at 19.75 m depth", [19.8, 20], "c") == []
    # integers with a unit are design labels, not measurements
    assert bad("in the 5 m column, 500 mm of rain", [-0.0729], "c") == []
    # but a decimal with a unit that no value backs is caught
    assert bad("the front stopped at 16.8 m", [11.8], "c")[0].startswith(
        'c: "16.8 m" is not one of this claim\'s values (11.8)')


def test_the_headline_and_answer_are_held_to_every_claims_values():
    spec = dict(GOOD, headline="Drainage reaches 84.3 percent at 1913 m",
                answer="It reached 12.5 m in one column.",
                claims=[{"claim": "x", "values": [0.843, 1913], "finding_id": "f"}])
    got = s3.check_readability(spec)
    assert any('answer: "12.5 m" is not among the values of any claim' in p
               for p in got)
    assert not any(p.startswith("headline:") and "percent" in p for p in got)


def test_validated_is_sent_back_as_compared():
    got = s3.number_problems("the split itself was never validated", [], "claim 8")
    assert len(got) == 1 and "validated" in got[0] and "compared with" in got[0]
    assert "never \"validated\"" in s3.TASK
