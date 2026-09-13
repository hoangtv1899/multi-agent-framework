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
    "claims": [{"claim": "QOVER/P is 0.367648204485183 at col_04.",
                "finding_id": "f1", "values": [0.367648204485183]}],
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
                "finding_id": "f1", "values": [0.367648204485183, 471.04]}],
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
    spec["claims"] = [{"claim": "QDRAI reaches 84 percent at 1913 m.",
                       "finding_id": "f1", "values": [0.843]}]
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
    assert spec["claims"][0]["values"] == [0.367648204485183]   # untouched
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
