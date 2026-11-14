"""The second reading covers what the report prints (the report rebuild,
package A8), on the small fake study of tests/test_suite_numbers_check.py,
with no model call.

    pytest tests/test_report_rebuild_printed.py -v

A record the report prints (`also`) is made again in full by the same code as
a cited one: a mismatch planted in it withholds the report, a screen's rows
that no claim states are made again, a record named both ways is made again
once, and a printed name that is no answered lookup is a mismatch. The
sentence for the report's "Checks and records" section is plain, in each
state the check can end in.
"""
import json

import pytest

from test_suite_numbers_check import _planted, plant, study  # noqa: F401

from agents.analysis import numbers_check as NC          # noqa: E402


def _check(study, rd=None, claims=None, also=()):
    return NC.check_evidence(rd or study["rd"], study["claims"] if claims is None else claims,
                             registry=study["entry"], write=False, also=also)


def test_a_printed_record_with_a_planted_mismatch_withholds_the_report(study, tmp_path):
    rd = _planted(study, tmp_path, 3, "difference", study["res"][3]["difference"] * 1.0001)
    first = study["claims"][:1]                                    # cites L1 alone
    assert _check(study, rd, first)["pass"]                        # not printed: not read
    rec = _check(study, rd, first, also=["L3"])
    assert not rec["pass"] and rec["printed"] == ["L3"]
    assert [(m["record"], m["path"]) for m in rec["mismatches"]] == [("L3", "difference")]
    assert "L3" in NC.evidence_reason(rec)


def test_a_printed_screen_is_made_again_in_full(study, tmp_path):
    res = study["res"][10]
    stated = {m.split(":", 1)[1] for c in study["claims"]
              for m in (c.get("evidence_trace") or {}).get("matched") or []
              if m.startswith("L10:")}
    row = next(k[:-len("change")] for k, v in res.items()
               if k.startswith("rows.") and k.endswith(".change") and isinstance(v, float)
               and v and not any(s.startswith(k[:-len("change")]) for s in stated))
    rd = _planted(study, tmp_path, 10, row + "change", plant(res[row + "change"], "*", 1.001))
    assert _check(study, rd)["pass"]                       # a row no claim states: not read
    for claims in (study["claims"], study["claims"][:1]):  # its other rows cited, or none
        rec = _check(study, rd, claims, also=["L10"])
        assert not rec["pass"]
        assert [(m["record"], m["path"]) for m in rec["mismatches"]] == [("L10", row + "change")]
    full = _check(study, also=["L10"])
    assert full["pass"] and full["records"]["L10"]["compared"] > \
        _check(study)["records"]["L10"]["compared"]


def test_a_record_named_both_ways_is_made_again_once(study, tmp_path):
    base = _check(study)
    both = _check(study, also=["L3", "L1"])
    assert both["pass"] and both["n_compared"] == base["n_compared"]
    assert both["printed"] == ["L1", "L3"] and list(both["records"]).count("L3") == 1
    rd = _planted(study, tmp_path, 3, "difference", study["res"][3]["difference"] * 1.0001)
    rec = _check(study, rd, also=["L3"])
    assert len(rec["mismatches"]) == 1


def test_a_printed_name_that_is_no_answered_lookup_withholds(study):
    rec = _check(study, also=["L99"])
    assert not rec["pass"] and rec["printed"] == ["L99"]
    assert "no answered lookup" in rec["mismatches"][0]["why"]


def test_the_written_record_names_what_the_report_prints(study):
    NC.check_evidence(study["rd"], study["claims"][:1], registry=study["entry"],
                      also=["L3", "L2", "L3"])
    rec = json.loads((study["out"] / NC.EVIDENCE_JSON).read_text())
    assert rec["printed"] == ["L2", "L3"] and rec["cited"] == ["L1"]
    assert set(rec["records"]) == {"L1", "L2", "L3"}
    NC.check_evidence(study["rd"], study["claims"], registry=study["entry"])
    assert "printed" not in json.loads((study["out"] / NC.EVIDENCE_JSON).read_text())


# ── the sentence for "Checks and records" ───────────────────────────────────

PASSED = {"pass": True, "n_compared": 1234, "cited": [f"L{i}" for i in range(1, 13)],
          "printed": [f"L{i}" for i in range(10, 40)], "records": {},
          "largest_relative_difference": {"rel_diff": 3e-11}, "not_recomputed": []}


@pytest.mark.parametrize("rec,figures,want", [
    (PASSED, True,
     "A second reading of the model output, by code written apart from the analysis, made "
     "1,234 numbers again from the 12 records the findings cite and the 27 more the report "
     "prints, with a largest relative difference of 3e-11. The values the figures draw were "
     "not made again."),
    (dict(PASSED, printed=[], not_recomputed=[{"n": 4}, {"why": "x"}]), False,
     "A second reading of the model output, by code written apart from the analysis, made "
     "1,234 numbers again from the 12 records the findings cite, with a largest relative "
     "difference of 3e-11. It did not make again 5 other numbers of these records, which no "
     "finding states."),
    (dict(PASSED, cited=["L1"], printed=["L1"], n_compared=1), False,
     "A second reading of the model output, by code written apart from the analysis, made "
     "1 number again from the 1 record the findings cite, with a largest relative difference "
     "of 3e-11."),
    (dict(PASSED, **{"pass": False, "mismatches": [{}, {}]}), True,
     "A second reading of the model output, by code written apart from the analysis, made "
     "1,234 numbers again from the 12 records the findings cite and the 27 more the report "
     "prints, and found 2 that did not reproduce, so the report is withheld."),
    ({"pass": False, "n_compared": 0, "error": "OSError: no history"}, True,
     "A second reading of the model output, by code written apart from the analysis, stopped "
     "before it compared any number, so the numbers in this report were not made again."),
    (None, True,
     "No second reading of the model output is on record, so the numbers in this report were "
     "not made again."),
], ids=["passed", "left numbers out", "one record", "did not pass", "did not run", "none"])
def test_the_sentence_in_each_state(rec, figures, want):
    got = NC.evidence_sentence(rec, figures=figures)
    assert got == want
    assert "**" not in got and "\u2014" not in got and "L1" not in got
    assert ".json" not in got


def test_the_page_line_stays_for_its_callers(study):
    rec = _check(study, also=["L3"])
    assert NC.evidence_page_line(rec).startswith("**Numbers check: passed.**")
