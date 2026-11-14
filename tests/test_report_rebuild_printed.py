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


def _with_lines(study, tmp_path, extra):
    """A copy of the fake study whose log ends with `extra` lines."""
    import shutil
    from agents.analysis import analysis_plan as AP
    rd = tmp_path / "more_lines"
    (rd / "04_analysis").mkdir(parents=True)
    for d in ("01_inputs", "domain_case"):
        shutil.copytree(study["rd"] / d, rd / d)
    shutil.copy(study["out"] / AP.PLAN_FILE, rd / "04_analysis" / AP.PLAN_FILE)
    lines = (study["out"] / "lookups.jsonl").read_text().splitlines()
    (rd / "04_analysis" / "lookups.jsonl").write_text(
        "\n".join(lines + [json.dumps(x) for x in extra]) + "\n")
    return rd, len(lines)


@pytest.mark.parametrize("mark", [{"refused": "the gate: not declared"},
                                  {"failed": "the store gives no value"},
                                  {"support": True},
                                  {"sha256": None}],
                         ids=["refused", "failed", "support", "no answer"])
def test_a_printed_line_that_is_no_answered_lookup_withholds(study, tmp_path, mark):
    """A refused, failed or support line, or one with no answer, named among
    the printed records is a mismatch with its reason, never skipped, though
    it still holds its numbers (each line here is an answered difference
    with the one mark added)."""
    good = json.loads((study["out"] / "lookups.jsonl").read_text().splitlines()[2])  # L3
    assert good["call"] == "difference" and "sha256" in good
    line = dict(good, **{k: v for k, v in mark.items() if v is not None})
    if mark.get("sha256", "") is None:
        line.pop("sha256")
    rd, n = _with_lines(study, tmp_path, [line])
    rid = f"L{n + 1}"
    assert _check(study, rd)["pass"]                       # not printed: not read
    rec = _check(study, rd, also=[rid])
    assert not rec["pass"] and rec["printed"] == [rid]
    assert [(m["record"], m["path"]) for m in rec["mismatches"]] == [(rid, None)]
    assert "no answered lookup" in rec["mismatches"][0]["why"]
    assert NC.evidence_sentence(rec).endswith(
        "and found 1 record it could not make again, so the report is withheld.")


def test_the_written_record_names_what_the_report_prints(study):
    NC.check_evidence(study["rd"], study["claims"][:1], registry=study["entry"],
                      also=["L3", "L2", "L3"])
    rec = json.loads((study["out"] / NC.EVIDENCE_JSON).read_text())
    assert rec["printed"] == ["L2", "L3"] and rec["cited"] == ["L1"]
    assert set(rec["records"]) == {"L1", "L2", "L3"}
    NC.check_evidence(study["rd"], study["claims"], registry=study["entry"])
    assert "printed" not in json.loads((study["out"] / NC.EVIDENCE_JSON).read_text())


# ── the sentence for "Checks and records" ───────────────────────────────────

RECORDS = {f"L{i}": {"compared": 5} for i in range(1, 40)}
PASSED = {"pass": True, "n_compared": 1234, "cited": [f"L{i}" for i in range(1, 13)],
          "printed": [f"L{i}" for i in range(10, 40)], "records": RECORDS,
          "largest_relative_difference": {"rel_diff": 3e-11}, "not_recomputed": []}
LEAD = NC.LEAD
# printed records L38 (nothing compared) and L39 (a call the check does not
# make again) had no number made again; L20 (printed) and L10 (cited and
# printed) left numbers out, and so did the cited L2
PART = dict(PASSED, records=dict({k: v for k, v in RECORDS.items() if k != "L39"},
                                 L38={"compared": 0}),
            not_recomputed=[{"record": "L39", "why": "the check makes no such call again"},
                            {"record": "L20", "n": 3}, {"record": "L10", "n": 1},
                            {"record": "L2", "n": 2}])


@pytest.mark.parametrize("rec,figures,want", [
    (PASSED, True,
     f"{LEAD} made 1,234 numbers again from the 12 records the findings cite and the 27 more "
     "the report prints, with a largest relative difference of 3e-11. The values the figures "
     "draw were not made again."),
    (dict(PASSED, printed=[], not_recomputed=[{"record": "L2", "n": 4},
                                              {"record": "L3", "why": "x"}]), False,
     f"{LEAD} made 1,234 numbers again from the 12 records the findings cite, with a largest "
     "relative difference of 3e-11. It did not make again 5 other numbers of the records the "
     "findings cite, which no finding states."),
    (dict(PASSED, cited=["L1"], printed=["L1"], n_compared=1), False,
     f"{LEAD} made 1 number again from the 1 record the findings cite, with a largest relative "
     "difference of 3e-11."),
    (PART, False,
     f"{LEAD} made 1,234 numbers again from the 12 records the findings cite and the 25 more "
     "the report prints, with a largest relative difference of 3e-11. It made no number again "
     "of 2 more records the report prints. It did not make again 2 other numbers of the "
     "records the findings cite, which no finding states. It did not make again 4 other "
     "numbers of the records the report prints."),
    (dict(PASSED, cited=[], printed=["L1", "L2"], n_compared=10), False,
     f"{LEAD} made 10 numbers again from the 2 records the report prints, with a largest "
     "relative difference of 3e-11."),
    (dict(PASSED, **{"pass": False, "mismatches": [{"path": "difference"}, {"path": "percent"},
                                                   {"path": None}]}), True,
     f"{LEAD} made 1,234 numbers again from the 12 records the findings cite and the 27 more "
     "the report prints, and found 2 numbers that did not reproduce and 1 record it could not "
     "make again, so the report is withheld."),
    ({"pass": False, "n_compared": 0, "cited": [], "printed": ["L99"], "records": {},
      "mismatches": [{"record": "L99", "path": None}]}, True,
     f"{LEAD} made no numbers again, and found 1 record it could not make again, so the report "
     "is withheld."),
    ({"pass": False, "error": "OSError: x", "n_compared": 50, "cited": ["L1"],
      "printed": ["L2"], "mismatches": []}, True,
     f"{LEAD} stopped before it finished, after it had made 50 numbers again, so the report is "
     "withheld."),
    ({"pass": False, "n_compared": 0, "error": "OSError: no history"}, True,
     f"{LEAD} stopped before it compared any number, so the report is withheld."),
    ({"pass": True, "n_compared": 0, "cited": [], "printed": ["L1"],
      "records": {"L1": {"compared": 0}}}, True,
     f"{LEAD} made no numbers again, because no record the findings cite or the report prints "
     "holds a number it makes again. The values the figures draw were not made again."),
    ({"pass": True, "n_compared": 0, "cited": [], "records": {}}, False,
     f"{LEAD} made no numbers again, because the findings cite no record and the report prints "
     "none."),
    (None, True,
     "No second reading of the model output is on record, so the numbers in this report were "
     "not made again."),
], ids=["passed", "left numbers out", "one record", "printed records not made again",
        "printed records alone", "did not pass", "a printed name not answered",
        "stopped after comparing", "stopped at once", "nothing to compare", "no record at all",
        "none"])
def test_the_sentence_in_each_state(rec, figures, want):
    got = NC.evidence_sentence(rec, figures=figures)
    assert got == want
    assert "**" not in got and "\u2014" not in got and "L1" not in got
    assert ".json" not in got
    assert "found 0" not in got and " 0 " not in got


def test_the_page_line_stays_for_its_callers(study):
    rec = _check(study, also=["L3"])
    assert NC.evidence_page_line(rec).startswith("**Numbers check: passed.**")
