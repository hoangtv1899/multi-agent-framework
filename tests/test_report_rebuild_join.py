"""The report rebuild after its two fixes were joined (2026-11-14): what the
details page says the second reading covered, when the tables print records
the findings do not cite, and a finding the Summary already says, which is
not said again under a figure.

Every record here is made up: its names, numbers and sentences belong to no
study.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.analysis import numbers_check as NC                  # noqa: E402
from agents.analysis import report_pages as RP                   # noqa: E402
from agents.analysis import step4_report as S4                   # noqa: E402
from test_report_pages import make_report                        # noqa: E402


def reading(**more):
    """A second reading that passed: two records the findings cite (L1, L2),
    three the tables print (L2, L3, L4), and of L4 no number made again."""
    rec = {"pass": True, "n_compared": 40, "largest_relative_difference": {"rel_diff": 3e-12},
           "cited": ["L1", "L2"], "printed": ["L2", "L3", "L4"],
           "records": {"L1": {"compared": 10}, "L2": {"compared": 10}, "L3": {"compared": 20},
                       "L4": {"compared": 0}},
           "not_recomputed": [{"record": "L1", "n": 3}, {"record": "L3", "n": 2},
                              {"record": "L4", "n": 5}]}
    rec.update(more)
    return rec


# ── the details page's line on the second reading ───────────────────────────
def test_the_line_counts_the_records_the_tables_print_apart():
    line = NC.evidence_page_line(reading())
    assert line.startswith("**Numbers check: passed.** 40 numbers of the 2 lookups the claims "
                           "cite and the 1 more the report's tables print were made again ")
    assert "No number was made again of 1 record the tables print: L4." in line
    assert ("Not made again: 3 number(s) of the cited records that no claim states and 2 "
            "number(s) of the records the tables print (listed in ") in line


def test_without_more_records_the_line_reads_as_before():
    for rec in (reading(printed=["L2"]), reading(printed=[]),
                {k: v for k, v in reading().items() if k not in ("cited", "printed")}):
        line = NC.evidence_page_line(rec)
        assert line.startswith(f"**Numbers check: passed.** 40 numbers of the "
                               f"{len(rec['records'])} lookups the claims cite were made again "
                               f"from the model output by an independent reading")
        assert "tables print" not in line


def test_what_code_checked_names_the_tables_when_they_print_more():
    report = {"kind": "whole_domain", "withheld": [], "evidence_numbers": reading(),
              "claims": [{"claim": "Rain fell by 3 mm.", "evidence": ["L1"]}]}
    text = "\n".join(S4.verification_limits(report))
    assert "- the numbers of the records the claims cite and the tables print: 40 numbers" in text
    assert ("What a figure draws beyond the records the claims cite and the tables print, a "
            "plotted series among it, was not made again") in text
    report["evidence_numbers"] = reading(printed=["L1"])
    text = "\n".join(S4.verification_limits(report))
    assert "- the numbers of the records the claims cite: 40 numbers" in text
    assert ("What a figure draws beyond the records the claims cite, a plotted series among "
            "it, was not made again") in text


def test_the_details_page_says_which_printed_records_were_made_again():
    report = {"kind": "whole_domain", "withheld": [], "claims": [],
              "evidence_numbers": reading(printed=["L4", "L3", "L2"])}
    text = "\n".join(RP._d_checks(report))
    assert "The records the report's tables print whose numbers were made again: L2, L3." in text
    assert ("The records the report's tables print of which no number was made again: L4."
            in text)


# ── a finding the Summary already says ──────────────────────────────────────
def test_a_finding_the_summary_says_is_not_said_again_below_it(tmp_path):
    """Finding 4, on topsoil water, would sit under the second figure. The
    Summary gives its words, without the code in brackets, so it stays on
    the details page and the figure shows none."""
    rf = {"figures": [
        {"id": "over_time", "file": "a.png", "number": 1, "caption": "A.", "comparisons": [1, 2],
         "panels": [{"letter": "a", "field": "GPP"}]},
        {"id": "over_time_2", "file": "b.png", "number": 2, "caption": "B.",
         "comparisons": [1, 2], "panels": [{"letter": "a", "field": "SOILWATER_10CM"}]}],
        "tables": []}
    report, adir = make_report(tmp_path, figures=rf)
    report["_plan"] = json.loads((adir / "analysis_plan.json").read_text())
    numbers = RP.numbers_of(report["_plan"], report)
    by_fig, others = RP.choose_findings(report, rf, numbers)
    assert by_fig["over_time_2"] == [4]                       # without a Summary that says it
    report["claims"][3]["claim"] = "Topsoil water (SOILWATER_10CM) was lower in summer 2033."
    report["answer_kept"] = "In the model, growth fell, and topsoil water was lower in summer 2033."
    by_fig, others = RP.choose_findings(report, rf, numbers)
    assert by_fig == {"over_time": [1, 2], "over_time_2": []} and others == [(3, 5)]
    # a finding that says more than the Summary stays where it was
    report["answer_kept"] = "In the model, topsoil water was lower."
    by_fig, _ = RP.choose_findings(report, rf, numbers)
    assert by_fig["over_time_2"] == [4]
