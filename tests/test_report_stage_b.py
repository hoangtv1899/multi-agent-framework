"""Stage B of the report rebuild (2026-11-14): step 3 sees the report's own
figures and tables before it writes, asks for one short finding for each
approved comparison beside them, and holds the Summary to five sentences; the
page leaves a finding out of the Results only when the tables print every
number it states.

Every record here is made up: its names, numbers and sentences belong to no
study.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.analysis import claim_trace as CT                     # noqa: E402
from agents.analysis import report_pages as RP                    # noqa: E402
from agents.analysis import step3_interpret as S3                 # noqa: E402

RF = {"figures": [
    {"id": "over_time", "number": 1, "file": "figure_over_time.png", "comparisons": [1, 3],
     "caption": "The new run minus the control run, for each month of 2031 and 2032."},
    {"id": "over_space", "number": 2, "file": "figure_over_space.png", "comparisons": [],
     "caption": "Each cell's growth."}],
    "tables": [{"number": 1, "follows": "over_time", "corner": "Quantity",
                "caption": "The new run minus the control run, over each period.",
                "columns": ["Summer 2031", "Year 2031"],
                "rows": [{"label": "Plant growth (GPP), gC/m2", "cells": ["-4.25 (-1.5 %)", "-6.1"],
                          "records": ["L2", "L3"]},
                         {"label": "Rain (RAIN), mm", "cells": ["", "-12.5"],
                          "records": ["L7"]}]}],
    "printed_records": ["L2", "L3", "L7"],
    "not_drawn": ["Air temperature (TBOT) has no panel, because the new run does not change it."],
    "problems": []}
PLAN = {"analyses": [{"id": "P1"}, {"id": "P2"}, {"id": "P3"}]}


# ── what step 3 is asked ────────────────────────────────────────────────────
def test_the_suite_task_asks_for_short_findings_beside_the_report():
    task = S3.suite_task()
    assert task.startswith(S3.TASK_BODY + S3.SUITE_RULES + S3.SUITE_FINDINGS)
    flat = " ".join(task.split())
    for words in ("Write one finding for each approved comparison that has results",
                  "Cite the comparison's own finding (required_p1 for P1, and so on)",
                  "A finding is at most 30 words and states at most two numbers",
                  "Your first-ranked finding answers the question with its key number",
                  "It does not repeat a number a table prints",
                  "write that a result cannot be stated only when no record holds it",
                  "at most five sentences, the first answering the question with its key "
                  "number"):
        assert words in flat, words
    assert S3.REPORT_VIEW_HEAD in flat
    assert "at most five plain sentences" in " ".join(" ".join(CT.claims_reply_lines()).split())


def test_the_brief_shows_the_reports_figures_and_tables():
    text = "\n".join(S3.report_view_lines(RF, PLAN))
    assert text.startswith(f"{S3.REPORT_VIEW_HEAD}. Code drew them")
    assert ("  Figure 1 (figure_over_time), for comparisons P1, P3: The new run minus the "
            "control run, for each month of 2031 and 2032.") in text
    assert "  Figure 2 (figure_over_space): Each cell's growth." in text
    assert "    | Quantity | Summer 2031 | Year 2031 |" in text
    assert "    | Plant growth (GPP), gC/m2 | -4.25 (-1.5 %) | -6.1 |" in text
    assert "    | Rain (RAIN), mm |  | -12.5 |" in text
    assert "  Not drawn: Air temperature (TBOT) has no panel" in text


def test_a_brief_with_nothing_drawn_says_so():
    text = "\n".join(S3.report_view_lines({"figures": [], "tables": [], "problems": ["no store"]},
                                          PLAN))
    assert text.endswith("None could be drawn (no store), so the report shows your findings in "
                         "their place.")
    assert "None could be drawn, so" in "\n".join(S3.report_view_lines(None, None))


def test_the_summary_holds_at_most_five_sentences():
    five = "One is here. Two is here. Three is here. Four is here. Five is here."
    six = five + " Six is here."
    held = [p for p in S3.check_readability({"answer": five, "claims": []}, headline=False)
            if "Summary" in p]
    assert held == []
    held = [p for p in S3.check_readability({"answer": six, "claims": []}, headline=False)
            if "Summary" in p]
    assert held == ["answer: 6 sentences; the report's Summary holds at most 5. Join sentences "
                    "that say one thing, keeping every number"]
    # a study without a suite writes a headline and keeps its own rules
    other = S3.check_readability({"headline": "Growth fell.", "answer": six, "claims": []},
                                 headline=True)
    assert not [p for p in other if "Summary" in p]


def test_step_3_sees_the_report_before_it_writes(tmp_path, monkeypatch):
    """The brief ends with the report's figures and tables, the task holds
    SUITE_FINDINGS, and each figure the report drew is attached by name."""
    from PIL import Image
    from agents.analysis import analysis_plan as AP
    from agents.analysis import analysis_recipes as AR
    adir = tmp_path / "04_analysis"
    adir.mkdir()
    Image.new("RGB", (8, 8), "white").save(adir / "figure_over_time.png")
    monkeypatch.setattr(S3, "suite_records", lambda ctx: {})
    monkeypatch.setattr(S3, "review_brief", lambda ctx, c, i: "THE BRIEF")
    monkeypatch.setattr(S3, "report_view", lambda ctx: RF)
    monkeypatch.setattr(AP, "analysis_dir", lambda ctx: adir)
    monkeypatch.setattr(AP, "load_plan", lambda d: PLAN)
    monkeypatch.setattr(AR, "report_lines", lambda q, p: [])
    monkeypatch.setattr(S3, "run_facts", lambda ctx: set())
    monkeypatch.setattr(S3, "_tracer", lambda ctx, records: None)
    monkeypatch.setattr(S3, "audit", lambda *a, **k: {"kept": [], "struck": [], "n_claims": 0,
                                                       "n_struck": 0})
    seen = []

    class Client:
        def ask(self, messages):
            seen.append(messages[0]["content"])
            return json.dumps({"claims": [], "headline_rank": [], "verdict": "sufficient",
                               "answer": "Growth fell in both summers."})

    ctx = SimpleNamespace(plan={"question": "What if?"}, caveats=[], run_dir=tmp_path)
    out = S3.interpret(ctx, {}, {"findings": [], "figures": []}, adir, client=Client())
    assert out["verdict"] == "sufficient" and len(seen) == 1
    text = seen[0][0]["text"]
    assert text.startswith(f"THE BRIEF\n\n{S3.REPORT_VIEW_HEAD}. Code drew them")
    assert S3.SUITE_FINDINGS in text
    # the figure on disk is attached by its name; the one not drawn is not
    labels = [p["text"] for p in seen[0][1:] if p["type"] == "text"]
    assert labels == ["figure: figure_over_time"]


# ── which findings the Results show ─────────────────────────────────────────
TABLE = {"figures": [], "tables": [{"number": 1, "columns": ["Summer 2031"], "rows": [
    {"label": "Growth", "cells": ["-4.25 (-1.5 %)"], "records": ["L2"]},
    {"label": "Area", "cells": ["1,295"], "records": ["L4"]}]}]}


def test_a_finding_is_left_out_only_when_the_tables_print_its_numbers():
    by = RP.numbers_by_record(TABLE)
    assert by == {"L2": ["-4.25", "-1.5"], "L4": ["1295"]}
    same = {"evidence": ["L2"], "values": [-4.2497, -1.5013]}
    assert RP.repeats_table(same, by)
    assert not RP.repeats_table(dict(same, values=[]), by)              # said in words
    assert not RP.repeats_table(dict(same, values=[-4.2497, -3.1]), by)  # one not printed
    assert not RP.repeats_table(dict(same, values=[4.2497]), by)        # the sign counts
    assert not RP.repeats_table(dict(same, evidence=["L9"]), by)        # another record
    assert RP.repeats_table({"evidence": ["L4"], "values": [1295.3]}, by)
    assert RP._prints("9.7e-05", 9.71e-05) and not RP._prints("9.7e-05", 9.8e-05)
    assert RP._prints("1295", 1295.3) and not RP._prints("1295", 1296.0)


def test_a_finding_in_words_on_the_tables_records_stays(tmp_path):
    from test_report_pages import make_report
    from test_report_pages_fixes import three_figures
    rf = three_figures()
    report, adir = make_report(tmp_path, figures=rf)
    report["_plan"] = json.loads((adir / "analysis_plan.json").read_text())
    numbers = RP.numbers_of(report["_plan"], report)

    def placed():
        by_fig, others = RP.choose_findings(report, rf, numbers)
        return {i for v in by_fig.values() for i in v} | {i for _, i in others}

    assert report["claims"][4]["claim"] == "Rain was lower in July 2033."
    assert 5 not in placed()                     # its number prints in Table 1
    report["claims"][4]["values"] = []           # the same records, said in words
    assert 5 in placed()
