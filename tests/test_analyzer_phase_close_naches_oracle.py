"""The two phase-closing fixes (2026-11-08) replayed on the accepted fourth
acceptance batch's recorded studies, with no model call. Real data: run with
--runlocaldata; the studies are read from tests/local_oracle_paths.json, key
"g5_acceptance" (each copied to tmp_path, never written in place). A replay
of recorded responses, not a new acceptance batch.

    pytest tests/test_analyzer_phase_close_naches_oracle.py --runlocaldata -v

U   Every figure that failed in the batch was a bars panel of several units.
    Its recorded specification, drawn again by the framework from the
    batch's own outputs: the three whose split fits four panels are drawn,
    one panel per unit, each output with its record; the other five are
    refused before anything is drawn, each mixed panel's units and outputs
    named. The brief now lists each output's unit by the same rule.
C   Question 1's headline is claim 1's text, whose record is a
    partial-record estimate: its restricted comparability (and its other
    warnings) now stand beside it on the page, on the slide, in the plain
    text and in the study record. Question 2's exploratory label stays, and
    no headline carries a caveat its claim does not.
"""
import json
import shutil
from pathlib import Path

import pytest

from g2_replay_support import forbid_model_clients

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).resolve().parent / "local_oracle_paths.json"
# the batch's failed figures, by question; the first three now fit by a split
DRAWN = {"q1": ["snow_loss_2015", "summer_gpp_2015"], "q2": ["carbon_2022_anomaly"]}
REFUSED = {"q1": ["competing_controls_2015"],
           "q2": ["season_and_timing_2022", "water_anomaly_2022_by_unit",
                  "water_cycle_2022_anomaly"],
           "q3": ["competing_explanations_2015_summer"]}


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


@pytest.fixture(scope="module")
def batch():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    d = (json.loads(CONFIG.read_text()).get("g5_acceptance") or {}).get("dir")
    if not d or not (Path(d) / "q1" / "04_analysis" / "lookups.jsonl").is_file():
        pytest.skip("no g5_acceptance block naming the accepted batch's studies")
    return Path(d)


def _copy(batch, q, tmp_path):
    rd = tmp_path / q
    shutil.copytree(batch / q, rd)
    return rd


def _j(p):
    return json.loads(Path(p).read_text())


def _replay(rd, fid):
    out = rd / "04_analysis"
    spec = _j(out / f"{fid}{S2.SPEC_SUFFIX}")
    f = {p.name: spec.get(p.name) for p in S2.SUITE_FIGURE_PARTS}
    f["id"] = f"{fid}_replay"
    ctx = S0.load(str(rd))
    return f, S2._spec_figure(ctx, f, out, AP.produce_required(ctx, out), AP.load_plan(out),
                              9, [])


def test_the_batchs_failed_figures_failed_on_mixed_units_as_recorded(batch):
    for q in ("q1", "q2", "q3"):
        ev = _j(batch / q / "evaluation.json")
        ph = _j(batch / q / "04_analysis" / "planned_phase.json")
        errs = {h["figure"]: h.get("error") for a in ph["analyses"]
                for h in a.get("attempts") or [] if h.get("figure") not in (None, "required_outputs")
                and h.get("ok") is False}
        assert set(errs) == set(DRAWN.get(q, []) + REFUSED.get(q, [])), (q, errs)
        assert all("its bars are in different units" in e for e in errs.values())
        assert ev["status"] == "supported"                            # the batch as accepted


@pytest.mark.parametrize("q,fid", [(q, f) for q, fs in DRAWN.items() for f in fs])
def test_a_failed_figure_whose_split_fits_is_drawn_one_unit_a_panel(batch, tmp_path, q, fid):
    rd = _copy(batch, q, tmp_path)
    f, (o, finding, cav) = _replay(rd, fid)
    assert o["ok"] and cav is None and (rd / "04_analysis" / f"{fid}_replay.png").is_file()
    asked = [n for p in f["panels"] for n in (p.get("outputs") or [p.get("output")])]
    drawn = [n for p in o["panels"] for n in p["outputs"]]
    assert sorted(drawn) == sorted(asked)                            # nothing dropped
    assert any(p.get("split_from") for p in o["panels"])
    units = AP.output_units(rd / "04_analysis", AP.load_plan(rd / "04_analysis"))
    for p in o["panels"]:
        if p["kind"] in ("bars", "series"):
            assert len({units[n] for n in p["outputs"]}) == 1, p     # one unit each
    lines = {c["label"]: c["line"] for c in _j(rd / "04_analysis" / AP.REQUIRED_FILE)["calls"]}
    assert [x["id"] for x in finding["lookups"]] == [f"L{lines[n]}" for n in
                                                     dict.fromkeys(asked)]
    spec = _j(rd / "04_analysis" / f"{fid}_replay{S2.SPEC_SUFFIX}")
    assert spec["panels"] == f["panels"] and spec["panels_drawn"]


@pytest.mark.parametrize("q,fid", [(q, f) for q, fs in REFUSED.items() for f in fs])
def test_a_failed_figure_whose_split_does_not_fit_is_refused_before_drawing(batch, tmp_path,
                                                                            q, fid):
    rd = _copy(batch, q, tmp_path)
    f, (o, finding, cav) = _replay(rd, fid)
    assert o["ok"] is False and finding is None
    assert o["error"].startswith("refused before drawing: its outputs are in different units "
                                 "within one panel (panel ")
    assert "where a figure holds 4: put each unit's outputs in a panel of their own" in \
        o["error"]
    assert not (rd / "04_analysis" / f"{fid}_replay.png").exists()
    assert cav["id"] == f"figure_failed:{fid}_replay"


def test_q1s_brief_lists_each_outputs_unit_and_no_value(batch, tmp_path):
    rd = _copy(batch, "q1", tmp_path)
    out = rd / "04_analysis"
    units = AP.output_units(out, AP.load_plan(out))
    assert units["P1.O1"] == "days" and units["P2.O1"] == "gC/m^2/season" and \
        units["P2.O2"] == "gC/m^2/yr"
    brief = S2.suite_brief(S0.load(str(rd)))
    assert "(unit: gC/m^2/season)" in brief and "(unit: days)" in brief
    vals = AP.produce_required(S0.load(str(rd)), out)["values"]
    v = float(vals["P2.O1"]["anomaly"])
    for shown in (repr(v), f"{v:.6g}", f"{v:.4g}"):
        assert shown not in brief, shown


def _report(rd, tmp_path):
    out = rd / "04_analysis"
    rep = S4.build(S0.load(str(rd)), {}, _j(out / "investigation.json"),
                   _j(out / "interpretation.json"), rd,
                   steps={"investigate": True, "interpret": True})
    return rep


def test_q1s_headline_keeps_its_restricted_comparability_wherever_it_is_shown(batch,
                                                                            tmp_path):
    rd = _copy(batch, "q1", tmp_path)
    out = rd / "04_analysis"
    accepted = _j(out / "analysis.json")
    rep = _report(rd, tmp_path)
    assert rep["headline"] == accepted["headline"]                   # the same claim's text
    assert rep["headline_selection"]["source"] == {
        "claim": 1, "evidence": ["L1"],
        "warnings": ["restricted_comparability", "flag:partial_record", "eligibility"]}
    line = CT.headline_caveat_line(rep)
    assert line.startswith("Caveats it carries from claim 1 (L1): restricted comparability (a "
                           "partial-record estimate is in a shift: the water year starts before "
                           "the run's records); partial record (")
    assert rep["headline_text"] == rep["headline"] + " " + line
    S4.write(rep, out, slides=True)
    page = (out / "REPORT.md").read_text()
    para = page[page.index("## The answer"):].split("\n\n")[1]
    assert para == f"**{rep['headline']}**  \n*{line}*"              # beside it, one paragraph
    from pptx import Presentation
    texts = [sh.text_frame.text for sh in Presentation(str(out / "analysis.pptx")).slides[1]
             .shapes if sh.has_text_frame]
    assert texts[texts.index(rep["headline"]) + 1] == line
    from agents import study_record as SRD
    sec = SRD.render(rd)
    sec = sec[sec.index("## The Analyzer"):]
    assert "restricted comparability (a partial-record estimate is in a shift" in \
        sec[:sec.index("- status")]


def test_q2_keeps_its_exploratory_label_and_no_headline_gains_a_caveat_its_claim_lacks(
        batch, tmp_path):
    for q in ("q2", "q3"):
        rd = _copy(batch, q, tmp_path)
        accepted = _j(rd / "04_analysis" / "analysis.json")
        rep = _report(rd, tmp_path)
        assert rep["headline"] == accepted["headline"]
        assert rep["headline_caveats"] == [] and CT.headline_caveat_line(rep) is None
        assert rep["headline_text"] == rep["headline"]
        if q == "q2":
            assert rep["headline"].startswith(S4.EXPLORATORY_PREFIX)
