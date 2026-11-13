"""Two wordings a live report used, which the claim trace read wrongly.

    pytest tests/test_claim_wording_cases.py -v

Both were found on 2026-11-12, in the first report written after a series
kept its values: one true claim was struck for four problems, none of them
real.

    a thousand or more,     The report writes its numbers to three significant
    at three figures        figures, so a value of 1,538.09 is "1,540". The
                            trace held a whole number to the unit and found no
                            value. A printed whole number of four digits or
                            more now also states the value whose three-figure
                            rounding it is, and nothing looser, and only where
                            it is no label of its records.
    "the grid cells"        The page's own words for runoff are "the water the
                            grid cells generate". Read as "the grid", they
                            named the region of every vegetated cell, and its
                            elevation band of the same edges.

The study is the made-up one of tests/test_series_values.py, whose yearly
values are above a thousand.
"""
import shutil

import numpy as np
import pytest

import domain_access as DAC
from test_series_values import LINE, REGION, _check, _claim, _value, study  # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402


# ── a thousand or more, at three significant figures ─────────────────────────

@pytest.mark.parametrize("printed, value, states", [
    ("1540", 1538.09, True), ("1,540", 1538.09, True), ("-1540", -1538.09, True),
    ("1540", 1535.0, True), ("1540", 1544.9, True),
    ("1540", 1548.0, False), ("1540", 1534.9, False),     # 1550 and 1530 at three figures
    ("12300", 12345.0, True), ("12400", 12345.0, False),
    ("1000", 999.6, True),                                 # 999.6 is 1000 at three figures
    ("154", 153.8, False),                                 # three digits: held to the unit
    ("1538.1", 1538.09, False), ("1540.2", 1538.09, False),  # a decimal: to its last digit
    ("1540", None, False), ("", 1538.09, False), ("1540", 0.0, False)])
def test_what_a_whole_number_states_at_three_figures(printed, value, states):
    assert CT.is_three_sig(printed, value) is states


def test_a_yearly_value_above_a_thousand_may_be_written_to_three_figures(study):
    v = _value(study, "year", "2016")
    assert 1000 < v < 10000 and CT._three_sig(v) != f"{v:.0f}"      # the rounding moves it
    rounded = f"{int(CT._three_sig(v)):,}"                           # as a report writes it
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {rounded} gC/m^2/yr in 2016.", values=[v]))
    assert t["problems"] == [], t["problems"]
    assert "L1:values.2016" in t["matched"]
    # to the unit it still passes, and a number that is neither is still refused
    ok = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {v:,.0f} gC/m^2/yr in 2016.", values=[v]))
    assert ok["problems"] == [], ok["problems"]
    off = int(CT._three_sig(v)) + 10
    bad = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {off:,} gC/m^2/yr in 2016.", values=[v]))
    assert any(f"the number {off} matches no value" in p for p in bad["problems"]), \
        bad["problems"]


def test_a_rounded_series_value_is_still_held_to_its_own_year(study):
    v = _value(study, "year", "2016")
    rounded = f"{int(CT._three_sig(v)):,}"
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {rounded} gC/m^2/yr in 2017.", values=[v]))
    assert any("a series' value for 2016 (L1), a period the claim does not name" in p
               for p in t["problems"]), t["problems"]


def _with_a_band(study, tmp_path, region, values):
    """The study's log with one more yearly series, of a region the made-up
    control does not hold, as the access would log it."""
    out = tmp_path / "04_analysis"
    shutil.copytree(study["out"], out)
    value = {"field": "GPP", "region": region, "step": "year", "labels": ["2015", "2016"],
             "values": np.array(values, dtype=np.float64),
             "covered": np.array([1.0, 1.0]), "units": "gC/m^2/yr", "how": "total"}
    AP.LookupLog(out, script="fig")({
        "call": "get_series", "keys": {"run": "control", "field": "GPP", "region": region,
                                       "step": "year"}, "method": None, "settings": None,
        "sha256": DAC.fingerprint(value), "shape": DAC._shape(value),
        "result": DAC.flat_result("get_series", value), "planned": True, "plan_items": ["P1"]})
    recs = CT.evidence_records(out)
    return recs, sorted(recs, key=lambda i: int(i[1:]))[-1]


def test_a_label_of_the_record_is_never_taken_for_a_value_near_it(study, tmp_path):
    """The 1000 of "below 1000 m" is the band's edge. The 2015 value of that
    band is 1,003.2, which is 1000 at three figures: read as that value, the
    edge would be a number of a year the claim does not name."""
    recs, rid = _with_a_band(study, tmp_path, "vegetated cells, below 1000 m", (1003.2, 52.5))
    assert CT.is_three_sig("1000", 1003.2)
    c = {"claim": "Over the vegetated cells below 1000 m, GPP was 52.5 gC/m^2/yr in 2016.",
         "finding_id": "f", "values": [52.5], "evidence": [rid],
         "evidence_class": "observation", "warnings": []}
    t = _check(study, c, records=recs)
    assert t["problems"] == [], t["problems"]
    assert t["matched"] == [f"{rid}:values.2016"]


def test_a_summary_sentence_may_round_to_three_figures_too(study):
    v = _value(study, "year", "2016")
    rounded = f"{int(CT._three_sig(v)):,}"
    claim = _claim(study, f"Over the vegetated cells, GPP was {v:.5g} gC/m^2/yr in 2016.",
                   values=[v])
    said = CT.summary_support(f"In 2016 GPP was {rounded} gC/m^2/yr.", [claim],
                              study["records"], {}, study["plan"])
    assert said[0]["claims"] == [1] and said[0]["unsupported"] == [], said
    off = f"{int(CT._three_sig(v)) + 10:,}"
    said = CT.summary_support(f"In 2016 GPP was {off} gC/m^2/yr.", [claim],
                              study["records"], {}, study["plan"])
    assert said[0]["claims"] == [] and said[0]["unsupported"], said


def test_a_headline_that_rounds_to_three_figures_states_its_claims_value():
    claims = [{"claim": "another", "values": [12.5]}, {"claim": "the one", "values": [1538.09]}]
    assert CT._headline_claims("Runoff was 1,540 mm in 2017.", claims) == [1]
    assert CT._headline_claims("Runoff was 1,560 mm in 2017.", claims) == []


# ── "the grid cells" names no region ─────────────────────────────────────────

@pytest.mark.parametrize("words, names", [
    ("the runoff the grid cells generate", False),
    ("runoff generated by the grid's cells", False),
    ("the water each of the grid cells generates, in one grid cell", False),
    ("over the grid", True), ("across the whole grid", True),
    ("over the grid, in the water the grid cells generate", True),
    ("over the vegetated cells", True), ("over the domain", True),
    ("over the watershed's grid cells", False)])
def test_which_words_name_the_region_of_all_cells(words, names):
    assert CT.names_all_cells(words) is names


def test_runoff_the_grid_cells_generate_names_no_region(study, tmp_path):
    recs, rid = _with_a_band(study, tmp_path, "box, vegetated, below 1000 m", (41.5, 52.5))
    regions = [REGION, "below 1000 m", "box, vegetated", "box, vegetated, below 1000 m"]

    def check(text):
        c = {"claim": text, "finding_id": "f", "values": [52.5], "evidence": [rid],
             "evidence_class": "observation", "warnings": []}
        return CT.check_claim(c, recs, study["plan"], {}, regions)["problems"]
    assert check("In the box below 1000 m, the runoff the grid cells generate was 52.5 mm "
                 "in 2016.") == []
    # the region of every cell, named outright, is still a region its record must name
    probs = check("Over the grid below 1000 m, the runoff was 52.5 mm in 2016.")
    assert any("names the region 'vegetated cells'" in p for p in probs), probs
    assert any("names the band 'below 1000 m'" in p for p in probs), probs
