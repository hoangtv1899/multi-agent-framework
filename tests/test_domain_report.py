"""A whole-domain experiment's report, rendered from its records. No LLM.

    pytest tests/test_domain_report.py -v

The records are written here, so every line the report should hold is known:
the headline is one claim with that claim's numbers, at 3 significant figures,
and what a run's difference does NOT mean stands beside what it does.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from agents.analysis import domain_report as DR  # noqa: E402

YEARS = [2015, 2016]
FIELDS = {"GPP": {"stream": "h0", "how": "sum", "units": "gC/m^2/yr", "note": None},
          "FAREA_BURNED": {"stream": "h0", "how": "sum", "units": "proportion/yr",
                           "note": "a fraction of the cell burned PER SECOND (FireMod.F90)"}}
REGIONS = {"vegetated cells": {"cells": 100, "area_km2": 60.0, "rule": "PCT_NATVEG >= 100"},
           "watershed, vegetated": {"cells": 40, "area_km2": 24.0, "rule": "inside"}}


def _values(gpp, fire):
    return {reg: {"GPP": {"2015": gpp[0] * k, "2016": gpp[1] * k},
                  "FAREA_BURNED": {"2015": fire[0], "2016": fire[1]}}
            for reg, k in (("vegetated cells", 1.0), ("watershed, vegetated", 1.1))}


def _table(with_twin=True, with_event=False):
    runs = {"control": {"fields": FIELDS, "values": _values((487.0, 536.5), (0.01, 0.005)),
                        "problems": []}}
    t = {"regions": REGIONS, "years": YEARS, "runs": runs, "problems": [],
         "differences": {}}
    if with_twin:
        runs["nofire"] = {"fields": FIELDS, "values": _values((480.0, 530.0), (0.0, 0.0)),
                          "problems": []}
        t["differences"] = {"nofire": {reg: {v: {y: runs["nofire"]["values"][reg][v][y]
                                                 - runs["control"]["values"][reg][v][y]
                                                 for y in ("2015", "2016")}
                                             for v in FIELDS} for reg in REGIONS}}
    if with_event:
        t["event"] = {"normal_years": [2016], "target_years": [2015], "problems": [],
                      "anomalies": {"vegetated cells": {"GPP": {
                          "normal": 536.5, "normal_years": [2016],
                          "years": {"2015": {"value": 487.0, "anomaly": -49.5,
                                             "anomaly_pct": -9.2264, "rank_lowest_first": 1,
                                             "of": 2}}}}}}
    # as the server makes it: the observed burned area per region and year, and
    # each run's own (its mean burned fraction times the region's area)
    model = {r: {reg: {y: frac * REGIONS[reg]["area_km2"] for y, frac in by["FAREA_BURNED"].items()}
                 for reg, by in runs[r]["values"].items()} for r in runs}
    t["observed"] = {"burned_area_km2": {
        "what": "burned area", "units": "km2", "source": "MODIS MCD64A1",
        "values": {"vegetated cells": {"2015": 0.2, "2016": 0.1},
                   "watershed, vegetated": {"2015": 0.1, "2016": 0.0}},
        "model_field": "FAREA_BURNED", "model": model}}
    return t


NOFIRE = {"run": "nofire", "case": "ctl_nofire", "reused": False, "status": "completed",
          "changes": [{"id": "fire_off", "params": {},
                       "meaning": "the difference from the control is what the model's own "
                                  "fire does",
                       "not_meaning": "the effect of an observed fire"}]}
CONTROL_ROW = {"run": "control", "reused": True, "status": "reused", "changes": []}


def _run_dir(tmp_path, table, runs, analysis, caveats=(), feasibility=None):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain", "regional": table,
        "experiment": {"control": {"name": "naches", "reuse": True}, "runs": runs,
                       "check": {"unusual": [{"where": "control",
                                              "what": "reused without a record"}]}},
        "chain": {"years_planned": YEARS, "years_passed": YEARS, "first_unpassed": None}}))
    (rd / "strategy.json").write_text(json.dumps({
        "domain_experiment": {"analysis": analysis, "caveats": list(caveats)},
        "feasibility": feasibility or {"verdict": "full", "why": "every change is available"},
        "requires": []}))
    (rd / "reception.json").write_text(json.dumps(
        {"brief": {"user_request": "What does the model's own fire do to GPP?"}}))
    return rd


ANALYSIS = {"regions": [{"name": "watershed"}, {"name": "vegetated cells"}],
            "variables": ["GPP", "FAREA_BURNED"], "normal_years": []}


class TestTheHeadline:

    def test_a_what_if_run(self, tmp_path):
        rd = _run_dir(tmp_path, _table(), [CONTROL_ROW, NOFIRE], ANALYSIS)
        out = DR.build(rd)
        # watershed values are 1.1 x: -7 x 1.1 = -7.7 (-1.44 %); -6.5 x 1.1 = -7.15 (-1.21 %)
        assert out["headline"] == ("With fire_off (run nofire), GPP over watershed, vegetated "
                                   "differed from the control by -7.7 gC/m^2/yr in 2015 "
                                   "(-1.44 %) and -7.15 gC/m^2/yr in 2016 (-1.21 %).")
        rec = json.loads((rd / "04_analysis" / "domain_report.json").read_text())
        assert rec["claim"]["kind"] == "difference" and rec["claim"]["run"] == "nofire"
        assert rec["claim"]["years"]["2015"]["difference"] == pytest.approx(-7.7)

    def test_an_event_on_the_control_alone(self, tmp_path):
        rd = _run_dir(tmp_path, _table(with_twin=False, with_event=True), [CONTROL_ROW],
                      dict(ANALYSIS, regions=[{"name": "vegetated cells"}], normal_years=[2016]))
        out = DR.build(rd)
        assert out["headline"] == ("GPP over vegetated cells was 9.23 % below its 2016 "
                                   "normal (536 gC/m^2/yr) in 2015.")
        page = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        assert "The control against its 2016 normal" in page
        assert "| vegetated cells | 536 | -9.23 % (rank 1 of 2) |" in page
        figs = sorted(p.name for p in (rd / "04_analysis").glob("*.png"))
        assert figs == ["domain_fig_GPP_anomaly.png", "domain_fig_GPP_by_year.png"]

    def test_an_event_of_two_years_names_the_normal_once(self, tmp_path):
        t = _table(with_twin=False, with_event=True)
        t["years"] = [2015, 2016, 2017]
        for reg in t["runs"]["control"]["values"].values():
            reg["GPP"]["2017"] = 547.23
        a = t["event"]["anomalies"]["vegetated cells"]["GPP"]
        a["years"]["2017"] = {"value": 547.23, "anomaly": 10.73, "anomaly_pct": 2.0,
                              "rank_lowest_first": 3, "of": 3}
        t["event"]["target_years"] = [2015, 2017]
        rd = _run_dir(tmp_path, t, [CONTROL_ROW],
                      dict(ANALYSIS, regions=["vegetated cells"], normal_years=[2016]))
        assert DR.build(rd)["headline"] == ("GPP over vegetated cells was 9.23 % below its 2016 "
                                            "normal (536 gC/m^2/yr) in 2015 and 2 % above it "
                                            "in 2017.")

    def test_an_event_in_kelvin_has_no_percent(self, tmp_path):
        t = _table(with_twin=False, with_event=True)
        t["runs"]["control"]["fields"] = {"TSA": {"stream": "h0", "how": "mean", "units": "K"}}
        t["runs"]["control"]["values"] = {"vegetated cells": {"TSA": {"2015": 281.5,
                                                                      "2016": 280.0}}}
        t["event"]["anomalies"] = {"vegetated cells": {"TSA": {
            "normal": 280.0, "years": {"2015": {"value": 281.5, "anomaly": 1.5,
                                                "anomaly_pct": None, "rank_lowest_first": 2,
                                                "of": 2}}}}}
        rd = _run_dir(tmp_path, t, [CONTROL_ROW],
                      {"regions": ["vegetated cells"], "variables": ["TSA"]})
        assert DR.build(rd)["headline"] == ("TSA over vegetated cells was 1.5 K above its "
                                            "2016 normal (280 K) in 2015.")

    def test_a_pure_number_has_no_unit_after_it(self, tmp_path):
        t = _table(with_twin=False)
        t["runs"]["control"]["fields"] = {"TLAI": {"stream": "h0", "how": "mean", "units": "1"}}
        t["runs"]["control"]["values"] = {"vegetated cells": {"TLAI": {"2015": 0.747,
                                                                       "2016": 0.768}}}
        rd = _run_dir(tmp_path, t, [CONTROL_ROW],
                      {"regions": ["vegetated cells"], "variables": ["TLAI"]})
        assert DR.build(rd)["headline"] == "TLAI over vegetated cells: 0.747 in 2015 and 0.768 in 2016."
        page = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        assert "### TLAI (dimensionless, mean over the year)" in page
        rec = json.loads((rd / "04_analysis" / "domain_report.json").read_text())
        assert rec["claim"]["units"] == "1"

    def test_values_alone(self, tmp_path):
        rd = _run_dir(tmp_path, _table(with_twin=False), [CONTROL_ROW],
                      dict(ANALYSIS, regions=[{"name": "vegetated cells"}]))
        assert DR.build(rd)["headline"] == ("GPP over vegetated cells: 487 gC/m^2/yr in 2015 "
                                            "and 536 gC/m^2/yr in 2016.")


@pytest.mark.parametrize("years,said", [([2016], "2016"),
                                        ([2020, 2016, 2017, 2018, 2019], "2016-2020"),
                                        ([2016, 2018, 2020], "2016, 2018 and 2020")])
def test_years_as_a_reader_says_them(years, said):
    assert DR.span(years) == said


class TestThePage:

    def _page(self, tmp_path, **kw):
        rd = _run_dir(tmp_path, _table(), [CONTROL_ROW, NOFIRE], ANALYSIS, **kw)
        DR.build(rd)
        return rd, (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()

    def test_what_was_run_and_what_it_means(self, tmp_path):
        rd, page = self._page(tmp_path)
        assert page.startswith("# What does the model's own fire do to GPP?")
        assert "status **complete**" in page
        assert "- run **nofire**: the control with fire_off (status completed)" in page
        assert ("  - fire_off means the difference from the control is what the model's own "
                "fire does; it is NOT the effect of an observed fire") in page

    def test_the_numbers(self, tmp_path):
        rd, page = self._page(tmp_path)
        assert "### GPP (gC/m^2/yr, total over the year)" in page
        assert "| watershed, vegetated | 536 | 590 |" in page
        assert "| watershed, vegetated | -7.7 (-1.44 %) | -7.15 (-1.21 %) |" in page
        assert "### Burned area: the model's own and the observed (km2)" in page
        # each run's own, as the server made it, then the observation
        assert "| region | year | model, control | model, nofire | observed |" in page
        assert "| vegetated cells | 2015 | 0.6 | 0 | 0.2 |" in page
        assert "FAREA_BURNED |" not in page and "| watershed, vegetated | 2016 | 0.12 | 0 | 0 |" \
            in page

    def test_the_caveats_and_the_sources(self, tmp_path):
        rd, page = self._page(tmp_path, caveats=["the model's fire \u2014 not the 2021 fire"])
        assert ("- Run nofire, fire_off: its difference from the control is NOT the effect "
                "of an observed fire") in page
        assert "- the model's fire, not the 2021 fire" in page          # a dash made a comma
        assert "- control: reused without a record" in page
        assert "- FAREA_BURNED: a fraction of the cell burned PER SECOND" in page
        assert "every number above: 03_results/regional.json" in page
        assert "\u2014" not in page and "validated" not in page.lower()

    def test_a_partial_design_says_what_it_needs(self, tmp_path):
        rd, page = self._page(tmp_path, feasibility={"verdict": "partial",
                                                     "why": "the moved burn cannot run."})
        assert "The design was judged **partial**: the moved burn cannot run." in page

    def test_the_figures(self, tmp_path):
        rd, page = self._page(tmp_path)
        figs = sorted(p.name for p in (rd / "04_analysis").glob("*.png"))
        assert figs == ["domain_fig_GPP_by_year.png", "domain_fig_GPP_difference.png"]
        assert "![domain_fig_GPP_by_year.png](domain_fig_GPP_by_year.png)" in page

    def test_an_incomplete_chain_says_so(self, tmp_path):
        rd = _run_dir(tmp_path, _table(), [CONTROL_ROW, NOFIRE], ANALYSIS)
        e = json.loads((rd / "experiment.json").read_text())
        e["chain"] = {"years_planned": YEARS, "years_passed": [2015],
                      "first_unpassed": "nofire: y2016"}
        (rd / "experiment.json").write_text(json.dumps(e))
        DR.build(rd)
        assert ("status **incomplete: the chain stopped at nofire: y2016 (years passed "
                "[2015] of [2015, 2016])**") in (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()

    def test_a_missing_year_is_incomplete_even_unnamed(self, tmp_path):
        rd = _run_dir(tmp_path, _table(), [CONTROL_ROW, NOFIRE], ANALYSIS)
        e = json.loads((rd / "experiment.json").read_text())
        e["chain"] = {"years_planned": YEARS, "years_passed": [2015], "first_unpassed": None}
        (rd / "experiment.json").write_text(json.dumps(e))
        DR.build(rd)
        assert "status **incomplete: years passed [2015] of [2015, 2016]**" in \
            (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()

    def test_no_regional_table_is_refused(self, tmp_path):
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "experiment.json").write_text("{}")
        with pytest.raises(ValueError, match="no regional table"):
            DR.build(rd)
