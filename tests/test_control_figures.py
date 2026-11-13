"""The control run in two figures, on the design pages (agents/control_figures).

    pytest tests/test_control_figures.py -v

Every number here is made up. The record is the shape the model server gives
(domain_measured.page_record): a made-up valley with snow at two pillows, a
river with a dam, and one map. The pages are test_design_page's.
"""
import json

import pytest

import domain_measured as DM
from agents import control_figures as CF
from agents import design_page as DP
from agents import design_review as dr
from agents.analysis import analysis_plan as AP
from agents.analysis import figure_spec as FS
from test_design_page import (PREVIEW, _flat, _reception, _record, _section, _strategy,
                              folder)  # noqa: F401
from test_domain_measured import built  # noqa: F401

YEARS = (2031, 2032, 2033)
LABELS = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
SNOW_MODEL = [10.0] * 36
SNOW_SEEN = [20.0] * 12 + [30.0] * 12 + [40.0] * 12
FLOW_MODEL = [3.0] * 12 + [4.0] * 12 + [5.0] * 12
FLOW_SEEN = [2.0] * 12 + [3.0] * 5 + [None] + [3.0] * 6 + [4.0] * 12       # June 2032 missing
FLOW_FREE = [2.5] * 12 + [3.5] * 12 + [4.5] * 12


def _pair(**kw):
    base = {"how": "total", "units": "mm/month", "where": "valley floor", "cells": 300,
            "table": {"path": "a.csv", "sha256": "0" * 64, "months": 36}, "months_both": 36}
    base.update(kw)
    return base


def _rec():
    return {
        "control": "valley", "problems": [],
        "measured": {"version": 1, "control": "valley", "step": "month", "labels": list(LABELS),
                     "problems": [], "series": [
            _pair(id="snow", field="SNOWX", how="mean", units="mm",
                  where="the 2 cells that hold the points", cells=2,
                  title="Snow at two made-up pillows", label="measured",
                  model_words="Snow at the cells of the two pillows",
                  measured_words="measured at the pillows", source="a made-up network",
                  model=list(SNOW_MODEL), measured=list(SNOW_SEEN)),
            _pair(id="flow", field="RUNX", title="Runoff and river flow",
                  label="measured river flow", model_words="Runoff from the valley floor",
                  measured_words="of measured river flow", source="a made-up gauge",
                  model=list(FLOW_MODEL), measured=list(FLOW_SEEN)),
            _pair(id="flow_free", field="RUNX", title="Runoff and river flow",
                  label="river flow without the dam", model_words="Runoff from the valley floor",
                  measured_words="without the dam", source="a made-up estimate",
                  model=list(FLOW_MODEL), measured=list(FLOW_FREE))]},
        "maps": {"lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0, 13.0],
                 "panels": [{"words": "Rain", "fields": ["RAINX", "SLEETX"], "units": "mm/yr",
                             "how": "total", "years": list(YEARS),
                             "values": [[100.0, 200.0, 300.0, 400.0], [150.0, None, 350.0, 450.0],
                                        [120.0, 220.0, 320.0, 420.0]]}],
                 "outline": {"region": "valley floor",
                             "mask": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]}}}


# ── the numbers a sentence states ────────────────────────────────────────────
class TestTheNumbers:
    @pytest.mark.parametrize("x, text", [(12.34, "12.3"), (456.7, "457"), (1234.5, "1,230"),
                                         (0.04567, "0.0457"), (0.0, "0"), (99.96, "100")])
    def test_three_significant_figures(self, x, text):
        assert CF.three(x) == text

    def test_two_measurements_of_one_quantity_are_one_panel(self):
        ps = CF.panels(_rec())
        assert [p["title"] for p in ps] == ["Snow at two made-up pillows", "Runoff and river flow"]
        assert [m["label"] for m in ps[1]["measured"]] == ["measured river flow",
                                                           "river flow without the dam"]
        assert ps[1]["model"] == FLOW_MODEL

    def test_a_mean_is_over_the_months_asked_about(self):
        n = CF.panel_numbers(CF.panels(_rec())[0], [2032, 2033])
        assert n == {"years": [2032, 2033], "total": False, "model": 10.0, "measured": [35.0]}

    def test_a_total_is_a_whole_years_and_a_year_with_a_gap_does_not_count(self):
        # 2032 lacks June in one of the two measurements: it counts for neither,
        # nor for the model, so the three numbers are of the same months
        n = CF.panel_numbers(CF.panels(_rec())[1], [2032, 2033])
        assert n == {"years": [2033], "total": True, "model": 60.0, "measured": [48.0, 54.0]}

    def test_a_total_over_two_whole_years_is_their_mean(self):
        n = CF.panel_numbers(CF.panels(_rec())[1], [2031, 2033])
        assert n["years"] == [2031, 2033]
        assert n["model"] == pytest.approx((36.0 + 60.0) / 2)
        assert n["measured"] == pytest.approx([(24.0 + 48.0) / 2, (30.0 + 54.0) / 2])

    def test_years_the_record_does_not_hold_give_nothing(self):
        assert CF.panel_numbers(CF.panels(_rec())[0], [2040]) is None
        assert CF.sentences(_rec(), [2040]) == []


class TestTheSentences:
    def test_one_sentence_per_panel_with_each_number_beside_what_it_is(self):
        assert CF.sentences(_rec(), [2033]) == [
            "Snow at the cells of the two pillows averages 10.0 mm in the control run in "
            "2033, compared with 40.0 mm measured at the pillows.",
            "Runoff from the valley floor totals 60.0 mm a year in the control run in 2033, "
            "compared with 48.0 mm a year of measured river flow and 54.0 mm a year without "
            "the dam."]

    def test_a_span_of_years_is_said_as_a_span_and_only_the_years_that_count(self):
        s = CF.sentences(_rec(), [2032, 2033])
        assert "in the control run over 2032 to 2033, compared with 35.0 mm" in s[0]
        assert "in the control run in 2033, compared with" in s[1]        # 2032 has a gap

    def test_no_sentence_is_written_by_a_model(self):
        # the words are the record's and the numbers are its values: nothing else
        s = " ".join(CF.sentences(_rec(), [2033]))
        for word in ("too", "well", "poor", "good", "bias", "under", "over 20"):
            assert word not in s


# ── the figures ──────────────────────────────────────────────────────────────
class TestTheFigures:
    def test_the_series_figure_names_the_models_line_and_marks_the_measured_ones(self):
        spec, refs = CF.series_figure(_rec(), [2033])
        assert spec["layout"] == "stack"
        assert [p["outputs"] for p in spec["panels"]] == [["S1.model", "S1.M1"],
                                                          ["S2.model", "S2.M1", "S2.M2"]]
        assert all(p["shade"] == [2033] and p["kind"] == "series" for p in spec["panels"])
        assert refs["S1.model"]["out"]["label"] == "control run"
        assert not refs["S1.model"]["out"].get("measured")
        assert refs["S2.M2"]["out"]["measured"] is True
        assert refs["S2.M2"]["out"]["label"] == "river flow without the dam"
        assert refs["S2.M1"]["value"]["values"] == FLOW_SEEN
        assert {r["out"]["title"] for n, r in refs.items() if n.startswith("S2")} == \
            {"Runoff and river flow"}

    def test_no_years_asked_means_no_shading(self):
        spec, _ = CF.series_figure(_rec())
        assert all("shade" not in p for p in spec["panels"])

    def test_the_maps_figure_carries_the_outline_and_each_maps_own_words(self):
        spec, refs, ll = CF.maps_figure(_rec())
        assert spec["panels"] == [{"kind": "map", "output": "M1"}]
        assert refs["M1"]["out"] == {"field": "RAINX+SLEETX", "title": "Rain, sum of a year",
                                     "units": "mm/yr"}
        assert ll["outline"] == [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]

    def test_both_are_drawn_into_the_study_folder(self, tmp_path):
        got = CF.draw(_rec(), tmp_path, [2033])
        assert got == {"series": "design_figures/control_run_series.png",
                       "maps": "design_figures/control_run_maps.png", "problems": []}
        for k in ("series", "maps"):
            assert (tmp_path / got[k]).stat().st_size > 5000

    def test_a_record_with_nothing_to_draw_draws_nothing(self, tmp_path):
        assert CF.draw(None, tmp_path) == {"series": None, "maps": None, "problems": []}
        bare = {"control": "valley", "measured": None, "maps": None, "problems": []}
        assert CF.draw(bare, tmp_path) == {"series": None, "maps": None, "problems": []}

    def test_a_figure_that_cannot_be_drawn_says_why_and_stops_nothing(self, tmp_path):
        rec = _rec()
        for s in rec["measured"]["series"]:
            s["model"] = [None] * 36
            s["measured"] = [None] * 36
        got = CF.draw(rec, tmp_path)
        assert got["series"] is None and got["maps"] is not None
        assert "the series were not drawn" in got["problems"][0]
        assert not (tmp_path / "design_figures" / CF.SERIES_PNG).exists()


# ── what the figure tool draws for it ────────────────────────────────────────
class TestTheTool:
    def _axes(self):
        plt = FS._plt()
        fig, ax = plt.subplots()
        return plt, fig, ax

    def test_a_run_of_years_is_one_band(self):
        assert FS._year_runs(LABELS, [2032]) == [(12, 23)]
        assert FS._year_runs(LABELS, [2031, 2033]) == [(0, 11), (24, 35)]
        assert FS._year_runs(LABELS, []) == [] and FS._year_runs(LABELS, None) == []

    def test_a_measured_line_is_black_with_its_points_and_the_years_are_shaded_once(self):
        plt, fig, ax = self._axes()
        _, refs = CF.series_figure(_rec(), [2032])
        rs = [refs[n] for n in ("S2.model", "S2.M1", "S2.M2")]
        n = FS._draw_series(ax, rs, {"shade": [2032]}, (), "panel 1")
        model, first, second = ax.get_lines()
        assert first.get_color() == "k" and first.get_marker() == "o"
        assert second.get_linestyle() == "--" and model.get_marker() in ("None", None, "")
        assert [t.get_text() for t in ax.get_legend().get_texts()] == \
            ["control run", "measured river flow", "river flow without the dam"]
        assert len(ax.patches) == 1                      # one band, not one per line
        assert n == 36 + 35 + 36 and ax.get_ylabel() == "mm/month"
        plt.close(fig)

    def test_without_the_new_keys_a_series_is_drawn_as_before(self):
        plt, fig, ax = self._axes()
        r = {"ref": "P1.O1", "kind": "series", "made": True, "outcome": None,
             "out": {"field": "LEAFX", "step": "month"},
             "value": {"labels": LABELS, "values": SNOW_MODEL, "step": "month", "units": "1"}}
        FS._draw_series(ax, [r], {}, (), "panel 1")
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["P1.O1 LEAFX"]
        assert len(ax.patches) == 0
        plt.close(fig)

    def test_a_panels_title_is_its_outputs_own_where_they_share_one(self):
        plt, fig, ax = self._axes()
        FS._title(ax, ["a", "b"], {"a": {"kind": "series", "title": "Snow at two pillows"},
                                   "b": {"kind": "series", "title": "Snow at two pillows"}})
        assert ax.get_title(loc="left") == "Snow at two pillows"
        FS._title(ax, ["a", "b"], {"a": {"kind": "series", "field": "X", "title": "One"},
                                   "b": {"kind": "series", "field": "Y"}})
        assert ax.get_title(loc="left").startswith("a: X")       # not shared: the keys, as before
        plt.close(fig)

    def test_a_map_takes_its_outputs_units_and_an_outline(self):
        plt, fig, ax = self._axes()
        _, refs, ll = CF.maps_figure(_rec())
        n = FS._draw_map(plt, ax, refs["M1"], {}, ll, "panel 1")
        assert n == 11                                    # twelve cells, one with no value
        kinds = [type(c).__name__ for c in ax.collections]
        assert "QuadMesh" in kinds and len(kinds) == 2    # the map, and the region's edge
        assert fig.axes[-1].get_ylabel() == "mm/yr"
        plt.close(fig)

    def test_the_new_keys_are_not_in_the_list_the_analyzer_is_shown(self):
        shown = " ".join(FS.panel_lines(()))
        for key in ("shade", "measured", "outline", "layout", "stack"):
            assert key not in shown


# ── the years the page sets off ──────────────────────────────────────────────
class TestTheYearsAskedAbout:
    def test_they_are_the_persons_years_that_the_record_holds(self):
        assert CF.asked_years(_reception()["brief"], _strategy(), _rec()) == [2033]

    def test_the_whole_record_is_not_set_off(self):
        b = _reception(run_settings={"requested_period": {"yr_start": 2031, "yr_end": 2033}})
        assert CF.asked_years(b["brief"], _strategy(), _rec()) == []

    def test_a_request_that_names_no_year_sets_none_off(self):
        b = _reception(run_settings={})
        assert CF.asked_years(b["brief"], _strategy(), _rec()) == []


# ── the pages ────────────────────────────────────────────────────────────────
def _written(folder, record="given", **kw):
    st = _strategy()
    (folder / "reception.json").write_text(json.dumps(_reception()))
    (folder / "strategy.json").write_text(json.dumps(st))
    args = {} if record == "left out" else {"control_figures": _rec() if record == "given"
                                            else record}
    dr.write_review(folder, reception=_reception(), strategy=st, preview=PREVIEW,
                    analysis_preview=_record(folder, st), **args, **kw)
    return (folder / "DESIGN_REVIEW.md").read_text(), (folder / DP.DETAILS_NAME).read_text()


class TestThePages:
    def test_the_short_page_shows_the_series_and_one_sentence_per_quantity(self, folder):
        page, _ = _written(folder)
        sec = _section(page, "The control run")
        assert "![The control run and the measurements, month by month]" \
               "(design_figures/control_run_series.png)" in sec
        flat = _flat(sec)
        assert "The figure sets the control run beside the measurements for each month of " \
               "2031 to 2033. The shaded year is the one you asked about." in flat
        for s in CF.sentences(_rec(), [2033]):
            assert s in flat
        assert "The maps of the control run, the sources of the measurements and the run's " \
               "full record are in `DESIGN_DETAILS.md`." in flat
        assert (folder / "design_figures" / CF.SERIES_PNG).is_file()

    def test_the_summarys_numbers_are_printed_only_where_there_is_no_figure(self, folder):
        st = _strategy()
        row = st["domain_rulebook"]["controls"]["controls"][0]
        row["page_summary"] = {"version": 4, "written": "2031-02-04",
                               "text": ["The control run is a made-up valley."],
                               "compared": ["Its snow in 2031 was half of the measured."]}
        figures = {"record": _rec(), "years": [2033], "drawn": {"series": "f/s.png"}}
        with_fig = _flat(_section(DP.render(_reception(), st, run_dir=folder, preview=PREVIEW,
                                            analysis_preview=_record(folder, st),
                                            control_figures=figures), "The control run"))
        without = _flat(_section(DP.render(_reception(), st, run_dir=folder, preview=PREVIEW,
                                           analysis_preview=_record(folder, st)),
                                 "The control run"))
        assert "The control run is a made-up valley." in with_fig and "f/s.png" in with_fig
        assert "half of the measured" not in with_fig
        assert "Its snow in 2031 was half of the measured." in without
        assert "The run's full record is in `DESIGN_DETAILS.md`." in without

    def test_the_details_show_both_figures_and_where_each_line_comes_from(self, folder):
        # (the details page rebuilt on 2026-11-13: the maps in "The control run",
        # and the series on the short page alone, with a table of where each
        # measurement comes from)
        _, details = _written(folder)
        sec = _section(details, "The control run")
        assert "(design_figures/control_run_maps.png)" in sec
        assert "control_run_series.png" not in details
        flat = _flat(sec)
        assert "Each map is the average of 2031 to 2033 over the model's grid. The line is " \
               "the edge of the region `valley floor`." in flat
        assert "The figure on the short page sets the control run beside these measurements." \
            in flat
        assert "| Runoff and river flow | a made-up estimate | `RUNX`, the monthly total in " \
               "mm/month, over valley floor |" in flat
        assert "| Snow at two made-up pillows | a made-up network | `SNOWX`, the monthly mean " \
               "in mm, over the 2 cells that hold the points |" in flat

    def test_the_record_is_kept_and_a_page_is_drawn_again_from_it(self, folder):
        first, _ = _written(folder)
        assert json.loads((folder / CF.RECORD_FILE).read_text()) == _rec()
        again, _ = _written(folder, record="left out")
        assert again == first

    def test_no_record_removes_an_earlier_one_and_the_page_has_no_figure(self, folder):
        _written(folder)
        page, details = _written(folder, record=None)
        assert not (folder / CF.RECORD_FILE).exists()
        assert "control_run_series.png" not in page and "control_run_maps" not in details
        assert "Its snow is about half of what was measured." in page

    def test_another_controls_record_is_not_drawn(self, folder):
        page, details = _written(folder, record=dict(_rec(), control="another"))
        assert "control_run_series.png" not in page and "control_run_maps" not in details

    def test_a_figure_that_fails_leaves_the_page_as_it_was_and_the_details_say_why(self, folder):
        rec = _rec()
        rec["maps"] = None
        for s in rec["measured"]["series"]:
            s["model"], s["measured"] = [None] * 36, [None] * 36
        page, details = _written(folder, record=rec)
        assert "control_run_series.png" not in page
        assert "Its snow is about half of what was measured." in page
        assert "control_run_maps" not in details


# ── from the model server to the page ────────────────────────────────────────
class TestFromTheServer:
    def test_a_control_the_registry_does_not_know_has_no_figures(self, monkeypatch):
        monkeypatch.setattr(AP, "_registry_entry", lambda name: {})
        assert CF.fetch("nowhere") is None and CF.fetch(None) is None

    def test_a_control_with_nothing_to_draw_has_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(AP, "_registry_entry", lambda name: {"suite_dir": str(tmp_path)})
        assert CF.fetch("bare") is None

    def test_the_servers_pairs_and_maps_reach_a_drawn_figure(self, built, monkeypatch,
                                                             tmp_path):
        units = built["suite"].units("GPP", "month")
        table = built["root"] / "seen.csv"
        table.write_text("month,value\n" + "".join(f"{y}-{m:02d},{7.5 + m}\n"
                                                   for y in (2015, 2016) for m in range(1, 13)))
        entry = dict(built["control"],
                     measured=[{"id": "growth", "field": "GPP", "region": built["region"],
                                "table": str(table), "units": units, "title": "Growth",
                                "label": "a sensor", "model_words": "Growth",
                                "measured_words": "seen by a sensor"}],
                     page_figures={"maps": [{"words": "Growth", "fields": ["GPP"]}],
                                   "outline": built["region"]})
        DM.build(entry)
        monkeypatch.setattr(AP, "_registry_entry", lambda name: entry)
        rec = CF.fetch("synth")
        assert rec["problems"] == [] and rec["control"] == "synth"
        assert CF.record_years(rec) == [2015, 2016]
        got = CF.draw(rec, tmp_path, [2016])
        assert got["problems"] == [] and (tmp_path / got["series"]).is_file() \
            and (tmp_path / got["maps"]).is_file()
        s = CF.sentences(rec, [2016])
        assert len(s) == 1 and s[0].startswith("Growth totals ")
        assert s[0].endswith(f"compared with {CF.three(sum(7.5 + m for m in range(1, 13)))} "
                             f"{units[:-len('/month')].replace('^', '')} a year seen by a "
                             f"sensor.")
        assert "^" not in s[0]

    def test_a_server_that_fails_is_a_reason_on_record_never_an_error(self, monkeypatch):
        monkeypatch.setattr(AP, "_registry_entry", lambda name: {"suite_dir": "/x"})

        def broke(entry):
            raise RuntimeError("made-up failure")
        monkeypatch.setattr(DM, "page_record", broke)
        rec = CF.fetch("valley")
        assert rec["measured"] is None and "made-up failure" in rec["problems"][0]
