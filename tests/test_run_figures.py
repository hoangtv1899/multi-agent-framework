"""What a new run is given, beside the control run, on the design page
(agents/run_figures, 2026-11-13).

    pytest tests/test_run_figures.py -v

Every number here is made up. The record has the shape the model server gives
(domain_given.page_record, the contract of 2026-11-13): a made-up valley, a
run that cuts summer rain over a region `floor`, and a run that switches the
fire off. The module domain_given is the server's, and here it is a made-up
module put into sys.modules.
"""
import json
import sys
import types

import pytest

from agents import run_figures as RF
from agents.analysis import analysis_plan as AP
from agents.analysis import figure_spec as FS

LABELS = [f"{y}-{m:02d}" for y in (2033, 2034) for m in range(1, 13)]
CHANGED = ["2033-06", "2033-07", "2033-08", "2034-06", "2034-07", "2034-08"]
CONTROL = [float(10 + i) for i in range(24)]
RUN = [c * 0.7 if lab in CHANGED else c for c, lab in zip(CONTROL, LABELS)]
RAIN = {"knob": "Rain and snow", "change": "multiplied by 0.7",
        "when": "June to August of 2033 and 2034", "where": "the region `floor`"}
WARM = {"knob": "Air temperature", "change": "2 K warmer",
        "when": "June to August of 2033 and 2034", "where": "the region `floor`"}
FIRE = {"knob": "Fire", "change": "switched off", "when": "every year of the run",
        "where": "everywhere"}
MASK = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]


def _dry(**over):
    run = {"name": "dry_summer", "rows": [dict(RAIN)],
           "panels": [{"title": "Rain and snow over the cells the change reaches",
                       "units": "mm/month", "step": "month", "labels": list(LABELS),
                       "control": list(CONTROL), "run": list(RUN), "changed": list(CHANGED)}],
           "map": {"title": "Change in rain and snow, mean of the changed months",
                   "units": "mm/month", "lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0, 13.0],
                   "values": [[None] * 4, [None, -5.0, -6.0, None], [None, -4.0, -7.0, None]],
                   "outline": MASK},
           "problems": []}
    run.update(over)
    return run


def _nofire(**over):
    run = {"name": "no_fire", "rows": [dict(FIRE)],
           "panels": [{"title": "Area the control run's own fire burns", "units": "km2",
                       "step": "year", "labels": ["2033", "2034"], "control": [50.0, None],
                       "run": [0.0, 0.0], "changed": ["2033", "2034"]}],
           "map": None, "problems": []}
    run.update(over)
    return run


def _rec(*runs):
    return {"control": "valley", "years": [2033, 2034], "problems": [],
            "runs": list(runs) or [_dry()]}


# ── the record, asked of the model server ────────────────────────────────────
STRATEGY = {"domain_experiment": {"control": {"name": "valley", "reuse": True},
                                  "runs": [{"name": "dry_summer", "changes": [
                                      {"id": "weather_scale",
                                       "params": {"precip_factor": 0.7}}]}],
                                  "years": [2033, 2034]}}


class TestFromTheServer:

    @pytest.fixture
    def server(self, monkeypatch):
        """A made-up domain_given, a control the registry knows, and its facts."""
        import domain_controls as DCT
        seen = {}

        def page_record(experiment, control, inputs_dir=None):
            seen.update(experiment=experiment, control=control, inputs_dir=inputs_dir)
            return _rec()
        monkeypatch.setitem(sys.modules, "domain_given",
                            types.SimpleNamespace(page_record=page_record))
        monkeypatch.setattr(AP, "_registry_entry",
                            lambda name: {"run_dir": "/made/up"} if name == "valley" else {})
        monkeypatch.setattr(DCT, "facts", lambda name, override=None: {"start_year": 2031})
        return seen

    def test_it_asks_with_the_design_the_control_and_the_inputs_folder(self, server,
                                                                       tmp_path):
        assert RF.fetch(STRATEGY, tmp_path) == _rec()
        de = STRATEGY["domain_experiment"]
        assert server["experiment"] == {"control": de["control"], "runs": de["runs"],
                                        "years": [2033, 2034]}
        assert server["control"] == {"run_dir": "/made/up", "name": "valley",
                                     "facts": {"start_year": 2031}}
        assert server["inputs_dir"] == tmp_path / "01_inputs"

    def test_an_unknown_control_gives_nothing(self, server, tmp_path):
        st = json.loads(json.dumps(STRATEGY))
        st["domain_experiment"]["control"]["name"] = "nowhere"
        assert RF.fetch(st, tmp_path) is None and not server

    def test_no_whole_domain_design_or_no_run_gives_nothing(self, server, tmp_path):
        assert RF.fetch({"archetype": "site"}, tmp_path) is None
        st = json.loads(json.dumps(STRATEGY))
        st["domain_experiment"]["runs"] = []
        assert RF.fetch(st, tmp_path) is None and not server

    def test_a_server_without_the_tool_gives_nothing(self, server, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "domain_given", None)     # the import fails
        assert RF.fetch(STRATEGY, tmp_path) is None

    def test_an_error_of_the_server_gives_nothing_and_raises_nothing(self, server,
                                                                     monkeypatch, tmp_path):
        def broke(*a, **k):
            raise RuntimeError("a made-up failure")
        monkeypatch.setitem(sys.modules, "domain_given",
                            types.SimpleNamespace(page_record=broke))
        assert RF.fetch(STRATEGY, tmp_path) is None
        monkeypatch.setitem(sys.modules, "domain_given",
                            types.SimpleNamespace(page_record=lambda *a, **k: ["not", "a", 1]))
        assert RF.fetch(STRATEGY, tmp_path) is None

    def test_facts_that_cannot_be_read_give_nothing(self, server, monkeypatch, tmp_path):
        import domain_controls as DCT

        def no_facts(name, override=None):
            raise DCT.ControlError("a made-up registry problem")
        monkeypatch.setattr(DCT, "facts", no_facts)
        assert RF.fetch(STRATEGY, tmp_path) is None


class TestTheRecordBesideTheDesign:

    def test_it_is_kept_and_read_back(self, tmp_path):
        RF.save(tmp_path, _rec())
        assert json.loads((tmp_path / RF.RECORD_FILE).read_text()) == _rec()
        assert RF.load(tmp_path) == _rec()

    def test_none_removes_an_earlier_one(self, tmp_path):
        RF.save(tmp_path, _rec())
        RF.save(tmp_path, None)
        assert not (tmp_path / RF.RECORD_FILE).exists() and RF.load(tmp_path) is None
        RF.save(tmp_path, None)                                    # nothing to remove

    def test_a_record_that_cannot_be_read_is_none(self, tmp_path):
        (tmp_path / "01_inputs").mkdir()
        (tmp_path / RF.RECORD_FILE).write_text("{not json")
        assert RF.load(tmp_path) is None
        (tmp_path / RF.RECORD_FILE).write_text("[1, 2]")
        assert RF.load(tmp_path) is None

    def test_it_is_of_this_design_by_its_control_and_its_years(self):
        de = {"control": {"name": "valley"}, "years": [2034, 2033],
              "runs": [{"name": "dry_summer", "changes": []}]}
        assert RF.of_design(_rec(), de)
        assert not RF.of_design(_rec(), dict(de, control={"name": "another"}))
        assert not RF.of_design(_rec(), dict(de, years=[2033, 2034, 2035]))
        assert not RF.of_design(None, de) and not RF.of_design(_rec(), None)

    def test_it_is_of_this_design_by_its_runs_and_their_changes_too(self):
        # a revised design keeps its control, its years and often a run's
        # name: the record an earlier design left is still not this design's
        change = {"id": "weather_scale", "params": {"precip_factor": 0.7}}
        de = {"control": {"name": "valley"}, "years": [2033, 2034],
              "runs": [{"name": "dry_summer", "changes": [dict(change)]}]}
        made = _rec(_dry(changes=[dict(change)]))
        assert RF.of_design(made, de)
        drier = {"id": "weather_scale", "params": {"precip_factor": 0.5}}
        assert not RF.of_design(made, dict(de, runs=[{"name": "dry_summer",
                                                      "changes": [drier]}]))
        assert not RF.of_design(made, dict(de, runs=[{"name": "another",
                                                      "changes": [dict(change)]}]))
        assert not RF.of_design(made, dict(de, runs=[]))
        assert not RF.of_design(_rec(_dry(changes=[dict(change)]), _nofire(changes=[])), de)
        # a run of the design that the record does not hold is no mismatch
        assert RF.of_design(made, dict(de, runs=de["runs"] + [{"name": "second",
                                                               "changes": []}]))
        # the order of a change's keys is no difference
        flipped = {"params": {"precip_factor": 0.7}, "id": "weather_scale"}
        assert RF.of_design(made, dict(de, runs=[{"name": "dry_summer",
                                                  "changes": [flipped]}]))
        # a record that does not say its changes is held to its run names
        assert RF.of_design(_rec(_dry()), de)

    def test_a_record_of_the_wrong_shape_has_no_run(self):
        for bad in ({"runs": 3}, {"runs": {"name": "x"}}, {"runs": [1, None]}, None):
            assert RF.entry(bad, "x") is None and RF.rows(bad, "x") == []
            assert RF.table_lines(bad, ["x"], False) == []


# ── the figures ──────────────────────────────────────────────────────────────
class TestTheFigures:

    def test_the_series_figure_puts_the_two_runs_in_a_panel_and_shades_the_changes(self):
        spec, refs = RF.series_figure(_dry())
        assert spec["layout"] == "stack" and spec["id"] == "new_run_dry_summer_series"
        assert spec["panels"] == [{"kind": "series", "outputs": ["G1.control", "G1.run"],
                                   "shade_labels": CHANGED}]
        assert refs["G1.control"]["out"]["label"] == "control run"
        assert refs["G1.run"]["out"]["label"] == "dry_summer"
        assert refs["G1.control"]["value"] == {"labels": LABELS, "values": CONTROL,
                                               "step": "month", "units": "mm/month"}
        assert refs["G1.run"]["value"]["values"] == RUN
        assert {r["out"]["title"] for r in refs.values()} == \
            {"Rain and snow over the cells the change reaches"}

    def test_one_panel_per_panel_of_the_record(self):
        two = _dry(panels=_dry()["panels"] + [dict(_dry()["panels"][0], title="Air",
                                                   units="degrees C", changed=[])])
        spec, refs = RF.series_figure(two)
        assert [p["outputs"] for p in spec["panels"]] == [["G1.control", "G1.run"],
                                                          ["G2.control", "G2.run"]]
        assert "shade_labels" not in spec["panels"][1]
        assert refs["G2.run"]["value"]["units"] == "degrees C"

    def test_the_map_is_a_change_on_a_scale_centred_on_zero_with_its_edge(self):
        spec, refs, ll = RF.map_figure(_dry())
        assert spec["panels"] == [{"kind": "map", "output": "D1"}]
        assert refs["D1"]["kind"] == "difference"
        assert refs["D1"]["value"]["difference"] == _dry()["map"]["values"]
        assert refs["D1"]["out"]["title"] == "Change in rain and snow, mean of the changed months"
        assert ll == {"lat": [1.0, 2.0, 3.0], "lon": [10.0, 11.0, 12.0, 13.0], "outline": MASK}

    def test_both_are_drawn_into_the_study_folder(self, tmp_path):
        got = RF.draw(_rec(_dry(), _nofire()), tmp_path)
        assert got == {"runs": {
            "dry_summer": {"series": "design_figures/new_run_dry_summer_series.png",
                           "map": "design_figures/new_run_dry_summer_map.png"},
            "no_fire": {"series": "design_figures/new_run_no_fire_series.png", "map": None}},
            "problems": []}
        for f in ("new_run_dry_summer_series.png", "new_run_dry_summer_map.png",
                  "new_run_no_fire_series.png"):
            assert (tmp_path / "design_figures" / f).stat().st_size > 5000

    def test_a_run_with_rows_and_no_panel_draws_nothing(self, tmp_path):
        got = RF.draw(_rec(_dry(panels=[], map=None)), tmp_path)
        assert got == {"runs": {"dry_summer": {"series": None, "map": None}}, "problems": []}
        assert not (tmp_path / "design_figures").exists()

    def test_a_figure_that_fails_is_left_out_and_named(self, tmp_path):
        empty = dict(_dry()["panels"][0], control=[None] * 24, run=[None] * 24)
        got = RF.draw(_rec(_dry(panels=[empty])), tmp_path)
        assert got["runs"]["dry_summer"]["series"] is None
        assert got["runs"]["dry_summer"]["map"] is not None        # the map still drawn
        assert len(got["problems"]) == 1
        assert got["problems"][0].startswith("the figure of `dry_summer` was not drawn: ")
        assert not (tmp_path / "design_figures" / RF.series_png("dry_summer")).exists()

    def test_a_map_that_fails_is_left_out_and_named(self, tmp_path):
        bad = dict(_dry()["map"], values=[[None] * 4] * 3)
        got = RF.draw(_rec(_dry(map=bad)), tmp_path)
        assert got["runs"]["dry_summer"] == {
            "series": "design_figures/new_run_dry_summer_series.png", "map": None}
        assert got["problems"][0].startswith("the map of `dry_summer` was not drawn: ")

    def test_nothing_raises_whatever_the_record(self, tmp_path):
        assert RF.draw(None, tmp_path) == {"runs": {}, "problems": []}
        assert RF.draw({"runs": 3}, tmp_path) == {"runs": {}, "problems": []}
        got = RF.draw(_rec(_dry(panels=[{"labels": 5}])), tmp_path)
        assert got["runs"]["dry_summer"]["series"] is None and got["problems"]


# ── what the figure tool draws for it ────────────────────────────────────────
class TestTheTool:

    def _axes(self):
        plt = FS._plt()
        fig, ax = plt.subplots()
        return plt, fig, ax

    def test_each_unbroken_run_of_labels_is_one_band(self):
        assert FS._label_runs(LABELS, CHANGED) == [(5, 7), (17, 19)]
        assert FS._label_runs(["2033", "2034", "2035"], ["2033", "2035"]) == [(0, 0), (2, 2)]
        assert FS._label_runs(LABELS, []) == [] and FS._label_runs(LABELS, None) == []

    def test_the_bands_are_drawn_once_per_panel(self):
        plt, fig, ax = self._axes()
        spec, refs = RF.series_figure(_dry())
        n = FS._draw_series(ax, [refs["G1.control"], refs["G1.run"]], spec["panels"][0], (),
                            "panel 1")
        spans = [(round(p.get_x() + 0.5), round(p.get_x() + p.get_width() - 0.5))
                 for p in ax.patches]
        assert sorted(spans) == [(5, 7), (17, 19)]               # two bands, not four
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["control run",
                                                                       "dry_summer"]
        assert n == 48 and ax.get_ylabel() == "mm/month"
        plt.close(fig)

    def test_year_labels_are_shaded_alike(self):
        plt, fig, ax = self._axes()
        spec, refs = RF.series_figure(_nofire())
        FS._draw_series(ax, [refs["G1.control"], refs["G1.run"]], spec["panels"][0], (),
                        "panel 1")
        assert len(ax.patches) == 1
        assert [t.get_text() for t in ax.get_xticklabels()] == ["2033", "2034"]
        plt.close(fig)

    def test_a_yearly_line_shows_its_points_and_a_monthly_one_is_as_before(self):
        plt, fig, ax = self._axes()
        one = _nofire(panels=[dict(_nofire()["panels"][0], labels=["2033"], control=[50.0],
                                   run=[0.0], changed=["2033"])])
        spec, refs = RF.series_figure(one)
        FS._draw_series(ax, [refs["G1.control"], refs["G1.run"]], spec["panels"][0], (),
                        "panel 1")
        assert [ln.get_marker() for ln in ax.get_lines()] == ["o", "o"]
        plt.close(fig)
        plt, fig, ax = self._axes()
        spec, refs = RF.series_figure(_dry())
        FS._draw_series(ax, [refs["G1.control"]], spec["panels"][0], (), "panel 1")
        assert ax.get_lines()[0].get_marker() in ("None", None, "")
        plt.close(fig)

    def test_shade_labels_is_not_in_the_list_the_analyzer_is_shown(self):
        assert "shade_labels" not in " ".join(FS.panel_lines(()))
        assert "shade_labels" not in [o.name for o in FS.PANEL_KINDS["series"].options]
        plan = {"analyses": [{"id": "P1", "outputs": [
            {"id": "O1", "kind": "series", "field": "X", "region": "r", "step": "month"}]}]}
        probs = FS.check_figure({"id": "f", "plan_items": ["P1"], "panels": [
            {"kind": "series", "outputs": ["P1.O1"], "shade_labels": ["2033-06"]}]}, plan, ())
        assert probs == ["f, panel 1: series has no option 'shade_labels' (its options: "
                         "years, normal)"]

    def test_the_docstring_lists_it_with_the_other_page_options(self):
        assert '"shade_labels"' in FS.__doc__


# ── the page's lines ─────────────────────────────────────────────────────────
class TestThePagesLines:

    def test_the_table_of_one_run(self):
        assert RF.table_lines(_rec(), ["dry_summer"], False) == [
            "| Knob | Change | When | Where |", "|---|---|---|---|",
            "| Rain and snow | multiplied by 0.7 | June to August of 2033 and 2034 | the "
            "region `floor` |", ""]

    def test_two_knobs_are_two_rows(self):
        lines = RF.table_lines(_rec(_dry(rows=[RAIN, WARM])), ["dry_summer"], False)
        assert lines[2:4] == [
            "| Rain and snow | multiplied by 0.7 | June to August of 2033 and 2034 | the "
            "region `floor` |",
            "| Air temperature | 2 K warmer | June to August of 2033 and 2034 | the region "
            "`floor` |"]

    def test_several_runs_get_a_first_column(self):
        lines = RF.table_lines(_rec(_dry(), _nofire()), ["dry_summer", "no_fire"], True)
        assert lines[:2] == ["| Run | Knob | Change | When | Where |", "|---|---|---|---|---|"]
        assert lines[3] == "| `no_fire` | Fire | switched off | every year of the run | " \
                           "everywhere |"

    def test_a_bar_in_a_cell_does_not_break_the_table(self):
        lines = RF.table_lines(_rec(_dry(rows=[dict(RAIN, where="a | b")])), ["dry_summer"],
                               False)
        assert lines[2].endswith("| a / b |")

    def test_a_run_drawn_whole_is_shown_and_said(self, tmp_path):
        drawn = RF.draw(_rec(), tmp_path)
        assert RF.figure_lines(_rec(), drawn, ["dry_summer"], False) == [
            "![What the new run is given](design_figures/new_run_dry_summer_series.png)", "",
            "![Where the change applies](design_figures/new_run_dry_summer_map.png)", "",
            "The figure shows what the new run is given beside the control run, month by "
            "month. The shaded months are the changed ones. The map shows the mean change in "
            "those months.", ""]

    def test_a_yearly_panel_is_said_year_by_year_and_no_map_is_not_said(self):
        drawn = {"runs": {"no_fire": {"series": "f/s.png", "map": None}}}
        said = " ".join(RF.figure_lines(_rec(_nofire()), drawn, ["no_fire"], False))
        assert "beside the control run, year by year. The shaded years are the changed " \
               "ones." in said
        assert "map" not in said.lower()

    def test_several_runs_each_introduced_by_its_name(self):
        drawn = {"runs": {"dry_summer": {"series": "f/a.png", "map": None},
                          "no_fire": {"series": "f/b.png", "map": None}}}
        lines = RF.figure_lines(_rec(_dry(), _nofire()), drawn, ["dry_summer", "no_fire"], True)
        assert lines[0] == "**`dry_summer`**" and "**`no_fire`**" in lines
        assert lines.index("**`dry_summer`**") < lines.index("![What dry_summer is given](f/a.png)") \
            < lines.index("**`no_fire`**") < lines.index("![What no_fire is given](f/b.png)")
        assert "The figure shows what `no_fire` is given beside the control run, year by " \
               "year." in " ".join(lines)

    def test_a_figure_not_drawn_has_no_image_and_no_sentence(self):
        drawn = {"runs": {"dry_summer": {"series": None, "map": None}}}
        assert RF.figure_lines(_rec(), drawn, ["dry_summer"], False) == []
        assert RF.figure_lines(_rec(), None, ["dry_summer"], False) == []

    def test_a_map_without_its_series_says_which_months(self):
        drawn = {"runs": {"dry_summer": {"series": None, "map": "f/m.png"}}}
        assert RF.figure_lines(_rec(), drawn, ["dry_summer"], False)[-2] == \
            "The map shows the mean change in the months the new run changes."

    def test_panels_with_no_change_shade_nothing_and_say_nothing_of_it(self):
        run = _dry(panels=[dict(_dry()["panels"][0], changed=[])], map=None)
        drawn = {"runs": {"dry_summer": {"series": "f/s.png", "map": None}}}
        said = " ".join(RF.figure_lines(_rec(run), drawn, ["dry_summer"], False))
        assert "shaded" not in said and "month by month." in said

    def test_the_details_name_what_was_not_given_or_not_drawn(self):
        rec = _rec(_dry(problems=["the weather file is missing"]))
        rec["problems"] = ["a made-up problem of the record"]
        drawn = {"runs": {"dry_summer": {"series": None, "map": None}},
                 "problems": ["the figure of `dry_summer` was not drawn: a made-up reason"]}
        lines = RF.details_lines(rec, drawn, ["dry_summer"])
        assert lines[0] == "## What the new runs are given"
        assert "- Not shown: a made-up problem of the record" in lines
        assert "- Not shown: `dry_summer`: the weather file is missing" in lines
        assert "- Not shown: the figure of `dry_summer` was not drawn: a made-up reason" in lines
        assert RF.details_lines(None, None, ["dry_summer"]) == []


# ── from the model server's own module to the drawn page ─────────────────────
class TestWithTheServersModule:
    """The page was built against a made-up `domain_given`. This holds the
    two halves together: the real module's record of a made-up control, the
    figures drawn from it, and the lines the page prints (2026-11-13)."""

    def test_the_real_record_is_drawn_and_listed(self, tmp_path, monkeypatch):
        import domain_given as DG
        import test_domain_given as TG
        from agents.analysis import analysis_plan as AP

        root = tmp_path / "server"
        run_dir = TG._history(root / "control_run")
        case = TG._control_case(root, domain=TG._domain(root / "domain.lnd.valley.nc"),
                                name="ctl")
        regions = {"floor": {"field": str(TG._field_region(root / "floor.nc",
                                                           TG._mask(TG.BOX))),
                             "var": "sel", "op": ">", "value": 0.5, "what": "a made-up floor"}}
        entry = {"run_dir": str(run_dir), "case_dir": str(case), "regions": regions}
        change = {"id": "weather_scale",
                  "params": {"precip_factor": 0.7, "months": [6, 7, 8], "region": "floor"}}
        study = tmp_path / "study"
        TG._build(case, study / "01_inputs", "dry_summer", [change], TG.YEARS, regions)
        strategy = {"domain_experiment": {"control": {"name": "valley", "reuse": True},
                                          "runs": [{"name": "dry_summer", "changes": [change]}],
                                          "years": list(TG.YEARS)}}
        monkeypatch.setattr(AP, "_registry_entry", lambda name: dict(entry))
        import domain_controls as DCT
        monkeypatch.setattr(DCT, "facts", lambda name, override=None: dict(TG.FACTS))

        rec = RF.fetch(strategy, study)
        assert rec is not None and rec["problems"] == [] and rec["control"] == "valley"
        (run,) = rec["runs"]
        assert run["problems"] == [] and run["changes"] == [change]
        assert RF.of_design(rec, strategy["domain_experiment"])
        assert [r["knob"] for r in run["rows"]] == ["Rain and snow"]
        assert run["rows"][0]["where"] == "the region `floor`"
        assert len(run["panels"]) == 1 and run["map"] is not None

        drawn = RF.draw(rec, study)
        assert drawn["problems"] == []
        files = drawn["runs"]["dry_summer"]
        assert (study / files["series"]).stat().st_size > 5000
        assert (study / files["map"]).stat().st_size > 5000
        table = "\n".join(RF.table_lines(rec, ["dry_summer"], False))
        assert "| Rain and snow | multiplied by 0.7 |" in table
        assert "the region `floor` |" in table

        # the same record, read back for a design that has since changed
        RF.save(study, rec)
        other = {"control": {"name": "valley"}, "years": list(TG.YEARS),
                 "runs": [{"name": "dry_summer", "changes": [
                     {"id": "weather_scale", "params": {"precip_factor": 0.5}}]}]}
        assert not RF.of_design(RF.load(study), other)
