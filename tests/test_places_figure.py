"""The places the numbers are reported for, as a map on the details page
(agents/places_figure, 2026-11-13).

    pytest tests/test_places_figure.py -v

Every name and number here is made up: a valley on a grid of 4 by 5 cells, its
region `floor` that holds two bands, `low` and `high`. The record has the
shape the model server gives (domain_places.page_record, interface I3). That
module is the server's, and here it is a made-up module put into sys.modules.
"""
import sys
import types

import pytest

from agents import places_figure as PF
from agents.analysis import analysis_plan as AP
from agents.analysis import figure_spec as FS

LAT = [40.0, 40.1, 40.2, 40.3]
LON = [-110.4, -110.3, -110.2, -110.1, -110.0]
FLOOR = [[0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
CLASSES = [[0, 1, 1, 0, 0], [0, 1, 2, 2, 0], [0, 0, 2, 2, 0], [0, 0, 0, 0, 0]]
REGIONS = ["floor", "floor, low", "floor, high"]


def _rec(**over):
    rec = {"control": "valley", "regions": list(REGIONS), "lat": list(LAT), "lon": list(LON),
           "filled": [{"label": "floor, low", "cells": 3, "area_km2": 1234.5},
                      {"label": "floor, high", "cells": 4, "area_km2": 22.49}],
           "classes": [list(r) for r in CLASSES],
           "outlines": [{"label": "floor", "cells": 7, "area_km2": 1257.0,
                         "mask": [list(r) for r in FLOOR]}],
           "problems": []}
    rec.update(over)
    return rec


# ── the record: asked of the server, kept beside the design ──────────────────
class TestFetch:
    @pytest.fixture
    def server(self, monkeypatch):
        calls = []

        def page_record(entry, regions):
            calls.append((entry, regions))
            return _rec(regions=list(regions))
        mod = types.SimpleNamespace(page_record=page_record)
        monkeypatch.setitem(sys.modules, "domain_places", mod)
        monkeypatch.setattr(AP, "_registry_entry", lambda name: (
            {"suite_dir": "/stores/valley"} if name == "valley" else {}))
        return mod, calls

    def test_it_asks_the_server_with_the_entry_and_the_regions(self, server):
        _, calls = server
        rec = PF.fetch("valley", REGIONS)
        assert rec["control"] == "valley" and rec["regions"] == REGIONS
        assert calls == [({"suite_dir": "/stores/valley", "name": "valley"}, REGIONS)]

    @pytest.mark.parametrize("name,regions", [(None, REGIONS), ("", REGIONS), ("valley", []),
                                              ("valley", None), ("other", REGIONS)])
    def test_nothing_to_ask_gives_none(self, server, name, regions):
        assert PF.fetch(name, regions) is None

    def test_a_server_without_the_module_gives_none(self, monkeypatch):
        monkeypatch.setattr(AP, "_registry_entry", lambda name: {"suite_dir": "/stores/valley"})
        monkeypatch.setitem(sys.modules, "domain_places", None)       # import fails
        assert PF.fetch("valley", REGIONS) is None

    def test_a_server_that_fails_or_answers_oddly_gives_none(self, server):
        mod, _ = server
        mod.page_record = lambda entry, regions: (_ for _ in ()).throw(OSError("no store"))
        assert PF.fetch("valley", REGIONS) is None
        mod.page_record = lambda entry, regions: ["not", "a", "record"]
        assert PF.fetch("valley", REGIONS) is None
        mod.page_record = lambda entry, regions: None             # a control with no store
        assert PF.fetch("valley", REGIONS) is None


class TestSaveAndLoad:
    def test_a_record_comes_back_as_it_was_saved(self, tmp_path):
        PF.save(tmp_path, _rec())
        assert PF.load(tmp_path) == _rec()
        assert (tmp_path / PF.RECORD_FILE).is_file()
        assert PF.RECORD_FILE == "01_inputs/places.json"

    def test_none_removes_an_earlier_record(self, tmp_path):
        PF.save(tmp_path, _rec())
        PF.save(tmp_path, None)
        assert not (tmp_path / PF.RECORD_FILE).exists() and PF.load(tmp_path) is None
        PF.save(tmp_path, None)                                    # nothing to remove

    @pytest.mark.parametrize("text", ["{broken", "[1, 2]", "", "null"])
    def test_a_broken_file_loads_as_none(self, tmp_path, text):
        p = tmp_path / PF.RECORD_FILE
        p.parent.mkdir(parents=True)
        p.write_text(text)
        assert PF.load(tmp_path) is None

    def test_a_missing_file_loads_as_none(self, tmp_path):
        assert PF.load(tmp_path) is None

    def test_the_saved_record_is_small(self, tmp_path):
        big = _rec(lat=[40.0 + i / 100 for i in range(96)],
                   lon=[-110.0 + i / 100 for i in range(144)],
                   classes=[[i % 5 for i in range(144)] for _ in range(96)],
                   outlines=[{"label": "floor", "area_km2": 1.0,
                              "mask": [[1] * 144 for _ in range(96)]}])
        PF.save(tmp_path, big)
        assert (tmp_path / PF.RECORD_FILE).stat().st_size < 100_000


class TestOfDesign:
    def test_the_same_control_and_the_same_set_of_regions(self):
        assert PF.of_design(_rec(), "valley", list(reversed(REGIONS)))

    @pytest.mark.parametrize("control,regions", [("other", REGIONS), ("valley", REGIONS[:2]),
                                                 ("valley", REGIONS + ["ridge"]),
                                                 ("valley", []), (None, REGIONS)])
    def test_another_design_is_not_drawn(self, control, regions):
        assert not PF.of_design(_rec(), control, regions)

    @pytest.mark.parametrize("rec", [None, [], {"control": "valley"},
                                     {"control": "valley", "regions": "floor"}])
    def test_a_record_of_another_shape_is_not_drawn(self, rec):
        assert not PF.of_design(rec, "valley", REGIONS)


# ── the legend's words ───────────────────────────────────────────────────────
class TestTheLegend:
    def test_labels_that_begin_with_the_outlines_are_said_after_it(self):
        words = PF.legend(_rec())
        assert words["title"] == "floor (1,257 km2)"
        assert [e["label"] for e in words["entries"]] == ["low", "high"]
        assert [e["area_km2"] for e in words["entries"]] == [1234.5, 22.49]
        assert [x["label"] for x in words["lines"]] == [None]       # the title names it

    def test_otherwise_each_label_whole_and_the_outline_beside_its_line(self):
        rec = _rec(filled=[{"label": "floor, low", "area_km2": 3.0},
                           {"label": "ridge", "area_km2": 4.0}])
        words = PF.legend(rec)
        assert words["title"] is None
        assert [e["label"] for e in words["entries"]] == ["floor, low", "ridge"]
        assert [x["label"] for x in words["lines"]] == ["floor"]

    def test_the_longest_outline_that_begins_every_label(self):
        rec = _rec(outlines=[{"label": "floor", "area_km2": 9.0, "mask": FLOOR},
                             {"label": "floor, wet", "area_km2": 8.0, "mask": FLOOR}],
                   filled=[{"label": "floor, wet, low", "area_km2": 1.0},
                           {"label": "floor, wet, high", "area_km2": 2.0}])
        words = PF.legend(rec)
        assert words["title"] == "floor, wet (8 km2)"
        assert [e["label"] for e in words["entries"]] == ["low", "high"]
        assert [x["label"] for x in words["lines"]] == ["floor", None]


# ── the drawing ──────────────────────────────────────────────────────────────
def _drawn_axes(rec):
    """The panel as the figure tool draws it, to read its legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec, refs, ll = PF.figure(rec)
    fig, ax = plt.subplots()
    FS._draw_classes(plt, ax, refs["C1"], spec["panels"][0], ll, "panel 1")
    return plt, fig, ax


class TestDraw:
    def test_several_classes_and_an_outline(self, tmp_path):
        drawn = PF.draw(_rec(), tmp_path)
        assert drawn == {"png": "design_figures/places_reported.png", "problems": []}
        assert (tmp_path / drawn["png"]).stat().st_size > 1000
        plt, fig, ax = _drawn_axes(_rec())
        leg = ax.get_legend()
        assert leg.get_title().get_text() == "floor (1,257 km2)"
        assert [t.get_text() for t in leg.get_texts()] == ["low (1,235 km2)", "high (22 km2)",
                                                           PF.OTHER]
        assert any(type(c).__name__ != "QuadMesh" for c in ax.collections)   # the edge
        plt.close(fig)

    def test_one_class(self, tmp_path):
        rec = _rec(filled=[{"label": "floor", "area_km2": 7.0}], outlines=[],
                   classes=[[1 if v else 0 for v in r] for r in FLOOR], regions=["floor"])
        drawn = PF.draw(rec, tmp_path)
        assert drawn["png"] and not drawn["problems"]
        plt, fig, ax = _drawn_axes(rec)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["floor (7 km2)", PF.OTHER]
        assert ax.get_legend().get_title().get_text() == ""
        plt.close(fig)

    def test_an_outline_alone(self, tmp_path):
        rec = _rec(filled=[], classes=[[0] * 5 for _ in range(4)])
        drawn = PF.draw(rec, tmp_path)
        assert drawn["png"] and not drawn["problems"]
        plt, fig, ax = _drawn_axes(rec)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["the model grid",
                                                                       "floor (1,257 km2)"]
        plt.close(fig)

    @pytest.mark.parametrize("lat,lon,classes,mask", [
        ([40.0], LON, [[0, 1, 1, 2, 0]], [[0, 1, 1, 1, 0]]),               # one row
        (LAT, [-110.4], [[0], [1], [2], [0]], [[0], [1], [1], [0]])])       # one column
    def test_a_grid_one_cell_wide_draws_the_classes_and_names_the_edge(self, tmp_path, lat,
                                                                       lon, classes, mask):
        """A contour needs two cells a side. The classes are drawn alone and
        the edge left out is named, never the whole map lost."""
        rec = _rec(lat=lat, lon=lon, classes=classes,
                   outlines=[{"label": "floor", "cells": 3, "area_km2": 1257.0, "mask": mask}])
        drawn = PF.draw(rec, tmp_path)
        assert drawn["png"] == PF.PNG and (tmp_path / PF.PNG).stat().st_size > 1000
        assert drawn["problems"] == ["the edge of 'floor' is left out of the map of the places, "
                                     "because the grid is one cell wide and an edge needs two "
                                     "cells a side"]
        plt, fig, ax = _drawn_axes(rec)
        assert [type(c).__name__ for c in ax.collections] == ["QuadMesh"]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["low (1,235 km2)",
                                                                       "high (22 km2)", PF.OTHER]
        plt.close(fig)

    def test_a_grid_one_cell_wide_with_no_edge_names_nothing(self, tmp_path):
        rec = _rec(lat=[40.0], classes=[[0, 1, 1, 2, 0]], outlines=[])
        assert PF.draw(rec, tmp_path) == {"png": PF.PNG, "problems": []}

    def test_nothing_to_draw_gives_no_figure(self, tmp_path):
        drawn = PF.draw(_rec(filled=[], outlines=[]), tmp_path)
        assert drawn["png"] is None
        assert drawn["problems"] == ["the map of the places was not drawn: the record holds no "
                                     "place to draw"]
        assert not (tmp_path / PF.PNG).exists()

    @pytest.mark.parametrize("over", [{"classes": [[0, 1]]}, {"lat": []}, {"lon": "x"},
                                      {"classes": [[0, 1, 1, 0, 9]] * 4},
                                      {"outlines": [{"label": "floor", "mask": [[1]]}]},
                                      {"classes": "not a grid"}])
    def test_a_record_that_does_not_hold_together_is_a_reason_not_a_crash(self, tmp_path, over):
        drawn = PF.draw(_rec(**over), tmp_path)
        assert drawn["png"] is None
        assert len(drawn["problems"]) == 1
        assert drawn["problems"][0].startswith("the map of the places was not drawn: ")

    @pytest.mark.parametrize("rec", [None, [], "x", {"filled": "x", "outlines": 3}])
    def test_draw_never_raises(self, tmp_path, rec):
        drawn = PF.draw(rec, tmp_path)
        assert drawn["png"] is None

    def test_a_longitude_past_180_is_drawn_west(self):
        _, _, ll = PF.figure(_rec(lon=[x + 360.0 for x in LON]))
        assert ll["lon"] == pytest.approx(LON)


class TestPageLines:
    def test_the_image_line_when_drawn(self, tmp_path):
        rec = _rec()
        drawn = PF.draw(rec, tmp_path)
        assert PF.page_lines(rec, drawn) == [
            "![The places the numbers are reported for](design_figures/places_reported.png)", ""]

    @pytest.mark.parametrize("drawn", [None, {}, {"png": None, "problems": ["x"]}, "x"])
    def test_nothing_when_nothing_was_drawn(self, drawn):
        assert PF.page_lines(_rec(), drawn) == []

    def test_nothing_without_a_record(self):
        assert PF.page_lines(None, {"png": PF.PNG}) == []
