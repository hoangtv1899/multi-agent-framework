"""A map of classes in the shared figure tool, an option for the design pages
(agents/analysis/figure_spec, 2026-11-13; drawn for agents/places_figure).

    pytest tests/test_figure_classes.py -v

Every number here is made up: a grid of 3 by 4 cells and its classes. The
option is not in the closed list the Analyzer's step 2b is shown, and a figure
the model writes is checked as before.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

from agents.analysis import figure_spec as FS  # noqa: E402

LL = {"lat": [45.0, 45.2, 45.4], "lon": [-100.6, -100.4, -100.2, -100.0]}
GRID = [[0, 1, 1, 0], [0, 2, 2, 3], [0, 0, 3, 3]]
ENTRIES = [{"label": "low", "area_km2": 10.4}, {"label": "middle", "area_km2": 1499.5},
           {"label": "high", "area_km2": None}]


def _ref(value=None):
    return {"ref": "C1", "kind": "map", "out": {"field": "places", "title": "places"},
            "value": GRID if value is None else value, "made": True, "error": None,
            "outcome": None, "why": None, "line": None}


def _panel(**over):
    c = {"entries": ENTRIES, "title": "valley (2,000 km2)", "other": "the rest of the grid",
         "lines": []}
    c.update(over)
    return {"kind": "map", "output": "C1", "classes": c}


def _draw(value=None, ll=None, **over):
    fig, ax = plt.subplots()
    n = FS._draw_classes(plt, ax, _ref(value), _panel(**over), ll or LL, "panel 1")
    return fig, ax, n


def _luminance(hex_colour):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in to_rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ── the drawing ──────────────────────────────────────────────────────────────
class TestTheMapOfClasses:
    def test_each_class_its_colour_and_its_legend_entry_with_its_area(self):
        fig, ax, n = _draw()
        leg = ax.get_legend()
        assert [t.get_text() for t in leg.get_texts()] == [
            "low (10 km2)", "middle (1,500 km2)", "high", "the rest of the grid"]
        assert leg.get_title().get_text() == "valley (2,000 km2)"
        faces = [to_hex(h.get_facecolor()) for h in leg.legend_handles]
        assert faces == FS.class_colours(3) + [FS.CLASS_OTHER]
        assert n == 7                                   # the cells in a class
        plt.close(fig)

    def test_the_mesh_takes_class_0_grey_and_class_k_the_kth_colour(self):
        fig, ax, _ = _draw()
        mesh = ax.collections[0]
        cmap, norm = mesh.get_cmap(), mesh.norm
        assert to_hex(cmap(norm(0))) == FS.CLASS_OTHER
        for k, col in enumerate(FS.class_colours(3), 1):
            assert to_hex(cmap(norm(k))) == col
        plt.close(fig)

    def test_no_title_ticks_in_degrees_and_the_legend_outside(self):
        fig, ax, _ = _draw()
        assert ax.get_title() == "" and ax.get_title(loc="left") == ""
        fig.canvas.draw()
        xs = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        ys = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert xs and all(x.endswith("°W") for x in xs)
        assert ys and all(y.endswith("°N") for y in ys)
        assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
        assert ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted()).x0 > 1
        assert ax.get_legend().get_texts()[0].get_fontsize() >= 16
        plt.close(fig)

    def test_the_title_render_figure_gives_every_panel_is_taken_off(self):
        # render_figure titles each panel from its outputs before it draws it
        fig, ax = plt.subplots()
        FS._title(ax, ["C1"], {"C1": _ref()["out"]})
        assert ax.get_title(loc="left") == "places"
        FS._draw_classes(plt, ax, _ref(), _panel(), LL, "panel 1")
        assert ax.get_title(loc="left") == "" and ax.get_title() == ""
        plt.close(fig)

    def test_a_line_for_each_edge_named_in_the_legend(self):
        mask = [[0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        fig, ax, n = _draw(lines=[{"label": "valley", "area_km2": 2000.0, "mask": mask},
                                  {"label": None, "mask": mask}])
        kinds = [type(c).__name__ for c in ax.collections]
        assert kinds.count("QuadMesh") == 1 and len(kinds) == 3   # the map and two edges
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts[-1] == "valley (2,000 km2)" and len(texts) == 5
        plt.close(fig)

    def test_two_edges_take_two_line_styles(self):
        """Two places drawn as edges are told apart by their lines, on the
        map and in the legend alike."""
        valley = [[0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        ridge = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        fig, ax, _ = _draw(lines=[{"label": "valley", "mask": valley},
                                  {"label": "ridge", "mask": ridge}])
        lines = [h for h in ax.get_legend().legend_handles if type(h).__name__ == "Line2D"]
        assert [h.get_linestyle() for h in lines] == ["-", "--"]
        edges = [c for c in ax.collections if type(c).__name__ != "QuadMesh"]
        assert len(edges) == 2
        assert edges[0].get_linestyle() != edges[1].get_linestyle()
        plt.close(fig)

    def test_the_grey_entry_only_where_a_cell_is_grey(self):
        """'The rest of the grid' names the grey cells; a map with none has
        no such entry."""
        full = [[1, 1, 2, 2], [2, 3, 3, 3], [1, 1, 3, 3]]
        fig, ax, _ = _draw(value=full)
        leg = ax.get_legend()
        assert [t.get_text() for t in leg.get_texts()] == ["low (10 km2)", "middle (1,500 km2)",
                                                          "high"]
        assert FS.CLASS_OTHER not in [to_hex(h.get_facecolor()) for h in leg.legend_handles]
        plt.close(fig)
        # one cell of no class, or one with no value, is grey and is named
        for g in ([[0, 1, 2, 2], [2, 3, 3, 3], [1, 1, 3, 3]],
                  [[np.nan, 1, 2, 2], [2, 3, 3, 3], [1, 1, 3, 3]]):
            fig, ax, _ = _draw(value=g)
            assert [t.get_text() for t in ax.get_legend().get_texts()][-1] == \
                "the rest of the grid"
            plt.close(fig)

    def test_todays_outline_is_drawn_as_a_line_too(self):
        mask = [[0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        fig, ax, _ = _draw(ll=dict(LL, outline=mask))
        assert len(ax.collections) == 2
        plt.close(fig)

    def test_an_outline_with_no_class_is_drawn(self):
        mask = [[0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]
        fig, ax, n = _draw(value=[[0] * 4] * 3, entries=[],
                           lines=[{"label": "valley", "mask": mask}])
        assert n == 7
        plt.close(fig)

    def test_a_cell_with_no_value_is_class_0(self):
        g = [[np.nan, 1, 1, 0], [0, 2, 2, 3], [0, 0, 3, 3]]
        fig, ax, n = _draw(value=g)
        assert n == 7
        plt.close(fig)

    @pytest.mark.parametrize("value,why", [
        ([[0, 1, 4, 0], [0, 2, 2, 3], [0, 0, 3, 3]], "class 4 has no legend entry"),
        ([[0, 1], [0, 2]], "is 2x2 cells, and the grid is 3x4"),
        ([[0, -1, 1, 0], [0, 2, 2, 3], [0, 0, 3, 3]], "not a class number"),
        ([[0, 1.5, 1, 0], [0, 2, 2, 3], [0, 0, 3, 3]], "not a class number"),
        ([[0] * 4] * 3, "holds no cell in a class and no region's edge"),
        ("nothing", "is not a grid of class numbers")])
    def test_a_grid_it_cannot_draw_fails_the_figure_naming_why(self, value, why):
        fig, ax = plt.subplots()
        with pytest.raises(FS.FigureFailed, match=why):
            FS._draw_classes(plt, ax, _ref(value), _panel(), LL, "panel 1")
        plt.close(fig)

    def test_a_line_off_the_grid_fails_the_figure(self):
        fig, ax = plt.subplots()
        with pytest.raises(FS.FigureFailed, match="line 1 is not on the grid"):
            FS._draw_classes(plt, ax, _ref(), _panel(lines=[{"label": "x", "mask": [[1]]}]),
                             LL, "panel 1")
        plt.close(fig)

    def test_render_figure_draws_it_alone_at_its_own_size(self, tmp_path, monkeypatch):
        sizes = []
        real = plt.subplots

        def spy(*a, **k):
            sizes.append(k.get("figsize"))
            return real(*a, **k)
        monkeypatch.setattr(plt, "subplots", spy)
        got = FS.render_figure({"id": "places", "panels": [_panel()]}, {"C1": _ref()},
                               tmp_path / "p.png", lat_lon=LL)
        assert got["ok"] and (tmp_path / "p.png").is_file()
        assert got["panels"][0]["status"] == "drawn" and got["panels"][0]["n"] == 7
        assert sizes == [FS.CLASS_SIZE]

    def test_every_tick_of_an_axis_has_the_same_decimals(self):
        """47 is never written beside 46.8: a whole degree takes the axis's
        decimals."""
        ll = {"lat": [46.6, 46.7, 46.8, 46.9, 47.0, 47.1], "lon": [-121.6, -121.4, -121.2, -121.0]}
        fig, ax = plt.subplots()
        FS._draw_classes(plt, ax, _ref([[1] * 4] * 6), _panel(entries=ENTRIES[:1]), ll,
                         "panel 1")
        fig.canvas.draw()
        for labels, end in ((ax.get_yticklabels(), "°N"), (ax.get_xticklabels(), "°W")):
            texts = [t.get_text() for t in labels if t.get_text()]
            assert len(texts) > 2 and all(t.endswith(end) for t in texts)
            decimals = {len(t[:-2].split(".")[1]) if "." in t else 0 for t in texts}
            assert decimals == {1}, texts
        assert "47.0°N" in [t.get_text() for t in ax.get_yticklabels()]
        plt.close(fig)

    @pytest.mark.parametrize("ticks,decimals", [
        ([46.8, 47.0, 47.2], 1), ([46.5, 47.0, 47.5], 1), ([46.0, 47.0], 0),
        ([121.25, 121.5], 2), ([46.800000000000004, 47.000000000000001], 1),
        ([0.1234567, 0.2], 6), ([], 0), ([46.8, float("nan")], 1)])
    def test_the_fewest_decimals_that_write_each_tick_exactly(self, ticks, decimals):
        assert FS.tick_decimals(ticks) == decimals

    @pytest.mark.parametrize("v,pos,neg,d,said", [
        (47.0, "N", "S", 1, "47.0°N"), (-121.2, "E", "W", 1, "121.2°W"),
        (-121.0, "E", "W", 0, "121°W"), (0.0, "E", "W", 1, "0.0°"),
        (-1e-12, "E", "W", 1, "0.0°")])
    def test_a_tick_in_degrees(self, v, pos, neg, d, said):
        assert FS._degrees(v, pos, neg, d) == said

    def test_a_map_without_the_option_is_drawn_as_before(self, tmp_path):
        r = {"ref": "P1.O1", "kind": "map", "out": {"field": "X", "units": "mm"},
             "value": [[1.0, 2.0, 3.0, 4.0]] * 3, "made": True, "error": None,
             "outcome": None, "why": None, "line": None}
        got = FS.render_figure({"id": "f", "panels": [{"kind": "map", "output": "P1.O1"}]},
                               {"P1.O1": r}, tmp_path / "m.png", lat_lon=LL)
        assert got["ok"] and got["panels"][0]["n"] == 12


# ── the scale ────────────────────────────────────────────────────────────────
class TestTheScale:
    @pytest.mark.parametrize("n", range(1, 13))
    def test_it_reads_light_to_dark_and_apart_from_the_grey(self, n):
        cols = FS.class_colours(n)
        assert len(cols) == n == len(set(cols))
        lum = [_luminance(c) for c in cols]
        assert all(a > b for a, b in zip(lum, lum[1:]))
        assert all(c.lower() != FS.CLASS_OTHER for c in cols)

    @pytest.mark.parametrize("n", [10, 11, 12, 20])
    def test_ten_or_more_read_light_to_dark_and_stay_apart_from_the_grey(self, n):
        """Past ColorBrewer's nine the colours are taken along its scale. They
        still read light to dark, and none comes near the grey of the cells
        of no place, on the scale and on the map alike."""
        cols = FS.class_colours(n)
        lum = [_luminance(c) for c in cols]
        assert len(set(cols)) == n and all(a > b for a, b in zip(lum, lum[1:]))
        grey = np.array(to_rgb(FS.CLASS_OTHER))
        assert min(float(np.linalg.norm(np.array(to_rgb(c)) - grey)) for c in cols) > 0.1
        ll = {"lat": [45.0, 45.2], "lon": [-100.0 + 0.2 * i for i in range(n + 1)]}
        entries = [{"label": f"band {k}", "area_km2": float(k)} for k in range(1, n + 1)]
        fig, ax = plt.subplots()
        FS._draw_classes(plt, ax, _ref([list(range(n + 1))] * 2), _panel(entries=entries), ll,
                         "panel 1")
        mesh = ax.collections[0]
        cmap, norm = mesh.get_cmap(), mesh.norm
        assert to_hex(cmap(norm(0))) == FS.CLASS_OTHER
        assert [to_hex(cmap(norm(k))) for k in range(1, n + 1)] == cols
        plt.close(fig)

    def test_four_classes_are_the_brewer_set_of_four(self):
        assert FS.class_colours(4) == ["#ffffcc", "#a1dab4", "#41b6c4", "#225ea8"]
        assert FS.class_colours(0) == []

    @pytest.mark.parametrize("area,said", [(382.5, "383 km2"), (1295.3, "1,295 km2"),
                                           (22.49, "22 km2"), (0.4, "0 km2"), (None, ""),
                                           ("x", ""), (float("nan"), "")])
    def test_areas_as_whole_km2_with_a_thousands_separator(self, area, said):
        assert FS.area_words(area) == said


# ── the Analyzer's closed list ───────────────────────────────────────────────
class TestTheAnalyzersListIsUnchanged:
    def test_the_option_is_not_in_the_list_the_analyzer_is_shown(self):
        shown = " ".join(FS.panel_lines(()))
        assert "classes" not in shown and "legend" not in shown
        assert [o.name for o in FS.PANEL_KINDS["map"].options] == ["value"]
        assert list(FS.PANEL_KINDS) == ["map", "series", "bars", "overlap", "timing", "screen"]

    def test_a_figure_the_model_writes_with_it_is_refused(self):
        plan = {"analyses": [{"id": "P1", "outputs": [
            {"id": "O1", "kind": "map", "field": "X", "region": None, "period": "2033"}]}]}
        ok = {"id": "f", "plan_items": ["P1"], "panels": [{"kind": "map", "output": "P1.O1"}]}
        assert FS.check_figure(ok, plan, ()) == []
        bad = {"id": "f", "plan_items": ["P1"], "panels": [
            {"kind": "map", "output": "P1.O1", "classes": {"entries": []}}]}
        assert FS.check_figure(bad, plan, ()) == [
            "f, panel 1: map has no option 'classes' (its options: value)"]

    def test_the_docstring_lists_it_with_the_other_page_options(self):
        assert '"classes"' in FS.__doc__ and "a map of classes" in FS.__doc__
