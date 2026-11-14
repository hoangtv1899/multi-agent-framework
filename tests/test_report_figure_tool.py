"""The figure tool's options for the report's own figures (package A5 of the
report rebuild): a stack of up to eight month panels, a month-by-month
difference panel with shaded months, a cell-by-cell panel, shared maps, a
letter per panel and labels in words. None of them is offered to the model:
the menu step 2b is shown is held, byte for byte, to its text at fdb7f07.

    pytest tests/test_report_figure_tool.py -v

Every number here is made up.
"""
import hashlib

import numpy as np
import pytest

from agents.analysis import figure_spec as FS

# THE MODEL'S MENU AT fdb7f07 (figure_spec.panel_lines with the server's six
# timing measures), the text step 2b's task carries. It must not change.
MENU_AT_FDB7F07 = (
    'THE PANELS (at most 4 a figure). Each panel is {"kind": ..., and "output" (one',
    'label) or "outputs" (a list of labels), and its options}; a label is a plan',
    "output's (P1.O1), or an extra output's of this figure (X1):",
    'ONE UNIT PER PANEL: a bars or series panel draws outputs of one unit, the unit',
    'the brief lists beside each output (percent, for every output of a panel drawn',
    'with "value": "percent"); a panel of several units is split into one panel per',
    'unit, in the order its outputs come, while the figure keeps to 4 panels, and',
    'otherwise the figure is refused, naming the units.',
    '  - "map": one per-cell map: a map, or a per-cell anomaly or difference (its',
    '      region null), a change drawn on a scale centred on zero. It draws map,',
    '      anomaly, difference; it takes one "output"; options: "value": "change" |',
    '      "percent" (for an anomaly or a difference: its change, or its percent of',
    '      the reference where the field allows one).',
    '  - "series": region series of one field and one step. It draws series; it',
    '      takes "outputs", a list; options: "years": [2015, ...] (the years drawn',
    '      on one seasonal axis; left out, the whole record is drawn); "normal":',
    "      true | false (true: the normal years' range and mean drawn beside them).",
    '  - "bars": one bar per output: a region\'s anomaly or difference, or a timing',
    "      measure's median shift with its settings' range; or one elevation split,",
    '      a bar per band. It draws anomaly, difference, a timing measure',
    '      (snow_peak, snow_gone, leaf_green_up, photosynthesis_onset, growth_peak,',
    '      half_runoff), elevation_split; it takes "outputs", a list; options:',
    '      "value": "change" | "percent" (for an anomaly or a difference: its',
    '      change, or its percent of the reference where the field allows one).',
    '  - "overlap": one overlap: the first map\'s mean in each quarter of the second,',
    '      with the correlation. It draws overlap; it takes one "output"; options:',
    '      "by_band": true | false (true: each band\'s correlation in place of the',
    '      quarters).',
    '  - "timing": one timing measure: its median shift at each stored setting, the',
    '      default marked. It draws a timing measure (snow_peak, snow_gone,',
    '      leaf_green_up, photosynthesis_onset, growth_peak, half_runoff); it takes',
    '      one "output".',
    '  - "screen": the plan\'s screen, drawn by its own display rule; a figure with',
    '      it draws nothing else. It draws screen; it takes no output.',
)
# step 2b's whole task for a study with a suite, at fdb7f07
SUITE_TASK_SHA256 = "938effa388551e1164abe0448b0e73bb50687383124daf2979bffbe98c5997a2"


def _timing_ids():
    from agents.analysis import analysis_plan as AP
    AP._elm()
    import domain_methods as M
    return M.TIMING_IDS


def test_the_menu_the_model_is_shown_is_the_text_at_fdb7f07():
    assert "\n".join(FS.panel_lines(_timing_ids())) == "\n".join(MENU_AT_FDB7F07)
    assert FS.MAX_PANELS == 4
    assert list(FS.PANEL_KINDS) == ["map", "series", "bars", "overlap", "timing", "screen"]
    for k in FS.CODE_KINDS:
        assert k not in FS.PANEL_KINDS and k not in " ".join(MENU_AT_FDB7F07)


def test_step_2b_task_is_the_text_at_fdb7f07():
    from agents.analysis import step2_investigate as S2
    assert hashlib.sha256(S2.suite_task().encode()).hexdigest() == SUITE_TASK_SHA256


def test_the_model_cannot_ask_for_a_code_panel():
    plan = {"analyses": [{"id": "P1", "outputs": [{"id": "O1", "kind": "difference",
                                                   "field": "GPP", "period": "2033",
                                                   "region": "r"}]}]}
    for kind in FS.CODE_KINDS:
        probs = FS.check_figure({"id": "f", "panels": [{"kind": kind, "output": "P1.O1"}]},
                                plan, ())
        assert any("no panel kind" in p for p in probs), (kind, probs)
    eight = {"id": "f", "layout": "stack",
             "panels": [{"kind": "bars", "outputs": ["P1.O1"]}] * 5}
    assert any("at most 4" in p for p in FS.check_figure(eight, plan, ()))


# ── made-up outputs ─────────────────────────────────────────────────────────

LABELS = [f"{y}-{m:02d}" for y in (2033, 2034) for m in range(1, 13)]


def _ref(ref, kind, out, value):
    return {"ref": ref, "kind": kind, "out": out, "value": value, "made": True, "error": None,
            "outcome": None, "why": None, "line": None}


def _months(n, values=None, **opts):
    refs, panels = {}, []
    for i in range(n):
        v = values if values is not None else [0.0] * 5 + [-1.0 - i, -2.0, -1.5] + [0.0] * 16
        refs[f"M{i}"] = _ref(f"M{i}", "months", {"field": "GPP", "units": "gC/m^2/month"},
                             {"labels": LABELS, "values": v})
        panels.append(dict({"kind": "months", "output": f"M{i}"}, **opts))
    return {"id": "t", "layout": "stack", "letters": True, "panels": panels}, refs


def test_a_stack_of_month_panels_holds_eight_and_no_more(tmp_path):
    spec, refs = _months(8)
    assert FS.panel_limit(spec) == 8
    assert FS.render_figure(spec, refs, tmp_path / "eight.png")["ok"]
    spec, refs = _months(9)
    done = FS.render_figure(spec, refs, tmp_path / "nine.png")
    assert not done["ok"] and "at most 8" in done["error"]
    assert not (tmp_path / "nine.png").exists()


def test_any_other_figure_keeps_four():
    assert FS.panel_limit({"layout": "stack", "panels": [{"kind": "series"}] * 5}) == 4
    assert FS.panel_limit({"layout": "grid", "panels": [{"kind": "months"}] * 5}) == 4
    assert FS.panel_limit({"panels": [{"kind": "months"}] * 2}) == 4
    # a row of maps over a row of cells holds two rows of four; a grid with a
    # kind the model may use keeps four, and so do maps and cells not in a grid
    grid = [{"kind": "shared_map"}] * 4 + [{"kind": "cells"}] * 4
    assert FS.panel_limit({"layout": "grid", "panels": grid}) == 8
    assert FS.panel_limit({"layout": "grid", "panels": grid[:3] + [{"kind": "map"}]}) == 4
    assert FS.panel_limit({"layout": "stack", "panels": grid}) == 4


def test_month_ticks_are_jan_apr_jul_oct_with_the_year_under_january():
    at, text = FS.month_ticks(LABELS)
    assert [LABELS[i] for i in at] == ["2033-01", "2033-04", "2033-07", "2033-10",
                                       "2034-01", "2034-04", "2034-07", "2034-10"]
    assert text == ["Jan\n2033", "Apr", "Jul", "Oct", "Jan\n2034", "Apr", "Jul", "Oct"]
    at, text = FS.month_ticks(["2033-05", "2033-06", "2033-07", "2033-08"])
    assert text == ["Jul\n2033"]                 # no January drawn: the year under the first


def test_a_month_panel_draws_bars_a_zero_line_shaded_months_and_words(tmp_path, monkeypatch):
    seen = {}
    real = FS._draw_months

    def spy(ax, r, opts, where):
        n = real(ax, r, opts, where)
        seen["ax"] = ax
        return n
    monkeypatch.setattr(FS, "_draw_months", spy)
    spec, refs = _months(1, shade_labels=["2033-06", "2033-07", "2033-08"])
    assert FS.render_figure(spec, refs, tmp_path / "one.png")["ok"]
    ax = seen["ax"]
    from matplotlib.colors import to_rgb
    bars = [p for p in ax.patches if p.get_height() != 0 and
            p.get_facecolor()[:3] == pytest.approx(to_rgb(FS.CHANGE_COLOUR))]
    assert len(bars) == 3 and all(p.get_height() < 0 for p in bars)
    spans = [p for p in ax.patches if p.get_facecolor()[:3] == pytest.approx((0.88,) * 3)]
    assert len(spans) == 1                       # one run of shaded months, June to August
    x0, w = spans[0].get_x(), spans[0].get_width()
    assert (x0, x0 + w) == (pytest.approx(4.5), pytest.approx(7.5))
    assert any(abs(line.get_ydata()[0]) < 1e-12 for line in ax.get_lines())
    assert ax.get_ylabel() == "Plant\ngrowth\n(gC/m2)"
    assert not [t for t in ax.texts if t.get_gid() == FS.LETTER_GID]   # one panel: no letter


def test_a_month_panel_against_the_normal_years_draws_its_range(tmp_path, monkeypatch):
    seen = {}
    real = FS._draw_months

    def spy(ax, r, opts, where):
        seen["ax"] = ax
        return real(ax, r, opts, where)
    monkeypatch.setattr(FS, "_draw_months", spy)
    n = len(LABELS)
    spec, refs = _months(1, values=[1.0] * n,
                         normal={"low": [0.5] * n, "high": [1.5] * n, "mean": [1.0] * n})
    assert FS.render_figure(spec, refs, tmp_path / "n.png")["ok"]
    ax = seen["ax"]
    assert ax.collections                        # the band
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())


def test_axis_words_keep_a_long_word_whole():
    assert FS.axis_words("QVEGT", "mm/month") == "Transpiration\n(mm)"
    assert FS.axis_words("TLAI", "1") == "Leaf area\nindex"
    assert FS.axis_words("NEWFIELD", "W/m^2") == "NEWFIELD\n(W/m2)"


def test_a_letter_goes_to_the_spot_with_the_least_drawn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax._report_points = np.array([[0.2, y] for y in np.linspace(0, 10, 60)])   # the left edge
    assert FS.place_letter(ax, "a") == "upper right"
    ax._report_points = np.r_[ax._report_points,
                              [[9.8, y] for y in np.linspace(6, 10, 20)]]       # and upper right
    assert FS.place_letter(ax, "b") == "center right"
    letters = [t for t in ax.texts if t.get_gid() == FS.LETTER_GID]
    assert [t.get_text() for t in letters] == ["a", "b"]
    assert letters[0].get_fontweight() == "bold"
    plt.close(fig)


def test_each_panel_of_a_stack_gets_its_letter_off_the_bars(tmp_path, monkeypatch):
    got = []
    real = FS.place_letter
    monkeypatch.setattr(FS, "place_letter",
                        lambda ax, letter: got.append((letter, real(ax, letter))) or got[-1][1])
    # a tall bar in the first month fills the upper left of each panel
    spec, refs = _months(2, values=[3.0, 0.0, -1.0] + [0.0] * 21)
    assert FS.render_figure(spec, refs, tmp_path / "l.png")["ok"]
    assert [a for a, _ in got] == ["a", "b"]
    assert all(spot != "upper left" for _, spot in got), got


def test_a_figure_without_the_option_has_no_letter(tmp_path):
    spec, refs = _months(2)
    spec.pop("letters")
    calls = []
    real = FS.place_letter
    FS.place_letter = lambda ax, letter: calls.append(letter) or real(ax, letter)
    try:
        assert FS.render_figure(spec, refs, tmp_path / "x.png")["ok"]
    finally:
        FS.place_letter = real
    assert calls == []


def _grid(nrow=6, ncol=8):
    lat = np.linspace(40.0, 40.5, nrow)
    lon = np.linspace(240.0, 240.7, ncol)        # east of Greenwich, as a grid may give it
    inside = np.zeros((nrow, ncol), dtype=bool)
    inside[1:5, 2:7] = True
    return lat, lon, inside


def test_shared_maps_are_on_one_scale_from_the_region_and_cells_share_limits(tmp_path):
    lat, lon, inside = _grid()
    a = np.where(inside, -2.0, 0.0)
    b = np.where(inside, -4.0, 0.0)
    b[0, 0] = 100.0                              # outside the region: not in the scale
    refs = {"U1": _ref("U1", "shared_map", {"title": "Summer 2033", "scale_label": "change"}, a),
            "U2": _ref("U2", "shared_map", {"title": "Summer 2034", "scale_label": "change"}, b),
            "C1": _ref("C1", "cells", {"x_label": "x", "y_label": "y", "c_label": "c"},
                       {"x": [1.0, 2.0, 3.0], "y": [-1.0, -2.0, -3.0], "c": [5.0, 6.0, 7.0]}),
            "C2": _ref("C2", "cells", {"x_label": "x", "y_label": "y", "c_label": "c"},
                       {"x": [4.0, 5.0], "y": [1.0, -6.0], "c": [3.0, 9.0]})}
    panels = [{"kind": "shared_map", "output": "U1"}, {"kind": "shared_map", "output": "U2"},
              {"kind": "cells", "output": "C1"}, {"kind": "cells", "output": "C2"}]
    ll = {"lat": lat, "lon": lon, "outline": inside}
    sc = FS.shared_scales(panels, refs, ll)
    assert sc["map"] == pytest.approx(np.percentile(np.r_[np.full(20, 2.0), np.full(20, 4.0)],
                                                    FS.MAP_SCALE_PERCENTILE))
    assert sc["c"] == (3.0, 9.0)
    assert sc["x"][0] < 1.0 and sc["x"][1] > 5.0 and sc["y"][0] < -6.0 and sc["y"][1] > 1.0
    spec = {"id": "g", "layout": "grid", "columns": 2, "letters": True, "panels": panels}
    done = FS.render_figure(spec, refs, tmp_path / "g.png", lat_lon=ll)
    assert done["ok"], done["error"]
    assert [p["status"] for p in done["panels"]] == ["drawn"] * 4


def test_a_cells_panel_with_nothing_to_draw_fails_the_figure(tmp_path):
    refs = {"C1": _ref("C1", "cells", {}, {"x": [np.nan], "y": [1.0], "c": [1.0]})}
    done = FS.render_figure({"id": "g", "layout": "grid", "columns": 1,
                             "panels": [{"kind": "cells", "output": "C1"}]}, refs,
                            tmp_path / "c.png")
    assert not done["ok"] and "no cell" in done["error"]
