#!/usr/bin/env python3
"""Analyzer step 1d — the combined spatial figure.

The point of this module is that it decides LAYOUT and nothing else: every
point it draws comes from a validator's own map_points(). So the tests that
matter are the parity ones — the combined figure and the standalone map must
draw the same points, or a reader comparing the two figures is comparing two
different answers to the same question.

The rest guard the two properties that are easy to get wrong and invisible in a
rendered PNG: a colour scale per ROW rather than per figure, and rows/overlays
that follow the data instead of being drawn empty.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step1_maps as maps                  # noqa: E402
from agents.analysis import step1_validate_swe as swe_mod       # noqa: E402
from agents.analysis import step1_validate_streamflow as f_mod  # noqa: E402
from agents.analysis import step1_validate_wtd as w_mod         # noqa: E402

SQUARE = [[[-108.0, 38.0], [-107.0, 38.0], [-107.0, 39.0],
           [-108.0, 39.0], [-108.0, 38.0]]]


class _Ctx:
    def __init__(self, cols, obs=None, boundary=SQUARE):
        self.data = {"observations": obs or {}, "boundary": boundary}
        self._c = cols

    @property
    def columns(self):
        return self._c


def _ctx():
    cols = [{"case_name": "col_01", "lat": 38.5, "lon": -107.5,
             "elevation_m": 3000.0, "fan_wtd_m": 25.0},
            {"case_name": "col_02", "lat": 38.2, "lon": -107.2,
             "elevation_m": 2600.0, "fan_wtd_m": 4.4}]
    obs = {"swe": {"stations": [
                {"name": "ST1", "lat": 38.4, "lon": -107.4},
                {"name": "ST2", "lat": 37.1, "lon": -107.4}]}}   # ST2 outside
    return _Ctx(cols, obs)


SWE = {"observed": [{"entity": "ST1", "available": True, "mean_swe_mm": 210.0},
                    {"entity": "ST2", "available": True, "mean_swe_mm": 190.0}],
       "model": [{"entity": "col_01", "available": True, "mean_swe_mm": 33.0},
                 {"entity": "col_02", "available": True, "mean_swe_mm": 410.0}]}

FLOW = {"gauges": [{"id": "G1", "lat": 38.5, "lon": -107.6,
                    "mean_mm_day": 1.2, "drainage_area_km2": 5000.0},
                   {"id": "G2", "lat": 38.3, "lon": -107.3,
                    "mean_mm_day": 0.8, "drainage_area_km2": 40.0}],
        "columns": [{"id": "col_01", "lat": 38.5, "lon": -107.5,
                     "mean_mm_day": 0.001},
                    {"id": "col_02", "lat": 38.2, "lon": -107.2,
                     "mean_mm_day": 5.2}]}

WTD_NO_WELLS = {"wells": [],
                "fan": [{"id": "col_01", "lat": 38.5, "lon": -107.5,
                         "wtd_m": 25.0},
                        {"id": "col_02", "lat": 38.2, "lon": -107.2,
                         "wtd_m": 4.4}],
                "model": [{"id": "col_01", "lat": 38.5, "lon": -107.5,
                           "wtd_m": 70.0},
                          {"id": "col_02", "lat": 38.2, "lon": -107.2,
                           "wtd_m": 2.1}]}

WTD_WELLS = dict(WTD_NO_WELLS,
                 wells=[{"id": "W1", "lat": 38.45, "lon": -107.35,
                         "wtd_m": 6.2, "n_obs": 11,
                         "series": [{"date": "2019-05-15", "wtd_m": 6.0},
                                    {"date": "2019-09-15", "wtd_m": 6.4}]},
                        {"id": "W2", "lat": 38.20, "lon": -107.15,
                         "wtd_m": 21.7, "n_obs": 3,
                         "series": [{"date": "2019-04-02", "wtd_m": 20.9}]}])


class TestRowsFollowTheData:

    def test_all_three_rows_when_all_three_are_given(self):
        rows = maps.build_rows(_ctx(), swe=SWE, streamflow=FLOW,
                               wtd=WTD_NO_WELLS)
        assert [r["label"] for r in rows] == [
            "mean SWE (mm)", "mean runoff (mm/day)", "water-table depth (m)"]

    def test_a_missing_observable_drops_its_row(self):
        """No streamflow record means no streamflow row — not an empty one."""
        rows = maps.build_rows(_ctx(), swe=SWE, wtd=WTD_NO_WELLS)
        assert [r["label"] for r in rows] == ["mean SWE (mm)",
                                              "water-table depth (m)"]

    def test_no_observables_at_all_gives_no_rows(self):
        assert maps.build_rows(_ctx()) == []


class TestScalePerRow:
    """mm, mm/day and m cannot share a colourbar; within a row they must."""

    def test_each_row_carries_its_own_units_label(self):
        rows = maps.build_rows(_ctx(), swe=SWE, streamflow=FLOW,
                               wtd=WTD_NO_WELLS)
        assert len({r["label"] for r in rows}) == 3

    def test_swe_is_linear_and_the_other_two_are_log(self):
        rows = maps.build_rows(_ctx(), swe=SWE, streamflow=FLOW,
                               wtd=WTD_NO_WELLS)
        assert [r["log"] for r in rows] == [False, True, True]

    def test_both_panels_of_a_row_are_in_that_rows_units(self):
        """A shared scale is only honest if both panels measure the same thing.
        Sizes differ (gauges are sized by area); the VALUES must be comparable.
        """
        rows = maps.build_rows(_ctx(), streamflow=FLOW)
        vals = [q[2] for p in rows[0]["panels"] for q in p["points"]]
        assert min(vals) >= 0 and max(vals) == 5.2      # all mm/day


class TestWtdOverlay:
    """Three sources, two slots: wells ride on the Fan panel."""

    def test_wells_overlay_the_fan_panel_when_they_exist(self):
        rows = maps.build_rows(_ctx(), wtd=WTD_WELLS)
        fan_panel = rows[0]["panels"][0]
        assert fan_panel["title"] == "Fan 2013"
        assert len(fan_panel["overlay"]["points"]) == 2
        assert fan_panel["overlay"]["title"] == "USGS wells"

    def test_no_overlay_when_no_well_lies_in_the_basin(self):
        """The 2019 Upper Gunnison case: all ten wells outside."""
        rows = maps.build_rows(_ctx(), wtd=WTD_NO_WELLS)
        assert rows[0]["panels"][0]["overlay"] is None

    def test_overlay_shares_the_rows_scale_not_its_own(self):
        """Wells carry raw depths, so plot_grid normalises them with the row —
        a separate scale would make a 6 m well and a 6 m column different
        colours."""
        rows = maps.build_rows(_ctx(), wtd=WTD_WELLS)
        ov = rows[0]["panels"][0]["overlay"]["points"]
        assert [q[2] for q in ov] == [6.2, 21.7]


class TestSizing:

    def test_gauges_are_sized_by_drainage_area(self):
        """A 40 km2 gauge and a 5,000 km2 gauge are not equivalent evidence."""
        rows = maps.build_rows(_ctx(), streamflow=FLOW)
        obs, mod = rows[0]["panels"]
        assert obs["sizes"] is not None and len(obs["sizes"]) == 2
        assert obs["sizes"][0] > obs["sizes"][1]        # G1 drains more
        assert mod.get("sizes") is None                 # columns are all 1 m2

    def test_swe_points_are_unsized(self):
        rows = maps.build_rows(_ctx(), swe=SWE)
        assert all(p.get("sizes") is None for p in rows[0]["panels"])


class TestParityWithStandaloneMaps:
    """If the combined figure and a standalone figure disagree about where a
    station is or what it measured, the bug is in the validator they share.
    These pin that they cannot drift apart."""

    def test_swe_points_match_the_validators_own(self):
        ctx = _ctx()
        obs, mod = swe_mod.map_points(SWE, ctx)
        rows = maps.build_rows(ctx, swe=SWE)
        assert rows[0]["panels"][0]["points"] == obs
        assert rows[0]["panels"][1]["points"] == mod

    def test_streamflow_points_and_sizes_match(self):
        ctx = _ctx()
        obs, mod, sizes = f_mod.map_points(FLOW, ctx)
        rows = maps.build_rows(ctx, streamflow=FLOW)
        assert rows[0]["panels"][0]["points"] == obs
        assert rows[0]["panels"][0]["sizes"] == sizes
        assert rows[0]["panels"][1]["points"] == mod

    def test_wtd_points_match(self):
        ctx = _ctx()
        wells, fan, model, _ = w_mod.map_points(WTD_WELLS, ctx)
        rows = maps.build_rows(ctx, wtd=WTD_WELLS)
        assert rows[0]["panels"][0]["points"] == fan
        assert rows[0]["panels"][0]["overlay"]["points"] == wells
        assert rows[0]["panels"][1]["points"] == model

    def test_swe_excludes_the_out_of_basin_station_the_same_way(self):
        """compare() drops out-of-basin stations; map_points must not
        resurrect them by looking coordinates up from raw reception data."""
        ctx = _ctx()
        only_in = {"observed": [SWE["observed"][0]], "model": SWE["model"]}
        obs, _mod = swe_mod.map_points(only_in, ctx)
        assert [q[3] for q in obs] == ["ST1"]


class TestRendering:

    def test_renders_a_file(self, tmp_path):
        out = tmp_path / "combined.png"
        # basemap off: a figure must not depend on a third-party raster being
        # reachable, and the test must not depend on the network.
        p = maps.create_validation_spatial_map(_ctx(), out, swe=SWE, streamflow=FLOW,
                          wtd=WTD_WELLS, basemap=False)
        assert Path(p).exists() and Path(p).stat().st_size > 5000

    def test_empty_rows_are_dropped_not_drawn(self, tmp_path):
        from agents.analysis.step1_geo import plot_grid
        rows = [{"label": "nothing (m)", "panels": [{"title": "A",
                                                     "points": []}]},
                {"label": "something (m)",
                 "panels": [{"title": "B",
                             "points": [(-107.5, 38.5, 1.0, "x")]}]}]
        out = tmp_path / "g.png"
        assert Path(plot_grid(rows, SQUARE, out, basemap=False)).exists()

    def test_nothing_to_map_is_stated_not_blank(self, tmp_path):
        from agents.analysis.step1_geo import plot_grid
        out = tmp_path / "empty.png"
        p = plot_grid([{"label": "x", "panels": [{"title": "A", "points": []}]}],
                      SQUARE, out, basemap=False)
        assert Path(p).exists()

    def test_figure_height_follows_the_extent_aspect(self, tmp_path):
        """A hardcoded row height left 2.6 in of dead space per row under
        cartopy's fixed aspect. A tall, narrow basin must give a taller figure
        than a wide, flat one for the same row count."""
        import matplotlib
        matplotlib.use("Agg")
        from PIL import Image
        from agents.analysis.step1_geo import plot_grid

        def height(ring):
            rows = [{"label": "v", "panels": [
                {"title": "A", "points": [(ring[0][0], ring[0][1], 1.0, "a")]}]}]
            out = tmp_path / f"{abs(hash(str(ring)))}.png"
            plot_grid(rows, [ring], out, basemap=False)
            return Image.open(out).size[1]

        wide = [[-109.0, 38.0], [-107.0, 38.0], [-107.0, 38.5],
                [-109.0, 38.5], [-109.0, 38.0]]
        tall = [[-107.5, 37.0], [-107.0, 37.0], [-107.0, 39.0],
                [-107.5, 39.0], [-107.5, 37.0]]
        assert height(tall) > height(wide)


def _clims(draw):
    """The (vmin, vmax) each scatter was actually drawn with, in call order."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    real, out = Axes.scatter, []

    def spy(self, *a, **kw):
        sc = real(self, *a, **kw)
        try:
            out.append(tuple(round(v, 6) for v in sc.get_clim()))
        except Exception:
            pass
        return sc

    Axes.scatter = spy
    try:
        draw()
    finally:
        Axes.scatter = real
    return out


class TestRenderedScales:
    """The central claim of the figure, asserted on what the panels were drawn
    with rather than on the row spec: a colour means ONE thing within a row and
    is free to mean something else in the next."""

    def test_within_a_row_the_panels_share_and_across_rows_they_differ(
            self, tmp_path):
        out = tmp_path / "g.png"
        clims = _clims(lambda: maps.create_validation_spatial_map(
            _ctx(), out, swe=SWE, streamflow=FLOW, wtd=WTD_NO_WELLS,
            basemap=False))
        assert len(clims) == 6, clims
        rows = [clims[0:2], clims[2:4], clims[4:6]]
        for i, r in enumerate(rows):
            assert r[0] == r[1], f"row {i} panels on different scales: {r}"
        assert len({r[0] for r in rows}) == 3, \
            f"rows share a scale across unlike units: {rows}"

    def test_a_rows_scale_covers_both_of_its_panels(self, tmp_path):
        """SWE runs 33-410 in the model and 190-210 observed. A scale fitted to
        one panel would clip the other."""
        out = tmp_path / "g2.png"
        clims = _clims(lambda: maps.create_validation_spatial_map(_ctx(), out, swe=SWE,
                                             basemap=False))
        vmin, vmax = clims[0]
        assert vmin <= 33.0 and vmax >= 410.0, clims

    def test_the_well_overlay_is_drawn_on_the_rows_scale(self, tmp_path):
        """A 6 m well and a 6 m column must be the same colour."""
        out = tmp_path / "g3.png"
        clims = _clims(lambda: maps.create_validation_spatial_map(_ctx(), out, wtd=WTD_WELLS,
                                             basemap=False))
        # Fan panel, then its overlay, then ELM — all one scale.
        assert len(clims) == 3, clims
        assert len(set(clims)) == 1, clims


class TestNoTightBbox:
    def test_bbox_inches_is_never_tight(self, tmp_path):
        """Under cartopy the tight bounding box is computed before the tiles
        exist and crops the whole figure down to the colourbar."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        real, seen = Figure.savefig, []

        def spy(self, *a, **kw):
            seen.append(dict(kw))
            return real(self, *a, **kw)

        Figure.savefig = spy
        try:
            maps.create_validation_spatial_map(_ctx(), tmp_path / "g.png", swe=SWE,
                          streamflow=FLOW, wtd=WTD_WELLS, basemap=False)
        finally:
            Figure.savefig = real
        assert seen
        assert all(kw.get("bbox_inches") is None for kw in seen), seen


class TestSymlogKeepsZeros:
    """Moved here from the streamflow suite, which asserted it against
    plot_two_maps. A linear scale was unreadable — one column's 5.2 mm/day
    flattened all seven gauges into one colour. But plain LogNorm cannot take a
    zero, and 15 of 19 columns produced EXACTLY zero, which is a different
    result from "very small"."""

    def test_a_log_row_containing_an_exact_zero_still_renders_it(self, tmp_path):
        flow = {"gauges": [{"id": "G", "lat": 38.5, "lon": -107.6,
                            "mean_mm_day": 1.2,
                            "drainage_area_km2": 100.0}],
                "columns": [{"id": "c1", "lat": 38.5, "lon": -107.5,
                             "mean_mm_day": 0.0},
                            {"id": "c2", "lat": 38.2, "lon": -107.2,
                             "mean_mm_day": 5.2}]}
        rows = maps.build_rows(_ctx(), streamflow=flow)
        assert rows[0]["log"] is True
        vals = [q[2] for p in rows[0]["panels"] for q in p["points"]]
        assert 0.0 in vals, "an exact zero must survive into the figure"

        clims = _clims(lambda: maps.create_validation_spatial_map(
            _ctx(), tmp_path / "z.png", streamflow=flow, basemap=False))
        assert clims, "nothing was drawn"
        # SymLogNorm is anchored at zero so a zero-flow column is a colour, not
        # a dropped point or a divide-by-zero.
        assert all(c[0] == 0.0 for c in clims), clims
