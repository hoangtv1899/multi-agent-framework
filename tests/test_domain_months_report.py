"""The numbers report's months, its monthly table and where its numbers come from.

    pytest tests/test_domain_months_report.py -v

The records of test_domain_report, given months, seasons and a provenance
block in the shape domain_analysis makes them (2026-11-03). No LLM.
"""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from agents.analysis import domain_report as DR  # noqa: E402
import test_domain_report as T                    # noqa: E402

UNITS = {"GPP": {"month": "gC/m^2/month", "season": "gC/m^2/season"},
         "FAREA_BURNED": {"month": "proportion/month", "season": "proportion/season"}}
PROVENANCE = {
    "surface_file": "/somewhere/surfdata_x.nc", "registered_files": {"watershed": "/w.geojson"},
    "regions": {"vegetated cells": {"cells": 100, "area_km2": 60.0,
                                    "rule": "PCT_NATVEG >= 100", "sources": ["surfdata_x.nc"]},
                "watershed, vegetated": {"cells": 40, "area_km2": 24.0, "rule": "inside",
                                         "sources": ["w.geojson", "surfdata_x.nc"]}},
    "runs": {"control": {"run_dir": "/runs/control", "streams": {"h0": {
        "fields": ["GPP", "FAREA_BURNED"], "cadence": "monthly", "calendar": "noleap",
        "years_read": [2015, 2016], "records_per_year": {"2015": 12, "2016": 12},
        "files": [{"name": "c.elm.h0.2015-02-01-00000.nc", "bytes": 1031029600,
                   "modified": "2026-10-14T19:21:55+00:00"}],
        "records_left_out": 0}}}}}


def _months(run, reg, y, m):
    return 10.0 * m + (5.0 if "watershed" in reg else 0.0) - \
        (2.0 if run == "nofire" and m in (6, 7, 8) else 0.0)


def with_months(t, event=False):
    for run, r in t["runs"].items():
        r["period_units"] = UNITS
        r["monthly"] = {reg: {"GPP": {f"{y}-{m:02d}": _months(run, reg, y, m)
                                      for y in T.YEARS for m in range(1, 13)}}
                        for reg in T.REGIONS}
        r["seasonal"] = {reg: {"GPP": {f"{y}-JJA": sum(_months(run, reg, y, m)
                                                       for m in (6, 7, 8))
                                       for y in T.YEARS}} for reg in T.REGIONS}
    if "nofire" in t["runs"]:
        for kind in ("monthly", "seasonal"):
            c, n = t["runs"]["control"][kind], t["runs"]["nofire"][kind]
            t[f"{kind}_differences"] = {"nofire": {
                reg: {"GPP": {p: x - c[reg]["GPP"][p] for p, x in n[reg]["GPP"].items()}}
                for reg in T.REGIONS}}
    if event:
        t["event"]["monthly_normal"] = {"vegetated cells": {"GPP": {
            f"{m:02d}": 10.0 * m + 1.0 for m in range(1, 13)}}}
        t["event"]["seasonal_normal"] = {"vegetated cells": {"GPP": {"JJA": 213.0}}}
    t["provenance"] = PROVENANCE
    return t


@pytest.fixture
def drawn(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    saved = {}
    real = Figure.savefig

    def keep(fig, path, *a, **kw):
        saved[Path(path).name] = fig
        return real(fig, path, *a, **kw)
    monkeypatch.setattr(Figure, "savefig", keep)
    return saved


def _build(tmp_path, table, runs=None, analysis=None):
    rd = T._run_dir(tmp_path, table, runs or [T.CONTROL_ROW, T.NOFIRE],
                    analysis or T.ANALYSIS)
    DR.build(rd)
    return rd, (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()


class TestTheMonthlyFigures:
    # ONE FIGURE PER FIELD, A PANEL PER REGION (2026-11-03): every field the plan
    # names is drawn, titled with the table's region names.

    def test_each_run_and_its_difference_a_figure_per_field(self, tmp_path, drawn):
        rd, page = _build(tmp_path, with_months(T._table()))
        n = "domain_fig_monthly_GPP.png"
        assert (rd / "04_analysis" / n).is_file() and f"![{n}]({n})" in page
        fig = drawn[n]
        # a panel per region (watershed first, the plan's order), each run's
        # months above and the run minus the control below
        top, top2, bottom = fig.axes[0], fig.axes[1], fig.axes[2]
        assert top.get_title() == "watershed, vegetated"
        assert top2.get_title() == "vegetated cells"
        assert top.get_ylabel() == "gC/m^2/month" and top2.get_ylabel() == ""
        assert [ln.get_label() for ln in top.get_lines()] == ["control", "nofire"]
        assert list(top.get_lines()[0].get_ydata()[:3]) == [15.0, 25.0, 35.0]
        d = [ln for ln in bottom.get_lines() if len(ln.get_ydata()) == 24][0]
        assert list(d.get_ydata()[4:9]) == [0.0, -2.0, -2.0, -2.0, 0.0]
        assert bottom.get_ylabel() == "minus the control\ngC/m^2/month"
        # the run's colour above is its colour below
        assert d.get_color() == top.get_lines()[1].get_color()
        assert [t.get_text() for t in top.get_xticklabels()] == ["2015", "2016"]
        assert fig._suptitle.get_text() == "GPP (gC/m^2/month), by month"
        # the page says which figure holds which field
        assert ("*domain_fig_monthly_GPP.png: GPP (gC/m^2/month) by month, a panel per "
                "region (watershed, vegetated; vegetated cells): each run, and beneath each "
                "panel each run minus the control.*") in page

    def test_a_missing_month_is_a_gap_not_a_zero(self, tmp_path, drawn):
        t = with_months(T._table())
        t["monthly_differences"]["nofire"]["watershed, vegetated"]["GPP"]["2015-07"] = None
        _build(tmp_path, t)
        bottom = drawn["domain_fig_monthly_GPP.png"].axes[2]
        y = bottom.get_lines()[0].get_ydata()
        assert y[6] != y[6] and 0.0 not in list(y[5:8])

    def test_an_event_against_the_normal_years_monthly_mean(self, tmp_path, drawn):
        t = with_months(T._table(with_twin=False, with_event=True), event=True)
        _build(tmp_path, t, runs=[T.CONTROL_ROW],
               analysis={"regions": [{"name": "vegetated cells"}], "variables": ["GPP"],
                         "normal_years": [2016]})
        fig = drawn["domain_fig_monthly_GPP.png"]
        (ax,) = fig.axes
        assert ax.get_title() == "vegetated cells"
        lines = {ln.get_label(): list(ln.get_ydata()) for ln in ax.get_lines()}
        assert set(lines) == {"2015", "2016 mean"}
        assert lines["2015"][0] == 10.0 and lines["2016 mean"][5] == 61.0
        assert [t.get_text() for t in ax.get_xticklabels()] == list("JFMAMJJASOND")

    def test_every_field_is_drawn(self, tmp_path, drawn):
        # five fields, the fifth the soil water a summer question is half about
        t = with_months(T._table())
        extra = ("A1", "A2", "A3", "SOILWATER_10CM")
        for r in t["runs"].values():
            r["fields"] = dict(r["fields"], **{v: r["fields"]["GPP"] for v in extra})
            for reg in r["monthly"].values():
                for v in extra:
                    reg[v] = dict(reg["GPP"])
        rd, page = _build(tmp_path, t, analysis=dict(T.ANALYSIS, variables=["GPP", *extra]))
        for v in ("GPP", *extra):
            n = f"domain_fig_monthly_{v}.png"
            assert n in drawn and f"![{n}]({n})" in page and f"*{n}: {v} (" in page
        assert "Not drawn by month" not in page

    def test_a_field_without_months_is_named(self, tmp_path, drawn):
        t = with_months(T._table())
        rd, page = _build(tmp_path, t, analysis=dict(T.ANALYSIS,
                                                     variables=["GPP", "FAREA_BURNED"]))
        assert "domain_fig_monthly_FAREA_BURNED.png" not in drawn
        assert ("Not drawn by month: FAREA_BURNED (the table holds no months of it over the "
                "regions reported).") in page

    def test_long_titles_fit_their_panels(self, tmp_path, drawn):
        import matplotlib
        from agents.analysis.domain_report import _fit_titles
        t = with_months(T._table())
        long = "a region with a long name, vegetated, 1000-1500 m"
        for r in t["runs"].values():
            for kind in ("values", "monthly", "seasonal"):
                r[kind][long] = r[kind]["watershed, vegetated"]
        for k in ("monthly_differences", "differences"):
            t[k]["nofire"][long] = t[k]["nofire"]["watershed, vegetated"]
        # a copy: T.REGIONS is shared by every test
        t["regions"] = {**t["regions"], long: dict(t["regions"]["watershed, vegetated"])}
        _build(tmp_path, t, analysis=dict(T.ANALYSIS, regions=[
            {"name": "watershed"}, {"name": long}, {"name": "vegetated cells"}]))
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = drawn["domain_fig_monthly_GPP.png"]
        FigureCanvasAgg(fig)                     # closed after saving: draw it again
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            if ax.get_title():
                assert ax.title.get_window_extent(r).width <= ax.get_window_extent(r).width
        assert fig.axes[1].get_title().replace("\n", " ") == long
        # a name with no space breaks after its underscores
        f = matplotlib.figure.Figure(figsize=(1.0, 1.0))
        FigureCanvasAgg(f)
        ax = f.add_subplot(1, 1, 1)
        ax.set_title("SOILWATER_10CM_MEAN")
        _fit_titles(f)
        assert "_\n" in ax.get_title() and ax.get_title().replace("\n", "") == \
            "SOILWATER_10CM_MEAN"

    def test_a_table_without_months_draws_none(self, tmp_path, drawn):
        rd, page = _build(tmp_path, T._table())
        assert not [n for n in drawn if "monthly" in n]
        assert not (rd / "03_results" / DR.MONTHLY_CSV).exists()
        assert "regional_monthly.csv" not in page and "### What each run" not in page


class TestTheMonthlyTable:

    def test_every_month_and_season_with_the_difference(self, tmp_path):
        rd, page = _build(tmp_path, with_months(T._table()))
        rows = list(csv.DictReader(open(rd / "03_results" / DR.MONTHLY_CSV)))
        assert len(rows) == 2 * 2 * (24 + 2)           # runs x regions x (months + JJA)
        r = [x for x in rows if x["run"] == "nofire" and x["region"] == "vegetated cells"
             and x["period"] == "2015-07"][0]
        assert r == {"run": "nofire", "region": "vegetated cells", "variable": "GPP",
                     "period_kind": "month", "period": "2015-07", "value": "68.0",
                     "units": "gC/m^2/month", "minus_control": "-2.0"}
        s = [x for x in rows if x["period"] == "2016-JJA" and x["run"] == "control"]
        assert s[0]["period_kind"] == "season" and s[0]["minus_control"] == ""
        link = "[03_results/regional_monthly.csv](../03_results/regional_monthly.csv)"
        assert page.count(link) == 2
        rec = json.loads((rd / "04_analysis" / DR.REPORT_JSON).read_text())
        assert rec["monthly_csv"] == "03_results/regional_monthly.csv"
        # the page holds no table of months
        assert "| 2015-07 |" not in page

    def test_the_normal_years_mean_is_in_the_table(self, tmp_path):
        t = with_months(T._table(with_twin=False, with_event=True), event=True)
        rd, _ = _build(tmp_path, t, runs=[T.CONTROL_ROW],
                       analysis={"regions": [{"name": "vegetated cells"}],
                                 "variables": ["GPP"], "normal_years": [2016]})
        rows = list(csv.DictReader(open(rd / "03_results" / DR.MONTHLY_CSV)))
        n = [x for x in rows if x["period_kind"] == "month, mean of 2016"]
        assert len(n) == 12 and n[5]["period"] == "06" and n[5]["value"] == "61.0"
        assert [x["value"] for x in rows if x["period_kind"] == "season, mean of 2016"] == \
            ["213.0"]


class TestWhereTheNumbersComeFrom:

    def test_the_files_and_the_regions(self, tmp_path):
        rd, page = _build(tmp_path, with_months(T._table()))
        tail = page[page.index("## Where the numbers come from"):]
        for line in (
                "### What each run was read from",
                "- run **control**: `/runs/control`",
                "  - stream h0 (monthly, noleap calendar): GPP, FAREA_BURNED; records read: "
                "12 in 2015, 12 in 2016; 1 file",
                "    - c.elm.h0.2015-02-01-00000.nc (1031 MB, modified 2026-10-14 19:21:55 UTC)",
                "| region | cells | area (km2) | rule | drawn from |",
                "| watershed, vegetated | 40 | 24 | inside | w.geojson, surfdata_x.nc |",
                "The vegetated cells are read from the control's surface file, "
                "surfdata_x.nc."):
            assert line in tail, line
        assert "\u2014" not in page and "/somewhere" not in page
