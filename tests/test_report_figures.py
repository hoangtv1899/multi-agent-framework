"""The report's figures and tables, drawn by code (src/agents/analysis/
report_figures.py, package A6 and interface I1 of the report rebuild).

    pytest tests/test_report_figures.py -v

Most tests draw from a stand-in for the four calls the module makes
(FakeAccess: get_series, get_map, region_mask, lat_lon) over made-up numbers.
The last ones draw from real made-up results stores (tests/test_domain_suite.py)
through the server's SuiteAccess, to hold the module's own gate and log: no
line is written to lookups.jsonl. Every number, name and path is made up.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from agents.analysis import analysis_plan as AP
from agents.analysis import figure_spec as FS
from agents.analysis import report_figures as RF
from agents.analysis.step4_report import format_value

REGION = "basin, vegetated"
RUN = "drier"
YEARS = [2033, 2034]
NORMAL = [2031, 2032, 2033, 2034]
NROW, NCOL = 5, 6
LAT = np.linspace(40.0, 40.4, NROW)
LON = np.linspace(240.0, 240.5, NCOL)
MASK = np.zeros((NROW, NCOL), dtype=bool)
MASK[1:4, 1:5] = True
SUMMER = [f"{y}-{m:02d}" for y in YEARS for m in (6, 7, 8)]
FIELDS = ["RAIN", "SNOW", "TBOT", "SOILWATER_10CM", "BTRAN", "QVEGT", "GPP", "NPP", "TLAI",
          "TOTVEGC", "FPG", "QCHARGE"]
# the made-up change of each field in the summer months of the new run
SUMMER_CHANGE = {"RAIN": -3.0, "SNOW": -0.5, "SOILWATER_10CM": -0.2, "BTRAN": -0.01,
                 "QVEGT": -0.4, "GPP": -2.0, "NPP": -1.5, "TLAI": -0.01, "TOTVEGC": -5.0,
                 "FPG": 0.001}
TOTALS = ("RAIN", "SNOW", "QVEGT", "GPP", "NPP", "QCHARGE")


def _units(field, step):
    base = {"RAIN": "mm", "SNOW": "mm", "QVEGT": "mm", "QCHARGE": "mm", "GPP": "gC/m^2",
            "NPP": "gC/m^2", "TBOT": "K", "SOILWATER_10CM": "kg/m2", "BTRAN": "1", "TLAI": "1",
            "TOTVEGC": "gC/m^2", "FPG": "proportion"}.get(field, "1")
    per = {"month": "month", "season": "season", "year": "yr", "wateryear": "water year"}
    return f"{base}/{per[step]}" if field in TOTALS else base


class FakeAccess:
    """SuiteAccess's four calls over made-up numbers, each call kept."""

    def __init__(self, months, maps):
        self.months, self.maps, self.calls = months, maps, []

    def get_series(self, run, field, region, step):
        self.calls.append(("get_series", run, field, region, step))
        d = self.months.get((run, field))
        if d is None:
            raise LookupError(f"the suite of {run} holds no {step} series of {field}")
        labs = sorted(d) if step == "month" else []
        return {"labels": labs, "values": np.array([d[x] for x in labs], dtype=float),
                "units": _units(field, step)}

    def get_map(self, run, field, period):
        self.calls.append(("get_map", run, field, period))
        if period.startswith("normal"):
            part = period[len("normal-"):] if period != "normal" else ""
            ps = [str(y) if not part else f"WY{y}" if part == "WY" else f"{y}-{part}"
                  for y in NORMAL]
            return np.mean([self.get_map(run, field, p) for p in ps], axis=0)
        a = self.maps.get((run, field, period))
        if a is None:
            raise LookupError(f"no map of {field} for {period}")
        return np.array(a, dtype=float)

    def region_mask(self, region):
        self.calls.append(("region_mask", region))
        return MASK.copy()

    def lat_lon(self):
        self.calls.append(("lat_lon",))
        return {"lat": LAT, "lon": LON}


def _gradient(k=1.0):
    return k * (1.0 + np.add.outer(np.arange(NROW), np.arange(NCOL)) / 10.0)


def world(change="summer", start_trace=True):
    """Made-up months and maps of a control run (2031 to 2034) and a new run
    (2033 and 2034). `change`: "summer" cuts rain and snow in June to August,
    by an amount that differs from cell to cell; "every" changes the plants
    in every month and leaves the weather alone; "none" changes nothing."""
    months, maps = {}, {}
    for i, f in enumerate(FIELDS):
        ctl = {f"{y}-{m:02d}": 10.0 + i + m + 0.1 * (y - 2031)
               for y in range(2031, 2035) for m in range(1, 13)}
        months[("control", f)] = ctl
        new = {}
        for lab in (x for x in ctl if int(x[:4]) in YEARS):
            d = 0.0
            if change == "summer" and lab in SUMMER:
                d = SUMMER_CHANGE.get(f, 0.0)
            if change == "every" and f not in ("RAIN", "SNOW", "TBOT", "QCHARGE"):
                d = 0.5 + 0.01 * i
            new[lab] = ctl[lab] + d
        if start_trace and f == "TBOT":
            new["2033-01"] += 0.002                # the later start's first month
        months[(RUN, f)] = new
        for y in range(2031, 2035):
            for p in (f"{y}-JJA", str(y), f"WY{y}"):
                base = _gradient(10.0 + i)
                maps[("control", f, p)] = base
                if y in YEARS:
                    d = np.zeros_like(base)
                    if change == "summer" and p.endswith("JJA") and f in ("RAIN", "SNOW"):
                        d = np.where(MASK, -0.3 * base, 0.0)
                    if change == "summer" and p.endswith("JJA") and f == "GPP":
                        d = np.where(MASK, -0.05 * _gradient(), 0.0)
                    if change == "every" and f not in ("RAIN", "SNOW", "TBOT"):
                        d = np.full_like(base, 0.5) + 0.01 * _gradient()
                    maps[(RUN, f, p)] = base + d
    return FakeAccess(months, maps)


def _out(o):
    return dict({"region": REGION, "run": RUN}, **o)


def what_if_plan(extra_analyses=(), first=("GPP", "NPP", "TLAI", "TOTVEGC", "FPG", "QCHARGE")):
    analyses = [
        {"id": "P1", "fields": list(first), "regions": [REGION], "periods": ["2033-JJA", "2033"],
         "outputs": [_out({"id": f"O{i}", "kind": "difference", "field": f, "period": p})
                     for i, (f, p) in enumerate([(f, p) for f in first
                                                 for p in ("2033-JJA", "2033")], 1)]},
        {"id": "P2", "fields": ["QVEGT", "TBOT", "BTRAN", "SOILWATER_10CM", "SNOW", "RAIN"],
         "regions": [REGION], "periods": ["2033-JJA"],
         "outputs": [_out({"id": "O1", "kind": "difference", "field": "QVEGT",
                           "period": "2033-JJA"})]},
        {"id": "P3", "fields": ["GPP"], "regions": [REGION, f"{REGION}, below 1000 m"],
         "periods": ["2033-JJA"],
         "outputs": [_out({"id": "O1", "kind": "difference", "field": "GPP",
                           "period": "2033-JJA", "region": None}),
                     {"id": "O2", "kind": "elevation_split", "region": REGION,
                      "inputs": ["O1"]}]},
    ] + list(extra_analyses)
    return {"comparisons": [{"id": "C1", "run": RUN, "against": "control", "years": YEARS}],
            "baseline": {"normal_years": NORMAL}, "analyses": analyses}


def study(tmp_path, changes, panels_changed=None, lookups=()):
    """A made-up study folder: the design's record of the new run and the
    analysis folder with its lookups."""
    rd = tmp_path / "study"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "04_analysis").mkdir(parents=True)
    run = {"name": RUN, "changes": changes, "rows": []}
    if panels_changed is not None:
        run["panels"] = [{"title": "a made-up input", "changed": panels_changed}]
    (rd / "01_inputs" / "run_given.json").write_text(json.dumps(
        {"control": "made_up", "years": YEARS, "runs": [run]}))
    (rd / "04_analysis" / "lookups.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lookups))
    return rd


CUT = [{"id": "weather_scale", "params": {"precip_factor": 0.7, "temperature_K": 0.0,
                                          "months": [6, 7, 8], "years": YEARS,
                                          "region": "basin"}}]
CO2 = [{"id": "co2_ppmv", "params": {"ppmv": 500}}]


@pytest.fixture
def drawn(monkeypatch):
    """Every figure specification the module hands the figure tool, by id."""
    specs = {}
    real = FS.render_figure

    def spy(spec, refs, out_path, **kw):
        specs[spec["id"]] = (spec, refs)
        return real(spec, refs, out_path, **kw)
    monkeypatch.setattr(FS, "render_figure", spy)
    return specs


def _build(rd, plan, acc):
    return RF.build(rd, plan, rd / "04_analysis", access=acc)


# ── figure 1, over time ─────────────────────────────────────────────────────

def test_the_panels_follow_the_change_then_the_catalog_and_split_above_eight(tmp_path, drawn):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world())
    assert rec["problems"] == []
    over = [f for f in rec["figures"] if f["id"].startswith("over_time")]
    assert [f["id"] for f in over] == ["over_time", "over_time_2"]
    assert [f["number"] for f in rec["figures"]] == [1, 2, 3]
    assert [p["field"] for p in over[0]["panels"]] == ["RAIN", "SNOW", "SOILWATER_10CM", "BTRAN",
                                                       "QVEGT"]
    assert [p["field"] for p in over[1]["panels"]] == ["GPP", "NPP", "TLAI", "TOTVEGC", "FPG"]
    assert [p["letter"] for p in over[1]["panels"]] == list("abcde")
    assert over[0]["heading"] == "Water and energy month by month"
    assert over[1]["heading"] == "Plants, carbon, nitrogen and fire month by month"
    for f in over:
        assert len(f["panels"]) <= FS.STACK_MAX_PANELS
        assert (rd / "04_analysis" / f["file"]).is_file()
    spec, refs = drawn["report_over_time"]
    assert spec["layout"] == "stack" and spec["letters"] is True
    assert all(p["kind"] == "months" for p in spec["panels"])
    # each bar is the new run minus the control run, month by month
    v = refs["M1"]["value"]
    assert v["labels"][0] == "2033-01" and len(v["labels"]) == 24
    assert v["values"][v["labels"].index("2033-06")] == pytest.approx(-3.0)
    assert v["values"][v["labels"].index("2033-05")] == pytest.approx(0.0)
    assert over[0]["caption"].startswith(
        "The new run minus the control run over the basin's vegetated land, for each month "
        "of 2033 and 2034. Panels: (a) rain, (b) snowfall, (c) topsoil water,")
    assert over[1]["comparisons"] == [1]


def test_the_quantity_the_run_changes_comes_before_its_catalog_place():
    """A changed quantity leads even where the catalog's order of groups puts
    another group first; the rest keep the catalog's order, and a field the
    catalog does not hold comes last, in the plan's order."""
    notes = {"LEAFX": {"group": "leaves"}, "WARMX": {"group": "weather in"},
             "WETX": {"group": "weather in"}, "SOILX": {"group": "soil"}}
    groups = ["leaves", "soil", "weather in"]
    assert RF.ordered(["WETX", "SOILX", "NEWX", "LEAFX", "WARMX"], notes, groups) == \
        ["LEAFX", "SOILX", "WARMX", "WETX", "NEWX"]
    assert RF.ordered(["WETX", "SOILX", "NEWX", "LEAFX", "WARMX"], notes, groups,
                      changed=["WETX"]) == ["WETX", "LEAFX", "SOILX", "WARMX", "NEWX"]


def test_eight_panels_stay_one_figure(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    plan = what_if_plan()
    plan["analyses"][1]["fields"] = ["RAIN", "SNOW", "QVEGT"]
    rec = _build(rd, plan, world())
    over = [f for f in rec["figures"] if f["id"].startswith("over_time")]
    assert len(over) == 1 and len(over[0]["panels"]) == 8
    assert over[0]["heading"] == "Each quantity month by month"


def test_a_quantity_the_runs_hold_alike_has_no_panel_and_is_named(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world())
    fields = [p["field"] for f in rec["figures"] for p in f["panels"]]
    assert "QCHARGE" not in fields
    water = rec["figures"][0]
    assert water["caption"].endswith("Aquifer recharge (QCHARGE) has no panel, because the two "
                                     "runs hold the same value of it in every month.")
    assert any(s.startswith("Aquifer recharge (QCHARGE) has no panel over time")
               for s in rec["not_drawn"])


def test_weather_the_design_leaves_alone_has_no_panel_though_its_first_month_differs(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world(start_trace=True))
    fields = [p["field"] for f in rec["figures"] for p in f["panels"]]
    assert "TBOT" not in fields
    assert "Air temperature (TBOT) has no panel, because the new run does not change it." in \
        rec["figures"][0]["caption"]


def test_the_months_of_a_seasonal_change_are_shaded(tmp_path, drawn):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world())
    spec, _ = drawn["report_over_time"]
    assert all(p["shade_labels"] == SUMMER for p in spec["panels"])
    assert "The shaded months are those with the change." in rec["figures"][0]["caption"]


def test_the_months_come_from_the_change_where_the_page_drew_none(tmp_path, drawn):
    rd = study(tmp_path, CUT, panels_changed=None)
    _build(rd, what_if_plan(), world())
    spec, _ = drawn["report_over_time"]
    assert spec["panels"][0]["shade_labels"] == SUMMER


def test_a_change_in_every_month_shades_nothing_and_says_so(tmp_path, drawn):
    rd = study(tmp_path, CO2)
    rec = _build(rd, what_if_plan(), world("every"))
    for fid, (spec, _) in drawn.items():
        if fid.startswith("report_over_time"):
            assert all("shade_labels" not in p for p in spec["panels"])
    over = [f for f in rec["figures"] if f["id"].startswith("over_time")]
    assert all("The change holds in every month, so no month is shaded." in f["caption"]
               for f in over)
    assert all("shaded months are" not in f["caption"] for f in over)
    # the run changes no weather: rain, snow and air temperature have no panel
    fields = [p["field"] for f in over for p in f["panels"]]
    assert not {"RAIN", "SNOW", "TBOT"} & set(fields)


def test_months_of_the_change_outside_the_years_drawn_are_said(tmp_path, drawn):
    """A design record whose changed months all fall in other years: nothing
    is shaded, and the caption and the record both say why."""
    later = [f"2036-{m:02d}" for m in (6, 7, 8)]
    rd = study(tmp_path, CUT, panels_changed=later)
    rec = _build(rd, what_if_plan(), world())
    spec, _ = drawn["report_over_time"]
    assert all("shade_labels" not in p for p in spec["panels"])
    over = [f for f in rec["figures"] if f["id"].startswith("over_time")]
    assert over and all(
        "The months of the change fall outside these years, so no month is shaded." in
        f["caption"] for f in over)
    assert all("shaded months are" not in f["caption"] for f in over)
    assert "No month is shaded over time, because the months the change alters fall outside " \
           "2033 and 2034." in rec["not_drawn"]
    # some of them inside the years drawn: those are shaded, and nothing is said
    rd2 = study(tmp_path / "part", CUT, panels_changed=later + SUMMER[:3])
    rec2 = _build(rd2, what_if_plan(), world())
    spec, _ = drawn["report_over_time"]
    assert spec["panels"][0]["shade_labels"] == SUMMER[:3]
    assert not any("fall outside" in s for s in rec2["not_drawn"])


def test_months_the_records_do_not_give_are_not_shaded_and_it_is_said(tmp_path, drawn):
    rd = study(tmp_path, CUT)
    (rd / "01_inputs" / "run_given.json").unlink()
    rec = _build(rd, what_if_plan(), world())
    spec, _ = drawn["report_over_time"]
    assert all("shade_labels" not in p for p in spec["panels"])
    assert any("do not say which months" in s for s in rec["not_drawn"])


# ── figure 2, over space ────────────────────────────────────────────────────

def test_the_lower_row_sets_the_cells_against_the_rain_change(tmp_path, drawn):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world())
    space = rec["figures"][-1]
    assert space["id"] == "over_space" and space["file"] == "figure_over_space.png"
    assert [p["kind"] for p in space["panels"]] == ["map", "map", "cells", "cells"]
    assert [p["period"] for p in space["panels"][:2]] == ["2033-JJA", "2034-JJA"]
    assert space["panels"][2]["against"] == "RAIN+SNOW"
    assert space["caption"].startswith("(a, b) Each cell's plant growth over June to August in "
                                       "2033 and 2034, the new run minus the control run.")
    assert "(c, d) The cells of the basin's vegetated land, the same change against the change "\
           "in rain and snowfall" in space["caption"]
    assert space["comparisons"] == [3]
    spec, refs = drawn["report_over_space"]
    assert spec["layout"] == "grid" and spec["columns"] == 2
    up = refs["U1"]["value"]
    assert np.allclose(up[MASK], (-0.05 * _gradient())[MASK]) and np.allclose(up[~MASK], 0.0)
    cells = refs["C1"]["value"]
    assert len(cells["x"]) == int(MASK.sum())             # the region's cells only
    assert np.allclose(cells["c"], _gradient(12.0)[MASK] - 273.15)   # air temperature, in °C


def test_a_uniform_change_uses_the_plans_overlap_pair(tmp_path, drawn):
    pair = {"id": "P4", "fields": ["GPP", "FPG"], "regions": [REGION], "periods": ["2033"],
            "outputs": [_out({"id": "O1", "kind": "difference", "field": "GPP", "period": "2033",
                              "region": None}),
                        {"id": "O2", "kind": "map", "run": "control", "field": "FPG",
                         "period": "2033"},
                        {"id": "O3", "kind": "overlap", "region": REGION,
                         "inputs": ["O2", "O1"]}]}
    rd = study(tmp_path, CO2)
    rec = _build(rd, what_if_plan(extra_analyses=[pair]), world("every"))
    space = rec["figures"][-1]
    assert [p["kind"] for p in space["panels"]] == ["map", "map", "cells"]
    assert space["panels"][2]["against"] == "FPG"
    assert "against the control run's nitrogen limitation index" in space["caption"]
    assert 4 in space["comparisons"]
    _, refs = drawn["report_over_space"]
    c = refs["C1"]
    assert np.allclose(c["value"]["x"], _gradient(20.0)[MASK])        # the FPG map, across
    assert "minus control run" in c["out"]["y_label"]                 # plant growth's change, up


def test_with_no_pair_the_lower_row_is_left_out_and_said(tmp_path):
    rd = study(tmp_path, CO2)
    rec = _build(rd, what_if_plan(), world("every"))
    space = rec["figures"][-1]
    assert [p["kind"] for p in space["panels"]] == ["map", "map"]
    assert "over the whole year" in space["caption"]
    assert any("has no lower row, because the change is the same in every cell" in s
               for s in rec["not_drawn"])


def test_a_change_the_same_in_every_cell_is_no_lower_row(tmp_path):
    """Two kelvin added to every cell: the change of the input is the same
    everywhere (but for the rounding of stored values), so the cells are not
    set against it."""
    warm = [{"id": "weather_scale", "params": {"temperature_K": 2.0, "months": [6, 7, 8],
                                               "years": YEARS}}]
    acc = world("none")
    for (run, f, p), a in list(acc.maps.items()):
        if run == RUN and f == "TBOT" and p.endswith("JJA"):
            acc.maps[(run, f, p)] = (a + 2.0).astype(np.float32).astype(float)
        if run == RUN and f == "GPP" and p.endswith("JJA"):
            acc.maps[(run, f, p)] = a - 0.1 * _gradient()
    for lab in SUMMER:
        acc.months[(RUN, "TBOT")][lab] += 2.0
        acc.months[(RUN, "GPP")][lab] -= 1.0
    rd = study(tmp_path, warm, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), acc)
    assert [p["field"] for p in rec["figures"][0]["panels"]][:1] == ["TBOT"]
    space = rec["figures"][-1]
    assert [p["kind"] for p in space["panels"]] == ["map", "map"]
    assert any("has no lower row, because the change is the same in every cell" in s
               for s in rec["not_drawn"])


def _four_years(acc, plan):
    """The made-up maps of both runs for the four years 2031 to 2034: a
    change in every cell over the whole year, and the summer's rain cut of
    world() in its summers."""
    plan["comparisons"][0]["years"] = [2031, 2032, 2033, 2034]
    for y in (2031, 2032):
        for f in FIELDS:
            base = acc.maps.setdefault(("control", f, str(y)), _gradient())
            acc.maps[(RUN, f, str(y))] = base + 0.5 + 0.01 * _gradient()
            summer = acc.maps[("control", f, f"{y}-JJA")]
            cut = {"RAIN": -0.3 * summer, "SNOW": -0.3 * summer, "GPP": -0.05 * _gradient()}
            acc.maps[(RUN, f, f"{y}-JJA")] = summer + np.where(MASK, cut.get(f, 0.0), 0.0)
    return acc, plan


def test_four_maps_with_no_lower_row_sit_two_to_a_row(tmp_path, drawn):
    """A new run whose change is the same in every cell, with no overlap of
    the plan to use instead, has no lower row: its four maps sit two to a
    row, as a control run's do. With a lower row they stay in one row."""
    rd = study(tmp_path, CO2)
    acc, plan = _four_years(world("every"), what_if_plan())
    rec = _build(rd, plan, acc)
    space = rec["figures"][-1]
    assert [p["kind"] for p in space["panels"]] == ["map"] * 4
    spec, _ = drawn["report_over_space"]
    assert spec["columns"] == 2
    pair = {"id": "P4", "fields": ["GPP", "FPG"], "regions": [REGION], "periods": ["2033"],
            "outputs": [_out({"id": "O1", "kind": "difference", "field": "GPP", "period": "2033",
                              "region": None}),
                        {"id": "O2", "kind": "map", "run": "control", "field": "FPG",
                         "period": "2033"},
                        {"id": "O3", "kind": "overlap", "region": REGION,
                         "inputs": ["O2", "O1"]}]}
    acc, plan = _four_years(world("every"), what_if_plan(extra_analyses=[pair]))
    rec = _build(study(tmp_path / "pair", CO2), plan, acc)
    assert rec["problems"] == []
    assert [p["kind"] for p in rec["figures"][-1]["panels"]] == ["map"] * 4 + ["cells"]
    spec, _ = drawn["report_over_space"]
    assert spec["columns"] == 4
    # four summers of a rain cut: four maps over four rows of cells, all drawn
    acc, plan = _four_years(world(), what_if_plan())
    rd = study(tmp_path / "cut", CUT, panels_changed=[f"{y}-{m:02d}" for y in range(2031, 2035)
                                                      for m in (6, 7, 8)])
    rec = _build(rd, plan, acc)
    assert rec["problems"] == []
    space = rec["figures"][-1]
    assert [p["kind"] for p in space["panels"]] == ["map"] * 4 + ["cells"] * 4
    assert [p["letter"] for p in space["panels"]] == list("abcdefgh")
    spec, _ = drawn["report_over_space"]
    assert spec["columns"] == 4 and (rd / "04_analysis" / space["file"]).is_file()


def test_more_than_four_years_map_the_first_four_and_say_so(tmp_path):
    rd = study(tmp_path, CO2)
    plan = what_if_plan()
    plan["comparisons"][0]["years"] = [2031, 2032, 2033, 2034, 2035]
    acc = world("every")
    for y in (2031, 2032, 2035):                 # the five years' maps of both runs
        for f in FIELDS:
            base = acc.maps.setdefault(("control", f, str(y)), _gradient())
            acc.maps[(RUN, f, str(y))] = base + 0.5
    rec = _build(rd, plan, acc)
    space = rec["figures"][-1]
    assert [p["period"] for p in space["panels"] if p["kind"] == "map"] == \
        ["2031", "2032", "2033", "2034"]
    assert any("in 2035 is not drawn, because a figure over space holds at most 4" in s
               for s in rec["not_drawn"])


# ── a study of the control run alone ────────────────────────────────────────

def control_plan():
    return {"comparisons": [{"id": "C1", "run": "control", "against": "normal years",
                             "years": [2033, 2034]}],
            "baseline": {"normal_years": NORMAL},
            "analyses": [{"id": "P1", "fields": ["SNOW", "RAIN"], "regions": [REGION],
                          "periods": ["WY2033", "WY2034", "normal-WY"],
                          "outputs": [{"id": "O1", "kind": "anomaly", "field": "SNOW",
                                       "period": "WY2033", "region": REGION}]}]}


def test_a_study_of_the_control_run_alone(tmp_path, drawn):
    lines = [{"call": "anomaly", "planned": True,
              "keys": {"run": "control", "field": "SNOW", "period": "WY2033", "region": REGION},
              "result": {"anomaly": 12.345, "percent": 6.789, "change_units": "mm/water year"}}]
    rd = study(tmp_path, [], lookups=lines)
    (rd / "01_inputs" / "run_given.json").unlink()
    rec = _build(rd, control_plan(), world("none"))
    assert rec["problems"] == []
    over = rec["figures"][0]
    assert over["heading"] == "Each quantity month by month, against the normal years"
    assert "The grey band is the range of the normal years 2031 to 2034" in over["caption"]
    spec, refs = drawn["report_over_time"]
    p = spec["panels"][0]
    assert "normal" in p and "shade_labels" not in p
    jan = refs["M1"]["value"]["labels"].index("2033-01")
    i = FIELDS.index(refs["M1"]["out"]["field"])
    assert p["normal"]["low"][jan] == pytest.approx(10.0 + i + 1)       # 2031, the lowest
    assert p["normal"]["high"][jan] == pytest.approx(10.0 + i + 1 + 0.3)
    space = rec["figures"][1]
    assert [x["kind"] for x in space["panels"]] == ["map", "map"]
    assert [x["period"] for x in space["panels"]] == ["WY2033", "WY2034"]
    assert "minus the mean of the normal years 2031 to 2034" in space["caption"]
    _, srefs = drawn["report_over_space"]
    assert np.allclose(srefs["U1"]["value"], 0.0)       # the made-up years are alike
    assert any("no lower row, because a study of the control run alone" in s
               for s in rec["not_drawn"])
    t = rec["tables"][0]
    assert t["columns"] == ["Water year 2033"]
    assert t["rows"][0]["cells"] == [f"{format_value(12.345)} ({format_value(6.789)} %)"]
    assert t["rows"][0]["label"] == "Snowfall (SNOW), mm"


# ── the tables ──────────────────────────────────────────────────────────────

def _diff(field, period, value, percent=None, planned=True, region=REGION, units="gC/m^2/season"):
    return {"call": "difference", "planned": planned,
            "keys": {"run": RUN, "control": "control", "field": field, "period": period,
                     "region": region},
            "result": {"difference": value, "percent": percent, "change_units": units}}


def test_the_tables_print_only_planned_records(tmp_path):
    lines = [_diff("GPP", "2033-JJA", -99.0, planned=False),           # L1: not planned
             _diff("GPP", "2033-JJA", -1.2345, -4.567),                # L2
             _diff("GPP", "2033", -2.0, -1.1, units="gC/m^2/yr"),      # L3
             _diff("NPP", "2033-JJA", -0.5),                           # L4: no percent
             _diff("GPP", "2033-06", -0.7, units="gC/m^2/month"),      # L5: a month, not a column
             _diff("GPP", "2034", -3.0, region="elsewhere, vegetated"),  # L6: another region
             _diff("QCHARGE", "2033", 0.0, units="mm/yr"),             # L7: not in Figure 1
             dict(_diff("TLAI", "2033", None), result={"difference": None, "why": "none"})]
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    before = (rd / "04_analysis" / "lookups.jsonl").read_bytes()
    rec = _build(rd, what_if_plan(), world())
    assert (rd / "04_analysis" / "lookups.jsonl").read_bytes() == before
    t = rec["tables"][0]
    assert t["id"] == "periods" and t["number"] == 1 and t["follows"] == "over_time_2"
    assert t["corner"] == "Quantity" and t["columns"] == ["Summer 2033", "Year 2033"]
    rows = {r["label"]: r for r in t["rows"]}
    assert set(rows) == {"Plant growth (GPP), gC/m2", "Net plant growth (NPP), gC/m2"}
    gpp = rows["Plant growth (GPP), gC/m2"]
    assert gpp["cells"] == [f"{format_value(-1.2345)} ({format_value(-4.567)} %)",
                            f"{format_value(-2.0)} ({format_value(-1.1)} %)"]
    assert gpp["records"] == ["L2", "L3"] and gpp["panel"] == "2a"
    assert rows["Net plant growth (NPP), gC/m2"]["cells"] == [format_value(-0.5), ""]
    assert rec["printed_records"] == ["L2", "L3", "L4"]
    printed = {n for r in t["rows"] for n in r["records"]}
    assert set(rec["printed_records"]) == printed


def test_the_period_table_keeps_six_periods_with_the_most_records(tmp_path):
    periods = ["2033-DJF", "2033-MAM", "2033-JJA", "2033-SON", "2034-JJA", "2033", "2034",
               "2034-SON"]
    lines = []
    for k, p in enumerate(periods):
        for f in ("GPP", "NPP", "TLAI")[:3 if k < 6 else 1]:
            lines.append(_diff(f, p, -1.0 - k))
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    rec = _build(rd, what_if_plan(), world())
    t = rec["tables"][0]
    assert t["columns"] == ["Winter 2032 to 2033", "Spring 2033", "Summer 2033", "Autumn 2033",
                            "Summer 2034", "Year 2033"]
    assert any("so autumn 2034 and year 2034 are not in it" in s for s in rec["not_drawn"])


def test_seasons_stand_in_time_order_whatever_the_order_they_were_logged(tmp_path):
    lines = [_diff("GPP", p, -1.0 - k) for k, p in
             enumerate(["2033-SON", "2033-JJA", "2033-DJF", "2033-MAM"])]      # L1 to L4
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    t = _build(rd, what_if_plan(), world())["tables"][0]
    assert t["columns"] == ["Winter 2032 to 2033", "Spring 2033", "Summer 2033", "Autumn 2033"]
    row = t["rows"][0]
    assert row["cells"] == [format_value(v) for v in (-3.0, -4.0, -2.0, -1.0)]
    assert row["records"] == ["L1", "L2", "L3", "L4"]        # by line, not by column


def test_a_difference_from_another_run_than_the_control_is_not_printed(tmp_path):
    other = _diff("GPP", "2033-JJA", -9.0)
    other["keys"]["control"] = "a_third_run"                           # L1
    lines = [other, _diff("NPP", "2033-JJA", -0.5)]                    # L2
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    rec = _build(rd, what_if_plan(), world())
    rows = {r["label"]: r for r in rec["tables"][0]["rows"]}
    assert set(rows) == {"Net plant growth (NPP), gC/m2"}
    assert rec["printed_records"] == ["L2"]


def test_a_split_of_a_map_logged_after_it_is_named_and_not_printed(tmp_path):
    split = {"call": "elevation_split", "planned": True, "keys": {"region": REGION, "x": "h-later"},
             "result": {"parts.below 1000 m.mean": 7.0, "parts.below 1000 m.area_km2": 5.0}}
    lines = [split,                                                     # L1: before its map
             {"call": "difference", "planned": True, "arrays": {"difference": "h-later"},
              "keys": {"run": RUN, "control": "control", "field": "GPP", "period": "2033-JJA",
                       "region": None},
              "result": {"change_units": "gC/m^2/season"}},             # L2: the map
             dict(split, result={"parts.below 1000 m.mean": -1.5,
                                 "parts.below 1000 m.area_km2": 5.0})]  # L3: after it
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    rec = _build(rd, what_if_plan(), world())
    t = [x for x in rec["tables"] if x["id"] == "elevation"][0]
    assert [r["cells"] for r in t["rows"]] == [[format_value(5.0), format_value(-1.5)]]
    assert [r["records"] for r in t["rows"]] == [["L3"]]
    assert rec["printed_records"] == ["L3"]
    assert [s for s in rec["not_drawn"] if "the map it splits is not on the record" in s] == \
        ["One elevation split is not in the table by elevation, because the map it splits is "
         "not on the record."]


def test_the_elevation_table_names_each_split_by_the_map_it_splits(tmp_path):
    lines = [{"call": "difference", "planned": True, "arrays": {"difference": "h-made-up"},
              "keys": {"run": RUN, "control": "control", "field": "GPP", "period": "2033-JJA",
                       "region": None},
              "result": {"change_units": "gC/m^2/season"}},
             {"call": "elevation_split", "planned": True,
              "keys": {"region": REGION, "x": "h-made-up"},
              "result": {"parts.below 1000 m.mean": -1.234, "parts.below 1000 m.area_km2": 56.78,
                         "parts.1000-1500 m.mean": -3.5, "parts.1000-1500 m.area_km2": 12.0,
                         "other_regions.x.mean": 9.0}},
             {"call": "elevation_split", "planned": True,
              "keys": {"region": REGION, "x": "h-not-on-record"},
              "result": {"parts.below 1000 m.mean": 7.0}}]
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=lines)
    rec = _build(rd, what_if_plan(), world())
    t = [x for x in rec["tables"] if x["id"] == "elevation"][0]
    assert t["follows"] == "over_space" and t["corner"] == "Elevation band"
    assert t["columns"] == ["Area, km2", "Summer 2033, gC/m2"]
    assert [r["label"] for r in t["rows"]] == ["Below 1000 m", "1000 to 1500 m"]
    assert t["rows"][0]["cells"] == [format_value(56.78), format_value(-1.234)]
    assert t["rows"][0]["records"] == ["L2"]
    assert all("panel" not in r for r in t["rows"])      # a band is no panel of a figure
    assert "L2" in rec["printed_records"] and "L3" not in rec["printed_records"]
    assert t["caption"].startswith("Plant growth (GPP), the new run minus the control run, in "
                                   "each elevation band")
    assert any("the map it splits is not on the record" in s for s in rec["not_drawn"])


# ── a second figure over space ──────────────────────────────────────────────

def mapping(pid, field, against=None, kind="difference", period="2033-JJA", split=True):
    """A made-up analysis that maps one quantity's change in each cell (a
    "difference", or an "anomaly" for the control run alone), splits it by
    elevation band and, with `against`, sets it against the control run's
    map of another quantity, cell by cell."""
    outs = [{"id": "O1", "kind": kind, "run": RUN if kind == "difference" else "control",
             "field": field, "period": period, "region": None}]
    if split:
        outs.append({"id": "O2", "kind": "elevation_split", "region": REGION, "inputs": ["O1"]})
    if against:
        outs += [{"id": "O3", "kind": "map", "run": "control", "field": against,
                  "period": period},
                 {"id": "O4", "kind": "overlap", "region": REGION, "inputs": ["O1", "O3"]}]
    return {"id": pid, "fields": [field] + ([against] if against else []), "regions": [REGION],
            "periods": [period], "outputs": outs}


def _water_use(acc):
    """world()'s new run with a made-up summer change of transpiration that
    differs from cell to cell."""
    for y in YEARS:
        p = f"{y}-JJA"
        acc.maps[(RUN, "QVEGT", p)] = acc.maps[("control", "QVEGT", p)] + \
            np.where(MASK, -0.02 * _gradient(), 0.0)
    return acc


def _spaces(rec):
    return [f for f in rec["figures"] if f["id"].startswith("over_space")]


def test_a_plan_that_maps_two_changes_gets_two_figures_over_space(tmp_path, drawn):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(extra_analyses=[mapping("P4", "QVEGT")]), _water_use(world()))
    assert rec["problems"] == []
    assert [f["id"] for f in rec["figures"]] == ["over_time", "over_time_2", "over_space",
                                                 "over_space_2"]
    first, second = _spaces(rec)
    assert (first["file"], second["file"]) == ("figure_over_space.png", "figure_over_space_2.png")
    assert (first["number"], second["number"]) == (3, 4)
    assert (first["heading"], second["heading"]) == ("Plant growth in each cell",
                                                     "Transpiration in each cell")
    assert [p["field"] for p in second["panels"]] == ["QVEGT"] * 4
    assert [p["kind"] for p in second["panels"]] == ["map", "map", "cells", "cells"]
    assert [p["letter"] for p in second["panels"]] == ["a", "b", "c", "d"]   # its own letters
    assert second["caption"].startswith("(a, b) Each cell's transpiration over June to August "
                                        "in 2033 and 2034, the new run minus the control run.")
    assert "(c, d) The cells of the basin's vegetated land, the same change against the change "\
           "in rain and snowfall" in second["caption"]
    # the first answers every elevation split, as it did alone; the second the
    # comparisons that map or split its own quantity (not P2's regional one)
    assert first["comparisons"] == [3, 4] and second["comparisons"] == [4]
    assert all((rd / "04_analysis" / f["file"]).is_file() for f in (first, second))
    spec, refs = drawn["report_over_space_2"]
    assert spec["layout"] == "grid" and spec["columns"] == 2
    assert np.allclose(refs["U1"]["value"][MASK], (-0.02 * _gradient())[MASK])
    assert np.allclose(refs["C1"]["value"]["y"], (-0.02 * _gradient())[MASK])
    assert not any("not mapped" in s for s in rec["not_drawn"])


def test_a_plan_that_maps_only_the_main_change_gets_todays_record(tmp_path):
    """Maps that are no change of another quantity draw no second figure: a
    run's own state (a "map"), a change over a region, a change with no
    period and a change no split or overlap takes. The record is the one
    the module gave before a second figure was drawn."""
    pair = mapping("P4", "GPP", against="FPG", period="2033", split=False)
    others = {"id": "P5", "fields": ["TLAI", "QVEGT", "BTRAN", "NPP"], "regions": [REGION],
              "periods": ["2033"],
              "outputs": [_out({"id": "O1", "kind": "difference", "field": "TLAI",
                                "period": "2033", "region": None}),
                          _out({"id": "O2", "kind": "difference", "field": "QVEGT",
                                "period": "2033"}),
                          {"id": "O3", "kind": "map", "run": "control", "field": "BTRAN",
                           "period": "2033"},
                          {"id": "O4", "kind": "overlap", "region": REGION,
                           "inputs": ["O2", "O3"]},
                          _out({"id": "O5", "kind": "difference", "field": "NPP",
                                "region": None}),
                          {"id": "O6", "kind": "elevation_split", "region": REGION,
                           "inputs": ["O5"]}]}
    rd = study(tmp_path, CO2)
    rec = _build(rd, what_if_plan(extra_analyses=[pair, others]), world("every"))
    assert rec["problems"] == [] and rec["tables"] == []
    assert [f["id"] for f in rec["figures"]] == ["over_time", "over_space"]
    words = "plant growth"
    assert rec["figures"][-1] == {
        "id": "over_space", "file": "figure_over_space.png", "number": 2,
        "heading": "Plant growth in each cell",
        "caption": "(a, b) Each cell's plant growth over the whole year in 2033 and 2034, the "
                   "new run minus the control run. The line is the edge of the basin's "
                   "vegetated land. (c) The cells of the basin's vegetated land, the change in "
                   "plant growth against the control run's nitrogen limitation index over the "
                   "whole year, coloured by the control run's air temperature.",
        "comparisons": [3, 4, 5],
        "panels": [{"field": "GPP", "words": words, "period": "2033", "kind": "map",
                    "letter": "a"},
                   {"field": "GPP", "words": words, "period": "2034", "kind": "map",
                    "letter": "b"},
                   {"field": "GPP", "words": words, "against": "FPG", "kind": "cells",
                    "letter": "c"}]}
    assert rec["not_drawn"] == [
        "Air temperature (TBOT) has no panel over time, because the new run does not change it.",
        "Snowfall (SNOW) has no panel over time, because the new run does not change it.",
        "Rain (RAIN) has no panel over time, because the new run does not change it.",
        "Aquifer recharge (QCHARGE) has no panel over time, because the two runs hold the same "
        "value of it in every month."]
    assert not (rd / "04_analysis" / "figure_over_space_2.png").exists()
    # with no lower row, the one figure's sentence still names it as before
    rec = _build(study(tmp_path / "alone", CO2), what_if_plan(), world("every"))
    assert "The figure over space has no lower row, because the change is the same in every "\
           "cell and no approved comparison sets another map against plant growth." \
           in rec["not_drawn"]


def test_a_third_mapped_change_is_named_and_not_drawn(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    plan = what_if_plan(extra_analyses=[mapping("P4", "QVEGT"), mapping("P5", "TLAI")])
    rec = _build(rd, plan, _water_use(world()))
    assert rec["problems"] == []
    assert [f["id"] for f in _spaces(rec)] == ["over_space", "over_space_2"]
    assert {p["field"] for f in _spaces(rec) for p in f["panels"]} == {"GPP", "QVEGT"}
    assert "Leaf area index (TLAI) is not mapped, because the report maps at most two "\
           "quantities." in rec["not_drawn"]
    assert sorted(p.name for p in (rd / "04_analysis").glob("figure_over_space*")) == \
        ["figure_over_space.png", "figure_over_space_2.png"]


def test_a_study_of_the_control_run_alone_maps_a_second_departure(tmp_path, drawn):
    plan = control_plan()
    # rain's departure, split and set against the control run's own leaf
    # area, which is a state and no departure
    plan["analyses"] += [mapping("P2", "RAIN", against="TLAI", kind="anomaly",
                                 period="WY2033")]
    rd = study(tmp_path, [])
    (rd / "01_inputs" / "run_given.json").unlink()
    rec = _build(rd, plan, world("none"))
    assert rec["problems"] == []
    first, second = _spaces(rec)
    assert (second["id"], second["file"], second["number"]) == \
        ("over_space_2", "figure_over_space_2.png", first["number"] + 1)
    assert second["heading"] == "Rain in each cell"
    assert [(p["period"], p["letter"]) for p in second["panels"]] == [("WY2033", "a"),
                                                                       ("WY2034", "b")]
    assert second["caption"].startswith("(a, b) Each cell's rain over each water year (October "
                                        "to September) of 2033 and 2034, minus the mean of the "
                                        "normal years 2031 to 2034.")
    assert second["comparisons"] == [2]
    assert [s for s in rec["not_drawn"] if "no lower row" in s] == [
        "The figures over space have no lower row, because a study of the control run alone "
        "has no change to set the cells against."]
    assert not any("not mapped" in s for s in rec["not_drawn"])
    _, refs = drawn["report_over_space_2"]
    assert refs["U1"]["out"]["field"] == "RAIN"


def test_a_uniform_change_uses_the_second_quantitys_own_overlap_pair(tmp_path, drawn):
    plan = what_if_plan(extra_analyses=[
        mapping("P4", "GPP", against="FPG", period="2033", split=False),
        mapping("P5", "QVEGT", against="BTRAN", period="2033")])
    rec = _build(study(tmp_path, CO2), plan, world("every"))
    assert rec["problems"] == []
    first, second = _spaces(rec)
    assert [p["against"] for p in first["panels"] if p["kind"] == "cells"] == ["FPG"]
    assert [p["against"] for p in second["panels"] if p["kind"] == "cells"] == ["BTRAN"]
    assert "(c) The cells of the basin's vegetated land, the change in transpiration against "\
           "the control run's water stress index over the whole year" in second["caption"]
    assert first["comparisons"] == [3, 4, 5] and second["comparisons"] == [5]
    _, refs = drawn["report_over_space_2"]
    i = FIELDS.index("BTRAN")
    assert np.allclose(refs["C1"]["value"]["x"], _gradient(10.0 + i)[MASK])   # BTRAN, across
    assert refs["C1"]["out"]["y_label"].startswith("Transpiration, new run minus")


def _map_line(field, h, units):
    """The made-up log line of a map of a change, found by its hash."""
    return {"call": "difference", "planned": True, "arrays": {"difference": h},
            "keys": {"run": RUN, "control": "control", "field": field, "period": "2033-JJA",
                     "region": None},
            "result": {"change_units": units}}


def _split_line(h, mean):
    return {"call": "elevation_split", "planned": True, "keys": {"region": REGION, "x": h},
            "result": {"parts.below 1000 m.mean": mean, "parts.below 1000 m.area_km2": 5.0}}


SPLIT_LINES = [_diff("GPP", "2033-JJA", -1.2, -4.5),                       # L1: by period
               _map_line("GPP", "h-g", "gC/m^2/season"),                    # L2
               _map_line("QVEGT", "h-q", "mm/season"),                      # L3
               _map_line("TLAI", "h-t", "1"),                               # L4
               _split_line("h-q", -0.25),                                   # L5: transpiration
               _split_line("h-g", -1.5),                                    # L6: plant growth
               _split_line("h-t", -0.01)]                                   # L7: no figure


def test_the_elevation_tables_split_by_quantity_each_after_its_own_figure(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=SPLIT_LINES)
    rec = _build(rd, what_if_plan(extra_analyses=[mapping("P4", "QVEGT")]), _water_use(world()))
    assert rec["problems"] == []
    assert [(t["id"], t["number"], t["follows"]) for t in rec["tables"]] == [
        ("periods", 1, "over_time_2"), ("elevation", 2, "over_space"),
        ("elevation_2", 3, "over_space_2")]
    first, second = rec["tables"][1:]
    # plant growth's split, and leaf area's, a quantity with no figure of its own
    assert first["columns"] == ["Area, km2", "Plant growth (GPP), summer 2033, gC/m2",
                                "Leaf area index (TLAI), summer 2033"]
    assert [r["records"] for r in first["rows"]] == [["L6", "L7"]]
    assert second["columns"] == ["Area, km2", "Summer 2033, mm"]
    assert [r["cells"] for r in second["rows"]] == [[format_value(5.0), format_value(-0.25)]]
    assert [r["records"] for r in second["rows"]] == [["L5"]]
    assert second["caption"].startswith("Transpiration (QVEGT), the new run minus the control "
                                        "run, in each elevation band")
    assert rec["printed_records"] == ["L1", "L5", "L6", "L7"]
    # with no second figure, every split stays in the one table after the first
    rec = _build(study(tmp_path / "one", CUT, panels_changed=SUMMER, lookups=SPLIT_LINES),
                 what_if_plan(), _water_use(world()))
    assert [(t["id"], t["number"], t["follows"]) for t in rec["tables"]] == [
        ("periods", 1, "over_time_2"), ("elevation", 2, "over_space")]
    assert [r["records"] for r in rec["tables"][1]["rows"]] == [["L5", "L6", "L7"]]


def test_the_report_page_places_figure_4_and_its_table_in_order(tmp_path):
    from agents.analysis import report_pages as RP
    from test_report_pages import make_report
    rd = study(tmp_path, CUT, panels_changed=SUMMER, lookups=SPLIT_LINES)
    rec = _build(rd, what_if_plan(extra_analyses=[mapping("P4", "QVEGT")]), _water_use(world()))
    report, adir = make_report(tmp_path / "page", figures=rec)
    written = RP.write_pages(report, adir)
    text = Path(written["REPORT.md"]).read_text()
    marks = ["### 3.3 Plant growth in each cell", "**Figure 3.**", "**Table 2.**",
             "### 3.4 Transpiration in each cell", "![Figure 4](figure_over_space_2.png)",
             "**Figure 4.**", "This figure answers comparison 4.", "**Table 3.**", "## 4."]
    at = [text.index(m) for m in marks]
    assert at == sorted(at)
    assert text.count("**Table 3.**") == 1 and "**Table 4.**" not in text
    details = Path(written["REPORT_DETAILS.md"]).read_text()
    assert "- Figure 4 (over_space_2): figure_over_space_2.png, comparisons 4" in details
    assert "- Table 3, after the figure over_space_2, 1 row:" in details


# ── the record, and what is never done ──────────────────────────────────────

def test_the_record_is_written_beside_the_figures(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), world())
    on_disk = json.loads((rd / "04_analysis" / RF.RECORD_FILE).read_text())
    assert on_disk == json.loads(json.dumps(rec))
    assert set(rec) == {"figures", "tables", "printed_records", "not_drawn", "problems"}
    for f in rec["figures"]:
        assert set(f) >= {"id", "file", "number", "heading", "caption", "comparisons", "panels"}
        for p in f["panels"]:
            assert set(p) >= {"letter", "field", "words"}


FRAMEWORK_WORDS = re.compile(r"\bplan(?:s|ned|'s)?\b|\bcomparisons?\b", re.I)


def test_no_caption_or_heading_names_the_plan_or_a_comparison(tmp_path):
    """The comparisons a figure answers are in its "comparisons" list, which
    the page states under the caption; a caption or a heading names neither
    the plan nor a comparison, and no sentence of what was not drawn names
    the plan. Every kind of figure and table, and every lower row."""
    pair = {"id": "P4", "fields": ["GPP", "FPG"], "regions": [REGION], "periods": ["2033"],
            "outputs": [_out({"id": "O1", "kind": "difference", "field": "GPP", "period": "2033",
                              "region": None}),
                        {"id": "O2", "kind": "map", "run": "control", "field": "FPG",
                         "period": "2033"},
                        {"id": "O3", "kind": "overlap", "region": REGION,
                         "inputs": ["O2", "O1"]}]}
    lines = [_diff("GPP", "2033-JJA", -1.2, -4.5),
             {"call": "anomaly", "planned": True,
              "keys": {"run": "control", "field": "SNOW", "period": "WY2033", "region": REGION},
              "result": {"anomaly": 1.5, "percent": 2.5, "change_units": "mm/water year"}}]
    cases = [(CUT, SUMMER, what_if_plan(), world()),                        # cells against rain
             (CO2, None, what_if_plan(extra_analyses=[pair]), world("every")),  # the overlap
             (CO2, None, what_if_plan(), world("every")),                   # no lower row
             ([], None, control_plan(), world("none")),                     # the control alone
             (CUT, None, {"analyses": [{"id": "P1", "fields": []}]}, world()),  # nothing to draw
             (CUT, SUMMER, what_if_plan(extra_analyses=[                    # three mapped
                 mapping("P4", "QVEGT"), mapping("P5", "TLAI")]), _water_use(world())),
             (CO2, None, what_if_plan(extra_analyses=[                      # two, no lower row
                 mapping("P4", "QVEGT", period="2033")]), world("every"))]
    seen = 0
    for k, (changes, changed, plan, acc) in enumerate(cases):
        rd = study(tmp_path / str(k), changes, panels_changed=changed, lookups=lines)
        rec = _build(rd, plan, acc)
        texts = [f["caption"] for f in rec["figures"]] + [f["heading"] for f in rec["figures"]]
        texts += [t["caption"] for t in rec["tables"]] + [c for t in rec["tables"]
                                                          for c in t["columns"]]
        for x in texts:
            assert not FRAMEWORK_WORDS.search(x), (k, x)
        for x in rec["not_drawn"]:
            assert not re.search(r"\bplan(?:s|ned|'s)?\b", x, re.I), (k, x)
        seen += len(texts) + len(rec["not_drawn"])
    assert seen > 20


def test_a_rebuild_removes_its_own_old_figures_and_nothing_else(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    out = rd / "04_analysis"
    (out / "figure_over_time_3.png").write_bytes(b"old")
    (out / "model_figure.png").write_bytes(b"the model's own")
    _build(rd, what_if_plan(), world())
    assert not (out / "figure_over_time_3.png").exists()
    assert (out / "model_figure.png").read_bytes() == b"the model's own"


class Broken(FakeAccess):
    def get_series(self, *a, **k):
        raise RuntimeError("the store is unreadable")

    def lat_lon(self):
        raise RuntimeError("the store is unreadable")


def test_build_never_raises(tmp_path):
    rd = study(tmp_path, CUT, panels_changed=SUMMER)
    rec = _build(rd, what_if_plan(), Broken({}, {}))
    assert rec["figures"] == [] and rec["problems"]
    assert json.loads((rd / "04_analysis" / RF.RECORD_FILE).read_text()) == rec
    for plan in (None, {}, {"analyses": "not a list"}):
        rec = RF.build(rd, plan, rd / "04_analysis", access=world())
        assert rec["figures"] == [] and rec["not_drawn"]
    # no access given and no study records to build one from: said, not raised
    bare = tmp_path / "bare"
    rec = RF.build(bare, what_if_plan(), bare / "04_analysis")
    assert rec["figures"] == [] and (rec["problems"] or rec["not_drawn"])


def test_the_weather_a_change_alters_is_the_servers_word():
    AP._elm()
    import domain_given as DG
    assert set(RF.WEATHER_KNOBS) == set(DG.KNOB_ORDER)
    assert RF.altered_inputs(CUT) == ["RAIN", "SNOW"]
    warm = [{"id": "weather_scale", "params": {"temperature_K": 2.0}}]
    assert RF.altered_inputs(warm) == ["TBOT"]
    both = [{"id": "weather_scale", "params": {"temperature_K": -1.0, "precip_factor": 1.2}}]
    assert RF.altered_inputs(both) == ["RAIN", "SNOW", "TBOT"]
    assert RF.altered_inputs(CO2) == []
    assert RF.altered_inputs([{"id": "fire_off", "params": {}}]) == []
    assert set(RF.altered_inputs([{"id": "weather_repeat_year", "params": {}}])) == \
        {f for g in RF.INPUTS for f in g}
    assert RF.altered_inputs([{"id": "no_such_change"}]) is None
    assert RF.altered_inputs(None) is None


def test_the_plan_names_the_region_years_and_run():
    plan = what_if_plan()
    assert RF.main_region(plan) == REGION
    assert RF.study_years(plan) == YEARS
    assert RF.new_run(plan) == RUN
    assert RF.new_run(control_plan()) is None
    assert RF.study_years({"analyses": [{"periods": ["2031-JJA", "WY2032", "normal"]}]}) == \
        [2031, 2032]


def test_the_gate_allows_only_the_four_calls():
    for call in RF.CALLS:
        marks = RF._gate({"call": call})
        assert marks["planned"] is False and marks["plan_items"] == []
    for call in ("anomaly", "difference", "screen", "overlap", "elevation_split"):
        with pytest.raises(PermissionError):
            RF._gate({"call": call})


# ── on real made-up stores, through the server's access ─────────────────────

@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    import domain_suite as S
    from test_domain_suite import FIELDS as STORE_FIELDS, make_control
    from test_suite_numbers_check import _scale_gpp
    root = tmp_path_factory.mktemp("report_figures")
    years = [2015, 2016]
    control, _ = make_control(root / "ctl", years)
    S.build(control, root / "suite", fields=STORE_FIELDS)
    warm, _ = make_control(root / "warm", years)
    _scale_gpp(warm["run_dir"], 1.1)
    S.build(control, root / "suite_warm", run_dir=warm["run_dir"], fields=STORE_FIELDS)
    return {"control": str(root / "suite"), "warm": str(root / "suite_warm")}


def test_real_stores_through_the_modules_own_gate_and_log(tmp_path, stores):
    plan = {"comparisons": [{"id": "C1", "run": "warm", "against": "control", "years": [2016]}],
            "baseline": {"normal_years": [2015, 2016]},
            "analyses": [{"id": "P1", "fields": ["GPP", "QCHARGE"], "regions": ["box, vegetated"],
                          "periods": ["2016"], "outputs": []}]}
    rd = tmp_path / "study"
    out = rd / "04_analysis"
    out.mkdir(parents=True)
    held = '{"call": "difference", "planned": true, "keys": {}}\n'
    (out / AP.LOOKUPS_FILE).write_text(held)
    rec = RF.build(rd, plan, out, suites=stores)
    assert (out / AP.LOOKUPS_FILE).read_text() == held              # never written
    assert rec["problems"] == [], rec["problems"]
    assert [f["id"] for f in rec["figures"]] == ["over_time", "over_space"]
    assert [p["field"] for p in rec["figures"][0]["panels"]] == ["GPP"]
    assert any(s.startswith("Aquifer recharge (QCHARGE) has no panel") for s in rec["not_drawn"])
    for f in rec["figures"]:
        assert (out / f["file"]).is_file()
    log = [json.loads(x) for x in (out / RF.LOG_FILE).read_text().splitlines() if x.strip()]
    assert log and all(x["script"] == RF.SCRIPT and x["call"] in RF.CALLS for x in log)
    assert all(x["planned"] is False for x in log)
    answered = [x for x in log if "sha256" in x]
    assert {x["call"] for x in answered} == set(RF.CALLS)
    assert not (out / "screen_record.json").exists()
    assert not (out / AP.CONTRACT_READS_FILE).exists()
