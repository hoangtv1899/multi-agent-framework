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
    assert "L2" in rec["printed_records"] and "L3" not in rec["printed_records"]
    assert t["caption"].startswith("Plant growth (GPP), the new run minus the control run, in "
                                   "each elevation band")
    assert any("the map it splits is not on the record" in s for s in rec["not_drawn"])


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
