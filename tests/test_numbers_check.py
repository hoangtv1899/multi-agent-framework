"""Every regional number made twice; every claim value traced to them.

    pytest tests/test_numbers_check.py -v
    pytest tests/test_numbers_check.py --runlocaldata -v   # + the demo runs on disk

A synthetic experiment (the run-gate fixture's case: daily, 3-hourly and
monthly history, a surface file, a polygon and a gridded-field region): the
server's domain_analysis makes the regional table, and the framework's
numbers_check makes every number again by its own path. A change of one number
by a little, of one mask by one row, or of one record must fail the check.
"""
import copy
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_analysis as A
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
from agents.analysis import claim_trace as T
from agents.analysis import numbers_check as N

YEARS = [2015, 2016]
VARS = ["GPP", "TSA", "FAREA_BURNED", "QVEGT", "QSOIL", "H2OSNO"]


def _case(root, name, gpp_scale=1.0):
    c = cf.write_case(root / name)
    fx.write_monthly(c["run_dir"], YEARS, case="synth",
                     variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    if gpp_scale != 1.0:
        for p in Path(c["run_dir"]).glob("synth.elm.h0.*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["GPP"][:] = d["GPP"][:] * gpp_scale
    return c


def _polygon(path, c, rows=2):
    lat, lon = c["grid"]["lat"], c["grid"]["lon"] - 360.0
    d = (lat[1] - lat[0]) / 2
    ring = [[float(lon[0] - d), float(lat[0] - d)], [float(lon[-1] + d), float(lat[0] - d)],
            [float(lon[-1] + d), float(lat[rows - 1] + d)],
            [float(lon[0] - d), float(lat[rows - 1] + d)], [float(lon[0] - d), float(lat[0] - d)]]
    path.write_text(json.dumps({"type": "Feature", "geometry": {
        "type": "Polygon", "coordinates": [ring]}}))
    return path


def _burn(path, c):
    g = c["grid"]
    burn = np.zeros(g["shape"], dtype=np.float32)
    burn[1, 2], burn[2, 3], burn[3, 4] = 0.5, 0.25, 0.1
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("lat", len(g["lat"]))
        d.createDimension("lon", len(g["lon"]))
        d.createDimension("year", 2)
        d.createVariable("lat", "f8", ("lat",))[:] = g["lat"]
        d.createVariable("lon", "f8", ("lon",))[:] = g["lon"] - 360.0
        d.createVariable("year", "i4", ("year",))[:] = YEARS
        d.createVariable("burned_fraction_year", "f4", ("year", "lat", "lon"))[:] = \
            np.stack([np.zeros_like(burn), burn])
    return path


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A control and a twin with 10 % more GPP, the server's table of them over
    every kind of region, and a run folder that holds it as the manager does."""
    root = tmp_path_factory.mktemp("nc")
    ctl, twin = _case(root, "ctl"), _case(root, "twin", 1.1)
    registry = {"case_dir": str(ctl["case_dir"]),
                "regions": {"box": {"polygon": str(_polygon(root / "box.geojson", ctl))},
                            "burned": {"field": str(_burn(root / "burn.nc", ctl)),
                                       "var": "burned_fraction_year", "op": ">",
                                       "value": 0.0, "over": "any"}}}
    analysis = {"regions": [A.VEGETATED, A.BANDS, "box", "burned",
                            {"name": A.BANDS, "within": "box"}],
                "variables": VARS, "normal_years": [2016]}
    table = A.experiment({"case_dir": str(ctl["case_dir"]), "regions": registry["regions"]},
                         {"control": ctl["run_dir"], "twin": twin["run_dir"]}, analysis,
                         YEARS, confined={"twin": "box"})
    assert table["problems"] == [], table["problems"]
    table = json.loads(json.dumps(table, default=str))      # as the manager writes it
    return {"root": root, "ctl": ctl, "twin": twin, "registry": registry, "table": table}


def _run_folder(base, made, table=None, records=None):
    """A run folder as the manager leaves it: the table, the control's record
    and, for each executed run, the year checks its chain wrote (each naming
    the run directory it checked); `records` overrides {run: run_dir}."""
    rd = base / "run"
    (rd / "03_results").mkdir(parents=True)
    (rd / "01_inputs").mkdir()
    (rd / "03_results" / "regional.json").write_text(
        json.dumps(table if table is not None else made["table"], indent=1))
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "reuse": True,
                     "case_dir": str(made["ctl"]["case_dir"]),
                     "run_dir": str(made["ctl"]["run_dir"])}}))
    for run, d in (records if records is not None else
                   {"twin": made["twin"]["run_dir"]}).items():
        ck = rd / "domain_case" / run / "checks"
        ck.mkdir(parents=True)
        for y in YEARS:
            (ck / f"y{y}_1.json").write_text(json.dumps({"run_dir": str(d), "pass": True}))
    return rd


# ── the recomputation ───────────────────────────────────────────────────────
class TestTheServersTableReproduces:

    def test_every_number_agrees(self, made, tmp_path):
        rd = _run_folder(tmp_path, made)
        rec = N.check(rd, registry=made["registry"])
        assert rec["pass"], rec["mismatches"][:5] or rec.get("error")
        assert rec["n_mismatched"] == 0 and rec.get("error") is None
        # far inside the tolerance: both paths reduce the same float32 values
        assert rec["largest_relative_difference"]["rel_diff"] < 1e-9
        assert rec["not_checked"] == [] and rec["notes"] == []
        saved = json.loads((rd / "04_analysis" / N.CHECK_JSON).read_text())
        assert saved["pass"] and saved["n_compared"] == rec["n_compared"]
        assert saved["tolerance"]["relative"] == N.RTOL and "method" in saved

    def test_every_section_is_counted(self, made, tmp_path):
        """regions (cells and area), each run's values (burned area among
        them), the differences, the event and the changed share."""
        t = made["table"]
        regions = list(t["regions"])
        assert {"box, vegetated", "burned, vegetated"} <= set(regions)
        assert any(r.startswith("box, vegetated, ") for r in regions)
        assert any(N.band_rule(r) for r in regions)
        n_values = sum(1 for r in t["runs"].values() for f in r["values"].values()
                       for ys in f.values() for v in ys.values() if v is not None)
        n_diff = sum(1 for r in t["differences"].values() for f in r.values()
                     for ys in f.values() for _ in ys.values())
        rec = N.check(_run_folder(tmp_path, made), registry=made["registry"], write=False)
        assert n_values and n_diff and t["event"]["anomalies"] and t["changed_share"]
        assert rec["n_compared"] > 2 * len(regions) + n_values + n_diff
        assert "burned_area_km2" in t["runs"]["control"]["values"][A.VEGETATED]

    def test_the_daily_and_three_hourly_streams_weigh_a_year_alike(self, made):
        """TSA is the same formula in the daily and the 3-hourly stream, each
        written in ELM's layout (the year's last day in the next file, a start
        step before the first): their annual means must agree."""
        h = N.History(made["ctl"]["run_dir"])
        g = h.grid()
        cells = np.arange(g["weight"].size)
        W = g["weight"].ravel()[:, None]
        for y in YEARS:
            out = {}
            for st in ("h1", "h2"):
                m, days, why = h.year(st, "TSA", y, W, cells)
                assert why is None and days.sum() == pytest.approx(365.0)
                out[st] = float((m[:, 0] * days).sum() / days.sum())
            assert out["h1"] == pytest.approx(out["h2"], rel=1e-9)

    def test_a_year_missing_its_last_day_is_not_whole(self, made, tmp_path):
        run = tmp_path / "cut"
        shutil.copytree(made["ctl"]["run_dir"], run)
        (run / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        h = N.History(run)
        g = h.grid()
        m, _, why = h.year("h1", "QVEGT", 2016, g["weight"].ravel()[:, None],
                           np.arange(g["weight"].size))
        assert m is None and "end at day 364" in why


class TestASmallChangeFails:

    # Every number, a difference, an anomaly and a percent among them, is held
    # to RTOL of itself (2026-11-03): each is moved by 1e-4 of itself. The
    # fixture's 2015 is 0.2 % below its normal, so its anomaly and percent are
    # small numbers made from large ones, the case the operands' scale hid.
    @pytest.mark.parametrize("where,by", [
        (("runs", "control", "values", A.VEGETATED, "GPP", "2015"), 1e-4),
        (("runs", "twin", "values", "box, vegetated", "H2OSNO", "2016"), 1e-4),
        (("runs", "control", "values", "burned, vegetated", "burned_area_km2", "2016"), 1e-4),
        (("differences", "twin", "box, vegetated, 1000-1500 m", "GPP", "2015"), 1e-4),
        (("event", "anomalies", A.VEGETATED, "GPP", "years", "2015", "anomaly_pct"), 1e-4),
        (("event", "anomalies", A.VEGETATED, "GPP", "years", "2015", "anomaly"), 1e-4),
        (("event", "anomalies", "box, vegetated", "TSA", "normal"), 1e-4),
        (("regions", "box, vegetated", "area_km2"), 1e-4),
    ])
    def test_one_number_changed_by_a_little(self, made, tmp_path, where, by):
        t = copy.deepcopy(made["table"])
        d = t
        for k in where[:-1]:
            d = d[k]
        d[where[-1]] = d[where[-1]] * (1.0 + by)
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"])
        assert not rec["pass"] and rec["n_mismatched"] == 1, rec["mismatches"]
        bad = rec["mismatches"][0]
        assert N.RTOL < bad["rel_diff"] < 2 * by
        assert bad["section"] == {"runs": "values"}.get(where[0], where[0])
        assert bad.get("region") in where

    def test_a_small_difference_of_large_values(self, made, tmp_path_factory):
        """A twin 0.05 % above the control: its GPP difference is 1/2000 of the
        values it subtracts. Moved by 1e-3 of itself (5e-7 of those values,
        inside RTOL of them) it must fail: the difference is held to itself."""
        root = tmp_path_factory.mktemp("near")
        near = _case(root, "near", 1.0005)
        t = A.experiment({"case_dir": str(made["ctl"]["case_dir"]),
                          "regions": made["registry"]["regions"]},
                         {"control": made["ctl"]["run_dir"], "near": near["run_dir"]},
                         {"regions": [A.VEGETATED], "variables": ["GPP"]}, YEARS)
        t = json.loads(json.dumps(t, default=str))
        d = t["differences"]["near"][A.VEGETATED]["GPP"]
        big = t["runs"]["control"]["values"][A.VEGETATED]["GPP"]["2015"]
        assert 0 < abs(d["2015"]) < 1e-3 * big
        rd = _run_folder(root, made, t, records={"near": near["run_dir"]})
        assert N.check(rd, registry=made["registry"], write=False)["pass"]
        d["2015"] *= 1.0 + 1e-3
        (rd / "03_results" / "regional.json").write_text(json.dumps(t))
        rec = N.check(rd, registry=made["registry"], write=False)
        assert not rec["pass"]
        assert [(b["section"], b["year"]) for b in rec["mismatches"]] == \
            [("differences", "2015")]
        assert rec["mismatches"][0]["abs_diff"] < N.RTOL * big

    def test_below_the_tolerance_it_passes(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["runs"]["control"]["values"][A.VEGETATED]["GPP"]["2015"] *= 1.0 + 1e-7
        assert N.check(_run_folder(tmp_path, made, t), registry=made["registry"],
                       write=False)["pass"]

    def test_a_cell_count_one_off(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["regions"]["box, vegetated"]["cells"] += 1
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert [b["what"] for b in rec["mismatches"]] == ["cells"]

    def test_a_changed_share(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["changed_share"]["twin"]["share"][A.VEGETATED] += 0.01
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert [b["section"] for b in rec["mismatches"]] == ["changed_share"]

    def test_a_region_drawn_one_row_larger(self, made, tmp_path):
        """The registry's polygon one row larger than the one the table was made
        with: the cells, the area and every number of that region differ."""
        reg = copy.deepcopy(made["registry"])
        reg["regions"]["box"]["polygon"] = str(_polygon(tmp_path / "big.geojson",
                                                        made["ctl"], rows=3))
        rec = N.check(_run_folder(tmp_path, made), registry=reg, write=False)
        assert not rec["pass"]
        assert {"box, vegetated"} <= {b.get("region") for b in rec["mismatches"]}

    def test_a_total_made_as_a_mean(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["runs"]["control"]["fields"]["QVEGT"]["how"] = "mean"
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert any(b["section"] == "fields" and b["field"] == "QVEGT"
                   for b in rec["mismatches"])

    def test_a_table_read_from_the_wrong_runs_history(self, made, tmp_path):
        """The twin's numbers made from the control's history, and the table
        naming the control's directory for it: the check must not read the
        directory from the table it checks (2026-11-03)."""
        t = copy.deepcopy(made["table"])
        t["runs"]["twin"] = copy.deepcopy(t["runs"]["control"])
        for reg in t["differences"]["twin"].values():
            for ys in reg.values():
                for y in ys:
                    ys[y] = 0.0
        t.pop("changed_share", None)
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"],
                      write=False)
        assert not rec["pass"]
        runs = [b for b in rec["mismatches"] if b["section"] == "runs"]
        assert [(b["run"], b["recomputed"]) for b in runs] == \
            [("twin", str(made["twin"]["run_dir"]))]
        assert "another output directory" in runs[0]["why"]
        assert rec["sources"]["runs"]["twin"] == str(made["twin"]["run_dir"])
        assert any(b["section"] == "differences" for b in rec["mismatches"])
        assert "do not match the manager's records" in N.reason(rec) or \
            "does not match the manager's records" in N.reason(rec)

    def test_two_runs_on_one_directory(self, made, tmp_path):
        rec = N.check(_run_folder(tmp_path, made,
                                  records={"twin": made["ctl"]["run_dir"]}),
                      registry=made["registry"], write=False)
        assert not rec["pass"]
        assert any("resolve to one output directory" in (b.get("why") or "")
                   for b in rec["mismatches"])

    def test_a_run_no_record_names(self, made, tmp_path):
        rec = N.check(_run_folder(tmp_path, made, records={}), registry=made["registry"],
                      write=False)
        assert not rec["pass"]
        assert [b["why"] for b in rec["mismatches"] if b["section"] == "runs"] == [
            "no record of the manager names this run's output directory, so its numbers "
            "cannot be recomputed"]

    def test_a_missing_input_is_not_a_pass(self, made, tmp_path):
        rec = N.check(_run_folder(tmp_path, made), registry={"regions": {}}, write=False)
        assert not rec["pass"] and "box" in rec["error"]


class TestTheRules:

    @pytest.mark.parametrize("units,rate", [
        ("gC/m^2/s", True), ("mm/s", True), ("kg m-2 s-1", True), ("proportion/sec", True),
        ("counts/km2/sec", True), ("umol/m2s", True), ("per second", True), ("s^-1", True),
        ("kg/m/s^2", False), ("m s-2", False), ("mm", False), ("K", False), ("1", False),
        ("W/m^2", False), ("gC/m^2", False)])
    def test_a_rate_per_second(self, units, rate):
        assert N.per_second("X", units) is rate

    def test_fire_is_per_second_whatever_its_label(self):
        assert N.per_second("FAREA_BURNED", "proportion")

    def test_band_names_give_their_edges(self):
        assert N.band_rule("below 1000 m") == (None, 1000.0)
        assert N.band_rule("1000-1500 m") == (1000.0, 1500.0)
        assert N.band_rule("2000 m and above") == (2000.0, None)
        assert N.band_rule("watershed") is None


# ── months, seasons and where the numbers come from ─────────────────────────
def _at(t, where):
    d = t
    for k in where[:-1]:
        d = d[k]
    return d, where[-1]


class TestMonthsAndSeasonsAreChecked:
    """MONTHS AND SEASONS ARE CHECKED (2026-11-03): every monthly and seasonal
    value, difference and normal, and the provenance block, made again by the
    check's own path; nothing of the table is left unchecked."""

    def test_every_period_is_compared(self, made, tmp_path):
        t = made["table"]
        assert {"monthly_differences", "seasonal_differences", "provenance"} <= set(t)
        n_periods = sum(1 for r in t["runs"].values() for k in ("monthly", "seasonal")
                        for f in r[k].values() for ys in f.values() for _ in ys)
        n_diff = sum(1 for k in ("monthly_differences", "seasonal_differences")
                     for r in t[k].values() for f in r.values() for ys in f.values()
                     for _ in ys)
        n_norm = sum(1 for k in ("monthly_normal", "seasonal_normal")
                     for f in t["event"][k].values() for ls in f.values() for _ in ls)
        assert n_periods and n_diff and n_norm
        rec = N.check(_run_folder(tmp_path, made), registry=made["registry"], write=False)
        assert rec["pass"] and rec["not_checked"] == []
        assert rec["n_compared"] > n_periods + n_diff + n_norm
        assert rec["largest_relative_difference"]["rel_diff"] < 1e-9

    def test_the_checks_months_add_up_to_its_year(self, made):
        """The check's own path: a rate's months sum to its year, a state's
        months weighted by their days average to it."""
        mine = N.recompute_runs(made["table"], {A.VEGETATED: _veg(made)},
                                N.History(made["ctl"]["run_dir"]).grid(),
                                dirs={"control": str(made["ctl"]["run_dir"])})["control"]
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for f, how in (("GPP", "sum"), ("TSA", "mean"), ("QVEGT", "sum")):
            assert mine["how"][f] == how
            mon = mine["monthly"][A.VEGETATED][f]
            xs = [mon[f"2016-{m:02d}"] for m in range(1, 13)]
            y = sum(xs) if how == "sum" else sum(x * d for x, d in zip(xs, days)) / 365
            assert y == pytest.approx(mine["values"][A.VEGETATED][f]["2016"], rel=1e-12)
            sea = mine["seasonal"][A.VEGETATED][f]
            jja = xs[5:8] if how == "sum" else None
            if jja:
                assert sea["2016-JJA"] == pytest.approx(sum(jja), rel=1e-12)

    def test_the_runs_first_step_is_not_a_december(self, made):
        """ELM writes the run's first model step before 1 January (folded into
        January in the monthly stream): DJF of the first year is left out, as
        the table leaves it out; DJF of the second year is made."""
        mine = N.recompute_runs(made["table"], {A.VEGETATED: _veg(made)},
                                N.History(made["ctl"]["run_dir"]).grid(),
                                dirs={"control": str(made["ctl"]["run_dir"])})["control"]
        for f in ("GPP", "QVEGT", "TSA"):
            assert mine["left_out"][f] == ["2015-DJF"]
            assert mine["seasonal"][A.VEGETATED][f]["2016-DJF"] is not None
        assert "2015-DJF" in made["table"]["runs"]["control"]["seasons_left_out"]

    # One number of each new kind moved by 1e-4 of itself: exactly that one
    # number fails.
    @pytest.mark.parametrize("where", [
        ("runs", "control", "monthly", A.VEGETATED, "GPP", "2015-07"),
        ("runs", "twin", "monthly", "box, vegetated", "TSA", "2016-01"),
        ("runs", "control", "monthly", "burned, vegetated", "burned_area_km2", "2016-08"),
        ("runs", "twin", "seasonal", "1000-1500 m", "QVEGT", "2016-DJF"),
        ("runs", "control", "seasonal", A.VEGETATED, "H2OSNO", "2015-SON"),
        ("monthly_differences", "twin", "box, vegetated", "GPP", "2015-06"),
        ("seasonal_differences", "twin", A.VEGETATED, "GPP", "2016-JJA"),
        ("event", "monthly_normal", A.VEGETATED, "GPP", "07"),
        ("event", "seasonal_normal", "box, vegetated", "TSA", "MAM"),
        ("provenance", "regions", "box, vegetated", "area_km2"),
    ])
    def test_one_value_changed_by_a_little(self, made, tmp_path, where):
        t = copy.deepcopy(made["table"])
        d, k = _at(t, where)
        d[k] *= 1.0 + 1e-4
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert not rec["pass"] and rec["n_mismatched"] == 1, rec["mismatches"]
        bad = rec["mismatches"][0]
        assert N.RTOL < bad["rel_diff"] < 2e-4
        sec = where[2] if where[0] == "runs" else where[0]
        assert bad["section"] == sec
        assert bad.get("period", bad.get("what")) == where[-1]

    def test_a_season_the_history_cannot_make_is_given_a_value(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["runs"]["control"]["seasonal"][A.VEGETATED]["GPP"]["2015-DJF"] = 1.0
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert [(b["section"], b["period"]) for b in rec["mismatches"]] == \
            [("seasonal", "2015-DJF")]
        assert "leaves out" in rec["mismatches"][0]["why"]

    def test_a_period_left_out_that_the_history_makes(self, made, tmp_path):
        """A month dropped, and a winter dropped with a reason the history
        does not bear out: both are mismatches."""
        t = copy.deepcopy(made["table"])
        del t["runs"]["twin"]["monthly"]["box, vegetated"]["GPP"]["2016-03"]
        del t["runs"]["twin"]["seasonal"]["box, vegetated"]["GPP"]["2016-DJF"]
        t["runs"]["twin"]["seasons_left_out"]["2016-DJF"] = "no December 2015"
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        rows = sorted((b["section"], b["period"], b["why"]) for b in rec["mismatches"])
        assert [r[:2] for r in rows] == [("monthly", "2016-03"), ("seasonal", "2016-DJF")]
        assert "(no December 2015)" in rows[1][2]
        assert "(with no reason)" in rows[0][2]

    def test_a_difference_left_out(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        del t["seasonal_differences"]["twin"][A.VEGETATED]["GPP"]["2016-SON"]
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert [(b["section"], b["period"]) for b in rec["mismatches"]] == \
            [("seasonal_differences", "2016-SON")]

    def test_the_normal_winter_and_its_years(self, made, tmp_path):
        """The fixture's normal year 2016 follows its event year 2015, so its
        winter holds the event's December: the DJF normal is of no year. A
        table that counts it, or names other years, fails."""
        ev = made["table"]["event"]
        assert ev["normal_years_used"] == {"DJF": []}
        assert ev["seasonal_normal"][A.VEGETATED]["GPP"]["DJF"] is None
        t = copy.deepcopy(made["table"])
        t["event"]["normal_years_used"]["DJF"] = [2016]
        t["event"]["normal_counts"][A.VEGETATED]["GPP"]["DJF"] = 1
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert sorted(b["what"] for b in rec["mismatches"]) == \
            ["normal_count", "normal_years_used"]

    @pytest.mark.parametrize("change", ["bytes", "name", "surface", "cells", "run_dir"])
    def test_the_provenance_block(self, made, tmp_path, change):
        t = copy.deepcopy(made["table"])
        pv = t["provenance"]
        fi = pv["runs"]["twin"]["streams"]["h0"]["files"][0]
        if change == "bytes":
            fi["bytes"] += 1
        elif change == "name":
            fi["name"] = "synth.elm.h0.2099-02-01-00000.nc"
        elif change == "surface":
            pv["surface_file"] = str(made["root"] / "other_surfdata.nc")
        elif change == "cells":
            pv["regions"]["burned, vegetated"]["cells"] -= 1
        else:
            pv["runs"]["twin"]["run_dir"] = str(made["ctl"]["run_dir"])
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert not rec["pass"]
        assert {b["section"] for b in rec["mismatches"]} == {"provenance"}
        want = {"bytes": "bytes", "name": "bytes", "surface": "surface_file",
                "cells": "cells", "run_dir": "run_dir"}[change]
        assert [b["what"] for b in rec["mismatches"]] == [want]
        if change == "name":
            assert rec["mismatches"][0]["why"] == "the file does not exist"

    def test_a_table_made_before_months(self, made, tmp_path):
        """A table from before months and seasons were reported carries none of
        them: its years are checked, and the notes say there are no months;
        a table that carries any of them is held to all of them."""
        t = copy.deepcopy(made["table"])
        for k in N.PERIOD_KEYS:
            t.pop(k)
        for r in t["runs"].values():
            for k in ("monthly", "seasonal", "seasons_left_out", "period_units"):
                r.pop(k, None)
        for k in ("monthly_normal", "seasonal_normal", "normal_counts", "normal_left_out",
                  "normal_years_used"):
            t["event"].pop(k)
        rec = N.check(_run_folder(tmp_path, made, t), registry=made["registry"], write=False)
        assert rec["pass"] and rec["notes"] == [
            "the table was made before months and seasons were reported: it has none "
            "to check"]
        t = copy.deepcopy(made["table"])
        t["runs"]["twin"].pop("monthly")
        rec = N.check(_run_folder(tmp_path / "b", made, t), registry=made["registry"],
                      write=False)
        assert not rec["pass"]
        assert {b["section"] for b in rec["mismatches"]} == {"monthly"}
        assert {b["run"] for b in rec["mismatches"]} == {"twin"}

    def test_a_monthly_mismatch_withholds_the_numbers_report(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["runs"]["control"]["monthly"][A.VEGETATED]["GPP"]["2015-07"] += 1.0
        g = N.gate(_run_folder(tmp_path, made, t), registry=made["registry"])
        assert not g["pass"] and g["n_mismatched"] == 1
        page = Path(g["report"]).read_text()
        assert "| monthly, control, vegetated cells, GPP, 2015-07 |" in page


def _veg(made):
    h = N.History(made["ctl"]["run_dir"])
    surf = N._fsurdat(made["ctl"]["case_dir"])
    return N.read_surface(surf, h.grid())["veg"]


# ── the gate ────────────────────────────────────────────────────────────────
class TestTheGate:

    def test_a_pass_writes_no_report(self, made, tmp_path):
        rd = _run_folder(tmp_path, made)
        g = N.gate(rd, registry=made["registry"])
        assert g["pass"] and g["report"] is None and g["n_mismatched"] == 0
        assert not (rd / "04_analysis" / "DOMAIN_REPORT.md").exists()

    def test_a_failure_withholds_the_numbers_report(self, made, tmp_path):
        t = copy.deepcopy(made["table"])
        t["differences"]["twin"][A.VEGETATED]["GPP"]["2016"] += 1.0
        rd = _run_folder(tmp_path, made, t)
        g = N.gate(rd, registry=made["registry"])
        assert not g["pass"]
        assert g["reason"] == (f"1 of {g['n_compared']} regional numbers did not reproduce "
                               f"when recomputed from the model output")
        page = Path(g["report"]).read_text()
        assert page.splitlines()[0] == f"# WITHHELD: {g['reason']}"
        assert "| differences, twin, vegetated cells, GPP, 2016 |" in page
        assert "the Analyzer was not run" in page

    def test_a_failure_sets_an_earlier_analysis_aside(self, made, tmp_path):
        """A run analysed before, whose numbers now fail: nothing of the earlier
        analysis stays beside the withheld page (2026-11-03)."""
        t = copy.deepcopy(made["table"])
        t["differences"]["twin"][A.VEGETATED]["GPP"]["2016"] *= 1.05
        rd = _run_folder(tmp_path, made, t)
        out = rd / "04_analysis"
        out.mkdir()
        earlier = ["REPORT.md", "analysis.json", "analysis.pptx", "domain_report.json",
                   "domain_fig_GPP_by_year.png", "gpp_by_band.py", "gpp_by_band.png",
                   "interpretation.json", "step3_round1_reply.txt"]
        for n in earlier:
            (out / n).write_text("the earlier answer")
        (out / "notes.txt").write_text("not an analysis product")
        g = N.gate(rd, registry=made["registry"])
        assert not g["pass"]
        assert sorted(p.name for p in out.iterdir() if p.is_file()) == \
            sorted(["DOMAIN_REPORT.md", N.CHECK_JSON, "notes.txt"])
        (moved,) = list((out / N.SUPERSEDED).iterdir())
        assert sorted(p.name for p in moved.iterdir()) == sorted(earlier)
        assert sorted(g["set_aside"]) == sorted(f"{N.SUPERSEDED}/{moved.name}/{n}"
                                                for n in earlier)
        page = (out / "DOMAIN_REPORT.md").read_text()
        assert page.startswith("# WITHHELD: 1 of ")
        assert f"9 file(s) were moved to 04_analysis/{N.SUPERSEDED}/{moved.name}/" in page

    def test_a_table_without_numbers_is_left_to_the_report(self, made, tmp_path):
        rd = _run_folder(tmp_path, made, {"error": "ValueError: boom"})
        assert N.gate(rd, registry=made["registry"]) is None


# ── through the manager: the analyze stage ──────────────────────────────────
from test_domain_manager import _stages, _notify, srv, mgr  # noqa: F401,E402
from test_domain_experiment import study  # noqa: F401,E402
from test_domain_experiment import TestAnExperimentThroughTheManager as _Through  # noqa: E402


class TestTheAnalyzeStage:

    def test_reproduced_numbers_are_reported(self, study, tmp_path):
        m, cfg = _Through._with_history(study, tmp_path)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        rec = json.loads((m.run_dir / "04_analysis" / N.CHECK_JSON).read_text())
        assert rec["pass"] and rec["n_compared"] > 0, rec
        assert _stages(m)["analyze"]["status"] == "done"
        assert not (m.run_dir / "04_analysis" / "DOMAIN_REPORT.md").read_text().startswith(
            "# WITHHELD")

    def test_a_number_that_does_not_reproduce_withholds_everything(self, study, tmp_path,
                                                                   monkeypatch):
        """The server's table one part in a thousand off: the numbers report is
        written withheld, the Analyzer is not run (the fixture fails on any LLM
        client), the stage says so and the mail leads with it."""
        m, cfg = _Through._with_history(study, tmp_path)
        cfg["analysis_llm"] = True
        DA = m._sibling_module("domain_analysis")                   # noqa: N806
        real = DA.event

        def off(run, normal, target):
            out = real(run, normal, target)
            a = out["anomalies"]["vegetated cells"]["GPP"]
            a["normal"] *= 1.001
            return out
        monkeypatch.setattr(DA, "event", off)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "withheld"
        assert an["reason"].startswith("1 of ") and "did not reproduce" in an["reason"]
        page = (m.run_dir / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        assert page.startswith(f"# WITHHELD: {an['reason']}")
        assert not (m.run_dir / "04_analysis" / "analysis.json").exists()
        assert not (m.run_dir / "04_analysis" / "domain_report.json").exists()
        body, rc = _notify().build(str(m.run_dir))
        assert rc == 1 and body.splitlines()[0] == f"WITHHELD: {an['reason']}"

    def test_a_check_that_cannot_run_withholds_and_the_run_survives(self, study, tmp_path,
                                                                    monkeypatch):
        """The gate itself raising ends neither the run nor its summary: the
        stage is withheld with why (2026-11-03)."""
        m, cfg = _Through._with_history(study, tmp_path)

        def boom(run_dir, registry=None):
            raise RuntimeError("no netCDF4 here")
        monkeypatch.setattr(N, "gate", boom)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        an = _stages(m)["analyze"]
        assert an["status"] == "withheld"
        assert an["reason"] == "the numbers check could not run (no netCDF4 here)"
        assert (m.run_dir / "experiment.json").is_file()
        assert (m.run_dir / "RUN_SUMMARY.json").is_file()
        assert not (m.run_dir / "04_analysis" / "domain_report.json").exists()


# ── claims traced to the numbers ────────────────────────────────────────────
TABLE = {
    "regions": {"vegetated cells": {"cells": 13394, "area_km2": 7872.5},
                "watershed, vegetated": {"cells": 4127, "area_km2": 2425.3}},
    "years": [2015, 2016],
    "runs": {"control": {"values": {
        "vegetated cells": {"GPP": {"2015": 487.0, "2016": 536.5}},
        "watershed, vegetated": {"GPP": {"2015": 643.70, "2016": 692.0}}}},
        "dry": {"values": {
            "vegetated cells": {"GPP": {"2015": 479.3, "2016": 530.0}},
            "watershed, vegetated": {"GPP": {"2015": 618.66, "2016": 690.0}}}}},
    "differences": {"dry": {"watershed, vegetated": {"GPP": {"2015": 618.66 - 643.70,
                                                             "2016": -2.0}}}},
    "event": {"normal_years": [2016], "target_years": [2015], "anomalies": {
        "vegetated cells": {"GPP": {"normal": 536.5, "years": {"2015": {
            "value": 487.0, "anomaly": -49.5, "anomaly_pct": -9.226467847157503}}}}}},
}


def _claim(text, values):
    return {"claim": text, "values": values, "finding_id": "f"}


class TestClaimsTracedToTheNumbers:

    def test_the_allowed_derivations(self):
        claims = [
            _claim("GPP over the watershed was 25.0 gC/m2 lower in the dry run in 2015, "
                   "about 3.9 percent.", [618.66 - 643.70, 100 * (618.66 - 643.70) / 643.70]),
            _claim("The control's 2015 GPP was 9.23 percent below its 2016 normal.",
                   [-9.226467847157503]),
            _claim("The watershed holds 4127 vegetated cells, 2430 km2.", [4127, 2425.3]),
            _claim("The dry run's 2015 GPP was 618.7 gC/m2 in the watershed, 139 above "
                   "the domain's.", [618.66, 618.66 - 479.3]),
        ]
        out = T.trace(claims, "Dry summers cut watershed GPP by 25.0 gC/m2.", TABLE)
        assert out["n_traced"] == out["n_values"] == 7 and out["claims_not_traced"] == []
        texts = [d["text"] for c in claims for d in c["derivations"]]
        assert texts[0] == "-25.0 = GPP, watershed: dry minus control, 2015"
        assert texts[1] == "-3.9 = GPP, watershed: dry against control, percent, 2015"
        assert texts[2] == ("-9.23 = GPP, vegetated cells: control, 2015 against the "
                            "2016 normal, percent")
        assert texts[3] == "4127 = watershed: its cells"
        assert texts[4] == "2430 = watershed: its area, km2"
        assert texts[6] == "139 = GPP, 2015: dry watershed minus dry vegetated cells"
        assert all(d["file"] == T.TABLE_FILE for c in claims for d in c["derivations"])
        assert "withheld_because" not in out

    def test_what_does_not_trace_is_marked(self):
        """A year-to-year change, a sum of two numbers and a value the claim's
        own wording does not support are not derivations."""
        claims = [_claim("GPP rose 49.5 gC/m2 from 2015 to 2016.", [49.5]),
                  _claim("GPP fell 25.2 gC/m2 in the watershed in 2015.", [-25.2]),
                  _claim("Together the two regions made 1131 gC/m2.", [1130.7])]
        out = T.trace(claims, None, TABLE)
        assert out["claims_not_traced"] == [1, 2, 3] and out["n_traced"] == 0
        # the same change against the normal year IS in the table, with its sign
        c = [_claim("2015 GPP sat 49.5 gC/m2 below its normal.", [-49.5])]
        T.trace(c, None, TABLE)
        assert c[0]["traced"] and c[0]["derivations"][0]["how"] == "event"
        for c in claims:
            assert c["trace"] == T.NOT_TRACED and not c["traced"]
            assert c["derivations"][0]["text"].endswith(T.NOT_TRACED)

    def test_a_year_on_the_list_is_a_date(self):
        c = [_claim("In 2015.", [2015])]
        T.trace(c, None, TABLE)
        assert c[0]["traced"] and c[0]["derivations"][0]["how"] == "year"

    def test_an_untraced_headline_withholds_the_report(self):
        from agents.analysis import step4_report as S4
        claims = [_claim("GPP fell 25.0 gC/m2 in the watershed in 2015.", [-25.04])]
        out = T.trace(claims, "A dry summer cuts GPP by 31.5 percent.", TABLE)
        assert out["withheld_because"] == ("the headline's value 31.5 cannot be traced to "
                                           "the numbers")
        report = {"question": "q", "headline": "A dry summer cuts GPP by 31.5 percent.",
                  "claims": claims, "numbers_trace": out,
                  "withheld_because": out["withheld_because"]}
        md = S4.render_markdown(report)
        assert md.splitlines()[0] == ("# WITHHELD: the headline's value 31.5 cannot be "
                                      "traced to the numbers")
        assert "- 31.5: no claim states this value" in md
        assert "-25.0 = GPP, watershed: dry minus control, 2015" in md

    def test_a_label_in_the_headline_is_not_a_value(self):
        out = T.trace([], "What 2 K of warming does to GPP in 2015.", TABLE)
        assert out["headline"] == [] and "withheld_because" not in out


class TestOnlyWhatTheClaimSays:
    """A derivation uses a field and region the claim names, pairs two runs in
    one region or two regions of one run, and matches exactly a value the
    claim gives with more digits than it prints (2026-11-03)."""

    def test_another_fields_value_does_not_trace(self):
        t = copy.deepcopy(TABLE)
        t["runs"]["control"]["values"]["watershed, vegetated"]["TSA"] = {"2015": 277.16}
        c = [_claim("GPP over the watershed was 277.2 gC/m2 in 2015.", [277.16])]
        T.trace(c, None, t)
        assert not c[0]["traced"]
        c = [_claim("Air temperature (TSA) over the watershed was 277.2 K in 2015.",
                    [277.16])]
        T.trace(c, None, t)
        assert c[0]["derivations"][0]["text"] == "277.2 = TSA, watershed: control, 2015"

    def test_a_pair_across_runs_and_regions_at_once_does_not_trace(self):
        # dry's watershed minus the control's vegetated cells: 618.66 - 487.0
        c = [_claim("GPP in 2015 was 131.66 gC/m2 higher.", [618.66 - 487.0])]
        T.trace(c, None, TABLE)
        assert not c[0]["traced"]

    def test_more_digits_than_printed_must_match_exactly(self):
        exact = 618.66 - 643.70
        c = [_claim("GPP over the watershed fell 25.0 gC/m2 in 2015.", [exact * 1.001])]
        T.trace(c, None, TABLE)
        assert not c[0]["traced"]
        assert "exactly" in c[0]["derivations"][0]["why"]
        c = [_claim("GPP over the watershed fell 25.0 gC/m2 in 2015.", [exact])]
        T.trace(c, None, TABLE)
        assert c[0]["derivations"][0]["precision"] == "exact"
        assert c[0]["derivations"][0]["match"] == "exact"

    def test_a_perturbed_value_on_a_dense_table(self, made):
        """The server's own table over every kind of region, bands among them:
        the twin's true difference traces, the same difference off by a few
        printed units does not, at either precision."""
        t = made["table"]
        d = t["differences"]["twin"]["box, vegetated"]["GPP"]["2015"]
        text = "GPP over the box was {:.1f} gC/m2 higher in the twin in 2015."
        good = [_claim(text.format(d), [d])]
        T.trace(good, None, t)
        assert good[0]["traced"], good[0]["derivations"]
        assert good[0]["derivations"][0]["source"] == \
            "GPP, box: twin minus control, 2015"
        for off in (0.3, 0.7, 1.3, -0.4):
            v = round(d + off, 1)
            for vals in ([v], [d + off]):
                c = [_claim(text.format(v), vals)]
                T.trace(c, None, t)
                assert not c[0]["traced"], (off, vals, c[0]["derivations"][0]["text"])

    def test_a_headline_number_needs_a_claim_value_of_its_unit(self):
        claims = [_claim("The watershed holds 4127 vegetated cells, 2430 km2.",
                         [4127, 2425.3])]
        out = T.trace(copy.deepcopy(claims), "Dry summers cut GPP by 2430 gC/m2.", TABLE)
        assert out["withheld_because"] == ("the headline's value 2430 cannot be traced "
                                           "to the numbers")
        assert "withheld_because" not in T.trace(
            copy.deepcopy(claims), "The watershed covers 2430 km2.", TABLE)

    def test_a_whole_number_with_a_measured_unit_must_trace(self):
        claims = [_claim("GPP over the watershed fell 25.0 gC/m2 in 2015.", [-25.04])]
        out = T.trace(claims, "A dry summer cuts watershed GPP by 40 gC/m2.", TABLE)
        assert out["withheld_because"] == ("the headline's value 40 cannot be traced to "
                                           "the numbers")
        assert out["headline"][0]["why"] == "no claim states this value in gc/m2"
        # a whole number states its units digit, not its tens: 25 is not 40
        out = T.trace(claims, "A dry summer cuts watershed GPP by 25 gC/m2.", TABLE)
        assert "withheld_because" not in out

    def test_a_value_shaped_like_a_year(self):
        """2050 gC/m2 is a value: only a year of the table with no unit after
        it is a date."""
        claims = [_claim("GPP over the watershed reached 2050 gC/m2 in 2016.", [2050.0])]
        out = T.trace(claims, "GPP reached 2050 gC/m2 in 2016.", TABLE)
        assert not claims[0]["traced"] and claims[0]["derivations"][0].get("how") != "year"
        assert out["withheld_because"] == ("the headline's value 2050 cannot be traced to "
                                           "the numbers")
        claims = [_claim("GPP over the watershed reached 2016 gC/m2 in 2015.", [2016.0])]
        T.trace(claims, None, TABLE)
        assert not claims[0]["traced"]


class TestClaimsAboutAMonthOrASeason:
    """A claim may trace to a month's or a season's number only when it names
    that month or season (2026-11-03); B's rules on fields, regions and the
    printed precision hold for them too."""

    def test_what_a_claim_names(self):
        assert T.named_periods("GPP in July 2015 fell") == {("month", "07")}
        assert T.named_periods("June-August GPP") == {("season", "JJA")}
        assert T.named_periods("over the summer of 2015") == {("season", "JJA")}
        assert T.named_periods("from Dec to Feb") == {("season", "DJF")}
        assert T.named_periods("JJA GPP") == {("season", "JJA")}
        assert T.named_periods("GPP may fall in 2015") == set()
        assert T.named_periods("GPP fell in May") == {("month", "05")}
        assert T.named_periods("in the fall of 2016") == {("season", "SON")}

    def test_a_months_and_a_seasons_numbers_trace(self, made):
        t = made["table"]
        md = t["monthly_differences"]["twin"]["box, vegetated"]["GPP"]["2015-07"]
        c = [_claim(f"GPP over the box was {md:.2f} gC/m2 higher in the twin in July 2015.",
                    [md])]
        T.trace(c, None, t)
        assert c[0]["traced"], c[0]["derivations"]
        assert c[0]["derivations"][0]["source"] == "GPP, box: twin minus control, July 2015"
        # the same claim naming June, or no month, does not trace to July's number
        for when in ("June 2015", "2015"):
            c = [_claim(f"GPP over the box was {md:.2f} gC/m2 higher in the twin in {when}.",
                        [md])]
            T.trace(c, None, t)
            assert not c[0]["traced"], c[0]["derivations"]
        sn = t["event"]["seasonal_normal"][A.VEGETATED]["TSA"]["JJA"]
        c = [_claim(f"The domain's summer air temperature (TSA) normal is {sn:.2f} K.", [sn])]
        T.trace(c, None, t)
        # the fixture's years repeat, so a summer's own value gives it too
        d = c[0]["derivations"][0]
        assert c[0]["traced"] and d["period"] == ("season", "JJA") and "JJA" in d["source"]
        assert d["value"] == sn
        a = t["runs"]["twin"]["seasonal"]["box, vegetated"]["GPP"]["2016-JJA"]
        b = t["runs"]["control"]["seasonal"]["box, vegetated"]["GPP"]["2016-JJA"]
        p = 100.0 * (a - b) / b
        c = [_claim(f"June-August GPP over the box was {p:.1f} percent higher in the twin "
                    f"in 2016.", [round(p, 1)])]
        T.trace(c, None, t)
        assert c[0]["derivations"][0]["source"] == \
            "GPP, box: twin against control, percent, JJA 2016"

    def test_a_normal_of_a_month(self):
        t = copy.deepcopy(TABLE)
        t["event"]["monthly_normal"] = {"vegetated cells": {"GPP": {"07": 61.3}}}
        c = [_claim("The July GPP normal over the domain is 61.3 gC/m2.", [61.3])]
        T.trace(c, None, t)
        assert c[0]["derivations"][0]["source"] == \
            "GPP, vegetated cells: control, the 2016 normal of July"
        c = [_claim("The August GPP normal over the domain is 61.3 gC/m2.", [61.3])]
        T.trace(c, None, t)
        assert not c[0]["traced"]

    def test_the_months_numbers_keep_the_other_rules(self, made):
        """Another field's July value, or July's value given with more digits
        than printed and off in them, does not trace."""
        t = made["table"]
        x = t["runs"]["control"]["monthly"]["box, vegetated"]["TSA"]["2015-07"]
        c = [_claim(f"GPP over the box was {x:.1f} in July 2015.", [x])]
        T.trace(c, None, t)
        assert not c[0]["traced"]
        c = [_claim(f"TSA over the box was {x:.1f} K in July 2015.", [x * 1.0001])]
        T.trace(c, None, t)
        assert not c[0]["traced"] and "exactly" in c[0]["derivations"][0]["why"]
        c = [_claim(f"TSA over the box was {x:.1f} K in July 2015.", [x])]
        T.trace(c, None, t)
        assert c[0]["derivations"][0]["source"] == "TSA, box: control, July 2015"


class TestThroughStep4AndTheAnalyzer:

    def test_report_md_lists_each_values_derivation(self, tmp_path):
        from test_domain_analyzer import write_run, Scripted, REVIEW
        from agents.analysis import step0_context as S0
        from agents.analysis import step2_investigate as S2
        from agents.analysis import step3_interpret as S3
        from agents.analysis import step4_report as S4
        ctx = S0.load(write_run(tmp_path))
        inv = S2.investigate(ctx, tmp_path, client=Scripted())
        it = S3.interpret(ctx, {}, inv, tmp_path, client=Scripted(review=REVIEW),
                          with_images=False)
        rep = S4.build(ctx, {}, inv, it, ctx.run_dir,
                       steps={"context": True, "compare": True, "investigate": True,
                              "interpret": True})
        (c,) = rep["claims"]
        assert c["traced"] and rep["numbers_trace"]["n_traced"] == 1
        assert "withheld_because" not in rep
        md = S4.render_markdown(rep)
        assert "   - 487 = GPP, vegetated cells: control, 2015" in md

    def test_an_untraced_headline_is_a_withheld_stage(self, tmp_path, monkeypatch):
        import agents.llm_agent as L
        from agents.analyzer import Analyzer
        from test_domain_analyzer import write_run, Scripted, REVIEW
        review = dict(REVIEW, headline="GPP over the vegetated cells was 512.3 gC/m^2/yr.")

        class Wrong(Scripted):
            def __init__(self, model=None):
                super().__init__(model, review=review)
        monkeypatch.setattr(L, "SimpleLLMClient", Wrong)
        rd = write_run(tmp_path)
        st = Analyzer(str(rd), verbose=False).run(config={"analysis_with_images": False})
        assert st["withheld"] == "the headline's value 512.3 cannot be traced to the numbers"
        md = (rd / "04_analysis" / "REPORT.md").read_text()
        assert md.startswith("# WITHHELD: the headline's value 512.3")
        a = json.loads((rd / "04_analysis" / "analysis.json").read_text())
        assert a["withheld_because"] == st["withheld"] and a["headline"] == review["headline"]
        assert not (rd / "04_analysis" / "analysis.pptx").exists()

    def test_an_earlier_deck_is_set_aside_when_the_report_is_withheld(self, tmp_path):
        from agents.analysis import step4_report as S4
        out = tmp_path / "04_analysis"
        out.mkdir()
        (out / "analysis.pptx").write_text("the earlier deck")
        (out / "gpp.py").write_text("an earlier figure's script")
        report = {"question": "q", "headline": "h", "claims": [], "numbers_trace": {},
                  "withheld_because": "the headline's value 1 cannot be traced"}
        S4.write(report, out)
        assert not (out / "analysis.pptx").exists() and (out / "gpp.py").exists()
        (moved,) = list((out / N.SUPERSEDED).iterdir())
        assert (moved / "analysis.pptx").read_text() == "the earlier deck"


# ── the demo runs on disk (opt-in) ──────────────────────────────────────────
CONFIG = Path(__file__).with_name("local_oracle_paths.json")


def _demo_dirs():
    if not CONFIG.is_file():
        return []
    return list((json.loads(CONFIG.read_text()).get("numbers_check_demos") or {})
                .get("run_dirs") or [])


@pytest.mark.localdata
@pytest.mark.parametrize("src", _demo_dirs() or ["(no numbers_check_demos block)"])
def test_a_demo_runs_numbers_reproduce(src, tmp_path):
    """The records a check reads, copied out of a finished demo run (read-only),
    recomputed from that run's model output on disk."""
    if not Path(src).is_dir():
        pytest.skip(f"{src} is not on this machine")
    rd = tmp_path / Path(src).name
    for sub in ("03_results", "01_inputs"):
        shutil.copytree(Path(src) / sub, rd / sub)
    for ck in Path(src).glob("domain_case/*/checks"):
        shutil.copytree(ck, rd / ck.relative_to(src))
    rec = N.check(rd)
    assert rec["pass"], (rec.get("error"), rec["mismatches"][:5])
    assert rec["largest_relative_difference"]["rel_diff"] < 1e-9 or \
        rec["largest_relative_difference"]["section"] == "changed_share"
    a = json.loads((Path(src) / "04_analysis" / "analysis.json").read_text())
    table = json.loads((rd / "03_results" / "regional.json").read_text())
    out = T.trace(copy.deepcopy(a["claims"]), a.get("headline"), table)
    assert "withheld_because" not in out
    # the headline's first value moved by three printed units, in the claim
    # that states it as well: it must no longer trace
    first = out["headline"][0]["printed"]
    nd = len(first.split(".")[1]) if "." in first else 0
    moved = f"{float(first) + 3 * 10.0 ** -nd:.{nd}f}"
    claims = copy.deepcopy(a["claims"])
    for c in claims:
        if first in c["claim"]:
            c["claim"] = c["claim"].replace(first, moved)
            c["values"] = [(-1 if v < 0 else 1) * float(moved)
                           if T.printed_form(v, first) == first else v
                           for v in c["values"]]
    out = T.trace(claims, a["headline"].replace(first, moved), table)
    assert out.get("withheld_because"), out["headline"]


@pytest.mark.localdata
@pytest.mark.parametrize("src", _demo_dirs() or ["(no numbers_check_demos block)"])
def test_a_demo_runs_months_and_seasons_reproduce(src, tmp_path):
    """The demo run's records copied (read-only), its regional.json made again
    by the server as the manager makes it, now with months, seasons and the
    provenance block: every number reproduces, nothing is left unchecked, and
    one monthly value moved by 1e-4 of itself fails (2026-11-03)."""
    import domain_controls as DCT
    if not Path(src).is_dir():
        pytest.skip(f"{src} is not on this machine")
    rd = tmp_path / Path(src).name
    for sub in ("03_results", "01_inputs"):
        shutil.copytree(Path(src) / sub, rd / sub)
    for ck in Path(src).glob("domain_case/*/checks"):
        shutil.copytree(ck, rd / ck.relative_to(src))
    old = json.loads((rd / "03_results" / "regional.json").read_text())
    analysis = json.loads((Path(src) / "strategy.json").read_text())[
        "domain_experiment"]["analysis"]
    name = json.loads((rd / "01_inputs" / "domain_control_facts.json").read_text())[
        "control"]["name"]
    t = A.experiment(DCT.get(name), {n: r["run_dir"] for n, r in old["runs"].items()},
                     analysis, old["years"],
                     confined={n: v["region"] for n, v in
                               (old.get("changed_share") or {}).items()} or None)
    t = json.loads(json.dumps(t, default=str))
    assert {"monthly_differences", "seasonal_differences", "provenance"} <= set(t)
    (rd / "03_results" / "regional.json").write_text(json.dumps(t, indent=1))
    rec = N.check(rd, write=False)
    assert rec["pass"], (rec.get("error"), rec["mismatches"][:5])
    assert rec["not_checked"] == []
    assert rec["largest_relative_difference"]["rel_diff"] < 1e-9 or \
        rec["largest_relative_difference"]["section"] == "changed_share"
    # the run's claims trace to the rebuilt table as to the one they were made
    # from, except a burned area the old table carried beside an observation
    # (the server stopped making it on 2026-10-27)
    a = json.loads((Path(src) / "04_analysis" / "analysis.json").read_text())
    before, after = copy.deepcopy(a["claims"]), copy.deepcopy(a["claims"])
    T.trace(before, None, old)
    out = T.trace(after, a.get("headline"), t)
    assert "withheld_because" not in out
    for cb, ca in zip(before, after):
        for db, da in zip(cb["derivations"], ca["derivations"]):
            if da.get("traced") or not db.get("traced"):
                assert da.get("text") == db.get("text"), (db, da)
            else:
                assert db.get("field") == "burned_area_km2" and "observed" in old, db
    reg = next(iter(t["regions"]))
    f = next(f for f in t["runs"]["control"]["monthly"][reg]
             if t["runs"]["control"]["monthly"][reg][f].get(f"{t['years'][0]}-07"))
    t["runs"]["control"]["monthly"][reg][f][f"{t['years'][0]}-07"] *= 1.0 + 1e-4
    (rd / "03_results" / "regional.json").write_text(json.dumps(t, indent=1))
    rec = N.check(rd, write=False)
    assert [(b["section"], b["field"], b["period"]) for b in rec["mismatches"]] == \
        [("monthly", f, f"{t['years'][0]}-07")]
