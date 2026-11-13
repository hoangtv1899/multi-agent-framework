"""What a new run is given, beside the control run (domain_given), on a made-up valley.

    pytest tests/test_domain_given.py -v

The design page shows what each new run is given before it is made: a table of
the changes and a figure of the changed inputs beside the control run's. The
model server works out the values. These tests hold them to numbers worked out
here from the arrays the made-up history and the weather file were written
with, never to the module's own reading.

The control run is a 4 x 5 grid of monthly history (elm_gridded_fixtures) with
TBOT, RAIN and SNOW, its rain and snow spread unevenly over the cells so that
the weighting matters. Its cell (3, 3) has no land and weighs nothing, and its
cell (3, 4) is half land. A run's weather file is made by domain_weather
itself, on a domain file of the same cells, with the checks
domain_changes writes beside it, as the server's build_domain_runs leaves them
before a page is written. Every number here is made up.
"""
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_changes as CH
import domain_given as DG
import domain_weather as W
import elm_gridded_fixtures as fx
from test_domain_weather import _control_case, _domain, _field_region

YEARS = [2015, 2016]
HISTORY_YEARS = [2015, 2016, 2017]
NLAT, NLON = fx.grid()["shape"]
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
NO_LAND = (3, 3)
BOX = [(0, 1), (0, 2), (1, 1), (1, 2)]               # a made-up region "floor"
CORNER = [(2, 4), (3, 3), (3, 4)]                     # holds the cell with no land
FACTS = {"settings": {"CCSM_CO2_PPMV": "367.0", "suplnitro": "NONE", "suplphos": "NONE"},
         "burned_area_km2": {"model": {"2015": 41.2, "2016": 7.5, "2017": 13.0},
                             "observed": {"2015": 900.0}, "pattern": "spread thin"}}
RAIN = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8]}}
WARM = {"id": "weather_scale", "params": {"temperature_K": 2.0, "months": [6, 7, 8]}}
FIRE = {"id": "fire_off", "params": {}}


# ── the made-up control run ─────────────────────────────────────────────────
def _rain_spread(j, i):
    return 1.0 + 0.1 * (j * NLON + i)


def _snow_spread(j, i):
    return 1.0 + 0.3 * j


def _history(run_dir, years=HISTORY_YEARS, variables=("TBOT", "RAIN", "SNOW")):
    """Monthly history in ELM's layout, rain and snow spread over the cells,
    and every year wetter and warmer than the one before, so that a year
    read in place of another shows. The cell with no land holds fill, as
    ELM writes it where there is no land."""
    fx.write_monthly(run_dir, years, case="valley", variables=variables)
    for p in sorted(Path(run_dir).glob("*.elm.h0.*.nc")):
        k = int(p.name.split(".h0.")[1][:4]) - years[0]
        with netCDF4.Dataset(p, "a") as d:
            for v, spread in (("RAIN", _rain_spread), ("SNOW", _snow_spread)):
                if v not in d.variables:
                    continue
                a = np.asarray(d[v][:], float)
                for j in range(NLAT):
                    for i in range(NLON):
                        a[:, j, i] *= spread(j, i) * (1.0 + 0.2 * k)
                d[v][:] = a.astype(np.float32)
            if "TBOT" in d.variables:
                d["TBOT"][:] = (np.asarray(d["TBOT"][:], float) + 0.7 * k).astype(np.float32)
            for v in variables:
                a = np.asarray(d[v][:], np.float32)
                a[:, NO_LAND[0], NO_LAND[1]] = fx.FILL
                d[v][:] = a
    return Path(run_dir)


def _set_fill(run_dir, var, year, month, cell):
    """One cell of one month written as ELM's fill."""
    p = Path(run_dir) / f"valley.elm.h0.{year:04d}-02-01-00000.nc"
    with netCDF4.Dataset(p, "a") as d:
        a = np.asarray(d[var][:], np.float32)
        a[month - 1, cell[0], cell[1]] = fx.FILL
        d[var][:] = a


def _arrays(run_dir, var):
    """{year: (12, nlat, nlon)} as the files hold the field, fill as NaN."""
    out = {}
    for p in sorted(Path(run_dir).glob("*.elm.h0.*.nc")):
        y = int(p.name.split(".h0.")[1][:4])
        with netCDF4.Dataset(p) as d:
            d.set_auto_mask(False)
            a = np.asarray(d[var][:], float)
        out[y] = np.where(np.abs(a) > 1e30, np.nan, a)
    return out


def _weights(run_dir):
    p = sorted(Path(run_dir).glob("*.elm.h0.*.nc"))[0]
    with netCDF4.Dataset(p) as d:
        d.set_auto_mask(False)
        return np.asarray(d["area"][:], float) * np.asarray(d["landfrac"][:], float)


def _precip(run_dir, year):
    """mm/month: RAIN + SNOW times the seconds of each month."""
    r, s = _arrays(run_dir, "RAIN")[year], _arrays(run_dir, "SNOW")[year]
    return [(r[m] + s[m]) * MONTH_DAYS[m] * 86400.0 for m in range(12)]


def _celsius(run_dir, year):
    return [t - 273.15 for t in _arrays(run_dir, "TBOT")[year]]


def _wmean(field, cells, w):
    """The weighted mean over a list of (j, i), worked out cell by cell."""
    num = den = 0.0
    for c in cells:
        if w[c] > 0:
            if not np.isfinite(field[c]):
                return None
            num += field[c] * w[c]
            den += w[c]
    return num / den


LAND = [(j, i) for j in range(NLAT) for i in range(NLON) if (j, i) != NO_LAND]
LABELS = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
SUMMER = [f"{y}-{m:02d}" for y in YEARS for m in (6, 7, 8)]


def _mask(cells):
    m = np.zeros((NLAT, NLON), bool)
    for c in cells:
        m[c] = True
    return m


@pytest.fixture(scope="module")
def valley(tmp_path_factory):
    root = tmp_path_factory.mktemp("given")
    run_dir = _history(root / "control_run")
    domain = _domain(root / "domain.lnd.valley.nc")
    case = _control_case(root, domain=domain, name="ctl")
    regions = {"floor": {"field": str(_field_region(root / "floor.nc", _mask(BOX))),
                         "var": "sel", "op": ">", "value": 0.5, "what": "a made-up floor"},
               "all_land": {"field": str(_field_region(root / "all_land.nc", _mask(LAND))),
                            "var": "sel", "op": ">", "value": 0.5, "what": "every land cell"},
               "corner": {"field": str(_field_region(root / "corner.nc", _mask(CORNER))),
                          "var": "sel", "op": ">", "value": 0.5,
                          "what": "a corner with the cell that has no land"}}
    control = {"name": "valley", "run_dir": str(run_dir), "case_dir": str(case),
               "regions": regions, "facts": FACTS}
    return {"root": root, "run_dir": run_dir, "case": case, "control": control,
            "regions": regions, "w": _weights(run_dir)}


def _build(case, inputs, name, changes, years=YEARS, regions=None):
    """What build_domain_runs leaves for a run before the page is written:
    its weather file under runs/<name>/weather/ and its checks beside it."""
    written = {}
    for c in changes:
        if c["id"] == "weather_scale":
            rc = CH.resolve(c)
            written["weather_scale"] = W.inputs(rc["params"], years, inputs / "runs" / name,
                                                case, {"regions": regions or {}})
    run = {"name": name, "changes": changes}
    d = inputs / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / DG.EXPECTATIONS).write_text(json.dumps(CH.expectations(run, str(case), written)))
    return inputs


def _experiment(runs, years=YEARS):
    return {"control": {"name": "valley", "reuse": True}, "runs": runs, "years": years}


def _one(valley, tmp_path, changes, name="dry_summer", years=YEARS, control=None):
    """The record of one run, built first."""
    inputs = _build(valley["case"], tmp_path / "01_inputs", name, changes, years,
                    valley["regions"])
    rec = DG.page_record(_experiment([{"name": name, "changes": changes}], years),
                         control or valley["control"], inputs)
    assert len(rec["runs"]) == 1
    return rec, rec["runs"][0]


def _f32(x):
    """A setting as the weather file stores it."""
    return float(np.float32(x))


# ── the words ────────────────────────────────────────────────────────────────
class TestWords:

    @pytest.mark.parametrize("months,words", [
        ([6, 7, 8], "June to August"), ([3, 10], "March and October"),
        (list(range(1, 13)), "every month"), ([12, 1, 2], "January to February and December"),
        ([1, 2, 3, 7, 11, 12], "January to March, July and November to December"),
        ([5], "May"), ([4, 6, 9], "April, June and September"), ([6, 7], "June to July")])
    def test_months(self, months, words):
        assert DG.months_words(months) == words

    @pytest.mark.parametrize("years,words", [
        ([2033, 2034], "2033 and 2034"), ([2033, 2034, 2035], "2033 to 2035"),
        ([2035, 2033, 2034], "2033 to 2035"), ([2033], "2033"),
        ([2033, 2035], "2033 and 2035"), ([2033, 2034, 2035, 2038], "2033 to 2035 and 2038"),
        ([2031, 2033, 2034], "2031, 2033 and 2034")])
    def test_years(self, years, words):
        assert DG.years_words(years) == words

    @pytest.mark.parametrize("x,words", [(0.7, "0.7"), (2.0, "2"), (2, "2"), (1.5, "1.5"),
                                         (550.0, "550"), (0.75, "0.75"), ("367.0", "367"),
                                         (0.125, "0.125")])
    def test_numbers_are_as_short_as_they_are_exact(self, x, words):
        assert DG.number_words(x) == words


# ── the rows ─────────────────────────────────────────────────────────────────
def _rows(changes, control=None, years=YEARS):
    """Rows need no built run and no history."""
    rec = DG.page_record(_experiment([{"name": "r", "changes": changes}], years),
                         control if control is not None else {"name": "valley",
                                                              "facts": FACTS}, None)
    return rec["runs"][0]["rows"]


class TestRows:

    def test_rain_and_snow(self):
        assert _rows([RAIN]) == [{"knob": "Rain and snow", "change": "multiplied by 0.7",
                                  "when": "June to August of every year of the run",
                                  "where": "everywhere"}]

    def test_warmer_and_cooler(self):
        warm = _rows([{"id": "weather_scale", "params": {"temperature_K": 2.0,
                                                         "months": [3, 10]}}])
        cool = _rows([{"id": "weather_scale", "params": {"temperature_K": -1.5}}])
        assert warm == [{"knob": "Air temperature", "change": "2 K warmer",
                         "when": "March and October of every year of the run",
                         "where": "everywhere"}]
        assert cool == [{"knob": "Air temperature", "change": "1.5 K cooler",
                         "when": "every month of every year of the run",
                         "where": "everywhere"}]

    def test_both_knobs_give_two_rows_rain_and_snow_first(self):
        got = _rows([{"id": "weather_scale", "params": {
            "temperature_K": 1.5, "precip_factor": 1.25, "months": [6, 7, 8],
            "years": [2016, 2015], "region": "floor"}}])
        when, where = "June to August of 2015 and 2016", "the region `floor`"
        assert got == [{"knob": "Rain and snow", "change": "multiplied by 1.25",
                        "when": when, "where": where},
                       {"knob": "Air temperature", "change": "1.5 K warmer",
                        "when": when, "where": where}]

    def test_a_knob_left_at_no_change_has_no_row(self):
        got = _rows([{"id": "weather_scale", "params": {"temperature_K": 0.0,
                                                        "precip_factor": 0.5}}])
        assert [r["knob"] for r in got] == ["Rain and snow"]

    def test_three_years_in_a_row_are_a_span(self):
        got = _rows([dict(RAIN, params=dict(RAIN["params"], years=[2015, 2016, 2017]))],
                    years=[2015, 2016, 2017])
        assert got[0]["when"] == "June to August of 2015 to 2017"

    def test_a_repeated_year_gives_one_row_per_recorded_year(self):
        got = _rows([{"id": "weather_repeat_year",
                      "params": {"map": {"2017": 2015, "2016": 2015, "2018": 2013,
                                         "2019": 2019}}}], years=[2016, 2017, 2018, 2019])
        assert got == [{"knob": "Weather", "change": "the recorded weather of 2013",
                        "when": "2018", "where": "everywhere"},
                       {"knob": "Weather", "change": "the recorded weather of 2015",
                        "when": "2016 and 2017", "where": "everywhere"}]

    def test_fire(self):
        assert _rows([FIRE]) == [{"knob": "Fire", "change": "switched off",
                                  "when": "every year of the run", "where": "everywhere"}]

    def test_co2_beside_the_control_runs_own_level(self):
        co2 = {"id": "co2_ppmv", "params": {"ppmv": 550.0}}
        assert _rows([co2]) == [{"knob": "CO2",
                                 "change": "550 ppmv, where the control run has 367 ppmv",
                                 "when": "every year of the run", "where": "everywhere"}]
        assert _rows([co2], {"name": "valley"})[0]["change"] == "550 ppmv"
        unread = {"name": "valley", "facts": {"settings": {"CCSM_CO2_PPMV": "$CO2"}}}
        assert _rows([co2], unread)[0]["change"] == "550 ppmv"

    def test_the_controls_level_is_read_as_the_server_reads_it(self):
        co2 = {"id": "co2_ppmv", "params": {"ppmv": 550.0}}

        def control(**settings):
            return {"name": "valley", "facts": {"settings": settings}}
        # the namelist's value comes before the case's own
        got = _rows([co2], control(co2_ppmv=401.5, CCSM_CO2_PPMV="367.0"))[0]["change"]
        assert got == "550 ppmv, where the control run has 401.5 ppmv"
        # a level that is not held constant is no one number to set beside it
        for kind in ("diagnostic", "'prognostic'"):
            assert _rows([co2], control(CCSM_CO2_PPMV="367.0",
                                        ELM_CO2_TYPE=kind))[0]["change"] == "550 ppmv"
        assert _rows([co2], control(CCSM_CO2_PPMV="367.0", co2_type="'constant'")
                     )[0]["change"] == "550 ppmv, where the control run has 367 ppmv"

    def test_nitrogen_deposition(self):
        assert _rows([{"id": "ndep_year", "params": {"year": 1850}}]) == [
            {"knob": "Nitrogen deposition", "change": "held at its level of 1850",
             "when": "every year of the run", "where": "everywhere"}]

    def test_a_supply_the_control_run_has_already_has_no_row(self):
        sup = {"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "NONE"}}
        assert _rows([sup]) == [{"knob": "Nitrogen supply", "change": "all that plants demand",
                                 "when": "every year of the run", "where": "everywhere"}]
        quoted = {"name": "valley", "facts": {"settings": {"suplnitro": "'ALL'",
                                                          "suplphos": "ALL"}}}
        assert _rows([sup], quoted) == [{"knob": "Phosphorus supply",
                                         "change": "none beyond the model's own",
                                         "when": "every year of the run",
                                         "where": "everywhere"}]

    def test_both_supplies_when_the_control_runs_are_not_known(self):
        sup = {"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "NONE"}}
        got = _rows([sup], {"name": "valley"})
        assert [(r["knob"], r["change"]) for r in got] == [
            ("Nitrogen supply", "all that plants demand"),
            ("Phosphorus supply", "none beyond the model's own")]

    def test_rows_are_given_when_no_figure_can_be_made(self):
        rec = DG.page_record(_experiment([{"name": "r", "changes": [RAIN, FIRE]}]),
                             {"name": "valley"}, None)
        run = rec["runs"][0]
        assert [r["knob"] for r in run["rows"]] == ["Rain and snow", "Fire"]
        assert run["panels"] == [] and run["map"] is None
        assert any("not built" in p for p in run["problems"])
        assert any("no burned area" in p for p in run["problems"])

    def test_a_change_that_cannot_be_read_is_named_and_the_others_kept(self):
        got = DG.page_record(_experiment([{"name": "r", "changes": [
            {"id": "no_such_change"}, {"id": "co2_ppmv", "params": {"ppmv": 5000.0}},
            FIRE]}]), {"name": "valley", "facts": FACTS}, None)["runs"][0]
        assert [r["knob"] for r in got["rows"]] == ["Fire"]
        assert sum("cannot be read" in p for p in got["problems"]) == 2


# ── weather_scale ────────────────────────────────────────────────────────────
class TestWeatherScale:

    def test_rain_and_snow_over_the_whole_grid(self, valley, tmp_path):
        rec, run = _one(valley, tmp_path, [RAIN])
        assert rec["control"] == "valley" and rec["years"] == YEARS
        assert rec["problems"] == [] and run["problems"] == []
        (p,) = run["panels"]
        assert p["title"] == "Rain and snow over the whole grid"
        assert (p["units"], p["step"], p["labels"]) == ("mm/month", "month", LABELS)
        assert p["changed"] == SUMMER
        w, f = valley["w"], _f32(0.7)
        for k, lab in enumerate(LABELS):
            y, m = int(lab[:4]), int(lab[5:])
            q = _precip(valley["run_dir"], y)[m - 1]
            assert p["control"][k] == pytest.approx(_wmean(q, LAND, w), rel=1e-12)
            want = _wmean(q * f, LAND, w) if lab in SUMMER else _wmean(q, LAND, w)
            assert p["run"][k] == pytest.approx(want, rel=1e-12)
        assert run["map"] is None

    def test_the_mean_is_weighted_by_area_times_land_fraction(self, valley, tmp_path):
        _, run = _one(valley, tmp_path, [RAIN])
        q = _precip(valley["run_dir"], 2015)[0]
        flat = float(np.mean([q[c] for c in LAND]))
        assert run["panels"][0]["control"][0] != pytest.approx(flat, rel=1e-6)
        # the half-land cell counts half: its weight is half its area
        w = valley["w"]
        assert w[3, 4] == pytest.approx(0.5 * (0.58 + 0.001 * 19), rel=1e-6)
        assert w[NO_LAND] == 0.0

    def test_a_region_reaches_its_cells_and_has_a_map(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch])
        (p,) = run["panels"]
        assert p["title"] == "Rain and snow over the cells the change reaches"
        w, f = valley["w"], _f32(0.7)
        for k, lab in enumerate(LABELS):
            y, m = int(lab[:4]), int(lab[5:])
            q = _precip(valley["run_dir"], y)[m - 1]
            assert p["control"][k] == pytest.approx(_wmean(q, BOX, w), rel=1e-12)
            want = _wmean(q * f, BOX, w) if lab in SUMMER else _wmean(q, BOX, w)
            assert p["run"][k] == pytest.approx(want, rel=1e-12)
        mp = run["map"]
        assert mp["title"] == "Change in rain and snow, mean of the changed months"
        assert mp["units"] == "mm/month"
        assert mp["outline"] == _mask(BOX).astype(int).tolist()
        g = fx.grid()
        assert mp["lat"] == pytest.approx([float(np.float32(x)) for x in g["lat"]], abs=1e-9)
        assert mp["lon"] == pytest.approx([float(np.float32(x)) - 360.0 for x in g["lon"]],
                                          abs=1e-4)
        for j in range(NLAT):
            for i in range(NLON):
                if (j, i) not in BOX:
                    assert mp["values"][j][i] is None
                    continue
                diffs = [_precip(valley["run_dir"], y)[m - 1][j, i] * (f - 1.0)
                         for y in YEARS for m in (6, 7, 8)]
                want = sum(diffs) / len(diffs)
                assert mp["values"][j][i] == pytest.approx(float(f"{want:.4g}"), rel=1e-12)
                assert mp["values"][j][i] == float(f"{mp['values'][j][i]:.4g}")

    def test_a_cell_the_domain_leaves_inactive_is_not_reached(self, valley, tmp_path):
        domain = _domain(tmp_path / "domain_hole.nc", inactive=[(0, 1)])
        case = _control_case(tmp_path, domain=domain, name="ctl_hole")
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "region": "floor"}}
        inputs = _build(case, tmp_path / "01_inputs", "dry_summer", [ch], YEARS,
                        valley["regions"])
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [ch]}]),
                             valley["control"], inputs)["runs"][0]
        reached = [c for c in BOX if c != (0, 1)]
        q = _precip(valley["run_dir"], 2015)[6]
        assert run["panels"][0]["control"][6] == pytest.approx(
            _wmean(q, reached, valley["w"]), rel=1e-12)
        assert run["map"]["outline"] == _mask(reached).astype(int).tolist()
        assert run["map"]["values"][0][1] is None

    def test_a_region_over_every_land_cell_is_the_whole_grid(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "region": "all_land"}}
        _, run = _one(valley, tmp_path, [ch])
        assert run["panels"][0]["title"] == "Rain and snow over the whole grid"
        assert run["map"] is None
        assert run["rows"][0]["where"] == "the region `all_land`"

    def test_a_cell_with_no_land_is_not_reached(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"temperature_K": 2.0, "months": [6, 7, 8],
                                                "region": "corner"}}
        _, run = _one(valley, tmp_path, [ch])
        land = [c for c in CORNER if c != NO_LAND]
        (p,) = run["panels"]
        assert p["title"] == "Air temperature over the cells the change reaches"
        t = _celsius(valley["run_dir"], 2016)[6]
        assert np.isnan(t[NO_LAND])                      # the history holds fill there
        assert p["run"][12 + 6] == pytest.approx(_wmean(t + 2.0, land, valley["w"]),
                                                 rel=1e-12)
        assert run["map"]["outline"] == _mask(land).astype(int).tolist()
        assert run["map"]["values"][3][3] is None and run["map"]["values"][3][4] == 2.0

    def test_the_air_temperature_and_its_map(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"temperature_K": 2.0, "months": [6, 7, 8],
                                                "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch])
        (p,) = run["panels"]
        assert p["title"] == "Air temperature over the cells the change reaches"
        assert p["units"] == "degrees C" and p["changed"] == SUMMER
        w = valley["w"]
        for k, lab in enumerate(LABELS):
            y, m = int(lab[:4]), int(lab[5:])
            t = _celsius(valley["run_dir"], y)[m - 1]
            assert p["control"][k] == pytest.approx(_wmean(t, BOX, w), rel=1e-12)
            want = _wmean(t + 2.0, BOX, w) if lab in SUMMER else _wmean(t, BOX, w)
            assert p["run"][k] == pytest.approx(want, rel=1e-12)
        mp = run["map"]
        assert mp["title"] == "Change in air temperature, mean of the changed months"
        assert mp["units"] == "K"
        assert [[v for v in row] for row in mp["values"]] == [
            [2.0 if (j, i) in BOX else None for i in range(NLON)] for j in range(NLAT)]

    def test_both_knobs_in_one_change(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"temperature_K": -1.5, "precip_factor": 1.25,
                                                "months": [3, 10], "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch])
        assert [p["title"] for p in run["panels"]] == [
            "Rain and snow over the cells the change reaches",
            "Air temperature over the cells the change reaches"]
        changed = [f"{y}-{m:02d}" for y in YEARS for m in (3, 10)]
        assert all(p["changed"] == changed for p in run["panels"])
        w = valley["w"]
        t = _celsius(valley["run_dir"], 2016)[9]
        assert run["panels"][1]["run"][12 + 9] == pytest.approx(_wmean(t - 1.5, BOX, w),
                                                                rel=1e-12)
        q = _precip(valley["run_dir"], 2015)[2]
        assert run["panels"][0]["run"][2] == pytest.approx(_wmean(q * _f32(1.25), BOX, w),
                                                           rel=1e-12)
        assert run["map"]["title"].startswith("Change in rain and snow")

    def test_a_change_in_one_year_changes_only_that_years_months(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "years": [2016]}}
        _, run = _one(valley, tmp_path, [ch])
        p = run["panels"][0]
        assert p["changed"] == ["2016-06", "2016-07", "2016-08"]
        assert p["run"][:12] == p["control"][:12]
        assert p["run"][12 + 6] == pytest.approx(p["control"][12 + 6] * _f32(0.7), rel=1e-9)
        assert run["rows"][0]["when"] == "June to August of 2016"

    def test_a_month_with_no_value_in_a_reached_cell_has_none(self, valley, tmp_path):
        run_dir = tmp_path / "holed_run"
        shutil.copytree(valley["run_dir"], run_dir)
        _set_fill(run_dir, "RAIN", 2015, 7, (0, 1))
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch],
                      control=dict(valley["control"], run_dir=str(run_dir)))
        p = run["panels"][0]
        assert p["control"][6] is None and p["run"][6] is None
        assert p["control"][5] is not None and p["control"][12 + 6] is not None
        diffs = [_precip(run_dir, y)[m - 1][0, 1] * (_f32(0.7) - 1.0)
                 for y in YEARS for m in (6, 7, 8) if (y, m) != (2015, 7)]
        assert run["map"]["values"][0][1] == pytest.approx(
            float(f"{sum(diffs) / len(diffs):.4g}"), rel=1e-12)

    def test_no_weather_file_gives_rows_and_says_why(self, valley, tmp_path):
        rec = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], None)
        run = rec["runs"][0]
        assert run["rows"] and run["panels"] == [] and run["map"] is None
        assert run["problems"] == ["weather_scale: no figure, the run's inputs are not "
                                   "built yet, so the weather file it reads cannot be read"]
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        for f in (inputs / "runs" / "dry_summer" / "weather").glob("weather_anomaly_*.nc"):
            f.unlink()
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], inputs)["runs"][0]
        assert run["panels"] == [] and run["map"] is None and run["rows"]
        assert any("weather file the run was built with is missing" in p
                   for p in run["problems"])
        run = DG.page_record(_experiment([{"name": "not_built", "changes": [RAIN]}]),
                             valley["control"], inputs)["runs"][0]
        assert any("is not built" in p for p in run["problems"]) and run["panels"] == []

    def test_a_weather_file_on_other_cells_gives_no_figure(self, valley, tmp_path):
        shifted = _domain(tmp_path / "domain_shifted.nc", shift_deg=0.01)
        case = _control_case(tmp_path, domain=shifted, name="ctl_shifted")
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8]}}
        inputs = _build(case, tmp_path / "01_inputs", "dry_summer", [ch, FIRE], YEARS)
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [ch, FIRE]}]),
                             valley["control"], inputs)["runs"][0]
        assert [r["knob"] for r in run["rows"]] == ["Rain and snow", "Fire"]
        assert run["map"] is None
        assert [p["title"] for p in run["panels"]] == [DG.FIRE_TITLE]
        assert run["problems"] == ["weather_scale: no figure, the weather file's cells are "
                                   "not the control run's cells"]

    def test_a_weather_file_changed_since_the_build_is_refused(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        (f,) = (inputs / "runs" / "dry_summer" / "weather").glob("weather_anomaly_*.nc")
        with netCDF4.Dataset(f, "a") as d:
            d["pr"][29, 0, 0] = np.float32(0.5)
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], inputs)["runs"][0]
        assert run["panels"] == []
        assert any("not the one the run was built with" in p for p in run["problems"])

    def test_a_design_changed_after_the_build_is_refused(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        now = {"id": "weather_scale", "params": {"precip_factor": 0.5, "months": [6, 7, 8]}}
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [now]}]),
                             valley["control"], inputs)["runs"][0]
        assert run["rows"][0]["change"] == "multiplied by 0.5"
        assert run["panels"] == []
        assert any("other settings than the run's change (precip_factor)" in p
                   for p in run["problems"])

    def test_years_changed_after_the_build_are_refused(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        for years in ([2015, 2016, 2017], [2016, 2017]):
            run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}],
                                             years), valley["control"], inputs)["runs"][0]
            assert run["panels"] == [] and run["rows"]
            assert any("other years than the experiment's" in p for p in run["problems"])

    def test_a_study_folder_moved_since_the_build_still_finds_its_file(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "old" / "01_inputs", "dry_summer", [RAIN],
                        YEARS)
        moved = tmp_path / "new" / "01_inputs"
        moved.parent.mkdir()
        shutil.move(str(inputs), str(moved))
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], moved)["runs"][0]
        assert run["problems"] == [] and len(run["panels"]) == 1

    def test_a_field_the_history_lacks_leaves_out_its_panel_only(self, valley, tmp_path):
        run_dir = _history(tmp_path / "no_tbot", variables=("RAIN", "SNOW"))
        ch = {"id": "weather_scale", "params": {"temperature_K": 2.0, "precip_factor": 0.7,
                                                "months": [6, 7, 8], "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch],
                      control=dict(valley["control"], run_dir=str(run_dir)))
        assert [p["title"] for p in run["panels"]] == [
            "Rain and snow over the cells the change reaches"]
        assert run["map"]["units"] == "mm/month"
        assert run["problems"] == ["Air temperature: the control run's monthly history "
                                   "holds no TBOT"]

    def test_a_temperature_map_when_rain_cannot_be_read(self, valley, tmp_path):
        run_dir = _history(tmp_path / "no_rain", variables=("TBOT", "SNOW"))
        ch = {"id": "weather_scale", "params": {"temperature_K": 2.0, "precip_factor": 0.7,
                                                "months": [6, 7, 8], "region": "floor"}}
        _, run = _one(valley, tmp_path, [ch],
                      control=dict(valley["control"], run_dir=str(run_dir)))
        assert run["map"]["units"] == "K"
        assert run["problems"] == ["Rain and snow: the control run's monthly history holds "
                                   "no RAIN"]

    def test_a_history_in_other_units_is_not_read(self, valley, tmp_path):
        run_dir = tmp_path / "kelvin_less"
        shutil.copytree(valley["run_dir"], run_dir)
        for p in run_dir.glob("*.elm.h0.*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["TBOT"].units = "degC"
                d["RAIN"].units = "kg/m2"
        _, run = _one(valley, tmp_path, [{"id": "weather_scale", "params": {
            "temperature_K": 2.0, "precip_factor": 0.7}}],
            control=dict(valley["control"], run_dir=str(run_dir)))
        assert run["panels"] == []
        assert sorted(run["problems"]) == [
            "Air temperature: the control run's TBOT is in 'degC', not K",
            "Rain and snow: the control run's RAIN is in 'kg/m2', not a rate in mm per second"]


# ── weather_repeat_year ──────────────────────────────────────────────────────
class TestRepeatYear:
    MODEL = [2016, 2017]

    def _rec(self, valley, ymap):
        ch = {"id": "weather_repeat_year", "params": {"map": ymap}}
        return DG.page_record(_experiment([{"name": "dry_again", "changes": [ch]}],
                                          self.MODEL), valley["control"], None)["runs"][0]

    def test_two_model_years_read_one_recorded_year(self, valley):
        run = self._rec(valley, {"2016": 2015, "2017": 2015})
        assert run["rows"] == [{"knob": "Weather", "change": "the recorded weather of 2015",
                                "when": "2016 and 2017", "where": "everywhere"}]
        assert run["problems"] == [] and run["map"] is None
        rain, temp = run["panels"]
        assert rain["title"] == "Rain and snow over the whole grid"
        assert temp["title"] == "Air temperature over the whole grid"
        labels = [f"{y}-{m:02d}" for y in self.MODEL for m in range(1, 13)]
        w = valley["w"]
        for p, units, value in ((rain, "mm/month", _precip), (temp, "degrees C", _celsius)):
            assert (p["units"], p["step"], p["labels"], p["changed"]) == (
                units, "month", labels, labels)
            for k, lab in enumerate(labels):
                y, m = int(lab[:4]), int(lab[5:])
                assert p["control"][k] == pytest.approx(
                    _wmean(value(valley["run_dir"], y)[m - 1], LAND, w), rel=1e-12)
                assert p["run"][k] == pytest.approx(
                    _wmean(value(valley["run_dir"], 2015)[m - 1], LAND, w), rel=1e-12)
        assert rain["run"][:12] == rain["run"][12:]

    def test_a_year_the_map_does_not_name_reads_its_own(self, valley):
        run = self._rec(valley, {"2017": 2015})
        rain = run["panels"][0]
        assert rain["changed"] == [f"2017-{m:02d}" for m in range(1, 13)]
        assert rain["run"][:12] == rain["control"][:12]
        assert rain["run"][12:] != rain["control"][12:]
        assert run["rows"][0]["when"] == "2017"

    def test_a_recorded_year_the_history_lacks_gives_no_figure(self, valley):
        run = self._rec(valley, {"2016": 2014})
        assert run["panels"] == [] and run["rows"]
        assert run["problems"] == [
            "Rain and snow: the control run's monthly history holds no 2014",
            "Air temperature: the control run's monthly history holds no 2014"]


# ── fire_off ─────────────────────────────────────────────────────────────────
class TestFire:

    def _run(self, facts, years=YEARS):
        ctl = {"name": "valley", **({"facts": facts} if facts is not None else {})}
        return DG.page_record(_experiment([{"name": "no_fire", "changes": [FIRE]}], years),
                              ctl, None)["runs"][0]

    def test_the_control_runs_own_fire_beside_none(self):
        run = self._run(FACTS)
        assert run["panels"] == [{"title": "Area the control run's own fire burns",
                                  "units": "km2", "step": "year", "labels": ["2015", "2016"],
                                  "control": [41.2, 7.5], "run": [0.0, 0.0],
                                  "changed": ["2015", "2016"]}]
        assert run["problems"] == [] and run["map"] is None

    def test_a_year_the_facts_lack_has_none(self):
        run = self._run({"burned_area_km2": {"model": {"2016": 7.5}}}, [2015, 2016, 2017])
        assert run["panels"][0]["control"] == [None, 7.5, None]
        assert run["panels"][0]["run"] == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("facts", [None, {}, {"burned_area_km2": {}},
                                       {"burned_area_km2": {"model": {"2030": 5.0}}},
                                       {"burned_area_km2": {"model": "a lot"}}])
    def test_no_burned_area_in_the_facts_gives_no_panel(self, facts):
        run = self._run(facts)
        assert run["panels"] == [] and run["rows"][0]["knob"] == "Fire"
        assert run["problems"] == ["fire_off: no figure, the control run's facts hold no "
                                   "burned area for the run's years"]


# ── several changes, a control with no facts ─────────────────────────────────
class TestTheRecord:

    def test_a_run_with_several_changes(self, valley, tmp_path):
        co2 = {"id": "co2_ppmv", "params": {"ppmv": 550.0}}
        _, run = _one(valley, tmp_path, [RAIN, FIRE, co2])
        assert [r["knob"] for r in run["rows"]] == ["Rain and snow", "Fire", "CO2"]
        assert [p["title"] for p in run["panels"]] == ["Rain and snow over the whole grid",
                                                       DG.FIRE_TITLE]
        assert run["problems"] == []

    def test_several_runs_each_have_their_own(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        _build(valley["case"], inputs, "warm_summer", [WARM], YEARS)
        rec = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]},
                                          {"name": "warm_summer", "changes": [WARM]},
                                          {"name": "no_fire", "changes": [FIRE]}]),
                             valley["control"], inputs)
        assert [r["name"] for r in rec["runs"]] == ["dry_summer", "warm_summer", "no_fire"]
        assert [[p["title"] for p in r["panels"]] for r in rec["runs"]] == [
            ["Rain and snow over the whole grid"], ["Air temperature over the whole grid"],
            [DG.FIRE_TITLE]]

    def test_a_control_with_no_facts(self, valley, tmp_path):
        no_facts = {k: v for k, v in valley["control"].items() if k != "facts"}
        sup = {"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "NONE"}}
        co2 = {"id": "co2_ppmv", "params": {"ppmv": 550.0}}
        rec, run = _one(valley, tmp_path, [RAIN, FIRE, co2, sup], control=no_facts)
        assert rec["problems"] == ["the control run's facts are not known, so its own "
                                   "settings and burned area are not given"]
        assert [(r["knob"], r["change"]) for r in run["rows"]] == [
            ("Rain and snow", "multiplied by 0.7"), ("Fire", "switched off"),
            ("CO2", "550 ppmv"), ("Nitrogen supply", "all that plants demand"),
            ("Phosphorus supply", "none beyond the model's own")]
        assert [p["title"] for p in run["panels"]] == ["Rain and snow over the whole grid"]
        assert run["problems"] == ["fire_off: no figure, the control run's facts hold no "
                                   "burned area for the run's years"]

    def test_the_keys_are_the_contracts(self, valley, tmp_path):
        ch = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                                "region": "floor"}}
        rec, run = _one(valley, tmp_path, [ch, FIRE])
        assert set(rec) == {"control", "years", "problems", "runs"}
        assert set(run) == {"name", "changes", "rows", "panels", "map", "problems"}
        assert all(set(r) == {"knob", "change", "when", "where"} for r in run["rows"])
        for p in run["panels"]:
            assert set(p) == {"title", "units", "step", "labels", "control", "run", "changed"}
            assert len(p["control"]) == len(p["run"]) == len(p["labels"])
            assert set(p["changed"]) <= set(p["labels"])
        assert set(run["map"]) == {"title", "units", "lat", "lon", "values", "outline"}
        assert len(run["map"]["values"]) == len(run["map"]["lat"]) == NLAT
        assert all(len(row) == len(run["map"]["lon"]) == NLON for row in run["map"]["values"])
        json.dumps(rec, allow_nan=False)


# ── nothing raises ───────────────────────────────────────────────────────────
class TestNothingRaises:

    @pytest.mark.parametrize("experiment,control,inputs", [
        (None, None, None), ({}, {}, None), ("an experiment", 7, 3.5),
        ({"runs": "dry_summer", "years": "2015"}, {"name": "valley"}, None),
        ({"runs": [None, 4, {"changes": "x"}, {"name": "r", "changes": [None, "fire_off"]}],
          "years": [2015]}, {"name": "valley", "facts": "none"}, "/no/such/folder"),
        ({"runs": [{"name": "r", "changes": [RAIN, FIRE]}], "years": [True, 2015]},
         {"name": "valley", "facts": FACTS, "run_dir": "/no/such/run"}, None),
        ({"runs": [{"name": "../r", "changes": [RAIN]}], "years": YEARS},
         {"name": "valley", "facts": FACTS}, "/no/such/folder")])
    def test_odd_input_is_named_not_raised(self, experiment, control, inputs):
        rec = DG.page_record(experiment, control, inputs)
        assert set(rec) == {"control", "years", "problems", "runs"}
        json.dumps(rec, allow_nan=False)
        assert rec["problems"] or any(r["problems"] for r in rec["runs"])

    def test_no_years_gives_rows_and_no_figure(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        rec = DG.page_record({"runs": [{"name": "dry_summer", "changes": [RAIN, FIRE]}]},
                             valley["control"], inputs)
        assert rec["years"] == [] and "names no years" in rec["problems"][0]
        assert len(rec["runs"][0]["rows"]) == 2 and rec["runs"][0]["panels"] == []

    def test_a_control_run_with_no_history(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        empty = tmp_path / "empty_run"
        empty.mkdir()
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             dict(valley["control"], run_dir=str(empty)), inputs)["runs"][0]
        assert run["panels"] == [] and run["rows"]
        assert "monthly history cannot be read" in run["problems"][0]

    def test_an_unreadable_record_of_checks(self, valley, tmp_path):
        d = tmp_path / "01_inputs" / "runs" / "dry_summer"
        d.mkdir(parents=True)
        (d / DG.EXPECTATIONS).write_text("{not json")
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], tmp_path / "01_inputs")["runs"][0]
        assert run["problems"] == [f"weather_scale: no figure, the run's "
                                   f"{DG.EXPECTATIONS} cannot be read"]
        (d / DG.EXPECTATIONS).write_text(json.dumps({"output": []}))
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN]}]),
                             valley["control"], tmp_path / "01_inputs")["runs"][0]
        assert run["problems"] == ["weather_scale: no figure, the run's checks name no "
                                   "weather file"]

    def test_a_run_name_that_leaves_the_inputs_is_refused(self, valley, tmp_path):
        _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        other = tmp_path / "other_study"
        other.mkdir()
        # this name leads from the other study's inputs to a run that is built
        name = "../../01_inputs/runs/dry_summer"
        assert (other / "runs" / name / DG.EXPECTATIONS).resolve().is_file()
        run = DG.page_record(_experiment([{"name": name, "changes": [RAIN]}]),
                             valley["control"], other)["runs"][0]
        assert run["panels"] == []
        assert run["problems"] == [f"weather_scale: no figure, {name!r} is not a run's name"]

    def test_a_year_the_weather_file_does_not_hold_is_refused(self, valley, tmp_path):
        inputs = _build(valley["case"], tmp_path / "01_inputs", "dry_summer", [RAIN], YEARS)
        (f,) = (inputs / "runs" / "dry_summer" / "weather").glob("weather_anomaly_*.nc")
        grid = DG._History(valley["run_dir"]).grid
        assert DG._file_fields(f, "pr", [2016], grid).shape == (12, NLAT, NLON)
        with pytest.raises(DG.GivenError, match="holds no record for every month of 2019"):
            DG._file_fields(f, "pr", [2016, 2019], grid)

    def test_a_failure_inside_is_named(self, valley, tmp_path, monkeypatch):
        def broke(*a, **k):
            raise RuntimeError("a made-up failure")
        monkeypatch.setattr(DG, "weather_scale_panels", broke)
        run = DG.page_record(_experiment([{"name": "dry_summer", "changes": [RAIN, FIRE]}]),
                             valley["control"], None)["runs"][0]
        assert run["problems"] == ["weather_scale: no figure (RuntimeError: a made-up "
                                   "failure)"]
        assert len(run["rows"]) == 2 and [p["title"] for p in run["panels"]] == [
            DG.FIRE_TITLE]
        monkeypatch.setattr(DG, "rows", broke)
        run = DG.page_record(_experiment([{"name": "r", "changes": [FIRE]}]),
                             valley["control"], None)["runs"][0]
        assert run["rows"] == [] and "its row cannot be written" in run["problems"][0]
        monkeypatch.setattr(DG, "_run_record", broke)
        rec = DG.page_record(_experiment([{"name": "r", "changes": [FIRE]}]),
                             valley["control"], None)
        assert rec["runs"] == [] and "a made-up failure" in rec["problems"][-1]
