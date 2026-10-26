"""A run's own weather: the anomaly file, the stream files that read it, the
namelist lines, and the check of every year's output.

    pytest tests/test_domain_weather.py -v

weather_scale (domain_changes.py) is written by domain_weather.py for the data
atmosphere's anomaly forcing. These tests hold the writer to what DATM reads
(one record per month at the month's first instant, on the model's own cells),
the namelist lines to the control's lists, and each year's check to the
arithmetic DATM does: the control's air temperature plus the amount, its rain
and its snow each times the factor, in the chosen months and active cells only.
The year-check tests use the rules W.inputs writes, the ones a chain applies.
"""
import bisect
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_case as DC
import domain_changes as CH
import domain_weather as W
import gridded as G
import run_gate as RG
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
from test_domain_experiment import _experiment, lab  # noqa: F401
from test_domain_manager import srv  # noqa: F401

YEARS = [2015, 2016]
TEMP, PRECIP = "Anomaly.Forcing.Temperature", "Anomaly.Forcing.Precip"
NAME = "anomaly.nc"                      # a file written by write_anomaly directly
SEL = "sel"                              # a registered field region of the tests

# The control's resolved datm_in, in the shape CIME writes for the Naches
# control (three streams); the paths are made up.
DATM_IN = """&datm_nml
  decomp = "1d"
  factorfn = "null"
  force_prognostic_true = .false.
  iradsw = 1
  presaero = .true.
  restfilm = "undefined"
  restfils = "undefined"
  wiso_datm = .false.
/
&shr_strdata_nml
  datamode = "CLMNCEP"
  domainfile = "{domain}"
  dtlimit = {dtlimit}
  fillalgo = {fillalgo}
  fillmask = {fillmask}
  fillread = {fillread}
  fillwrite = {fillwrite}
  mapalgo = {mapalgo}
  mapmask = {mapmask}
  mapread = {mapread}
  mapwrite = {mapwrite}
  readmode = {readmode}
  streams = {streams}
  taxmode = {taxmode}
  tintalgo = {tintalgo}
  vectors = "null"
/
"""
CONTROL_LISTS = {
    "dtlimit": ["1.5"] * 3, "fillalgo": ['"nn"'] * 3, "fillmask": ['"nomask"'] * 3,
    "fillread": ['"NOT_SET"'] * 3, "fillwrite": ['"NOT_SET"'] * 3,
    "mapalgo": ['"bilinear"'] * 3, "mapmask": ['"nomask"'] * 3,
    "mapread": ['"NOT_SET"'] * 3, "mapwrite": ['"NOT_SET"'] * 3,
    "readmode": ['"single"'] * 3,
    "streams": ['"datm.streams.txt.CLMMOSARTTEST 2013 2013 2023"',
                '"datm.streams.txt.presaero.clim_2000 1 1 1"',
                '"datm.streams.txt.topo.observed 1 1 1"'],
    "taxmode": ['"cycle"'] * 3, "tintalgo": ['"linear"', '"linear"', '"lower"']}
ONE_STREAM = {k: v[:1] for k, v in CONTROL_LISTS.items()}
# What CIME appends for each anomaly stream (its defaults, and the tintalgo
# the run's user_nl_datm sets).
ANOMALY_DEFAULTS = {
    "dtlimit": "1.5", "fillalgo": '"nn"', "fillmask": '"nomask"', "fillread": '"NOT_SET"',
    "fillwrite": '"NOT_SET"', "mapalgo": '"bilinear"', "mapmask": '"nomask"',
    "mapread": '"NOT_SET"', "mapwrite": '"NOT_SET"', "readmode": '"single"',
    "taxmode": '"cycle"', "tintalgo": '"lower"'}


def _datm_in(domain, streams=(), edit=None, base=None, cime_form=False):
    """A resolved datm_in: the control's, or a run's with anomaly streams in
    the shape CIME writes. edit(lists) may change the lists before writing;
    base is the control's lists; cime_form writes CIME's own form exactly
    (anomaly_forcing single-quoted, the streams over continuation lines, as in
    the proof's CaseDocs/datm_in)."""
    lists = {k: list(v) for k, v in (base or CONTROL_LISTS).items()}
    for s in streams:
        for k, v in ANOMALY_DEFAULTS.items():
            lists[k].append(v)
        lists["streams"].append(f'"datm.streams.txt.{s} 2006 2006 2300"')
    if edit:
        edit(lists)
    joined = {k: ", ".join(v) for k, v in lists.items()}
    if cime_form:
        joined["streams"] = ",\n      ".join(lists["streams"])
    text = DATM_IN.format(domain=domain, **joined)
    if streams:
        q = "'" if cime_form else '"'
        text = text.replace("&datm_nml\n", "&datm_nml\n  anomaly_forcing = "
                            + ", ".join(f"{q}{s}{q}" for s in streams) + "\n", 1)
    return text


def _domain(path, g=None, shift_deg=0.0, shift_lat=0.0, inactive=()):
    """A land domain file on the fixture grid: xc, yc in float64, area in
    square radians, as a domain file carries them; `inactive` cells get
    mask 0."""
    g = g or fx.grid()
    lon2, lat2 = np.meshgrid(np.asarray(g["lon"], float) + shift_deg,
                             np.asarray(g["lat"], float) + shift_lat)
    nj, ni = g["shape"]
    mask = np.ones((nj, ni))
    for c in inactive:
        mask[c] = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("nj", nj)
        f.createDimension("ni", ni)
        for name, data, typ in (("xc", lon2, "f8"), ("yc", lat2, "f8"),
                                ("area", g["area"] / W.EARTH_RADIUS_KM ** 2, "f8"),
                                ("mask", mask, "i4"), ("frac", g["landfrac"], "f8")):
            v = f.createVariable(name, typ, ("nj", "ni"))
            v[:] = data
    return Path(path)


def _control_case(tmp_path, domain=None, name="ctl", base=None, streams=(),
                  cime_form=False):
    """A control case folder whose CaseDocs hold datm_in and lnd_in."""
    domain = domain or _domain(tmp_path / "domain.lnd.synth.nc")
    cd = tmp_path / name
    (cd / "CaseDocs").mkdir(parents=True)
    (cd / "CaseDocs" / "datm_in").write_text(_datm_in(domain, streams=streams, base=base,
                                                      cime_form=cime_form))
    (cd / "CaseDocs" / "lnd_in").write_text("&elm_inparm\n use_nofire = .false.\n/\n")
    return cd


def _params(**kw):
    p = {"temperature_K": 2.0, "precip_factor": 0.8, "months": [6, 7, 8], "region": None}
    p.update(kw)
    return p


def _record_month(start):
    """The calendar month a daily record of the fixture belongs to (its first
    day), or None for the run's 30-minute start record."""
    if start < 0:
        return None
    m, _ = fx.doy_to_md(int(math.floor(start + 1e-9)) % 365 + 1)
    return m


def _history(run_dir, case, years=YEARS):
    return fx.write_daily(run_dir, years, case=case, variables=("TBOT", "RAIN", "SNOW"))


def _as_datm_would(run_dir, offset=0.0, factor=1.0, months=W.ALL_MONTHS, mask=None,
                   only=None):
    """Make a run's daily history what DATM's anomaly forcing would make of
    the control's: TBOT plus the amount, RAIN and SNOW times the factor, in
    the chosen months and cells. `only` limits it to some fields."""
    for p in sorted(Path(run_dir).glob("*.elm.h1.*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            tb = np.asarray(d["time_bounds"][:])
            sel = np.ones(d["TBOT"].shape[1:], bool) if mask is None else mask
            for r in range(tb.shape[0]):
                if _record_month(tb[r, 0]) not in months:
                    continue
                for v, how in (("TBOT", "offset"), ("RAIN", "factor"), ("SNOW", "factor")):
                    if only and v not in only:
                        continue
                    a = np.asarray(d[v][r], float)
                    d[v][r] = np.where(sel, a + offset if how == "offset" else a * factor, a)


def _set_cells(run_dir, fields, cells, value, day=None):
    """Write `value` (or value(old)) into some cells of some fields, on every
    day or on one day (0-based, counted from 1 January of the first year)."""
    for p in sorted(Path(run_dir).glob("*.elm.h1.*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            tb = np.asarray(d["time_bounds"][:])
            for v in fields:
                x = np.asarray(d[v][:], float)
                for r in range(tb.shape[0]):
                    if tb[r, 1] - tb[r, 0] < 0.5:
                        continue                         # the 30-minute start record
                    if day is not None and int(round(tb[r, 0])) != day:
                        continue
                    for c in cells:
                        k = (r,) + tuple(c)
                        x[k] = value(x[k]) if callable(value) else value
                d[v][:] = x


def _anomaly(tmp_path, params, mask=None, domain=None):
    dom = W.read_domain(domain or _domain(tmp_path / "domain.nc"))
    if mask is None:
        mask = np.ones(dom["xc"].shape, bool)
    return W.write_anomaly(tmp_path / "w" / NAME, dom, params, YEARS, mask)


def _box(path, g=None, cols=(1, 2)):
    """A polygon around the centres of the given columns of rows 0-1."""
    g = g or fx.grid()
    lon = np.where(g["lon"] > 180, g["lon"] - 360, g["lon"])
    x0, x1 = lon[cols[0]] - 0.004, lon[cols[1]] + 0.004
    y0, y1 = g["lat"][0] - 0.004, g["lat"][1] + 0.004
    Path(path).write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}))
    return Path(path)


def _field_region(path, mask, g=None):
    """A gridded field that selects `mask`'s cells (sel > 0.5), on the
    fixture grid's own centres, as a registered field region is stored."""
    g = g or fx.grid()
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("lat", g["shape"][0])
        f.createDimension("lon", g["shape"][1])
        f.createVariable("lat", "f8", ("lat",))[:] = g["lat"]
        f.createVariable("lon", "f8", ("lon",))[:] = g["lon"]
        f.createVariable(SEL, "f4", ("lat", "lon"))[:] = mask.astype(np.float32)
    return Path(path)


def _built(tmp_path, params, mask=None, domain=None, years=YEARS):
    """What W.inputs writes for a run on a synthetic control, and the rules a
    chain applies (domain_changes adds the change's id). A mask goes in as a
    registered field region, the path a real run's region takes."""
    ctl = _control_case(tmp_path, domain=domain or _domain(tmp_path / "domain.nc"))
    p, entry = dict(params), None
    if mask is not None:
        entry = {"regions": {SEL: {"field": str(_field_region(tmp_path / "sel.nc", mask)),
                                   "var": SEL, "op": ">", "value": 0.5}}}
        p["region"] = SEL
    out = W.inputs(p, years, tmp_path / "inputs", ctl, entry)
    return out, [dict(r, change="weather_scale") for r in out["output"]]


# ── the file ─────────────────────────────────────────────────────────────────
class TestTheAnomalyFile:

    def test_one_record_per_month_at_its_first_instant_with_a_year_either_side(self):
        t, ym, y0 = W.month_starts(YEARS)
        assert y0 == 2014 and len(t) == 48 and ym[0] == (2014, 1) and ym[-1] == (2017, 12)
        assert list(t[:3]) == [0.0, 31.0, 59.0] and t[12] == 365.0 and t[24] == 730.0

    def test_its_spacing_is_inside_datms_dtlimit(self):
        # shr_stream refuses a stream whose largest record spacing over its
        # smallest exceeds dtlimit (1.5 in the control's datm_in)
        dt = np.diff(W.month_starts(YEARS)[0])
        assert dt.max() / dt.min() <= 1.5

    def test_lower_interpolation_gives_every_day_its_own_months_record(self):
        t, ym, y0 = W.month_starts(YEARS)
        for y in YEARS:
            for doy in range(1, 366):
                day = 365.0 * (y - y0) + doy - 1 + 0.5          # midday
                k = bisect.bisect_right(list(t), day) - 1
                assert ym[k] == (y, fx.doy_to_md(doy)[0])

    def test_the_change_sits_in_the_chosen_months_and_cells_and_identity_elsewhere(
            self, tmp_path):
        mask = np.zeros(fx.grid()["shape"], bool)
        mask[1:3, 2:4] = True
        a = _anomaly(tmp_path, _params(), mask=mask)
        assert a["records"] == 48 and a["years"] == [2014, 2017]
        with netCDF4.Dataset(a["path"]) as f:
            assert f["time"].units == "days since 2014-01-01 00:00:00"
            assert f["time"].calendar == "noleap"
            tas, pr = np.asarray(f["tas"][:]), np.asarray(f["pr"][:])
        for k, (y, m) in enumerate(W.month_starts(YEARS)[1]):
            on = m in (6, 7, 8)
            assert np.all(tas[k][mask] == (2.0 if on else 0.0))
            assert np.all(tas[k][~mask] == 0.0)
            assert np.allclose(pr[k][mask], 0.8 if on else 1.0)
            assert np.all(pr[k][~mask] == 1.0)

    def test_it_carries_the_domains_own_cells(self, tmp_path):
        d = _domain(tmp_path / "domain.nc")
        a = _anomaly(tmp_path, _params(), domain=d)
        dom = W.read_domain(d)
        with netCDF4.Dataset(a["path"]) as f:
            for v in ("xc", "yc", "area", "mask"):
                assert np.array_equal(np.asarray(f[v][:]), dom[v]), v
            assert f["mask"].dtype == np.int32
            assert f["tas"].dimensions == ("time", "nj", "ni")

    def test_a_mask_of_another_shape_is_refused(self, tmp_path):
        with pytest.raises(W.WeatherError, match="mask is"):
            _anomaly(tmp_path, _params(), mask=np.ones((2, 2), bool))

    def test_records_are_read_back_by_month(self, tmp_path):
        a = _anomaly(tmp_path, _params(months=[12]))
        by_month, xc, yc = W.anomaly_by_month(a["path"], "tas", 2016)
        assert sorted(by_month) == list(range(1, 13))
        assert np.all(by_month[12] == 2.0) and np.all(by_month[11] == 0.0)
        assert xc.shape == fx.grid()["shape"]

    def test_a_time_axis_it_did_not_write_is_refused(self, tmp_path):
        a = _anomaly(tmp_path, _params())
        with netCDF4.Dataset(a["path"], "a") as f:
            f["time"][5] = f["time"][5] + 14.0              # mid-month
        with pytest.raises(W.WeatherError, match="record 5 .* is not the first instant "
                                                  "of 2014-06"):
            W.anomaly_by_month(a["path"], "tas", 2015)


# ── when a month's value takes effect ────────────────────────────────────────
# DATM's own rules, replayed step by step on the file and the stream file as
# written (E3SM maint-2.0):
#   the driver advances the clock, runs the land on what the atmosphere sent
#   before, then runs the atmosphere: a land step takes the weather DATM
#   computed at the step's start          (driver-mct/main/cime_comp_mod.F90)
#   a new pair of records is found only when the time is outside the current
#   pair, its ends included                  (shr_dmodel_mod.F90, readLBUB)
#   the pair found is the last record at or before the time and the next one
#                                    (shr_stream_mod.F90, shr_stream_findBounds)
#   "lower" gives the earlier record full weight     (shr_tInterp_getFactors)
#   the stream's <offset> is added to every record's time
#                                          (shr_stream_mod.F90, readTCoord)
def _effective_times_s(anomaly_path, stream_text_):
    root = ET.fromstring(stream_text_)
    offset = int(root.find("fieldInfo").find("offset").text.strip())
    with netCDF4.Dataset(anomaly_path) as f:
        return [int(round(x * 86400.0)) + offset for x in np.asarray(f["time"][:], float)]


def _land_receives(rec_s, values, start_s, n_steps, dt_s):
    """The value each land step receives, from a fresh start (each chain year
    is one job, so DATM starts again on 1 January)."""
    out, lb = [], None
    for j in range(n_steps):
        t = start_s + j * dt_s
        if lb is None or t < rec_s[lb] or t > rec_s[lb + 1]:
            lb = bisect.bisect_right(rec_s, t) - 1
        out.append(values[lb])
    return np.asarray(out)


def _daily_means(tmp_path, offset_text=None, dt_s=1800, year=2015):
    """Each history day's mean of what the land receives (the steps that
    start in the day), for a +2 K change in June-August."""
    a = _anomaly(tmp_path, _params(precip_factor=1.0))
    text = W.stream_text(TEMP, tmp_path / "w", NAME)
    if offset_text is not None:
        text = text.replace(f"-{W.LEAD_SECONDS}\n", f"{offset_text}\n")
    rec_s = _effective_times_s(a["path"], text)
    with netCDF4.Dataset(a["path"]) as f:
        values = [float(x) for x in np.asarray(f["tas"][:, 0, 0])]
    per_day = 86400 // dt_s
    start = 365 * 86400 * (year - W.month_starts(YEARS)[2])
    got = _land_receives(rec_s, values, start, 365 * per_day, dt_s)
    return got.reshape(365, per_day).mean(axis=1)


def _month_of_day(d):
    return fx.doy_to_md(d + 1)[0]


class TestWhenAMonthTakesEffect:

    @pytest.mark.parametrize("dt_s", [1800, 3600, 900])
    def test_every_day_takes_its_own_months_value(self, tmp_path, dt_s):
        for year in YEARS:
            means = _daily_means(tmp_path / f"{dt_s}_{year}", dt_s=dt_s, year=year)
            want = [2.0 if _month_of_day(d) in (6, 7, 8) else 0.0 for d in range(365)]
            assert np.array_equal(means, want), (year, np.flatnonzero(means != want))

    def test_records_at_midnight_would_miss_the_first_step_of_each_change(self, tmp_path):
        # the first design: no lead; 1 June keeps May's value for its first
        # step and 1 September August's
        means = _daily_means(tmp_path, offset_text="0")
        june1, sep1 = 151, 243
        assert means[june1] == pytest.approx(2.0 * 47 / 48)
        assert means[sep1] == pytest.approx(2.0 / 48)
        others = [d for d in range(365) if d not in (june1, sep1)]
        assert all(means[d] == (2.0 if _month_of_day(d) in (6, 7, 8) else 0.0)
                   for d in others)


# ── the stream file ──────────────────────────────────────────────────────────
class TestTheStreamFile:

    @pytest.mark.parametrize("stream,pair", [(TEMP, ["tas", "tbot_af"]),
                                             (PRECIP, ["pr", "prec_af"])])
    def test_the_file_is_its_own_domain_and_names_the_models_field(self, tmp_path,
                                                                   stream, pair):
        root = ET.fromstring(W.stream_text(stream, tmp_path / "w", NAME))
        dom = root.find("domainInfo")
        names = dict(line.split() for line in
                     dom.find("variableNames").text.strip().splitlines())
        assert names == {"time": "time", "xc": "lon", "yc": "lat", "area": "area",
                         "mask": "mask"}
        for part in (dom, root.find("fieldInfo")):
            assert part.find("filePath").text.strip() == str(tmp_path / "w")
            assert part.find("fileNames").text.strip() == NAME
        fi = root.find("fieldInfo")
        assert fi.find("variableNames").text.split() == pair
        assert fi.find("offset").text.strip() == f"-{W.LEAD_SECONDS}"
        assert root.find("dataSource").text.strip() == "GENERIC"


# ── the namelist lines and the settings check ────────────────────────────────
class TestTheNamelist:

    def test_the_lines_keep_the_controls_interpolation_and_add_lower(self, tmp_path):
        ctl = _control_case(tmp_path)
        assert W.datm_lines(ctl, [TEMP, PRECIP]) == [
            ("datm", "anomaly_forcing", f"'{TEMP}', '{PRECIP}'"),
            ("datm", "tintalgo", '"linear", "linear", "lower", "lower", "lower"')]
        assert W.datm_lines(ctl, [TEMP])[1][2] == '"linear", "linear", "lower", "lower"'

    def test_a_control_whose_lists_disagree_is_refused(self, tmp_path):
        ctl = _control_case(tmp_path)
        (ctl / "CaseDocs" / "datm_in").write_text(_datm_in(
            "/d.nc", edit=lambda L: L["tintalgo"].pop()))
        with pytest.raises(W.WeatherError, match="2 tintalgo entries for 3 streams"):
            W.datm_lines(ctl, [TEMP])
        shutil.rmtree(ctl / "CaseDocs")
        with pytest.raises(W.WeatherError, match="0 tintalgo entries"):
            W.datm_lines(ctl, [TEMP])

    def test_what_the_run_must_hold_is_every_per_stream_list_grown(self, tmp_path):
        want = W.settings_expected(_control_case(tmp_path), [TEMP, PRECIP])["datm_in"]
        assert want["anomaly_forcing"] == [TEMP, PRECIP]
        assert want["streams"] == {"appends": 2, "starts": [f"datm.streams.txt.{TEMP}",
                                                            f"datm.streams.txt.{PRECIP}"]}
        assert want["tintalgo"] == {"appends": 2, "values": ["lower", "lower"]}
        grown = sorted(k for k, v in want.items() if v == {"appends": 2})
        assert grown == sorted(set(CONTROL_LISTS) - {"streams", "tintalgo"})
        assert set(W.PER_STREAM_KEYS) == set(CONTROL_LISTS) - {"streams"}
        assert "domainfile" not in want and "vectors" not in want

    # Case names are put on one footing in namelist values before comparing,
    # so they must not occur in the test's own paths.
    def _diff(self, tmp_path, streams=(TEMP, PRECIP), edit=None, expect=(TEMP, PRECIP),
              base=None, cime_form=False):
        ctl = _control_case(tmp_path, name="synthctl", base=base, cime_form=cime_form)
        run = tmp_path / "synthctl_warm"
        (run / "CaseDocs").mkdir(parents=True)
        (run / "CaseDocs" / "datm_in").write_text(_datm_in(
            W.domain_file(ctl), streams=streams, edit=edit, base=base, cime_form=cime_form))
        shutil.copy(ctl / "CaseDocs" / "lnd_in", run / "CaseDocs" / "lnd_in")
        want = W.settings_expected(ctl, list(expect))
        return CH.settings_differences(run, ctl, want, run.name, ctl.name)

    def test_a_run_as_cime_writes_it_passes(self, tmp_path):
        sd = self._diff(tmp_path)
        assert sd["pass"], sd["reasons"]
        f = sd["files"]["datm_in"]
        assert f["unexpected"] == [] and f["wrong"] == [] and f["no_op"] == []
        assert "anomaly_forcing" in f["differences"] and "domainfile" not in f["differences"]
        assert self._diff(tmp_path / "t", streams=(TEMP,), expect=(TEMP,))["pass"]

    def test_cimes_own_form_is_read_as_written(self, tmp_path):
        # single-quoted anomaly_forcing, the streams over continuation lines
        sd = self._diff(tmp_path, cime_form=True)
        assert sd["pass"], sd["reasons"]
        assert sd["files"]["datm_in"]["differences"]["anomaly_forcing"]["run"] == [TEMP,
                                                                                   PRECIP]

    def test_a_control_with_one_stream_has_its_lists_grown_too(self, tmp_path):
        sd = self._diff(tmp_path, base=ONE_STREAM)
        assert sd["pass"], sd["reasons"]
        assert sorted(sd["files"]["datm_in"]["differences"]) == sorted(
            set(CONTROL_LISTS) | {"anomaly_forcing"})

    @pytest.mark.parametrize("what,edit,key", [
        ("a list not grown", lambda L: L["mapalgo"].__delitem__(slice(3, None)), "mapalgo"),
        ("a control entry changed", lambda L: L["tintalgo"].__setitem__(0, '"lower"'),
         "tintalgo"),
        ("interpolation not lower", lambda L: L["tintalgo"].__setitem__(4, '"linear"'),
         "tintalgo"),
        ("the streams swapped", lambda L: L["streams"].__setitem__(
            slice(3, None), [L["streams"][4], L["streams"][3]]), "streams")])
    def test_a_run_whose_lists_are_not_the_controls_plus_the_streams_fails(
            self, tmp_path, what, edit, key):
        sd = self._diff(tmp_path, edit=edit)
        assert not sd["pass"] and sd["files"]["datm_in"]["wrong"] == [key], what

    def test_the_reason_says_what_the_list_should_be(self, tmp_path):
        # CIME's own default for the anomaly streams is "nearest" (seen on a
        # scratch clone of the Naches control, 2026-10-25): the second half of
        # every month would take the next month's record
        sd = self._diff(tmp_path, edit=lambda L: L["tintalgo"].__setitem__(
            slice(3, None), ['"nearest"', '"nearest"']))
        assert sd["reasons"] == [
            "datm_in: tintalgo is ['linear', 'linear', 'lower', 'nearest', 'nearest'], the "
            "change sets the control's ['linear', 'linear', 'lower'] with 'lower', 'lower' "
            "appended"]

    def test_a_stream_the_run_did_not_ask_for_fails(self, tmp_path):
        sd = self._diff(tmp_path, streams=(TEMP, PRECIP), expect=(TEMP,))
        assert not sd["pass"]
        assert set(sd["files"]["datm_in"]["wrong"]) >= {"anomaly_forcing", "streams"}

    def test_a_difference_no_change_explains_still_fails(self, tmp_path):
        sd = self._diff(tmp_path, edit=lambda L: L["taxmode"].__setitem__(0, '"extend"'))
        assert not sd["pass"] and "taxmode" in sd["files"]["datm_in"]["wrong"]
        ctl = _control_case(tmp_path / "x", name="synthctl")
        run = tmp_path / "x" / "synthctl_warm"
        (run / "CaseDocs").mkdir(parents=True)
        (run / "CaseDocs" / "datm_in").write_text(_datm_in("/another/domain.nc",
                                                           streams=(TEMP, PRECIP)))
        shutil.copy(ctl / "CaseDocs" / "lnd_in", run / "CaseDocs" / "lnd_in")
        sd = CH.settings_differences(run, ctl, W.settings_expected(ctl, [TEMP, PRECIP]),
                                     run.name, ctl.name)
        assert sd["files"]["datm_in"]["unexpected"] == ["domainfile"]


class TestTheCaseFiles:

    def test_each_written_file_must_be_in_the_case_unchanged(self, tmp_path):
        (tmp_path / "CaseDocs").mkdir()
        p = tmp_path / "CaseDocs" / f"datm.streams.txt.{TEMP}"
        p.write_text("as written")
        want = {f"CaseDocs/datm.streams.txt.{TEMP}": W.sha256(p)}
        assert CH.case_files_differences(tmp_path, want)["pass"]
        p.write_text("rewritten")
        bad = CH.case_files_differences(tmp_path, want)
        assert bad["reasons"] == [f"CaseDocs/datm.streams.txt.{TEMP} is not the file "
                                  f"written for the run (sha256 differs)"]
        p.unlink()
        assert "is missing" in CH.case_files_differences(tmp_path, want)["reasons"][0]

    def test_the_settings_stage_checks_them(self, tmp_path):
        ctl = _control_case(tmp_path)
        run = tmp_path / "run"
        (run / "CaseDocs").mkdir(parents=True)
        for f in ("datm_in", "lnd_in"):
            shutil.copy(ctl / "CaseDocs" / f, run / "CaseDocs" / f)
        p = run / "CaseDocs" / f"datm.streams.txt.{TEMP}"
        p.write_text("as written")
        expect = tmp_path / "expectations.json"
        expect.write_text(json.dumps({"settings": {}, "output": [], "case_files": {
            f"CaseDocs/datm.streams.txt.{TEMP}": W.sha256(p)}}))
        argv = ["settings", "--case-dir", str(run), "--control-case-dir", str(ctl),
                "--expect", str(expect), "--job", "7", "--out", str(tmp_path / "chk")]
        assert RG.main(argv) == 0
        rec = json.loads((tmp_path / "chk" / "settings_7.json").read_text())
        assert rec["case_files"] == {f"CaseDocs/datm.streams.txt.{TEMP}":
                                     {"present": True, "same": True}}
        p.write_text("rewritten")
        assert RG.main(argv) == 1


# ── one run's inputs ─────────────────────────────────────────────────────────
class TestInputs:

    def test_everything_one_run_needs_is_written_and_named(self, tmp_path):
        ctl = _control_case(tmp_path)
        out = W.inputs(_params(), YEARS, tmp_path / "inputs", ctl)
        wdir = (tmp_path / "inputs" / "weather").resolve()
        sha = out["anomaly"]["sha256"]
        assert out["anomaly"]["path"] == str(wdir / f"weather_anomaly_{sha[:12]}.nc")
        assert W.sha256(out["anomaly"]["path"]) == sha
        assert out["anomaly"]["streams"] == [TEMP, PRECIP]
        assert out["anomaly"]["cells"] == 20 and out["anomaly"]["region"] is None
        assert sorted(out["user_files"]) == [f"user_datm.streams.txt.{PRECIP}",
                                             f"user_datm.streams.txt.{TEMP}"]
        for name, m in out["user_files"].items():
            assert Path(m["from"]).parent == wdir and m["sha256"] == W.sha256(m["from"])
            text = Path(m["from"]).read_text()
            assert str(wdir) in text and Path(out["anomaly"]["path"]).name in text
        assert out["case_files"] == {
            f"CaseDocs/datm.streams.txt.{s}": out["user_files"][
                f"user_datm.streams.txt.{s}"]["sha256"] for s in (TEMP, PRECIP)}
        assert out["user_nl"] == W.datm_lines(ctl, [TEMP, PRECIP])
        assert out["settings"] == W.settings_expected(ctl, [TEMP, PRECIP])
        rules = {r["anomaly_var"]: r for r in out["output"]}
        assert sorted(rules) == ["pr", "tas"]
        assert {r["anomaly_sha256"] for r in out["output"]} == {sha}
        assert {(k, r["stream"], tuple(r["fields"]), r["how"], r["tolerance"], r["floor"])
                for k, r in rules.items()} == {
            ("tas", "h1", ("TBOT",), "offset", 1e-3, W.PRECIP_FLOOR),
            ("pr", "h1", ("RAIN", "SNOW"), "factor", 1e-5, W.PRECIP_FLOOR)}

    def test_the_file_is_named_by_its_content_and_never_overwritten(self, tmp_path):
        ctl = _control_case(tmp_path)
        a = W.inputs(_params(), YEARS, tmp_path / "i", ctl)
        b = W.inputs(_params(), YEARS, tmp_path / "i", ctl)
        assert a["anomaly"]["path"] == b["anomaly"]["path"]
        c = W.inputs(_params(temperature_K=3.0), YEARS, tmp_path / "i", ctl)
        assert c["anomaly"]["path"] != a["anomaly"]["path"]
        assert W.sha256(a["anomaly"]["path"]) == a["anomaly"]["sha256"]
        assert not list((tmp_path / "i" / "weather").glob("*.writing.nc"))

    def test_a_field_left_alone_is_still_checked_to_be_the_controls(self, tmp_path):
        ctl = _control_case(tmp_path)
        out = W.inputs(_params(precip_factor=1.0), YEARS, tmp_path / "i", ctl)
        assert out["anomaly"]["streams"] == [TEMP]
        assert out["user_nl"][0] == ("datm", "anomaly_forcing", f"'{TEMP}'")
        assert list(out["user_files"]) == [f"user_datm.streams.txt.{TEMP}"]
        assert [r["why"] for r in out["output"]] == [W.ANOMALY["temperature_K"]["why"],
                                                     W.ANOMALY["precip_factor"]["unchanged"]]
        with netCDF4.Dataset(out["anomaly"]["path"]) as f:
            assert np.all(np.asarray(f["pr"][:]) == 1.0)

    def test_a_registered_region_takes_its_cells_only(self, tmp_path):
        g = fx.grid()
        poly = _box(tmp_path / "box.geojson")
        entry = {"regions": {"upper": {"polygon": str(poly), "what": "a box"}}}
        out = W.inputs(_params(region="upper"), YEARS, tmp_path / "i",
                       _control_case(tmp_path), entry)
        assert out["anomaly"]["cells"] == 4 and out["anomaly"]["region"] == "upper"
        with netCDF4.Dataset(out["anomaly"]["path"]) as f:
            june = np.asarray(f["tas"][17])                  # 2015-06
        want = np.zeros(g["shape"], bool)
        want[0:2, 1:3] = True
        assert np.array_equal(june == 2.0, want)
        with pytest.raises(W.WeatherError, match="not registered"):
            W.inputs(_params(region="lower"), YEARS, tmp_path / "j",
                     _control_case(tmp_path / "c2"), entry)

    def test_a_region_given_as_a_gridded_field_takes_its_cells_only(self, tmp_path):
        mask = np.zeros(fx.grid()["shape"], bool)
        mask[2, 1:4] = True
        mask[0, 0] = True
        out, _ = _built(tmp_path, _params(), mask=mask)
        assert out["anomaly"]["cells"] == 4 and out["anomaly"]["region"] == SEL
        with netCDF4.Dataset(out["anomaly"]["path"]) as f:
            assert np.array_equal(np.asarray(f["tas"][17]) == 2.0, mask)

    @pytest.mark.parametrize("kind", ["polygon", "field"])
    def test_the_region_is_the_analysis_region_before_its_vegetated_cut(self, tmp_path,
                                                                        kind):
        # the weather falls on every cell of the region; the numbers report
        # the region's vegetated cells: the two must be one region. Both
        # regions hold a cell that is not vegetated (the urban cell (0, 0),
        # the lake cell (3, 1)), so the cut is exercised.
        import domain_analysis as DA
        g = fx.grid()
        surf = cf.write_surface(tmp_path / "surf.nc", g)
        if kind == "polygon":
            registered = {"r": {"polygon": str(_box(tmp_path / "box.geojson", cols=(0, 2)))}}
            n_weather, n_reported = 6, 5
        else:
            m = np.zeros(g["shape"], bool)
            m[0, 0:2] = True
            m[3, 1:3] = True
            registered = {"r": {"field": str(_field_region(tmp_path / "f.nc", m)),
                                "var": SEL, "op": ">", "value": 0.5}}
            n_weather, n_reported = 4, 2
        _history(tmp_path / "run", "run")
        ms = DA.masks(G.open_stream(tmp_path / "run", "h1").grid, surf, ["r"], registered)
        assert ms["problems"] == []
        veg = ms["masks"].mask(DA.VEGETATED).array
        reported = ms["masks"].mask("r, vegetated").array
        dom = W.read_domain(_domain(tmp_path / "domain.nc"))
        weather = W.region_mask(W.grid_from_domain(dom), "r", registered)
        assert weather.sum() == n_weather and reported.sum() == n_reported
        assert np.array_equal(weather & veg, reported)

    def test_a_region_off_the_grid_is_refused_naming_the_run(self, tmp_path):
        far = tmp_path / "far.geojson"
        far.write_text(json.dumps({"type": "Polygon", "coordinates": [[
            [10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0], [10.0, 10.0]]]}))
        entry = {"regions": {"far": {"polygon": str(far)}}}
        ctl = _control_case(tmp_path)
        with pytest.raises(W.WeatherError, match="holds no cell"):
            W.inputs(_params(region="far"), YEARS, tmp_path / "i", ctl, entry)
        recipe = DC.Recipe(case="ctl", case_root=str(tmp_path / "cases"),
                           srcroot="/src/E3SM", compset="C",
                           xml=[("RUN_STARTDATE", "2015-01-01")], user_nl={},
                           start="2015-01-01", copied_from=str(ctl))
        run = {"name": "warm", "changes": [{"id": "weather_scale",
                                            "params": _params(region="far")}]}
        with pytest.raises(CH.ChangeError, match="run warm: weather_scale: .*holds no cell"):
            CH.build_run(recipe, run, tmp_path / "j", YEARS, control_case_dir=str(ctl),
                         control_entry=entry)

    def test_only_active_cells_are_changed(self, tmp_path):
        d = _domain(tmp_path / "domain.nc", inactive=[(1, 1), (1, 2)])
        ctl = _control_case(tmp_path, domain=d)
        out = W.inputs(_params(), YEARS, tmp_path / "i", ctl)
        assert out["anomaly"]["cells"] == 18
        with netCDF4.Dataset(out["anomaly"]["path"]) as f:
            june = np.asarray(f["tas"][17])
        assert june[1, 1] == 0.0 and june[1, 2] == 0.0 and june[0, 0] == 2.0
        m = np.zeros(fx.grid()["shape"], bool)
        m[1, 1:3] = True
        entry = {"regions": {SEL: {"field": str(_field_region(tmp_path / "s.nc", m)),
                                   "var": SEL, "op": ">", "value": 0.5}}}
        with pytest.raises(W.WeatherError, match="reaches no active cell .* in region "
                                                  "'sel'"):
            W.inputs(_params(region=SEL), YEARS, tmp_path / "j", ctl, entry)

    def test_a_region_needs_a_regular_grid(self, tmp_path):
        dom = W.read_domain(_domain(tmp_path / "domain.nc"))
        dom["yc"] = dom["yc"] + 0.001 * np.arange(dom["yc"].shape[1])[None, :]
        with pytest.raises(W.WeatherError, match="not a regular latitude-longitude"):
            W.grid_from_domain(dom)
        dom = W.read_domain(_domain(tmp_path / "domain.nc"))
        dom["xc"] = dom["xc"] + 0.001 * np.arange(dom["xc"].shape[0])[:, None]
        with pytest.raises(W.WeatherError, match="not a regular latitude-longitude"):
            W.grid_from_domain(dom)

    def test_a_one_row_domain_is_refused(self, tmp_path):
        d = _domain(tmp_path / "domain.nc", g=fx.grid(nlat=1, nlon=5))
        with pytest.raises(W.WeatherError, match="one row of cells"):
            W.inputs(_params(), YEARS, tmp_path / "i", _control_case(tmp_path, domain=d))

    def test_a_control_that_already_reads_anomaly_streams_is_refused(self, tmp_path):
        ctl = _control_case(tmp_path, streams=(TEMP,))
        with pytest.raises(W.WeatherError, match="already reads anomaly streams"):
            W.inputs(_params(), YEARS, tmp_path / "i", ctl)

    @pytest.mark.parametrize("years,match", [([2004, 2005, 2006], r"lie in 2007-2299; "
                                                                  r"this one has \[2004, "
                                                                  r"2005, 2006\]"),
                                             ([2299, 2300], r"has \[2300\]")])
    def test_years_the_anomaly_streams_do_not_cover_are_refused(self, tmp_path, years,
                                                                match):
        with pytest.raises(W.WeatherError, match=match):
            W.inputs(_params(), years, tmp_path / "i", _control_case(tmp_path))

    def test_nothing_to_change_or_no_control_case_is_refused(self, tmp_path):
        with pytest.raises(W.WeatherError, match="changes nothing"):
            W.inputs(_params(temperature_K=0.0, precip_factor=1.0), YEARS, tmp_path / "i",
                     _control_case(tmp_path))
        with pytest.raises(W.WeatherError, match="changes nothing"):
            W.inputs(_params(temperature_K=0.0, precip_factor=1.00000001), YEARS,
                     tmp_path / "i", _control_case(tmp_path / "f"))
        with pytest.raises(W.WeatherError, match="no resolved datm_in"):
            W.inputs(_params(), YEARS, tmp_path / "i", tmp_path / "nowhere")
        ctl = _control_case(tmp_path / "x", domain=tmp_path / "gone.nc")
        with pytest.raises(W.WeatherError, match="no readable domainfile"):
            W.inputs(_params(), YEARS, tmp_path / "i", ctl)


# ── the year check ───────────────────────────────────────────────────────────
class TestTheYearCheck:

    def _pair(self, tmp_path, params, mask=None, domain=None, years=YEARS):
        ctl_run, run = tmp_path / "ctl_run", tmp_path / "run"
        _history(ctl_run, "ctl")
        _history(run, "run")
        out, rules = _built(tmp_path, params, mask=mask, domain=domain, years=years)
        return G.open_stream(ctl_run, "h1"), run, rules

    def _check(self, run, ref, rules, year=2015):
        return [W.check_year(G.open_stream(run, "h1"), ref, r, year) for r in rules]

    def test_a_run_made_as_datm_makes_it_passes_every_year(self, tmp_path):
        mask = np.zeros(fx.grid()["shape"], bool)
        mask[1:, :3] = True
        ref, run, rules = self._pair(tmp_path, _params(), mask=mask)
        _as_datm_would(run, offset=2.0, factor=0.8, months=[6, 7, 8], mask=mask)
        for y in YEARS:
            t, p = self._check(run, ref, rules, y)
            assert t["pass"] and t["max_error"] < 1e-4 and t["compared"] == 365 * 20, t
            assert t["changed_compared"] == 92 * 9 and t["unpaired"] == 0
            assert p["pass"] and p["max_relative_error"] < 1e-6, p
            assert p["per_field"]["RAIN"]["compared"] == 365 * 20
            assert p["dry_days_with_precipitation"] == 0 and "reason" not in p

    @pytest.mark.parametrize("what,kw", [
        ("a month too early", dict(months=[5, 6, 7])),
        ("every cell, not the region's", dict(mask=None)),
        ("half the amount", dict(offset=1.0)),
        ("no amount at all", dict(offset=0.0))])
    def test_temperature_not_as_the_file_says_fails(self, tmp_path, what, kw):
        mask = np.zeros(fx.grid()["shape"], bool)
        mask[1:, :3] = True
        ref, run, rules = self._pair(tmp_path, _params(), mask=mask)
        args = dict(offset=2.0, factor=0.8, months=[6, 7, 8], mask=mask)
        args.update(kw)
        _as_datm_would(run, **args)
        t, _ = self._check(run, ref, rules)
        assert not t["pass"] and t["max_error"] >= 1.0 - 1e-3, what
        assert t["reason"].startswith("off by up to")

    def test_precipitation_not_as_the_file_says_fails(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params())
        _as_datm_would(run, offset=2.0, factor=0.9, months=[6, 7, 8])
        _, p = self._check(run, ref, rules)
        assert not p["pass"] and abs(p["max_relative_error"] - 0.125) < 1e-4
        assert p["reason"].startswith("relative error up to 0.125")

    def test_rain_and_snow_are_compared_one_by_one(self, tmp_path):
        # DATM multiplies each, after the split: a run that moved snow into
        # rain keeps the sum, and must fail
        ref, run, rules = self._pair(tmp_path, _params(temperature_K=0.0,
                                                       precip_factor=1.0 + 1e-3,
                                                       months=[1]))
        _as_datm_would(run, factor=1.0 + 1e-3, months=[1])
        assert all(c["pass"] for c in self._check(run, ref, rules))
        for p in sorted(run.glob("*.elm.h1.*.nc")):
            with netCDF4.Dataset(p, "a") as d:
                rain, snow = np.asarray(d["RAIN"][:], float), np.asarray(d["SNOW"][:], float)
                d["RAIN"][:] = rain + 0.5 * snow
                d["SNOW"][:] = 0.5 * snow
        _, pr = self._check(run, ref, rules)
        assert not pr["pass"]
        assert pr["per_field"]["SNOW"]["max_relative_error"] > 0.4

    def test_a_field_left_alone_that_moved_fails(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params(precip_factor=1.0))
        _as_datm_would(run, offset=2.0, months=[6, 7, 8], only=("TBOT",))
        _as_datm_would(run, factor=1.1, months=[3], only=("RAIN", "SNOW"))
        t, p = self._check(run, ref, rules)
        assert t["pass"] and not p["pass"] and abs(p["max_relative_error"] - 0.1) < 1e-4

    def test_dry_days_must_stay_dry(self, tmp_path):
        ref_dir = tmp_path / "ctl_run"
        ref, run, rules = self._pair(tmp_path, _params(temperature_K=0.0, precip_factor=2.0))
        _as_datm_would(run, factor=2.0, months=[6, 7, 8])
        july = list(range(181, 212))                     # 0-based days of July 2015
        for d_ in (ref_dir, run):                        # a dry July in cell (0, 0)
            for day in july:
                _set_cells(d_, ("RAIN", "SNOW"), [(0, 0)], 0.0, day=day)
        ref = G.open_stream(ref_dir, "h1")
        _, p = self._check(run, ref, rules)
        assert p["pass"] and p["per_field"]["RAIN"]["compared"] == 365 * 20 - 31, p
        _set_cells(run, ("RAIN",), [(0, 0)], 1e-6, day=july[0])
        _, p = self._check(run, ref, rules)
        assert not p["pass"] and p["dry_days_with_precipitation"] == 1
        assert "1 dry cell-day(s) of the control have precipitation" in p["reason"]

    def test_a_control_day_drier_than_the_floor_may_grow_by_the_factor_only(self, tmp_path):
        ref_dir = tmp_path / "ctl_run"
        ref, run, rules = self._pair(tmp_path, _params(temperature_K=0.0, precip_factor=2.0,
                                                       months=W.ALL_MONTHS))
        _as_datm_would(run, factor=2.0)
        _set_cells(ref_dir, ("RAIN",), [(0, 0)], 0.9e-9)  # below the floor, not zero
        _set_cells(run, ("RAIN",), [(0, 0)], 1.8e-9)
        ref = G.open_stream(ref_dir, "h1")
        assert all(c["pass"] for c in self._check(run, ref, rules))
        _set_cells(run, ("RAIN",), [(0, 0)], 2.5e-9)
        _, p = self._check(run, ref, rules)
        assert not p["pass"] and p["dry_days_with_precipitation"] == 365

    def test_a_factor_of_zero_reads_as_the_share_kept(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params(temperature_K=0.0, precip_factor=0.0))
        _as_datm_would(run, factor=0.0, months=[6, 7, 8])
        assert all(c["pass"] for c in self._check(run, ref, rules))
        ref2, run2, rules2 = self._pair(tmp_path / "b", _params(temperature_K=0.0,
                                                                precip_factor=0.0))
        _, p = self._check(run2, ref2, rules2)           # untouched: it kept everything
        assert not p["pass"] and p["max_relative_error"] == pytest.approx(1.0)
        assert p["reason"].startswith("relative error up to 1 ")

    def test_a_one_step_slip_at_a_month_boundary_is_seen_for_a_small_change(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params(temperature_K=0.3, precip_factor=1.0))
        _as_datm_would(run, offset=0.3, months=[6, 7, 8], only=("TBOT",))
        assert self._check(run, ref, rules)[0]["pass"]
        # what the lead removes: 1 June keeping May's value for one step in
        # 48, and 1 September August's
        every = [(j, i) for j in range(4) for i in range(5)]
        _set_cells(run, ("TBOT",), every, lambda x: x - 0.3 / 48, day=151)
        _set_cells(run, ("TBOT",), every, lambda x: x + 0.3 / 48, day=243)
        t, _ = self._check(run, ref, rules)
        assert not t["pass"] and t["max_error"] == pytest.approx(0.3 / 48, abs=1e-4)

    def test_a_cell_day_reported_by_one_run_only_fails(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params())
        _as_datm_would(run, offset=2.0, factor=0.8, months=[6, 7, 8])
        for day in range(9, 19):
            _set_cells(run, ("TBOT",), [(2, 2)], np.nan, day=day)
        t, p = self._check(run, ref, rules)
        assert not t["pass"] and t["unpaired"] == 10
        assert t["reason"] == "10 cell-day(s) have a value in one run and not the other " \
                              "(TBOT 10)"
        assert p["pass"]

    def test_a_change_that_reached_no_reported_cell_fails(self, tmp_path):
        mask = np.zeros(fx.grid()["shape"], bool)
        mask[1, 1:3] = True
        ref_dir = tmp_path / "ctl_run"
        ref, run, rules = self._pair(tmp_path, _params(), mask=mask)
        for d_ in (ref_dir, run):                        # neither reports those cells
            _set_cells(d_, ("TBOT", "RAIN", "SNOW"), [(1, 1), (1, 2)], np.nan)
        ref = G.open_stream(ref_dir, "h1")
        t, p = self._check(run, ref, rules)              # a run identical to its control
        for c in (t, p):
            assert not c["pass"] and c["changed_compared"] == 0 and c["unpaired"] == 0
            assert "the change reached no cell the history reports" in c["reason"]

    def test_nothing_to_compare_fails_saying_so(self, tmp_path):
        ref_dir = tmp_path / "ctl_run"
        ref, run, rules = self._pair(tmp_path, _params())
        every = [(j, i) for j in range(4) for i in range(5)]
        for d_ in (ref_dir, run):
            _set_cells(d_, ("TBOT",), every, np.nan)
        t = W.check_year(G.open_stream(run, "h1"), G.open_stream(ref_dir, "h1"), rules[0],
                         2015)
        assert not t["pass"] and t["reason"] == "no cell-day of 2015 could be compared"

    def test_an_anomaly_file_on_other_cells_is_refused(self, tmp_path):
        # one cell east, or one cell north: DATM itself would take either
        # within its 0.01 degrees
        for k, kw in enumerate((dict(shift_deg=0.0083333), dict(shift_lat=0.0083333))):
            d = _domain(tmp_path / f"shifted{k}.nc", **kw)
            ref, run, rules = self._pair(tmp_path / str(k), _params(), domain=d)
            with pytest.raises(W.WeatherError, match="not the run's history cells"):
                self._check(run, ref, rules)

    def test_float32_history_coordinates_match_the_domains(self, tmp_path):
        ref, run, rules = self._pair(tmp_path, _params())
        with netCDF4.Dataset(sorted(run.glob("*.nc"))[0]) as f:
            assert f["lat"].dtype == np.float32 and f["lon"].dtype == np.float32
        _as_datm_would(run, offset=2.0, factor=0.8, months=[6, 7, 8])
        assert all(c["pass"] for c in self._check(run, ref, rules))


class TestTheGateRule:

    def _setup(self, tmp_path):
        ctl_run, run = tmp_path / "ctl_run", tmp_path / "run"
        _history(ctl_run, "ctl")
        _history(run, "run")
        out, rules = _built(tmp_path, _params())
        _as_datm_would(run, offset=2.0, factor=0.8, months=[6, 7, 8])
        return G.open_stream(ctl_run, "h1"), run, out, rules

    def test_the_rule_passes_and_its_record_names_both_fields(self, tmp_path):
        ref, run, _, rules = self._setup(tmp_path)
        ex = RG.expected_output(run, 2015, rules, reference=ref)
        assert [e["pass"] for e in ex] == [True, True]
        assert ex[0]["fields"] == ["TBOT"] and ex[1]["fields"] == ["RAIN", "SNOW"]

    def test_without_the_controls_output_it_fails_saying_so(self, tmp_path):
        ref, run, _, rules = self._setup(tmp_path)
        ex = RG.expected_output(run, 2015, rules)
        assert not any(e["pass"] for e in ex)
        assert "--reference" in ex[0]["reason"]
        ex = RG.expected_output(run, 2015, rules, reference_error="/x: no h1 history")
        assert ex[0]["reason"] == "cannot read the control's output: /x: no h1 history"

    def test_the_controls_output_must_be_the_rules_stream(self, tmp_path):
        ref, run, _, rules = self._setup(tmp_path)
        ex = RG.expected_output(run, 2015, [dict(rules[0], stream="h2")], reference=ref)
        assert ex[0]["reason"] == "the control's output was opened as stream h1, and the " \
                                  "rule reads h2"

    def test_an_anomaly_file_changed_since_the_build_fails(self, tmp_path):
        ref, run, out, rules = self._setup(tmp_path)
        with netCDF4.Dataset(out["anomaly"]["path"], "a") as f:
            f["tas"][17] = np.full(fx.grid()["shape"], 3.0, np.float32)
        ex = RG.expected_output(run, 2015, rules, reference=ref)
        assert ex[0]["reason"] == "the anomaly file is not the one the run was built " \
                                  "with (sha256 differs)"

    def test_a_year_the_file_lacks_or_the_run_lacks_fails_without_raising(self, tmp_path):
        ref, run, _, rules = self._setup(tmp_path)
        _, rules2010 = _built(tmp_path / "b", _params(), years=[2010])
        ex = RG.expected_output(run, 2015, rules2010, reference=ref)
        assert not any(e["pass"] for e in ex)
        assert all(e["reason"].startswith("cannot check: the anomaly file holds no 2015 "
                                          "record") for e in ex)
        ex = RG.expected_output(run, 2019, rules, reference=ref)
        assert all(e["reason"].startswith("cannot check") and "2019" in e["reason"]
                   for e in ex)

    def test_failures_read_as_sentences(self, tmp_path):
        e = {"change": "weather_scale", "fields": ["TBOT"], "rule": "matches_anomaly",
             "why": "w", "max_error": 1.5, "tolerance": 0.001,
             "reason": "off by up to 1.5 (tolerance 0.001)"}
        assert RG._expected_reason(e) == ("expected weather_scale: TBOT matches_anomaly (w); "
                                          "off by up to 1.5 (tolerance 0.001)")
        ref, run, _, rules = self._setup(tmp_path)
        _as_datm_would(run, factor=1.1, months=[3], only=("RAIN", "SNOW"))
        ex = RG.expected_output(run, 2015, rules, reference=ref)
        assert RG._expected_reason(ex[1]).startswith(
            "expected weather_scale: RAIN+SNOW matches_anomaly (daily rain and daily snow "
            "are each the control's times the factor in the chosen months and cells, and "
            "the control's elsewhere); relative error up to 0.1 (tolerance 1e-05), RAIN "
            "0.1, SNOW 0.1")

    def test_the_year_gate_applies_it_with_the_controls_output(self, tmp_path):
        c = cf.write_case(tmp_path / "x")
        ctl = cf.write_case(tmp_path / "c", case="ctl")
        for d_ in (c["run_dir"], ctl["run_dir"]):
            for p in sorted(d_.glob("*.elm.h1.*.nc")):
                with netCDF4.Dataset(p, "a") as d:
                    v = d.createVariable("TBOT", "f4", ("time", "lat", "lon"))
                    v.units = "K"
                    v[:] = np.full(d["RAIN"].shape, 280.0, np.float32)
        _, rules = _built(tmp_path, _params())
        _as_datm_would(c["run_dir"], offset=2.0, factor=0.8, months=[6, 7, 8])
        expect = tmp_path / "expectations.json"
        expect.write_text(json.dumps({"settings": {}, "output": rules}))
        argv = ["year", "--case-dir", str(c["case_dir"]), "--run-dir", str(c["run_dir"]),
                "--year", "2016", "--job", "111", "--out", str(tmp_path / "chk"),
                "--expect", str(expect), "--reference", str(ctl["run_dir"]),
                "--reference-report-only"]

        def year(args):
            RG.main(args)
            return json.loads((tmp_path / "chk" / "year_111.json").read_text())["years"][
                "2016"]
        y = year(argv)
        assert [e["pass"] for e in y["expected"]] == [True, True], y["expected"]
        _as_datm_would(c["run_dir"], offset=1.0, months=[7], only=("TBOT",))
        y = year(argv)
        assert [e["pass"] for e in y["expected"]] == [False, True]
        assert any(r.startswith("expected weather_scale: TBOT matches_anomaly")
                   for r in y["reasons"])
        no_ref = [x for x in argv if x not in ("--reference", str(ctl["run_dir"]),
                                               "--reference-report-only")]
        y = year(no_ref)
        assert all("--reference" in e["reason"] for e in y["expected"])
        gone = [x if x != str(ctl["run_dir"]) else str(tmp_path / "gone") for x in argv]
        y = year(gone)
        assert all(e["reason"].startswith(f"cannot read the control's output: "
                                          f"{tmp_path / 'gone'}") for e in y["expected"])


# ── the change, end to end on the server's side ──────────────────────────────
class TestTheChange:

    def _control_recipe(self, tmp_path, ctl_case):
        return DC.Recipe(case="ctl", case_root=str(tmp_path / "cases"), srcroot="/src/E3SM",
                         compset="C", xml=[("RUN_STARTDATE", "2015-01-01")],
                         user_nl={"elm": ["use_nofire = .false."]}, start="2015-01-01",
                         copied_from=str(ctl_case))

    def test_build_run_writes_the_files_and_the_recipe_reads_them(self, tmp_path):
        ctl = _control_case(tmp_path)
        run = {"name": "warm", "changes": [{"id": "weather_scale", "params": {
            "temperature_K": 2.0, "months": [7, 6, 8]}}]}
        r, e = CH.build_run(self._control_recipe(tmp_path, ctl), run, tmp_path / "inputs",
                            YEARS, control_case_dir=str(ctl))
        assert r.case == "ctl_warm"
        assert f"anomaly_forcing = '{TEMP}'" in r.user_nl["datm"]
        assert 'tintalgo = "linear", "linear", "lower", "lower"' in r.user_nl["datm"]
        assert list(r.user_files) == [f"user_datm.streams.txt.{TEMP}"]
        assert r.check() == []
        assert e["changes"][0]["params"]["months"] == [6, 7, 8]
        assert e["changes"][0]["meaning"].startswith(
            "the difference from the control is the response to air 2 K warmer, in June, "
            "July, August, with every other input the control's")
        assert e["settings"]["datm_in"]["anomaly_forcing"] == [TEMP]
        assert list(e["case_files"]) == [f"CaseDocs/datm.streams.txt.{TEMP}"]
        assert [o["anomaly_var"] for o in e["output"]] == ["tas", "pr"]
        assert all(o["change"] == "weather_scale" for o in e["output"])
        with netCDF4.Dataset(e["output"][0]["anomaly_file"]) as f:
            assert f["time"].units == "days since 2014-01-01 00:00:00"
            assert len(f["time"]) == 48

    def test_the_recipe_alone_refuses_a_change_that_writes_files(self, tmp_path):
        run = {"name": "warm", "changes": [{"id": "weather_scale",
                                            "params": {"temperature_K": 2.0}}]}
        with pytest.raises(CH.ChangeError, match="build it with build_run"):
            CH.run_recipe(self._control_recipe(tmp_path, tmp_path), run)
        with pytest.raises(CH.ChangeError, match="build it with build_run"):
            CH.expectations(run, "/c/ctl")

    def test_a_writer_failure_names_the_run(self, tmp_path):
        run = {"name": "warm", "changes": [{"id": "weather_scale",
                                            "params": {"temperature_K": 2.0}}]}
        with pytest.raises(CH.ChangeError, match="run warm: weather_scale: .*datm_in"):
            CH.build_run(self._control_recipe(tmp_path, tmp_path), run, tmp_path / "i",
                         YEARS, control_case_dir=str(tmp_path / "none"))

    def test_the_text_of_cooling_wetting_drying_and_a_region(self):
        c = CH.resolve({"id": "weather_scale", "params": {
            "temperature_K": -1.5, "precip_factor": 1.2, "months": [1, 2],
            "region": "upper"}})
        assert c["meaning"] == ("the difference from the control is the response to air "
                                "1.5 K cooler and rain and snow multiplied by 1.2, in "
                                "January, February, over upper, with every other input "
                                "the control's")
        warm = CH.resolve({"id": "weather_scale", "params": {
            "temperature_K": 2.0, "precip_factor": 0.8, "months": [6, 7, 8],
            "region": "watershed"}})
        assert warm["meaning"] == ("the difference from the control is the response to "
                                   "air 2 K warmer and rain and snow multiplied by 0.8, in "
                                   "June, July, August, over watershed, with every other "
                                   "input the control's")
        dry = CH.resolve({"id": "weather_scale", "params": {"precip_factor": 0.5}})
        assert dry["meaning"] == ("the difference from the control is the response to "
                                  "rain and snow multiplied by 0.5, in every month, with "
                                  "every other input the control's")
        assert dry["params"]["temperature_K"] == 0.0

    def test_the_not_meaning_says_what_the_land_model_sees(self):
        nm = CH.resolve({"id": "weather_scale", "params": {"temperature_K": 2.0}})[
            "not_meaning"]
        for part in ("does not turn snow into rain", "relative humidity", "drier air",
                     "fire model", "dew and frost", "from the temperature it receives"):
            assert part in nm, part

    @pytest.mark.parametrize("params,match", [
        ({"temperature_K": 11.0}, "temperature_K"),
        ({"precip_factor": -0.1}, "precip_factor"),
        ({"temperature_K": float("nan")}, "is not a number"),
        ({"temperature_K": float("inf")}, "is not a number"),
        ({"precip_factor": 1.00000001}, "changes nothing"),
        ({"temperature_K": 1.0, "months": [0, 1]}, "not a list of distinct months"),
        ({"temperature_K": 1.0, "months": [3, 3]}, "not a list of distinct months"),
        ({"temperature_K": 1.0, "months": []}, "not a list of distinct months"),
        ({"temperature_K": 1.0, "region": ""}, "not a region's name")])
    def test_parameters_out_of_bounds_are_refused(self, params, match):
        with pytest.raises(CH.ChangeError, match=match):
            CH.resolve({"id": "weather_scale", "params": params})

    H1 = {"fields": ["RAIN", "SNOW", "TBOT"], "years": list(range(2006, 2301))}

    def _check(self, reuse=True, years=(2015, 2016), regions=None, h1=H1, other=(),
               **params):
        exp = {"control": {"name": "naches", "reuse": reuse}, "years": list(years),
               "runs": [{"name": "warm", "changes": [{"id": "weather_scale",
                                                     "params": params}] + list(other)}]}
        facts = {"start_year": years[0], "forcing_years": [2006, 2300],
                 "output_years": [2006, 2300]}
        if h1 is not None:
            facts["streams"] = {"h1": h1}
        return CH.check(exp, {"name": "naches", "facts": facts, "regions": regions or {}})

    def test_the_experiment_check_says_what_will_not_build(self):
        ok = self._check(temperature_K=2.0)
        assert ok["ok"], ok["wont_build"]
        assert not [u for u in ok["unusual"] if "weather_scale" in u["where"]], ok["unusual"]
        assert any("meaning: the difference from the control is the response to air 2 K "
                   "warmer" in f["what"] for f in ok["facts"])
        bad = self._check(reuse=False, temperature_K=2.0)
        assert any("needs a reused control" in w["what"] for w in bad["wont_build"])
        bad = self._check(region="upper", temperature_K=2.0)
        assert any("region 'upper' is not registered" in w["what"] for w in bad["wont_build"])
        ok = self._check(region="upper", regions={"upper": {"polygon": "/p.geojson"}},
                         temperature_K=2.0)
        assert ok["ok"], ok["wont_build"]

    def test_the_controls_daily_history_must_hold_the_fields_and_years(self):
        bad = self._check(h1={"fields": ["RAIN", "SNOW"], "years": [2015, 2016]},
                          temperature_K=2.0)
        assert any("(h1) lacks ['TBOT']" in w["what"] for w in bad["wont_build"])
        bad = self._check(h1=dict(self.H1, years=[2015]), temperature_K=2.0)
        assert any("has no complete year [2016]" in w["what"] for w in bad["wont_build"])
        odd = self._check(h1=None, temperature_K=2.0)
        assert odd["ok"] and any("could not confirm that the control's daily history"
                                 in u["what"] for u in odd["unusual"])

    def test_a_cooler_run_and_a_run_with_other_changes_are_flagged(self):
        cool = self._check(temperature_K=-2.0)
        assert cool["ok"] and any("a cooler run" in u["what"] for u in cool["unusual"])
        both = self._check(temperature_K=2.0, other=[{"id": "fire_off"}])
        assert both["ok"] and any("changes the weather and fire_off" in u["what"]
                                  and "joint effect" in u["what"] for u in both["unusual"])

    def test_two_weather_changes_in_one_run_are_refused(self):
        exp = {"control": {"name": "naches", "reuse": True}, "years": [2015],
               "runs": [{"name": "w", "changes": [
                   {"id": "weather_scale", "params": {"temperature_K": 1.0}},
                   {"id": "weather_scale", "params": {"precip_factor": 2.0}}]}]}
        out = CH.check(exp, {"name": "naches", "facts": {"start_year": 2015,
                                                         "forcing_years": [2006, 2300],
                                                         "output_years": [2006, 2300]}})
        assert any("both alter user_nl_datm:anomaly_forcing" in w["what"]
                   for w in out["wont_build"])

    def test_years_outside_the_anomaly_streams_will_not_build(self):
        out = self._check(years=(2005, 2006), temperature_K=1.0)
        assert any("must lie in 2007-2299" in w["what"] for w in out["wont_build"])
        assert self._check(years=(2007, 2008), temperature_K=1.0)["ok"]


# ── what the server publishes, and who reads it ──────────────────────────────
class TestWhatIsPublished:

    def test_the_published_meaning_names_no_placeholder_a_run_cannot_fill(self):
        c = next(c for c in CH.declare()["changes"] if c["id"] == "weather_scale")
        assert "menu_text" not in c and "{" not in c["meaning"]
        assert "temperature_K" in c["meaning"] and "precip_factor" in c["meaning"]
        for x in CH.declare()["changes"]:
            assert set(re.findall(r"\{(\w+)\}", x["meaning"])) <= set(x["params"]), x["id"]

    def test_no_change_claims_the_weather_is_the_controls(self):
        for c in CH.declare()["changes"]:
            if c["id"] != "weather_scale":
                assert "the weather is the control's" not in c["not_meaning"], c["id"]
                assert "weather unchanged" not in c["meaning"], c["id"]

    def test_the_servers_note_on_differences_holds_for_a_weather_run(self):
        import domain_analysis as DA
        text = " ".join(c["caveat"] for c in DA.LIMITATIONS["configuration"])
        assert "reads the control's own weather" not in text
        assert "weather_scale" in text

    def test_the_review_page_shows_the_run_and_the_whole_not_meaning(self):
        from agents.design_review import render
        from test_domain_front import CONTROLS
        brief = {"design_archetype": "whole_domain", "user_request": "what does +2 K do?",
                 "domain": {"name": "naches"},
                 "run_settings": {"resolved_period": {"yr_start": 2015, "yr_end": 2015,
                                                      "source": "user"}}}
        strategy = {"archetype": "whole_domain",
                    "domain_experiment": {
                        "control": {"name": "naches", "reuse": True},
                        "runs": [{"name": "warm", "changes": [{"id": "weather_scale",
                                                               "params": _params()}]}],
                        "years": [2015], "analysis": {"regions": [], "normal_years": []},
                        "caveats": []},
                    "domain_rulebook": {"controls": CONTROLS, "changes": CH.declare()},
                    "feasibility": {"verdict": "full", "why": "every change is available"}}
        page = render({"brief": brief}, strategy, run_name="r")
        line = next(x for x in page.splitlines() if "weather_scale means" in x)
        assert "{" not in line and "the chosen months" in line
        nm = next(c for c in CH.declare()["changes"] if c["id"] == "weather_scale")[
            "not_meaning"]
        assert f"  - it is NOT {nm}" in page


# ── through the server's tools ───────────────────────────────────────────────
WARM = [{"name": "warm", "changes": [{"id": "weather_scale", "params": {
    "temperature_K": 2.0, "months": [6, 7, 8]}}]}]


class TestThroughTheServer:

    def _with_datm(self, lab, tmp_path):
        dom = _domain(tmp_path / "domain.lnd.synth.nc")
        (lab["control_dir"] / "CaseDocs" / "datm_in").write_text(_datm_in(dom))

    def test_a_weather_run_is_built_with_its_files_and_checks(self, lab, tmp_path):
        self._with_datm(lab, tmp_path)
        _experiment(lab, WARM)
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"], out
        inp = lab["run_dir"] / "01_inputs" / "runs" / "warm"
        assert len(list((inp / "weather").glob("weather_anomaly_*.nc"))) == 1
        r = DC.Recipe.from_json((inp / "domain_recipe.json").read_text())
        assert list(r.user_files) == [f"user_datm.streams.txt.{TEMP}"]
        assert f"anomaly_forcing = '{TEMP}'" in r.user_nl["datm"]
        e = json.loads((inp / "domain_expectations.json").read_text())
        assert list(e["case_files"]) == [f"CaseDocs/datm.streams.txt.{TEMP}"]
        assert [o["rule"] for o in e["output"]] == ["matches_anomaly"] * 2
        spec = json.loads((inp / "domain_chain.json").read_text())
        assert spec["reference"]["run_dir"] == str(lab["control_run"])
        w = json.loads(lab["srv"].write_domain_case(str(lab["run_dir"]), run="warm"))
        assert w["ok"], w
        rec = lab["run_dir"] / "domain_case" / "warm"
        kept = rec / "user_files" / f"user_datm.streams.txt.{TEMP}"
        assert W.sha256(kept) == r.user_files[kept.name]["sha256"]
        case_script = (rec / DC.CASE_SCRIPT).read_text()
        assert f"{r.user_files[kept.name]['sha256']}  {kept.name}" in case_script
        stage = (rec / DC.STAGE_SCRIPT).read_text()
        assert f"--reference {lab['control_run']}" in stage
        assert '--expect "$REC/expectations.json"' in stage

    def test_a_region_and_two_years_reach_the_file_and_the_gate(self, lab, tmp_path):
        self._with_datm(lab, tmp_path)
        reg_path = tmp_path / "domain_controls.json"             # the lab's registry
        reg = json.loads(reg_path.read_text())
        reg["controls"]["synth"]["regions"] = {"box": {
            "polygon": str(_box(tmp_path / "box.geojson")), "what": "a box"}}
        reg_path.write_text(json.dumps(reg))
        _experiment(lab, [{"name": "warm", "changes": [{"id": "weather_scale", "params": {
            "temperature_K": 2.0, "precip_factor": 0.8, "months": [6, 7, 8],
            "region": "box"}}]}])
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"], out
        inp = lab["run_dir"] / "01_inputs" / "runs" / "warm"
        (anom,) = list((inp / "weather").glob("weather_anomaly_*.nc"))
        with netCDF4.Dataset(anom) as f:
            assert f["time"].units == "days since 2014-01-01 00:00:00"
            assert np.array_equal(np.asarray(f["time"][:]), W.month_starts(YEARS)[0])
            june16 = np.asarray(f["tas"][29])                    # 2016-06
        box = np.zeros(fx.grid()["shape"], bool)
        box[0:2, 1:3] = True
        assert np.array_equal(june16 == 2.0, box)
        # the gate, on the expectations the server wrote, over histories made
        # as DATM would make them
        c = cf.write_case(tmp_path / "x")
        ctl = cf.write_case(tmp_path / "c", case="ctl")
        for d_ in (c["run_dir"], ctl["run_dir"]):
            for p in sorted(d_.glob("*.elm.h1.*.nc")):
                with netCDF4.Dataset(p, "a") as d:
                    v = d.createVariable("TBOT", "f4", ("time", "lat", "lon"))
                    v.units = "K"
                    v[:] = np.full(d["RAIN"].shape, 280.0, np.float32)
        _as_datm_would(c["run_dir"], offset=2.0, factor=0.8, months=[6, 7, 8], mask=box)
        argv = ["year", "--case-dir", str(c["case_dir"]), "--run-dir", str(c["run_dir"]),
                "--year", "2016", "--job", "111", "--out", str(tmp_path / "chk"),
                "--expect", str(inp / "domain_expectations.json"),
                "--reference", str(ctl["run_dir"]), "--reference-report-only"]

        def year():
            RG.main(argv)
            return json.loads((tmp_path / "chk" / "year_111.json").read_text())["years"][
                "2016"]
        assert [e["pass"] for e in year()["expected"]] == [True, True]
        _as_datm_would(c["run_dir"], offset=-1.0, months=[6, 7, 8], mask=box,
                       only=("TBOT",))                             # half the amount
        assert [e["pass"] for e in year()["expected"]] == [False, True]

    def test_inputs_are_not_rebuilt_while_a_chain_may_run(self, lab, tmp_path):
        self._with_datm(lab, tmp_path)
        _experiment(lab, WARM)
        assert json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))["ok"]
        s = json.loads(lab["srv"].submit_domain_chain(str(lab["run_dir"]), run="warm"))
        assert s["ok"], s
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"] is False
        assert out["error"].startswith("run warm: ")
        assert out["error"].endswith("its inputs are not rebuilt while its chain may still "
                                     "run")

    def test_a_weather_run_beside_a_control_that_runs_too_will_not_build(self, lab, tmp_path):
        self._with_datm(lab, tmp_path)
        _experiment(lab, WARM, reuse=False)
        ctl = DC.recipe_from_case(lab["control_dir"], case="ctl")
        (lab["run_dir"] / "01_inputs" / "domain_recipe.json").write_text(ctl.to_json())
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"] is False and out["error"] == "the experiment will not build"
        assert any("needs a reused control" in w["what"] for w in out["wont_build"])
