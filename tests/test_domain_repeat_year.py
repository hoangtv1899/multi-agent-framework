"""A recorded year read in another model year: weather_repeat_year.

    pytest tests/test_domain_repeat_year.py -v

Each chain year is its own job, and a job sets which recorded years the
forcing reads before it runs (DATM_CLMNCEP_YR_START, _END, _ALIGN). These
tests hold the writer to the two-year form every year of a run gets and to
the controls it refuses, the chain's job script to setting it year by year,
and each year's two checks to what the run must show: the data atmosphere's
log for the year's job names the recorded year, and the run's daily TBOT,
RAIN and SNOW are the control's for that year, day for day. Across a
restart (measured on the Naches control, job 791045) the new years take
effect, and the first model step after it has the atmosphere's last output
from before the restart: at a year boundary, the record after the previous
year's recorded year. 1 January of such a year is reported, not held.
"""
import gzip
import json
import math
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
from test_domain_case import _run_stage, _stage_case
from test_domain_experiment import _experiment, lab  # noqa: F401
from test_domain_manager import srv  # noqa: F401
from test_domain_weather import _control_case, _datm_in, _domain

YEARS = [2015, 2016, 2017]
FORCING = "datm.streams.txt.CLMMOSARTTEST"
ENV_RUN = """<?xml version="1.0"?>
<file id="env_run.xml" version="2.0">
  <group id="run_component_datm">
    <entry id="DATM_CLMNCEP_YR_ALIGN" value="{align}">
    </entry>
    <entry id="DATM_CLMNCEP_YR_START" value="{start}">
    </entry>
    <entry id="DATM_CLMNCEP_YR_END" value="{end}">
    </entry>
  </group>
</file>
"""
STREAM = """<?xml version="1.0"?>
<file id="stream" version="1.0">
<dataSource>
   CLMNCEP
</dataSource>
<domainInfo>
  <variableNames>
     time    time
     LONGXY  lon
     LATIXY  lat
     area    area
     mask    mask
  </variableNames>
  <filePath>
     {path}
  </filePath>
  <fileNames>
     domain.nc
  </fileNames>
</domainInfo>
<fieldInfo>
   <variableNames>
     TBOT     tbot
   </variableNames>
   <filePath>
     {path}
   </filePath>
   <fileNames>
     clmforc.nldas.2013-01.nc
clmforc.nldas.2013-02.nc
   </fileNames>
   <offset>
      {offset}
   </offset>
</fieldInfo>
</file>
"""


def _forcing(dirpath, first_hours=0.0):
    """A forcing file whose records are hourly, the first `first_hours`
    after 00:00 on 1 January."""
    dirpath.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(dirpath / "clmforc.nldas.2013-01.nc", "w") as f:
        f.createDimension("time", None)
        t = f.createVariable("time", "f8", ("time",))
        t.units = "days since 2013-01-01 00:00:00"
        t[:] = (first_hours + np.arange(3)) / 24.0
    return dirpath


def _repeat_control(tmp_path, align=2013, start=2013, end=2023, name="ctl", first_hours=0.0,
                    offset=0):
    """A control case whose datm_in reads the forcing for its case's years,
    with the forcing's stream file and its first file."""
    cd = _control_case(tmp_path, name=name)
    (cd / "env_run.xml").write_text(ENV_RUN.format(align=align, start=start, end=end))
    path = _forcing(tmp_path / f"{name}_forcing", first_hours)
    (cd / "CaseDocs" / FORCING).write_text(STREAM.format(path=path, offset=offset))
    return cd


def _params(**m):
    return {"map": m or {"2016": 2015}}


def _atm_log(run_dir, job, *entries):
    """The data atmosphere's log of one job, with its streams lines."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(f"(shr_strdata_print)   streams ({k + 1:2d}) = {e}\n"
                   for k, e in enumerate(entries))
    with gzip.open(run_dir / f"atm.log.{job}.261026-101010.gz", "wt") as f:
        f.write("(shr_stream_init) Reading file " + FORCING + "\n" + text)


# ── histories whose years differ ─────────────────────────────────────────────
# The fixture's TBOT, RAIN and SNOW are the same every year, which would let a
# run that read the wrong year pass. Year Y gets TBOT + (Y - 2015) K and rain
# and snow times 1 + (Y - 2015) / 10.
def _year_of(start_day):
    return 2015 + int(math.floor(start_day + 1e-9)) // 365


def _distinct(run_dir, reads=None):
    """Make each year of a daily history distinct; `reads` {model year:
    recorded year} writes a run's year as the control's recorded year."""
    for p in sorted(Path(run_dir).glob("*.elm.h1.*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            tb = np.asarray(d["time_bounds"][:])
            for v in ("TBOT", "RAIN", "SNOW"):
                x = np.asarray(d[v][:], float)
                for r in range(tb.shape[0]):
                    if tb[r, 1] - tb[r, 0] < 0.5:
                        continue
                    y = _year_of(tb[r, 0])
                    w = (reads or {}).get(y, y)
                    x[r] = x[r] + (w - 2015) if v == "TBOT" else x[r] * (1 + (w - 2015) / 10)
                d[v][:] = x


def _pair(tmp_path, reads):
    ctl, run = tmp_path / "ctl_run", tmp_path / "run"
    fx.write_daily(ctl, YEARS, case="ctl", variables=("TBOT", "RAIN", "SNOW"))
    fx.write_daily(run, YEARS, case="run", variables=("TBOT", "RAIN", "SNOW"))
    _distinct(ctl)
    _distinct(run, reads)
    return G.open_stream(ctl, "h1"), run


def _rules(tmp_path, m):
    out = W.repeat_inputs({"map": m}, YEARS, _repeat_control(tmp_path / "c"))
    return [dict(r, change="weather_repeat_year") for r in out["output"]]


def _shift(run_dir, day, by, field="TBOT", cells=None):
    """Add `by` to one field on one day (0-based from 2015-01-01), in every
    cell or in `cells`; `by` NaN writes NaN."""
    for p in sorted(Path(run_dir).glob("*.elm.h1.*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            tb = np.asarray(d["time_bounds"][:])
            x = np.asarray(d[field][:], float)
            hit = False
            for r in range(tb.shape[0]):
                if tb[r, 1] - tb[r, 0] > 0.5 and int(round(tb[r, 0])) == day:
                    sel = (slice(None),) * 2 if cells is None else tuple(np.array(cells).T)
                    x[r][sel] = np.nan if (isinstance(by, float) and math.isnan(by)) \
                        else x[r][sel] + by
                    hit = True
            if hit:
                d[field][:] = x


# ── the writer ───────────────────────────────────────────────────────────────
class TestTheWriter:

    def test_every_year_reads_its_recorded_year_and_the_next_aligned_to_itself(self):
        m = W.year_map({"map": {"2016": 2015}}, YEARS)
        assert m == {2015: 2015, 2016: 2015, 2017: 2017}
        assert W.repeat_settings(m) == {
            2015: [["DATM_CLMNCEP_YR_START", 2015], ["DATM_CLMNCEP_YR_END", 2016],
                   ["DATM_CLMNCEP_YR_ALIGN", 2015]],
            2016: [["DATM_CLMNCEP_YR_START", 2015], ["DATM_CLMNCEP_YR_END", 2016],
                   ["DATM_CLMNCEP_YR_ALIGN", 2016]],
            2017: [["DATM_CLMNCEP_YR_START", 2017], ["DATM_CLMNCEP_YR_END", 2018],
                   ["DATM_CLMNCEP_YR_ALIGN", 2017]]}

    @pytest.mark.parametrize("m,carried", [
        ({"2016": 2015}, [2016, 2017]),               # 2015 after 2015, then 2017 after 2015
        ({"2016": 2015, "2017": 2016}, [2016]),       # a shifted sequence runs on
        ({"2015": 2018, "2016": 2019, "2017": 2020}, []),   # the first year starts afresh
        ({"2017": 2015}, [2017])])
    def test_a_year_whose_recorded_year_does_not_follow_carries_one_step(self, m, carried):
        assert W.carried_first_days(W.year_map({"map": m}, YEARS)) == carried

    def test_the_forcing_streams_are_the_ones_the_cases_years_set(self, tmp_path):
        f = W.forcing_streams(_repeat_control(tmp_path))
        assert f == {"align": 2013, "first": 2013, "last": 2023, "streams": [FORCING]}
        with pytest.raises(W.WeatherError, match="does not say, as whole years"):
            W.forcing_streams(_control_case(tmp_path / "c"))

    def test_cimes_unresolved_reference_is_followed(self, tmp_path):
        cd = _repeat_control(tmp_path, align="$DATM_CLMNCEP_YR_START")
        assert W.forcing_streams(cd)["align"] == 2013

    @pytest.mark.parametrize("what,make,match", [
        ("model years that are not recorded years",
         lambda t: _repeat_control(t, align=2000), "reads recorded year 2013 \\+ \\(Y - 2000\\)"),
        ("a forcing that starts at 01:30",
         lambda t: _repeat_control(t, first_hours=1.5), "start at 01:30 on 1 January"),
        ("a user stream file", lambda t: _with(_repeat_control(t), f"user_{FORCING}", "x"),
         "its file list stays fixed"),
        ("streams set in user_nl_datm",
         lambda t: _with(_repeat_control(t), "user_nl_datm", ' streams = "a 1 1 1"\n'),
         "sets the streams itself"),
        ("bias correction", lambda t: _datm_extra(_repeat_control(t), "bias_correct",
                                                  "BC.CRUNCEP.GPCP.Precip"),
         "also reads bias_correct streams"),
        ("anomaly forcing", lambda t: _datm_extra(_repeat_control(t), "anomaly_forcing",
                                                  "Anomaly.Forcing.Precip"),
         "also reads anomaly_forcing streams")])
    def test_a_control_the_two_year_form_cannot_serve_is_refused(self, tmp_path, what, make,
                                                                 match):
        with pytest.raises(W.WeatherError, match=match):
            W.forcing_streams(make(tmp_path))

    def test_an_offset_that_brings_the_first_record_to_midnight_is_accepted(self, tmp_path):
        cd = _repeat_control(tmp_path, first_hours=0.5, offset=-1800)
        assert W.forcing_streams(cd)["streams"] == [FORCING]

    def test_what_one_run_needs(self, tmp_path):
        out = W.repeat_inputs(_params(), YEARS, _repeat_control(tmp_path))
        assert out["repeat"] == {"map": {"2015": 2015, "2016": 2015, "2017": 2017},
                                 "carried_first_days": [2016, 2017], "streams": [FORCING]}
        assert sorted(out["year_settings"]) == ["2015", "2016", "2017"]
        assert out["settings"] == {} and "user_files" not in out
        years, equals = out["output"]
        assert years == {"stream": "datm_in", "rule": "datm_stream_years",
                         "streams": [FORCING],
                         "years": {"2015": [2015, 2015, 2016], "2016": [2016, 2015, 2016],
                                   "2017": [2017, 2017, 2018]},
                         "why": years["why"]}
        assert (equals["rule"], equals["stream"], equals["fields"], equals["tolerance"],
                equals["years"], equals["carried_first_days"]) == (
            "equals_control_year", "h1", ["TBOT", "RAIN", "SNOW"], 0.0,
            {"2015": 2015, "2016": 2015, "2017": 2017}, [2016, 2017])

    def test_a_recorded_year_needs_the_year_after_it_in_the_forcing(self, tmp_path):
        with pytest.raises(W.WeatherError, match=r"holds 2013-2023, .*\[2023\] do not"):
            W.repeat_inputs(_params(**{"2016": 2023}), YEARS, _repeat_control(tmp_path))
        with pytest.raises(W.WeatherError, match=r"\[2012\] do not"):
            W.repeat_inputs(_params(**{"2016": 2012}), YEARS, _repeat_control(tmp_path / "b"))

    def test_a_map_that_changes_nothing_is_refused(self, tmp_path):
        with pytest.raises(W.WeatherError, match="changes nothing"):
            W.repeat_inputs(_params(**{"2016": 2016}), YEARS, _repeat_control(tmp_path))


def _with(cd, name, text):
    (cd / name).write_text(text)
    return cd


def _datm_extra(cd, key, value):
    p = cd / "CaseDocs" / "datm_in"
    p.write_text(p.read_text().replace("&datm_nml\n", f'&datm_nml\n  {key} = "{value}"\n', 1))
    return cd


# ── the year's forcing years, from the data atmosphere's log ─────────────────
class TestTheYearsForcingYears:

    def test_the_run_read_the_recorded_year(self, tmp_path):
        rule = _rules(tmp_path, {"2016": 2015})[0]
        _atm_log(tmp_path / "run", 222, f"{FORCING} 2016 2015 2016",
                 "datm.streams.txt.presaero.clim_2000 1 1 1")
        out = W.check_stream_years(tmp_path / "run", rule, 2016, job=222)
        assert out["pass"] and out["streams"] == {FORCING: [2016, 2015, 2016]}
        assert out["log"] == "atm.log.222.261026-101010.gz"

    def test_a_run_that_kept_the_controls_years_fails_saying_so(self, tmp_path):
        rule = _rules(tmp_path, {"2016": 2015})[0]
        _atm_log(tmp_path / "run", 222, f"{FORCING} 2013 2013 2023")
        out = W.check_stream_years(tmp_path / "run", rule, 2016, job=222)
        assert not out["pass"]
        assert out["reason"] == (f"{FORCING} reads 2013 2013 2023, not 2016 2015 2016 "
                                 f"(align, first, last)")
        ex = RG.expected_output(tmp_path / "run", 2016, [rule], job=222)
        assert ex[0]["pass"] is False
        assert RG._expected_reason(ex[0]) == (
            f"expected weather_repeat_year: {FORCING} datm_stream_years ({rule['why']}); "
            f"{FORCING} reads 2013 2013 2023, not 2016 2015 2016 (align, first, last)")

    def test_a_log_that_names_no_forcing_stream_fails(self, tmp_path):
        rule = _rules(tmp_path, {"2016": 2015})[0]
        _atm_log(tmp_path / "run", 222, "datm.streams.txt.topo.observed 1 1 1")
        out = W.check_stream_years(tmp_path / "run", rule, 2016, job=222)
        assert not out["pass"] and out["reason"].startswith(f"{FORCING} reads nothing")

    def test_another_jobs_log_is_not_read(self, tmp_path):
        rule = _rules(tmp_path, {"2016": 2015})[0]
        _atm_log(tmp_path / "run", 111, f"{FORCING} 2016 2015 2016")
        ex = RG.expected_output(tmp_path / "run", 2016, [rule], job=222)
        assert ex[0]["pass"] is False
        assert ex[0]["reason"].startswith("cannot check: no atm.log.222.* in")
        ex = RG.expected_output(tmp_path / "run", 2016, [rule])
        assert ex[0]["reason"] == "cannot check: no job id: the year's DATM log cannot be found"

    def test_a_year_the_rule_does_not_name_says_so(self, tmp_path):
        rule = _rules(tmp_path, {"2016": 2015})[0]
        _atm_log(tmp_path / "run", 222, f"{FORCING} 2018 2018 2019")
        ex = RG.expected_output(tmp_path / "run", 2018, [rule], job=222)
        assert ex[0]["reason"] == ("cannot check: the rule names no recorded year for model "
                                   "year 2018")


# ── the year's output ────────────────────────────────────────────────────────
class TestTheYearsOutput:

    def test_a_run_that_read_the_recorded_year_passes_exactly(self, tmp_path):
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        for y in YEARS:
            out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, y)
            assert out["pass"] and out["max_abs_difference"] == 0.0, (y, out)
            assert out["reads"] == {2015: 2015, 2016: 2015, 2017: 2017}[y]
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert out["first_day_carried"] and out["per_field"]["TBOT"][
            "first_day_max_abs_difference"] == 0.0

    def test_a_run_that_read_its_own_year_fails(self, tmp_path):
        ref, run = _pair(tmp_path, {})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert not out["pass"]
        assert out["reason"].startswith("differs from the control's 2015 by up to 1 ")

    @pytest.mark.parametrize("field,by,day", [
        ("RAIN", 1e-6, 365 + 100), ("SNOW", 1e-6, 365 + 10),       # rain or snow alone
        ("TBOT", 0.01, 365 + 200),                                    # a small error
        ("TBOT", 0.25, 365 + 31)])                                    # 1 February, carried year
    def test_any_field_off_on_any_held_day_fails(self, tmp_path, field, by, day):
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        _shift(run, day, by, field=field)
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert not out["pass"] and out["per_field"][field]["max_abs_difference"] > 0

    def test_the_carried_first_day_is_reported_not_held(self, tmp_path):
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        for day in (365, 730):                          # 1 January 2016 and 2017
            _shift(run, day, 0.25)
        for y in (2016, 2017):                          # both carried
            out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, y)
            assert out["pass"], out
            assert out["per_field"]["TBOT"]["first_day_max_abs_difference"] == \
                pytest.approx(0.25, abs=1e-4)
        rule_none = dict(rule, carried_first_days=[])
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule_none, 2016)
        assert not out["pass"] and out["max_abs_difference"] == pytest.approx(0.25, abs=1e-4)

    def test_a_day_off_later_in_a_carried_year_still_fails(self, tmp_path):
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        _shift(run, 365 + 1, 0.25)                       # 2 January 2016
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert not out["pass"]

    def test_31_december_is_held_though_it_lives_in_the_next_years_file(self, tmp_path):
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        _shift(run, 729, 0.25)                           # 31 December 2016
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert not out["pass"]

    @pytest.mark.parametrize("where", ["run", "control"])
    def test_a_cell_day_reported_by_one_run_only_fails(self, tmp_path, where):
        ref_dir = tmp_path / "ctl_run"
        ref, run = _pair(tmp_path, {2016: 2015})
        rule = _rules(tmp_path, {"2016": 2015})[1]
        if where == "run":
            _shift(run, 365 + 40, np.nan, cells=[(1, 1)])
        else:
            _shift(ref_dir, 40, np.nan, cells=[(1, 1)])  # the control's 2015 day 41
            ref = G.open_stream(ref_dir, "h1")
        out = W.check_control_year(G.open_stream(run, "h1"), ref, rule, 2016)
        assert not out["pass"] and out["unpaired"] == 1
        assert out["reason"].startswith("1 cell-day(s) have a value in one run")

    def test_the_gate_rule_needs_the_controls_output(self, tmp_path):
        ref, run = _pair(tmp_path, {2016: 2015})
        rules = _rules(tmp_path, {"2016": 2015})
        ex = RG.expected_output(run, 2016, rules[1:], reference=ref)
        assert ex[0]["pass"] is True and ex[0]["reads"] == 2015
        ex = RG.expected_output(run, 2016, rules[1:])
        assert "--reference" in ex[0]["reason"]
        ex = RG.expected_output(run, 2016, rules[1:], reference_error="/x: no h1 history")
        assert ex[0]["reason"] == "cannot read the control's output: /x: no h1 history"
        ex = RG.expected_output(run, 2016, [dict(rules[1], stream="h2")], reference=ref)
        assert ex[0]["reason"] == ("the control's output was opened as stream h1, and the "
                                   "rule reads h2")
        named = G.Stream("reference", sorted((tmp_path / "ctl_run").glob("*.elm.h1.*.nc")))
        ex = RG.expected_output(run, 2016, rules[1:], reference=named)
        assert ex[0]["pass"] is True
        ex = RG.expected_output(run, 2019, rules[1:], reference=ref)
        assert ex[0]["reason"] == ("cannot check: the rule names no recorded year for model "
                                   "year 2019")


# ── the change ───────────────────────────────────────────────────────────────
class TestTheChange:

    def test_its_meaning_names_the_years(self):
        c = CH.resolve({"id": "weather_repeat_year",
                        "params": {"map": {"2017": 2015, "2016": 2015, "2019": 2018}}})
        assert c["params"]["map"] == {"2016": 2015, "2017": 2015, "2019": 2018}
        assert c["meaning"] == ("the difference from the control is the response to "
                                "2015's recorded weather in 2016 and 2017; 2018's recorded "
                                "weather in 2019, with every other input the control's")
        own = CH.resolve({"id": "weather_repeat_year",
                          "params": {"map": {"2016": 2015, "2017": 2017}}})
        assert own["meaning"].startswith("the difference from the control is the response "
                                         "to 2015's recorded weather in 2016, with")
        assert "its first model step has the record after the previous year's recorded " \
               "year" in c["not_meaning"]

    @pytest.mark.parametrize("m,match", [
        ({}, "not a map of model years"), ({"16": 2015}, "not a map of model years"),
        ({"2016": "2015"}, "not a map of model years"), ({"2016": True}, "not a map"),
        ({"2016": 2016}, "changes nothing")])
    def test_a_map_that_is_not_one_is_refused(self, m, match):
        with pytest.raises(CH.ChangeError, match=match):
            CH.resolve({"id": "weather_repeat_year", "params": {"map": m}})

    def test_what_the_list_publishes(self):
        d = CH.declare()
        c = next(c for c in d["changes"] if c["id"] == "weather_repeat_year")
        assert c["available"] and "{" not in c["meaning"] and "menu_text" not in c
        assert any("(h1) must hold TBOT, RAIN and SNOW" in x for x in c["limits"])
        assert any("weather at most once" in r for r in d["rules"])
        for x in d["changes"]:
            assert not any("something else" in lim for lim in x.get("limits") or []), x["id"]
        assert len(json.dumps(d, separators=(",", ":"))) < 12000   # the reception cut

    def test_build_run_gives_each_year_its_settings_and_checks(self, tmp_path):
        ctl = _repeat_control(tmp_path)
        recipe = DC.Recipe(case="ctl", case_root=str(tmp_path / "cases"), srcroot="/src/E3SM",
                           compset="C", xml=[("RUN_STARTDATE", "2015-01-01")],
                           user_nl={"elm": ["use_nofire = .false."]}, start="2015-01-01",
                           copied_from=str(ctl))
        run = {"name": "dry", "changes": [{"id": "weather_repeat_year",
                                           "params": {"map": {"2016": 2015}}}]}
        r, e = CH.build_run(recipe, run, tmp_path / "inputs", YEARS, control_case_dir=str(ctl))
        assert r.case == "ctl_dry" and r.user_nl == recipe.user_nl and r.check() == []
        assert any(n.startswith("each year's job sets, before it runs: 2015: "
                                "DATM_CLMNCEP_YR_START=2015,DATM_CLMNCEP_YR_END=2016,"
                                "DATM_CLMNCEP_YR_ALIGN=2015; 2016: ") for n in r.notes)
        assert e["year_settings"]["2016"] == [["DATM_CLMNCEP_YR_START", 2015],
                                              ["DATM_CLMNCEP_YR_END", 2016],
                                              ["DATM_CLMNCEP_YR_ALIGN", 2016]]
        assert [o["rule"] for o in e["output"]] == ["datm_stream_years", "equals_control_year"]
        assert all(o["change"] == "weather_repeat_year" for o in e["output"])
        assert e["settings"] == {}
        with pytest.raises(CH.ChangeError, match="sets each year's settings; build it"):
            CH.run_recipe(recipe, run)
        with pytest.raises(CH.ChangeError, match="sets each year's settings, and its checks"):
            CH.expectations(run, str(ctl))

    FACTS = {"start_year": 2015, "forcing_years": [2013, 2023], "output_years": [2015, 2021],
             "settings": {"DATM_CLMNCEP_YR_START": "2013", "DATM_CLMNCEP_YR_END": "2023",
                          "DATM_CLMNCEP_YR_ALIGN": "2013"},
             "streams": {"h1": {"fields": ["RAIN", "SNOW", "TBOT"],
                                "years": list(range(2015, 2022))}}}

    def _check(self, m, years=YEARS, reuse=True, facts=None, other=()):
        exp = {"control": {"name": "naches", "reuse": reuse}, "years": years,
               "runs": [{"name": "dry", "changes": [{"id": "weather_repeat_year",
                                                    "params": {"map": m}}] + list(other)}]}
        return CH.check(exp, {"name": "naches", "facts": facts or self.FACTS})

    def test_the_experiment_check(self):
        ok = self._check({"2016": 2015, "2017": 2015})
        assert ok["ok"], ok["wont_build"]
        assert any("2015's recorded weather in 2016 and 2017" in f["what"] for f in ok["facts"])
        for m, match in (({"2019": 2015}, "names model year(s) [2019]"),
                         ({"2016": 2014}, "not in the control's output (2015-2021)"),
                         ({"2016": 2022}, "not in the control's output")):
            out = self._check(m)
            assert any(match in w["what"] for w in out["wont_build"]), (m, out)
        for m in ({"2016": 2021}, {"2016": 2015}):        # the output's last and first years
            assert self._check(m)["ok"], m
        facts = dict(self.FACTS, output_years=[2015, 2023],
                     streams={"h1": {"fields": ["RAIN", "SNOW", "TBOT"],
                                     "years": list(range(2015, 2024))}})
        out = self._check({"2016": 2023}, facts=facts)
        assert any("needs the year after it there too: [2023]" in w["what"]
                   for w in out["wont_build"])
        out = self._check({"2016": 2015}, reuse=False)
        assert any("needs a reused control" in w["what"] for w in out["wont_build"])
        lack = dict(self.FACTS, streams={"h1": {"fields": ["RAIN"], "years": YEARS}})
        out = self._check({"2016": 2015}, facts=lack)
        assert any("(h1) lacks ['TBOT', 'SNOW']" in w["what"] for w in out["wont_build"])
        gap = dict(self.FACTS, streams={"h1": {"fields": ["RAIN", "SNOW", "TBOT"],
                                               "years": [2016, 2017]}})
        out = self._check({"2016": 2015}, facts=gap)
        assert any("has no complete year [2015]" in w["what"] for w in out["wont_build"])

    def test_a_control_whose_model_years_are_not_its_recorded_years_is_refused(self):
        moved = dict(self.FACTS, settings=dict(self.FACTS["settings"],
                                               DATM_CLMNCEP_YR_ALIGN="2015"))
        out = self._check({"2016": 2015}, facts=moved)
        assert any("reads recorded year 2013 + (Y - 2015), not Y" in w["what"]
                   for w in out["wont_build"])
        ref = dict(self.FACTS, settings=dict(self.FACTS["settings"],
                                             DATM_CLMNCEP_YR_ALIGN="$DATM_CLMNCEP_YR_START"))
        assert self._check({"2016": 2015}, facts=ref)["ok"]
        far = dict(self.FACTS, settings=dict(self.FACTS["settings"],
                                             DATM_CLMNCEP_YR_ALIGN="${OTHER_YEAR}",
                                             OTHER_YEAR="2015"))
        out = self._check({"2016": 2015}, facts=far)
        assert any("reads recorded year 2013 + (Y - 2015), not Y" in w["what"]
                   for w in out["wont_build"])

    def test_malformed_years_are_reported_once(self):
        out = self._check({"2016": 2015}, years="2015")
        assert any("years must be a non-empty list" in w["what"] for w in out["wont_build"])
        assert not any("names model year" in w["what"] for w in out["wont_build"])

    def test_two_weather_changes_in_one_run_are_refused_and_others_noted(self):
        both = self._check({"2016": 2015}, other=[{"id": "weather_scale",
                                                   "params": {"temperature_K": 1.0}}])
        assert any("changes the weather twice" in w["what"] for w in both["wont_build"])
        fire = self._check({"2016": 2015}, other=[{"id": "fire_off"}])
        assert fire["ok"] and any("changes the weather and fire_off" in u["what"]
                                  for u in fire["unusual"])
        twice = self._check({"2016": 2015}, other=[{"id": "weather_repeat_year",
                                                    "params": {"map": {"2017": 2015}}}])
        assert not twice["ok"]
        assert not any("changes the weather and :" in u["what"] for u in twice["unusual"])


class TestTheReports:

    def test_a_year_map_reads_as_words_in_the_numbers_report_and_the_analyzer(self):
        import agents.analysis.domain_report as DR
        import agents.analysis.step2_investigate as S2
        run = {"changes": [{"id": "weather_repeat_year",
                            "params": {"map": {"2016": 2015, "2017": 2015}}},
                           {"id": "weather_scale", "params": {"months": [6, 7, 8]}}]}
        assert DR._changes_text(run) == ("weather_repeat_year map 2016 to 2015, 2017 to "
                                         "2015, weather_scale months 6, 7, 8")
        assert S2._param_text("map", {"2016": 2015}) == "map 2016 to 2015"
        assert S2._param_text("ppmv", 400.0) == "ppmv 400.0"


# ── the chain's job script ───────────────────────────────────────────────────
class TestTheJobScript:

    def _expect(self, ctl):
        return {"control_case_dir": str(ctl), "settings": {}, "output": [],
                "year_settings": {"2015": [["DATM_CLMNCEP_YR_START", 2015],
                                           ["DATM_CLMNCEP_YR_END", 2016],
                                           ["DATM_CLMNCEP_YR_ALIGN", 2015]],
                                  "2016": [["DATM_CLMNCEP_YR_START", 2015],
                                           ["DATM_CLMNCEP_YR_END", 2016],
                                           ["DATM_CLMNCEP_YR_ALIGN", 2016]]}}

    def test_each_year_and_the_smoke_run_set_their_own_years(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        ctl = tmp_path / "ctl"
        (ctl / "CaseDocs").mkdir(parents=True)
        script = DC.write_stage_script(r, rec, expect=self._expect(ctl))
        text = script.read_text()
        assert "2016) YS=,DATM_CLMNCEP_YR_START=2015,DATM_CLMNCEP_YR_END=2016," \
               "DATM_CLMNCEP_YR_ALIGN=2016 ;;" in text
        _run_stage(script, "y2016")
        assert (c["case_dir"] / "xmlchange.log").read_text().strip().splitlines()[-1] == (
            "CONTINUE_RUN=TRUE,STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1,"
            "DATM_CLMNCEP_YR_START=2015,DATM_CLMNCEP_YR_END=2016,DATM_CLMNCEP_YR_ALIGN=2016")
        _run_stage(script, "y2015")
        assert (c["case_dir"] / "xmlchange.log").read_text().strip().splitlines()[-1] == (
            "CONTINUE_RUN=FALSE,STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1,"
            "RUN_STARTDATE=2015-01-01,DATM_CLMNCEP_YR_START=2015,DATM_CLMNCEP_YR_END=2016,"
            "DATM_CLMNCEP_YR_ALIGN=2015")
        _run_stage(script, "smoke")
        assert (c["case_dir"] / "xmlchange.log").read_text().strip().splitlines()[-1] \
            .endswith(",DATM_CLMNCEP_YR_START=2015,DATM_CLMNCEP_YR_END=2016,"
                      "DATM_CLMNCEP_YR_ALIGN=2015")

    def test_a_year_without_settings_stops_before_its_model_runs(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        ctl = tmp_path / "ctl"
        (ctl / "CaseDocs").mkdir(parents=True)
        script = DC.write_stage_script(r, rec, expect=self._expect(ctl))
        p = _run_stage(script, "y2017")
        assert p.returncode == 2
        assert not (c["case_dir"] / "xmlchange.log").exists()
        assert "FAILED no year settings for 2017 in the expectations" in \
            (rec / DC.STATUS_FILE).read_text()

    def test_a_run_without_year_settings_is_unchanged(self, tmp_path):
        c, r = _stage_case(tmp_path)
        script = DC.write_stage_script(r, tmp_path / "rec")
        _run_stage(script, "y2016")
        assert (c["case_dir"] / "xmlchange.log").read_text().strip().splitlines()[-1] == (
            "CONTINUE_RUN=TRUE,STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1")

    @pytest.mark.parametrize("bad,match", [
        ([["STOP_N", 3]], "set run control"), ([["DATM_CLMNCEP_YR_START", "20 15"]],
                                               "not NAME=value pairs")])
    def test_settings_that_are_not_plain_pairs_are_refused(self, tmp_path, bad, match):
        c, r = _stage_case(tmp_path)
        ctl = tmp_path / "ctl"
        (ctl / "CaseDocs").mkdir(parents=True)
        e = dict(self._expect(ctl), year_settings={"2016": bad})
        with pytest.raises(DC.DomainCaseError, match=match):
            DC.write_stage_script(r, tmp_path / "rec", expect=e)


# ── through the server's tools, and the gate's year mode ─────────────────────
class TestThroughTheServer:

    def _built(self, lab, tmp_path):
        cd = lab["control_dir"]
        (cd / "CaseDocs" / "datm_in").write_text(_datm_in(_domain(tmp_path / "d.nc")))
        (cd / "env_run.xml").write_text(ENV_RUN.format(align=2013, start=2013, end=2023))
        path = _forcing(tmp_path / "forcing")
        (cd / "CaseDocs" / FORCING).write_text(STREAM.format(path=path, offset=0))
        _experiment(lab, [{"name": "dry", "changes": [{"id": "weather_repeat_year",
                                                       "params": {"map": {"2016": 2015}}}]}])
        out = json.loads(lab["srv"].build_domain_runs(str(lab["run_dir"])))
        assert out["ok"], out
        return lab["run_dir"] / "01_inputs" / "runs" / "dry"

    def test_a_repeat_run_is_built_with_its_year_settings(self, lab, tmp_path):
        inp = self._built(lab, tmp_path)
        e = json.loads((inp / "domain_expectations.json").read_text())
        assert sorted(e["year_settings"]) == ["2015", "2016"]
        assert [o["rule"] for o in e["output"]] == ["datm_stream_years", "equals_control_year"]
        spec = json.loads((inp / "domain_chain.json").read_text())
        assert spec["reference"]["run_dir"] == str(lab["control_run"])
        w = json.loads(lab["srv"].write_domain_case(str(lab["run_dir"]), run="dry"))
        assert w["ok"], w
        stage = (lab["run_dir"] / "domain_case" / "dry" / DC.STAGE_SCRIPT).read_text()
        assert "2016) YS=,DATM_CLMNCEP_YR_START=2015" in stage
        assert f"--reference {lab['control_run']}" in stage

    def test_a_chain_past_the_built_years_is_refused(self, lab, tmp_path):
        inp = self._built(lab, tmp_path)
        spec = json.loads((inp / "domain_chain.json").read_text())
        (inp / "domain_chain.json").write_text(json.dumps(dict(spec, years=[2015, 2016,
                                                                            2017])))
        w = json.loads(lab["srv"].write_domain_case(str(lab["run_dir"]), run="dry"))
        assert w["ok"] is False and "which the run's year settings do not cover" in w["error"]

    def test_the_gate_applies_both_rules_from_the_servers_expectations(self, lab, tmp_path):
        inp = self._built(lab, tmp_path)
        c = cf.write_case(tmp_path / "x")
        ctl = cf.write_case(tmp_path / "c", case="ctl")
        for d_ in (c["run_dir"], ctl["run_dir"]):
            for p in sorted(d_.glob("*.elm.h1.*.nc")):
                with netCDF4.Dataset(p, "a") as d:
                    v = d.createVariable("TBOT", "f4", ("time", "lat", "lon"))
                    v.units = "K"
                    v[:] = np.full(d["RAIN"].shape, 280.0, np.float32)
        _distinct(ctl["run_dir"])
        _distinct(c["run_dir"], {2016: 2015})
        _atm_log(c["run_dir"], 111, f"{FORCING} 2016 2015 2016")
        argv = ["year", "--case-dir", str(c["case_dir"]), "--run-dir", str(c["run_dir"]),
                "--year", "2016", "--job", "111", "--out", str(tmp_path / "chk"),
                "--expect", str(inp / "domain_expectations.json"),
                "--reference", str(ctl["run_dir"]), "--reference-report-only"]

        def year():
            RG.main(argv)
            return json.loads((tmp_path / "chk" / "year_111.json").read_text())["years"][
                "2016"]
        y = year()
        assert [(e["rule"], e["pass"]) for e in y["expected"]] == [
            ("datm_stream_years", True), ("equals_control_year", True)], y["expected"]
        assert y["expected"][1]["reads"] == 2015 and y["expected"][1]["first_day_carried"]
        _shift(c["run_dir"], 729, 0.25)                 # 31 December 2016, in the 2017 file
        y = year()
        assert [e["pass"] for e in y["expected"]] == [True, False]
        assert any(r.startswith("expected weather_repeat_year: TBOT+RAIN+SNOW "
                                "equals_control_year") for r in y["reasons"])
