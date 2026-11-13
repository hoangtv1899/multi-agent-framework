"""How to make the new runs without the framework, RUN_BY_HAND.md
(agents/run_by_hand, 2026-11-13).

    pytest tests/test_run_by_hand.py -v

Every name, path and number here is made up: a valley's control run, a run
that cuts summer rain (dry_summer) and one that warms the air (warm_air), as
the model server would have built them before the design page. The files the
preview names are written into a temporary study folder.
"""
import copy
import json
import re

import pytest

import run_gate
from agents import run_by_hand as RB

CODE_FP = "c0de" * 16
MOD_SHA = "1" * 64
USER_SHA = "2" * 64
WEATHER_SHA = "3" * 64
SCRIPT_MARK = "MADE_UP_SCRIPT_BODY_LINE"
STREAM_MARK = "MADE_UP_STREAM_TEXT"
START = "/scratch/valley_ctl/run/valley_ctl.elm.r.2033-01-01-00000.nc"

CTL_RECIPE = {
    "case": "valley_ctl", "case_root": "/cases", "srcroot": "/code/e3sm",
    "compset": "I2000VALLEY", "res": "USRDAT", "mach": "box", "compiler": "gnu",
    "project": "acct1",
    "xml": [["LND_DOMAIN_PATH", "/inputs/valley"], ["LND_DOMAIN_FILE", "domain.valley.nc"],
            ["NTASKS", "40"], ["RUN_STARTDATE", "2031-01-01"]],
    "user_nl": {"elm": ["finidat = '/inputs/valley/start_2031.nc'",
                        "fsurdat = '/inputs/valley/surf.nc'",
                        "paramfile = '/inputs/params.nc'",
                        "stream_fldfilename_ndep = '/inputs/ndep.nc'",
                        "! a note kept in the case",
                        "hist_fincl1 = 'GPP','NPP',",
                        "              'TLAI'"]},
    "sourcemods": {"src.elm/LeafMod.F90": {
        "from": "/cases/valley_ctl/SourceMods/src.elm/LeafMod.F90", "sha256": MOD_SHA}},
    "user_files": {}, "create_options": {}, "start": "2031-01-01"}

STREAM_TEXT = f"""<?xml version="1.0"?>
<file id="stream" version="1.0">
<!-- {STREAM_MARK} -->
<fieldInfo>
   <variableNames>
     pr   prec_af
   </variableNames>
</fieldInfo>
</file>
"""

CASE_SCRIPT = f"""#!/bin/bash
# {SCRIPT_MARK}
CASEDIR=/cases/valley_ctl_dry_summer
[ -e "$CASEDIR" ] && {{ echo "case exists: $CASEDIR" >&2; exit 1; }}
./case.setup
"""

STAGE_SCRIPT = f"""#!/bin/bash
#SBATCH --account=acct1
#SBATCH --nodes=1
#SBATCH --exclusive
# {SCRIPT_MARK}
  smoke)  into_case
          gate smoke --days 3 ;;
"""

FACTOR_RULE = {"stream": "h1", "fields": ["RAIN", "SNOW"], "rule": "matches_anomaly",
               "how": "factor", "anomaly_file": "WEATHER", "anomaly_var": "pr",
               "tolerance": 1e-05, "floor": 1e-09, "change": "weather_scale",
               "why": "daily rain and daily snow are each the control's times the factor in "
                      "the chosen months and cells, and the control's elsewhere"}
OFFSET_RULE = {"stream": "h1", "fields": ["TBOT"], "rule": "matches_anomaly", "how": "offset",
               "anomaly_file": "WEATHER", "anomaly_var": "tas", "tolerance": 0.001,
               "floor": 1e-09, "change": "weather_scale",
               "why": "the daily air temperature the land model receives is the control's"}


def _weather(rd, name):
    return {"file": "weather_anomaly_abc.nc", "sha256": WEATHER_SHA,
            "path": str(rd / f"01_inputs/runs/{name}/weather/weather_anomaly_abc.nc"),
            "fields": {
                "tas": {"how": "offset", "units": "K", "identity": 0.0, "acts_on": "tbot_af",
                        "months": [], "cells": 0, "values": []},
                "pr": {"how": "factor", "units": "1", "identity": 1.0, "acts_on": "prec_af",
                       "months": ["2033-06", "2033-07", "2034-06", "2034-07"], "cells": 120,
                       "values": [0.6]}},
            "records": {"first": "2032-01", "last": "2035-12", "n": 48,
                        "each": "the first instant of each month"},
            "grid": {"shape": [12, 10], "active_cells": 120}}


def _jobs(tag, years, smoke=True):
    jobs = [{"stage": "case", "name": f"{tag}_case", "partition": "small",
             "walltime": "01:00:00", "after": None}]
    if smoke:
        jobs.append({"stage": "smoke", "name": f"{tag}_smoke", "partition": "small",
                     "walltime": "00:45:00", "after": "case"})
    for y in years:
        jobs.append({"stage": f"y{y}", "name": f"{tag}_y{y}", "partition": "small",
                     "walltime": "01:55:00", "after": jobs[-1]["stage"]})
    jobs.append({"stage": "report", "name": f"{tag}_report", "partition": "small",
                 "walltime": "00:10:00", "after": None})
    return jobs


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def _run(rd, name="dry_summer", weather=True, after=None):
    """One run as the server builds it: its recipe, chain spec, expectations,
    scripts and the files written for it; the preview's entry for it."""
    inp, case = rd / f"01_inputs/runs/{name}", rd / f"domain_case/{name}"
    recipe = copy.deepcopy(CTL_RECIPE)
    recipe["case"] = f"valley_ctl_{name}"
    recipe["xml"][-1] = ["RUN_STARTDATE", "2033-01-01"]
    recipe["start"] = "2033-01-01"
    recipe["user_nl"]["elm"][0] = f"finidat = '{START}'"
    rules = []
    if weather:
        recipe["user_nl"]["datm"] = ["anomaly_forcing = 'Anomaly.Forcing.Precip'",
                                     'tintalgo = "linear", "lower"']
        stream = inp / "weather/user_datm.streams.txt.Anomaly.Forcing.Precip"
        _write(stream, STREAM_TEXT)
        recipe["user_files"] = {stream.name: {"from": str(stream), "sha256": USER_SHA}}
        w = str(inp / "weather/weather_anomaly_abc.nc")
        _write(inp / "weather/weather_anomaly_abc.nc", "not read by the page")
        rules = [dict(OFFSET_RULE, anomaly_file=w), dict(FACTOR_RULE, anomaly_file=w)]
    expect = {"run": name, "changes": [],
              "settings": {"datm_in": {"anomaly_forcing": ["Anomaly.Forcing.Precip"],
                                       "streams": {"appends": 1},
                                       "dtlimit": {"appends": 1}, "fillalgo": {"appends": 1}},
                           "lnd_in": {"finidat": START}},
              "output": rules,
              "case_files": ({"CaseDocs/datm.streams.txt.Anomaly.Forcing.Precip": USER_SHA}
                             if weather else {})}
    spec = {"years": [2033, 2034], "tag": f"valley_ctl_{name}",
            "reference": {"run_dir": "/scratch/valley_ctl/run", "stream": "h1",
                          "gate_years": []}}
    _write(inp / "domain_recipe.json", recipe)
    _write(inp / "domain_chain.json", spec)
    _write(inp / "domain_expectations.json", expect)
    _write(case / "expectations.json", expect)
    _write(case / "create_case.sh", CASE_SCRIPT.replace("dry_summer", name))
    _write(case / "run_stage.sh", STAGE_SCRIPT)
    return {"name": name, "case": f"valley_ctl_{name}",
            "case_dir": f"/cases/valley_ctl_{name}",
            "changes": ["weather_scale"] if weather else [], "after": after,
            "recipe": f"01_inputs/runs/{name}/domain_recipe.json",
            "chain_spec": f"01_inputs/runs/{name}/domain_chain.json",
            "expectations": f"01_inputs/runs/{name}/domain_expectations.json",
            "case_script": f"domain_case/{name}/create_case.sh",
            "stage_script": f"domain_case/{name}/run_stage.sh",
            "jobs": _jobs(f"valley_ctl_{name}", [2033, 2034]),
            "notes": ["the reference gates no year: every year is compared and reported only"],
            "weather_files": [_weather(rd, name)] if weather else []}


def _preview(rd, runs=("dry_summer",)):
    _write(rd / "01_inputs/control_recipe.json", CTL_RECIPE)
    _write(rd / "domain_case/inputs_check.json", {"pass": True})
    return {
        "made": "2033-01-01T00:00:00", "refused": [], "error": None, "problems": [],
        "runs": [_run(rd, n) for n in runs],
        "check": {"unusual": [], "facts": [], "cost": {"node_hours": 3.2}},
        "control": {"name": "valley", "reuse": True, "case_dir": "/cases/valley_ctl",
                    "run_dir": "/scratch/valley_ctl/run",
                    "reproduced": {"fingerprint": CODE_FP, "years": [2031, 2032],
                                   "how": "made again from its own records"}},
        "inputs_check": {"path": "domain_case/inputs_check.json", "pass": True,
                         "reasons": [], "surface": "/inputs/valley/surf.nc",
                         "domain": "/inputs/valley/domain.valley.nc", "restart": START,
                         "checks": {"active_cells": 120, "layout_matches": True}},
        "control_recipe": "01_inputs/control_recipe.json",
        "forcing": [
            {"stream": "datm.streams.txt.VALLEYWX", "years": [2031, 2031, 2035],
             "domain": "/inputs/weather/grid.nc", "data_dir": "/inputs/weather", "n_files": 60,
             "first": "wx.2031-01.nc", "last": "wx.2035-12.nc",
             "fields": ["TBOT", "WIND", "PRECTmms"]},
            {"stream": "datm.streams.txt.presaero.clim_2000", "years": [1, 1, 1],
             "domain": "/inputs/aero.nc", "data_dir": "/inputs", "n_files": 1,
             "first": "aero.nc", "last": "aero.nc", "fields": ["BCDEPWET"]}],
        "checks_in_words": dict(run_gate.CHECKS_IN_WORDS),
        "source": {"srcroot": "/code/e3sm", "commit": "a1" * 20,
                   "changed": ["M cime", "M components/elm/src/x.F90"],
                   "untracked": ["components/elm/y.bak"], "diff_sha256": "4" * 64,
                   "fingerprint": CODE_FP, "patch_left_out": [],
                   "patch": "01_inputs/source_changes.patch"},
        "analysis": {"regions": [{"name": "valley floor", "cells": 40, "area_km2": 30.0,
                                  "rule": "below 900 m", "sources": ["surf.nc"]}],
                     "fields": [{"name": "GPP", "stream": "h0", "how": "total over the year",
                                 "units": "gC/m^2/yr"}]}}


def _page(rd, preview=None, **kw):
    return RB.render(preview if preview is not None else _preview(rd, **kw), rd, {}, {})


def _flat(text):
    return " ".join(text.split())


def _section(text, head):
    """The lines of one section, from its heading to the next of its level."""
    level = head.split(" ", 1)[0]
    out, on = [], False
    for ln in text.splitlines():
        if ln.startswith(head):
            on = True
            continue
        if on and re.match(rf"^#{{1,{len(level)}}} ", ln):
            break
        if on:
            out.append(ln)
    return "\n".join(out)


# ── nothing to say ───────────────────────────────────────────────────────────
class TestNothingBuilt:
    @pytest.mark.parametrize("preview", [None, {}, {"refused": [{"where": "x", "what": "y"}]},
                                         {"error": "it would not build", "runs": []}])
    def test_no_page_without_a_built_design(self, tmp_path, preview):
        assert RB.render(preview, tmp_path, {}, {}) == ""

    def test_a_record_with_nothing_in_it_gives_no_page(self, tmp_path):
        assert RB.render({"runs": [], "made": "x"}, tmp_path, {}, {}) == ""

    @pytest.mark.parametrize("runs", [["dry_summer"], [None, 3], "dry_summer",
                                      {"name": "dry_summer"}])
    def test_runs_listed_but_none_a_record_gives_no_page(self, tmp_path, runs):
        """The study makes runs, so a page that says it makes none would be
        untrue; with no record of them there is nothing true to say."""
        pv = _preview(tmp_path)
        assert "This study makes no new run" in _page(tmp_path, dict(pv, runs=[]))
        assert RB.render(dict(pv, runs=runs), tmp_path, {}, {}) == ""

    @pytest.mark.parametrize("over", [{"refused": [{"where": "runs", "what": "too long"}]},
                                      {"error": "run dry_summer: its scripts could not be "
                                                "written"}])
    def test_a_whole_record_that_was_refused_or_failed_gives_no_page(self, tmp_path, over):
        pv = _preview(tmp_path)
        assert _page(tmp_path, pv)                      # it would give a page as it is
        assert RB.render(dict(pv, **over), tmp_path, {}, {}) == ""


# ── one run ──────────────────────────────────────────────────────────────────
class TestOneRun:
    def test_the_title_names_the_run_and_the_opening_says_what_the_page_is(self, tmp_path):
        page = _page(tmp_path)
        assert page.startswith("# How to make this run by hand: dry_summer\n")
        assert "This page is for a modeller who wants to make the new run without the " \
               "framework." in page
        assert "Nothing has been submitted." in page
        assert "A path that does not start with a slash is inside this study's folder." in page

    def test_the_steps_come_in_order(self, tmp_path):
        heads = [ln for ln in _page(tmp_path).splitlines() if ln.startswith("## ")]
        assert heads == ["## Step 1. Get the code", "## Step 2. Start from the control run's case",
                         "## Step 3. Make the changes", "## Step 4. Run the jobs",
                         "## Step 5. Check each job", "## Step 6. The input files"]

    def test_step_1_the_code(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 1.")
        assert "| E3SM checkout | `/code/e3sm` |" in s
        assert f"| Commit | `{'a1' * 20}` |" in s
        assert ("| Local changes | `01_inputs/source_changes.patch`, which changes two paths and "
                "adds one new file. Apply it with `git apply` at the checkout root, after the "
                "commit and its submodules are checked out. |") in s
        assert ("| Source file copied into the case | `LeafMod.F90`, from "
                "`/cases/valley_ctl/SourceMods/src.elm/` |") in s
        assert f"| Fingerprint of the code | `{CODE_FP}` |" in s
        assert "Each case job records the fingerprint of the code it built" in s
        assert "made again" not in s                 # said for the control run alone

    def test_step_2_the_control_runs_case(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 2.")
        assert "| Case | `/cases/valley_ctl` |" in s
        assert "| Output | `/scratch/valley_ctl/run` |" in s
        assert "| Recipe, as read from the case | `01_inputs/control_recipe.json` |" in s
        assert "The new run is this recipe with the changes of step 3. Nothing is run for the " \
               "control run." in s

    def test_step_3_a_row_per_difference_with_both_values(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 3.")
        assert "The new case is `/cases/valley_ctl_dry_summer`. It differs from the control " \
               "run's case in three ways." in s
        assert "| Change | The control run | The new run |" in s
        assert "| Start date, `RUN_STARTDATE` | 2031-01-01 | 2033-01-01 |" in s
        assert (f"| Starting state, `finidat` in `user_nl_elm` | the state the control run "
                f"started from in 2031 | `{START}` |") in s
        assert ("| Weather, `user_nl_datm` | nothing set | `anomaly_forcing = "
                "'Anomaly.Forcing.Precip'` and `tintalgo = \"linear\", \"lower\"`, read from one "
                "more stream file |") in s

    def test_step_3_the_files_written_and_what_each_holds(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 3.")
        assert "Two files carry the change of weather. Both are in " \
               "`01_inputs/runs/dry_summer/weather/`." in s
        assert ("| `weather_anomaly_abc.nc` | A factor on precipitation for each month from "
                "January 2032 to December 2035. The factor is 0.6 in 120 cells in June and July "
                "of 2033 and 2034, and 1 everywhere else. The amount added to the air "
                "temperature is 0 everywhere. |") in s
        assert ("| `user_datm.streams.txt.Anomaly.Forcing.Precip` | The stream file that tells "
                "the data atmosphere to read the factor. |") in s

    def test_step_3_names_the_case_script(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 3.")
        assert "The script `domain_case/dry_summer/create_case.sh` creates the case with these " \
               "changes. It refuses a case that already exists." in s

    def test_step_4_the_stage_script_and_a_row_per_stage(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 4.")
        assert ("The script `domain_case/dry_summer/run_stage.sh` runs one stage as its own job, "
                "on one node of the queue `small` under the account `acct1`. Submit it once for "
                "each stage, in this order.") in s
        assert "| Stage | What it does | Wall time | Waits for |" in s
        assert "| case | creates and builds the case | 1 h | nothing |" in s
        assert "| smoke | runs 3 days from the starting state | 45 min | case |" in s
        assert "| y2033 | runs 2033 | 1 h 55 min | smoke |" in s
        assert "| y2034 | runs 2034 | 1 h 55 min | y2033 |" in s
        assert "| report | writes each job's state and time | 10 min | every job above |" in s
        assert "Each job starts only when the one before it ended well, and a failure cancels " \
               "the jobs after it. The report job runs whatever happens." in s
        assert "valley_ctl_dry_summer_case" not in s          # the job names are left out

    def test_step_5_a_row_per_check_with_its_tolerance(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 5.")
        assert "`run_stage.sh` makes these checks itself. The values that each check must meet " \
               "are in `domain_case/dry_summer/expectations.json`." in s
        assert ("| case | Before the model is built, the run's settings differ from the control "
                "run's only in the settings its changes alter, with the values they set. The "
                "stream file is in the case, unchanged. |") in s
        assert ("| smoke | The job's own logs end cleanly. The first day's means match the saved "
                "state the run starts from, over the vegetated cells. |") in s
        assert "| each year | The job's own logs show the model dates it ran and a clean end." in s
        assert ("| each year, the change | The daily air temperature the land model receives is "
                "the control run's, within 0.001 K. Daily rain and daily snow are each the "
                "control run's times the factor in the chosen months and cells, and the control "
                "run's elsewhere, within a relative difference of 0.00001. |") in s
        assert ("| each year, against the control run | Each field's bias and daily correlation "
                "against the control run's same year are reported. They are not held to a "
                "limit, because the run is meant to differ. |") in s

    def test_step_6_the_input_files(self, tmp_path):
        s = _section(_page(tmp_path), "## Step 6.")
        assert "| Surface data | `/inputs/valley/surf.nc` |" in s
        assert "| Grid file | `/inputs/valley/domain.valley.nc` |" in s
        assert "| Starting state | the saved state of step 3 |" in s
        assert ("| Weather | the 60 files `wx.2031-01.nc` to `wx.2035-12.nc` in `/inputs/weather` "
                "|") in s
        assert ("The model server checked the first three as one set of 120 cells, and the set "
                "passed. Its record is `domain_case/inputs_check.json`.") in s
        assert ("The other files that the model reads are those of the control run's case. They "
                "include the parameter file, the nitrogen deposition file and the aerosol "
                "deposition file.") in s

    def test_the_page_ends_with_the_input_files(self, tmp_path):
        page = _page(tmp_path)
        assert page.endswith("deposition file.\n")


# ── what is left out, and the words ──────────────────────────────────────────
class TestWhatIsLeftOut:
    def test_no_script_is_printed(self, tmp_path):
        page = _page(tmp_path)
        assert SCRIPT_MARK not in page
        assert "```" not in page and "#!/bin/bash" not in page and "#SBATCH" not in page
        assert "create_newcase" not in page and "./xmlchange" not in page

    def test_no_file_hash_but_the_codes(self, tmp_path):
        page = _page(tmp_path)
        assert CODE_FP in page
        for h in (MOD_SHA, USER_SHA, WEATHER_SHA, "4" * 64):
            assert h not in page
        assert "sha256" not in page

    def test_no_text_of_a_written_file(self, tmp_path):
        page = _page(tmp_path)
        assert STREAM_MARK not in page and "<fieldInfo>" not in page

    def test_no_list_of_settings_that_grow_by_one(self, tmp_path):
        page = _page(tmp_path)
        assert "grows by" not in page and "dtlimit" not in page and "fillalgo" not in page

    def test_no_closing_section_on_the_numbers_of_the_runs(self, tmp_path):
        page = _page(tmp_path)
        for gone in ("tabulated", "question type", "valley floor", "years reported",
                     "a year's value", "| region |"):
            assert gone not in page

    def test_the_words_follow_the_house_rules(self, tmp_path):
        for page in (_page(tmp_path / "a"), _page(tmp_path / "b",
                                                  runs=("dry_summer", "warm_air"))):
            assert "\u2014" not in page and "\u2013" not in page
            assert "prepared run" not in page
            assert not re.search(r"\bthe control\b(?! run)", page)
            for word in ("Analyzer", "analysis plan", "preview", "coordinator", "rulebook",
                         "planning step"):
                assert word not in page

    def test_the_page_is_short(self, tmp_path):
        assert len(_page(tmp_path).split()) < 900


# ── several runs ─────────────────────────────────────────────────────────────
class TestSeveralRuns:
    def test_steps_1_2_and_6_once_and_3_to_5_for_each_run(self, tmp_path):
        page = _page(tmp_path, runs=("dry_summer", "warm_air"))
        assert page.startswith("# How to make these runs by hand\n")
        assert "make the new runs `dry_summer` and `warm_air` without the framework" in page
        assert "checked and built the runs for the design page" in page
        heads = [ln for ln in page.splitlines() if ln.startswith("#")]
        assert heads == [
            "# How to make these runs by hand",
            "## Step 1. Get the code", "## Step 2. Start from the control run's case",
            "## Run `dry_summer`", "### Step 3. Make the changes", "### Step 4. Run the jobs",
            "### Step 5. Check each job",
            "## Run `warm_air`", "### Step 3. Make the changes", "### Step 4. Run the jobs",
            "### Step 5. Check each job",
            "## Step 6. The input files"]
        assert "Each new run is this recipe with its own changes of step 3." in page

    @pytest.mark.parametrize("name", [None, "", "  "])
    def test_a_run_with_no_name_is_left_unnamed(self, tmp_path, name):
        pv = _preview(tmp_path)
        pv["runs"][0]["name"] = name
        page = _page(tmp_path, pv)
        assert page.startswith("# How to make this run by hand\n")
        pv = _preview(tmp_path, runs=("dry_summer", "warm_air"))
        pv["runs"][1]["name"] = name
        page = _page(tmp_path, pv)
        heads = [ln for ln in page.splitlines() if ln.startswith("## Run")]
        assert heads == ["## Run `dry_summer`", "## Run"]
        assert "make the new runs without the framework" in page
        assert "None" not in page and "``" not in page

    def test_each_run_has_its_own_case_and_scripts(self, tmp_path):
        page = _page(tmp_path, runs=("dry_summer", "warm_air"))
        warm = _section(page, "## Run `warm_air`")
        assert "The new case is `/cases/valley_ctl_warm_air`." in warm
        assert "`domain_case/warm_air/create_case.sh`" in warm
        assert "`domain_case/warm_air/run_stage.sh`" in warm
        assert "dry_summer" not in warm

    def test_a_run_that_waits_for_the_control_runs_case_job(self, tmp_path):
        pv = _preview(tmp_path, runs=("dry_summer",))
        pv["runs"].insert(0, dict(_run(tmp_path, "control", weather=False),
                                  case_dir="/cases/valley_ctl_control"))
        pv["runs"][1]["after"] = "control"
        pv["control"] = {"name": None, "reuse": False, "case_dir": "/cases/valley_ctl_control",
                         "run_dir": None}
        page = _page(tmp_path, pv)
        assert "The control run is made too, as the run `control` below. Each new run is this " \
               "recipe with its own changes." in page
        assert "The first job also waits for the case job of the run `control`." in page
        assert "Nothing is run for the control run" not in page


# ── the control run alone ────────────────────────────────────────────────────
class TestTheControlRunAlone:
    def _alone(self, rd, **over):
        pv = _preview(rd, runs=())
        pv.pop("inputs_check")
        pv.update(over)
        return pv

    def test_nothing_is_run_and_the_page_says_so(self, tmp_path):
        page = _page(tmp_path, self._alone(tmp_path))
        assert page.startswith("# How to make the control run by hand\n")
        assert "This study makes no new run. It reads the control run's output as it stands, " \
               "so nothing is run and nothing has been submitted." in page
        heads = [ln for ln in page.splitlines() if ln.startswith("## ")]
        assert heads == ["## Step 1. Get the code", "## Step 2. The control run's case",
                         "## Step 3. The input files"]
        assert "Run the jobs" not in page and "Each case job" not in page

    def test_the_recipe_is_described_not_printed(self, tmp_path):
        s = _section(_page(tmp_path, self._alone(tmp_path)), "## Step 2.")
        assert "The recipe holds what the case was created with." in s
        assert "| Compset | `I2000VALLEY` |" in s and "| Start date | 2031-01-01 |" in s
        assert "| Machine | `box` |" in s and "| Project | `acct1` |" in s
        assert "It also holds the four settings made with `xmlchange` after that, the lines of " \
               "`user_nl_elm` and the source file of step 1." in s
        assert "./xmlchange" not in s and "```" not in s

    def test_the_code_it_was_made_again_from(self, tmp_path):
        s = _section(_page(tmp_path, self._alone(tmp_path)), "## Step 1.")
        assert "The control run was made again from this code for 2031 and 2032, and it came " \
               "out the same bit for bit." in s
        pv = self._alone(tmp_path)
        pv["control"]["reproduced"]["fingerprint"] = "e" * 64
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert f"from code with the fingerprint `{'e' * 64}`, and it came out the same bit for " \
               f"bit. That is not the code of this checkout now." in s

    def test_its_input_files_come_from_its_recipe(self, tmp_path):
        s = _section(_page(tmp_path, self._alone(tmp_path)), "## Step 3.")
        assert "| Surface data | `/inputs/valley/surf.nc` |" in s
        assert "| Grid file | `/inputs/valley/domain.valley.nc` |" in s
        assert "| Starting state | `/inputs/valley/start_2031.nc` |" in s
        assert "checked" not in s
        assert "The other files that the model reads are named in the control run's recipe." in s


# ── each part with its record and without it ─────────────────────────────────
class TestAPartWithoutItsRecord:
    def test_no_source_leaves_step_1_to_the_source_file_alone(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("source")
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert "Source file copied into the case" in s and "Commit" not in s
        assert "fingerprint" not in s

    def test_no_code_at_all_renumbers_the_steps(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("source")
        (tmp_path / "01_inputs/control_recipe.json").unlink()
        for r in pv["runs"]:
            (tmp_path / r["recipe"]).unlink()
        heads = [ln for ln in _page(tmp_path, pv).splitlines() if ln.startswith("## ")]
        assert heads[0] == "## Step 1. Start from the control run's case"
        assert heads[1] == "## Step 2. Make the changes"

    def test_a_checkout_that_could_not_be_read(self, tmp_path):
        pv = _preview(tmp_path)
        pv["source"] = {"error": "not a git checkout"}
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert "The model server could not read the E3SM checkout. Not a git checkout." in s

    def test_local_changes_left_out_of_the_patch_or_not_saved(self, tmp_path):
        pv = _preview(tmp_path)
        pv["source"]["patch_left_out"] = ["big/data.bin"]
        pv["source"]["untracked"] = ["big/data.bin"]
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert "which changes two paths. Apply it" in s
        assert "| Left out of the patch | `big/data.bin`, too large to carry. Copy it by hand. |" \
               in s
        pv["source"].pop("patch")
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert "| Local changes | two paths differ from the commit, and no patch of them was " \
               "saved. |" in s
        pv["source"]["changed"] = []
        s = _section(_page(tmp_path, pv), "## Step 1.")
        assert "| Local changes | none |" in s

    def test_no_control_recipe_points_to_the_runs_own(self, tmp_path):
        pv = _preview(tmp_path)
        (tmp_path / "01_inputs/control_recipe.json").unlink()
        pv.pop("control_recipe")
        page = _page(tmp_path, pv)
        s = _section(page, "## Step 3.")
        assert "The new case is `/cases/valley_ctl_dry_summer`. Its recipe is " \
               "`01_inputs/runs/dry_summer/domain_recipe.json`." in s
        assert "| Change |" not in s
        assert "Recipe" not in _section(page, "## Step 2.")
        assert "## Step 6." in page

    def test_a_broken_recipe_is_a_missing_one(self, tmp_path):
        pv = _preview(tmp_path)
        (tmp_path / pv["runs"][0]["recipe"]).write_text("{not json")
        s = _section(_page(tmp_path, pv), "## Step 3.")
        assert "| Change |" not in s and "The new case is" in s

    def test_no_jobs_leaves_step_4_out_and_keeps_step_5(self, tmp_path):
        pv = _preview(tmp_path)
        pv["runs"][0]["jobs"] = []
        page = _page(tmp_path, pv)
        assert "Run the jobs" not in page
        assert "## Step 5. Check each job" in page and "| smoke |" in page

    def test_no_stage_script_still_gives_the_stages(self, tmp_path):
        pv = _preview(tmp_path)
        (tmp_path / pv["runs"][0]["stage_script"]).unlink()
        s = _section(_page(tmp_path, pv), "## Step 4.")
        assert "Each stage below is its own job, in the queue `small`. They run in this order." \
               in s
        assert "| smoke | runs a few days from the starting state | 45 min | case |" in s

    def test_no_expectations_leaves_the_change_row_out(self, tmp_path):
        pv = _preview(tmp_path)
        for rel in (pv["runs"][0]["expectations"], "domain_case/dry_summer/expectations.json"):
            (tmp_path / rel).unlink()
        s = _section(_page(tmp_path, pv), "## Step 5.")
        assert "the change |" not in s and "expectations" not in s
        assert "| case | Before the model is built" in s
        assert "The stream file is in the case" not in s

    def test_no_words_for_the_checks_keeps_the_rules(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("checks_in_words")
        s = _section(_page(tmp_path, pv), "## Step 5.")
        rows = [ln for ln in s.splitlines() if ln.startswith("| ") and "---" not in ln]
        assert [r.split(" | ")[0] for r in rows] == ["| Job", "| each year, the change"]

    def test_no_checks_at_all_leaves_step_5_out(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("checks_in_words")
        for rel in (pv["runs"][0]["expectations"], "domain_case/dry_summer/expectations.json"):
            (tmp_path / rel).unlink()
        page = _page(tmp_path, pv)
        assert "Check each job" not in page and "## Step 6. The input files" in page

    def test_no_input_check_takes_the_files_from_the_recipe(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("inputs_check")
        s = _section(_page(tmp_path, pv), "## Step 6.")
        assert "| Surface data | `/inputs/valley/surf.nc` |" in s
        assert "| Starting state | the saved state of step 3 |" in s
        assert "checked" not in s

    def test_a_failed_input_check_says_why(self, tmp_path):
        pv = _preview(tmp_path)
        pv["inputs_check"].update({"pass": False, "reasons": ["the grids differ"]})
        s = _section(_page(tmp_path, pv), "## Step 6.")
        assert "as one set of 120 cells, and the set failed. The grids differ." in s

    def test_a_stream_that_could_not_be_read_is_named_not_called_the_weather(self, tmp_path):
        pv = _preview(tmp_path)
        pv["forcing"][1] = {"stream": "datm.streams.txt.presaero.clim_2000", "years": [1, 1, 1],
                            "error": "no such file"}
        s = _section(_page(tmp_path, pv), "## Step 6.")
        assert "| The stream `datm.streams.txt.presaero.clim_2000` | not readable. No such file. |" \
            in s
        assert s.count("| Weather") == 1 and "aerosol" not in s

    def test_a_namelist_file_a_run_changes_is_not_called_the_control_runs(self, tmp_path):
        pv = _preview(tmp_path)
        rp = tmp_path / pv["runs"][0]["recipe"]
        r = json.loads(rp.read_text())
        r["user_nl"]["elm"][2] = "paramfile = '/inputs/params_other.nc'"
        rp.write_text(json.dumps(r))
        page = _page(tmp_path, pv)
        assert ("| Parameter file, `paramfile` in `user_nl_elm` | `/inputs/params.nc` | "
                "`/inputs/params_other.nc` |") in page
        s = _section(page, "## Step 6.")
        assert "parameter file" not in s and "the nitrogen deposition file" in s

    def test_no_weather_streams_leave_the_weather_row_out(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("forcing")
        s = _section(_page(tmp_path, pv), "## Step 6.")
        assert "| Weather |" not in s and "aerosol" not in s

    def test_no_weather_file_leaves_the_stream_file_alone(self, tmp_path):
        pv = _preview(tmp_path)
        pv["runs"][0]["weather_files"] = []
        s = _section(_page(tmp_path, pv), "## Step 3.")
        assert "One file is written for the run. It is in `01_inputs/runs/dry_summer/weather/`." \
               in s
        assert "to read the file of the change" in s

    def test_no_control_record_leaves_step_2_out(self, tmp_path):
        pv = _preview(tmp_path)
        pv.pop("control")
        pv.pop("control_recipe")
        heads = [ln for ln in _page(tmp_path, pv).splitlines() if ln.startswith("## ")]
        assert "## Step 2. Make the changes" in heads


# ── what the server changed since the page's words were written ─────────────
class TestTheServersWords:
    def test_the_pages_sentences_were_written_from_the_servers_words(self):
        for kind, words in RB.SERVER_WORDS.items():
            assert run_gate.CHECKS_IN_WORDS[kind] == words, kind

    def test_words_the_server_changed_are_printed_as_it_gives_them(self, tmp_path):
        pv = _preview(tmp_path)
        pv["checks_in_words"]["year"] = ("the year's logs end cleanly \u2014 against the "
                                         "control's")
        s = _section(_page(tmp_path, pv), "## Step 5.")
        assert "| each year | The year's logs end cleanly, against the control run's. |" in s

    def test_an_en_dash_in_the_servers_words_does_not_reach_the_page(self, tmp_path):
        pv = _preview(tmp_path)
        pv["checks_in_words"]["year"] = "the logs of 2033\u20132034 end cleanly"
        pv["inputs_check"].update({"pass": False,
                                   "reasons": ["the grids differ in rows 1\u20133"]})
        page = _page(tmp_path, pv)
        assert "\u2013" not in page and "\u2014" not in page
        assert "| each year | The logs of 2033-2034 end cleanly. |" in page
        assert "The grids differ in rows 1-3." in page

    def test_a_year_the_reference_holds(self, tmp_path):
        pv = _preview(tmp_path)
        spec = tmp_path / pv["runs"][0]["chain_spec"]
        d = json.loads(spec.read_text())
        d["reference"]["gate_years"] = [2033]
        spec.write_text(json.dumps(d))
        s = _section(_page(tmp_path, pv), "## Step 5.")
        assert "In 2033 they are also held to the model server's limits." in s
        assert "not held" not in s


# ── the parts a change of another kind writes ────────────────────────────────
class TestOtherChanges:
    def test_a_year_job_that_sets_its_own_settings(self, tmp_path):
        pv = _preview(tmp_path)
        ex = tmp_path / pv["runs"][0]["expectations"]
        d = json.loads(ex.read_text())
        d["year_settings"] = {"2034": [["DATM_YR_ALIGN", 2031], ["DATM_YR_START", 2031]]}
        ex.write_text(json.dumps(d))
        s = _section(_page(tmp_path, pv), "## Step 4.")
        assert "| y2034 | runs 2034, after setting `DATM_YR_ALIGN=2031` and `DATM_YR_START=2031` " \
               "| 1 h 55 min | y2033 |" in s

    def test_rules_of_other_kinds(self):
        zero = {"rule": "all_zero", "field": "FIRE_AREA", "stream": "h0",
                "why": "no fire can burn with the fire model off"}
        same = {"rule": "equals_control_year", "fields": ["TBOT"], "tolerance": 0.0,
                "why": "daily TBOT is the control's for the recorded year"}
        assert RB.rule_words(zero, []) == ("`FIRE_AREA` is 0 in every cell and every record of "
                                           "the year, because no fire can burn with the fire "
                                           "model off.")
        assert RB.rule_words(same, []) == ("Daily TBOT is the control run's for the recorded "
                                           "year. The tolerance is zero.")

    def test_a_change_in_the_land_namelist_and_the_xml(self):
        ctl = copy.deepcopy(CTL_RECIPE)
        run = copy.deepcopy(CTL_RECIPE)
        run["user_nl"]["elm"].append("use_nofire = .true.")
        run["xml"].append(["CCSM_CO2_PPMV", "560"])
        run["xml"] = [e for e in run["xml"] if e[0] != "NTASKS"]
        rows = RB.change_rows(run, ctl)
        assert ("CO2, `CCSM_CO2_PPMV`", "not set", "560") in rows
        assert ("`NTASKS`", "40", "not set") in rows
        assert ("Fire, `use_nofire` in `user_nl_elm`", "not set", "`.true.`") in rows
        assert len(rows) == 3

    def test_the_same_recipe_under_a_new_name(self):
        run = dict(copy.deepcopy(CTL_RECIPE), case="valley_ctl_control")
        assert RB.change_rows(run, CTL_RECIPE) == []

    def test_a_source_file_or_user_file_the_run_does_not_copy(self):
        ctl = dict(copy.deepcopy(CTL_RECIPE), user_files={"user_x": {"from": "/a/x"}})
        run = dict(copy.deepcopy(CTL_RECIPE), sourcemods={})
        rows = RB.change_rows(run, ctl)
        assert ("Source file `src.elm/LeafMod.F90`", "copied into the case", "not copied") in rows
        assert ("`user_x`", "copied into the case", "not copied") in rows

    def test_several_queues_get_a_column(self, tmp_path):
        pv = _preview(tmp_path)
        pv["runs"][0]["jobs"][-1]["partition"] = "long"
        s = _section(_page(tmp_path, pv), "## Step 4.")
        assert "| Stage | What it does | Wall time | Waits for | Queue |" in s
        assert "| report | writes each job's state and time | 10 min | every job above | `long` |" \
               in s
        assert "of the queue" not in s

    def test_a_study_folder_that_was_moved(self, tmp_path):
        rd = tmp_path / "moved" / "study_1"
        pv = _preview(rd)
        old = str(tmp_path / "before" / "study_1")
        for w in pv["runs"][0]["weather_files"]:
            w["path"] = w["path"].replace(str(rd), old)
        s = _section(_page(rd, pv), "## Step 3.")
        assert "`01_inputs/runs/dry_summer/weather/`" in s


# ── the small helpers ────────────────────────────────────────────────────────
class TestHelpers:
    @pytest.mark.parametrize("wall,said", [("01:55:00", "1 h 55 min"), ("01:00:00", "1 h"),
                                           ("00:45:00", "45 min"), ("00:10:00", "10 min"),
                                           ("1-02:00:00", "1 d 2 h"), ("00:00:30", "30 s"),
                                           ("soon", "soon")])
    def test_wall_times_read_as_a_person_says_them(self, wall, said):
        assert RB.wall_words(wall) == said

    def test_months_of_the_same_kind_each_year(self):
        assert RB.months_words(["2033-06", "2033-07", "2033-08", "2034-06", "2034-07",
                                "2034-08"]) == "June, July and August of 2033 and 2034"
        assert RB.months_words(["2033-06", "2034-08"]) == "June of 2033 and August of 2034"

    def test_a_namelist_with_a_continued_value_and_comments(self):
        nl = RB.namelist(["a = 'x!y' ! a comment", "! a whole comment", "b = 1,",
                          "    2", "", "c = .true."])
        assert nl == {"a": "'x!y'", "b": "1, 2", "c": ".true."}

    def test_a_smoke_stage_of_one_day_says_one_day(self, tmp_path):
        assert RB.stage_words("smoke", {"smoke_days": 1}, None, {}) == \
            "runs 1 day from the starting state"
        assert RB.stage_words("smoke", {}, "gate smoke --days 1", {}) == \
            "runs 1 day from the starting state"
        assert RB.stage_words("smoke", {"smoke_days": 2}, None, {}) == \
            "runs 2 days from the starting state"
        pv = _preview(tmp_path)
        script = tmp_path / pv["runs"][0]["stage_script"]
        script.write_text(script.read_text().replace("--days 3", "--days 1"))
        s = _section(_page(tmp_path, pv), "## Step 4.")
        assert "| smoke | runs 1 day from the starting state | 45 min | case |" in s

    def test_tolerances_as_plain_decimals(self):
        assert RB._decimal(1e-05) == "0.00001" and RB._decimal(0.001) == "0.001"
        assert RB._decimal(2.0) == "2"

    def test_the_weather_file_with_both_fields_acting(self, tmp_path):
        w = _weather(tmp_path, "warm")
        w["fields"]["tas"].update(months=["2033-07"], cells=10, values=[2.0])
        assert RB.weather_words(w) == (
            "An amount added to the air temperature and a factor on precipitation for each "
            "month from January 2032 to December 2035. The amount is 2 K in 10 cells in July of "
            "2033, and 0 K everywhere else. The factor is 0.6 in 120 cells in June and July of "
            "2033 and 2034, and 1 everywhere else.")
