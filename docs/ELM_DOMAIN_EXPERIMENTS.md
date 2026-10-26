# Whole-domain ELM experiments: a control run and named changes (piece 4)

A whole-domain experiment answers a question about one prepared domain, such as
*"how does the Naches behave in the years after a drought?"* or *"what does the
model's own fire do to its carbon?"*. It is never a sample of columns. It is a
**control run** (one CIME case over the whole domain, rebuilt from its recorded
recipe, piece 3) and, for a what-if question, one or more **runs that differ from
the control by named changes**: fire off, a different CO2 level, nitrogen
deposition held at another year.

This document is the plan for that. It is not a description of what exists; the
status table at the end says what is built.

---

## Three rules that constrain everything below

**1 · Additive only.** The site, conceptual and coupling paths must not move. Every
change is a new file, a new key the other paths never set, or a branch whose
`else` is the existing code. The suite's FAILED list before this work is the
baseline (69 failed and 2 collection errors, 2026-10-23), and every step diffs
against it.

**2 · No ELM knowledge in the framework.** What a change does to ELM, what it
means, how to check it took effect, what a history field's units really are:
all of it lives in `mcp/elm-mcp/`. The framework holds the experiment's shape
(a control and runs), the regions and the arithmetic of comparing runs.

**3 · A design may use only what the server declares.** The planner designs from
the server's change list, the gate refuses a change that is not on it, and a
question that needs a change the server does not offer comes back `partial` or
`infeasible` with that change named in `requires`. The planner's verdict is
acted on: an `infeasible` plan stops before materialize.

---

## What the server declares

Four new tools, in the order a study uses them, following the conceptual
sweep's declare, check and build trio (`describe_conceptual_factors`,
`check_conceptual_design`, `build_conceptual_columns`):

| Tool | Returns |
|---|---|
| `describe_domain_controls` | the prepared controls this machine has: name, case, the years with output, how the control was reproduced, named regions (polygons, burn footprints), gridded observations on the model grid, and the control's facts |
| `describe_domain_changes` | the change list: every change the server can make to a control, available or not |
| `check_domain_experiment` | for a proposed experiment: `wont_build`, `unusual` and `facts`, the same three lists the sweep check returns. The server reports; it does not grade |
| `build_domain_runs` | the per-run recipes and chain specs, written under `01_inputs/runs/<run>/` |

Each result stays well under the 12,000 characters a reception tool result is
cut at, which is why controls and changes are separate tools rather than more
keys in `describe_elm_capabilities`.

### A change

```
id            fire_off
summary       switch ELM's fire model off
params        {}                       (a schema: name -> type, valid values, units)
alters        user_nl_elm use_nofire = .true.
meaning       the difference from the control is the effect of the model's own
              fire, as the control's facts measure it
not_meaning   the effect of an observed fire: the control's fire is the model's,
              in its own places and years
checks        settings: the run's lnd_in has use_nofire = .true. and differs from
              the control's lnd_in in nothing else
              output:   FAREA_BURNED is 0 in every cell and month (h0)
cost          the same as the control: one job per model year
available     true
requires      []                       (for an unavailable change: the tool it needs)
```

The first list:

| Change | Alters | Available |
|---|---|---|
| `fire_off` | `use_nofire = .true.` | yes |
| `co2_ppmv` | `CCSM_CO2_PPMV` (the control: 367, constant) | yes |
| `ndep_year` | `stream_year_first_ndep` and `_last_ndep` (the control: 2000) | yes |
| `nutrient_supplement` | `suplnitro`, `suplphos` (the control: NONE, ALL) | yes |
| `weather_repeat_year` | which recorded year each model year's weather comes from, set by that year's job | yes |
| `weather_scale` | an added temperature and a precipitation factor, by month and cell, through the data atmosphere's anomaly forcing | yes |
| `burn_footprint` | a burn imposed on chosen cells on a chosen date | no: a later piece |
| `parameter_value` | one value in the parameter file | no: a later piece |

Setting-only changes go into the run's recipe (`user_nl` lines or XML settings)
and need nothing new to run. Their meaning is checked twice: in the case stage,
the run's resolved namelists must differ from the control's only by the change;
after each year, the output checks must hold.

A change that cannot be written as a setting writes files for its run
(`alters: [{"inputs": <writer>}]`). `build_run` writes them into the run's
inputs folder before the recipe is made; the recipe then carries the user_nl
lines and user files that read them, and the writer adds its own checks.
`weather_scale` is the first (see "A run's own weather" below). A change can
also set each year's settings in that year's job (`alters: [{"years":
<writer>}]`): they are in neither the recipe nor the case stage's check, and
the writer's own rules check them year by year (`weather_repeat_year`, below).

### A control's facts

Computed once from the control run's own output and cached beside the
registry, because reading seven years takes minutes:

- per year: GPP, NPP, peak LAI, ET, runoff, 1 April snow, water closure, the
  model's own burned area and, where a burned-area observation is on the grid,
  the observed one;
- the forcing years the case's weather stream covers, the history streams and
  the fields each one holds;
- the control's settings that changes alter (CO2, fire, nitrogen deposition,
  nutrients) and the source tree it was reproduced with.

For the Naches control these facts carry the finding that made rule 3 matter:
the model's fire burns 32-132 km2 a year spread over 1,864-5,296 cells, while
MODIS maps 208, 286 and 344 km2 of concentrated burns in 2017, 2020 and 2021
(computed 2026-10-23 in 12.6 minutes, weighted by the model's own cell areas).
A design that reads the facts cannot describe a no-fire twin as the effect of
the 2021 fire.

### The registry

`describe_domain_controls` reads a local JSON file named by `paths.json` key
`domain_controls` (or `IDEAS_DOMAIN_CONTROLS`), git-ignored because it names
local paths; `domain_controls.json.example` shows its shape. A control may be
**reused**: an existing run whose case the recipe was read from, with a record
of its reproduction (for the Naches: rebuilt bit for bit, 2015-2016, chain
790947-790951). A reused control spends no compute; the check reports, as a
fact, whether a run is built from the source tree the reproduction used.

### Field meanings

A table in the server gives, for each gridded history field the framework
reads: its units as ELM really computes them, whether it is a state or a flux,
and how a year is totalled. It carries the traps found by hand: `FAREA_BURNED`
is a fraction per second despite its label, and `FIRE` is emitted longwave
radiation, not fire.

---

## The experiment record

`01_inputs/domain_experiment.json`, written by the manager from the plan:

```
control   {"name": "naches", "reuse": true}     or a recipe, as in piece 3
runs      [{"name": "nofire", "changes": [{"id": "fire_off"}]}]
years     [2015, 2021]
analysis  {"regions": [...], "variables": [...], "normal_years": [...]}
```

The rulebook the plan was made under (the change list and the control's facts,
as fetched) is written beside `strategy.json`, as the pinning block already is,
so the gate and the manager enforce exactly what the planner read.

Each executed run has its own case (`<control case>_<run>`) and its own record
directory `domain_case/<run>/`, laid out as piece 3's: recipe, scripts,
`chain.json`, `STATUS.log`, `checks/`.

`experiment.json` gains `runs` (their changes, with meaning and not-meaning
quoted from the change list), regional rows per run and per year, the
difference of each run from the control per region and year, and the control's
facts.

---

## The flow

```
reception    archetype whole_domain; reads describe_domain_controls and
             describe_domain_changes through local tools; settles the control,
             the runs and the years with the user, as it settles a sweep
planner      a whole-domain section: carries the runs; chooses the regions,
             variables, observations and normal years; gives the verdict
gate         checks the runs against the rulebook (check_domain_experiment);
             STOPs on wont_build or an infeasible verdict
review v2    before the runs: the control, each run as control plus changes
             with meaning, not-meaning, checks and cost; after them: the
             recipe a person could rerun by hand, with job ids and check results
manager      build_domain_runs, then one chain per executed run and job B after
             all of them; extract regional rows per run; package
report       rendered from the records, no LLM: one headline claim with its own
             numbers, the tables by region and year, each change's meaning with
             its not-meaning, the caveats and two figures (DOMAIN_REPORT.md)
analyzer     the LLM Analyzer on the same records: step 0's regional frame, the
             server's burned area beside the observed one as step 1's finding,
             a whole-domain brief for the figures, audited claims (REPORT.md)
```

---

## Order of work, and how each step is tested

| Step | What | Tested by |
|---|---|---|
| 1 | server: field meanings, change list, check, per-run recipes; the registry and the control's facts | unit tests on synthetic cases; the facts of the Naches control against the numbers in this document |
| 2 | server: per-run layout in the chain tools; settings check in the case stage; output checks in the year gate | stage scripts run against stub CIME commands, as piece 3's are |
| 3 | framework: experiment record, gate branch, verdict acted on, manager running one chain per run | the manager driving the real server tools with a fake SLURM |
| 4 | framework: reception and planner sections, local tools, review v2 | prompt-text tests and fake clients; one dry planner call on a recorded reception (LLM cost only) |
| 5 | framework: the regional numbers (5a), the report rendered from them (5b, option b) and the LLM Analyzer on the same rows (5b, option a) | offline on the Naches control's output: the hand analysis's numbers and headline; one live Analyzer run |
| 6 | proof: a fire-off twin of the Naches, 2015-2016, against the reused control | real jobs, asked for first |

Out of scope here: the rest of the weather tool (a lapse rate), an imposed
burn, and input sets for a new watershed.

---

## A run's own weather (piece 4b, first part: `weather_scale`)

The part of E3SM that hands recorded weather to the land model, the data
atmosphere (DATM), can add a fixed amount to the air temperature and multiply
the precipitation, cell by cell and at every model step, after the recorded
weather has been mapped onto the model's cells (E3SM maint-2.0,
`datm_comp_mod.F90`; E3SM calls this anomaly forcing). `weather_scale` uses it,
so no forcing file is copied or rewritten. `domain_weather.py` writes, for each
run:

| File | What it holds |
|---|---|
| `weather/weather_anomaly_<checksum>.nc` | the amount added to the temperature (`tas`, K) and the precipitation factor (`pr`), one record per month at the month's first instant, from the year before the run's first to the year after its last; the change in the chosen months and active cells, 0 and 1 elsewhere; the domain file's own `xc`, `yc`, `area` and `mask`. Named by its content, so building a run again never overwrites a file a chain reads |
| `weather/user_datm.streams.txt.Anomaly.Forcing.Temperature`, `...Precip` | the stream file through which DATM reads each changed field; the anomaly file describes its own grid |
| `user_nl_datm` lines | `anomaly_forcing` naming the streams, and `tintalgo` as the control's list with `lower` for each stream |

Six facts of DATM and CIME fix this shape, each read in the code:

- the anomaly streams' years are 2006-2300, with the data year equal to the
  model year (`namelist_definition_datm.xml`), and DATM reads no record dated
  before 1 January 2006; the first year's January value takes effect just
  before that year begins (the lead, below), and the last time DATM is called
  in a year's job is 00:00 on 1 January after the run's last year, so a run's
  years must lie in 2007-2299;
- a stream's record spacing may vary by at most `dtlimit` (1.5), so the
  records are monthly, and with `lower` interpolation each moment takes its
  own month's record (CIME's default for these streams is `nearest`, which
  would give the second half of every month the next month's record);
- the file's cells are the model's to well within DATM's 0.01 degrees, so DATM
  maps nothing (`shr_strdata_mod.F90`) and each cell takes its own value; only
  active cells (domain mask 1) are changed, and a change that reaches none is
  refused;
- CIME copies a user stream file into the case unchanged (`CIME/nmlgen.py`);
- each land step takes the weather DATM computed at the step's start (the
  driver runs the land, then the atmosphere, after each clock advance:
  `cime_comp_mod.F90`), and DATM moves to the next pair of records only when
  the time is strictly past the later one (`shr_dmodel_readLBUB`). With
  records at midnight, the step that starts at 00:00 on the 1st would keep the
  previous month's value, so each stream file moves its records 60 s earlier
  (the lead, written as the stream's `<offset>`). `tests/test_domain_weather.py`
  replays these rules step by step on the written files, and the proof found
  1 June and 1 September exact;
- DATM splits rain from snow, caps the air's moisture (specific humidity) at
  saturation and sets the potential temperature from the recorded temperature,
  before it adds the amount and multiplies rain and snow, each, by the factor;
  the land model then computes relative humidity and air density itself, from
  the temperature it receives (`lnd_import_export.F90`).

The checks:

| When | What must hold |
|---|---|
| the experiment check | the control is reused (the file is written on the grid of its resolved datm_in, and every year is checked against its output); a region is one registered for the control; the years lie in 2007-2299; the control's daily history (h1) holds TBOT, RAIN and SNOW for every year of the experiment. It notes, without refusing, a cooler run (below) and a run that changes the weather and something else (each change's meaning is stated against the control alone, so the difference is their joint effect) |
| the case stage | the run's datm_in names the anomaly streams; every per-stream list is the control's with the streams' entries appended and nothing else changed; the case holds the stream files written for the run (checksums) |
| every year | in every cell-day both runs report, the run's daily TBOT minus the control's equals the amount added (to 0.001 K), and its daily rain and its daily snow over the control's each equal the factor (to 1e-5), so a change of phase is caught; a day without precipitation in the control stays dry. Both fields are checked even when one is left alone, and outside the chosen months and cells each must be the control's. A cell-day reported by one run only fails, and so does a change that reached no cell the history reports |

Building a run's inputs again is refused while a chain armed for any run of
the experiment may still run, since the model reads the weather files while it
runs.

What a run does not mean (the change's `not_meaning`): a consistent change of
climate. A warmer run does not turn snow into rain, and the air's moisture,
radiation and wind are the control's; so warmer air is also drier air to the
land model, its relative humidity about 6-7 % of its value lower per kelvin,
and ELM's fire model and leaves respond to that. In a cooler run the recorded
moisture can exceed saturation for the cooled air, so dew and frost form that
the recorded weather would not make.

---

## A recorded year in another model year (piece 4b, second part: `weather_repeat_year`)

A run can make chosen model years read another recorded year's weather: a
dry year three times running, say. The change's parameter is a map from model
year to recorded year; a year it does not name reads its own. No file is
written for the run. Every chain year is its own job, and the job sets which
years of the forcing the data atmosphere reads before it runs
(`DATM_CLMNCEP_YR_START`, `_END` and `_ALIGN`); CIME writes the namelists, and
the forcing's list of files, anew when each run starts. The smoke run takes
the first year's settings. These settings are not in the run's recipe, which
keeps the control's forcing years: they are in its expectations
(`year_settings`), in the year arms of its stage script, and in a line of the
recipe's notes.

Every year of such a run, its own years included, reads the same two-year
form: its recorded year W and W+1, aligned so that the model year reads W.
Model year 2016 reading 2015 sets 2015-2016 aligned to 2016. The last time
the data atmosphere is called in that year's job, at 00:00 on the next 1
January, it then reads 2016's first record, as the control's own 2015 does.

What happens across a restart was read in E3SM's code and then tested on a
clone of the Naches control (job 791045: three 5-day runs, restarted on 6
and 11 January 2015, inside one year; 0.14 node-hours; records in
`workflow_outputs/elm_restart_test_20261026`):

- after a restart the data atmosphere takes the years from the namelist, not
  from its restart file (`shr_stream_restRead` reads the stored years and
  discards them). It restores its stored record positions unless a file whose
  times it had read has another name at the same place in the new list, and
  finds them afresh otherwise. Both paths were run. Each run's largest daily
  difference from the control's recorded year, 0.36 K and 0.074 K in
  temperature, is what one carried step on the first day gives; the proof's
  2016 checks every other day at a tolerance of zero;
- the first land step after a restart takes the atmosphere's last output from
  the coupler's restart (`seq_rest_mod.F90`, `a2x_ax`), which the previous job
  computed at the restart instant with its own years. At a year boundary that
  is the record for 00:00 on 1 January of the year after the previous year's
  recorded year: for 2016 reading 2015 after 2015, 2016's own first record,
  the control's value. So where a year's recorded year does not follow the
  previous year's, its first model step (30 minutes on the Naches) has that
  record rather than its own recorded year's;
- restored positions go wrong when the new list of files starts with the old
  one but is shorter (they point past its end) or longer (the old end caps it,
  and the year wraps without an error). The two-year form rules both out:
  every year's list has the same length, and two different lists differ in
  their first file, which is always read.

The form needs a forcing whose records of each year start at 00:00 on 1
January, as the Naches control's NLDAS records do. Where they start later
(GSWP3 at 01:30, CRUNCEP at 03:00), the data atmosphere would take the hours
before from the end of the next recorded year, so every 1 January would
differ from the control's: such a forcing is refused, and so is a control
whose model years are not its recorded years (`_ALIGN` other than `_START`),
whose forcing has a fixed list of files (a user stream file), or which also
reads bias-correction or anomaly streams (they stay on the model year). A
three-year form, W-1 to W+1, would serve the later-starting forcings, after a
proof of its own.

The checks:

| When | What must hold |
|---|---|
| the experiment check | the control is reused, and its model years are its recorded years; every model year the map names is one the experiment runs; every recorded year read is one of the control's output years, with the year after it in the forcing the control's case reads; the control's daily history holds TBOT, RAIN and SNOW for those years. A run changes the weather at most once: not with `weather_scale` too, since each change's checks read the other's weather as the control's |
| building the run | the forcing as above (records from 00:00, a list of files CIME makes, no bias-correction or anomaly streams); a chain whose years the run's year settings do not cover is refused, and a year job without settings stops before its model runs |
| every year | the data atmosphere's log for the year's job names the forcing streams with its recorded year and the year after, aligned so that the model year reads it; the run's daily TBOT, RAIN and SNOW equal the control's for the recorded year, day of the year by day of the year and cell by cell, exactly. 1 January of a year whose recorded year does not follow the previous year's is reported, not held |

What a run does not mean (the change's `not_meaning`, in full): a drought of a
kind the record has never held, a forecast or a climate scenario. Each
repeated year is the recorded year, with its storms on the same dates, and the
land carries its state over, so a repeated dry year meets the soil and
vegetation the years before it left. Where a year's recorded year does not
follow the previous year's, its first model step has the record after the
previous year's recorded year, since E3SM carries the atmosphere's last output
across the restart.

---

## The Analyzer on a whole domain (step 5b, option a)

The Analyzer's five steps read a whole-domain experiment through the same
records the rendered report reads; nothing is computed twice and no ELM name
enters the framework.

| Step | What it does for a whole domain |
|---|---|
| 0 | classifies the experiment's keys (`kind`, `regional`, `control_facts` and `chain` are evidence; `experiment` is the design, carried as `ctx.plan["experiment"]`); builds the regional frame (run, region, variable, year, value, units, how, source); holds the plan to its years, runs, regions, fields and normal years; makes caveats from the records: each change's not-meaning (blocking), the server's check notes, the planner's caveats, field notes, regional problems, an incomplete chain, and the server's whole-domain limitations |
| 1 | no station comparison (none is fetched for a whole domain, by design); each observation the server set beside the model's own (the burned area per region and year, the model's side made in the server from its own field) becomes a finding step 3 may cite |
| 2 | a whole-domain brief and task: the control and each run with what its difference means and does not mean, the regions with their cells and areas, the normal years, the fields of `reg`; figure scripts get `reg`, `regions` and `runs` |
| 3 | facts of record: each region's cells and area, the years, the normal years, the counts of regions and runs |
| 4 | the compute from each chain's report stage (the job times it read from the scheduler); the scope line names years and runs, not columns, and REPORT.md points to DOMAIN_REPORT.md |

The analyze stage writes DOMAIN_REPORT.md first, then runs the Analyzer;
`config["analysis_llm"] = False` stops after the first. A rendered report that
fails, or regional numbers that could not be made, fail the stage; the
Analyzer's preflight then stops before any model call.

---

## Status

| Step | State |
|---|---|
| 1 | built 2026-10-23: domain_changes.py, domain_controls.py, gridded.FIELD_NOTES, the registry path; four tools and the facts refresh in main.py; 41 + 7 tests; the Naches control's facts computed |
| 2 | built 2026-10-23: chain tools take run=<name>; the case stage's settings check and the year stages' --expect; a run waits for an executed control's case job; 6 + 9 tests |
| 3 | built 2026-10-23: strategy `domain_experiment`; the gate's shape checks and the infeasible verdict acted on (whole-domain only); the manager's experiment mode (rows per run, one chain per executed run, job B after all, differences from the control); 13 tests |
| 4 | built 2026-10-24: reception's whole_domain archetype, STEP 2c and two menu tools; its gather skips every fetch; the planner's whole-domain section and rulebook block, written beside strategy.json; the gate needs no basin and takes the experiment's years; Design Review v2's whole-domain page; the server states that sampled columns have no carbon cycle; an experiment may be the control alone; 11 tests. One live planner call (2026-10-24) on a fire question with a moved burn returned `partial`, the burn tool in `requires`, the change's not-meaning and the control's burned-area facts as caveats, and passed the gate |
| 5 | 5a built 2026-10-24: domain_analysis.py, per run, region and year (a rate totalled, a state averaged), differences from the control, anomalies against normal years, the observed burned area per region; written to 03_results/regional.json and experiment.json; the Naches control's GPP and LAI by year, normals, 2015 and 2021 anomalies and the watershed's 4,127 cells reproduced from the monthly stream (7 real-data checks). 5b built 2026-10-25 as a report rendered from the records with no LLM (`src/agents/analysis/domain_report.py`): the headline is one claim with that claim's own numbers (the first run's difference from the control, else the target years against the normal, else the values), tables by region and year, what each change means beside what it does NOT mean, the caveats and two figures; the analyze stage renders it in place of the Analyzer, and a failed render, or regional numbers that could not be made, is a failed analysis rather than a skip; a control whose facts are stale says so instead of passing; 17 unit tests, 4 through the manager, 1 on the Naches output, 22/22 mutants killed. On the Naches control the headline is the hand analysis's: GPP 5.86 % above its 2016-2020 normal (460 gC/m^2/yr) in 2015 and 18.6 % below it in 2021. 5b option a built 2026-10-25: the Analyzer reads a whole-domain experiment (see "The Analyzer on a whole domain" below); the rendered report moved to DOMAIN_REPORT.md beside the Analyzer's REPORT.md; 22 tests of the Analyzer's steps, 5 of the server's additions (the model's burned area, the limitations, the chain's job times), 3 more through the manager, 2 more on the Naches output; 31 of 32 mutants killed (the survivor was a redundant clause, since removed); one live run on the Naches control (3 model calls) kept 10 of 10 claims, each recomputed from the table |
| 6 | done 2026-10-25: a fire-off twin of the Naches, 2015-2016, against the reused control, through the framework's back half (one live planner call chose the analysis; the gate, the server's check, the run's recipe and checks, its chain and job B); chain 791021-791025, job B 791026. The first arming stopped before any job: materialize demanded a bounding box of a whole-domain study, which a real whole-domain reception does not have (the tests' brief carried one); fixed, and the tests now use a brief without one. Every check passed: the case stage's settings check (lnd_in differs from the control's only by use_nofire, datm_in not at all), the build from the control's own source tree (fingerprint 503a94c5d1a0), the smoke test, both years (FAREA_BURNED zero in every cell of every month; vegetation carbon the only field outside the reproduction limits, as fire off should give), the report stage's job times (6,842 s of model time) and job B: the numbers report, then the Analyzer (3 model calls, 6 of 6 claims kept, each recomputed from the table). The model's own fire lowers vegetation carbon over the vegetated domain by 6.96 gC/m2 (2015 mean) and 23.2 (2016 mean) while it burns 115 and 60.3 km2 against MODIS's 3.72 and 2.64 |
| 4b, weather_scale | built 2026-10-25, PROVEN 2026-10-26: `domain_weather.py` (the anomaly file, the stream files, the user_nl_datm lines, the year check), writers in `domain_changes.py` (`build_run`, the `appends` expectation, the case files), the gate's `matches_anomaly` rule with the control's output as reference, and the case stage's checksum of each file written for the run; 95 tests in `tests/test_domain_weather.py`; 37 of 38 deliberate breaks caught before the review (the survivor converts longitudes the region code converts again), 23 of 23 of the review's fixes. Checked on the Naches control without a job: the file sits on its 96 x 144 cells with no offset from the history's; its watershed is 4,142 cells, exactly the analysis's 4,127 once cut to vegetated cells; on a scratch clone resolved by preview_namelists the case stage's check passed (14 differences in datm_in, all expected, none in lnd_in, both stream files unchanged), and failed when the tintalgo line was dropped, CIME's default for the anomaly streams being `nearest`. The first arming of the proof (chain 791029-791032, 2026-10-25) passed its case stage's settings check on the real job and was cancelled during the build, before any model year, when reading DATM's timing showed that records at midnight would leave the first step of 1 June with May's value (the 60 s offset above). The proof, re-armed with the offset (chain 791033-791036: a 2015 twin of the Naches with the air 2 K warmer and rain and snow times 0.8 in June-August over the watershed, against the reused control), passed every stage: the settings check, the build from the control's source (fingerprint 503a94c5d1a0), the smoke test and the year, whose gate compared 5,045,760 cell-days of TBOT (largest error 0.0 K) and 2,608,692 wet cell-days of rain plus snow (largest relative error 1.04e-7, no dry day made wet). An independent reading of the raw files agreed: the change in exactly the 4,142 watershed cells in June-August, 1 June and 1 September exact (2.0 and 0.0 K), rain and snow each times 0.8 (the 1,100 summer snow cell-days included, so the warming moved no snow into rain), the control's everywhere else. Reviewed 2026-10-26 by four independent reviewers, each finding then challenged by a skeptic: 34 findings, all confirmed, all fixed. The largest: the not-meaning said humidity and air density stay the control's, but the land model computes both from the warmed air, so a warmer run is drier air to it; the server's whole-domain note said every run reads the control's own weather; the year check now compares rain and snow one by one, with tolerances of 0.001 K and 1e-5, fails a cell-day reported by one run only and a change that reached no reported cell, and the experiment check confirms the control's daily history. The proof's output passes the stricter check: TBOT 0.0 K off over 5,045,760 cell-days, rain and snow 1.04e-7 and 9.5e-8 over 2,373,059 and 1,028,242 wet cell-days |
| 4b, weather_repeat_year | built and PROVEN 2026-10-26: the year map, each year's DATM settings in the chain's job script (the smoke run takes the first year's), two year rules in the gate (`datm_stream_years` on the datm_in the year's run read, `equals_control_year` against the control's output for the recorded year), and the experiment check's rules; 55 tests in `tests/test_domain_repeat_year.py`; 18 of 18 deliberate breaks caught before the review, 20 of 20 of the review's fixes. Reviewed 2026-10-26 by four independent reviewers, each finding challenged by a skeptic: 34 findings, all confirmed, all fixed. The largest: the Analyzer's note on whole-domain differences named only `weather_scale`; the not-meaning described the carried step as the previous year's weather, where at a year boundary it is the record after the previous year's recorded year; forcings whose records start after midnight, fixed file lists, bias-correction or anomaly streams and model years that are not recorded years are now refused; and the year rule read the run directory's datm_in, which the next year's job overwrites, and now reads the data atmosphere's log for the year's job. The restart test (job 791045) measured the behaviour this rests on. The proof, a 2015-2016 twin of the Naches in which 2016 reads 2015 (chain 791046-791050, 1.8 node-hours), passed every stage: the case's settings check (no difference, as the year settings come later), the build from the control's source (fingerprint 503a94c5d1a0), the smoke run and both years, each year's job setting its own forcing years. 2015 equals the control's 2015 exactly on every day; 2016 equals the control's 2015 exactly on 364 days, over 5,031,936 cell-days of each of TBOT, RAIN and SNOW, and 1 January, the carried day, differs only in TBOT, by up to 0.14 K; against the control's own 2016 it differs by up to 18 K. The data atmosphere's logs show 2016's job restoring its stored positions (the same file list as 2015's) at the year boundary. An independent reading of the raw files agreed, and the reviewed checks pass on the same output |
