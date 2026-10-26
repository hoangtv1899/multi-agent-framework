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
| `weather_repeat_year` | which forcing year each model year reads | no: piece 4b |
| `weather_scale` | rain, snow or temperature scaled in the forcing | no: piece 4b |
| `burn_footprint` | a burn imposed on chosen cells on a chosen date | no: a later piece |
| `parameter_value` | one value in the parameter file | no: a later piece |

Setting-only changes go into the run's recipe (`user_nl` lines or XML settings)
and need nothing new to run. Their meaning is checked twice: in the case stage,
the run's resolved namelists must differ from the control's only by the change;
after each year, the output checks must hold.

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

Out of scope here: weather changes (piece 4b), an imposed burn, and input sets
for a new watershed.

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
