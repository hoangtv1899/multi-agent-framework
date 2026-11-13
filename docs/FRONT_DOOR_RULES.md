# What the front door runs, and what counts as a yes

Status on 2026-11-10. Written by hand; the generated documents
(`ARCHITECTURE.md`, `ANALYSIS_METHODS.md`) are not edited to hold it.

These rules are in code. A prompt asks a language model to follow them; it
cannot make it, so none of them rests on a prompt alone.

## The scope is said before the first question

`src/core/scope_banner.py`. Before the first question the coordinator prints
"WHAT THIS VERSION ANSWERS": the prepared runs with their own descriptions
and output years, where each run's comparison with measurements is shown
(the design page), what a what-if run may change (a change no run has
completed with is marked), and what is not answered. Every word about a run
or a change is read from the model server's control registry and change list
at that moment, so the banner names nothing itself and cannot go stale. It
calls no language model. A banner that cannot be read is said in one line
and never stops a request.

## Only the servers needed are started

`supported_routes.servers_to_start`, passed to `MCPManager`. A whole-domain
study reads its control and makes its runs through one model server and
fetches nothing, so the front door starts that server and no other. The
other configured servers (another model's, the data servers) are not
started; their code and their entries in the configuration file are
untouched, and the developer's switch starts them all. A script that makes
its own `MCPManager` with no rule starts every server, as before.

## This version runs whole-domain studies only

A whole-domain study reads a prepared control, and may add runs that differ
from it by named changes. The other kinds of study (single model columns
sampled across a basin, a controlled sweep of single columns, one model driven
by another's output) keep their code and their tests. They are set aside: no
request reaches them from the front door, whatever the person answers.

One rule, `src/core/supported_routes.py`, read at three places:

| Where | When | What happens to a study of a kind set aside |
|---|---|---|
| Reception (`reception_llm.py`, `process`) | after its conversation, before a run directory is made or anything is fetched | the route becomes `unsupported`; the coordinator prints why and stops |
| The coordinator (`workflow.py`, `_workflow_design_and_run`) | after the plan, before the design page could offer a yes | the page is written as a record; the prompt offers no yes |
| The gate (`src/core/strategy_check.py`) | the Experiment Manager's first step, on every way into compute | a STOP |

A developer reaches the kinds set aside by setting `IDEAS_DEVELOPER_ROUTES=1`
in the environment of the process. Nothing typed into the conversation can do
that. The test session sets it (`tests/conftest.py`), since most tests of
those routes predate the rule; the tests of the rule remove it.

Reception is no longer one of the ways in, switch or no switch
(2026-11-11). Its prompt (`prompts/reception_agentic.txt`) is written for
whole-domain studies alone, and its code for the kinds set aside was deleted
with the instructions: the gather of a basin's terrain, stations, rain and
subsurface, the carry-forward of a prior run for a coupling, and the sweep's
branch. A design brief of another kind is refused by Reception itself. With
the switch, the coordinator and the gate still let such a study through when
it arrives another way (a saved brief, a resumed study, a manager called
directly); the managers, the samplers and the data servers of those kinds
are unchanged.

## An infeasible design has no yes

The Planner's verdict `infeasible` stops a design of any kind: the page is
written, the prompt reads "This design cannot run as it is", and the gate
stops it again before compute. Until 2026-11-10 the verdict was acted on for
a whole-domain study only.

## Only a typed yes runs

The prompt is "To run this design, type yes. Type no to stop, or type a
change:". An empty answer approves nothing: the prompt is asked again, and
after three empty answers the design is left unapproved. Any other text is a
change, which goes back to Reception and gives a new design in a new folder,
with a new page and a new prompt.

An unattended run (no `--ask`) has no prompt. A design that only reads a
control's existing output continues and records that it did. A design that
would run the model stops at the design page unless the run was launched
with `--yes`, which is the person's word, given at launch, that the design
the request leads to may be run; the approval then records "approved at
launch".

## An approval is of one design

When a design is approved, `01_inputs/design_approval.json` records how, when
and a fingerprint (sha256) of what the design would run: its control, its
runs and its years. Before the coordinator hands anything to an Experiment
Manager it reads that record back (`design_review.approval_problem`):

- no approval on record: not run;
- the design in hand is not the design approved: not run;
- a resumed study whose page says declined, not run or revision requested:
  not resumed.

A change to the analysis alone (the fields or regions reported) leaves the
fingerprint as it was. A study made before approvals were recorded resumes as
before.

## The years simulated and the years reported

`src/core/analysis_years.py`. The years the numbers are reported for are the
years the person asked about (`run_settings.requested_period`) and the normal
years they are compared with, or the plan's own `analysis.years`. The design
page, the regional table and the study record read this one function.

- The page's When line labels each part: the years asked about, the normal
  years chosen by the design, and years simulated and not reported, which are
  the design's own span. A widened span is never called the person's.
- The control read alone may cover any consecutive years inside its output.

## Where a run starts

Since 2026-11-12 the model server lists the years a run can start from, and
nothing else decides it (`domain_controls.start_years`).

- A control saves its state on every 1 January. The server reads its run
  folder each time it is asked and lists its own first year and each later
  year it has a saved state for, inside its output. The list is in the
  control's description under `facts.start_years`.
- A run simulates the years the question asks about and no others. Its first
  year is one of the listed years, and it starts on 1 January of that year
  from the control's saved state of that day (`domain_changes.start_from`).
  Its case job expects that one more difference from the control's settings.
- A first year that is not listed is refused, with the list.
- Until 2026-11-12 every run started at the control's own first year, so a
  question about 2018 and 2019 simulated 2015 to 2019.
- No run started from a later saved state has been completed here yet. The
  server's check says so in a note, which the design page prints, until
  `domain_changes.LATER_START_DEMONSTRATED` is set after one has passed every
  check of its jobs.

## What Reception reads, and how it ends

- **One prompt, for whole-domain studies alone** (2026-11-11). It is about a
  third of the length it had as the prompt of four kinds of study with a
  block appended that set three of them aside. It names no model, no control
  and no change: those are read from the model server in the conversation.
  `tests/test_reception_prompt.py` holds it to the code that reads its
  output.
- **Four framework tools and the question to the person.** Reception may
  call `list_servers`, `describe_server`, `describe_domain_controls` and
  `describe_domain_changes`, and `ask_user`. It is given no data tool and no
  tool of a kind of study set aside.
- **A framework tool is answered once.** Asked again with the same
  arguments in one conversation, it answers with a short note that points
  back to its first answer. A menu of twelve thousand characters is not
  carried twice.
- **A revision carries the years asked about** as well as the years
  simulated, so a year the design adds is not reported as a year asked about
  after a change at the design page.
- **The run is described once, and only what it is.** The start banner says
  what each prepared run is. Reception's card says the same, and is shown
  only for a run the banner did not name (`LLMReceptionAgent.introduced`,
  set by the coordinator), so the person does not read it twice. How the run
  compares with measurements, the years its anomalies are measured against
  and what it does not contain are printed on the design page, which the
  person reads before approving anything.
- **Reception is not shown the run's results** (2026-11-11,
  `reception_llm.controls_for_the_llm`). Its copy of a control's record has
  neither the comparison with measurements nor the numbers by year; a note
  in place of each says who reports it (the design page; the analysis).
  Handed those numbers, it
  quoted them and then judged them in its own words, against its prompt. It
  still reads the run's years, settings, fire, grid, regions and what the
  run does not contain.
- **The cost is not Reception's to compute.** It says how many model years
  will be simulated; the design page states the cost from the server's own
  figure.
- **The servers' menus are read whole.** A framework tool's answer (the
  prepared runs, the list of changes) may be 24,000 characters; a data
  tool's result keeps the 12,000 cap. A result that is cut all the same is
  said on the terminal and marked `truncated` in the trace. At 12,000 the
  list of changes had 2 characters to spare.
- **Run names already taken.** Each control's record lists the names whose
  case exists already (`run_names_taken`), so a run is named free the first
  time and not refused for its name.
- **A question that cannot be answered ends as a statement.** Reception
  emits `cannot_answer` with a reason; the person reads "Nothing was designed
  or run: ...", not a clarification.
- **An empty question is never shown.** A call of the ask tool with no text
  is answered to the model.

## A change applies in the years it is given

`weather_scale` takes `years`: the model years it applies in, among the
run's years; every simulated year when absent. The anomaly file holds the
identity in a year left out, and the year's check, which reads the file,
holds the run to the control there. A change with no such setting is in
effect from the run's first year, and the page says so by code ("in effect
in EVERY simulated year"). The Planner's verdict is `partial`, by its
prompt, when a change reaches years the question did not name and cannot be
limited to them.

## The Planner designs without the results

`planner._rulebook_for_prompt`. The Planner's copy of a control's record
holds neither its comparison with measurements nor, since 2026-11-12, its own
results: the numbers by year and the model's burned area by year. A note in
their place says the analysis reports them. The Planner's goals and choices
are handed to the step that plans the analysis, so a design written by a
reader of the results would not be written before them. It still reads the
control's years, fields, settings, fire and regions.

## The page says what its tabulated numbers are

The second part of the design page lists the places and quantities the design
names. Where the control has a results store, its heading is "The numbers
tabulated from the runs", and the page says that the answer is read by an
analysis plan written after the runs, which may read other quantities and is
not on the page. For a control with no store the heading stays "The numbers
the answer is read from".

## The design page is short, with two files beside it

`src/agents/design_page.py` (2026-11-12). A whole-domain design is decided on
a short page, `DESIGN_REVIEW.md`: the question, what will run and its cost,
the prepared run in a summary written once, how the answer will be read, what
the study cannot tell, the points to weigh and the decision. The design in
full is in `DESIGN_DETAILS.md`, and the steps to make the runs by hand are in
`RUN_BY_HAND.md`. The page names both.

Short does not mean silent. The page and the details are drawn by code from
the same records, and no model writes either. A caveat, a finding of the
model server, a remark of a plan's review or a thing decided without the
person is printed whole. What the page leaves to the details is description:
the control's set-up, each change's mechanism, the scripts.

- **Where a run differs from the question's own word.** A change in the
  server's list may say what a question's word misses (`asked_otherwise`): a
  question about rain, and a change that multiplies rain and snow together.
  The page prints the server's sentence where the run is described.
- **The places and the years.** Each place an analysis reads is given with
  its cells, its area and its rule. A daily or monthly series is given with
  the years it covers.
- **The prepared run's summary** is a few sentences kept in the control's
  registry entry (`page_summary`), with a version, and printed as it stands.
- **A no is not said to cost nothing.** It submits nothing and uses no
  computing time; the model calls that prepared the page are already spent.
- **Titles.** The planning step writes a short title for each analysis in the
  same call as its plan. The question and the id stay, and a plan with no
  title is listed by its question.
- **The rules for the words** a person reads are in
  `src/agents/prompts/house_style.txt`, and the Planner and the planning step
  are sent them in full. An agent is a model call of its own and inherits no
  style from the tool that wrote the code.

## The analysis plan may be shown before the yes (transitional)

`src/agents/analysis_preview.py`, off unless `IDEAS_PLAN_BEFORE_APPROVAL=1`.
With the switch set, the Analyzer's own planning step (step 2a, with its
check, its review and its one rewrite: `step3_interpret.plan_and_review`)
writes its plan from the design before the page. The short page shows what
is compared, each analysis with its places and periods, and every remark of
the plan's review. The details file shows the plan in full.

This is a look before the decision and nothing more:

- The yes is unchanged. It approves the control, the runs and the years.
- The Analyzer does not carry this plan out. After the runs it writes its own
  plan in the study's `04_analysis`, as before, and that plan may differ. With
  the switch set, a study that runs is therefore planned twice. The next
  section describes the switch that changes this.
- A plan that could not be written, or that did not pass its check, is shown
  as it is and stops nothing. A design that cannot be approved is not planned
  for.
- The plan and its records are kept under the study's
  `01_inputs/analysis_preview/`. No record of the Experiment Manager is
  touched.

Binding the yes to a plan is not built. Until it is, the page must not be
read as the approval of an analysis, and it says so wherever it shows a plan.

## The plan shown is the plan approved and carried out

`src/agents/analysis_handoff.py`, off unless `IDEAS_CARRY_OUT_SHOWN_PLAN=1`,
and only with the preview of the section above. With both switches set, a
plan that passed its check and its review is promised to the Analyzer when
the page is written. The page then says that a yes approves this plan
together with the runs and the years, and that the Analyzer carries it out as
it stands and writes no second one, after the runs or, for a design that
makes no new run, after the yes. It also says that the analysis stops, and
its report says why, if the plan cannot be carried out.

- The promise is kept in the preview's record (`handoff`), and the Analyzer
  reads that record after the runs. It does not read the switch, since the
  analysis is the last line of a batch job, where a switch set in a login
  shell does not exist.
- Before the Analyzer plans, the plan shown, its review, its first version
  when it was rewritten and the model's exchanges are copied into the study's
  `04_analysis`, byte for byte. The planning step keeps a plan and a review
  already on disk, so it makes no model call. Its gate holds the plan to the
  hash its review recorded.
- The plan is used only when the design approved is the design the plan was
  written for, the plan file still hashes to what the page's record holds, its
  review still passes, and the study's `04_analysis` holds no other plan.
- When any of these fails the handoff is blocked, and the Analyzer does not
  write another plan in its place. `04_analysis/plan_handoff.json` records the
  stop with its reason. The planning step is not called, the gate that every
  look-up passes refuses, and `REPORT.md` is a page that says the handoff is
  blocked, why, and what is kept. Nothing is deleted. The runs, their numbers
  and the plan shown stay as they were.
- A resume checks a blocked handoff again, and the analysis goes ahead only if
  the plan shown can then be handed over unchanged. To analyse such a study
  with another plan is a new decision, which is not built.
- A study whose page made no promise is as before. Its Analyzer writes its
  own plan after the runs.
- The plan's own check is made again against the runs as they finished, and
  what it finds is recorded and printed in the report. It changes nothing. An
  analysis whose year a run did not produce is closed as failed by the
  Analyzer's close of the planned phase, as before.
- The report says, above its planned analyses, that the plan was shown on the
  design page, approved with the yes and carried out as it stood.

## A yes is of the analysis plan too

Since stage 4 (2026-11-12), under the same two switches. Nothing changes for
a study planned after its runs.

- **The record of the yes names the plan.** `01_inputs/design_approval.json`
  holds the design's fingerprint, as before, and beside it the hash of the
  plan the page showed (`analysis_plan_sha256`), the plan's file and, in
  words, what the yes covers. The plan is no part of the design's
  fingerprint. The two are kept apart, so a difference can be named.
- **The Analyzer is handed the plan the yes names.** A yes on record that
  names no plan, or another plan, blocks the handoff.
- **A design whose approved plan changed is not run.** Before anything goes
  to the Experiment Manager, the plan on disk must still hash to what the yes
  names (`design_review.approval_problem`). No computing is spent on a study
  whose approved analysis can no longer be carried out as approved.
- **No plan, no yes.** A design whose plan did not pass its check and its
  review, or could not be written, cannot be approved, at the terminal or
  unattended. The page says so, and the prompt takes a change or stops. A
  change plans the design again. A study with no analysis plan at all (no
  results store) is not held to one.
- **A record made before stage 4** promised its plan under a yes that named
  none. Such a study is handed its plan as it was promised then, and its
  report says the yes was not tied to the plan.

## The status of each analysis, before the yes

`analysis_preview.analysis_support`. Code gives each analysis of a plan that
is shown one of three statuses, from the plan and its check. No model writes
it. The page's table has the status in its last column, and a line below the
table gives the reason for each one that is not supported.

- **blocked**: code's check of the plan names a problem of this analysis.
  Such a plan did not pass.
- **conditional**: it can be carried out, under a condition a reader must
  know before the yes. One condition is known to code today. Every output of
  the analysis is a map or a daily series, whose records keep no value, so
  the analysis gives figures and no number the report may state.
- **supported**: neither.

Not built: a status for a calculation the outputs only approximate, such as a
monthly water budget from stored water kept as a monthly mean. The plan has
no place to say what a calculation needs of a field.

## The computing a yes allows

`src/core/approved_spending.py`, for every whole-domain study, with no switch.

- The design page gives the model server's estimate of the cost, in
  node-hours, and says that a yes allows that figure and no more.
- The approval records the figure under `authorized`, apart from the design's
  fingerprint and from the plan's hash. A cost estimate identifies neither.
- The Experiment Manager checks it after the model server has checked the
  design again and before anything is built or submitted. A design the server
  now estimates above the figure is refused, with both figures, and nothing
  is submitted. A person then approves it again on a page that gives the
  present estimate.
- A study with no approval on record, an approval made before this rule, or a
  design with no estimate is held to no limit.

Only computing is limited this way. The paid model calls of the analysis are
bounded by the number of rounds the Analyzer may make, not by a figure a
person approves.

## Every caveat is on the page

Up to eight caveats are printed, each up to 400 characters, and any beyond
that are counted with where to find them. What Reception lists as data gaps
is labelled "noted by Reception" on a whole-domain page, since such a study
fetches no data.

## A change with no completed run

The server's change list marks each available change `demonstrated: true` or
`false` (`domain_changes.py`). A change that has never completed a run in
this framework may be designed. The server's check lists it as unusual, and
the design page says "NOT DEMONSTRATED" under the run, by code. The words
printed there are the server's and say what is true of the change, nothing
else; what a designer does about it is a rule of the server's list, which no
page prints.

## Known limits

- What a model does with a prompt is shown only by live conversations. The
  prompt's tests hold its contract with the code, not its behaviour.
- The other changes (fire off, CO2 level, deposition year, nutrient supply)
  have no year setting: they are in effect from the run's first year.
- A year setting that begins in January, or ends in December, next to a year
  left out has not been run: the first step after a year's restart carries
  the atmosphere's last output of the job before.
