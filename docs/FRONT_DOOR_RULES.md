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

`src/core/analysis_years.py`. A run starts from the control's saved state, so
the years it simulates start at the control's start year. The years the
numbers are reported for are the years the person asked about
(`run_settings.requested_period`) and the normal years they are compared
with, or the plan's own `analysis.years`. The design page, the regional table
and the study record read this one function.

- The page's When line labels each part: the years asked about, the normal
  years chosen by the design, and years simulated and not reported, with the
  rule that requires them. A widened span is never called the person's.
- The server's rule that a run's years start at the saved state applies only
  when a run executes. The control read alone may cover any consecutive
  years inside its output.

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
  simulated, so the start year a run needs is not reported as a year asked
  about after a change at the design page.
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
