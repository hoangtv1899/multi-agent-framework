# What the front door runs, and what counts as a yes

Status on 2026-11-10. Written by hand; the generated documents
(`ARCHITECTURE.md`, `ANALYSIS_METHODS.md`) are not edited to hold it.

These rules are in code. A prompt asks a language model to follow them; it
cannot make it, so none of them rests on a prompt alone.

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

Reception's prompt gains a block that says the same
(`prompts/reception_whole_domain_only.txt`), left out when the switch is set.
The block only stops Reception from offering a way forward that the code
would refuse.

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

An unattended run (no `--ask`) has no prompt: it continues on a feasible
design of a supported kind and records that it did. Launching it is the
person's decision.

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

## A change with no completed run

The server's change list marks each available change `demonstrated: true` or
`false` (`domain_changes.py`). A change that has never completed a run in
this framework may be designed. The server's check lists it as unusual, and
the design page says "NOT DEMONSTRATED" under the run, by code.

## Known limits

- Reception may still spend a tool call on a place it will not study.
- The weather-scaling change has month and region selectors and no year
  selector: a design for "the summers of two named years" scales the summers
  of every simulated year. The page says so in a caveat written by the
  Planner; no code checks that it does.
- An unattended run is not asked for a yes.
