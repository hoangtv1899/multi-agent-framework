# The Analyzer's report: recipe, preview renderer, editorial revisions and verification coverage

Status on 2026-11-10. This note is written by hand; the generated documents
(`ARCHITECTURE.md`, `ANALYSIS_METHODS.md`) are not edited to hold it.

## What a live run writes

A live Analyzer run writes the report it has written since the analysis suite
was accepted: `04_analysis/REPORT.md`, one page, rendered by
`src/agents/analysis/step4_report.py` from `04_analysis/analysis.json`.
Nothing under "Preview only" below changes that page.

## Live since 2026-11-10: what a first run of the whole path changed

The first live studies through Reception, the Planner and the Analyzer
together, with and without a new simulation, led to these changes:

- **A claim that fails the evidence trace is struck alone.** Step 3's audit
  now applies the trace as its last rule (`rule_trace`,
  `claim_trace.claim_tracer`): the failing claim is struck with the trace's
  own reasons and the claims that pass are reported. Before, one such claim
  withheld the whole report at step 4. The gate at step 4 is unchanged, so no
  claim reaches a report without passing it.
- **The printed summary holds to the kept claims.** When the audit struck a
  claim, a summary sentence that gives a number no kept claim's record gives
  is left out of the page; the summary as written stays in `analysis.json`.
- **A new run gets its results store.** The Experiment Manager builds it in
  the study tail (`_build_run_suites`), in `03_results/suite/<run>`, where
  the Analyzer reads a run's maps, series and dates.
- **A control may name its own normal years** in the registry
  (`normal_years`). They are shown on Reception's card and on the design
  page, the Planner uses them unless the person names others, and a study
  whose Planner names none falls back to them.
- **A study whose analysis failed can be resumed for the analysis alone**
  (`core/resumable.py`), and a plan amendment whose settings name no method
  is refused with a reason.

- **The page says what code did not check** (2026-11-10,
  `step4_report.verification_limits`). A whole-domain report now ends with a
  section "What code checked, and what it did not": the numbers and the
  claim trace on one side; ranking words (with the numbers of the kept claims
  that use one), figures and plotted series, and interpretation on the
  other. Until then only the preview pages said so.

## Live: the recipe "seasonal evolution and spatial context"

`src/agents/analysis/analysis_recipes.py`. A recipe is guidance for the
analysis plan. It says which of the catalog's existing methods a kind of
question needs. It names no watershed, no value and no result to expect, and
it defines no numerical method.

- Which parts apply is read from the question's own words, by code
  (`applies`). A season or a month calls for the month-by-month part. A word
  about place, or about two things going together, calls for the spatial part.
  One question therefore gets the same guidance whatever model plans it.
- The plan step reads the guidance before its task (`plan_lines`). Step 3
  reads a matching block for the parts the plan carried out (`report_lines`).
- The plan check does not enforce a recipe, and the module still marks it as
  a draft.

## Preview only: the two-view report

`src/agents/analysis/report_views.py` renders a finished study a second way,
from the same `analysis.json` and the same evidence records:

- `REPORT.md`, the scientific report: the answer, the limitations beside it,
  the verified claims as evidence, the figures, what remains uncertain and a
  verification section;
- `AUDIT.md`, the audit appendix: every check, record and decision, with the
  whole page a live run writes.

**It is preview only.** No step of the pipeline calls it, and a live run
still produces the existing report format. It is called by hand on a finished
study (`write_views`). It calls no language model and changes no value, check
or decision; it chooses where each recorded thing is shown.

## Preview only: editorial revisions

The synthesis of a preview page may be an editorial revision: sentences
written after the run by an editor, not by the Analyzer, from the claims the
audit kept. The synthesis is data beside the study (an `editorial.json`
file), never code, and the page says whose words it is.

- Each sentence names the kept claims it rests on and is read against their
  records as a claim is (`sentence_problems`): its numbers, years, periods and
  regions are theirs, and it uses no causal word.
- It names no field those records do not read, and it uses no word of a claim
  the audit withheld. The synthesis is therefore no way back in for a
  withheld claim.
- Each competing explanation the plan named gets one status: consistent, not
  supported, not distinguished or not assessed. "Not supported" means the
  evidence gives no support at the scale it was read; it does not mean ruled
  out.
- A kept claim that a later reading finds wrong can carry a correction
  (`amendment_problems`). The claim stays on the page as the audit kept it,
  marked, with the corrected wording and its reason under it. A record the
  correction adds must have been made again by a numbers check. The run's own
  record is never changed.
- A later revision lists in the appendix what changed since the earlier one.
- A synthesis or a correction with any problem is not rendered at all.

## Verification coverage

What code checks, and what it does not:

| Statement | Checked by code |
|---|---|
| The numbers of the regional table | Yes. Made again from the model output by an independent reading (`numbers_check.py`). |
| The numbers of every lookup a claim cites | Yes. The same, by `suite_check.py`. |
| Each claim against the records it cites | Yes. The claim trace (`claim_trace.py`), by the audit and again at the report's gate. |
| A lookup that a figure draws and no claim cites | No, unless a review check is run on it. |
| A plotted series | No. A series leaves only a fingerprint in the log. |
| A ranking ("most", "least", "largest", "every band") against records the claim leaves out | No. |
| Whether an interpretation follows from its claims | No. A person reviews it. |

A live report labels this coverage in its last section before Provenance
(above). The preview page labels it in more detail. Its verification section has a
"Checked by code" list and a "Not checked by code" list (`coverage_lines`),
and a note under each figure says what the figure draws beyond the claims'
records (`figure_check_line`). A review check is the same independent check
run later, read only, on the lookups the figures draw (`suite_check.check`
with `write=False`).

## Backlog, not built

- The numbers check covers what the figures draw, a plotted series included.
- A ranking is held to its whole comparison set.

Reviewing an interpretation stays manual. No general reasoning checker is
planned.

## Tests

`tests/test_analysis_recipes.py`, `tests/test_report_views.py`,
`tests/test_figure_bars.py` and `tests/test_audit_trace_rule.py`. They are
synthetic: every number in them is made
up. A test of the renderer on real studies exists, reads local data and is
kept outside this repository, since it quotes the studies' own reports.
