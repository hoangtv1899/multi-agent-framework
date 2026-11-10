# The Analyzer's report: recipe, preview renderer, editorial revisions and verification coverage

Status on 2026-11-09. The Analyzer's scope is frozen at this state. This note
is written by hand; the generated documents (`ARCHITECTURE.md`,
`ANALYSIS_METHODS.md`) are not edited to hold it.

## What a live run writes

A live Analyzer run writes the report it has written since the analysis suite
was accepted: `04_analysis/REPORT.md`, one page, rendered by
`src/agents/analysis/step4_report.py` from `04_analysis/analysis.json`.
Nothing under "Preview only" below changes that page.

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
| Each claim against the records it cites | Yes. The claim trace (`claim_trace.py`). |
| A lookup that a figure draws and no claim cites | No, unless a review check is run on it. |
| A plotted series | No. A series leaves only a fingerprint in the log. |
| A ranking ("most", "least", "largest", "every band") against records the claim leaves out | No. |
| Whether an interpretation follows from its claims | No. A person reviews it. |

The preview page labels this coverage. Its verification section has a
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

`tests/test_analysis_recipes.py`, `tests/test_report_views.py` and
`tests/test_figure_bars.py`. They are synthetic: every number in them is made
up. A test of the renderer on real studies exists, reads local data and is
kept outside this repository, since it quotes the studies' own reports.
