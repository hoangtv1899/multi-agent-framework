"""
The Analyzer, one module per step
src/agents/analysis/

    step0_context      3 boundary files  -> {plan, data, caveats}
    step1_compare      data + obs        -> comparison record, new caveats
    step2_investigate  ctx + step 1      -> figures an LLM chose, pandas computed
    step3_interpret    everything        -> claims, audited; may send 2 back once
    step4_report       everything        -> analysis.json

The order is the dataflow, and the numbers are in the filenames because the
order is the thing that was confusing when these were siblings with no
apparent sequence.

Comparison comes FIRST because it is the step that can RAISE CAVEATS, and the
caveats bind everything after it. Investigating first would plan figures — and
spend an LLM call on them — without knowing whether the water table is
simulated at all in this run, or whether any station exists to compare against.

Step 1 is COMPARISON, not validation. Validation would mean measuring the
model against measurements of the same quantity in the same place, and almost
none of that holds: most stations of every network lie outside the watershed
that was simulated, a gauge integrates a routed catchment while a column is
1 m2 of unrouted local generation, and the water-table reference is itself a
model. Comparing fields that overlap partially and on different terms is worth
doing; calling it validation would claim more than the data supports.

STEP 1 NO LONGER DOES THE COMPARING (2026-08-12). Four modules here knew
H2OSNO from ZWT, what a snow pillow is sited for, and how deep ELM's soil
column goes — none of it model-agnostic, in the one box that has to be, since
this same Analyzer reads PFLOTRAN runs. That knowledge moved to the ELM
server's `compare` package. What step 1 kept is the half that is not about any
model: turning measurements into the CAVEATS that bind what may be claimed
from them.

Two rules hold across every step:

  A step takes the context plus what earlier steps returned, and RETURNS data.
  No step opens a file for itself — step 0 is the only reader. That is what
  lets any step be run alone, against an archived run, without the rest.

  Numbers come only from `data`. `plan` supplies the question, `caveats` can
  veto a claim, and neither is ever a source of results.

The Analyzer that sequences these does nothing else: it decides order, not
content.
"""
