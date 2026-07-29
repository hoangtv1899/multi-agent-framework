"""
The Analyzer, one module per step
src/agents/analysis/

    step0_context    3 boundary files  -> {plan, data, caveats}
    step1_validate   data + obs        -> comparison verdicts, new caveats
    step2_derive     data + verdicts   -> correlations, aggregates, soil
    step3_figures    derived + verdicts-> figures
    step4_interpret  everything        -> grounded prose
    step5_report     everything        -> analysis.json

The order is the dataflow, and the numbers are in the filenames because the
order is the thing that was confusing when these were siblings with no
apparent sequence.

Validation comes before derivation even though neither consumes the other.
Both need only the context, so the order is a choice — and validation is the
one that can RAISE CAVEATS. Deriving first would state correlations without
knowing whether the gauges say the model has any purchase on reality.

Two rules hold across every step:

  A step takes the context plus what earlier steps returned, and RETURNS data.
  No step opens a file for itself — step 0 is the only reader. That is what
  lets any step be run alone, against an archived run, without the rest.

  Numbers come only from `data`. `plan` supplies the question, `caveats` can
  veto a claim, and neither is ever a source of results.

The Analyzer that sequences these does nothing else: it decides order, not
content.
"""
