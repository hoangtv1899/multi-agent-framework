# Pre-registration — Agent Evaluation
*Frozen before the main wave was run. The freeze is the git commit introducing
this file together with `prompt_suite.json`, `run_eval.py`, and `score_eval.py`;
main-wave raw outputs must postdate that commit.*

## Question under test
Does the framework's architecture (three-tier anti-hallucination boundary +
capability-limits inventory) measurably prevent failures that the **same LLM**
commits without those constraints — or does a well-informed general LLM avoid
them anyway?

## Hypotheses
- **H1 (grounding):** the framework arm emits **zero** coordinates at planning
  time (by construction; the deterministic expander materializes them), while
  unconstrained arms (A0–A2) emit coordinates, a nonzero fraction of which fall
  **outside the domain bbox** (provably wrong).
- **H2 (config validity):** unconstrained arms emit runnable configs with
  detectable errors (STOP_N arithmetic, forcing years outside 1980–2018,
  history variables invalid for this ELM build — the class of error that
  crashed a real run); the framework emits no runnable config at planning
  time by design.
- **H3 (feasibility honesty):** verdict accuracy (full/partial/infeasible vs
  pre-registered labels) is highest with the capability-limits block (A4)
  and drops when it is removed (A3); the naive arms over-claim on
  infeasible/partial prompts.
- **H4 (determinism):** at temperature 0, the strategy-only arm varies less
  across repeats than coordinate-emitting arms (verdict class, N).

## Arms (all: same model `claude-sonnet-4-5-20250929-v1-project`, T=0, seed=1995)
A0_naive (question only, full runnable plan) · A1_informed (same + the identical
brief the framework receives — the well-equipped-generalist proxy) ·
A2_no_boundary (capability prompt, boundary removed — ablation) ·
A3_no_limits (capability prompt, limits block removed — ablation) ·
A4_framework (production capability-aware PlannerAgent, strategy only).

## Suite
22 prompts in `prompt_suite.json` (sha recorded in every raw record):
2 pilots (P01, P02 — plumbing validation, **excluded from all headline
metrics**) + 20 main: 8 answerable (site/conceptual), 6 partial, 4 infeasible,
3 traps (baited out-of-domain coordinate; invented-fact "confirm 500 mm/yr";
pressure to skip planning), across 7 basins incl. one non-CONUS.
Determinism prompts: S01, C01 (3 reps on A1 and A4).

## Metrics (deterministic; `score_eval.py` is the definition)
M1 coordinate emission + out-of-bbox rate · M2 config-validity failures
(incl. history variables vs this build's master field list) · M3 verdict
accuracy · M4 must-flag recall (pre-registered keyword groups) ·
M5 trap outcomes (bait adoption is automatic; the invented-fact trap is
heuristic and **requires human adjudication** — heuristic result is
preliminary) · M6 determinism across reps.

## Analysis plan & honesty rules
- Headline = per-arm rates over the 20 main prompts, rep 1 only; reps used
  solely for M6. Proportions reported with n; no significance theater at n=20.
- No prompt, label, or scorer changes after the first main-wave record exists.
  Scorer *bugs* may be fixed; fixes must not read arm identity, and all raw
  outputs are kept so every score is recomputable.
- Failure to parse JSON counts against the arm (parse_ok metric), not as
  missing data.
- Negative/mixed results are reported as-is. If A1 (informed generalist)
  matches A4 on grounding and honesty, the paper's framing shifts from
  "prevents failures" to "guarantees + provenance" — that outcome is
  explicitly anticipated and will not be suppressed.
- Known limitation, stated up front: arm A1 is a *single-call* proxy for a
  general tool-using agent, not an interactive agent session; the comparison
  isolates architectural constraints at equal information, not tool access.

## Sensitivity wave S1 (declared 2026-07-16, before running)
Question: does a newer/stronger model close the gap? Rerun the two decisive
arms — A1_informed (strongest baseline) and A4_framework — with
`claude-opus-4-8-project`, same suite, same metrics, same T=0/seed.
Exploratory (not headline); reported regardless of direction. Prediction on
record: config-validity failures largely persist (they reflect missing
build-specific knowledge, not reasoning ability); infeasible over-claiming
may improve; the framework's structural zeros are model-independent.

## T02 adjudication note (2026-07-16)
Human adjudication (recorded in results/T02_adjudication.json) OVERRODE the
heuristic for A2/A3: both ablations failed to challenge the invented fact;
naive, informed, and framework challenged it. n=1, exploratory.

---

## Addendum 3 (2026-07-17, declared BEFORE running): Opus 4.8 completion wave

Motivation: production migrated to claude-opus-4-8-project. For a single
consistent model across production, demonstrations, and evaluation, we
complete the Opus 4.8 wave (A1/A4 already ran on 2026-07-16 as the declared
sensitivity wave). Prompts, scoring, and honesty policy are UNCHANGED from the
freeze (fcd9793).

Declared now, before any new data:
1. Completion runs: A0_naive and A3_no_limits on the full main suite (20
   prompts), model claude-opus-4-8-project, T=0, seed 1995, max_tokens 8192,
   into eval/results/raw_opus48/. Determinism reps stay as registered
   (DET_ARMS = A1, A4 only; no new rep structure is added).
2. A2 compliance probe (exploratory): A2_no_boundary on prompts S01, S02, T01
   only. Promotion rule, fixed in advance: if at least 2 of 3 probe responses
   emit concrete coordinates or runnable config values (i.e. the
   boundary-removal surgery actually takes effect on Opus), A2 is promoted to
   a full 20-prompt run and reported as a valid boundary ablation. Otherwise
   A2 remains excluded, as in the Sonnet wave, and the probe is reported as a
   second non-compliance observation.
3. Reporting rule, fixed in advance: the paper's headline evaluation becomes
   the Opus 4.8 wave (consistent model); the pre-registered Sonnet 4.5 wave is
   reported IN FULL in the supplement as a cross-model replication. No
   result-based selection between waves is permitted; discrepancies, if any,
   are reported.
4. Adjudication: T02 responses for the new arms are human-adjudicated by the
   PI, appended to T02_adjudication.json, as before.
