# The run directory: four files, one per box

A run directory's *surface* is four JSON files, one per box of the framework.
Each is written by the stage that produced it, and nothing else writes it.

```
elm_run_YYYYMMDD_HHMMSS/
    reception.json      ← Reception    what/where/when + everything fetched
    strategy.json       ← Planner      the sampling strategy, purely LLM
    experiment.json     ← Exp Manager  what was run and what came out
    analysis.json       ← Analyzer     what it means
```

That is the contract. Anything else in the directory is working state, and
working state does not live at the top level.

## Why not literally four files

`columns.json`, `cases.json` and `run_plan.json` look like clutter but are not
aliases — they are how the Experiment Manager hands state between its own
stages and the `tools/` CLIs. `make_warmstart`, `make_finidat_subset`,
`build_column_inputs`, `plot_columns` and `build_cases` all read them, and that
is what lets a stage be re-run without re-running the pipeline. Deleting them
would trade a real capability for a tidier listing.

So they move DOWN, into `01_inputs/`, rather than away.

## The mapping

### Stays, at the surface — the four

| File | Written by | Holds |
|---|---|---|
| `reception.json` | coordinator, after Reception | route, brief, observations, grid, provenance |
| `strategy.json` | coordinator, after Planner | archetype, goals, feasibility, sampling, validation, **pinning** — the model server's block the design was made against, written by the coordinator so the sampler enforces the same rules the planner was shown (2026-08-18) |
| `experiment.json` | Exp Manager `_package()` | domain, period, strategy check, per-column results, field semantics, artifact pointers |
| `analysis.json` | Analyzer | figures drawn, validation verdicts, interpretation — TO BE ADDED |

### Moves into `01_inputs/` — working state, still read by the CLIs

| File | Read by |
|---|---|
| `columns.json` | make_warmstart, make_finidat_subset, build_column_inputs, plot_columns, validate_run, analyze_run, analyze_agentic |
| `cases.json` | build_cases, build_pflotran_cases, plot_columns, make_warmstart, validate_run, analyze_run |
| `run_plan.json` | make_warmstart, validate_run, analyze_run, analyze_pflotran_coupled |
| `assumptions.json` | build_cases, analyze_run, analyze_agentic — the honesty ledger |

### Deleted — the content already lives in one of the four

| File | Superseded by | Note |
|---|---|---|
| `reception_brief.json` | `reception.json["brief"]` | a strict subset; existed only because the tools opened it by that name |
| `plan.json` | `strategy.json` | the planner's product, written twice under two names |
| `RUN_SUMMARY.json` | `experiment.json` | counts and timings fold in |
| `LLM_ANALYSIS_INPUT.json` | `experiment.json` | this IS the package the Analyzer reads; two names for one job |
| `ANALYSIS_REPORT.json` | `analysis.json` | the written report becomes a field |

### Folds into `analysis.json` — currently scattered across `04_analysis/`

| File | Note |
|---|---|
| `validation.json` | the observation comparison and its verdicts |
| `analysis_plan.json` | which figures the Analyzer chose, and why |
| `figure_captions.json` | per-figure provenance |

`interpretation.md` stays as a file. It is the one artifact meant to be read by
a person rather than a program, and burying prose inside JSON helps nobody.

`04_analysis/hydro_summary.json` also stays: it is the raw per-column
extraction with the full variable series, and `experiment.json` deliberately
carries the summary rather than the series. Keeping it means the expensive
part of a run survives even when the packaging changes.

## Reading old runs

Every path goes through `src/core/run_layout.py`, which resolves a logical name
to wherever the file actually is — the new location first, then the legacy one.
Runs produced before this change stay readable, which matters because they are
the only record of experiments that cost hours of compute.
