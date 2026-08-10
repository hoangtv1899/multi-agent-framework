# The Experiment Manager, once ELM lives behind the MCP

**Status:** decisions recorded 2026-08-10. Implementation pending.

**This is not a second boundary document.** `docs/ELM_MCP_PLAN.md` §1–8 is the
boundary and stays the authority; that design was agreed 2026-08-07 and a
conversation on 2026-08-10 re-derived it independently, which is the strongest
evidence it is right. Read that first.

This document covers the half that plan does not: **what happens to the
experiment manager itself** — `ExperimentManagerBase` and its per-model
subclasses — and the four decisions of 2026-08-10, two of which amend the plan.

---

## 1. The rule, restated for this side

`ELM_MCP_PLAN.md` §1 gives the rule as *anything that requires knowing what ELM
is lives in `elm-mcp`*. The experiment manager needs the same rule stated from
its own end, because a manager is not knowledge — it is sequencing:

> The framework decides **what to ask for** and **what the answer means**.
> The server decides **everything about how the model produces it**.

A stage that only sequences model-specific work has nothing left to sequence
once the model-specific work is one call. That is why the manager dissolves
rather than shrinking.

---

## 2. The four decisions of 2026-08-10

### 2.1 PFLOTRAN gets the same treatment — **new**

`ELM_MCP_PLAN.md` is silent on PFLOTRAN. It is not silent by accident: under
that plan `ExperimentManagerBase` survives to serve
`pflotran_exp_manager` and `lambda_pflotran_exp_manager`, and the base's
`_build_case_inputs` → `List[Dict]` contract survives with it.

**Decided:** PFLOTRAN follows ELM behind its own MCP boundary. So the base class
is not a permanent fixture serving one remaining backend — it is scaffolding on
a schedule.

*Consequence, and the reason this decision matters more than it looks:* it
settles a question that was live for a week. When ELM's `_build_case_inputs`
starts returning something the base did not previously expect, **do not** amend
the base contract to accommodate it, and **do not** make PFLOTRAN absorb churn
to keep the shapes aligned. Both backends are leaving. Anything spent making the
shared contract elegant is spent on a class scheduled for deletion.

### 2.2 The MCP's `columns.json` is the final one — **amends the plan**

`ELM_MCP_PLAN.md` §3 has the framework write `01_inputs/columns.json` and says
"the framework's own `columns.json` must reflect the moved version" — the MCP
snaps the coordinates and writes them back into the caller's file.

**Decided:** that back-write goes away. The sampler's `columns.json` is a
**temporary input to the MCP**, not a durable artifact. The final column
metadata is the MCP's, fetched by an explicit tool call
(`get_column_metadata` or similar), not by two writers sharing one path.

*Why this is worth the extra tool:* the warm start **moves the columns** — it
snaps each to its CONUS donor gridcell and adopts that cell's soil, so anything
persisted before it describes a run that will not happen (`ELM_MCP_PLAN.md` §3,
"why the warm start is the hinge"). A file written by the framework and then
silently rewritten by the server has two producers and no owner. Making the
handoff a return value gives it one owner and makes the staleness impossible
rather than merely unlikely.

### 2.3 The MCP owns the run-directory layout — **amends the plan**

`src/core/run_layout.py` is framework-side today and answers "where is X" for
every tool. The MCP writes into `01_inputs/`, `warmstart/` and `02_setup_plots/`
regardless.

**Decided:** the MCP owns the layout of everything it writes.

**Interpretation, to be confirmed:** the framework's four boundary files —
`reception.json`, `strategy.json`, `experiment.json`, `analysis.json`
(`docs/RUN_LAYOUT.md`) — are the framework's surface and stay framework-owned.
What moves is ownership of the *simulation subtree*. If the intent was that the
MCP owns the run directory including those four, say so; it is a larger change
and it makes the framework a client of its own run directory.

`run_layout.py` also carries the legacy-path fallbacks that let a tool read runs
from any era. Those are about **reading old runs**, not about where new work
writes, so they stay in the framework whatever else moves.

### 2.4 The MCP owns setup and comparison figures — **confirms the plan**

Consistent with `ELM_MCP_PLAN.md` §7 (`plots.py`, "every ELM figure"). The rule
that resolves future cases: **a figure follows the knowledge it needs.**
`column_surfaces.png` reads each case's generated FSURDAT — model knowledge, so
MCP. `sampling_design.png` reads columns and bands — design knowledge, so
framework (`tools/`, alongside `figstyle.py`).

Comparison figures are stubbed for now; they are part of the Analyzer redesign.

---

## 3. Deferred to the Analyzer redesign

Recorded here so they are not rediscovered as surprises.

**`src/core/limitations.py`.** A catalog of caveats that travel with every
result, split into `structural` (never fixable within 1-D columns) and
`configuration` (fixable by rerunning). It exists to **bind the Analyzer LLM**:
`analyzer_agent.py`'s prompt states the catalogue binds it, and
`step0_context.py` promotes structural entries to *blocking* — "local runoff
generation on a 1 m² column, not routed discharge" does not qualify a gauge
comparison, it forbids the naive one.

Its structural entries are all statements about ELM's physics, so under the rule
they are MCP content sitting in a framework file — and it is imported from both
sides today (`analyzer_agent`, `step0_context`, `exp_manager_base`,
`interpret_run` on one side; `elm_results_analyzer`, `analyze_run`,
`analyze_agentic` on the other), which is a live server → framework dependency.

**Not resolved here.** The Analyzer is being redesigned; this file is an input
to that design, not a loose end to tidy first.

**The observation contract.** `compare(observations)` makes the observation
shape a public API the framework must produce. Deliberately not designed yet: a
signature written before it has a caller will be wrong by the time it has one.

---

## 4. What the experiment manager becomes

Today, for ELM, with the stage ledger and resume machinery around each step:

```
_materialize / _sample_columns   framework
_build_case_inputs              framework, by direct import of the MCP's inputs.py
_build_cases                    MCP job
_run                            MCP job
_extract                        framework base, ELM reader already MCP-side
_package                        framework
Analyzer                        framework
```

After:

```
sample columns                  framework
run the study                   ONE MCP call -> job id
poll                            MCP
read the package                framework
interpret and report            framework
```

Five compute stages become one call and a poll. The ledger, `Pending`, and the
resume machinery were built to make a five-stage pipeline restartable at any
point; with one job that either finished or did not, most of that has nothing
left to guard. **Do not port it forward on the assumption it is still needed** —
`ELM_MCP_PLAN.md` §6 already establishes that the study job finalises itself, so
a resume finds every stage already marked done.

### The duplication to remove first

`elm_exp_manager.py` calls `inputs.warm_start` (line ~204) and then
`inputs.build_case_inputs` separately, by direct import. The MCP tool
`build_elm_inputs_from_location` calls `warm_start`, `attach_donor_soil`,
`to_run_plan`, `build_case_inputs`, `serialise_case_inputs`, `write_case_inputs`
— the same sequence, in one call.

So the same six steps have two callers today. The local by-import path is
**deleted**, not kept as a fallback: ELM stops being buildable without the
server. Decided 2026-08-10.

---

## 5. Backwards dependencies to straighten

These exist now and are the concrete measure of whether the boundary is real.

| what | direction today | fix |
|---|---|---|
| `tools/run_study.sh` | MCP tool `run_elm_study` sbatches a **framework** script | moves into the MCP (`ELM_MCP_PLAN.md` §7 deletes it in favour of jobs A+B) |
| `workflow.py --finalize` | that script calls back into the framework | deleted with it |
| `core.limitations` | `analyze_run.py` imports it from the framework | Analyzer redesign (§3) |
| `elm_exp_manager` | inherits `src/core/exp_manager_base`, imports `agents.analyzer` | dissolves with the class |

---

## 6. What is already true

Not everything here is prospective; some of it is where the code already lives,
and it is worth knowing which so effort goes to the real gaps.

* **ELM's history-file reader is already MCP-side** —
  `mcp/elm-mcp/src/elm_results_analyzer.py`, "single responsibility: read ELM
  NetCDF history files". Only the *orchestration* of extract is framework-side.
* **The input builders are already MCP-side** — `inputs.py`,
  `build_column_inputs.py`, `make_finidat_subset.py`.
* **The CIME build through the moved modules is verified.** Job 773080,
  2026-08-10: two columns, `COMPLETED 0:0`, 7 min 17 s, reference case compiled
  and the clone sharing its executable via `--keepexe`. Each column received its
  **own** `finidat` and its **own** surface file at its own coordinates, which is
  what proves the path resolution across the boundary rather than merely that
  something compiled.

  Two things that run did **not** cover: it built cases but did not run them, and
  n=2 exercises both the cold-compile and clone paths but says nothing about
  clone fan-out at ensemble scale.

  One latent finding: the cloned case is left `BUILD_COMPLETE=FALSE` —
  `create_clone --keepexe` does not set it and `_clone_case` does not either. It
  does not bite today because `run_simulation` resolves `EXEROOT` via `xmlquery`
  and `srun`s the executable directly, never consulting CIME. **So
  `BUILD_COMPLETE` is not a usable readiness signal for these cases**, and
  anything that later starts trusting it — a resume path, a health check, a move
  to `./case.submit` — will wrongly conclude the clone was never built.

---

## 7. How this gets proven

`ELM_MCP_PLAN.md` §9 settles this and it is not re-opened here: **every phase is
proven by a real run**, the existing suite is used only as an import check during
relocation, and a replacement suite is written from scratch at the end.

The reasoning is worth repeating because it is easy to backslide on: of the
failures of 2026-08-06, every one shipped green. `_as_extract` is the sharpest —
a test for that exact bug existed, passed, and was *structurally incapable* of
catching it, having been written against PFLOTRAN whose rows are a list, so it
could never reach the dict branch.

**Standing constraint:** ask before submitting any SLURM job. Reaffirmed
2026-08-07 and again by the explicit authorisation required for job 773080.
