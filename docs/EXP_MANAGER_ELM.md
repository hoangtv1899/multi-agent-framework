# The Experiment Manager, once ELM lives behind the MCP

**Status:** decisions recorded 2026-08-10; the ELM side implemented and cleaned
up the same day (§8). `ELMExpManager` still exists and the Analyzer half is
untouched — see §8, "Still open".

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

**Confirmed 2026-08-10:** the framework's four boundary files —
`reception.json`, `strategy.json`, `experiment.json`, `analysis.json`
(`docs/RUN_LAYOUT.md`) — are the framework's surface and stay framework-owned.
What moves is ownership of the *simulation subtree*: `01_inputs/`,
`warmstart/`, `02_setup_plots/`, and the case directories under `$PSCRATCH`.

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

Five compute stages become one call and a poll.

> **The ledger paragraph that stood here was wrong, and §8 replaces it.** It
> argued that with one job, the ledger and resume machinery had nothing left to
> guard, and reasoned from `ELM_MCP_PLAN.md` §6 — where the study job finalises
> *itself*, so a resume finds every stage already done. Jobs A and B removed
> that premise: the framework submits and **exits**, and job B is a different
> process on a different node hours later. `run_state.json` is the only thing
> that crosses that gap. Keep it.

### The agreed call sequence — settled 2026-08-10

```
1  framework   sample columns          bands, spread, pin at stations
2  elm_mcp     build inputs            warm start, donor soil, surfaces,
                                       domains, case_inputs.json — ONE call
3  elm_mcp     build cases             the CIME compile
4  elm_mcp     run
5  framework   package                 from elm_mcp's extract
```

`get_column_metadata` alongside, whenever the framework needs the final
columns — the design figure, area weights. `compare` is MCP-side and is
deferred until the experiment manager is done.

**`conus_restart` is passed; the CONUS surfdata is NOT.** The restart names its
own surfdata in its `surface_dataset` attribute and the code then asserts the
gridcell indices fall inside that mesh ("restart and surfdata are different
bands"). Passing the two independently would let a caller pair a restart with
the wrong surfdata — a mismatch the current design makes structurally
impossible. The warm-start directory is not passed either; the MCP owns the
layout of what it writes (§2.3).

**Why `package` is the framework's and `extract` is not.** Extract knows what
ELM's variables mean, so it is model knowledge. Packaging is rows → the bundle
the Analyzer reads, and that shape is model-independent — putting it in the
framework means one packaging format rather than one per model. Comparison is
worth duplicating per server (decided: cross-model duplication is fine);
packaging is not.

### A naming hazard, recorded because it has already caused one misreading

`case_inputs.json` is a FILE — the columns and their settings, written by
step 2. `build_elm_cases` is an ACTION — the CIME compile, step 3, which reads
that file. "build case inputs" and "build cases" are adjacent steps whose names
differ by one letter and mean entirely different things: one writes a settings
file in seconds, the other compiles Fortran for ten minutes. If anything here
is ever renamed, that pair is the one worth doing.

### The duplication to remove first

`elm_exp_manager.py` calls `inputs.warm_start` (line ~204) and then
`inputs.build_case_inputs` separately, by direct import. The MCP tool
`build_elm_inputs_from_location` calls `warm_start`, `attach_donor_soil`,
`to_run_plan`, `build_case_inputs`, `serialise_case_inputs`, `write_case_inputs`
— the same sequence, in one call.

So the same six steps have two callers today. The local by-import path is
**deleted**, not kept as a fallback: ELM stops being buildable without the
server. Decided 2026-08-10.

### Phase 1e was the wrong shape — found 2026-08-10

The pending task read *"`_build_case_inputs` calls the MCP tool, delete the
local path."* That cannot be done as written, and the reason is structural
rather than incidental.

The manager does not split those six steps where the tool does. It splits them
across **two** stages:

* `_refine_columns` — warm start + donor soil, and the base enforces that it
  runs **before `columns.json` is written**, because the snap moves the columns
  and a file written earlier describes a run that will not happen. (The base
  states this as one of two structural ordering constraints; every column of the
  2026-07-28 run was 200-700 m from where `columns.json` claimed.)
* `_build_case_inputs` — surfaces, `runtime_config`, `case_inputs.json`.

`build_elm_inputs_from_location` does all six in one call. So:

* routing **`_build_case_inputs`** through the tool re-runs the warm start,
  which has already happened in `_refine_columns`; and
* routing **`_refine_columns`** through the tool was impossible, because the
  tool read `columns.json` and at that moment the file does not exist yet.

**Fix, done 2026-08-10:** the tool now accepts `columns` as data
(`build_elm_inputs_from_location(..., columns=[...])`), the file form kept for
an agent driving the server from a run directory. Passing them as data removes
the ordering problem and the temporary file with it — which is what decision
2.2 implies anyway: a temporary input that never has to be written beats one
written and then ignored.

**Consequence for sequencing:** Phase 1e is not a separate step. The manager
change is *one* call at `_refine_columns` time that returns snapped columns and
a written `case_inputs.json`, after which `_build_case_inputs` has nothing left
to do but read what the MCP produced. That is the same edit as the dissolution
in section 4, so it folds into it rather than being done twice.

Verified both forms: with columns passed as data and **no `columns.json` on
disk at all**, two perturbed columns snap to 39.6292/-75.6875 at 20.5 m and
40.1292/-75.4875 at 32.1 m — identical to what the file form produces, and the
file form still works.

---

## 5. Backwards dependencies — where they stand

These are the concrete measure of whether the boundary is real. Four of the
seven are closed.

| what | direction | state |
|---|---|---|
| `tools/run_study.sh` | MCP tool `run_elm_study` sbatched a **framework** script | **gone** — tool and script both deleted, e7ecc1b |
| `workflow.py --finalize` | that script called back into the framework | no longer called by anything on the server side. The *entry point* stays: it adopts a landed job from disk without needing SLURM or a live MCP client, which `--resume` cannot do |
| `tools/submit_cases.sh` | MCP tool `submit_elm_ensemble` sbatched it; `_run_batch` also ran it | **server side gone** — both callers deleted. The file stays in `tools/` as the legacy `run_watershed.sh` CLI's plumbing, which nothing in the framework flow touches |
| `core.exp_manager_base` | `check_elm_job` imported it for `_slurm_state` | **gone** — copied across the boundary. Thirty lines about `squeue` is not framework knowledge |
| `agents.analyzer` | `elm_exp_manager` imported `Analyzer` | **gone** — the import was unused |
| `core.limitations` | `elm_exp_manager._extract` and `scripts/analyze_run.py` import it | open — Analyzer redesign (§3) |
| `core.exp_manager_base` | `elm_exp_manager` inherits `ExperimentManagerBase` | open — dissolves with the class (§8) |
| `core.model_agent_base` | `elm_input_agent` inherits `ModelAgentBase` | open |
| `agents.analysis`, `core.figure_registry` | `scripts/analyze_agentic.py` imports both | open — Analyzer redesign (§3) |

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
## 8. The cleanup of 2026-08-10, and what is left

### What the manager became

`ELMExpManager` is 819 lines, down from 1035, and **every ELM computation in it
is now an MCP call**. The reduction is not refactoring: `_refine_columns` has
required the `elm` client since 8f32a2d, so everything guarded by
`client is None` was unreachable from the first stage onward and only *looked*
like a fallback. Deleted: `_run_batch`, `_submit_via_mcp`, `_build_cases_via_mcp`,
`_poll_via_mcp`, the `ELMExperimentBuilder` handle, the serial-`srun` path, and
the local half of `_poll`.

What is left is the four things that are not ELM knowledge: **when** to call the
server, **where** the run directory is, **what** the derived fields mean
(`FIELD_SEMANTICS`), and arranging the framework's own follow-up job.

### The bug the collapse found

`_poll` dispatched on `record['stage']`. A polled job A came back down the
`build_cases` branch — which hands back case directories and lets `execute_plan`
walk into `_run`, **which submitted the ensemble job A had already run**. Every
column would have run a second time, over the top of the first pass's history
files. Nothing had hit it only because the analysis is deferred, so no resume
had reached that line since the switch to A+B.

`_run` now collects from disk and never submits anything; `_poll` has one shape,
because there is one job.

### The ledger question — settled: keep it

Raised as "how much of the resume machinery survives one-job studies", on the
theory that a study which is one `sbatch` does not need a stage ledger.

**It needs it more, not less.** The framework submits A and B and then *exits* —
the Python process is gone. Job B starts hours later, on a different node, in a
different process, and `run_state.json` is the only thing carrying the job id
and the record of which stages finished across that gap. Without it B has
nothing to resume from. The ledger is a process-boundary carrier now, not a
convenience for an interrupted session.

Two things did change, and are worth writing down so nothing is built on the old
shape:

* **`build_cases` is the only stage that ever returns `Pending`.** `run` no
  longer participates in the pending protocol at all. `_advance`'s three-path
  logic stays — it is the base's, and PFLOTRAN uses it — but on the ELM side it
  is exercised once per study.
* **The base's comment "PHASE 1 WRITES IT AND NOTHING READS IT" is stale** and
  has been corrected. It is read, by every job B.

### Still open

* `ELMExpManager` **still exists** and still inherits `ExperimentManagerBase`.
  The class dissolving is a separate move and belongs with §12's PFLOTRAN work,
  because the base is what the two backends share.
* `_extract` is still split — `ELMResultsAnalyzer` computes MCP-side, the
  orchestration is here. That is the Analyzer redesign (§3), not this.
* **Unproven:** no study has been driven end to end through the collapsed
  manager. The build+run half is verified (job 773089); `_run`-as-collect and
  the setup figure in job A are verified by inspection only.

---

