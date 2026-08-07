# The ELM MCP

**Status:** design agreed 2026-08-07. Not yet built — §9 is the implementation plan.

This is a **rework**, not an increment. The previous design is archived at
`docs/archive/ELM_MCP_PLAN_ruleA_2026-08-01.md`; it is worth keeping because the
reasoning that produced its boundary is what tells us the cost of moving it, and
because its measurements and bug records still hold. Nothing in this document
depends on reading it.

---

## 1. The rule

> **Anything that requires knowing what ELM is lives in `elm-mcp`.**

One sentence, applied without exception. Everything else in this document
follows from it.

### Why the previous rule had to go

The old boundary was *"does it need the scheduler and the model binary?"* — the
server compiles and runs, the framework decides and reads.

That rule is coherent, and it is **why `ELMExpManager` existed**. Under it,
ELM-specific *deciding* — the warm start, the donor soil, the run-plan shape —
is barred from the server, so it needs a home somewhere else, and that home is a
per-model manager class. Three separate attempts to delete that class failed
because the rule kept regenerating the need for it.

| | old rule | this rule |
|---|---|---|
| boundary | needs scheduler/binary | needs ELM knowledge |
| input generation | framework | **MCP** |
| extraction | framework | **MCP** |
| observation comparison | Analyzer | **MCP** |
| ELM figures | Analyzer | **MCP** |
| `FIELD_SEMANTICS` | framework | **MCP**, returned as data |
| `ELMExpManager` | **must exist** | **deleted** |

### What it costs

The server's `_requirements()` currently advertises being short — *"no CONUS
surfdata, no ELM input-file templates."* Under this rule it needs bulk data
again, not just code and a scheduler.

That price is smaller than it looks: the server already requires the E3SM source
tree, `$PSCRATCH` and `sbatch`. It is not portable now and will not become so. A
CONUS restart does not move it into a different class of thing.

### The objection that does not survive

The old design withdrew "an agent can drive this directly" because
`build_elm_cases(columns=[...])` produced *"a cold, template-soil column at a
coordinate — runnable, and quietly worse science… worse than not offering it,
because the run looks fine."*

That is an argument about a **signature**, not a location. A tool that requires
the warm-start inputs and refuses to run cold does not have the defect. The
capability is reinstated, with that guard (§5).

---

## 2. Architecture

```
FRAMEWORK        model-agnostic, permanently
  _sample_columns    watershed → column locations   (terrain · fan_wtd · geology MCPs)
  stage sequence     run_state.json · Pending · STOP · --resume
  _package           experiment.json — the Analyzer's input contract
  Analyzer           the user's question · which comparisons are worth running
                     claims · caveats · verdict
                     fetches observations (snotel · usgs) and passes them IN
  job-B composition  sbatch --dependency=afterany

ELM MCP          everything that knows ELM
  inputs · surfaces · cases · run · results · compare · plots

ELM / CIME
```

Two properties worth stating plainly:

* The framework **already** drives three MCPs inside `_sample_columns`, so
  calling a *model* server from there is not a new pattern.
* Dependencies are **one-directional**: framework → servers. No server calls the
  framework, and no server calls another server. This is what the previous
  design violated (§6).

---

## 3. What each side owns

### Framework

`_sample_columns` (renamed from `_materialize` — named for `columns.json`, the
artifact it writes, matching the `_build_case_inputs`/`case_inputs.json`
convention). In order:

```
1  bbox from the reception brief
2  self.check(plan, config)              strategy check + corrections
3  terrain MCP get_watershed_boundary    no HUC ⇒ loud raw-bbox warning
4  expand_sampling.expand(...)           ← the column locations
     sample_elevation_grid → clip to polygon → elevation bands
     → _allocate per band → _farthest_point_select → fan_wtd enrichment
5  provenance: clipped_to_watershed, or the bbox caveat
6  write 01_inputs/columns.json          locations + grid + boundary + bands
```

Then the stage ledger, `_package`, the Analyzer's question-driven half, and the
job-B composition.

### ELM MCP

Everything downstream of a column location: the warm start, the donor soil,
surface and domain generation, `runtime_config`, CIME, the run, extraction,
observation comparison, and every ELM figure.

### Why the warm start is the hinge

`_refine_columns` runs in the **middle** of the old `_materialize`, not at the
end, because the warm start **moves the columns** — it snaps each to its CONUS
donor gridcell and adopts that cell's soil. Anything persisted before it
describes a plan the run will not follow.

That is the single fact that makes this a boundary and not a handoff: the
framework samples, the MCP moves what was sampled, and the framework's own
`columns.json` must reflect the moved version.

---

## 4. The tool surface — seven tools

| tool | its one job | returns |
|---|---|---|
| `describe_elm_capabilities` | what it can and cannot do, **+ the variable registry** | data |
| `build_elm_inputs_from_location` | locations → runnable ELM inputs | snapped columns, inputs, `case_inputs.json`, `sampling_design.png` |
| `build_elm_inputs_conceptual` | idealized spec → runnable ELM inputs | same, stamped `provenance: conceptual` |
| `run_elm_ensemble` | build + clone + run, ONE job, restartable | job id |
| `check_elm_job` | scheduler state, nothing else | `{state, active}` |
| `collect_elm_results` | read the history files | rows, units, named series |
| `compare_elm_to_observations` | observations in → skill out | metrics, caveats, figure |

### One function, one job

Each tool does exactly one thing. Decomposition happens **inside** the server as
private functions, never as more tools.

The reason is measurable. The reaction MCP has **43 tools**, with overlapping
pairs throughout:

```
create_pflotran_input · create_column_deck · configure_reaction_sandbox    3 ways to make an input
run_pflotran_simulation · submit_pflotran_ensemble                         2 ways to run
check_pflotran_job · check_simulation_status                               2 ways to check
collect_pflotran_results · extract_observations                            2 ways to read
```

and `create_pflotran_input` takes **18 parameters**, nearly all science-bearing,
most with defaults. A model driving that guesses twice — which tool, then what
values — and nothing rejects a plausible invention. The archived plan predicted
this in its own words: *"two tools that differ only in input shape is the kind of
surface that grows to 40."* It reached 43.

### Why build and run are one tool

**Nobody wants CIME cases.** The deliverable is model output; cases are an
intermediate. Splitting them cost two queue waits and produced a job id whose
product was a half-finished thing.

Restartability preserves every use the split had: `run_elm_ensemble` skips the
build when `built_cases.json` already reports success — so a wall-time kill costs
a queue wait rather than the ~8–10 min compile, and calling it on already-built
cases simply runs them. That is the recovery path used on job 770923.

`get_elm_cases` is therefore **not** a tool: `collect_elm_results` reads
`built_cases.json` from the same run directory.

### `check_elm_job` is scheduler state and nothing else

Today it does three jobs — `squeue`/`sacct`, *then* reads `built_cases.json` and
returns case directories, *or* reads `run.log` and returns a tail. With build and
run merged there is no half-finished stage needing somewhere to hand back an
answer, so it collapses to `{state, active, scheduler_answered}`.

`active: false` with `state: null` must remain impossible. A `squeue` timeout is
**no answer**, not a finished ensemble.

---

## 5. The two input builders

They fork on one question the caller can always answer — **do I have real
coordinates?** — and merge at `case_inputs.json`. Everything downstream is
identical. That is what keeps a seventh tool from becoming a forty-third.

```
REAL         _sample_columns → build_elm_inputs_from_location ─┐
                                                               ├→ case_inputs.json
CONCEPTUAL             build_elm_inputs_conceptual ────────────┘
                                                                    ↓
                                     run_elm_ensemble → check_elm_job → collect_elm_results
```

### `build_elm_inputs_from_location`

```
reads   01_inputs/columns.json          locations + grid + boundary + bands
  _warm_start(columns)                  subset the CONUS restart; snap to donor gridcell
  _attach_donor_soil(columns, map)      the donor's soil — the only soil the run has
  _generate_column_inputs(...)          domain.nc + surface.nc per column
  _to_runtime_config(...)               the 13 CIME keys per case
  _write_case_inputs(...)               01_inputs/case_inputs.json
  rewrite columns.json                  with the SNAPPED columns
  plots.sampling_design(...)            drawn where the snapped columns live
```

The run directory is the medium, as it already is for `case_inputs.json` and
`built_cases.json`. The design figure needs the full DEM grid and the boundary
rings — which `expand()` keeps but deliberately does not put in `columns.json`
(*"full DEM sample kept for the hypsometry / map plot"*) — so passing them over
stdio would mean ~14k points per call. Reading the file avoids that.

### `build_elm_inputs_conceptual`

**Not new work.** `tools/make_soil_sweep.py` already is it:

> Controlled SOIL SWEEP — emit an executable ELM plan that varies ONLY soil
> texture (a clay gradient, sand → clay) at a **SINGLE location**, so forcing,
> PFT and everything else are identical across runs. This is the
> **conceptual-archetype counterpart to the spatial expander**.

It also already answers the design question a conceptual column raises: ELM needs
DATM forcing, which is gridded by coordinate, so a column with *no* location
cannot be forced. Using one real location and varying only soil is the right
answer, and it is what this script does.

What changes is the output shape (`CONDITIONS_COUPLERS` → `case_inputs.json`)
plus four guards, drawn from what the reaction MCP gets wrong:

1. **Closed vocabulary, not free numbers.** `soil: Literal["sand","loam",…]`,
   not an arbitrary texture array. A model cannot invent an enum member — it
   errors. This collapses the guess space more than anything else.
2. **No science-bearing defaults.** Anything that changes the answer is required,
   or it is an error naming what is missing. `time_unit="d"` is how a fabricated
   run comes out looking complete.
3. **Provenance in the artifact.** `provenance: "conceptual"` rides into
   `experiment.json` and the caveats read it. This does not prevent
   hallucination; it prevents it being **invisible** — the failure that let a
   `recharge_fraction` of 2.72 through on 2026-08-06.
4. **Cold start is a cliff, not a flag.** No donor gridcell means no warm start,
   so soil moisture and carbon begin at nothing and a short run measures
   relaxation rather than soil. A blocking caveat.

Shares `_generate_column_inputs` and `_to_runtime_config`. **Must not share
`_warm_start`** — there is no donor to snap to.

#### Optional: validate against the CONUS distribution

A model can invent a number; it cannot invent a matching gridcell. Compare the
requested profile against the real CONUS distribution and report where it sits.

Three constraints:

* **Advisory, never blocking.** Idealized experiments are *supposed* to be
  unphysical sometimes.
* **Say "outside the CONUS distribution", not "unphysical."** CONUS is one
  continent; a tropical or permafrost profile is real and absent. The check
  reports a measurement; claims belong in the caveat record.
* **Precompute it.** Ship a small reference table rather than opening the dataset
  per call — in `expand_sampling`, the file *open* was the entire cost, which is
  why enrichment there is batched.

Output shape:

```json
{"col_01": {"in_distribution": false,
            "flags": [{"field": "clay_fraction[0:3]", "value": 0.71,
                       "conus_percentile": 99.8,
                       "nearest_real_example": {"lat": 34.2, "lon": -89.6}}]}}
```

Run it once over our own preset table too: if what we call `"loam"` does not sit
near CONUS loam, that is a bug in the presets.

---

## 6. The unattended flow — jobs A and B

```
framework        _sample_columns                     → 01_inputs/columns.json
framework → MCP  build_elm_inputs_from_location(run_dir)
framework → MCP  run_elm_ensemble                    → JOB A   build + clone + run
framework        sbatch --dependency=afterany:A      → JOB B   then EXITS

                 …SLURM starts B by itself when A ends…

JOB B            the ordinary --resume path:
                   collect_elm_results → package → analyze
                   Analyzer fetches observations → compare_elm_to_observations
                 Slurm mails on B ⇒ "mail arrived" means "analysis is on disk"
```

**The framework never waits.** It records the job id in `run_state.json` and
exits. Job B is the resume path a person would trigger, with SLURM pressing the
button. Same ledger, same stages, no polling loop, no held allocation.

**`afterany`, not `afterok`.** With `afterok`, a failed ensemble means B never
runs and *no mail is ever sent* — the silent failure that occurred twice on
2026-08-06. `afterany` means B always runs and always reports, including "the
ensemble failed, here is why."

**This removes a circular dependency.** `run_elm_study` sbatched a job that
called `workflow.py --finalize` — the server's job executing the client's code.
Jobs A and B invert it correctly. `run_elm_study` and `workflow.py --finalize`
are both deleted; `--finalize` existed only because the analysis ran *inside* the
job it would otherwise have had to poll, and job B does not have that problem.

### Verified on Compy, 2026-08-07

Jobs **770940** (A) and **770941** (B):

```
depA   COMPLETED   08:39:37 → 08:40:25
depB   COMPLETED   08:40:25 → 08:40:27      started 2 s after A ended
B.log  "B sees A's log:" followed by A's output
```

B sat in `Dependency` state holding no allocation while A ran; SLURM started it
unprompted; B read what A wrote; the submitting shell exited immediately.

**A failure worth keeping.** The first attempt (770938/770939) put the sbatch
scripts and logs under node-local `/tmp`. Both jobs failed in **2 seconds with
completely empty logs** — no error, no message, nothing to diagnose. Everything
in the chain must be on shared filesystem, and the framework should refuse to
submit when the run directory is not on a shared mount.

---

## 7. Layout

```
mcp/elm-mcp/
  main.py            PINNED by mcp_config.json — the 7 tool definitions, thin
  src/
    inputs.py        warm start · donor soil · runtime_config · case_inputs.json
    surfaces.py      surface + domain generation
    cases.py         CIME: builder · wrapper · adapter
    results.py       extraction · named series · variable registry
    compare.py       observation comparison
    plots.py         every ELM figure
  scripts/
    ensemble_job.py  runs inside job A          ← build_cases_job.py
    analyze_run.py   ┐
    analyze_agentic.py │ run by hand on a finished run
    replot.py        ┘
```

`main.py` stays at the top level because `mcp_config.json` pins its absolute
path. It adds `src/` to `sys.path` the same way it already adds the framework's.

**The subdirectory is `scripts/`, not `tools/`.** `_load_tool` resolves to the
*framework's* `tools/` and caches by bare module name:

```python
path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
if name in sys.modules: return sys.modules[name]
```

A same-named file in both directories would collide in `sys.modules`, and
whichever imported first would silently serve both. That presents as a wrong
figure three steps later.

### What moves — ~264 KB

Nothing in `src/core/` or `src/agents/` imports any of it. Only four `tools/`
CLIs do, and they move with it.

```
src/core/    elm_exp_manager 48K · elm_results_analyzer 41K
             elm_surface_generator 40K · elm_wrapper 25K
             elm_experiment_builder 19K · elm_input_agent 5K
             elm_domain_generator 5K                             ≈ 182 KB

analysis/    step1_compare_swe 25K · step2_investigate 21K
             step1_compare_wtd 16K · step1_compare_streamflow 16K
             step1_compare 3K                                    ≈  82 KB

tools/       build_column_inputs.py · make_warmstart.py · make_finidat_subset.py
             plot_columns.py · submit_cases.sh · run_cases.sh
scripts/     analyze_run.py · analyze_agentic.py · replot.py
```

The four imported ones (`make_finidat_subset`, `make_warmstart`,
`build_column_inputs`, `plot_columns`) become modules and keep their `__main__`
blocks, so they stay runnable.

### Deleted, not moved

| what | why |
|---|---|
| `ELMExpManager` (the class) | dissolved across `src/`; nothing left to hold |
| `_run_batch` | `submit_elm_ensemble` is a faithful copy — same `cases.json` + `exe_path.txt`, same `submit_cases.sh`, same job-id regex |
| `_build_cases`' local builder branch | duplicates `build_elm_cases`; a second dead fallback sits under it |
| `_run`'s serial `srun` path | needs a live `elm_agent` and a held allocation |
| `tools/build_cases.py` | the CLI form of the same duplication |
| `tools/run_study.sh` | replaced by jobs A + B |
| `workflow.py --finalize` | existed only for the in-job polling problem |

### Stays in the framework

`expand_sampling.py`, and the question-driven half of the Analyzer:
`step0_context` 18K, `step3_interpret` 20K, `step2_derive` 15K, `step1_geo` 12K,
`step4_report` 11K, `step1_maps` 6K.

`_couple_pflotran` belongs to neither server and stays.

---

## 8. Why observation comparison moves

The four comparison steps hardcode ELM variable names — `H2OSNO`, `QDRAI`,
`QOVER`, `ZWT`, `H2OSOI` — with **no model gating anywhere**, because there was
never a second model:

```
step1_compare_streamflow.py   12 hardcoded ELM variable names
step2_investigate.py          11
step1_compare_wtd.py           6
step1_compare_swe.py           3
```

They are ELM code that happened to live in the Analyzer.

**The framework fetches the observations and passes them in.** The MCP does not
call `snotel-mcp` or `usgs-water-mcp` itself: dependencies stay one-directional,
and the framework already drives those clients.

```
compare_elm_to_observations(run_dir, observations, variable)
  ├─ map variable → ELM field       SWE → H2OSNO, mm
  ├─ extract the matching series    reuses the collect internals
  └─ compare + figure
```

What stays with the Analyzer is the judgment: what the user asked, which
comparisons are worth running, what may be claimed, and what must be withheld.
The MCP answers *"how did ELM do against this observation"*; the Analyzer decides
*"does that answer the question, and what can we honestly say."*

### The figure style

`plots.py` ends up owning the sampling design, `column_surfaces.png`, and the
comparison figures. The house rule — *no interpretation on the plot, large fonts,
no redundant labels* — would then live inside a model server, and would be
defined twice once PFLOTRAN grows its own `plots.py`.

Pull the rcParams and helpers into a small standalone style module both servers
import. It is neither model knowledge nor orchestration, so it does not violate
the rule; it is the house style, defined once.

---

## 9. Implementation plan

**Built in workflow order** — inputs, then run, then results — so that after
every phase there is a working end-to-end path, made of new tools in front and
existing tools behind.

### How phases are proven — decided 2026-08-07

**Not by the existing test suite.** It is not evidence, and this was measured
rather than assumed:

```
963 collected · 894 pass · 69 skip
   34 of the skips are tests/test_elm_integration.py
   plus test_elm_e2e_minimal.py, the whole-pipeline test
```

The two files most capable of reproducing a real failure never run — they need a
node, `$PSCRATCH` and an E3SM build. What does run is largely structural (855
uses of `tmp_path`; `test_elm_exp_manager_structure.py` is 82 tests asserting
that source code *says* things) and only 18 of 46 test files touch real data at
all.

The decisive evidence is the failures of 2026-08-06, every one of which shipped
green: the RUNDIR clobber (live since 07-25), `_as_extract` packaging 19 clean
columns as `columns_total: 0`, an analyzer failure recorded as done, a study
reporting its script's exit code instead of its own, mail that was never
delivered, and 19 columns when 3 were asked for. Each was found by running the
real thing.

`_as_extract` is the sharpest case: a test for that exact bug existed, was
green, and was **structurally incapable** of catching it — written against
PFLOTRAN, whose rows are a list, so it could never reach the dict branch. Green
does not mean covered.

So: **every phase below is proven by a real run**, and the suite is used only as
an import check during relocation, which is the one thing it reliably detects.
No new tests are added to it. A replacement suite is written from scratch at the
end (phase 7).

### Why input generation goes first

It is the **only phase that tests the rule itself.** `collect_elm_results`
already worked under the old rule and was parity-verified there, so building it
first proves nothing about this design. Input generation is what crosses the
boundary the rule moved: the warm start, the CONUS donor lookup, and the bulk
data dependency the server currently does not have. If that cannot work behind an
MCP boundary, this design is wrong — and everything built before finding out is
wasted.

It also slots in without breaking anything. `case_inputs.json` is already a
stable contract: `build_elm_cases` reads that file and nothing else. So the new
input builder goes in *front* of the existing build/submit/collect tools and they
consume its output unchanged.

---

### Phase 0 — preconditions

* Split `tests/test_requested_columns.py`: the requested-column tests are about
  the strategy check and stay; the new `TestRowsSurviveEitherShape` class is
  about the extract contract and belongs in `tests/test_extract_contract.py`.
* Commit that plus the outstanding `_as_extract` fix (881 passing, uncommitted).
  **This is the instrument, not housekeeping** — Phase 4's parity check runs
  *through* `_as_extract`, and with the bug present the ELM baseline reports zero
  rows. Comparing against a ruler that reads zero either passes falsely or fails
  for the wrong reason.
* Record the baseline test count, and pick a completed run directory to serve as
  the parity fixture.
* Create `mcp/elm-mcp/src/` and `scripts/`; move `build_cases_job.py` →
  `scripts/ensemble_job.py` with no behaviour change. Proves the layout and the
  `sys.path` wiring before anything depends on it.

---

### Phase 1 — `build_elm_inputs_from_location`

The largest move and the one that validates the rule. Split into five steps
because ~182 KB of relocation plus a new capability in one commit is where silent
breakage lives.

#### 1a — probe the data dependency

Can the MCP **process** open the CONUS restart and surfdata? The server's
`_requirements()` checks E3SM, CIME, `$PSCRATCH` and the scheduler — no bulk
data, because it never needed any. Recall that MCP clients forward only `HOME`,
`LOGNAME`, `PATH`, `SHELL`, `USER`, so anything the warm start reads through an
environment variable will not be there.

Minutes of work, and if the answer is no the rule is in trouble before anything
has moved. **Do this first.**

#### 1a RESULT — PASSED, 2026-08-07

`mcp/elm-mcp/scripts/probe_conus_access.py`, run normally and with `--stripped`
(which re-execs carrying only `HOME`, `LOGNAME`, `PATH`, `SHELL`, `USER`).
**Identical output, all checks pass, in both.**

```
paths      manifest · conus_surfdata · input_files          all readable
open       netCDF4 1.7.2 · manifest parses 12 bands
           OPENS a restart   62840 gridcells, 237 variables
           OPENS a surfdata  91 variables
lookup     38.9N → band lat7 → grid1d_lat/lon argmin
           → cell 1276906 at 38.8958N -107.0042E
```

The lookup is the real one, not an approximation: `make_finidat_subset.py:119`
uses `grid1d_lat`/`grid1d_lon` with the same `>180 → -360` conversion and the
same `argmin`, and returns the same cell.

**Why it passes:** every data root is a hardcoded absolute path. The only
environment variable anywhere in the warm-start chain is
`IDEAS_WARMSTART_WORKERS`, a tuning knob with a default. Nothing resolves
through the environment, so nothing is lost when the environment is stripped.

**So Rule B's data dependency is viable.** The warm start can move.

#### But the probe found a real fragility

All 12 CONUS restart bands live in **another user's scratch directory**:

```
/compyfs/bish218/e3sm_scratch/conus_lat*/run/*.elm.r.2040-01-01-00000.nc
   owner bish218 · world-readable · 3.5 GB each · ~43 GB total · dated May 17
   /compyfs is 83% full (1.5P of 1.8P)
```

Readable today, and not ours to protect. Scratch is what gets purged when a
filesystem fills, and the warm start is what makes every column credible — so
this is a single point of failure for the scientific validity of every run, not
just for a build step.

Two consequences, the first of which is **done** (see below):

* `_requirements()` must check the restart manifest **and** that at least one
  band resolves, so a missing restart is diagnosed by
  `describe_elm_capabilities` rather than eight minutes into a case build.
* Owning a copy of these files is worth costing out separately. Not a blocker
  for this rework; it is a standing risk that the rework makes the MCP's
  problem rather than the framework's.

#### 1a-bis — the path resolver — DONE, 2026-08-07

`mcp/elm-mcp/src/paths.py`. Hardcoded paths were what made 1a pass; making them
configurable must not give that back.

```
precedence   1  explicit tool argument       recorded in provenance
             2  IDEAS_CONUS_RESTART etc.     works via our MCPManager
             3  paths.json beside main.py    works under ANY client
             4  the hardcoded default        unchanged
```

**Layer 3 is the point.** `mcp_client.py:78` does `os.environ.copy()`, so an
environment variable reaches the server under *our* manager and vanishes under a
standard client — an override that works in the framework and fails silently in
Claude Code is worse than none. A file the server reads itself has no such
asymmetry. `mcp_config.json` has no `env` block to use instead; the manager
reads only `command` and `args`.

`describe()` reports, per path: the value, **which layer supplied it**, whether
it is readable, owner, mtime, and for the manifest how many bands actually
resolve. On Compy today that renders as:

```
conus_restart_manifest  source=default  present=True  bands=12/12
   WARN: restarts: owned by bish218, not by you — you cannot protect it
   WARN: restarts: on a personal scratch tree, which is what gets purged
                   when the filesystem fills
```

Surfaced under `data_paths` in `describe_elm_capabilities`, deliberately **not**
under `requirements`: the server does not generate inputs yet, so a missing
restart cannot stop it and must not gate `ready`. Phase 1c moves them.

`provenance()` returns value **and** source for every path, for
`case_inputs.json` — because which restart a column warm-started from is a fact
about the science, and today it survives only as a NetCDF attribute
(`make_finidat_subset.py:259`) that nothing the Analyzer reads ever sees.

13 tests in `tests/test_elm_mcp_paths.py`, including the two failures that would
otherwise be silent: a manifest whose restarts have vanished (`present: False`,
not "fine"), and a `paths.json` with a typo (reported, not indistinguishable
from absence).

#### 1b — move the modules, change no behaviour

`elm_exp_manager`, `elm_results_analyzer`, `elm_surface_generator`,
`elm_wrapper`, `elm_experiment_builder`, `elm_input_agent`,
`elm_domain_generator` → `mcp/elm-mcp/src/`; `build_column_inputs`,
`make_warmstart`, `make_finidat_subset`, `plot_columns` with them, keeping their
`__main__` blocks.

**Proof:** the suite still imports everything (its one reliable use), *and* a
real study runs end-to-end through the moved modules. A pure relocation that
only satisfies the first has not been shown to work.

#### 1c — build the tool

`_warm_start`, `_attach_donor_soil`, `_to_runtime_config`,
`_generate_column_inputs`, `_write_case_inputs` as private functions in
`src/inputs.py` and `src/surfaces.py`; the tool itself ~15 lines calling them in
order. It reads `01_inputs/columns.json`, rewrites it with the **snapped**
columns, and writes `case_inputs.json`.

#### 1d — parity

The `case_inputs.json` it writes must match what `_build_case_inputs` writes
today, **field for field, including all 13 `runtime_config` keys** — the same
check `4651d5a` ran. Plus: no `ELMAgentAdapter` repr anywhere in the file.

**Risk, from experience:** deleting `build_elm_cases` previously took the
`runtime_config` extraction with it, and a job was submitted against a case list
naming **no FSURDAT, no FINIDAT and no domain paths** — a file that existed and
looked plausible. Assert on the *content* of what is written, never on the
absence of an exception.

#### 1e — wire it in

The framework calls it after `_sample_columns`. The existing `build_elm_cases`
consumes the result untouched, so the pipeline runs end-to-end from here.

**`ELMExpManager` is NOT deletable after this** — corrected 2026-08-07. It
still owns 22 methods Phase 1 does not touch: `_run`, `_poll`, `_collect`,
`_build_cases`, `_run_batch`, `_extract`, `_outcomes_from_disk`,
`_run_summary_for`, `adopt_completed_run`, the two report writers,
`_plot_setups` and `_couple_pflotran`. Those move in phases 2 and 4, so the
class dies **after phase 4**, at ~923 lines until then.

### The stage boundary — decided 2026-08-07

The tool spans two existing stages: the warm start must run inside `materialize`
(it MOVES the columns, so `columns.json` must be written after it), while
`case_inputs.json` belongs to `build_case_inputs`. One tool call now does both,
and a stage is a save point — the granularity `--resume` skips at.

Measured, because a save point is worth what it protects: surfaces are
content-hash cached and re-run near-free, the warm start is not (`build_finidats`
has no `exists()` check, ~1-2 min for 19 columns). So the current split protects
about a minute.

**Chosen: merge, and put the boundary where the layers already divide.**

```
sample_columns    framework    where the columns go   (3 MCP calls, worth protecting)
build_inputs      ELM MCP      what ELM sees there    (one tool call)
```

The save point becomes the layer boundary, so the ledger describes the
architecture instead of cutting across it. `STAGES` entries are ledger keys, so
`materialize → sample_columns` and `build_case_inputs → build_inputs` need an
alias map or in-flight runs find neither.

**Bigger than a rename.** `_materialize` currently returns plan + executable
payload, and two things read `CONDITIONS_COUPLERS` downstream:
`_already_executable` and `_extract` (for the limitations payload's forcing
years). So `build_inputs` must return the run plan for merging, not only
`case_inputs.json` — otherwise the caveats lose their years.

---

### Phase 2 — `run_elm_ensemble`

Merge `build_elm_cases` and `submit_elm_ensemble` into one job: reserve the node,
build the reference case, clone the rest with `--keepexe`, run all columns
concurrently, write `built_cases.json`, end `run.log` with `ALL_DONE`. Skip the
build when `built_cases.json` already reports success.

Default partition `slurm` (4 d), since the walltime must now cover build + run
together.

**Proof:** a real ensemble end-to-end. Then a second submission against the same
run directory, which must skip the build and only run.

**Then delete** `_run_batch`, the local builder branch, the serial `srun`
fallback, and `tools/build_cases.py`.

---

### Phase 3 — job B

Wire `sbatch --dependency=afterany:<A>` into the framework, running the ordinary
`--resume` path. Add the shared-filesystem precondition check.

The mechanism is already verified (§6); what remains is the wiring.

**Proof:** an unattended run from `_sample_columns` to a mail whose arrival means
the analysis is on disk. Also prove the failure path: kill job A and confirm B
still runs and reports honestly.

**Then delete** `tools/run_study.sh` and `workflow.py --finalize`.

---

### Phase 4 — `collect_elm_results`

Lowest risk, so it comes late rather than early. Restore from `ca5e9a6` and adapt
the renamed constants (`build_manifest.json` → `case_inputs.json`,
`prepared_cases.json` → `built_cases.json`). Move `_extract`,
`_outcomes_from_disk` and `_run_summary_for`'s disk branch into `src/results.py`.

The limitations/assumptions payload **does not move** — `select_limitations`
weighs warm start, spinup and forcing to decide how much a run can be trusted.
That is judgment, not a reading. Rows and units come back from the MCP; the
framework attaches the caveats.

**Proof:** re-run the parity check against the local `_extract`. It passed once —
*52 metric values identical across 4 columns, same statuses, same history-file
counts* — and must pass again before anything depends on it.

**Risk:** `ELMResultsAnalyzer.results` is a **dict keyed by case name**;
PFLOTRAN's is a list. `list()` on the dict yields the case *names*, and packaging
then drops every non-dict — that is how 19 clean columns were packaged as
`columns_total: 0` on 2026-08-06 while the ledger recorded `n_rows=19`. Compare
**row contents**, never counts.

---

### Phase 5 — `compare_elm_to_observations` and `plots.py`

Move the four `step1_compare_*` files and `step2_investigate.py` into
`src/compare.py`; the figures into `src/plots.py`; extract the shared style
module. Add the variable registry to `describe_elm_capabilities`, and named
series to `collect_elm_results`.

**Proof:** the same run analysed before and after produces the same metrics and
the same figures.

---

### Phase 6 — `build_elm_inputs_conceptual`

Promote `tools/make_soil_sweep.py`: change the output shape and add the four
guards from §5. The CONUS-distribution check is optional and can follow.

**Proof:** a conceptual ensemble runs end-to-end, and `experiment.json` carries
`provenance: conceptual` and a blocking cold-start caveat.

---

---

### Phase 7 — a test suite written from scratch

Deferred deliberately to the end: the current suite's shape is a record of past
bugs in code that is about to stop existing, so porting it would carry that
shape forward.

What the replacement has to do differently, taken from why the old one missed
everything:

* **Run the model.** The integration and end-to-end tests must actually execute,
  which means running under `sbatch` rather than requiring an interactive node.
  A suite whose most valuable third is skipped is a suite that reports on its
  own least important part.
* **Use real data as fixtures.** `workflow_outputs/elm_run_20260806_162707` —
  19 columns, 19 case dirs, 17 history files each — is a better fixture than any
  `tmp_path` tree, because the bugs live in the shapes real output takes.
* **Verify each backend on its own data shape.** ELM's rows are a dict,
  PFLOTRAN's a list; a check that passes on one proves nothing about the other.
* **Assert on content, never on the absence of an exception.** The case list
  written with no FSURDAT existed, parsed, and looked plausible.

---

### Ordering constraints

* **1a before everything.** It is the cheapest possible test of the rule.
* **Phases 1 and 4 are parity checks against the existing local path** — the only
  way to know the server is *right* rather than merely quiet. Do not delete a
  local path before its replacement has passed one.
* Every phase leaves a working pipeline: new tools in front, existing tools
  behind.

---

## 10. Known risks

* **A wrong parity check.** The `_as_extract` regression was invisible because it
  was verified against PFLOTRAN, whose rows are a list, while ELM's are a dict.
  Verify each backend on its own data shape.
* **Silent no-op patches.** Three edits on 2026-08-06 were `replace()` calls that
  matched nothing and reported success. Assert after every mechanical edit.
* **Method shadowing.** Python accepts a duplicate `def` in a class body
  silently. `tests/test_no_shadowed_methods.py` guards this; keep it running as
  code moves between modules.
* **Node-local paths.** See §6. Failures are silent and produce empty logs.
* **Environment loss across the boundary.** MCP clients forward only `HOME`,
  `LOGNAME`, `PATH`, `SHELL`, `USER`. `LD_LIBRARY_PATH` and `MODULEPATH` vanish;
  every path the server needs must be set with a default in the launcher.
  Measured clean for the warm-start chain (§9 phase 1a) — but that is a fact
  about today's hardcoded paths, and any new `os.environ` read reintroduces it.
* **The CONUS restarts sit in another user's scratch** (§9 phase 1a):
  `/compyfs/bish218/e3sm_scratch/`, ~43 GB, on a filesystem at 83%. Readable,
  not ours, and purgeable. The warm start is what makes a column credible, so
  losing these does not break a build — it silently removes the basis for every
  claim the ensemble supports.

---

## 11. Decisions recorded, so they read as intent and not drift

1. **Extraction returns to the MCP.** The previous design removed it because
   *"reading history files needs no scheduler, no long wait and no login-node
   CPU."* True, and irrelevant under this rule.
2. **Agent-driven entry reinstated** — previously withdrawn as *"an
   overstatement"* — now with the §5 guards.
3. **Observation comparison and ELM figures leave the Analyzer.** They were never
   model-agnostic; they were ungated ELM code.
4. **Build and run merged.** Nobody wants CIME cases.
5. **The sampling figure moves to the MCP** because the two models want different
   sampling figures — *not*, as was claimed during design, because
   `expand_sampling`'s design plot has an ELM code dependency. It does not: the
   ELM import is in `tools/plot_columns.py`, which draws the ELM *case* plots.
   The conclusion stands; that particular argument for it was wrong.

---

## 11b. The sampler ignores most of the planner's strategy — found 2026-08-07

**Highest-value outstanding item. Larger in scientific impact than the remaining
stage merge, because it restores validation the framework has never had.**

The planner emits a sampling design. The sampler reads two numbers of it.

```
planner said                          sampler did
  n_bands: 5                          5 bands
  n_columns: 19                       19 columns
  per_band: 3                         proportional allocation instead
  n_validation: 4                     nothing
  approach: "...plus columns          nothing — the words "pinned", "station",
    pinned to observation stations"     "per_band", "n_validation" appear
                                        NOWHERE in expand_sampling.py
```

So a design of 15 stratified + 4 station-pinned columns became 19 stratified and
nothing at any station.

**This is the root of a scar elsewhere.** `step1_compare_swe` abandoned station
pairing because *"five stations collapsed onto two columns with elevation offsets
up to 846 m… any agreement that produced was arithmetic, not skill."* Pairing
failed because no column was ever placed AT a station — the instruction that
would have put one there was dropped upstream, and the comparison rebuilt itself
around elevation gradients to work around a gap nobody had noticed.

**The data is already on disk.** Verified on the 19-column run: `reception.json`
carries 37 stations with coordinates, keyed by exactly the IDs the planner cites.

```
streamflow   USGS-09119000          38.52111, -106.94096
swe          538:CO:SNTL            37.93389, -107.67620   2980.9 m
swe          762:CO:SNTL            37.99076, -107.20392   3523.5 m
water_table  USGS-382715107514501   38.45417, -107.86250
```

All four found; none invented. The SNOTEL pair spans the snow-band elevations
exactly as the strategy claimed. Pinning needs no new MCP call.

### What the revision does

```
read strategy.sampling      n_bands · per_band · n_validation · approach
read strategy.validation[]  station IDs per variable
read reception.observations resolve ID -> lat/lon/elevation

PINNED   one column per named station, at the station's own coordinates
BANDED   per_band columns per band, AS STATED
total    pinned + banded, reconciled against n_columns
```

**DECIDED: `_allocate` is dropped for the banded columns.** The planner said 3
per band and `_allocate` overrode it with area-proportional counts plus a
`max(1, ...)` floor. Obeying the planner is simpler and removes a bias worth
naming: the floor gives a sliver band with 3 DEM points the same guaranteed
column as a band covering a third of the basin, which fights the area-weighting
`_merge_column_metadata` expects downstream.

**Open:** what to do when a cited station is not in the fetched set. Given this
whole finding is a dropped instruction nobody noticed for months, fail loudly.

### IMPLEMENTED 2026-08-07 — and it uncovered a second bug

`tools/expand_sampling.py`

| added | does |
|---|---|
| `_station_index` | flattens `reception.observations` to `{id: record}`. Three fetchers, three shapes: streamflow/water_table key on `id`, SNOTEL on `triplet`, and only SNOTEL reports an elevation |
| `_pinned_from_plan` | resolves `plan.validation[].stations`; **raises** on an id reception never fetched |
| `_place_pinned` | one column at each station's own coordinates |
| `_even_allocate` | replaces `_allocate` (deleted) |
| `_farthest_point_select(..., seeds=)` | pinned columns seed the distance array, so a stratified pick never lands on ground a station column already covers. No seeds ⇒ identical picks to before |
| `expand(..., per_band=, pinned=)` | assembles bands ascending, pinned before stratified within each |
| `sampling_design` in the output | asked-for vs built, in `columns.json`, so the design is auditable without holding `plan.json` open beside it — which is what nobody did |

`_materialize` reads `per_band` and resolves stations from `config["reception"]`;
warns when a plan names validation stations but no reception was passed.

**Verified against the 19-column Gunnison run** — the planner's design, reproduced
exactly: **19 = 4 pinned + 15 stratified, 3 per band.** Budget is never exceeded
across four scenarios (as-written / rewritten to 2 / no-pin / budget 6).

**Elevation comes from a 3DEP point query AT the station, not the nearest grid
point.** Measured, and this is the whole ballgame:

```
station                 3DEP    reported   Δ        nearest grid point
538:CO:SNTL           2982.17    2980.9   +1.3      3988.72   (+1006.5 m)
762:CO:SNTL           3518.79    3523.5   -4.7      3621.81   (+103.0 m)
USGS-09119000         2328.11         -      -      2591.83   (+263.7 m)
USGS-382715107514501  1792.51         -      -      2061.21   (+268.7 m)
```

±5 m against the instrument. A grid-based lookup would have been **1006 m** out
on 538:CO:SNTL — the same order as the 846 m offsets that made
`step1_compare_swe` abandon station pairing. Pinning without the point query
would have rebuilt the scar it exists to remove.

**Second bug, pre-existing, found by this work.** `_assign_band` returned the
LAST band for anything it could not place, so an elevation *below* the sampled
minimum was filed with the alpine columns. Invisible while every caller passed a
grid point — the bands are built from those points, so nothing could fall
outside. Pinning is the first caller that can pass an outside elevation, and the
Gunnison well does: **1792 m against a 2031 m sampled minimum → band 5 of 5.**
Now clamps to the nearest band and says so.

Three `_allocate` tests in `test_mcp_tools.py` were removed with the function.
The new suite owes coverage of `_even_allocate`, pinning, and the band clamp.

**Still unproven:** the CLI path (`--run-dir` reading `reception.json`, writing
`columns.json`). The selection logic was verified offline against the real saved
DEM grid; the 3DEP numbers above are real MCP calls. Only the wiring is untested.

---

## 12. Open

* **Fan WTD sits awkwardly in `sample_columns`, and moves when PFLOTRAN does.**
  Sampling selects on elevation alone — `_farthest_point_select` over DEM grid
  points within bands. Fan is queried *afterwards*, at the chosen coordinates,
  and never influences placement. So it looks like decoration in a stage whose
  job is selection.

  It is not decoration. `pflotran_exp_manager` uses `fan_wtd_m` as an initial
  condition and to set `wt_in_domain` — *"a column whose water table is below
  the domain runs fully unsaturated, and on the 2019 Gunnison sample that was 10
  of 19 columns"* — and its header states the reason the fetch is shared:
  PFLOTRAN needs the same fields, so both models run on one column definition
  rather than two samples of the same basin. Moving the fetch into the model
  servers would mean two reads of one dataset, free to drift, and would weaken
  exactly that comparability.

  The honest boundary is therefore **shared vs model-specific**, not selection
  vs enrichment — note that soil is deliberately NOT gathered at sampling time,
  because ELM's soil comes from the warm-start donor and a second profile would
  be a field the model never sees. Left as is; revisit when PFLOTRAN is
  restructured, where a third stage (`sample_columns` → `enrich_columns` →
  `build_inputs`) is the clean split if one is wanted.

  Measured while checking this: Fan's grid is ~0.00834° ≈ 0.93 km, the warm
  start snaps columns by ≤0.409 km, and displacing all 19 columns by 0.4 km
  changes the Fan value for **0 of 19**. So `fan_wtd_m` describing the pre-snap
  coordinate is a real inconsistency with no measured effect — worth knowing if
  the snap distance ever grows.

* **`_merge_column_metadata`** reads `columns.json` at package time to join band,
  priors and soil onto the rows (*"the row's own `soil` field comes back null
  without this"*). Under §6 the MCP rewrites `columns.json`, so it works — but
  ownership of that file is not settled.
* **`step3_interpret.py`** stays in the Analyzer with 3 hardcoded ELM variable
  names, to be resolved through the registry.
* **A range check for impossible derived values.** Nothing flagged
  `recharge_fraction` at 2.72, −0.41, 2.65 and 1.15 on the 19-column run of
  2026-08-06; the Analyzer caught the water balance not closing but never that
  a fraction lay outside [0, 1].
* **PFLOTRAN is not symmetric.** Its server is third-party upstream with 43
  tools, of which the framework uses 4 — `run_pflotran_simulation`,
  `submit_pflotran_ensemble`, `check_pflotran_job`, `collect_pflotran_results`.
  Run/check/collect already match this design's shape; the gap is input
  generation, currently framework-side in `pflotran_input_agent.py`. The clean
  path is registering our own input tool in **our** launcher
  (`mcp/reaction-sandbox-mcp/main.py`) rather than forking upstream. Note also
  there are **three** managers, not two: `pflotran_exp_manager.py` 38K and
  `lambda_pflotran_exp_manager.py` 31K.
