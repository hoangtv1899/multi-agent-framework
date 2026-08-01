# Packaging ELM as an MCP server

Status: **BUILT** (`ef15874`, `ca5e9a6`). Phase 4 of the resumability work.
Everything below is the design as agreed; §12 records where the build differed
from it and what the parity checks found.

Phases 1–3b (`b49f9b9`, `0fe9c3c`, `51878a9`, `11e4bea`) and the discovery/CLI
layer (`f7fe3da`) are in and live-tested.

---

## 1. Why

Today the framework runs on a login node and ELM's stages run *inside the
framework process*. That couples three things that want to be separate:

* **where the science is decided** — reception and the planner, which are LLM
  calls and want a terminal and a person;
* **where the compute is arranged** — case building, submission, collection,
  which want a machine with CIME, E3SM and a scheduler;
* **how long a session has to live** — which today is "as long as the
  compute", because the stages block.

Phase 3 removed the third coupling for `_run`. This removes the second: an ELM
MCP puts the model-specific work behind a tool boundary, so the framework
becomes an orchestrator that holds the ledger and the packaging, and the server
holds the file formats and the job model.

The secondary benefit is real but should not drive the design: once ELM is a
tool surface, **an agent can drive it directly** — Claude Code, or the framework's
own Analyzer — without the exp manager. See §8.

---

## 2. The one constraint that shapes everything

`MCPManager` opens a **fresh stdio session per call**, and tearing that session
down kills the server and its children. `call_tool_json` returns `None` on
timeout, and the per-server timeout in `mcp_config.json` is 30–300 s today.

That is fine for a terrain query and fatal for ELM, whose two slowest stages are
(by this repo's own measurements, `elm_exp_manager.py:412` and
`pflotran_exp_manager.py:26`):

| stage | cost | source |
|---|---|---|
| first CIME case build | ~8–10 min | `_prepare` docstring |
| each subsequent clone (`--keepexe`) | ~30 s, parallel | `_prepare` docstring |
| 19-column ensemble through SLURM | 2406 s | measured |

A 10-minute `prepare` under a 300 s timeout is not a slow call — it is a CIME
build killed halfway, leaving a half-written case directory.

**So the rule for this server is: every tool either returns quickly, or returns
a job id. No tool blocks on the science.**

That is the generalisation of Phase 3, and it is the single idea the whole
design hangs on. Two ways to satisfy it, and this proposal uses both:

* **submit-and-return** for the ensemble — proven in Phase 3 against job 770603.
* **per-call timeout sizing** for prepare — the same trick
  `_run_via_mcp` already uses for PFLOTRAN (`client.timeout` raised for the
  duration of one call, restored in a `finally`). Prepare is *local* work with
  a bounded cost, so blocking for it is honest; the alternative is inventing a
  task-tracking layer for one stage.

> **Decision needed (D1).** Is a ~10-minute blocking `prepare_elm_cases` call
> acceptable, or should case building also be sbatch'd and become job-shaped?
> Blocking is simpler and matches what happens today. Job-shaping is more
> uniform and survives a dropped session. **Recommendation: blocking now,**
> because Phase 2 already makes a killed prepare cheap — it is not marked done,
> so a resume re-runs it — and because sbatch'ing a CIME build is a separate
> risk not worth bundling into this change.

---

## 3. What goes in the server, and what does not

### Goes in

| stage | why it crosses the boundary cleanly |
|---|---|
| `_build` | Produces per-column surface/domain files and an experiment manifest. Both are files and JSON. |
| `_prepare` | Pure side effect on disk; returns case directory paths. |
| `_run` | Already reduced to "submit, return a job id" in Phase 3. |
| `_extract` | `ELMResultsAnalyzer(experiments, analysis_dir)` needs only `case_dir` paths — verified: `elm_results_analyzer.py:155`. |

### Stays in the framework

**`_materialize` must not move.** It is model-agnostic — PFLOTRAN and ELM share
it — and the base's docstring records why: *"a PFLOTRAN run over columns ELM
already produced is only a comparison if both used one materialisation."*
Forking the sampler into an ELM server would quietly break the coupling
guarantee, and it is already MCP-driven anyway (terrain, fan_wtd, geology).

**The ledger, resume, and packaging stay too.** `run_state.json`, the
rehydration, `experiment.json` — these are how a *study* is tracked, not how a
model is run. A server that also owned them could not be swapped out.

### Undecided

**Warm start** (`_refine_columns` → CONUS restart donors) is ELM-specific and
sits inside the shared materialize step. It is a natural future tool
(`warm_start_columns`) but should not be in the first cut: it would drag restart
file handling across the boundary before the simpler stages have proven the
pattern.

---

## 4. The tool surface

Six tools. Small on purpose — the reaction MCP has 40, and the ones we actually
use are a handful.

```
describe_elm_capabilities()
    → what this server can do, what it needs, what it does NOT do.
      The reaction MCP has no such tool and it cost real time working out
      which of its 40 were usable.

build_elm_cases(columns | run_plan, run_dir, soil_config, substrate)
    → {manifest_path, cases: [{case_name, column_id, lat, lon, ...}],
       inputs_built: n, failed: {column_id: reason}}
    FAST-ish. Generates per-column surface/domain inputs and the manifest.

prepare_elm_cases(run_dir, manifest_path)                    [~10 min, D1]
    → {case_dirs: [...], n_ok, n_failed, failures: {...}}
    Builds the reference case, clones the rest with --keepexe.

submit_elm_ensemble(run_dir, case_dirs, queue, walltime, email?)
    → {job_id, n_cases, log_path}
    RETURNS IMMEDIATELY. This is Phase 3's shape, moved behind the boundary.

check_elm_job(job_id)
    → {state, active: bool, elapsed, reason?}
    squeue, then sacct. `active: false` with `state: null` must be impossible —
    "no answer" is its own outcome (see §6).

collect_elm_results(run_dir, manifest_path)
    → {analysis_path, columns: [{case_name, ok, n_history_files}], units}
    Writes the analysis to disk and returns a POINTER plus a compact summary.
```

### Why `collect` returns a path, not the data

Nineteen columns of daily output over twenty years is tens of megabytes. MCP
responses are JSON over stdio. Returning the series inline would work in testing
and fail on a real watershed — the classic shape of a bug that ships. The
manager reads the file; the file is in the run directory where the ledger can
point at it.

### Why `build` accepts columns *or* a run plan

`columns_to_elm_plan` produces a deeply nested `run_plan`, which the exp manager
has and an agent does not. Accepting a plain list of columns
(`lat/lon/elevation/soil`) is what makes §8 possible. The exp manager passes the
plan it already has; an agent passes columns.

> **Decision needed (D2).** Should `build_elm_cases` accept both, or should the
> plan-shaped entry be a separate tool? **Recommendation: both, one tool** —
> two tools that differ only in input shape is the kind of surface that grows to
> 40.

---

## 5. How the exp manager uses it

The pattern is the one PFLOTRAN already uses, and it changes nothing about
Phases 1–3:

```python
def _run(self, experiments, config):
    client = (config.get("mcp_clients") or {}).get("elm")
    if client is not None and config.get("run_via_mcp", True):
        out = self._submit_via_mcp(experiments, client, config)
        if out is None:
            raise RuntimeError(...)      # no silent demotion — d61eaed
        return out                        # a Pending, exactly as today
    ...
```

`_poll` then calls `check_elm_job` instead of `_slurm_state`, and `_collect`
calls `collect_elm_results`. **The ledger, the resume path, the pending summary
and `--resume` all keep working untouched**, because Phase 3 defined the
contract in terms of `Pending(job_id)` and not in terms of who produced it.

Two rules carried forward from earlier work, both learned the hard way:

* **Configured means used.** If the `elm` client is present and `run_via_mcp` is
  on, a failure raises. It does not fall back to the local path — commit
  `d61eaed`, after a session in which "more reliable" turned out to mean
  "attributable by construction", not "better".
* **Attribution or nothing.** A result that cannot be tied back to a named
  column is not a result. The reaction MCP needed `results_by_input` added for
  exactly this; the ELM server should key everything by `case_name` from the
  start.

---

## 6. The environment problem

This is the trap that cost the most time on the reaction MCP, and it will
recur here in a worse form.

`mcp.client.stdio.get_default_environment()` forwards **only** `HOME`,
`LOGNAME`, `PATH`, `SHELL`, `USER`. Our `MCPManager` happens to forward
`os.environ.copy()` (`mcp_client.py:78`) — a standard client does not. That
difference is why every LAMBDA tool worked through the framework and failed
under Claude Code, same server, same machine.

ELM needs far more than LAMBDA did: the conda environment, the E3SM source
tree, CIME, `$PSCRATCH` / `/compyfs`, and the machine's module environment.

**So the launcher shim must set every path it needs, with a default, exactly as
`mcp/reaction-sandbox-mcp/main.py` does.** Anything read from the ambient
environment is a tool that works for us and fails for anyone else — and, worse,
fails *differently* rather than loudly.

> **Decision needed (D3).** Should the shim source `env_compy.sh`, or hard-code
> the handful of variables with `os.environ.setdefault`? **Recommendation:
> setdefault**, following the reaction shim: sourcing a shell script from a
> stdio server means a subshell and makes the failure mode "some variables"
> rather than "this one, missing".

---

## 7. Where the server runs

**The login node.** Phase 3 settled this: a server that *submits* rather than
*runs* does not need to live where the compute does. Compute nodes were verified
to have full outbound network, so the compute-node option remains open, but it
buys nothing here and costs an allocation held for the server's lifetime.

This also preserves the split you wanted: reception and the planner on the login
node, the ensemble on compute, and nothing in between holding a node open to
wait.

---

## 8. What an agent can do with it

Once the six tools exist, the loop `build → prepare → submit → check → collect`
is drivable by an LLM with no exp manager at all. That makes ELM available to
Claude Code sessions and to the framework's own agents.

Two honest caveats, so this is not oversold:

* **It is not the framework.** No ledger, no strategy gate, no caveat records,
  no `experiment.json`. An agent driving these tools gets an ensemble, not a
  study.
* **`describe_elm_capabilities` is load-bearing.** Without it an agent has to
  infer the workflow from six signatures, and the reaction MCP shows where that
  ends — `configure_reaction_sandbox` is a no-op upstream and
  `validate_pflotran_input` passes decks PFLOTRAN then refuses.

---

## 9. Decisions — SETTLED 2026-08-01

| | question | decision |
|---|---|---|
| **D1** | Blocking `prepare_elm_cases` (~10 min) or sbatch it? | **sbatch it — job-shaped.** (Overrides the recommendation; see §9a.) |
| **D2** | One `build_elm_cases` taking columns *or* a plan, or two tools? | **One tool, both shapes** |
| **D3** | Shim sources `env_compy.sh` or uses `setdefault`? | **`os.environ.setdefault`** |
| **D4** | New repo, or `mcp/elm-mcp/` alongside the others? | **`mcp/elm-mcp/`** |
| **D5** | MCP path default for ELM, or opt-in? | **Default.** (Overrides the recommendation; see §9b.) |

### 9a. What D1 costs — and it is not in the server

Phase 3 made **`run`** job-shaped. It did not make *stages* job-shaped: the
Pending/poll logic is inlined in `execute_plan` for the `run` stage alone.

Making `prepare` job-shaped therefore changes the base before it changes
anything about ELM:

* any stage must be able to return `Pending`;
* `execute_plan` must record it, stop, and return the pending summary from
  *that* stage;
* a resume must poll the right stage — so `_poll` has to be told **which**
  stage it is polling, since a backend will answer differently for a CIME
  build than for an ensemble.

This is foundational, has nothing to do with MCP, and is testable on its own.
It is built first, as **Phase 3b**, before any server code.

The upside of D1 beyond uniformity: case building moves off the login node,
where an 8–10 minute CIME compile is currently a login-node citizen, and a
dropped session stops costing a rebuild.

### 9b. What D5 costs

"Default" means `run_via_mcp` defaults to **True when an `elm` client is
registered**. With no client in `mcp_config.json` the local path still runs —
the same semantics PFLOTRAN already has, and not a silent demotion, because
the choice is made by what is configured rather than by a failure.

The risk taken here is that a 40-minute ensemble is the first real exercise of
new code. It is mitigated by build order, not by hedging the decision: the
parity checks in §10 (steps 2 and 5) compare the server's manifest and rows
against the local path **before** step 6 wires it in. The default goes live
only after those agree.

---

## 10. Build order, and how each step is proved

Each step ends with something demonstrable, in the order that fails cheapest
first.

1. **Shim + `describe_elm_capabilities`.** Proves the server starts, the
   environment is complete, and `MCPManager` can see it. Costs nothing to
   throw away if the shape is wrong.
2. **`build_elm_cases`.** Compare its manifest against the manifest the local
   `_build` produces for the same plan — they must agree column for column.
3. **`submit_elm_ensemble` + `check_elm_job`.** Test against a *sleep* job
   first, as Phase 3 did — a real scheduler, no ELM, nothing destructible.
4. **`prepare_elm_cases`.** The slowest and riskiest; by now the boundary is
   proven and only CIME is new.
5. **`collect_elm_results`.** Compare rows against the local `_extract` on a
   run that already completed — `elm_run_20260730_113010` is on disk and has
   both `experiment.json` and its history files.
6. **Wire into `ELMExpManager`,** opt-in (D5), and run one real watershed
   end-to-end through `--resume`.

Steps 2 and 5 are **parity tests against the existing local path**, which is the
only way to know the server is right rather than merely quiet. That is how the
reaction MCP's `run_pflotran_simulation` was validated, and it is what caught
that the local runner and the MCP agreed exactly — the difference was
attribution, not correctness.

---

## 11. What could go wrong

* **CIME in a non-interactive stdio child.** Case building shells out heavily
  and may assume a terminal or a login shell. Step 4 is where this shows up;
  the mitigation is that steps 1–3 will already have proven the environment.
* **Payload size.** Guarded by returning paths (§4), but `build_elm_cases`
  manifests for a large ensemble should be checked too.
* **Two sources of truth for job state.** The framework has `_slurm_state` and
  the server would have `check_elm_job`. They must not disagree. Simplest fix:
  the server's version is authoritative when the MCP path is on, and
  `_slurm_state` stays for the local path — not shared, not "unified", just
  clearly owned.
* **A killed `prepare` leaving half-built cases.** Real today too. Worth a
  `--clean` or an idempotent rebuild, but not in the first cut.

---

## 12. What was actually built (2026-08-01)

### Seven tools, not six

`collect_prepared_cases` was not in the design, and D1 is why. The design had
`prepare_elm_cases` blocking, so it could return its case directories. Once it
became job-shaped it hands back a job id instead — and a stage that hands back
a job id needs somewhere to hand back its **answer**. The alternative was the
framework reading the server's `prepared_cases.json` directly, which is not a
boundary.

### The parity checks, and what they found

Both agreed exactly, which is the result worth having *because* it was checked
rather than assumed:

| check | result |
|---|---|
| `build_elm_cases` vs local `_build` | identical field for field, including all 13 `runtime_config` keys |
| `collect_elm_results` vs local `_extract` | 52 metric values identical across 4 columns, same statuses, same history-file counts |

The manifest was also asserted to contain no `ELMAgentAdapter` repr — the exact
Phase 3 failure, checked for rather than hoped against.

### D1 on real hardware

Job **770694** ran a CIME case build inside a SLURM job and wrote
`PREPARE_DONE 2/2` in **10:14**, both cases on `/compyfs`. This was the riskiest
item in the plan: building cases in a non-interactive batch child had never been
done here, and it is the one thing no amount of local testing could establish.

### The payload argument is now a measurement

§4 argued that `collect` must return a path. For **four** columns the extraction
is **1320 KB** against a **0.8 KB** response. Nineteen columns would be ~6 MB
inline over stdio.

### Two things the wiring forced that the design did not anticipate

**`_prepare` now takes `config`.** It needs to see `mcp_clients` the same way
`_build` and `_run` do, and the base was calling it with `experiments` alone.
Defaulted, so a caller that predates the change still works.

**`_mcp_call` raises on `error` only when the payload has no `ok` key.** A tool
returning `ok` is reporting an *outcome* — a build that failed, an ensemble with
no results — and the caller has more to say about it, including the job's log.
Raising on both made those richer messages dead code. Found by a test that was
asserting on a message which could never actually be produced.

### Still open

* **Warm start** (§3, "Undecided") remains in the framework. Nothing changed.
* **`describe_elm_capabilities` is the contract.** It reports `available` vs
  `planned` per tool and checks every requirement against the filesystem; keep
  it truthful when adding tools, or it becomes the thing it was written to
  prevent.
* **A killed `prepare` still leaves half-built cases** (§11). Real before this
  change and real after it.
