# Contribution bundle for `reaction_sandbox_mcp`

Changes developed and tested on Compy against PFLOTRAN v7.0 / PETSc
3.21.6: a deadlock fix and per-run timeouts (01), a runnable column deck (02),
a withdrawn scheduler mode (03), and repairs to upstream's own tools (04).

Apply to a pristine checkout with:

```bash
cd <reaction_sandbox_mcp>
patch -p0 tools/simulation.py < 01-simulation-timeout-and-threads.patch
patch -p0 server.py           < 02-server-timeout-and-create_column_deck.patch
cp tools_column_builder.py       tools/column_builder.py
cp tools_pflotran_input_agent.py tools/pflotran_input_agent.py
```

Verified: applying these to pristine copies reproduces our working tree
byte-identically for all four files.

---

## 1. `ensemble_parallel` deadlocked — always

`_run_ensemble_parallel` used `ProcessPoolExecutor`, which defaults to **fork**
on Linux. The MCP stdio server runs anyio reader/writer threads, so a forked
child inherits a lock that may be held at fork time and hangs before doing any
work.

Isolated in four steps:

| what was run | result |
|---|---|
| `_run_ensemble_parallel`, 3 decks, **outside** the server | 1.8 s, 12 output files, exit 0 |
| the same through `run_pflotran_simulation` | **300 s client timeout, no PFLOTRAN spawned, nothing returned** |
| `ProcessPoolExecutor` in a single-threaded asyncio loop | fine, 0.0 s |
| the same with background threads holding a lock | **deadlock, killed by timeout** |

**Fix:** `ThreadPoolExecutor`. `_run_single` does nothing but wait on
`subprocess.run`, which is I/O-bound and releases the GIL, so separate
interpreters bought nothing even when they worked.

**After:** the same call takes **3.1 s**, exit codes `[0, 0, 0]`. At scale,
19 decks run in 7.1 s at `max_parallel=4` and 5.3 s at 8.

## 2. No simulation could be bounded

`_run_single` called `subprocess.run(...)` with no `timeout=`, and nothing
upstream of it supplied one. The only thing that could stop a non-converging
solve was the MCP client abandoning the session — which kills the whole call.
In an ensemble that means **one bad column discards every other column**,
including those that already finished, and the caller gets nothing.

**Fix:** a `timeout` parameter threaded through `run_pflotran_simulation` →
`run_simulation` → `_run_single`, with `TimeoutExpired` returned as a failed
result rather than raised. `_run_ensemble_parallel` needs no logic change —
each worker bounds itself.

**Verified** on a 4-deck ensemble containing one column whose timestep
collapses:

```
timeout=20 →  elapsed 21.4 s
              validation_status: partial
              exit_codes: [0, 0, 0, -1]
              num_successful/failed: 3 / 1
              failed_runs: ['col_HANG.in']
              output from the 3 good columns: preserved
```

Before, that call returned `None` after 300 s with nothing.

Default is `None` (no limit), so existing behaviour is unchanged unless asked
for.

### 2b. Ensemble results could not be attributed

`_run_ensemble_parallel` collected via `as_completed`, so the aggregate
`exit_codes` come back in **completion order** — `exit_codes[i]` does not
belong to `input_files[i]`. A caller could count failures but could not say
which realization failed, how long any took, or where one realization's
output went.

**Fix:** every run is now timed (`execution_time` on each `_run_single`
return), and `_run_ensemble_parallel` returns `results_by_input` — a map from
each input file to its own result. The aggregate lists are unchanged, so
nothing existing breaks.

```
results_by_input: 4 entries
   col_02_r100.in   exit=[0]     success  1.727s   4 out
   col_03_r100.in   exit=[0]     success  1.627s   4 out
   col_06_r100.in   exit=[0]     success  1.926s   4 out
   col_HANG.in      exit=[None]  failed  20.016s   3 out  exceeded 20.0s
```

## 3. New tool — `create_column_deck`

`create_pflotran_input` writes the skeleton of a deck: simulation type, grid,
time, output. It has no parameters for material properties, regions, strata or
flow conditions, so its output has none — and PFLOTRAN rejects it. Measured:
its own generated skeleton fails with exit 87, while `validate_pflotran_input`
reports it as valid.

`create_column_deck` fills that gap for the case the LAMBDA workflow needs — a
1-D variably-saturated soil column:

- soil horizons become cells, van Genuchten parameters carried through, deepest
  horizon extended downward as substrate on a geometrically coarsening grid
- hydrostatic initial condition pinned at the site's water table
- recharge flux on top, steady or transient
- hydrostatic or no-flow bottom

**Model-agnostic by design.** A transient upper boundary is supplied as plain
`[[time_y, flux_m_per_y], ...]` pairs, so it can be driven from a
land-surface model, a measured infiltration record, or anywhere else. The
module imports no land model and knows about none.

**Verified:** the deck it produces is **byte-identical** (879/879 lines) to the
one our own tested builder writes for the same site, and PFLOTRAN runs it —
exit 0, 5 output files. Transient, fully-unsaturated, and no-flow paths each
tested separately.

`tools/pflotran_input_agent.py` is the deck writer it uses. Self-contained:
`os`, `subprocess`, `numpy`.

---

## Note for reviewers

`validate_pflotran_input` currently passes decks PFLOTRAN refuses — a truncated
deck, a primary species absent from the database, a species missing from a
constraint, and the skeleton described above. It checks which blocks are
present, not whether the deck is runnable. Not addressed here, but worth
knowing before trusting it as a gate.

---

## 03 — scheduler job mode — WITHDRAWN 2026-08-18

The three scheduler tools (`submit_pflotran_ensemble`, `check_pflotran_job`,
`collect_pflotran_results`) and their two modules were deleted from our branch
(`fac929a`): nothing called them, and a tool that can `sbatch` behind the
conversation is what nobody wanted. The patch and module copies are gone from
this bundle. Two lessons from it are still worth carrying to any batch path:
`results_by_input` (the aggregate `exit_codes` list is in completion order, so
without the map an ensemble can be counted but not attributed), and waiting
briefly for a result file written from a compute node before declaring it
absent.

## 04 — repairs to upstream tools (on our branch as `fac929a`, not yet a patch)

Found by exercising upstream's own tools on Compy (2026-08-18); every one is
present at `bb773fd`: `run_pflotran_simulation` raised `TypeError` when
`num_cores` was omitted; `check_simulation_status` said `running` for every
finished run (grepped a banner v7 never prints); `create_parameter_ensemble`'s
docstring named keys the sampler does not read; `validate_pflotran_input`
called an unrunnable skeleton valid; `configure_reaction_sandbox` reported
success while inserting nothing on any real CHEMISTRY block; `output_files`
omitted `-mas.dat`/`.regression`. See that commit's message and
`tests/test_column_tools.py` for the tests. To be turned into a patch when the
bundle is next sent.

### Applying

    patch -p0 < 01-simulation-timeout-and-threads.patch
    patch -p0 < 02-server-timeout-and-create_column_deck.patch

Verified when written: applying 01→02 in sequence to the pristine `server.py`
reproduced the working file byte for byte. Our branch has moved on since
(`create_decks_from_columns`, `extract_column_series`, the 04 repairs), so a
fresh bundle should be cut from `git diff bb773fd..HEAD` rather than from
these files.
