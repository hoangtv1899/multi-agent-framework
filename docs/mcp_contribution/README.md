# Contribution bundle for `reaction_sandbox_mcp`

Three changes, developed and tested on Compy against PFLOTRAN v7.0 / PETSc
3.21.6. Two are bug fixes, one is a new tool.

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
