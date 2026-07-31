# reaction_sandbox_mcp: Perlmutter port notes (step 1, 2026-07-20)

Status: PORTED, REGISTERED, and reactive transport VERIFIED end to end.
Upstream repo (/global/homes/h/hvtran/RCSFA/reaction_sandbox_mcp, commit
79d8c3b) is UNMODIFIED so it stays pullable.

## What was blocking
1. `server.py` does `from fastmcp import FastMCP`. The standalone `fastmcp`
   distribution is not installed here, so it fell back to `mcp_server.py`,
   whose `run()` is a placeholder that prints instead of speaking MCP.
2. `server.py` has NO entry point: it defines 39 tools but never calls
   `mcp.run()`, so it cannot be launched as a server at all.
3. Config points at a WSL dev machine
   (lambda_config.yaml LAMBDA_BASE_DIR = /mnt/c/Users/gara009/OneDrive...),
   PFLOTRAN_EXECUTABLE defaults to /usr/local/bin/pflotran, MPI_COMMAND to
   mpirun.

## The port (all in THIS repo, mcp/reaction-sandbox-mcp/)
- `_shim/fastmcp.py` re-exports `mcp.server.fastmcp.FastMCP`, the same
  implementation our other servers use, under the name upstream expects.
- `main.py` sets PFLOTRAN_EXECUTABLE to our build and MPI_COMMAND to srun,
  puts the shim and the upstream dir on sys.path, chdirs into upstream (it
  resolves some paths relatively), imports `server`, and runs stdio.
- Registered in mcp_config.json as `reaction` (backup: mcp_config.json.bak).
- Verified: MCPManager brings up the client and lists all 39 tools.

## Reactive transport verified with OUR PFLOTRAN build
Build: /global/homes/h/hvtran/petsc/pflotran (commit 7d24fd57c, PETSc 3.24).
Sandboxes compiled in this build: base, biohill, bioTH, calcite, chromium,
clm_cn, equilibrate, example, flexbiohill, gas, pnnl_cyber, pnnl_lambda,
pnnl_N, radon, simple, ufd_wp.
Note `clm_cn` (carbon-nitrogen), `pnnl_lambda` (what this MCP targets), and
`pnnl_N` (nitrogen) are all present.

Smoke test: the build's own shipped LAMBDA regression case
(regression_tests/default/reaction_sandbox/reaction_sandbox_lambda.in) runs
to completion (exit 0, 0.05 s) and its regression output MATCHES the shipped
gold standard. Chemistry evolves as expected (pH 7.62-7.67, HCO3-, NH4+,
HPO4-- all active).

## Traps found (encode these before any production use)
- The upstream example `batch_reaction_example.in` is NOT runnable here:
  (a) uses deprecated `GAS_SPECIES` (this build requires
      `PASSIVE_GAS_SPECIES`);
  (b) places `REACTION_SANDBOX` OUTSIDE the `CHEMISTRY` block (PFLOTRAN
      requires it nested inside);
  (c) its `CYBERNETIC` block uses a `REACTION` sub-keyword this build's
      cybernetic sandbox does not accept (build has `pnnl_cyber`).
  So the MCP server's generated decks must be validated against THIS build,
  which is exactly the Tier-3 configurator's job.
- LAMBDA cases need three files, and two carry their own paths: the .in, the
  reaction network .txt (which itself contains a `DATABASE` line), and the
  network file referenced from the .txt. Relative paths in the shipped tests
  assume the regression_tests working directory.

## Next (step 2, not started)
Add reactive transport to the planner v2 capability inventory, re-run the
frozen nitrate prompt, and show the verdict move off `infeasible`. Keep the
eval's frozen v0.1 prompt pinned so Section 4 results stay reproducible.

## Reactive-transport demonstration (step 2b, 2026-07-21)

Deck builder: tools/build_reactive_demo.py. Takes the flow-only Gunnison
column deck produced by build_pflotran_cases.py (real coordinates, SSURGO
layering, Fan water table) and grafts on the LAMBDA organic-matter sandbox,
encoding the three deck traps from the port: PASSIVE_GAS_SPECIES, the
sandbox nested inside CHEMISTRY, and absolute database/network paths.

Setup: col_01 (Fan WTD 26.1 m, 31.1 m domain, 20 cells), coupled RICHARDS
flow + GIRT transport, 20 years, two recharge scenarios (100 and 10 mm/yr,
the same contrast as the flow-only sweep). Both ran on the login node,
2.9 s and 1.8 s, first attempt, no solver failures.

Result: a recharge-controlled organic-matter degradation front.
| recharge | column carbon consumed | front depth |
|---|---|---|
| 100 mm/yr | 36.8% | reaches about 10-13 m |
| 10 mm/yr  | 16.9% | confined to the top about 2-3 m |
Depth profiles (t=20 y, CH2O(s), initial 110 M): at 100 mm/yr the surface
cell is nearly stripped (0.25) and depletion is still visible at 4.8 m
(73.7); at 10 mm/yr the surface retains 4.05 and 4.8 m is essentially
pristine (109.95). pH carries the signature, dropping from about 8.04 at the
surface to 7.71 at the front.

Interpretation: recharge sets how deep surface-driven biogeochemistry
reaches, the chemical counterpart of the vadose-zone low-pass filtering
result in the flow-only coupling. Figure: docs/paper/fig_reactive_demo.png.

Honest scope. This is a capability demonstration, not a science result:
the reaction network and kinetics are the build's own verified regression
configuration, not calibrated to the Gunnison; there are no in-domain
pore-water chemistry observations, so nothing here is validated; oxygen is
held by the sandbox EQUILIBRATE block (an open re-aerating system by
design), so O2 is not a free diagnostic here; and the nitrogen source is
prescribed inlet chemistry, not ELM-derived, because no solute coupler
exists yet.

---

## LOCAL MODIFICATION to upstream — `tools/simulation.py` (2026-07-31)

**Upstream is no longer pristine.** Everything else in this port was done from
outside the upstream tree precisely so it stayed pullable; this one is not.
A re-clone, a re-unzip, or a `git pull` WILL silently drop it and
`ensemble_parallel` will go back to hanging. Backup of the original is at
`tools/simulation.py.orig`.

### What changed

`_run_ensemble_parallel` used `ProcessPoolExecutor`; it now uses
`ThreadPoolExecutor`. Two lines: the import at the top of the file, and the
`with` statement inside the function.

### Why

`ProcessPoolExecutor` defaults to **fork** on Linux. The MCP stdio server runs
anyio reader/writer threads, so the forked child inherits a lock that may be
held at fork time and deadlocks before doing any work.

Measured, in the order that isolates each variable:

| what was run | result |
|---|---|
| `_run_ensemble_parallel`, 3 decks, OUTSIDE the server | 1.8 s, 12 `.tec`, exit 0 |
| the same through `run_pflotran_simulation` | **300 s MCP timeout, no PFLOTRAN spawned, no output** |
| `ProcessPoolExecutor` in a single-threaded asyncio loop | fine, 0.0 s |
| the same with background threads holding a lock | **deadlock, killed by timeout** |

Threads are also simply correct here: `_run_single` does nothing but wait on
`subprocess.run`, which is I/O-bound and releases the GIL, so a separate
interpreter per job bought nothing even when it worked.

### After the fix

Same call that previously hung: **3.1 s**, exit codes `[0, 0, 0]`, 12 `.tec`
files on disk.

### THE UPSTREAM TREE IS NOT UNDER VERSION CONTROL

`reaction_sandbox_mcp-main` was unzipped, not cloned, so this edit is not
tracked anywhere by git. A recoverable copy lives in THIS repo at
`docs/reaction_mcp_ensemble_parallel.patch` — apply it with

```bash
cd $REACTION_MCP_DIR && patch -p0 < .../docs/reaction_mcp_ensemble_parallel.patch
```

### If you re-pull upstream

Re-apply both lines, or `cp tools/simulation.py.orig tools/simulation.py` and
redo them. Worth sending upstream — it is a two-line fix with a reproducer.
