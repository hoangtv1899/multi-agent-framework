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
