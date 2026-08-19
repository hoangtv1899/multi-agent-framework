# mcp/pflotran-mcp — the framework's side of the PFLOTRAN server

The PFLOTRAN MCP server itself is **not** here. It lives in a separate repo,
`reaction_sandbox_mcp-upstream` (branch `compy-port`), installed in the
`ideas` conda env and launched as the `pflotran-mcp` console script that
`mcp_config.json` names.

This directory holds only what the framework needs to drive that server,
mirroring `mcp/elm-mcp/src/elm_exp_manager.py` for ELM:

- `pflotran_exp_manager.py` — the Experiment Manager for PFLOTRAN: field
  semantics, one build call (`create_decks_from_columns` with the run
  directory as `site_dir`), and the run / extract pass-throughs. Imported by
  path from `src/core/resumable.py::_manager_for`.
- `compare/` — the Analyzer's step-1 comparison for PFLOTRAN columns (one
  observable today, `water_table`: the pressure-crossing depth and the given
  CONUS2 depth against wells), a registry over the framework's
  `agents/analysis/compare_common`, mirroring `mcp/elm-mcp/src/compare/`.
- `sampling_design.py` — the design figure the manager draws after
  `columns.json` (columns over the terrain, rain, the given water table, each
  column's depth on the CONUS2 ladder, unsaturated column), mirroring ELM's.
- `lambda_pflotran_exp_manager.py` — the Lambda-PFLOTRAN subclass, moved
  here unchanged; **does not import** since the parent went MCP-only on
  2026-08-18 (see its header).
