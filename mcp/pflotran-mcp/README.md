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
- `lambda_pflotran_exp_manager.py` — the Lambda-PFLOTRAN subclass, moved
  here unchanged; **does not import** since the parent went MCP-only on
  2026-08-18 (see its header).
