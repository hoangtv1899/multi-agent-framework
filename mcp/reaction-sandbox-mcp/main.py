#!/usr/bin/env python3
"""
Launcher for the reaction_sandbox_mcp server (PFLOTRAN reactive transport).

The upstream repository is developed on WSL and has no entry point: server.py
defines the tools but never calls mcp.run(), and it imports the standalone
`fastmcp` package that is absent here. This launcher supplies both without
editing upstream code:

  * puts a `fastmcp` shim on sys.path that re-exports mcp.server.fastmcp
  * points PFLOTRAN_EXECUTABLE at the build this project already uses
  * imports the upstream module and serves it over stdio

Registered in mcp_config.json as `reaction`.
"""
import os
import sys
from pathlib import Path

SERVER_DIR = Path(os.getenv(
    "REACTION_MCP_DIR", "/qfs/people/tran289/IDEAS/reaction_sandbox_mcp-main"))
SHIM_DIR = Path(__file__).resolve().parent / "_shim"
PFLOTRAN = ("/qfs/people/tran289/pflotran/src/pflotran/pflotran")

os.environ.setdefault("PFLOTRAN_EXECUTABLE", PFLOTRAN)
os.environ.setdefault("MPI_COMMAND", "mpirun")

if not SERVER_DIR.is_dir():
    sys.exit(f"reaction MCP: server dir not found: {SERVER_DIR}")

sys.path.insert(0, str(SHIM_DIR))       # `fastmcp` -> mcp.server.fastmcp
sys.path.insert(0, str(SERVER_DIR))     # upstream `server` and `tools` package
os.chdir(SERVER_DIR)                    # upstream resolves some paths relatively

import server  # noqa: E402  (upstream module; defines `mcp` with its tools)

if __name__ == "__main__":
    server.mcp.run(transport="stdio")
