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

# EVERY PATH THIS SERVER NEEDS IS SET HERE, WITH A DEFAULT, because an MCP
# client does not pass the environment through. mcp.client.stdio's
# get_default_environment() forwards only HOME, LOGNAME, PATH, SHELL and USER
# — so a var exported in env_compy.sh reaches this process only when the
# launching client happens to forward the full environment, which our
# MCPManager does and a standard client does not.
#
# LAMBDA_PFLOTRAN_DIR was the one that got missed. Upstream's
# tools/lambda_pipeline.py reads it to put the Lambda package on sys.path and
# otherwise falls back to its author's home directory, so under any ordinary
# MCP client every lambda tool returned
#     "Lambda-PFLOTRAN not available: No module named 'preprocessing'"
# while the same call through the framework succeeded. Same server, same
# machine, different launcher.
os.environ.setdefault("PFLOTRAN_EXECUTABLE", PFLOTRAN)
os.environ.setdefault("MPI_COMMAND", "mpirun")
os.environ.setdefault("LAMBDA_PFLOTRAN_DIR",
                      str(SERVER_DIR / "lambda_pflotran_refactor"))

if not SERVER_DIR.is_dir():
    sys.exit(f"reaction MCP: server dir not found: {SERVER_DIR}")

sys.path.insert(0, str(SHIM_DIR))       # `fastmcp` -> mcp.server.fastmcp
sys.path.insert(0, str(SERVER_DIR))     # upstream `server` and `tools` package
os.chdir(SERVER_DIR)                    # upstream resolves some paths relatively

import server  # noqa: E402  (upstream module; defines `mcp` with its tools)

if __name__ == "__main__":
    server.mcp.run(transport="stdio")
