"""Shim: the reaction_sandbox_mcp server does `from fastmcp import FastMCP`.

The standalone `fastmcp` distribution is not installed in the NERSC runtime,
and the server's bundled fallback (mcp_server.SimpleMCP) is a placeholder whose
run() does not speak the MCP protocol. The official `mcp` package ships an
equivalent FastMCP, which is what our other servers use, so we re-export it
under the name the server expects. This keeps the upstream repository
unmodified and pullable.
"""
from mcp.server.fastmcp import FastMCP  # noqa: F401
