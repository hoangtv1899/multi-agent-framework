# src/core/mcp_client.py
"""
MCP Client - synchronous wrapper for MCP server communication

Changes from original:
    - Fixed session lifecycle (per-call sessions, no shared state)
    - Added asyncio conflict guard for HPC/Jupyter environments
    - Added timeout support
"""
import asyncio
import os
import json
import sys
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─────────────────────────────────────────────────────────────────────────────
# Where the MCP SERVERS' stderr goes.
#
# Every stdio server logs "Processing request of type CallToolRequest" at INFO
# through rich, and its stderr is inherited by this process — so an interactive
# session is buried under timestamped log lines between every prompt, and the
# reception trace the user is actually reading scrolls away.
#
# The output is worth keeping (it is the only view into a server that fails to
# start), so it goes to a file rather than being discarded. Set IDEAS_MCP_LOG to
# another path to move it, or to "stderr"/"-" to put it back on the terminal
# when debugging a server.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MCP_LOG = "logs/mcp_servers.log"
_ERRLOG_LOCK = threading.Lock()
_ERRLOGS: Dict[str, Any] = {}


def mcp_errlog():
    """Open TextIO for the child's stderr — one cached handle per path.

    Cached because a fresh session is opened per tool call; reopening the file
    hundreds of times per run would be wasteful and would race between threads.
    Falls back to stderr if the path cannot be opened, since losing the log is
    better than failing the call.
    """
    path = os.environ.get("IDEAS_MCP_LOG", DEFAULT_MCP_LOG)
    if path in ("stderr", "-", ""):
        return sys.stderr
    with _ERRLOG_LOCK:
        handle = _ERRLOGS.get(path)
        if handle is None or handle.closed:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                handle = open(path, "a", buffering=1, encoding="utf-8")
            except OSError:
                return sys.stderr
            _ERRLOGS[path] = handle
        return handle


class MCPClient:
    """
    Synchronous MCP client using per-call sessions.

    Each tool call opens a fresh connection to the MCP server,
    makes the call, then closes cleanly.
    """

    def __init__(self,
                 server_name: str,
                 command: str,
                 args: List[str],
                 timeout: float = 30.0):
        self.server_name = server_name
        self.command     = command
        self.args        = args
        self.env         = os.environ.copy()
        self.timeout     = timeout

    # =========================================================================
    # ASYNC CORE (per-call session)
    # =========================================================================

    async def _call_tool_async(self,
                                tool_name: str,
                                arguments: Dict[str, Any]) -> Optional[str]:
        """Open fresh session → call tool → close. Returns raw text or None."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )
        async with stdio_client(server_params, errlog=mcp_errlog()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.content:
                    return result.content[0].text
                return None

    async def _list_tools_async(self) -> List[Dict[str, Any]]:
        """Open fresh session → list tools → close."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )
        async with stdio_client(server_params, errlog=mcp_errlog()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return [
                    {
                        'name':        tool.name,
                        'description': tool.description,
                        'parameters':  tool.inputSchema
                    }
                    for tool in tools_result.tools
                ]

    # =========================================================================
    # ASYNCIO CONFLICT GUARD
    # =========================================================================

    def _run_async(self, coro):
        """
        Run async coroutine from sync code safely.

        Handles two cases:
            1. No event loop running (normal)  → asyncio.run()
            2. Event loop already running      → new thread with own loop
               (Jupyter / Perlmutter HPC environments)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # Normal case
            return asyncio.run(coro)
        else:
            # Already inside an event loop - run in separate thread
            result_container   = {}
            exception_container = {}

            def _thread_target():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_container['value'] = new_loop.run_until_complete(coro)
                except Exception as e:
                    exception_container['error'] = e
                finally:
                    new_loop.close()

            thread = threading.Thread(target=_thread_target, daemon=True)
            thread.start()
            thread.join(timeout=self.timeout)

            if thread.is_alive():
                raise TimeoutError(
                    f"MCP call to {self.server_name} timed out "
                    f"after {self.timeout}s"
                )
            if 'error' in exception_container:
                raise exception_container['error']

            return result_container.get('value')

    # =========================================================================
    # PUBLIC SYNC API
    # =========================================================================

    def call_tool(self,
                  tool_name: str,
                  arguments: Dict[str, Any]) -> Optional[str]:
        """
        Call an MCP tool synchronously. Returns raw text or None on error.

        Example:
            raw = client.call_tool("get_streamflow", {"site_id": "11463000"})
        """
        try:
            return self._run_async(
                asyncio.wait_for(
                    self._call_tool_async(tool_name, arguments),
                    timeout=self.timeout
                )
            )
        except TimeoutError:
            print(f"⚠️  MCP timeout ({self.server_name}.{tool_name}) "
                  f"after {self.timeout}s")
            return None
        except Exception as e:
            print(f"⚠️  MCP tool call failed ({self.server_name}.{tool_name}): {e}")
            return None

    def call_tool_json(self,
                       tool_name: str,
                       arguments: Dict[str, Any]) -> Optional[Dict]:
        """
        Call a tool and parse result as JSON automatically.

        Example:
            data = client.call_tool_json("get_streamflow", {"site_id": "11463000"})
        """
        raw = self.call_tool(tool_name, arguments)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse failed ({self.server_name}.{tool_name}): {e}")
            print(f"   Raw: {raw[:200]}")
            return None

    def list_tools(self) -> List[str]:
        """Return list of tool names available on this MCP server."""
        try:
            tools = self._run_async(self._list_tools_async())
            return [t['name'] for t in tools]
        except Exception as e:
            print(f"⚠️  Could not list tools for {self.server_name}: {e}")
            return []

    def list_tools_detailed(self) -> List[Dict[str, Any]]:
        """Return full tool specs (name, description, parameters) in one session.

        Used by the agentic tool-loop to build OpenAI tool schemas.
        """
        try:
            return self._run_async(self._list_tools_async())
        except Exception as e:
            print(f"⚠️  Could not list detailed tools for {self.server_name}: {e}")
            return []

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get schema/description for a specific tool."""
        try:
            tools = self._run_async(self._list_tools_async())
            for tool in tools:
                if tool['name'] == tool_name:
                    return tool
            return None
        except Exception as e:
            print(f"⚠️  Could not get tool info "
                  f"({self.server_name}.{tool_name}): {e}")
            return None

    def __repr__(self):
        return (f"MCPClient(server='{self.server_name}', "
                f"timeout={self.timeout}s)")