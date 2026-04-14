# src/core/mcp_manager.py
"""
MCP Manager - manages multiple MCP server connections

Changes from original:
    - Forwards timeout from config to MCPClient
"""
import json
from pathlib import Path
from typing import Dict, Optional

from .mcp_client import MCPClient


class MCPManager:
    """Manages multiple MCP client connections."""

    def __init__(self, config_file: str = "mcp_config.json"):
        self.config_file = config_file
        self.clients: Dict[str, MCPClient] = {}
        self._load_config()
        self._initialize_clients()

    def _load_config(self):
        """Load MCP configuration from JSON file."""
        config_path = Path(self.config_file)
        if not config_path.exists():
            print(f"⚠️  MCP config not found: {self.config_file}")
            print(f"   No MCP tools will be available")
            self.config = {"mcp_servers": {}}
            return
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading MCP config: {e}")
            self.config = {"mcp_servers": {}}

    def _initialize_clients(self):
        """Initialize MCP clients from config."""
        for server_name, server_config in \
                self.config.get('mcp_servers', {}).items():
            try:
                client = MCPClient(
                    server_name = server_name,
                    command     = server_config['command'],
                    args        = server_config['args'],
                    timeout     = server_config.get('timeout', 30.0)  # ← NEW
                )
                self.clients[server_name] = client
                print(f"✓ MCP client ready: {server_name}")
            except Exception as e:
                print(f"✗ Failed to initialize MCP {server_name}: {e}")

    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """Get specific MCP client by name."""
        return self.clients.get(server_name)

    def get_all_clients(self) -> Dict[str, MCPClient]:
        """Get all MCP clients."""
        return self.clients

    def has_server(self, server_name: str) -> bool:
        """Check if a specific MCP server is available."""
        return server_name in self.clients