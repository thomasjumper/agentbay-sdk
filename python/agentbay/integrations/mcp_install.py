"""One-command MCP installation for Claude Code, Cursor, Windsurf.

Provides a helper to install the AgentBay MCP server using ``npx mcp-add``,
or can be run directly as a module.

Usage::

    # From Python
    from agentbay.integrations.mcp_install import install_mcp
    install_mcp("claude code")

    # From command line
    python -m agentbay.integrations.mcp_install
    python -m agentbay.integrations.mcp_install cursor
    python -m agentbay.integrations.mcp_install windsurf

    # Or directly with npx (no Python needed)
    npx mcp-add --name agentbay --type http --url https://www.aiagentsbay.com/api/mcp

Environment variables:
    AGENTBAY_API_KEY: Your API key (ab_live_...). If set, it will be passed
        as a Bearer token header to the MCP server.
"""

import os
import subprocess
import sys
from typing import Optional

MCP_URL = "https://www.aiagentsbay.com/api/mcp"
SUPPORTED_CLIENTS = ["claude code", "cursor", "windsurf", "cline", "continue"]


def install_mcp(
    client: str = "claude code",
    api_key: Optional[str] = None,
    url: str = MCP_URL,
) -> int:
    """Install AgentBay MCP for the specified client.

    Args:
        client: Target client name. One of: claude code, cursor, windsurf,
            cline, continue. Defaults to "claude code".
        api_key: AgentBay API key. Falls back to ``AGENTBAY_API_KEY`` env var.
        url: MCP server URL. Defaults to the AgentBay production endpoint.

    Returns:
        Exit code from npx mcp-add (0 = success).
    """
    api_key = api_key or os.environ.get("AGENTBAY_API_KEY", "")

    cmd = [
        "npx", "mcp-add",
        "--name", "agentbay",
        "--type", "http",
        "--url", url,
        "--clients", client,
    ]

    if api_key:
        cmd.extend(["--header", f"Authorization:Bearer {api_key}"])

    print(f"Installing AgentBay MCP for {client}...")
    print(f"  URL: {url}")
    if api_key:
        print(f"  Auth: Bearer {api_key[:10]}...")
    else:
        print("  Auth: none (set AGENTBAY_API_KEY for authenticated access)")
    print()

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("\nAgentBay MCP installed successfully!")
            print(f"  51+ tools available in {client}")
            print("  Docs: https://www.aiagentsbay.com/docs/mcp")
        else:
            print(f"\nInstallation failed (exit code {result.returncode}).")
            print("Try installing manually:")
            print(f"  npx mcp-add --name agentbay --type http --url {url}")
        return result.returncode
    except FileNotFoundError:
        print("Error: npx not found. Install Node.js first:")
        print("  https://nodejs.org/")
        return 1


def uninstall_mcp(client: str = "claude code") -> int:
    """Remove AgentBay MCP from the specified client.

    Args:
        client: Target client name. Defaults to "claude code".

    Returns:
        Exit code from npx mcp-remove (0 = success).
    """
    cmd = ["npx", "mcp-remove", "--name", "agentbay", "--clients", client]
    print(f"Removing AgentBay MCP from {client}...")
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except FileNotFoundError:
        print("Error: npx not found.")
        return 1


if __name__ == "__main__":
    client = sys.argv[1] if len(sys.argv) > 1 else "claude code"
    if client in ("-h", "--help"):
        print("Usage: python -m agentbay.integrations.mcp_install [client]")
        print(f"Clients: {', '.join(SUPPORTED_CLIENTS)}")
        sys.exit(0)
    if client == "uninstall":
        target = sys.argv[2] if len(sys.argv) > 2 else "claude code"
        sys.exit(uninstall_mcp(target))
    sys.exit(install_mcp(client))
