# github_mcp/github_session.py

import os
import asyncio
from contextlib import asynccontextmanager
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


def _get_npx_command() -> str:
    """
    Return platform-safe npx command.
    """
    return "npx.cmd" if os.name == "nt" else "npx"


@asynccontextmanager
async def create_github_mcp_session():
    """
    Creates and initializes a GitHub MCP session.

    IMPORTANT:
    - Async-only
    - Must be awaited by caller
    - Owns stdio lifecycle
    """

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN environment variable is required")

    # Use minimal env dict (matching working example)
    # Only include GITHUB_TOKEN to avoid potential conflicts
    env = {
        "GITHUB_TOKEN": github_token,
    }

    server_params = StdioServerParameters(
        command=_get_npx_command(),
        args=["-y", "@modelcontextprotocol/server-github"],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            # CRITICAL: ClientSession must be used as a context manager
            async with ClientSession(read, write) as session:
                # Initialize the session (no timeout wrapper - let context manager handle it)
                await session.initialize()
                
                # Yield the session for use
                yield session
    except ExceptionGroup as e:
        # TaskGroup exceptions from stdio_client - extract the actual error
        if e.exceptions:
            first_exception = e.exceptions[0]
            raise RuntimeError(
                f"MCP server connection failed: {type(first_exception).__name__}: {first_exception}"
            ) from first_exception
        raise RuntimeError(f"MCP server connection failed: {e}") from e
