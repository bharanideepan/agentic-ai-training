# GitHub MCP Package Initialization
# =================================
# This package contains Model Context Protocol (MCP) integration modules for GitHub.
# Note: This is a local package, not the 'mcp' SDK package.

from .github_session import create_github_mcp_session

__all__ = [
    "create_github_mcp_session",
]

