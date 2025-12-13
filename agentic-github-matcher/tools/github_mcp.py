"""
GitHub MCP Integration - MCP-Only Implementation (Async-Only)
=============================================================

This module provides GitHub search functionality using the Model Context Protocol (MCP) ONLY.
It integrates with the official GitHub MCP server (@modelcontextprotocol/server-github)
to provide standardized GitHub API access.

NO REST API FALLBACKS - This module uses MCP exclusively.

ALL FUNCTIONS ARE ASYNC-ONLY - No sync wrappers, no asyncio.run(), no threads.
The caller must be in an async context.

The MCP server runs via:
  npx @modelcontextprotocol/server-github

Environment Variables Required:
    - GITHUB_TOKEN: Personal access token for GitHub API
"""

import json
from typing import List, Dict, Any
from github_mcp.github_session import create_github_mcp_session


# ==============================================
# HELPER CONSTANTS
# ==============================================

_LANGUAGES = {
    "python", "javascript", "typescript", "java", "c", "c++", "cpp",
    "c#", "csharp", "go", "rust", "ruby", "php", "swift", "kotlin",
    "scala", "r", "matlab", "julia", "perl", "haskell", "elixir",
    "clojure", "dart", "lua", "shell", "bash", "powershell", "sql",
    "html", "css", "sass", "less", "vue", "react", "angular"
}

_KNOWN_ORGANIZATIONS = {
    "microsoft", "google", "facebook", "meta", "apple", "amazon", "netflix",
    "twitter", "x", "linkedin", "github", "gitlab", "bitbucket", "atlassian",
    "adobe", "salesforce", "oracle", "ibm", "intel", "nvidia", "amd",
    "uber", "airbnb", "spotify", "stripe", "paypal", "square",
    "apache", "eclipse", "mozilla", "linux", "kubernetes", "docker",
    "tensorflow", "pytorch", "opencv", "scikit-learn",
    "django", "rails", "spring", "angular", "vuejs", "reactjs",
    "nodejs", "expressjs", "nestjs", "fastapi", "flask",
    "jquery", "lodash", "underscore", "moment", "axios", "request",
    "bootstrap", "foundation", "material-ui", "ant-design",
    "webpack", "rollup", "vite", "parcel", "babel",
    "googleapis", "aws", "azure", "gcp",
    "helm", "istio", "prometheus", "grafana",
    "elastic", "mongodb", "redis", "postgresql", "mysql",
    "nginx", "traefik", "envoy",
    "ansible", "terraform", "vagrant", "packer", "consul",
    "vault", "nomad", "nomadproject",
}


def _is_organization_or_company(username: str) -> bool:
    """Check if a username appears to be an organization."""
    username_lower = username.lower()
    if username_lower in _KNOWN_ORGANIZATIONS:
        return True
    
    org_patterns = [
        username_lower.endswith("-org"),
        username_lower.endswith("-team"),
        username_lower.endswith("-labs"),
        username_lower.endswith("-inc"),
        username_lower.endswith("-llc"),
        username_lower.endswith("-corp"),
        username_lower.startswith("team-"),
        username_lower.startswith("org-"),
        "-" in username_lower and len(username_lower.split("-")) > 2,
    ]
    
    return sum(org_patterns) >= 2


# ==============================================
# USER SEARCH
# ==============================================

async def search_users_by_skills_mcp(
    skills: List[str],
    max_results: int = 10,
    min_followers: int = 50,
) -> Dict[str, Any]:
    """
    Search GitHub users using GitHub MCP.
    
    Returns dict with 'users' list and 'success' flag to match agent expectations.
    """
    query = " ".join(skills[:3])  # Limit to top 3 skills for query

    try:
        async with create_github_mcp_session() as session:
            # Use "q" parameter (not "query") as per working example
            result = await session.call_tool(
                "search_users",
                {
                    "q": f"{query} followers:>{min_followers} type:User",
                    "sort": "followers",  # allowed: followers, repositories, joined
                    "order": "desc",
                    "per_page": min(max_results, 100),  # max 100 per GitHub API
                    "page": 1,
                },
            )

            # MCP result format: result.content is a list of TextContent items
            # Each item has item.type == "text" and item.text contains JSON string
            # JSON structure: {"total_count": int, "incomplete_results": bool, "items": [...]}
            users = []
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'type') and item.type == "text":
                        # Parse JSON from item.text
                        data = json.loads(item.text)
                        # GitHub search returns {"items": [...]}
                        items = data.get("items", [])
                        for u in items[:max_results]:
                            username = u.get("login", "")
                            user_type = u.get("type", "User")
                            
                            # Skip organizations
                            if user_type != "User" or _is_organization_or_company(username):
                                continue
                            
                            users.append({
                                "username": username,
                                "html_url": u.get("html_url", ""),
                                "avatar_url": u.get("avatar_url", ""),
                                "type": user_type,
                                "score": u.get("score", 0)
                            })
                        break  # Only process first text item

            return {
                "users": users,
                "total_count": len(users),
                "query": query,
                "success": True
            }
    except ExceptionGroup as e:
        # Handle TaskGroup exceptions from stdio_client
        error_msg = f"TaskGroup error: {e}"
        if e.exceptions:
            first_exception = e.exceptions[0]
            error_msg = f"MCP server error: {type(first_exception).__name__}: {first_exception}"
        print(f"  [MCP] ⚠ Exception in search_users_by_skills: {error_msg}")
        import traceback
        print(f"  [MCP] 🔍 Traceback: {traceback.format_exc()}")
        return {
            "users": [],
            "total_count": 0,
            "query": query,
            "success": False,
            "error": error_msg
        }
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"  [MCP] ⚠ Exception in search_users_by_skills ({error_type}): {error_msg}")
        import traceback
        print(f"  [MCP] 🔍 Traceback: {traceback.format_exc()}")
        return {
            "users": [],
            "total_count": 0,
            "query": query,
            "success": False,
            "error": f"{error_type}: {error_msg}"
        }


# Export helper functions for use in other modules
__all__ = [
    "search_users_by_skills_mcp",
    "_is_organization_or_company",
    "_LANGUAGES"
]
