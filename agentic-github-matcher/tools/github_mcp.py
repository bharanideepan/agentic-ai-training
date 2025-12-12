"""
GitHub MCP Integration
======================

This module provides GitHub search functionality using the Model Context Protocol (MCP).
It integrates with the official GitHub MCP server (@modelcontextprotocol/server-github)
to provide standardized GitHub API access.

The MCP approach offers:
- Standardized tool interface
- Better abstraction and error handling
- Built-in rate limiting
- Easier integration with other MCP-compatible tools

Environment Variables Required:
    - GITHUB_TOKEN: Personal access token for GitHub API
"""

import os
import json
import asyncio
import shutil
import sys
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

# Connection timeout settings
# Note: First connection can take 10-30 seconds as npx downloads the MCP server package
# Subsequent connections should be much faster (2-5 seconds) as the package is cached
MCP_CONNECTION_TIMEOUT = 30.0  # 30 seconds timeout for transport connection
MCP_INITIALIZATION_TIMEOUT = 10.0  # 10 seconds timeout for session initialization

# Load environment variables
load_dotenv()

# Try to import MCP SDK
MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("  [MCP] ⚠ MCP SDK not available. Install with: pip install mcp")
    print("  [MCP] ℹ System will fall back to direct REST API calls")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


# ==============================================
# NODE.JS/NPX DETECTION
# ==============================================

def _check_npx_available() -> Tuple[bool, Optional[str]]:
    """
    Check if npx is available on the system.
    
    Returns:
        tuple: (is_available, error_message)
    """
    # Check for npx
    npx_path = shutil.which("npx")
    if npx_path:
        return True, None
    
    # On Windows, try npx.cmd
    if sys.platform == "win32":
        npx_cmd_path = shutil.which("npx.cmd")
        if npx_cmd_path:
            return True, None
        
        # Try to find npm and check for npx.cmd in same directory
        npm_path = shutil.which("npm")
        if npm_path:
            npm_dir = os.path.dirname(npm_path)
            npx_cmd = os.path.join(npm_dir, "npx.cmd")
            if os.path.exists(npx_cmd):
                return True, None
    
    # Check for node
    node_path = shutil.which("node")
    if not node_path:
        return False, "Node.js is not installed. Please install Node.js from https://nodejs.org/"
    
    return False, "npx not found in PATH. Please ensure Node.js and npm are properly installed."


def _get_npx_command() -> Optional[str]:
    """
    Get the correct npx command for the current platform.
    
    Returns:
        str: The npx command to use, or None if not available
    """
    if shutil.which("npx"):
        return "npx"
    
    if sys.platform == "win32":
        if shutil.which("npx.cmd"):
            return "npx.cmd"
        npm_path = shutil.which("npm")
        if npm_path:
            npm_dir = os.path.dirname(npm_path)
            npx_cmd = os.path.join(npm_dir, "npx.cmd")
            if os.path.exists(npx_cmd):
                return npx_cmd
    
    return None


# ==============================================
# MCP CLIENT
# ==============================================

class GitHubMCPClient:
    """
    Client for interacting with GitHub via Model Context Protocol (MCP).
    
    This client connects to the GitHub MCP server and provides
    standardized access to GitHub API functionality.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub MCP client.
        
        Args:
            token: GitHub Personal Access Token (uses GITHUB_TOKEN env var if not provided)
        """
        self.token = token or GITHUB_TOKEN
        self.session: Optional[ClientSession] = None
        self._transport = None
        self._initialized = False
        
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP packages not installed. Install with: pip install mcp")
    
    async def __aenter__(self):
        """Enter async context manager."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to the GitHub MCP server with timeout and progress feedback."""
        if self._initialized:
            return
        
        try:
            print("  [MCP] 🔌 Connecting to GitHub MCP server...")
            print("  [MCP] ⏱ This may take 10-30 seconds on first run (downloading package)...")
            
            # Check if npx is available
            print("  [MCP] 📋 Checking npx availability...")
            is_available, error_msg = _check_npx_available()
            if not is_available:
                raise RuntimeError(f"npx not available: {error_msg}")
            print("  [MCP] ✓ npx found")
            
            # Get the correct npx command
            npx_cmd = _get_npx_command()
            if not npx_cmd:
                raise RuntimeError("Could not find npx command")
            
            # Create server parameters
            print("  [MCP] 📦 Starting MCP server process (this may download package on first run)...")
            server_params = StdioServerParameters(
                command=npx_cmd,
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.token}
            )
            
            # Create stdio client and connect with timeout
            print("  [MCP] 🔗 Establishing connection...")
            self._transport = stdio_client(server_params)
            
            try:
                # Add timeout for transport connection
                read_stream, write_stream = await asyncio.wait_for(
                    self._transport.__aenter__(),
                    timeout=MCP_CONNECTION_TIMEOUT
                )
                print("  [MCP] ✓ Transport connection established")
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Connection timeout after {MCP_CONNECTION_TIMEOUT}s. "
                    "The MCP server may be taking too long to start. "
                    "Try again or check your network connection."
                )
            
            # Create client session
            print("  [MCP] 🔧 Creating client session...")
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize the session with timeout
            print("  [MCP] ⚙️  Initializing session...")
            try:
                await asyncio.wait_for(
                    self.session.initialize(),
                    timeout=MCP_INITIALIZATION_TIMEOUT
                )
                print("  [MCP] ✓ Session initialized")
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Initialization timeout after {MCP_INITIALIZATION_TIMEOUT}s. "
                    "The MCP server may not be responding properly."
                )
            
            self._initialized = True
            print("  [MCP] ✅ Connected to GitHub MCP server successfully")
            
            # List available tools for debugging
            try:
                print("  [MCP] 📋 Listing available tools...")
                tools_result = await asyncio.wait_for(
                    self.session.list_tools(),
                    timeout=5.0
                )
                if hasattr(tools_result, 'tools'):
                    tool_names = [t.name for t in tools_result.tools]
                    print(f"  [MCP] ℹ Available tools ({len(tool_names)}): {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}")
            except asyncio.TimeoutError:
                print("  [MCP] ⚠ Tool listing timed out (continuing anyway)")
            except Exception as e:
                print(f"  [MCP] ⚠ Could not list tools: {e}")
                
        except asyncio.TimeoutError as e:
            error_msg = str(e) if str(e) else "Connection timeout"
            print(f"  [MCP] ❌ Connection timeout: {error_msg}")
            print("  [MCP] 💡 Tip: First connection can take 10-30 seconds while downloading the package")
            print("  [MCP] 💡 Tip: Subsequent connections should be faster (2-5 seconds)")
            self._initialized = False
            # Clean up on timeout
            try:
                if self._transport:
                    await self._transport.__aexit__(None, None, None)
                    self._transport = None
            except:
                pass
            raise RuntimeError(error_msg)
        except Exception as e:
            print(f"  [MCP] ❌ Failed to connect: {e}")
            print("  [MCP] 💡 The system will automatically fall back to direct REST API")
            self._initialized = False
            # Clean up on error
            try:
                if self._transport:
                    await self._transport.__aexit__(None, None, None)
                    self._transport = None
            except:
                pass
            raise
    
    async def disconnect(self):
        """Disconnect from the GitHub MCP server."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            if self._transport:
                await self._transport.__aexit__(None, None, None)
                self._transport = None
        except Exception as e:
            print(f"  [MCP] ⚠ Error disconnecting: {e}")
        finally:
            self._initialized = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if not self._initialized:
            await self.connect()
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract content from MCP response
            if hasattr(result, 'content'):
                content_items = []
                for content_block in result.content:
                    if hasattr(content_block, 'text'):
                        try:
                            parsed = json.loads(content_block.text)
                            if isinstance(parsed, list):
                                content_items.extend(parsed)
                            else:
                                content_items.append(parsed)
                        except json.JSONDecodeError:
                            content_items.append(content_block.text)
                
                return {
                    "success": True,
                    "content": content_items,
                    "isError": getattr(result, 'isError', False)
                }
            else:
                return {"success": False, "error": "Unexpected MCP response format"}
                
        except Exception as e:
            error_msg = f"MCP tool call failed: {str(e)}"
            print(f"  [MCP] ⚠ {error_msg}")
            return {"success": False, "error": error_msg, "exception": str(e)}


# ==============================================
# GLOBAL MCP CLIENT INSTANCE
# ==============================================

_mcp_client: Optional[GitHubMCPClient] = None
_mcp_initialization_attempted = False
_mcp_connection_failed = False


async def initialize_mcp_client() -> bool:
    """
    Initialize the MCP client at application startup.
    
    Returns:
        bool: True if MCP client was successfully initialized, False otherwise
    """
    global _mcp_client, _mcp_connection_failed, _mcp_initialization_attempted
    
    if _mcp_initialization_attempted:
        return _mcp_client is not None and _mcp_client._initialized
    
    _mcp_initialization_attempted = True
    
    if not MCP_AVAILABLE:
        print("  [MCP] ⚠ MCP SDK not available, will use direct REST API")
        return False
    
    if not GITHUB_TOKEN:
        print("  [MCP] ⚠ GITHUB_TOKEN not set, will use direct REST API")
        return False
    
    print("  [MCP] 🔌 Initializing MCP client...")
    print("  [MCP] ⏱ Connection may take 10-30 seconds on first run...")
    try:
        _mcp_client = GitHubMCPClient()
        # Add overall timeout for the entire initialization process
        await asyncio.wait_for(
            _mcp_client.connect(),
            timeout=MCP_CONNECTION_TIMEOUT + MCP_INITIALIZATION_TIMEOUT + 5.0  # Add buffer
        )
        print("  [MCP] ✓ MCP client initialized successfully")
        return True
    except asyncio.TimeoutError:
        print(f"  [MCP] ❌ Initialization timed out after {MCP_CONNECTION_TIMEOUT + MCP_INITIALIZATION_TIMEOUT + 5.0}s")
        print("  [MCP] 💡 First connection can take longer while downloading the MCP server package")
        print("  [MCP] 💡 Try increasing MCP_CONNECTION_TIMEOUT if you have a slow connection")
        print("  [MCP] ℹ Will use direct REST API for GitHub operations")
        _mcp_connection_failed = True
        _mcp_client = None
        return False
    except Exception as e:
        print(f"  [MCP] ❌ Failed to initialize: {e}")
        print("  [MCP] ℹ Will use direct REST API for GitHub operations")
        _mcp_connection_failed = True
        _mcp_client = None
        return False


async def get_mcp_client() -> Optional[GitHubMCPClient]:
    """
    Get the global MCP client instance.
    
    Returns:
        Optional[GitHubMCPClient]: The initialized MCP client, or None if unavailable
    """
    global _mcp_client
    
    if _mcp_connection_failed:
        return None
    
    if _mcp_client is not None and _mcp_client._initialized:
        return _mcp_client
    
    if not _mcp_initialization_attempted:
        await initialize_mcp_client()
        return _mcp_client if _mcp_client and _mcp_client._initialized else None
    
    return None


# ==============================================
# GITHUB MCP TOOL FUNCTIONS
# ==============================================

def search_repositories_by_skills_mcp(skills: List[str], max_results: int = 10) -> Dict[str, Any]:
    """
    Search GitHub repositories by skills using MCP.
    
    Args:
        skills: List of skill keywords to search for
        max_results: Maximum number of results to return
        
    Returns:
        dict: Contains 'repositories' list with repo details
    """
    if not MCP_AVAILABLE:
        return _fallback_to_direct_api("search_repositories_by_skills", skills, max_results)
    
    async def _search():
        client = await get_mcp_client()
        if client is None:
            return _fallback_to_direct_api("search_repositories_by_skills", skills, max_results)
        
        # Build GitHub search query
        query_parts = []
        languages = [s for s in skills if s.lower() in _LANGUAGES]
        if languages:
            lang_query = " OR ".join([f"language:{lang.lower()}" for lang in languages[:3]])
            query_parts.append(f"({lang_query})")
        
        frameworks = [s for s in skills if s.lower() not in _LANGUAGES]
        if frameworks:
            topic_query = " OR ".join([f"topic:{fw.lower()}" for fw in frameworks[:3]])
            query_parts.append(f"({topic_query})")
        
        if not query_parts:
            query = " ".join(skills[:3])
        elif len(query_parts) > 1:
            query = " OR ".join(query_parts)
        else:
            query = query_parts[0]
        
        # Try common GitHub MCP tool names
        tool_names = ["search_repositories", "github_search_repositories"]
        
        for tool_name in tool_names:
            try:
                result = await client.call_tool(
                    tool_name,
                    {
                        "query": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": min(max_results, 30)
                    }
                )
                
                if result.get("success"):
                    items = result.get("content", [])
                    repositories = []
                    for item in items[:max_results]:
                        if isinstance(item, dict):
                            repo = {
                                "name": item.get("name", ""),
                                "full_name": item.get("full_name", ""),
                                "description": item.get("description", ""),
                                "html_url": item.get("html_url", ""),
                                "stars": item.get("stargazers_count", 0),
                                "forks": item.get("forks_count", 0),
                                "language": item.get("language", ""),
                                "topics": item.get("topics", []),
                                "owner": item.get("owner", {}).get("login", "") if isinstance(item.get("owner"), dict) else ""
                            }
                            repositories.append(repo)
                    
                    return {
                        "repositories": repositories,
                        "total_count": len(repositories),
                        "query": query,
                        "success": True
                    }
            except Exception:
                continue
        
        # If all tool names failed, fall back to direct API
        return _fallback_to_direct_api("search_repositories_by_skills", skills, max_results)
    
    try:
        return asyncio.run(_search())
    except Exception as e:
        print(f"  [MCP] ⚠ Exception in search: {e}")
        return _fallback_to_direct_api("search_repositories_by_skills", skills, max_results)


def fetch_user_profile_mcp(username: str) -> Dict[str, Any]:
    """
    Fetch user profile using MCP.
    
    Args:
        username: GitHub username
        
    Returns:
        dict: User profile data
    """
    if not MCP_AVAILABLE:
        return _fallback_to_direct_api("fetch_user_profile", username)
    
    async def _fetch():
        client = await get_mcp_client()
        if client is None:
            return _fallback_to_direct_api("fetch_user_profile", username)
        
        # Try common GitHub MCP tool names
        tool_names = ["get_user", "github_get_user", "fetch_user"]
        
        for tool_name in tool_names:
            try:
                result = await client.call_tool(tool_name, {"username": username})
                
                if result.get("success"):
                    data = result.get("content", {})
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    
                    if isinstance(data, dict):
                        return {
                            "username": data.get("login", ""),
                            "name": data.get("name", ""),
                            "bio": data.get("bio", ""),
                            "html_url": data.get("html_url", ""),
                            "avatar_url": data.get("avatar_url", ""),
                            "public_repos": data.get("public_repos", 0),
                            "followers": data.get("followers", 0),
                            "following": data.get("following", 0),
                            "location": data.get("location", ""),
                            "company": data.get("company", ""),
                            "blog": data.get("blog", ""),
                            "hireable": data.get("hireable", None),
                            "created_at": data.get("created_at", ""),
                            "type": data.get("type", "User"),
                            "success": True
                        }
            except Exception:
                continue
        
        return _fallback_to_direct_api("fetch_user_profile", username)
    
    try:
        return asyncio.run(_fetch())
    except Exception as e:
        print(f"  [MCP] ⚠ Exception in fetch_user_profile: {e}")
        return _fallback_to_direct_api("fetch_user_profile", username)


def fetch_user_repos_mcp(username: str, max_repos: int = 10) -> Dict[str, Any]:
    """
    Fetch user repositories using MCP.
    
    Args:
        username: GitHub username
        max_repos: Maximum number of repos to return
        
    Returns:
        dict: Contains 'repositories' list
    """
    if not MCP_AVAILABLE:
        return _fallback_to_direct_api("fetch_user_repos", username, max_repos)
    
    async def _fetch():
        client = await get_mcp_client()
        if client is None:
            return _fallback_to_direct_api("fetch_user_repos", username, max_repos)
        
        # Try common GitHub MCP tool names
        tool_names = ["list_repositories", "get_user_repositories", "github_list_repositories"]
        
        for tool_name in tool_names:
            try:
                result = await client.call_tool(
                    tool_name,
                    {
                        "username": username,
                        "sort": "stars",
                        "direction": "desc",
                        "per_page": min(max_repos, 100),
                        "type": "owner"
                    }
                )
                
                if result.get("success"):
                    items = result.get("content", [])
                    repositories = []
                    for item in items[:max_repos]:
                        if isinstance(item, dict):
                            repo = {
                                "name": item.get("name", ""),
                                "full_name": item.get("full_name", ""),
                                "description": item.get("description", ""),
                                "html_url": item.get("html_url", ""),
                                "stars": item.get("stargazers_count", 0),
                                "forks": item.get("forks_count", 0),
                                "language": item.get("language", ""),
                                "topics": item.get("topics", []),
                                "created_at": item.get("created_at", ""),
                                "updated_at": item.get("updated_at", "")
                            }
                            repositories.append(repo)
                    
                    return {
                        "username": username,
                        "repositories": repositories,
                        "total_count": len(repositories),
                        "success": True
                    }
            except Exception:
                continue
        
        return _fallback_to_direct_api("fetch_user_repos", username, max_repos)
    
    try:
        return asyncio.run(_fetch())
    except Exception as e:
        print(f"  [MCP] ⚠ Exception in fetch_user_repos: {e}")
        return _fallback_to_direct_api("fetch_user_repos", username, max_repos)


def search_users_by_skills_mcp(skills: List[str], max_results: int = 10) -> Dict[str, Any]:
    """
    Search for GitHub users by skills using MCP.
    
    Args:
        skills: List of skill keywords
        max_results: Maximum number of users to return
        
    Returns:
        dict: Contains 'users' list
    """
    if not MCP_AVAILABLE:
        return _fallback_to_direct_api("search_users_by_skills", skills, max_results)
    
    async def _search():
        client = await get_mcp_client()
        if client is None:
            return _fallback_to_direct_api("search_users_by_skills", skills, max_results)
        
        # Build GitHub search query
        languages = [s for s in skills if s.lower() in _LANGUAGES]
        query_parts = []
        if languages:
            lang_query = " OR ".join([f"language:{lang.lower()}" for lang in languages[:2]])
            query_parts.append(f"({lang_query})")
        
        if query_parts:
            query = f"{' '.join(query_parts)} repos:>5 type:User"
        else:
            query = " ".join(skills[:2]) + " repos:>5 type:User"
        
        # Try common GitHub MCP tool names
        tool_names = ["search_users", "github_search_users"]
        
        for tool_name in tool_names:
            try:
                result = await client.call_tool(
                    tool_name,
                    {
                        "query": query,
                        "sort": "followers",
                        "order": "desc",
                        "per_page": min(max_results, 30)
                    }
                )
                
                if result.get("success"):
                    items = result.get("content", [])
                    users = []
                    for item in items[:max_results]:
                        if isinstance(item, dict):
                            user_type = item.get("type", "User")
                            if user_type != "User":
                                continue
                            
                            username = item.get("login", "")
                            if _is_organization_or_company(username):
                                continue
                            
                            users.append({
                                "username": username,
                                "html_url": item.get("html_url", ""),
                                "avatar_url": item.get("avatar_url", ""),
                                "type": user_type
                            })
                    
                    return {
                        "users": users,
                        "total_count": len(users),
                        "query": query,
                        "success": True
                    }
            except Exception:
                continue
        
        return _fallback_to_direct_api("search_users_by_skills", skills, max_results)
    
    try:
        return asyncio.run(_search())
    except Exception as e:
        print(f"  [MCP] ⚠ Exception in search_users: {e}")
        return _fallback_to_direct_api("search_users_by_skills", skills, max_results)


# ==============================================
# FALLBACK TO DIRECT API
# ==============================================

def _fallback_to_direct_api(function_name: str, *args, **kwargs) -> Dict[str, Any]:
    """
    Fallback to direct REST API when MCP is unavailable.
    
    Args:
        function_name: Name of the function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result from direct API
    """
    print(f"  [MCP] ⚠ Falling back to direct REST API for {function_name}")
    try:
        from tools.github_search import (
            search_repositories_by_skills,
            fetch_user_profile,
            fetch_user_repos,
            search_users_by_skills
        )
        
        if function_name == "search_repositories_by_skills":
            return search_repositories_by_skills(*args, **kwargs)
        elif function_name == "fetch_user_profile":
            return fetch_user_profile(*args, **kwargs)
        elif function_name == "fetch_user_repos":
            return fetch_user_repos(*args, **kwargs)
        elif function_name == "search_users_by_skills":
            return search_users_by_skills(*args, **kwargs)
        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}
    except ImportError as e:
        return {"success": False, "error": f"Direct API fallback failed: {e}"}


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

