"""
GitHub Search Tool Functions
============================

This module provides tool functions for searching GitHub repositories
and fetching user profiles. These functions are designed to be called
by the GitHubSearchAgent in the agentic workflow.

Functions:
    - search_repositories_by_skills: Search GitHub repos by skill keywords
    - fetch_user_profile: Get detailed user profile information
    - fetch_user_repos: Get repositories for a specific user

Environment Variables Required:
    - GITHUB_TOKEN: Personal access token for GitHub API
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================
# CONFIGURATION
# ==============================================

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Default headers for GitHub API requests
def get_headers() -> dict:
    """Get headers for GitHub API requests with authentication."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


# ==============================================
# DATA CLASSES FOR TYPE SAFETY
# ==============================================

@dataclass
class Repository:
    """Represents a GitHub repository."""
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    stars: int
    forks: int
    language: Optional[str]
    topics: list[str] = field(default_factory=list)
    owner: str = ""
    

@dataclass
class UserProfile:
    """Represents a GitHub user profile."""
    username: str
    name: Optional[str]
    bio: Optional[str]
    html_url: str
    avatar_url: str
    public_repos: int
    followers: int
    following: int
    location: Optional[str]
    company: Optional[str]
    blog: Optional[str]
    hireable: Optional[bool]


@dataclass
class SearchResult:
    """Container for search results."""
    repositories: list[Repository] = field(default_factory=list)
    users: list[UserProfile] = field(default_factory=list)
    total_count: int = 0
    query: str = ""


# ==============================================
# TOOL FUNCTIONS
# ==============================================

def is_individual_developer(username: str, profile: dict = None) -> bool:
    """
    Check if a GitHub account belongs to an individual developer (not organization).
    
    Args:
        username: GitHub username
        profile: Optional profile dict (if already fetched)
        
    Returns:
        bool: True if individual developer, False if organization/company
    """
    # Check known organizations
    if _is_organization_or_company(username):
        return False
    
    # If profile provided, check type
    if profile:
        if profile.get("type") != "User" or profile.get("is_organization"):
            return False
    
    return True


def search_repositories_by_skills(skills: list[str], max_results: int = 10) -> dict:
    """
    Search GitHub repositories by skill keywords.
    
    This function queries the GitHub Search API to find repositories
    that match the given skills/technologies. Results are sorted by
    stars to surface the most popular and relevant projects.
    
    Args:
        skills: List of skill keywords to search for (e.g., ["python", "django", "postgresql"])
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        dict: Contains 'repositories' list with repo details, 'total_count', and 'query'
    
    Example:
        >>> results = search_repositories_by_skills(["python", "machine-learning"])
        >>> print(results['repositories'][0]['name'])
        'scikit-learn'
    """
    # Build enhanced search query with multiple strategies
    query_parts = []
    
    # Strategy 1: Language-based search for programming languages
    languages = [s for s in skills if s.lower() in _LANGUAGES]
    if languages:
        lang_query = " OR ".join([f"language:{lang.lower()}" for lang in languages[:3]])
        query_parts.append(f"({lang_query})")
    
    # Strategy 2: Topic-based search for frameworks/tools
    frameworks = [s for s in skills if s.lower() not in _LANGUAGES]
    if frameworks:
        topic_query = " OR ".join([f"topic:{fw.lower()}" for fw in frameworks[:3]])
        query_parts.append(f"({topic_query})")
    
    # Combine with OR if we have both types
    if len(query_parts) > 1:
        query = " OR ".join(query_parts)
    elif query_parts:
        query = query_parts[0]
    else:
        # Fallback: simple keyword search
        query = " ".join(skills[:3])
    
    # Make API request
    url = f"{GITHUB_API_BASE}/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(max_results, 30)  # GitHub API max is 30 per page
    }
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse repositories
        repositories = []
        for item in data.get("items", [])[:max_results]:
            repo = {
                "name": item.get("name", ""),
                "full_name": item.get("full_name", ""),
                "description": item.get("description", ""),
                "html_url": item.get("html_url", ""),
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", ""),
                "topics": item.get("topics", []),
                "owner": item.get("owner", {}).get("login", "")
            }
            repositories.append(repo)
        
        return {
            "repositories": repositories,
            "total_count": data.get("total_count", 0),
            "query": query,
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "repositories": [],
            "total_count": 0,
            "query": query,
            "success": False,
            "error": str(e)
        }


def fetch_user_profile(username: str) -> dict:
    """
    Fetch detailed profile information for a GitHub user.
    
    Retrieves comprehensive profile data including bio, location,
    company, follower counts, and more.
    
    Args:
        username: GitHub username to fetch profile for
    
    Returns:
        dict: User profile data or error information
    
    Example:
        >>> profile = fetch_user_profile("torvalds")
        >>> print(profile['name'])
        'Linus Torvalds'
    """
    url = f"{GITHUB_API_BASE}/users/{username}"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check if this is an organization (not a person)
        user_type = data.get("type", "User")
        if user_type != "User":
            return {
                "success": False,
                "error": f"Not a user account (type: {user_type})",
                "is_organization": True
            }
        
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
            "type": user_type,
            "success": True
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"success": False, "error": f"User '{username}' not found"}
        return {"success": False, "error": str(e)}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def fetch_user_repos(username: str, max_repos: int = 10) -> dict:
    """
    Fetch repositories for a specific GitHub user.
    
    Retrieves the user's public repositories, sorted by stars
    to highlight their most popular projects.
    
    Args:
        username: GitHub username to fetch repos for
        max_repos: Maximum number of repos to return (default: 10)
    
    Returns:
        dict: Contains 'repositories' list with repo details
    
    Example:
        >>> repos = fetch_user_repos("torvalds", max_repos=5)
        >>> print(repos['repositories'][0]['name'])
        'linux'
    """
    url = f"{GITHUB_API_BASE}/users/{username}/repos"
    params = {
        "sort": "stars",
        "direction": "desc",
        "per_page": min(max_repos, 100),
        "type": "owner"  # Only repos owned by user, not forks
    }
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        repositories = []
        for item in data[:max_repos]:
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
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"success": False, "error": f"User '{username}' not found"}
        return {"success": False, "error": str(e)}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def search_users_by_skills(skills: list[str], max_results: int = 10) -> dict:
    """
    Search for GitHub users who have repositories matching the given skills.
    
    This function searches for users based on their programming languages
    and repository topics.
    
    Args:
        skills: List of skill keywords to search for
        max_results: Maximum number of users to return
    
    Returns:
        dict: Contains 'users' list with user profile summaries
    """
    # Build enhanced search query for users
    # GitHub user search works best with language: and repos: qualifiers
    query_parts = []
    
    # Use language qualifier for programming languages
    languages = [s for s in skills if s.lower() in _LANGUAGES]
    if languages:
        # Search for users who have repos in these languages
        lang_query = " OR ".join([f"language:{lang.lower()}" for lang in languages[:2]])
        query_parts.append(f"({lang_query})")
    
    # Add repos qualifier to ensure they have repositories
    # Add type:User to filter out organizations
    if query_parts:
        query = f"{' '.join(query_parts)} repos:>5 type:User"  # Only individual users
    else:
        # Fallback: simple keyword search
        query = " ".join(skills[:2]) + " repos:>5 type:User"
    url = f"{GITHUB_API_BASE}/search/users"
    params = {
        "q": query,
        "sort": "followers",
        "order": "desc",
        "per_page": min(max_results, 30)
    }
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        users = []
        for item in data.get("items", [])[:max_results]:
            # Only include actual users, not organizations
            user_type = item.get("type", "User")
            if user_type != "User":
                continue
            
            username = item.get("login", "")
            # Skip known organizations/companies/libraries
            if _is_organization_or_company(username):
                continue
            
            user = {
                "username": username,
                "html_url": item.get("html_url", ""),
                "avatar_url": item.get("avatar_url", ""),
                "type": user_type
            }
            users.append(user)
        
        return {
            "users": users,
            "total_count": data.get("total_count", 0),
            "query": query,
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "users": [],
            "total_count": 0,
            "query": query,
            "success": False,
            "error": str(e)
        }


# ==============================================
# HELPER CONSTANTS
# ==============================================

# Common programming languages recognized by GitHub
_LANGUAGES = {
    "python", "javascript", "typescript", "java", "c", "c++", "cpp",
    "c#", "csharp", "go", "rust", "ruby", "php", "swift", "kotlin",
    "scala", "r", "matlab", "julia", "perl", "haskell", "elixir",
    "clojure", "dart", "lua", "shell", "bash", "powershell", "sql",
    "html", "css", "sass", "less", "vue", "react", "angular"
}

# Known organizations, companies, and libraries to exclude
_KNOWN_ORGANIZATIONS = {
    # Tech Companies
    "microsoft", "google", "facebook", "meta", "apple", "amazon", "netflix",
    "twitter", "x", "linkedin", "github", "gitlab", "bitbucket", "atlassian",
    "adobe", "salesforce", "oracle", "ibm", "intel", "nvidia", "amd",
    "uber", "airbnb", "spotify", "stripe", "paypal", "square",
    
    # Open Source Organizations
    "apache", "eclipse", "mozilla", "linux", "kubernetes", "docker",
    "tensorflow", "pytorch", "opencv", "scikit-learn",
    "django", "rails", "spring", "angular", "vuejs", "reactjs",
    "nodejs", "expressjs", "nestjs", "fastapi", "flask",
    
    # Well-known Libraries/Frameworks
    "jquery", "lodash", "underscore", "moment", "axios", "request",
    "bootstrap", "foundation", "material-ui", "ant-design",
    "webpack", "rollup", "vite", "parcel", "babel",
    
    # Other Common Organizations
    "googleapis", "aws", "azure", "gcp",
    "helm", "istio", "prometheus", "grafana",
    "elastic", "mongodb", "redis", "postgresql", "mysql",
    "nginx", "traefik", "envoy",
    
    # Popular Developer Tools
    "ansible", "terraform", "vagrant", "packer", "consul",
    "vault", "nomad", "nomadproject",
}

def _is_organization_or_company(username: str) -> bool:
    """
    Check if a username appears to be an organization, company, or library.
    
    Args:
        username: GitHub username to check
        
    Returns:
        bool: True if appears to be an organization/company
    """
    username_lower = username.lower()
    
    # Check against known organizations
    if username_lower in _KNOWN_ORGANIZATIONS:
        return True
    
    # Check for common organization patterns
    org_patterns = [
        username_lower.endswith("-org"),
        username_lower.endswith("-team"),
        username_lower.endswith("-labs"),
        username_lower.endswith("-inc"),
        username_lower.endswith("-llc"),
        username_lower.endswith("-corp"),
        username_lower.startswith("team-"),
        username_lower.startswith("org-"),
        "-" in username_lower and len(username_lower.split("-")) > 2,  # Multi-word orgs
    ]
    
    # If multiple patterns match, likely an organization
    if sum(org_patterns) >= 2:
        return True
    
    return False


# ==============================================
# TOOL CLASS FOR AUTOGEN INTEGRATION
# ==============================================

class GitHubSearchTool:
    """
    A wrapper class that provides GitHub search tools for AutoGen agents.
    
    This class encapsulates all GitHub search functionality and provides
    methods that can be registered as tools with AutoGen agents.
    """
    
    def __init__(self):
        """Initialize the GitHub search tool."""
        self.api_base = GITHUB_API_BASE
        
    def search_repos(self, skills: list[str], max_results: int = 10) -> dict:
        """Search repositories by skills."""
        return search_repositories_by_skills(skills, max_results)
    
    def get_profile(self, username: str) -> dict:
        """Fetch user profile."""
        return fetch_user_profile(username)
    
    def get_user_repos(self, username: str, max_repos: int = 10) -> dict:
        """Fetch user repositories."""
        return fetch_user_repos(username, max_repos)
    
    def search_developers(self, skills: list[str], max_results: int = 10) -> dict:
        """Search for developers by skills."""
        return search_users_by_skills(skills, max_results)


# ==============================================
# MODULE TEST
# ==============================================

if __name__ == "__main__":
    # Simple test to verify the module works
    print("Testing GitHub Search Tools...")
    print("-" * 50)
    
    # Test repository search
    print("\n1. Testing search_repositories_by_skills:")
    results = search_repositories_by_skills(["python", "machine-learning"], max_results=3)
    if results["success"]:
        print(f"   Found {results['total_count']} total repos")
        for repo in results["repositories"]:
            print(f"   - {repo['full_name']}: ⭐ {repo['stars']}")
    else:
        print(f"   Error: {results.get('error', 'Unknown error')}")
    
    # Test user profile fetch
    print("\n2. Testing fetch_user_profile:")
    profile = fetch_user_profile("torvalds")
    if profile["success"]:
        print(f"   Name: {profile['name']}")
        print(f"   Followers: {profile['followers']}")
    else:
        print(f"   Error: {profile.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 50)
    print("Tests complete!")

