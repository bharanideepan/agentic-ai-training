import asyncio
import json
import os
from typing import List, Dict, Any

import requests

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
SERVER_COMMAND = "npx"
SERVER_ARGS = ["-y", "@modelcontextprotocol/server-github"]

GITHUB_API_BASE = "https://api.github.com"


def build_user_query(skills: List[str]) -> str:
    """
    Build a GitHub user search query from a list of skills.

    - Matches skills in login, full name, bio, or email (partial match).
    - ORs between skills to be permissive.
    - Adds a mild followers threshold to bias toward active users.
    """
    # For each skill, look for it in login, name, bio, or email
    # Example piece: (python in:login in:name in:bio in:email)
    skill_chunks = []
    for s in skills:
        s = s.strip()
        if not s:
            continue
        skill_chunks.append(f'language:{s}')

    # Join with OR so any skill match is accepted
    skills_expr = " OR ".join(skill_chunks) if skill_chunks else ""

    # Add a soft followers filter so we don't get empty or very low-signal accounts
    # You can tune this, or remove it entirely:
    # followers_filter = "followers:>50"

    # if skills_expr:
    #     q = f"{skills_expr} {followers_filter}"
    # else:
    #     q = followers_filter

    q = skills_expr
    return q


async def search_users_by_skills_mcp(skills: List[str], max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Use GitHub MCP `search_users` to get candidate users by skill query.
    Returns a list of dicts: {login, html_url, score}.
    """
    q = build_user_query(skills)
    print("q:", q)
    print("GITHUB_TOKEN:", GITHUB_TOKEN)
    server_params = StdioServerParameters(
        command=SERVER_COMMAND,
        args=SERVER_ARGS,
        env={
            "GITHUB_TOKEN": GITHUB_TOKEN or "",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Optional: confirm search_users exists
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            if "search_users" not in tool_names:
                raise RuntimeError(f"search_users tool not found. Tools: {tool_names}")

            result = await session.call_tool(
                "search_users",
                {
                    "q": q,
                    "sort": "followers",   # followers | repositories | joined
                    "order": "desc",
                    "per_page": max_count,
                    "page": 1,
                },
            )
            print("result:", result)
            users = []

            for item in result.content:
                # Server returns text content containing JSON
                if item.type == "text":
                    data = json.loads(item.text)
                    for u in data.get("items", [])[:max_count]:
                        users.append(
                            {
                                "login": u.get("login"),
                                "html_url": u.get("html_url"),
                                "type": u.get("type"),
                                "score": u.get("score"),
                            }
                        )

            return users[:max_count]


def fetch_user_profile(login: str) -> Dict[str, Any]:
    """
    Use GitHub REST API to fetch user profile details:
    followers, public_repos, etc.
    """
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    resp = requests.get(f"{GITHUB_API_BASE}/users/{login}", headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return {
        "login": data.get("login"),
        "name": data.get("name"),
        "bio": data.get("bio"),
        "html_url": data.get("html_url"),
        "followers": data.get("followers"),
        "public_repos": data.get("public_repos"),
    }


def fetch_user_repos_and_stars(login: str, max_repos: int = 100) -> Dict[str, Any]:
    """
    Fetch user repos and compute:
    - total_stars: sum of stargazers_count
    - skills: set of languages used across repos
    """
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    page = 1
    per_page = 100  # GitHub max per page
    total_stars = 0
    languages = set()

    while len(languages) < max_repos:
        resp = requests.get(
            f"{GITHUB_API_BASE}/users/{login}/repos",
            headers=headers,
            params={"page": page, "per_page": per_page, "sort": "updated"},
            timeout=15,
        )
        resp.raise_for_status()
        repos = resp.json()
        if not repos:
            break

        for r in repos:
            total_stars += r.get("stargazers_count", 0)
            lang = r.get("language")
            if lang:
                languages.add(lang)

        if len(repos) < per_page:
            break
        page += 1

    return {
        "total_stars": total_stars,
        "skills": sorted(languages),
    }


async def find_top_users_by_skills(skills: List[str], max_count: int = 10) -> List[Dict[str, Any]]:
    """
    High-level helper:
    1. Search users by skills using MCP.
    2. For each user, fetch profile + repo stats.
    3. Sort by total_stars desc and return top N.
    """
    # 1) search candidates
    candidates = await search_users_by_skills_mcp(skills, max_count=max_count)

    enriched: List[Dict[str, Any]] = []

    for u in candidates:
        login = u["login"]
        profile = fetch_user_profile(login)
        repo_stats = fetch_user_repos_and_stars(login)

        enriched.append(
            {
                "login": login,
                "name": profile["name"],
                "bio": profile["bio"],
                "html_url": profile["html_url"],
                "followers": profile["followers"],
                "public_repos": profile["public_repos"],
                "total_stars": repo_stats["total_stars"],
                "skills": repo_stats["skills"],
            }
        )

    # 3) sort by stars desc locally
    enriched.sort(key=lambda x: x["total_stars"], reverse=True)

    # 4) cap to max_count
    return enriched[:max_count]


async def main():
    # Example list of skills (exact or partial)
    skills = ["python"]

    users = await find_top_users_by_skills(skills, max_count=10)

    print(f"Top {len(users)} users for skills {skills}:")
    for u in users:
        print(
            f"- {u['login']} ({u['html_url']})\n"
            f"  Name: {u['name']}\n"
            f"  Followers: {u['followers']}, Repos: {u['public_repos']}, Stars: {u['total_stars']}\n"
            f"  Skills: {', '.join(u['skills'])}\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
