import asyncio
import os

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
SERVER_COMMAND = "npx"
SERVER_ARGS = ["-y", "@modelcontextprotocol/server-github"]


async def search_github_users_by_skills(skills: list[str], per_page: int = 10):
    skills_query = " ".join(skills)
    query = f"{skills_query} in:bio in:fullname"

    server_params = StdioServerParameters(
        command=SERVER_COMMAND,
        args=SERVER_ARGS,
        env={
            "GITHUB_TOKEN": GITHUB_TOKEN or "",
        },
    )

    # 1) Open stdio transport (spawns `npx -y @modelcontextprotocol/server-github`)
    async with stdio_client(server_params) as (read, write):
        # 2) Open MCP client session on that transport
        async with ClientSession(read, write) as session:
            # 3) Initialize the session
            await session.initialize()

            # Optional: list tools
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            print("Tools from server:", tool_names)

            if "search_users" not in tool_names:
                raise RuntimeError("search_users tool not found")

            # 4) Call the tool by name with arguments (newer SDK lets you pass name & args directly)
            result = await session.call_tool(
                "search_users",
                {
                    "q": query,
                    "page": 1,
                    "per_page": per_page,
                },
            )

            users = []
            for item in result.content:
                if item.type == "json" and isinstance(item.data, dict):
                    for u in item.data.get("items", []):
                        users.append(
                            {
                                "login": u.get("login"),
                                "html_url": u.get("html_url"),
                                "score": u.get("score"),
                            }
                        )
            return users


async def main():
    skills = ["python"]
    users = await search_github_users_by_skills(skills, per_page=5)
    print(f"Found {len(users)} users")
    for u in users:
        print(f"- {u['login']} ({u['html_url']}) score={u['score']}")


if __name__ == "__main__":
    asyncio.run(main())
