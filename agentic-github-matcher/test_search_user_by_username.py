#!/usr/bin/env python3
"""Test if search_users can get profile data for a specific user"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test_search_user():
    github_token = os.getenv("GITHUB_TOKEN")
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    env = {"GITHUB_TOKEN": github_token}
    
    server_params = StdioServerParameters(
        command=npx_cmd,
        args=["-y", "@modelcontextprotocol/server-github"],
        env=env,
    )
    
    # Test with a known username
    test_username = "torvalds"  # Linus Torvalds
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Try searching for the specific user
            print(f"Searching for user: {test_username}")
            result = await session.call_tool(
                "search_users",
                {
                    "q": f"user:{test_username}",
                    "per_page": 1,
                },
            )
            
            print("\nResult structure:")
            print(f"  Type: {type(result)}")
            print(f"  Has content: {hasattr(result, 'content')}")
            
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    print(f"\n  Item type: {type(item)}")
                    print(f"  Item has type attr: {hasattr(item, 'type')}")
                    if hasattr(item, 'type'):
                        print(f"  Item.type: {item.type}")
                    if hasattr(item, 'text'):
                        data = json.loads(item.text)
                        print(f"\n  Parsed data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        if isinstance(data, dict) and "items" in data:
                            if data["items"]:
                                user = data["items"][0]
                                print(f"\n  User data keys: {list(user.keys())}")
                                print(f"  Sample fields:")
                                for key in ["login", "html_url", "type", "score", "avatar_url", "followers_url", "repos_url"]:
                                    if key in user:
                                        print(f"    {key}: {user[key]}")

if __name__ == "__main__":
    asyncio.run(test_search_user())

