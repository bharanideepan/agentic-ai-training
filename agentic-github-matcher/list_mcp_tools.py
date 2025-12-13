#!/usr/bin/env python3
"""List available MCP tools from GitHub server"""

import asyncio
import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def list_tools():
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set")
        return
    
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    env = {"GITHUB_TOKEN": github_token}
    
    server_params = StdioServerParameters(
        command=npx_cmd,
        args=["-y", "@modelcontextprotocol/server-github"],
        env=env,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools_result = await session.list_tools()
            print(f"\nAvailable MCP Tools ({len(tools_result.tools)}):")
            print("=" * 60)
            for tool in tools_result.tools:
                print(f"  - {tool.name}")
                if hasattr(tool, 'description') and tool.description:
                    print(f"    {tool.description[:80]}...")
            print("=" * 60)
            
            # Check for user-related tools
            user_tools = [t.name for t in tools_result.tools if 'user' in t.name.lower()]
            print(f"\nUser-related tools: {user_tools}")

if __name__ == "__main__":
    asyncio.run(list_tools())

