# GitHub MCP Tool Limitations

## Available Tools

The GitHub MCP server (`@modelcontextprotocol/server-github`) provides these tools:

✅ **Available:**

- `search_users` - Search for GitHub users by query
- `search_repositories` - Search for GitHub repositories by query
- `search_code` - Search for code across repositories
- `search_issues` - Search for issues and pull requests
- And 22 other tools (see `list_mcp_tools.py`)

❌ **NOT Available:**

- `get_user` - Get detailed user profile (NOT in MCP server)
- `get_user_repositories` - Get user's repositories (NOT in MCP server)
- `get_repository` - Get detailed repository info (NOT in MCP server)

## Workarounds Implemented

### 1. `fetch_user_profile_mcp(username)`

**Original Plan:** Use `get_user` tool  
**Actual Implementation:** Uses `search_users` with `user:username` query

**Limitations:**

- Only returns basic info: `login`, `html_url`, `avatar_url`, `type`
- Does NOT return: `bio`, `location`, `company`, `blog`, `followers`, `following`, `public_repos`, `hireable`, `created_at`
- Returns a note: "Limited profile data - full profile not available via MCP"

### 2. `fetch_user_repos_mcp(username)`

**Original Plan:** Use `get_user_repositories` tool  
**Actual Implementation:** Uses `search_repositories` with `user:username` query

**Limitations:**

- May not return all repositories (search has limits)
- Returns repos sorted by stars
- Returns a note: "Repositories found via search - may not include all repos"

## Impact on Application

The agent will work but with limited profile data:

- ✅ Can search for users by skills
- ✅ Can get basic user info (username, URL, avatar)
- ❌ Cannot get full profile details (bio, location, company)
- ⚠️ Repository data may be incomplete

## Recommendations

1. **For now:** The application will work with limited data
2. **Future:** Monitor for updated GitHub MCP server with `get_user` support
3. **Alternative:** Consider using GitHub REST API directly for profile/repo data (but violates MCP-only requirement)
