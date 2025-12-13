"""
GitHub Search Agent - MCP-Only Implementation
==============================================

This module implements the GitHubSearchAgent, responsible for:
- Searching GitHub repositories based on extracted skills
- Finding relevant developer profiles
- Fetching detailed repository and user information

The agent uses GitHub MCP (Model Context Protocol) exclusively for all
GitHub operations. NO REST API calls are made.

MCP Integration:
- All GitHub operations go through @modelcontextprotocol/server-github
- MCP tools are discovered dynamically at runtime
- Session management handled via mcp/github_session.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass, field

# AutoGen imports (pyautogen 0.2.x classic API)
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent

# LiteLLM for model gateway
import litellm

# For enriching finalized candidates with profile/repo data
import requests

# Import GitHub MCP tools (MCP-only, no REST fallback)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.github_mcp import (
    search_users_by_skills_mcp as search_users_by_skills,
    _is_organization_or_company,
    _LANGUAGES
)


# ==============================================
# DATA STRUCTURES
# ==============================================

@dataclass
class DeveloperMatch:
    """Represents a matched developer profile."""
    username: str
    name: str
    html_url: str
    bio: str = ""
    location: str = ""
    followers: int = 0
    public_repos: int = 0
    top_repositories: list[dict] = field(default_factory=list)
    matching_skills: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    skill_match_percentage: float = 0.0  # Percentage of required skills matched
    is_exact_match: bool = False  # True if matches all or most required skills


@dataclass  
class SearchResults:
    """Container for all search results."""
    query_skills: list[str] = field(default_factory=list)
    repositories: list[dict] = field(default_factory=list)
    developers: list[DeveloperMatch] = field(default_factory=list)
    total_repos_found: int = 0
    total_developers_found: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_skills": self.query_skills,
            "repositories": self.repositories,
            "developers": [
                {
                    "username": d.username,
                    "name": d.name,
                    "html_url": d.html_url,
                    "bio": d.bio,
                    "location": d.location,
                    "followers": d.followers,
                    "public_repos": d.public_repos,
                    "top_repositories": d.top_repositories,
                    "matching_skills": d.matching_skills,
                    "relevance_score": d.relevance_score,
                    "skill_match_percentage": d.skill_match_percentage,
                    "is_exact_match": d.is_exact_match
                }
                for d in self.developers
            ],
            "total_repos_found": self.total_repos_found,
            "total_developers_found": self.total_developers_found
        }


# ==============================================
# GITHUB AGENT SYSTEM PROMPT
# ==============================================

GITHUB_AGENT_SYSTEM_PROMPT = """You are a GitHub Search Agent specialized in finding developers matching job requirements.

Your role is to:
1. Take a list of required skills and technologies
2. Search for GitHub users/developers matching those skills using search_users_by_skills
3. Return the list of candidates found

You have access to the following tool (via GitHub MCP - Model Context Protocol):
- search_users_by_skills: Search for GitHub users/developers by skill keywords. This is your ONLY tool - use it to find candidates.

All GitHub operations use MCP (Model Context Protocol) exclusively - no REST API calls.

SIMPLE STRATEGY:
1. Use search_users_by_skills with max_results matching the requested candidate count
2. The tool returns users with: username, html_url, avatar_url, type, and score
3. Return the list of users found - they will be processed further

IMPORTANT:
- Use search_users_by_skills with max_results matching the requested candidate count
- You can make multiple calls with different skill combinations if needed
- Return all users found - provide a summary of candidates
"""


# ==============================================
# GITHUB SEARCH AGENT CLASS
# ==============================================

class GitHubSearchAgent:
    """
    The GitHubSearchAgent searches GitHub for developers and repositories
    matching the required skills extracted from job descriptions.
    
    This agent uses AutoGen's tool calling feature to interact with
    the GitHub API through defined tool functions.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.5,
        name: str = "GitHubSearchAgent"
    ):
        """
        Initialize the GitHubSearchAgent.
        
        Args:
            model: The LLM model to use (via LiteLLM)
            temperature: Temperature for generation
            name: Name identifier for the agent
        """
        self.model = model
        self.temperature = temperature
        self.name = name
        
        # Tool functions available to this agent
        # Using GitHub MCP exclusively (no REST API)
        # Only search_users_by_skills is available - no profile fetching needed
        self.tools = {
            "search_users_by_skills": search_users_by_skills,
        }
        
        print(f"  [GitHubSearchAgent] ✓ Using GitHub MCP for all GitHub operations")
        
        # Configure LiteLLM settings with tool definitions
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm_config = {
            "config_list": [
                {
                    "model": model,
                    "api_type": "openai",
                    "api_key": api_key,  # Explicitly pass API key
                    "temperature": temperature,
                }
            ],
            "timeout": 120,
            "tools": self._get_tool_definitions(),
        }
        
        # Log API key status
        if api_key:
            if api_key.startswith("sk-"):
                print(f"  [GitHubSearchAgent] ✓ API key format valid")
            else:
                print(f"  [GitHubSearchAgent] ⚠ API key format unusual: {api_key[:10]}...")
        else:
            print(f"  [GitHubSearchAgent] ⚠ No API key found")
        
        # Create the AutoGen agent
        self.agent = ConversableAgent(
            name=name,
            system_message=GITHUB_AGENT_SYSTEM_PROMPT,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,  # Increased to allow more tool call iterations
        )
        
        # Register tools with the agent for LLM and execution
        # This enables true agentic tool calling
        self._register_tools()
    
    def _get_tool_definitions(self) -> list[dict]:
        """
        Get OpenAI-compatible tool definitions.
        
        Returns:
            list: Tool definitions for the LLM
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_users_by_skills",
                    "description": "Search for GitHub users/developers by skill keywords. Returns users matching the skills (partial or full match).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skills": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of skill keywords to search for"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            }
                        },
                        "required": ["skills"]
                    }
                }
            },
        ]
    
    def _register_tools(self):
        """
        Register tool functions with the AutoGen agent for LLM tool calling.
        This enables the agent to decide which tools to call and when.
        
        NOTE: Only search_users_by_skills is available via MCP. No profile fetching needed.
        """
        # Register search_users_by_skills for both LLM (so agent knows about it) and execution (so it can be called)
        @self.agent.register_for_llm(description="Search for GitHub users/developers by skill keywords. Returns users matching the skills (partial or full match). This is the ONLY tool available - use it to find candidates.")
        @self.agent.register_for_execution()
        async def search_users_by_skills(skills: list[str], max_results: int = 10) -> dict:
            """Search for GitHub users/developers by skill keywords (async MCP call)."""
            print(f"  [AGENT TOOL CALL] 🤖 search_users_by_skills(skills={skills[:3]}, max_results={max_results})")
            # MCP functions are async - await them
            result = await self.tools["search_users_by_skills"](skills, max_results)
            print(f"  [AGENT TOOL CALL] ✓ search_users_by_skills returned {len(result.get('users', []))} users")
            return result
        
        print(f"  [GitHubSearchAgent] ✓ search_users_by_skills tool registered for agentic tool calling")
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Execute a tool function by name (fallback method).
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name in self.tools:
            return self.tools[tool_name](**arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _extract_tool_results_from_messages(self, messages: list) -> dict:
        """
        Extract tool call results from AutoGen agent messages.
        
        Args:
            messages: List of messages from agent conversation
            
        Returns:
            dict: Extracted results with 'users' list
        """
        extracted = {
            "users": []
        }
        
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            
            # AutoGen stores tool responses in 'tool_responses' or as separate tool role messages
            tool_responses = msg.get("tool_responses", [])
            
            # Also check if message itself is a tool response
            if msg.get("role") == "tool":
                tool_responses.append(msg)
            
            for tool_resp in tool_responses:
                content = tool_resp.get("content", "")
                
                # Try to parse content
                result = None
                if isinstance(content, str):
                    try:
                        result = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        # Content might already be a dict or other type
                        result = content
                elif isinstance(content, dict):
                    result = content
                else:
                    continue
                
                if not isinstance(result, dict):
                    continue
                
                # Identify tool by result structure - only search_users_by_skills is available
                # Check result keys to determine which tool was called
                has_users = "users" in result and isinstance(result.get("users"), list)
                
                # Extract users from search_users_by_skills results
                if has_users:
                    users = result.get("users", [])
                    if users:
                        extracted["users"].extend(users)
        
        return extracted
    
    async def search(
        self, 
        skills: list[str], 
        job_analysis: dict = None,
        max_candidates: int = 10,
        progress_callback: Callable[[str, float], None] = None
    ) -> SearchResults:
        """
        Search GitHub for developers and repositories matching the given skills.
        
        This method uses true agentic behavior - the LLM agent decides which tools
        to call and in what order to find the best candidates.
        
        Args:
            skills: List of skills to search for
            job_analysis: Optional job analysis dict for better matching
            max_candidates: Maximum number of candidates to return
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchResults: Compiled search results
        """
        results = SearchResults(query_skills=skills)
        
        print(f"  [GitHubSearchAgent] 🤖 Starting agentic search with {len(skills)} skills")
        print(f"  [GitHubSearchAgent] ✓ Using LLM agent for tool calling decisions")
        print(f"  [EXECUTION MODE] 🤖 AGENTIC MODE - Tools will be called by LLM agent")
        
        if progress_callback:
            progress_callback("Agent is analyzing requirements and planning tool calls...", 20)
        
        # Build the search prompt for the agent
        job_info = ""
        if job_analysis:
            job_title = job_analysis.get("title", "N/A")
            exp_level = job_analysis.get("experience_level", "N/A")
            job_info = f"\nJob Title: {job_title}\nExperience Level: {exp_level}\n"
        
        search_prompt = f"""Search GitHub for {max_candidates} developers matching these skills: {', '.join(skills)}.
        
{job_info}
Your task:
1. Call search_users_by_skills with max_results={max_candidates} to find candidates
2. You can make multiple calls with different skill combinations if needed to find enough candidates
3. Return the list of users found

CRITICAL REQUIREMENTS:
- Use search_users_by_skills - this is your ONLY available tool
- Request at least {max_candidates} users (you can request more to ensure you get enough after filtering)
- The tool returns: username, html_url, avatar_url, type, and score for each user
- After finding users, provide a summary of the candidates found"""
        
        try:
            # Clear agent history for fresh start
            self.agent.clear_history()
            
            if progress_callback:
                progress_callback("Agent is making tool calls to search GitHub...", 30)
            
            # Use agent to make tool calls - the agent will decide which tools to use
            # Create a simple user proxy to initiate the conversation
            print("  🤖 Agent is analyzing and making tool calls...")
            
            # Create a user proxy agent to initiate conversation
            # In AutoGen, when an agent makes tool calls, the UserProxyAgent executes them
            # max_consecutive_auto_reply must be > 0 to allow tool execution
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,  # Allow auto-reply to execute tool calls
                code_execution_config=False,  # Don't execute code, just tools
            )
            
            # Register tools with user_proxy for execution
            # When the agent suggests tool calls, user_proxy will execute them
            # NOTE: Since initiate_chat is sync, we need sync wrappers for async MCP tools
            # Run initiate_chat in a thread to avoid blocking the async event loop
            import concurrent.futures
            
            @user_proxy.register_for_execution()
            def search_users_by_skills(skills: list[str], max_results: int = 10) -> dict:
                # This is called from a sync context (initiate_chat thread)
                # Create a new event loop for this thread
                import asyncio
                try:
                    # Try to get the event loop for this thread
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(self.tools["search_users_by_skills"](skills, max_results))
                except RuntimeError:
                    # No event loop in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.tools["search_users_by_skills"](skills, max_results))
                    finally:
                        loop.close()
            
            
            # Initiate chat - the agent will make tool calls as needed
            # The user_proxy will execute the tool calls when the agent suggests them
            # Run initiate_chat in a thread since we're in an async context
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: user_proxy.initiate_chat(
                        recipient=self.agent,
                        message=search_prompt,
                        max_turns=10,  # Allow multiple tool call iterations
                        silent=False
                    )
                )
                chat_result = future.result(timeout=300)  # 5 minute timeout
            
            if progress_callback:
                progress_callback("Extracting results from agent's tool calls...", 60)
            
            # Extract tool results from agent's conversation history
            # AutoGen stores messages in chat_messages dict keyed by sender
            messages = []
            if hasattr(self.agent, 'chat_messages'):
                # Try different keys - might be keyed by user_proxy or agent
                if user_proxy in self.agent.chat_messages:
                    messages = self.agent.chat_messages[user_proxy]
                elif self.agent in self.agent.chat_messages:
                    messages = self.agent.chat_messages[self.agent]
                elif hasattr(self.agent.chat_messages, 'values'):
                    # Get all messages from all senders
                    all_messages = []
                    for msg_list in self.agent.chat_messages.values():
                        all_messages.extend(msg_list)
                    messages = all_messages
            
            if hasattr(self.agent, '_oai_messages'):
                # Also check _oai_messages
                if user_proxy in self.agent._oai_messages:
                    messages.extend(self.agent._oai_messages[user_proxy])
                elif self.agent in self.agent._oai_messages:
                    messages.extend(self.agent._oai_messages[self.agent])
            
            extracted = self._extract_tool_results_from_messages(messages)
            
            # Also try to extract from chat result if available
            if hasattr(chat_result, 'chat_history'):
                additional = self._extract_tool_results_from_messages(chat_result.chat_history)
                # Merge results
                extracted["users"].extend(additional["users"])
            
            # Also check user_proxy's message history
            if hasattr(user_proxy, 'chat_messages'):
                for sender, msg_list in user_proxy.chat_messages.items():
                    additional = self._extract_tool_results_from_messages(msg_list)
                    extracted["users"].extend(additional["users"])
            
            # Process extracted users (primary source - from search_users_by_skills)
            user_candidates = []
            if extracted["users"]:
                for user in extracted["users"]:
                    username = user.get("username", "")
                    if username:
                        user_candidates.append(username)
                print(f"  ✓ Found {len(user_candidates)} user candidates via agent search")
            
            # Remove duplicates while preserving order
            all_candidates = []
            seen = set()
            for username in user_candidates:
                if username and username not in seen:
                    all_candidates.append(username)
                    seen.add(username)
            
            # If we don't have enough candidates, we'll retry with less strict criteria
            if len(all_candidates) < max_candidates:
                print(f"  ⚠ Found {len(all_candidates)} candidates, need {max_candidates}. Will retry with expanded search if needed.")
            
            # Limit to max_candidates for processing
            candidates_to_process = all_candidates[:max_candidates]
            
            if progress_callback:
                progress_callback(f"Processing {len(candidates_to_process)} candidates...", 70)
            
            # Create developer objects from search results (no profile fetching needed)
            developers = []
            # Map usernames to user data from extracted results
            user_data_map = {u.get("username"): u for u in extracted["users"] if u.get("username")}
            
            for username in candidates_to_process:
                user_data = user_data_map.get(username)
                if not user_data:
                    continue
                
                # Skip organizations
                if _is_organization_or_company(username) or user_data.get("type") != "User":
                    continue
                
                # Create DeveloperMatch from search_users result data
                # search_users returns: username, html_url, avatar_url, type, score
                developer = DeveloperMatch(
                    username=username,
                    name=username,  # No name available from search_users
                    html_url=user_data.get("html_url", ""),
                    bio="",  # No bio available from search_users
                    location="",  # No location available from search_users
                    followers=0,  # No followers count available from search_users
                    public_repos=0,  # No repo count available from search_users
                    top_repositories=[],  # No repos available from search_users
                    matching_skills=[],  # Will be determined by LLM scoring
                    relevance_score=user_data.get("score", 0) * 10,  # Use search score as initial relevance
                    skill_match_percentage=0.0,  # Will be determined by LLM scoring
                    is_exact_match=False  # Will be determined by LLM scoring
                )
                developers.append(developer)
            
            # Always score candidates with LLM (batch scoring for efficiency)
            if developers:
                if progress_callback:
                    progress_callback("Scoring candidates with AI...", 80)
                
                # Use batch scoring if job_analysis available, otherwise use skill-based scoring
                if job_analysis:
                    developers = self._llm_batch_score_candidates(developers, skills, job_analysis)
                else:
                    # Fallback: use skill match percentage as relevance score
                    for dev in developers:
                        dev.relevance_score = max(10, dev.skill_match_percentage * 0.8)
            
            # Sort by relevance score (from LLM) first, then skill match percentage
            developers.sort(key=lambda d: (d.relevance_score, d.skill_match_percentage), reverse=True)
            
            # If we have fewer candidates than requested, try to get more
            if len(developers) < max_candidates and len(all_candidates) > len(developers):
                print(f"  ⚠ Only {len(developers)} valid candidates after filtering, requested {max_candidates}")
                # Try to process remaining candidates from the list
                remaining = [u for u in all_candidates[len(developers):] if u not in [d.username for d in developers]]
                if remaining:
                    print(f"  ↻ Processing {min(len(remaining), max_candidates - len(developers))} additional candidates...")
                    user_data_map = {u.get("username"): u for u in extracted["users"] if u.get("username")}
                    for username in remaining[:max_candidates - len(developers)]:
                        user_data = user_data_map.get(username)
                        if not user_data:
                            continue
                        
                        # Skip organizations
                        if _is_organization_or_company(username) or user_data.get("type") != "User":
                            continue
                        
                        try:
                            developer = DeveloperMatch(
                                username=username,
                                name=username,
                                html_url=user_data.get("html_url", ""),
                                bio="",
                                location="",
                                followers=0,
                                public_repos=0,
                                top_repositories=[],
                                matching_skills=[],
                                relevance_score=user_data.get("score", 0) * 10,
                                skill_match_percentage=0.0,
                                is_exact_match=False
                            )
                            developers.append(developer)
                        except Exception as e:
                            print(f"  ⚠ Error processing additional candidate {username}: {e}")
                            continue
                    
                    # Re-score all candidates if we added more
                    if job_analysis and len(developers) > 0:
                        developers = self._llm_batch_score_candidates(developers, skills, job_analysis)
                        developers.sort(key=lambda d: (d.relevance_score, d.skill_match_percentage), reverse=True)
            
            # Finalize candidates
            finalized_developers = developers[:max_candidates]
            
            # Enrich finalized candidates with profile and repo data
            if finalized_developers:
                if progress_callback:
                    progress_callback("Enriching finalized candidates with detailed data...", 70)
                finalized_developers = self._enrich_finalized_candidates(
                    finalized_developers,
                    skills,
                    progress_callback
                )
                
                # Score enriched candidates with LLM
                if job_analysis:
                    if progress_callback:
                        progress_callback("Scoring enriched candidates with AI...", 85)
                    finalized_developers = self._llm_batch_score_candidates(
                        finalized_developers,
                        skills,
                        job_analysis
                    )
                    # Re-sort after scoring
                    finalized_developers.sort(key=lambda d: (d.relevance_score, d.skill_match_percentage), reverse=True)
            
            results.developers = finalized_developers
            results.total_developers_found = len(results.developers)
            
            print(f"  ✓ Agentic search complete: {len(results.developers)} candidates found (requested {max_candidates})")
            return results
            
        except Exception as e:
            print(f"  ⚠ Agentic search encountered an error: {e}")
            print(f"  ⚠ Falling back to direct search method...")
            # Fallback to direct method if agent fails
            return await self._direct_search(skills, job_analysis, max_candidates, progress_callback)
    
    async def _direct_search(
        self,
        skills: list[str],
        job_analysis: dict = None,
        max_candidates: int = 10,
        progress_callback: Callable[[str, float], None] = None
    ) -> SearchResults:
        """
        Direct search method (fallback) - uses direct function calls.
        This is the original implementation kept as fallback.
        """
        results = SearchResults(query_skills=skills)
        
        # Step 1: Direct user search via MCP
        strategy_info = "via GitHub MCP"
        print(f"  [GitHubSearchAgent] Starting direct search with {len(skills)} skills")
        print(f"  [EXECUTION MODE] 📝 MCP MODE - Tools called via GitHub MCP")
        if progress_callback:
            progress_callback(f"Searching GitHub repositories ({strategy_info})...", 35)
        
        # Note: Repository search removed from direct search - focus on user search only
        # Repositories can be fetched via user profiles if needed
        
        # Deduplicate repositories by full_name
        seen_repos = set()
        unique_repos = []
        for repo in results.repositories:
            repo_id = repo.get("full_name", "")
            if repo_id and repo_id not in seen_repos:
                seen_repos.add(repo_id)
                unique_repos.append(repo)
        
        # Sort by stars and take top ones
        unique_repos.sort(key=lambda r: r.get("stars", 0), reverse=True)
        results.repositories = unique_repos[:30]
        results.total_repos_found = len(results.repositories)
        
        # Extract unique owners from top repositories (prioritize high-star repos)
        # Filter out organizations/companies
        repo_owners = []
        for repo in results.repositories[:20]:
            owner = repo.get("owner", "")
            if owner and owner not in repo_owners:
                # Skip known organizations and companies
                if not _is_organization_or_company(owner):
                    repo_owners.append(owner)
        
        # Step 2: Enhanced user search with agentic behavior
        if progress_callback:
            progress_callback(f"Searching for developers ({strategy_info})...", 40)
        print(f"  🔍 Searching for developers ({strategy_info})...")
        
        # Use agentic behavior: search with multiple strategies to ensure we get enough candidates
        # Key insight: Search by individual skills, not all skills together
        # This ensures we find candidates even if they don't have ALL skills
        user_candidates = []
        seen_candidates = set()
        
        # Strategy 1: Search by individual skills (more flexible than searching all at once)
        # Store full user data, not just usernames
        for skill in skills[:5]:  # Try top 5 skills individually
            if len(user_candidates) >= max_candidates:
                break
            print(f"  [MCP CALL] 📝 search_users_by_skills(skills=[{skill}], max_results={max_candidates})")
            skill_results = await search_users_by_skills([skill], max_results=max_candidates)
            if skill_results.get("success"):
                new_users = skill_results.get("users", [])
                print(f"  [MCP CALL] ✓ Found {len(new_users)} users for skill '{skill}'")
                for user in new_users:
                    username = user.get("username", "")
                    if username and username not in seen_candidates and len(user_candidates) < max_candidates:
                        user_candidates.append(user)  # Store full user object
                        seen_candidates.add(username)
        
        # Strategy 2: Also search by primary skills combination (for exact matches)
        if len(user_candidates) < max_candidates and len(skills) >= 2:
            primary_skills = skills[:3] if len(skills) >= 3 else skills
            print(f"  [MCP CALL] 📝 search_users_by_skills(skills={primary_skills}, max_results={max_candidates})")
            user_results = await search_users_by_skills(primary_skills, max_results=max_candidates)
            if user_results.get("success"):
                results.total_developers_found = user_results.get("total_count", 0)
                print(f"  [MCP CALL] ✓ Found {len(user_results.get('users', []))} users for combined skills")
                for user in user_results.get("users", []):
                    username = user.get("username", "")
                    if username and username not in seen_candidates and len(user_candidates) < max_candidates:
                        user_candidates.append(user)  # Store full user object
                        seen_candidates.add(username)
        
        # Combine and prioritize candidates (repo owners first, then direct search)
        # Convert repo_owners to user objects if needed, or just use user_candidates
        all_candidates_data = user_candidates[:max_candidates]
        
        # Step 3: Create DeveloperMatch objects from search results (no profile/repo fetching)
        if progress_callback:
            progress_callback(f"Processing {len(all_candidates_data)} candidates...", 45)
        print(f"  👤 Processing {len(all_candidates_data)} candidates from search results...")
        developers = []
        filtered_count = 0
        
        for idx, user_data in enumerate(all_candidates_data):
            if progress_callback:
                progress = 45 + int((idx / max(len(all_candidates_data), 1)) * 10)
                progress_callback(f"Processing candidate {idx + 1}/{len(all_candidates_data)}...", progress)
            
            if not isinstance(user_data, dict):
                continue
            
            username = user_data.get("username", "")
            if not username:
                continue
            
            # Skip organizations
            if _is_organization_or_company(username) or user_data.get("type") != "User":
                filtered_count += 1
                continue
            
            # Create DeveloperMatch from search_users result data
            # search_users returns: username, html_url, avatar_url, type, score
            developer = DeveloperMatch(
                username=username,
                name=username,  # No name available from search_users
                html_url=user_data.get("html_url", ""),
                bio="",  # No bio available from search_users
                location="",  # No location available from search_users
                followers=0,  # No followers count available from search_users
                public_repos=0,  # No repo count available from search_users
                top_repositories=[],  # No repos available from search_users
                matching_skills=[],  # Will be determined by LLM scoring
                relevance_score=user_data.get("score", 0) * 10,  # Use search score as initial relevance
                skill_match_percentage=0.0,  # Will be determined by LLM scoring
                is_exact_match=False  # Will be determined by LLM scoring
            )
            developers.append(developer)
        
        # Step 4: Batch score all candidates with a single LLM call (much faster)
        if developers and job_analysis:
            if progress_callback:
                progress_callback(f"Analyzing skills for {len(developers)} candidates...", 55)
            print(f"  🤖 Scoring {len(developers)} candidates with AI...")
            if progress_callback:
                progress_callback("Scoring candidates with AI...", 65)
            developers = self._llm_batch_score_candidates(
                developers,
                skills,
                job_analysis
            )
        
        # Step 5: Prioritize exact matches, then fall back to partial matches
        # Separate candidates into exact matches and partial matches
        exact_matches = [d for d in developers if d.is_exact_match]
        partial_matches = [d for d in developers if not d.is_exact_match]
        
        # Sort exact matches by skill match percentage (descending), then by relevance score
        exact_matches.sort(key=lambda d: (d.skill_match_percentage, d.relevance_score), reverse=True)
        
        # Sort partial matches by skill match percentage (descending), then by relevance score
        partial_matches.sort(key=lambda d: (d.skill_match_percentage, d.relevance_score), reverse=True)
        
        # Combine: exact matches first, then partial matches
        developers = exact_matches + partial_matches
        
        # Step 6: Final sorting (already scored, just sort by score)
        
        # Final sort: exact matches first (by skill match % and relevance), then partial matches
        developers = sorted(
            developers, 
            key=lambda d: (
                not d.is_exact_match,  # False (0) for exact matches comes first
                -d.skill_match_percentage,  # Higher percentage first
                -d.relevance_score  # Higher relevance first
            )
        )
        
        if filtered_count > 0:
            print(f"  ✓ Filtered out {filtered_count} organizations/companies")
        
        # Log match statistics
        if exact_matches:
            print(f"  ✓ Found {len(exact_matches)} exact match(es) (≥80% skills)")
        if partial_matches:
            print(f"  ℹ Found {len(partial_matches)} partial match(es) (<80% skills)")
        
        # Agentic behavior: If we don't have enough candidates, expand search
        if len(developers) < max_candidates:
            print(f"  ⚠ Only found {len(developers)} candidates, expanding search...")
            
            # Expand search: try searching with broader terms
            expanded_candidates = []
            
            # Try searching with individual skills (limited to max_candidates)
            existing_usernames = {d.username for d in developers}
            for skill in skills[:5]:  # Try top 5 skills
                if len(expanded_candidates) >= max_candidates:
                    break
                print(f"  [MCP CALL] 📝 search_users_by_skills(skills=[{skill}], max_results={max_candidates}) [expanded search]")
                skill_results = await search_users_by_skills([skill], max_results=max_candidates)
                if skill_results.get("success"):
                    new_users = skill_results.get("users", [])
                    for user in new_users:
                        username = user.get("username", "")
                        if username and username not in existing_usernames:
                            expanded_candidates.append(user)  # Store full user object
                            existing_usernames.add(username)
            
            # Process expanded candidates from search results (no profile fetching)
            for user_data in expanded_candidates[:max_candidates - len(developers)]:
                if not isinstance(user_data, dict):
                    continue
                
                username = user_data.get("username", "")
                if not username or username in existing_usernames:
                    continue
                
                # Skip organizations
                if _is_organization_or_company(username) or user_data.get("type") != "User":
                    continue
                
                # Create DeveloperMatch from search results
                developer = DeveloperMatch(
                    username=username,
                    name=username,
                    html_url=user_data.get("html_url", ""),
                    bio="",
                    location="",
                    followers=0,
                    public_repos=0,
                    top_repositories=[],
                    matching_skills=[],  # Will be determined by LLM scoring
                    relevance_score=user_data.get("score", 0) * 10,
                    skill_match_percentage=0.0,
                    is_exact_match=False
                )
                developers.append(developer)
                existing_usernames.add(username)
        
        # Select candidates: prioritize exact matches, fall back to partial if needed
        # Always return up to max_candidates (even if scores are low)
        selected = []
        
        # Sort by initial relevance score for selection
        developers.sort(key=lambda d: d.relevance_score, reverse=True)
        
        # Select top candidates
        selected = developers[:max_candidates]
        
        # Finalize candidates (before enrichment)
        finalized_developers = selected
        
        # Enrich finalized candidates with profile and repo data
        if finalized_developers:
            if progress_callback:
                progress_callback("Enriching finalized candidates with detailed data...", 70)
            finalized_developers = self._enrich_finalized_candidates(
                finalized_developers,
                skills,
                progress_callback
            )
            
            # Score enriched candidates with LLM
            if job_analysis:
                if progress_callback:
                    progress_callback("Scoring enriched candidates with AI...", 85)
                finalized_developers = self._llm_batch_score_candidates(
                    finalized_developers,
                    skills,
                    job_analysis
                )
                # Re-sort after scoring
                finalized_developers.sort(key=lambda d: (d.relevance_score, d.skill_match_percentage), reverse=True)
        
        results.developers = finalized_developers
        
        # Log final count
        if len(results.developers) < max_candidates:
            print(f"  ℹ Returning {len(results.developers)} candidates (requested {max_candidates})")
        else:
            print(f"  ✓ Returning {len(results.developers)} candidates")
        
        return results
    
    def _is_language(self, skill: str) -> bool:
        """Check if a skill is a programming language."""
        return skill.lower() in _LANGUAGES
    
    # ==============================================
    # USER ENRICHMENT (REST API for finalized candidates only)
    # ==============================================
    
    def _fetch_user_profile_rest(self, username: str) -> dict:
        """
        Fetch user profile details using GitHub REST API.
        Used only for enriching finalized candidates.
        
        Args:
            username: GitHub username
            
        Returns:
            dict: User profile data
        """
        github_token = os.getenv("GITHUB_TOKEN")
        headers = {}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        
        try:
            resp = requests.get(
                f"https://api.github.com/users/{username}",
                headers=headers,
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            
            return {
                "username": data.get("login", username),
                "name": data.get("name", ""),
                "bio": data.get("bio", ""),
                "html_url": data.get("html_url", ""),
                "avatar_url": data.get("avatar_url", ""),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "public_repos": data.get("public_repos", 0),
                "location": data.get("location", ""),
                "company": data.get("company", ""),
                "blog": data.get("blog", ""),
                "hireable": data.get("hireable", None),
                "created_at": data.get("created_at", ""),
                "type": data.get("type", "User"),
            }
        except Exception as e:
            print(f"  ⚠ Failed to fetch profile for {username}: {e}")
            return {
                "username": username,
                "name": username,
                "bio": "",
                "html_url": f"https://github.com/{username}",
                "avatar_url": "",
                "followers": 0,
                "following": 0,
                "public_repos": 0,
                "location": "",
                "company": "",
                "blog": "",
                "hireable": None,
                "created_at": "",
                "type": "User",
            }
    
    def _fetch_user_repos_rest(self, username: str, max_repos: int = 10) -> list[dict]:
        """
        Fetch user repositories using GitHub REST API.
        Used only for enriching finalized candidates.
        
        Args:
            username: GitHub username
            max_repos: Maximum number of repos to fetch
            
        Returns:
            list: Repository data with name, description, stars, language, etc.
        """
        github_token = os.getenv("GITHUB_TOKEN")
        headers = {}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        
        repos = []
        page = 1
        per_page = min(max_repos, 100)  # GitHub max per page
        
        try:
            while len(repos) < max_repos:
                resp = requests.get(
                    f"https://api.github.com/users/{username}/repos",
                    headers=headers,
                    params={"page": page, "per_page": per_page, "sort": "stars", "direction": "desc"},
                    timeout=15
                )
                resp.raise_for_status()
                page_repos = resp.json()
                
                if not page_repos:
                    break
                
                for r in page_repos:
                    if len(repos) >= max_repos:
                        break
                    repos.append({
                        "name": r.get("name", ""),
                        "full_name": r.get("full_name", ""),
                        "description": r.get("description", ""),
                        "html_url": r.get("html_url", ""),
                        "stars": r.get("stargazers_count", 0),
                        "forks": r.get("forks_count", 0),
                        "language": r.get("language", ""),
                        "topics": r.get("topics", []),
                        "created_at": r.get("created_at", ""),
                        "updated_at": r.get("updated_at", ""),
                    })
                
                if len(page_repos) < per_page:
                    break
                page += 1
                
        except Exception as e:
            print(f"  ⚠ Failed to fetch repos for {username}: {e}")
        
        return repos
    
    def _enrich_finalized_candidates(
        self,
        developers: list[DeveloperMatch],
        skills: list[str] = None,
        progress_callback: Callable[[str, float], None] = None
    ) -> list[DeveloperMatch]:
        """
        Enrich finalized candidates with profile and repository data.
        This is called only after candidates are finalized, before LLM scoring.
        
        Args:
            developers: List of finalized DeveloperMatch objects
            skills: List of skills to match against (for extracting matching skills)
            progress_callback: Optional progress callback
            
        Returns:
            list: Enriched DeveloperMatch objects
        """
        if not developers:
            return developers
        
        print(f"  📊 Enriching {len(developers)} finalized candidates with profile and repo data...")
        
        skills_lower = {s.lower(): s for s in (skills or [])}
        
        enriched = []
        for idx, dev in enumerate(developers):
            if progress_callback:
                progress = 70 + int((idx / len(developers)) * 10)
                progress_callback(f"Enriching candidate {idx + 1}/{len(developers)}...", progress)
            
            print(f"  📝 Fetching details for {dev.username}...")
            
            # Fetch profile data
            profile = self._fetch_user_profile_rest(dev.username)
            
            # Fetch repository data
            repos = self._fetch_user_repos_rest(dev.username, max_repos=10)
            
            # Update DeveloperMatch with enriched data
            dev.name = profile.get("name", "") or dev.username
            dev.bio = profile.get("bio", "") or ""
            dev.location = profile.get("location", "") or ""
            dev.followers = profile.get("followers", 0)
            dev.public_repos = profile.get("public_repos", 0)
            dev.top_repositories = repos[:5]  # Top 5 repos
            
            # Extract matching skills from repos and bio (matching against search skills)
            matching_skills = set()
            
            # Check repos for matching skills
            for repo in repos:
                # Check language
                lang = (repo.get("language") or "").lower()
                if lang and lang in skills_lower:
                    matching_skills.add(skills_lower[lang])
                
                # Check topics
                for topic in repo.get("topics", []):
                    topic_lower = topic.lower()
                    if topic_lower in skills_lower:
                        matching_skills.add(skills_lower[topic_lower])
                    # Partial match
                    for skill_lower, skill_orig in skills_lower.items():
                        if skill_lower in topic_lower or topic_lower in skill_lower:
                            matching_skills.add(skill_orig)
                
                # Check description
                desc = (repo.get("description") or "").lower()
                for skill_lower, skill_orig in skills_lower.items():
                    if skill_lower in desc:
                        matching_skills.add(skill_orig)
            
            # Check bio for matching skills
            bio_lower = (dev.bio or "").lower()
            for skill_lower, skill_orig in skills_lower.items():
                if skill_lower in bio_lower:
                    matching_skills.add(skill_orig)
            
            dev.matching_skills = list(matching_skills)
            # Update skill match percentage
            if skills:
                dev.skill_match_percentage = (len(matching_skills) / len(skills) * 100) if skills else 0.0
                dev.is_exact_match = dev.skill_match_percentage >= 80.0
            
            enriched.append(dev)
        
        print(f"  ✓ Enrichment complete for {len(enriched)} candidates")
        return enriched
    
    def _calculate_enhanced_relevance(
        self, 
        profile: dict, 
        repos: list[dict], 
        skills: list[str],
        job_analysis: dict = None
    ) -> float:
        """
        Calculate an enhanced relevance score for a developer.
        
        Enhanced factors:
        - Skill overlap percentage (most important)
        - Repository quality (stars, forks, recency)
        - Activity level (recent commits, updated repos)
        - Profile indicators (followers, hireable status)
        - Experience indicators (years active, repo diversity)
        
        Args:
            profile: User profile dict
            repos: User's repositories
            skills: Target skills
            job_analysis: Optional job analysis for context
            
        Returns:
            float: Relevance score (0-100)
        """
        score = 0.0
        skills_lower = {s.lower(): s for s in skills}
        skills_found = set()
        
        # === SKILL MATCHING (Max 50 points) ===
        # This is the most important factor
        
        # Check languages in repos
        repo_languages = set()
        for repo in repos:
            lang = (repo.get("language") or "").lower()
            if lang and lang in skills_lower:
                skills_found.add(skills_lower[lang])
                repo_languages.add(lang)
        
        # Check topics in repos
        repo_topics = set()
        for repo in repos:
            topics = [t.lower() for t in repo.get("topics", [])]
            for topic in topics:
                repo_topics.add(topic)
                # Exact match
                if topic in skills_lower:
                    skills_found.add(skills_lower[topic])
                # Partial match
                for skill_lower, skill_orig in skills_lower.items():
                    if skill_lower in topic or topic in skill_lower:
                        skills_found.add(skill_orig)
        
        # Check descriptions in repos
        for repo in repos:
            desc = (repo.get("description") or "").lower()
            for skill_lower, skill_orig in skills_lower.items():
                if skill_lower in desc:
                    skills_found.add(skill_orig)
        
        # Check bio
        bio = (profile.get("bio") or "").lower()
        for skill_lower, skill_orig in skills_lower.items():
            if skill_lower in bio:
                skills_found.add(skill_orig)
        
        # Skill overlap percentage (most important - 50 points)
        # Even if no repos, if skills found in bio, give base score
        if skills:
            skill_overlap = len(skills_found) / len(skills)
            score += skill_overlap * 50
            # Minimum base score if any skills match (even without repos)
            if len(skills_found) > 0 and not repos:
                score += 10  # Base score for bio match without repos
        
        # === REPOSITORY QUALITY (Max 25 points) ===
        if repos:
            total_stars = sum(r.get("stars", 0) for r in repos)
            total_forks = sum(r.get("forks", 0) for r in repos)
            
            # Stars factor (max 15 points)
            # Logarithmic scale: 100 stars = 5pts, 1000 = 10pts, 10000 = 15pts
            if total_stars > 0:
                stars_score = min(15, 5 + (10 * (total_stars / 1000) ** 0.5))
                score += stars_score
            
            # Forks factor (max 10 points)
            if total_forks > 0:
                forks_score = min(10, 3 + (7 * (total_forks / 500) ** 0.5))
                score += forks_score
        
        # === ACTIVITY & EXPERIENCE (Max 15 points) ===
        # Public repos count (indicator of experience)
        public_repos = profile.get("public_repos", 0)
        if public_repos > 0:
            # More repos = more experience (capped)
            repos_score = min(8, public_repos / 5)
            score += repos_score
        
        # Followers (indicator of reputation)
        followers = profile.get("followers", 0)
        if followers > 0:
            # Logarithmic: 100 followers = 2pts, 1000 = 5pts, 10000 = 7pts
            followers_score = min(7, 2 + (5 * (followers / 1000) ** 0.4))
            score += followers_score
        
        # === JOB-SPECIFIC MATCHING (Max 10 points) ===
        if job_analysis:
            # Check if experience level matches
            exp_level = job_analysis.get("experience_level", "").lower()
            if exp_level:
                # Senior/Lead roles: prefer more followers and repos
                if exp_level in ["senior", "lead", "principal"]:
                    if followers > 500 or public_repos > 20:
                        score += 5
                # Mid-level: moderate indicators
                elif exp_level == "mid":
                    if followers > 100 or public_repos > 10:
                        score += 5
                # Junior: any activity is good
                else:
                    if public_repos > 0:
                        score += 5
            
            # Check for specific tech stack matches
            tech_stack = job_analysis.get("tech_stack", [])
            if tech_stack:
                tech_matches = sum(1 for tech in tech_stack if tech.lower() in skills_found)
                if tech_matches > 0:
                    score += min(5, (tech_matches / len(tech_stack)) * 5)
        
        # === BONUS FACTORS ===
        # Hireable status
        if profile.get("hireable"):
            score += 5
        
        # Location match (if specified in job)
        if job_analysis and profile.get("location"):
            # Could add location matching logic here
            pass
        
        return min(score, 100)
    
    def _extract_matching_skills(
        self, 
        repos: list[dict], 
        skills: list[str]
    ) -> list[str]:
        """
        Extract skills from repositories that match the target skills.
        
        Args:
            repos: User's repositories
            skills: Target skills to match
            
        Returns:
            list: Matching skills found
        """
        matching = set()
        skills_lower = {s.lower(): s for s in skills}
        
        for repo in repos:
            # Check language
            lang = (repo.get("language") or "").lower()
            if lang in skills_lower:
                matching.add(skills_lower[lang])
            
            # Check topics
            for topic in repo.get("topics", []):
                topic_lower = topic.lower()
                if topic_lower in skills_lower:
                    matching.add(skills_lower[topic_lower])
                # Partial match
                for skill_lower, skill_orig in skills_lower.items():
                    if skill_lower in topic_lower or topic_lower in skill_lower:
                        matching.add(skill_orig)
        
        return list(matching)
    
    def _llm_batch_score_candidates(
        self,
        candidates: list[DeveloperMatch],
        required_skills: list[str],
        job_analysis: dict
    ) -> list[DeveloperMatch]:
        """
        Score all candidates in a single LLM call for efficiency.
        
        This is much faster than scoring candidates individually.
        
        Args:
            candidates: List of DeveloperMatch objects to score
            required_skills: Required skills from JD
            job_analysis: Job analysis dictionary
            
        Returns:
            list: Candidates with updated relevance scores
        """
        if not candidates:
            return candidates
        
        # Safely extract job analysis values
        job_title = (job_analysis.get('title') if job_analysis else None) or 'N/A'
        exp_level = (job_analysis.get('experience_level') if job_analysis else None) or 'N/A'
        tech_stack = job_analysis.get('tech_stack', []) if job_analysis else []
        frameworks = job_analysis.get('frameworks', []) if job_analysis else []
        tech_stack_str = ', '.join(tech_stack) if tech_stack else 'N/A'
        frameworks_str = ', '.join(frameworks) if frameworks else 'N/A'
        
        # Prepare candidate summaries
        candidate_summaries = []
        for idx, candidate in enumerate(candidates):
            bio_text = candidate.bio or ""
            bio_text = bio_text[:200] if isinstance(bio_text, str) else ""
            
            repo_summaries = []
            if candidate.top_repositories:
                for repo in candidate.top_repositories[:3]:
                    if repo:
                        repo_desc = repo.get("description") or ""
                        repo_summaries.append({
                            "name": repo.get("name", ""),
                            "description": repo_desc[:150] if repo_desc else "",
                            "language": repo.get("language", ""),
                            "stars": repo.get("stars", 0)
                        })
            
            candidate_summaries.append({
                "index": idx,
                "username": candidate.username or "N/A",
                "name": candidate.name or "N/A",
                "bio": bio_text,
                "location": candidate.location or "N/A",
                "followers": candidate.followers or 0,
                "public_repos": candidate.public_repos or 0,
                "matching_skills": candidate.matching_skills or [],
                "skill_match_percentage": candidate.skill_match_percentage,
                "repositories": repo_summaries
            })
        
        prompt = f"""You are an expert technical recruiter evaluating GitHub developer candidates for a job position.

JOB REQUIREMENTS:
- Title: {job_title}
- Experience Level: {exp_level}
- Required Skills: {', '.join(required_skills) if required_skills else 'N/A'}
- Tech Stack: {tech_stack_str}
- Frameworks: {frameworks_str}

CANDIDATES TO SCORE:
{json.dumps(candidate_summaries, indent=2)}

Your task: Generate a relevance score (0-100) for EACH candidate based on:
1. Skill match quality (exact matches get higher scores than partial)
2. Experience level alignment with job requirements
3. Repository quality and relevance
4. Overall fit for the role

Scoring Guidelines:
- 90-100: Excellent match - has all or most required skills, strong experience
- 70-89: Good match - has majority of required skills, relevant experience
- 50-69: Moderate match - has some required skills, some relevant experience
- 30-49: Partial match - has few required skills, limited relevant experience
- 10-29: Weak match - minimal skill overlap, but some relevance
- 0-9: Poor match - very little relevance

Return ONLY a JSON object with this exact format:
{{
  "scores": [
    {{"index": 0, "score": 85, "reasoning": "Brief explanation"}},
    {{"index": 1, "score": 72, "reasoning": "Brief explanation"}},
    ...
  ]
}}

Do not include any other text."""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical recruiter. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Safe access with None checks
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid response from LLM")
            
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                raise ValueError("Invalid message in response")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in response")
            
            score_data = json.loads(content)
            scores_list = score_data.get("scores", [])
            
            # Create index to score mapping
            score_map = {item.get("index"): float(item.get("score", 0)) for item in scores_list}
            
            # Update candidate scores
            for idx, candidate in enumerate(candidates):
                if idx in score_map:
                    candidate.relevance_score = max(0, min(100, score_map[idx]))
                else:
                    # Fallback: use skill match percentage
                    candidate.relevance_score = max(10, candidate.skill_match_percentage * 0.8)
            
            return candidates
            
        except Exception as e:
            print(f"  ⚠ Batch LLM scoring failed: {e}, using fallback scores")
            # Fallback: use skill match percentage as base score
            for candidate in candidates:
                candidate.relevance_score = max(10, candidate.skill_match_percentage * 0.8)
            return candidates
    
    def _llm_score_candidate(
        self,
        profile: dict,
        repos: list[dict],
        matching_skills: list[str],
        required_skills: list[str],
        job_analysis: dict
    ) -> float:
        """
        Use LLM to generate a relevance score for a single candidate.
        
        This replaces code-based scoring with AI-based evaluation that considers:
        - Skill match quality (not just count)
        - Job requirements alignment
        - Experience level match
        - Overall fit
        
        Args:
            profile: User profile dict
            repos: User's repositories
            matching_skills: Skills found in candidate's profile/repos
            required_skills: Required skills from JD
            job_analysis: Job analysis dictionary
            
        Returns:
            float: Relevance score (0-100) generated by LLM
        """
        # Prepare candidate summary
        repo_summaries = []
        if repos:
            for repo in repos[:5]:
                if repo:  # Check if repo is not None
                    repo_desc = repo.get("description") or ""
                    repo_summaries.append({
                        "name": repo.get("name", ""),
                        "description": repo_desc[:200] if repo_desc else "",
                        "language": repo.get("language", ""),
                        "stars": repo.get("stars", 0),
                        "topics": (repo.get("topics") or [])[:5]
                    })
        
        # Safely extract values with None checks
        job_title = (job_analysis.get('title') if job_analysis else None) or 'N/A'
        exp_level = (job_analysis.get('experience_level') if job_analysis else None) or 'N/A'
        tech_stack = job_analysis.get('tech_stack', []) if job_analysis else []
        frameworks = job_analysis.get('frameworks', []) if job_analysis else []
        tech_stack_str = ', '.join(tech_stack) if tech_stack else 'N/A'
        frameworks_str = ', '.join(frameworks) if frameworks else 'N/A'
        
        bio_text = profile.get('bio') or 'N/A'
        bio_text = bio_text[:300] if isinstance(bio_text, str) else 'N/A'
        
        matching_skills_str = ', '.join(matching_skills) if matching_skills else 'None'
        skill_match_pct = (len(matching_skills) / len(required_skills) * 100) if required_skills and len(required_skills) > 0 else 0.0
        
        prompt = f"""You are an expert technical recruiter evaluating a GitHub developer candidate for a job position.

JOB REQUIREMENTS:
- Title: {job_title}
- Experience Level: {exp_level}
- Required Skills: {', '.join(required_skills) if required_skills else 'N/A'}
- Tech Stack: {tech_stack_str}
- Frameworks: {frameworks_str}

CANDIDATE PROFILE:
- Username: {profile.get('username', 'N/A')}
- Name: {profile.get('name', 'N/A')}
- Bio: {bio_text}
- Location: {profile.get('location', 'N/A')}
- Followers: {profile.get('followers', 0)}
- Public Repos: {profile.get('public_repos', 0)}
- Matching Skills Found: {matching_skills_str}
- Skills Match Percentage: {skill_match_pct:.1f}%

TOP REPOSITORIES:
{json.dumps(repo_summaries, indent=2)}

Your task: Generate a relevance score (0-100) for this candidate based on:
1. Skill match quality (exact matches get higher scores than partial)
2. Experience level alignment with job requirements
3. Repository quality and relevance
4. Overall fit for the role

Scoring Guidelines:
- 90-100: Excellent match - has all or most required skills, strong experience
- 70-89: Good match - has majority of required skills, relevant experience
- 50-69: Moderate match - has some required skills, some relevant experience
- 30-49: Partial match - has few required skills, limited relevant experience
- 10-29: Weak match - minimal skill overlap, but some relevance
- 0-9: Poor match - very little relevance

Return ONLY a JSON object with this exact format:
{{"score": 85, "reasoning": "Brief explanation of the score"}}

Do not include any other text."""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical recruiter. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent scoring
                response_format={"type": "json_object"}
            )
            
            # Safe access with None checks
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid response from LLM")
            
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                raise ValueError("Invalid message in response")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in response")
            
            score_data = json.loads(content)
            score = float(score_data.get("score", 0))
            
            # Clamp score to 0-100 range
            return max(0, min(100, score))
            
        except Exception as e:
            print(f"  ⚠ LLM scoring failed for {profile.get('username', 'unknown')}: {e}, using fallback score")
            # Fallback: use skill match percentage as base score
            skill_match_pct = (len(matching_skills) / len(required_skills) * 100) if required_skills and len(required_skills) > 0 else 0
            return max(10, skill_match_pct * 0.8)  # At least 10 points if any skills match
    
    def _llm_rank_candidates(
        self,
        candidates: list[DeveloperMatch],
        skills: list[str],
        job_analysis: dict
    ) -> list[DeveloperMatch]:
        """
        Use LLM to intelligently rank and re-order candidates based on job fit.
        
        This method uses the LLM to evaluate how well each candidate matches
        the job requirements beyond just skill keywords.
        
        Args:
            candidates: List of candidate DeveloperMatch objects
            skills: Required skills
            job_analysis: Job analysis dictionary
            
        Returns:
            list: Re-ranked candidates
        """
        if not candidates or len(candidates) <= 1:
            return candidates
        
        # Prepare candidate summaries for LLM evaluation
        candidate_summaries = []
        for idx, candidate in enumerate(candidates[:15]):  # Limit to top 15 for LLM
            bio_text = candidate.bio or ""
            bio_text = bio_text[:200] if isinstance(bio_text, str) else ""
            
            top_repos = []
            if candidate.top_repositories:
                for r in candidate.top_repositories[:3]:
                    if r:  # Check if repo is not None
                        repo_desc = r.get("description") or ""
                        top_repos.append({
                            "name": r.get("name", ""),
                            "description": repo_desc[:100] if repo_desc else "",
                            "stars": r.get("stars", 0),
                            "language": r.get("language", "")
                        })
            
            summary = {
                "index": idx,
                "username": candidate.username or "N/A",
                "name": candidate.name or "N/A",
                "bio": bio_text,
                "followers": candidate.followers or 0,
                "public_repos": candidate.public_repos or 0,
                "matching_skills": candidate.matching_skills or [],
                "top_repos": top_repos
            }
            candidate_summaries.append(summary)
        
        # Safely extract job analysis values
        job_title = (job_analysis.get('title') if job_analysis else None) or 'N/A'
        exp_level = (job_analysis.get('experience_level') if job_analysis else None) or 'N/A'
        tech_stack = job_analysis.get('tech_stack', []) if job_analysis else []
        frameworks = job_analysis.get('frameworks', []) if job_analysis else []
        tech_stack_str = ', '.join(tech_stack) if tech_stack else 'N/A'
        frameworks_str = ', '.join(frameworks) if frameworks else 'N/A'
        
        # Create LLM prompt for ranking
        prompt = f"""You are evaluating GitHub developer candidates for a job position.

JOB REQUIREMENTS:
- Title: {job_title}
- Experience Level: {exp_level}
- Required Skills: {', '.join(skills) if skills else 'N/A'}
- Tech Stack: {tech_stack_str}
- Frameworks: {frameworks_str}

CANDIDATES TO EVALUATE:
{json.dumps(candidate_summaries, indent=2)}

Your task: Rank these candidates from best to worst match for the job.
Consider:
1. Skill overlap with requirements
2. Repository quality and relevance
3. Experience level indicators
4. Overall fit for the role

Return a JSON array of candidate indices in order of best match (best first):
{{"ranked_indices": [0, 3, 1, 5, ...]}}

Only return the JSON, no other text."""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical recruiter evaluating GitHub profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent ranking
                response_format={"type": "json_object"}
            )
            
            # Safe access with None checks
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid response from LLM")
            
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                raise ValueError("Invalid message in response")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in response")
            
            ranking_data = json.loads(content)
            ranked_indices = ranking_data.get("ranked_indices", [])
            
            # Re-order candidates based on LLM ranking
            if ranked_indices and len(ranked_indices) == len(candidate_summaries):
                ranked_candidates = []
                index_to_candidate = {i: candidates[i] for i in range(len(candidate_summaries))}
                
                for idx in ranked_indices:
                    if idx in index_to_candidate:
                        ranked_candidates.append(index_to_candidate[idx])
                
                # Add any remaining candidates not ranked by LLM
                for i, candidate in enumerate(candidates):
                    if i >= len(candidate_summaries) or i not in ranked_indices:
                        ranked_candidates.append(candidate)
                
                return ranked_candidates
            
        except Exception as e:
            print(f"  ⚠ LLM ranking failed: {e}, using original ranking")
        
        # Fallback to original order if LLM ranking fails
        return candidates
    
    async def search_async(self, skills: list[str], max_candidates: int = 10) -> SearchResults:
        """
        Async version of the search method.
        
        Args:
            skills: List of skills to search for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            SearchResults: Compiled search results
        """
        # For now, delegate to sync version
        # Could be enhanced with async HTTP calls
        return self.search(skills, max_candidates)


# ==============================================
# FACTORY FUNCTION
# ==============================================

def create_github_agent(
    model: str = "gpt-4o",
    temperature: float = 0.5
) -> GitHubSearchAgent:
    """
    Factory function to create a GitHubSearchAgent.
    
    Args:
        model: The LLM model to use
        temperature: Generation temperature
        
    Returns:
        GitHubSearchAgent: Configured GitHub search agent
    """
    return GitHubSearchAgent(model=model, temperature=temperature)


# ==============================================
# MODULE TEST
# ==============================================

if __name__ == "__main__":
    print("Testing GitHubSearchAgent...")
    print("-" * 50)
    
    agent = create_github_agent()
    results = agent.search(["python", "django", "machine-learning"], max_candidates=5)
    
    print(f"\nFound {len(results.repositories)} repositories")
    print(f"Found {len(results.developers)} developers")
    
    for dev in results.developers:
        print(f"\n👤 {dev.name} (@{dev.username})")
        print(f"   Followers: {dev.followers} | Repos: {dev.public_repos}")
        print(f"   Skills: {', '.join(dev.matching_skills)}")
        print(f"   Score: {dev.relevance_score:.1f}")

