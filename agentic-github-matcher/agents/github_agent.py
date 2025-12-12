"""
GitHub Search Agent
===================

This module implements the GitHubSearchAgent, responsible for:
- Searching GitHub repositories based on extracted skills
- Finding relevant developer profiles
- Fetching detailed repository and user information

The agent uses tool calling to interact with the GitHub API
and finds candidates matching the job requirements.
"""

import json
import os
from typing import Callable, Any
from dataclasses import dataclass, field

# AutoGen imports (pyautogen 0.2.x classic API)
from autogen import ConversableAgent, AssistantAgent

# LiteLLM for model gateway
import litellm

# Import GitHub tools (direct REST API only - MCP disabled)
from tools.github_search import (
    search_repositories_by_skills,
    fetch_user_profile,
    fetch_user_repos,
    search_users_by_skills
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

GITHUB_AGENT_SYSTEM_PROMPT = """You are a GitHub Search Agent specialized in finding developers and repositories.

Your role is to:
1. Take a list of required skills and technologies
2. Search GitHub for relevant repositories and developers
3. Compile a list of potential candidates who match the requirements

You have access to the following tools (via direct GitHub REST API):
- search_repositories_by_skills: Search for repositories matching skills
- search_users_by_skills: Search for developers matching skills
- fetch_user_profile: Get detailed profile for a GitHub user
- fetch_user_repos: Get repositories for a specific user

Note: These tools use the GitHub REST API directly for reliable GitHub API access.

WORKFLOW:
1. First, search for repositories matching the key skills
2. Identify repository owners who might be good candidates
3. Search for users directly based on skills
4. Fetch detailed profiles for promising candidates
5. Compile results with relevant information

Always aim to find 5-10 relevant candidates with their key information.
Return results in a structured format that can be used for formatting.
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
        # Using direct REST API calls only
        self.tools = {
            "search_repositories_by_skills": search_repositories_by_skills,
            "search_users_by_skills": search_users_by_skills,
            "fetch_user_profile": fetch_user_profile,
            "fetch_user_repos": fetch_user_repos,
        }
        
        print(f"  [GitHubSearchAgent] ✓ Using direct REST API for GitHub operations")
        
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
            max_consecutive_auto_reply=5,
        )
    
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
                    "name": "search_repositories_by_skills",
                    "description": "Search GitHub repositories by skill keywords. Returns repos sorted by stars.",
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
            {
                "type": "function",
                "function": {
                    "name": "search_users_by_skills",
                    "description": "Search for GitHub users/developers by skill keywords.",
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
            {
                "type": "function",
                "function": {
                    "name": "fetch_user_profile",
                    "description": "Fetch detailed profile information for a GitHub user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "GitHub username to fetch profile for"
                            }
                        },
                        "required": ["username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_user_repos",
                    "description": "Fetch repositories for a specific GitHub user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "GitHub username to fetch repos for"
                            },
                            "max_repos": {
                                "type": "integer",
                                "description": "Maximum number of repos to return",
                                "default": 10
                            }
                        },
                        "required": ["username"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Execute a tool function by name.
        
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
    
    def search(
        self, 
        skills: list[str], 
        job_analysis: dict = None,
        max_candidates: int = 10,
        progress_callback: Callable[[str, float], None] = None
    ) -> SearchResults:
        """
        Search GitHub for developers and repositories matching the given skills.
        
        Enhanced search strategy:
        1. Multi-strategy repository search (language, topic, description)
        2. User search with multiple query types
        3. Fetch detailed profiles and repositories
        4. Intelligent scoring based on skill overlap and job requirements
        5. LLM-based candidate evaluation for top matches
        
        Args:
            skills: List of skills to search for
            job_analysis: Optional job analysis dict for better matching
            max_candidates: Maximum number of candidates to return
            
        Returns:
            SearchResults: Compiled search results
        """
        results = SearchResults(query_skills=skills)
        
        # Step 1: Enhanced repository search with multiple strategies
        strategy_info = "via Direct REST API"
        print(f"  [GitHubSearchAgent] Starting search with {len(skills)} skills")
        print(f"  [GitHubSearchAgent] Strategy: {strategy_info}")
        if progress_callback:
            progress_callback(f"Searching GitHub repositories ({strategy_info})...", 35)
        print(f"  🔍 Searching repositories for skills: {skills[:5]} ({strategy_info})")
        
        # Strategy 1: Search by languages
        language_skills = [s for s in skills if self._is_language(s)]
        if language_skills:
            repo_results_lang = search_repositories_by_skills(language_skills, max_results=20)
            if repo_results_lang.get("success"):
                results.repositories.extend(repo_results_lang.get("repositories", []))
            else:
                error_msg = repo_results_lang.get("error", "Unknown error")
                print(f"  ⚠ Repository search failed: {error_msg}")
        
        # Strategy 2: Search by frameworks/tools (topic-based)
        framework_skills = [s for s in skills if not self._is_language(s)]
        if framework_skills:
            repo_results_fw = search_repositories_by_skills(framework_skills, max_results=20)
            if repo_results_fw.get("success"):
                results.repositories.extend(repo_results_fw.get("repositories", []))
            else:
                error_msg = repo_results_fw.get("error", "Unknown error")
                print(f"  ⚠ Repository search failed: {error_msg}")
        
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
        from tools.github_search import _is_organization_or_company
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
        # Optimized: Only fetch max_candidates to reduce API calls
        for skill in skills[:5]:  # Try top 5 skills individually
            if len(user_candidates) >= max_candidates:
                break
            skill_results = search_users_by_skills([skill], max_results=max_candidates)
            if skill_results.get("success"):
                new_candidates = [u.get("username") for u in skill_results.get("users", [])]
                for candidate in new_candidates:
                    if candidate not in seen_candidates and len(user_candidates) < max_candidates:
                        user_candidates.append(candidate)
                        seen_candidates.add(candidate)
        
        # Strategy 2: Also search by primary skills combination (for exact matches)
        if len(user_candidates) < max_candidates and len(skills) >= 2:
            primary_skills = skills[:3] if len(skills) >= 3 else skills
            user_results = search_users_by_skills(primary_skills, max_results=max_candidates)
            if user_results.get("success"):
                results.total_developers_found = user_results.get("total_count", 0)
                for candidate in [u.get("username") for u in user_results.get("users", [])]:
                    if candidate not in seen_candidates and len(user_candidates) < max_candidates:
                        user_candidates.append(candidate)
                        seen_candidates.add(candidate)
        
        # Combine and prioritize candidates (repo owners first, then direct search)
        all_candidates = repo_owners + [u for u in user_candidates if u not in repo_owners]
        # Only fetch max_candidates to optimize speed
        all_candidates = all_candidates[:max_candidates]
        
        # Step 3: Fetch detailed profiles and calculate enhanced scores
        if progress_callback:
            progress_callback(f"Fetching profiles for {len(all_candidates)} candidates...", 45)
        print(f"  👤 Fetching profiles for {len(all_candidates)} candidates...")
        developers = []
        filtered_count = 0
        low_relevance_count = 0
        no_repos_count = 0
        
        for idx, username in enumerate(all_candidates):
            # Update progress during fetching (update for every candidate for better responsiveness)
            if progress_callback:
                progress = 45 + int((idx / max(len(all_candidates), 1)) * 10)
                progress_callback(f"Fetching profile {idx + 1}/{len(all_candidates)}...", progress)
            if not username:
                continue
                
            profile = fetch_user_profile(username)
            
            if not profile.get("success"):
                filtered_count += 1
                continue
            
            # Skip organizations and companies
            if profile.get("is_organization") or profile.get("type") != "User":
                filtered_count += 1
                continue
            
            # Additional check: skip if username looks like an organization
            from tools.github_search import _is_organization_or_company
            if _is_organization_or_company(username):
                filtered_count += 1
                continue
            
            # Additional heuristics: very high repo count often indicates organization
            public_repos = profile.get("public_repos", 0)
            if public_repos > 200:  # Organizations typically have many repos
                # But allow if it's a very active individual contributor
                followers = profile.get("followers", 0)
                if followers < 1000:  # Low followers + high repos = likely org
                    filtered_count += 1
                    continue
            
            # Fetch user's top repos (more repos for better analysis)
            user_repos = fetch_user_repos(username, max_repos=10)
            top_repos = user_repos.get("repositories", []) if user_repos.get("success") else []
            
            # Extract matching skills from repos
            matching_skills = self._extract_matching_skills(top_repos, skills) if top_repos else []
            
            # Also check bio and profile for skill mentions (even if no repos)
            bio = (profile.get("bio") or "").lower()
            skills_lower = {s.lower(): s for s in skills}
            for skill_lower, skill_orig in skills_lower.items():
                if skill_lower in bio and skill_orig not in matching_skills:
                    matching_skills.append(skill_orig)
            
            # If no matching skills found at all, skip this candidate
            if not matching_skills:
                no_repos_count += 1
                continue
            
            # Calculate skill match percentage
            skill_match_percentage = (len(matching_skills) / len(skills) * 100) if skills else 0.0
            
            # Determine if this is an exact match (80%+ skill match)
            is_exact_match = skill_match_percentage >= 80.0
            
            # Include ALL candidates with at least 1 matching skill
            # We'll score them all in batch later for efficiency
            if len(matching_skills) > 0:
                developer = DeveloperMatch(
                    username=profile.get("username", ""),
                    name=profile.get("name", "") or profile.get("username", ""),
                    html_url=profile.get("html_url", ""),
                    bio=profile.get("bio", "") or "",
                    location=profile.get("location", "") or "",
                    followers=profile.get("followers", 0),
                    public_repos=profile.get("public_repos", 0),
                    top_repositories=top_repos[:5],  # Top 5 repos for better context
                    matching_skills=matching_skills,
                    relevance_score=0.0,  # Will be set by batch scoring
                    skill_match_percentage=skill_match_percentage,
                    is_exact_match=is_exact_match
                )
                developers.append(developer)
            else:
                low_relevance_count += 1
        
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
        if no_repos_count > 0:
            print(f"  ℹ Skipped {no_repos_count} candidates with no repositories")
        if low_relevance_count > 0:
            print(f"  ℹ Skipped {low_relevance_count} candidates with low relevance scores")
        
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
            for skill in skills[:5]:  # Try top 5 skills
                if len(expanded_candidates) >= max_candidates:
                    break
                skill_results = search_users_by_skills([skill], max_results=max_candidates)
                if skill_results.get("success"):
                    new_users = [u.get("username") for u in skill_results.get("users", [])]
                    expanded_candidates.extend([u for u in new_users if u not in all_candidates])
            
            # Fetch profiles for expanded candidates (limited to max_candidates)
            existing_usernames = {d.username for d in developers}
            for username in expanded_candidates[:max_candidates]:
                if username in existing_usernames:
                    continue
                    
                profile = fetch_user_profile(username)
                if not profile.get("success") or profile.get("is_organization") or profile.get("type") != "User":
                    continue
                
                # Check bio for skills
                bio = (profile.get("bio") or "").lower()
                matching_skills_bio = []
                skills_lower = {s.lower(): s for s in skills}
                for skill_lower, skill_orig in skills_lower.items():
                    if skill_lower in bio:
                        matching_skills_bio.append(skill_orig)
                
                # If bio has matching skills, include this candidate
                if matching_skills_bio:
                    user_repos = fetch_user_repos(username, max_repos=5)
                    top_repos = user_repos.get("repositories", []) if user_repos.get("success") else []
                    
                    skill_match_percentage = (len(matching_skills_bio) / len(skills) * 100) if skills else 0.0
                    
                    developer = DeveloperMatch(
                        username=profile.get("username", ""),
                        name=profile.get("name", "") or profile.get("username", ""),
                        html_url=profile.get("html_url", ""),
                        bio=profile.get("bio", "") or "",
                        location=profile.get("location", "") or "",
                        followers=profile.get("followers", 0),
                        public_repos=profile.get("public_repos", 0),
                        top_repositories=top_repos[:5],
                        matching_skills=matching_skills_bio,
                        relevance_score=0.0,  # Will be set by batch scoring
                        skill_match_percentage=skill_match_percentage,
                        is_exact_match=skill_match_percentage >= 80.0
                    )
                    developers.append(developer)
                    existing_usernames.add(username)
        
        # Re-score expanded candidates if any were added
        if developers and job_analysis:
            developers = self._llm_batch_score_candidates(developers, skills, job_analysis)
        
        # Re-sort after scoring
        exact_matches = [d for d in developers if d.is_exact_match]
        partial_matches = [d for d in developers if not d.is_exact_match]
        exact_matches.sort(key=lambda d: (d.skill_match_percentage, d.relevance_score), reverse=True)
        partial_matches.sort(key=lambda d: (d.skill_match_percentage, d.relevance_score), reverse=True)
        developers = exact_matches + partial_matches
        
        # Select candidates: prioritize exact matches, fall back to partial if needed
        # Always return up to max_candidates (even if scores are low)
        selected = []
        
        if exact_matches:
            # If we have exact matches, prioritize them
            selected = exact_matches[:max_candidates]
            # Add partial matches if we don't have enough exact matches
            if len(selected) < max_candidates:
                remaining_slots = max_candidates - len(selected)
                selected.extend(partial_matches[:remaining_slots])
        else:
            # No exact matches found, use partial matches
            selected = partial_matches[:max_candidates] if partial_matches else []
        
        # Final fallback: if we still don't have enough, use all candidates sorted by score
        if len(selected) < max_candidates and len(developers) > len(selected):
            # Get remaining candidates sorted by score
            remaining = [d for d in developers if d not in selected]
            remaining.sort(key=lambda d: d.relevance_score, reverse=True)
            needed = max_candidates - len(selected)
            selected.extend(remaining[:needed])
        
        # Ensure we return at least what we have (even if below max_candidates)
        results.developers = selected if selected else developers[:max_candidates]
        
        # Log final count
        if len(results.developers) < max_candidates:
            print(f"  ℹ Returning {len(results.developers)} candidates (requested {max_candidates})")
        else:
            print(f"  ✓ Returning {len(results.developers)} candidates")
        
        return results
    
    def _is_language(self, skill: str) -> bool:
        """Check if a skill is a programming language."""
        from tools.github_search import _LANGUAGES
        return skill.lower() in _LANGUAGES
    
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

