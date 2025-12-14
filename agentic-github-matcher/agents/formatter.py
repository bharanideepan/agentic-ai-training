"""
Formatter Agent
===============

This module implements the FormatterAgent, responsible for:
- Converting raw search results into professional output
- Generating formatted reports with candidate information
- Creating summary tables and detailed profiles

The agent produces clean, readable output suitable for
recruitment and hiring purposes.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

# AutoGen imports (pyautogen 0.2.x classic API)
from autogen import ConversableAgent, UserProxyAgent

# LiteLLM for model gateway
import litellm

# Rich for beautiful console output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text


# ==============================================
# FORMATTER AGENT SYSTEM PROMPT
# ==============================================

FORMATTER_SYSTEM_PROMPT = """You are a Professional Report Formatter Agent.

Your role is to take raw data about job requirements and candidate matches,
and format them into a professional, readable report.

When formatting reports:
1. Create clear sections with headers
2. Use bullet points for lists
3. Highlight key information
4. Provide a summary at the top
5. Include all relevant details without being verbose
6. Use professional language appropriate for HR/recruitment

Your output should be in Markdown format and include:
- Executive Summary
- Job Requirements Overview
- Candidate Profiles (with GitHub links, skills, and relevance)
- Repository Highlights
- Recommendations

Make the report actionable and easy to scan quickly.
"""


# ==============================================
# FORMATTER AGENT CLASS
# ==============================================

class FormatterAgent:
    """
    The FormatterAgent converts analysis results and GitHub search data
    into professional, formatted reports using agentic behavior.
    
    This agent uses AutoGen's ConversableAgent to autonomously format reports
    and generate summaries. The agent decides how to structure and present
    the information for optimal readability.
    
    It supports multiple output formats:
    - Rich console output (colorful, tabular)
    - Markdown format
    - Plain text
    - JSON
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        name: str = "FormatterAgent"
    ):
        """
        Initialize the FormatterAgent.
        
        Args:
            model: The LLM model to use (via LiteLLM)
            temperature: Temperature for generation
            name: Name identifier for the agent
        """
        self.model = model
        self.temperature = temperature
        self.name = name
        self.console = Console()
        
        # Configure LiteLLM settings
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
        }
        
        # Log API key status
        if api_key:
            if api_key.startswith("sk-"):
                print(f"  [FormatterAgent] ✓ API key format valid")
            else:
                print(f"  [FormatterAgent] ⚠ API key format unusual: {api_key[:10]}...")
        else:
            print(f"  [FormatterAgent] ⚠ No API key found")
        
        # Create the AutoGen agent for agentic behavior
        self.agent = ConversableAgent(
            name=name,
            system_message=FORMATTER_SYSTEM_PROMPT,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )
        
        print(f"  [FormatterAgent] ✓ Agentic behavior enabled - using AutoGen agent")
    
    def format_results(
        self,
        job_analysis: dict,
        search_results: dict,
        output_format: str = "rich"
    ) -> str:
        """
        Format the combined results into a professional report using agentic behavior.
        
        This method uses the AutoGen agent to intelligently structure and format reports,
        generating professional summaries, insights, and recommendations.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            search_results: Dictionary from SearchResults.to_dict()
            output_format: One of "rich", "markdown", "text", "json"
            
        Returns:
            str: Formatted report string
        """
        if output_format == "rich":
            return self._format_rich(job_analysis, search_results)
        elif output_format == "markdown":
            return self._format_markdown_agentic(job_analysis, search_results)
        elif output_format == "json":
            return self._format_json(job_analysis, search_results)
        else:
            return self._format_text_agentic(job_analysis, search_results)
    
    def _format_rich(self, job_analysis: dict, search_results: dict) -> str:
        """
        Format results using Rich library for beautiful console output.
        """
        output_parts = []
        
        # Header
        self.console.print()
        self.console.print(Panel.fit(
            "[bold blue]🎯 GitHub Talent Matcher Report[/bold blue]",
            border_style="blue"
        ))
        
        # Job Analysis Section
        self.console.print()
        self.console.print("[bold cyan]📋 JOB REQUIREMENTS ANALYSIS[/bold cyan]")
        self.console.print("-" * 50)
        
        if job_analysis.get("title"):
            self.console.print(f"[bold]Position:[/bold] {job_analysis['title']}")
        
        if job_analysis.get("experience_level"):
            exp = job_analysis.get("experience_years", "N/A")
            level = job_analysis.get("experience_level", "")
            self.console.print(f"[bold]Experience:[/bold] {exp} years ({level})")
        
        if job_analysis.get("skills"):
            skills_str = ", ".join(job_analysis["skills"][:10])
            self.console.print(f"[bold]Key Skills:[/bold] {skills_str}")
        
        if job_analysis.get("tech_stack"):
            tech_str = ", ".join(job_analysis["tech_stack"][:8])
            self.console.print(f"[bold]Tech Stack:[/bold] {tech_str}")
        
        if job_analysis.get("frameworks"):
            fw_str = ", ".join(job_analysis["frameworks"][:6])
            self.console.print(f"[bold]Frameworks:[/bold] {fw_str}")
        
        # Candidate Table
        self.console.print()
        self.console.print("[bold cyan]👥 MATCHED CANDIDATES[/bold cyan]")
        self.console.print("-" * 50)
        
        developers = search_results.get("developers", [])
        # Ensure candidates are sorted by relevance score (descending)
        developers = sorted(
            developers, 
            key=lambda d: d.get("relevance_score", 0), 
            reverse=True
        )
        
        if developers:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", style="dim", width=5)
            table.add_column("Developer", style="cyan", width=20)
            table.add_column("Followers", justify="right", width=10)
            table.add_column("Repos", justify="right", width=8)
            table.add_column("Skills Match", width=25)
            table.add_column("Score", justify="right", width=8)
            
            for idx, dev in enumerate(developers[:10], 1):
                skills_match = ", ".join(dev.get("matching_skills", [])[:3])
                score = dev.get("relevance_score", 0)
                score_style = "green" if score >= 50 else "yellow" if score >= 30 else "red"
                
                table.add_row(
                    str(idx),
                    f"@{dev.get('username', 'N/A')}",
                    str(dev.get("followers", 0)),
                    str(dev.get("public_repos", 0)),
                    skills_match or "N/A",
                    f"[{score_style}]{score:.0f}[/{score_style}]"
                )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No matching candidates found.[/yellow]")
        
        # Top Repositories
        self.console.print()
        self.console.print("[bold cyan]📦 TOP MATCHING REPOSITORIES[/bold cyan]")
        self.console.print("-" * 50)
        
        repos = search_results.get("repositories", [])[:5]
        
        if repos:
            repo_table = Table(show_header=True, header_style="bold magenta")
            repo_table.add_column("Repository", style="cyan", width=30)
            repo_table.add_column("⭐ Stars", justify="right", width=10)
            repo_table.add_column("🍴 Forks", justify="right", width=10)
            repo_table.add_column("Language", width=15)
            
            for repo in repos:
                repo_table.add_row(
                    repo.get("full_name", "N/A")[:30],
                    f"{repo.get('stars', 0):,}",
                    f"{repo.get('forks', 0):,}",
                    repo.get("language", "N/A") or "N/A"
                )
            
            self.console.print(repo_table)
        
        # Summary Statistics with AI-generated insights
        self.console.print()
        try:
            print(f"  [FormatterAgent] 🤖 Generating insights summary using agentic behavior...")
            insights_summary = self.generate_llm_summary(job_analysis, search_results)
            # Truncate for panel display
            summary_text = insights_summary[:200] + "..." if len(insights_summary) > 200 else insights_summary
            self.console.print(Panel(
                f"[bold]AI-Generated Insights:[/bold]\n\n{summary_text}\n\n"
                f"[bold]Summary:[/bold] Found [green]{len(developers)}[/green] candidates "
                f"and [green]{search_results.get('total_repos_found', 0):,}[/green] repositories "
                f"matching the job requirements.",
                title="📊 Results Summary & Insights",
                border_style="green"
            ))
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate agentic insights: {e}")
            # Fallback to simple summary
            self.console.print(Panel(
                f"[bold]Summary:[/bold] Found [green]{len(developers)}[/green] candidates "
                f"and [green]{search_results.get('total_repos_found', 0):,}[/green] repositories "
                f"matching the job requirements.",
                title="📊 Results Summary",
                border_style="green"
            ))
        
        # Generate text version for return
        return self._format_text_agentic(job_analysis, search_results)
    
    def _format_markdown_agentic(self, job_analysis: dict, search_results: dict) -> str:
        """
        Format results as Markdown using agentic behavior for intelligent formatting.
        
        Uses the AutoGen agent to generate professional summaries, insights, and recommendations.
        """
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines.append("# 🎯 GitHub Talent Matcher Report")
        lines.append(f"\n*Generated: {timestamp}*\n")
        
        # Use agentic behavior to generate executive summary
        try:
            print(f"  [FormatterAgent] 🤖 Generating executive summary using agentic behavior...")
            executive_summary = self.generate_llm_summary(job_analysis, search_results)
            lines.append("## 📝 Executive Summary\n")
            lines.append(executive_summary)
            lines.append("")
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate agentic summary: {e}")
            # Fallback to simple summary
            developers = search_results.get("developers", [])
            repos = search_results.get("repositories", [])
            lines.append("## 📝 Executive Summary\n")
            lines.append(f"This report identifies **{len(developers)} potential candidates** "
                        f"and **{len(repos)} relevant repositories** based on the job requirements.\n")
        
        # Job Requirements
        lines.append("## 📋 Job Requirements Analysis\n")
        
        if job_analysis.get("title"):
            lines.append(f"**Position:** {job_analysis['title']}\n")
        
        if job_analysis.get("experience_years") or job_analysis.get("experience_level"):
            exp = job_analysis.get("experience_years", "N/A")
            level = job_analysis.get("experience_level", "")
            lines.append(f"**Experience Required:** {exp} years ({level})\n")
        
        if job_analysis.get("skills"):
            lines.append("**Required Skills:**")
            for skill in job_analysis["skills"][:10]:
                lines.append(f"- {skill}")
            lines.append("")
        
        if job_analysis.get("tech_stack"):
            lines.append("**Tech Stack:**")
            for tech in job_analysis["tech_stack"][:8]:
                lines.append(f"- {tech}")
            lines.append("")
        
        if job_analysis.get("summary"):
            lines.append(f"**Summary:** {job_analysis['summary']}\n")
        
        # Matched Candidates
        lines.append("## 👥 Matched Candidates\n")
        
        # Get and sort developers
        developers = search_results.get("developers", [])
        # Ensure candidates are sorted by relevance score (descending)
        developers = sorted(
            developers, 
            key=lambda d: d.get("relevance_score", 0), 
            reverse=True
        )
        
        if developers:
            lines.append("| Rank | Developer | Followers | Repos | Score |")
            lines.append("|------|-----------|-----------|-------|-------|")
            
            for idx, dev in enumerate(developers[:10], 1):
                username = dev.get("username", "N/A")
                url = dev.get("html_url", "#")
                followers = dev.get("followers", 0)
                repos_count = dev.get("public_repos", 0)
                score = dev.get("relevance_score", 0)
                
                lines.append(f"| {idx} | [@{username}]({url}) | {followers:,} | {repos_count} | {score:.0f} |")
            
            lines.append("")
            
            # Use agentic behavior to generate candidate highlights
            try:
                print(f"  [FormatterAgent] 🤖 Generating candidate highlights using agentic behavior...")
                candidate_highlights = self._generate_candidate_highlights(job_analysis, developers[:5])
                if candidate_highlights:
                    lines.append("### 🌟 Top Candidate Highlights\n")
                    lines.append(candidate_highlights)
                    lines.append("")
            except Exception as e:
                print(f"  [FormatterAgent] ⚠ Could not generate candidate highlights: {e}")
            
            # Detailed profiles
            lines.append("### Candidate Details\n")
            
            for idx, dev in enumerate(developers[:5], 1):
                lines.append(f"#### {idx}. {dev.get('name', dev.get('username', 'N/A'))}")
                lines.append(f"- **GitHub:** [@{dev.get('username', 'N/A')}]({dev.get('html_url', '#')})")
                if dev.get("location"):
                    lines.append(f"- **Location:** {dev['location']}")
                if dev.get("bio"):
                    lines.append(f"- **Bio:** {dev['bio'][:150]}...")
                if dev.get("matching_skills"):
                    lines.append(f"- **Matching Skills:** {', '.join(dev['matching_skills'])}")
                
                if dev.get("top_repositories"):
                    lines.append("- **Top Repositories:**")
                    for repo in dev["top_repositories"][:3]:
                        lines.append(f"  - [{repo.get('name', 'N/A')}]({repo.get('html_url', '#')}) "
                                   f"⭐ {repo.get('stars', 0):,}")
                lines.append("")
        else:
            lines.append("*No matching candidates found.*\n")
        
        # Top Repositories
        lines.append("## 📦 Top Matching Repositories\n")
        
        if repos:
            lines.append("| Repository | Stars | Forks | Language |")
            lines.append("|------------|-------|-------|----------|")
            
            for repo in repos[:10]:
                name = repo.get("full_name", "N/A")
                url = repo.get("html_url", "#")
                stars = repo.get("stars", 0)
                forks = repo.get("forks", 0)
                lang = repo.get("language", "N/A") or "N/A"
                
                lines.append(f"| [{name}]({url}) | {stars:,} | {forks:,} | {lang} |")
            
            lines.append("")
        
        # Add recommendations section using agentic behavior
        try:
            print(f"  [FormatterAgent] 🤖 Generating recommendations using agentic behavior...")
            recommendations = self._generate_recommendations(job_analysis, search_results)
            if recommendations:
                lines.append("## 💡 Recommendations\n")
                lines.append(recommendations)
                lines.append("")
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate recommendations: {e}")
        
        # Footer
        lines.append("---")
        lines.append("*Report generated by GitHub Talent Matcher - Agentic AI Workflow*")
        
        return "\n".join(lines)
    
    def _format_text_agentic(self, job_analysis: dict, search_results: dict) -> str:
        """
        Format results as plain text using agentic behavior for narrative summaries.
        
        Uses the AutoGen agent to generate professional text-based reports with insights.
        """
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines.append("=" * 60)
        lines.append("        GITHUB TALENT MATCHER REPORT")
        lines.append(f"        Generated: {timestamp}")
        lines.append("=" * 60)
        
        # Use agentic behavior to generate executive summary
        try:
            print(f"  [FormatterAgent] 🤖 Generating narrative summary using agentic behavior...")
            executive_summary = self.generate_llm_summary(job_analysis, search_results)
            lines.append("\nEXECUTIVE SUMMARY")
            lines.append("-" * 60)
            lines.append(executive_summary)
            lines.append("")
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate agentic summary: {e}")
            # Continue with deterministic formatting
        
        # Job Analysis
        lines.append("\nJOB REQUIREMENTS ANALYSIS")
        lines.append("-" * 40)
        
        if job_analysis.get("title"):
            lines.append(f"Position: {job_analysis['title']}")
        
        if job_analysis.get("experience_years") or job_analysis.get("experience_level"):
            exp = job_analysis.get("experience_years", "N/A")
            level = job_analysis.get("experience_level", "")
            lines.append(f"Experience: {exp} years ({level})")
        
        if job_analysis.get("skills"):
            lines.append(f"Skills: {', '.join(job_analysis['skills'][:10])}")
        
        if job_analysis.get("tech_stack"):
            lines.append(f"Tech Stack: {', '.join(job_analysis['tech_stack'][:8])}")
        
        # Candidates
        lines.append("\n\nMATCHED CANDIDATES")
        lines.append("-" * 40)
        
        developers = search_results.get("developers", [])
        # Ensure candidates are sorted by relevance score (descending)
        developers = sorted(
            developers, 
            key=lambda d: d.get("relevance_score", 0), 
            reverse=True
        )
        
        if developers:
            for idx, dev in enumerate(developers[:10], 1):
                lines.append(f"\n{idx}. @{dev.get('username', 'N/A')}")
                if dev.get("name") and dev.get("name") != dev.get("username"):
                    lines.append(f"   Name: {dev['name']}")
                lines.append(f"   Followers: {dev.get('followers', 0):,} | Repos: {dev.get('public_repos', 0)}")
                lines.append(f"   Score: {dev.get('relevance_score', 0):.0f}/100")
                if dev.get("matching_skills"):
                    lines.append(f"   Skills: {', '.join(dev['matching_skills'])}")
                lines.append(f"   URL: {dev.get('html_url', 'N/A')}")
        else:
            lines.append("\nNo matching candidates found.")
        
        # Repositories
        lines.append("\n\nTOP MATCHING REPOSITORIES")
        lines.append("-" * 40)
        
        repos = search_results.get("repositories", [])[:5]
        
        for repo in repos:
            lines.append(f"\n- {repo.get('full_name', 'N/A')}")
            lines.append(f"  Stars: {repo.get('stars', 0):,} | Forks: {repo.get('forks', 0):,}")
            lines.append(f"  Language: {repo.get('language', 'N/A')}")
        
        # Add recommendations using agentic behavior
        try:
            print(f"  [FormatterAgent] 🤖 Generating recommendations using agentic behavior...")
            recommendations = self._generate_recommendations(job_analysis, search_results)
            if recommendations:
                lines.append("\n\nRECOMMENDATIONS")
                lines.append("-" * 40)
                lines.append(recommendations)
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate recommendations: {e}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def _format_json(self, job_analysis: dict, search_results: dict) -> str:
        """
        Format results as JSON using agentic behavior for intelligent insights.
        
        Uses the AutoGen agent to generate professional summaries, insights, and recommendations
        that are included in the structured JSON response.
        """
        output = {
            "report_type": "github_talent_matcher",
            "generated_at": datetime.now().isoformat(),
            "job_analysis": job_analysis,
            "search_results": search_results
        }
        
        # Add AI-generated content using agentic behavior
        try:
            print(f"  [FormatterAgent] 🤖 Generating executive summary using agentic behavior...")
            executive_summary = self.generate_llm_summary(job_analysis, search_results)
            output["executive_summary"] = executive_summary
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate agentic summary: {e}")
            # Fallback to simple summary
            developers = search_results.get("developers", [])
            repos = search_results.get("repositories", [])
            output["executive_summary"] = (
                f"This report identifies {len(developers)} potential candidates "
                f"and {len(repos)} relevant repositories based on the job requirements."
            )
        
        # Generate candidate highlights for top candidates
        try:
            developers = search_results.get("developers", [])
            top_candidates = sorted(
                developers,
                key=lambda d: d.get("relevance_score", 0),
                reverse=True
            )[:5]
            
            if top_candidates:
                print(f"  [FormatterAgent] 🤖 Generating candidate highlights using agentic behavior...")
                candidate_highlights = self._generate_candidate_highlights(job_analysis, top_candidates)
                if candidate_highlights:
                    output["candidate_highlights"] = candidate_highlights
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate candidate highlights: {e}")
        
        # Generate recommendations
        try:
            print(f"  [FormatterAgent] 🤖 Generating recommendations using agentic behavior...")
            recommendations = self._generate_recommendations(job_analysis, search_results)
            if recommendations:
                output["recommendations"] = recommendations
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate recommendations: {e}")
        
        return json.dumps(output, indent=2)
    
    def print_report(self, job_analysis: dict, search_results: dict) -> None:
        """
        Print a formatted report to the console using Rich.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            search_results: Dictionary from SearchResults.to_dict()
        """
        self._format_rich(job_analysis, search_results)
    
    def generate_llm_summary(self, job_analysis: dict, search_results: dict) -> str:
        """
        Use the agentic agent to generate a natural language summary of the results.
        
        This method uses the AutoGen agent to autonomously analyze the results
        and generate a professional summary with recommendations.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            search_results: Dictionary from SearchResults.to_dict()
            
        Returns:
            str: Natural language summary
        """
        prompt = f"""Based on the following job requirements and candidate search results, 
provide a brief professional summary (2-3 paragraphs) with recommendations.

JOB REQUIREMENTS:
{json.dumps(job_analysis, indent=2)}

SEARCH RESULTS:
- Found {len(search_results.get('developers', []))} candidates
- Found {search_results.get('total_repos_found', 0)} matching repositories

TOP CANDIDATES:
{json.dumps(search_results.get('developers', [])[:5], indent=2)}

Please provide:
1. A summary of how well the search matched the requirements
2. Highlights of the top 2-3 candidates
3. A recommendation for next steps
"""

        try:
            print(f"  [FormatterAgent] 🤖 Generating summary using agentic behavior...")
            
            # Clear agent history for fresh start
            self.agent.clear_history()
            
            # Create a user proxy agent to initiate conversation
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            # Initiate chat - the agent will generate the summary
            chat_result = user_proxy.initiate_chat(
                recipient=self.agent,
                message=prompt,
                max_turns=1,
                silent=False
            )
            
            # Extract the agent's response from conversation history
            agent_response = None
            
            # Method 1: Check user_proxy's chat_messages (most reliable)
            if hasattr(user_proxy, 'chat_messages'):
                if self.agent in user_proxy.chat_messages:
                    messages = user_proxy.chat_messages[self.agent]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            # Method 2: Check agent's chat_messages
            if not agent_response and hasattr(self.agent, 'chat_messages'):
                if user_proxy in self.agent.chat_messages:
                    messages = self.agent.chat_messages[user_proxy]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            # Method 3: Check _oai_messages
            if not agent_response and hasattr(self.agent, '_oai_messages'):
                if user_proxy in self.agent._oai_messages:
                    messages = self.agent._oai_messages[user_proxy]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            # Method 4: Check chat_result.chat_history
            if not agent_response and hasattr(chat_result, 'chat_history'):
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "assistant" and content and content.strip() and content != prompt:
                            agent_response = content
                            break
            
            # Method 5: Check user_proxy's _oai_messages
            if not agent_response and hasattr(user_proxy, '_oai_messages'):
                if self.agent in user_proxy._oai_messages:
                    messages = user_proxy._oai_messages[self.agent]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip() and content != prompt:
                                agent_response = content
                                break
            
            # Validate we got a real response
            if not agent_response or agent_response.strip() == prompt.strip():
                raise ValueError("No valid response received from agent")
            
            if agent_response:
                print(f"  [FormatterAgent] ✓ Summary generated successfully")
                return agent_response.strip()
            else:
                raise ValueError("No response received from agent")
            
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Agentic summary generation failed: {e}")
            print(f"  [FormatterAgent] ⚠ Falling back to direct LLM call...")
            # Fallback to direct LLM call
            return self._fallback_generate_summary(job_analysis, search_results)
    
    def _generate_candidate_highlights(self, job_analysis: dict, top_candidates: list) -> str:
        """
        Use agentic behavior to generate highlights for top candidates.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            top_candidates: List of top candidate dictionaries
            
        Returns:
            str: Formatted candidate highlights
        """
        if not top_candidates:
            return ""
        
        prompt = f"""Based on the job requirements and these top candidates, provide a brief highlight (1-2 sentences per candidate) explaining why each candidate is a strong match.

JOB REQUIREMENTS:
{json.dumps(job_analysis, indent=2)}

TOP CANDIDATES:
{json.dumps(top_candidates, indent=2)}

Provide highlights in markdown format with bullet points, one per candidate. Be specific about their strengths and relevance to the role.
"""

        try:
            # Clear agent history
            self.agent.clear_history()
            
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            chat_result = user_proxy.initiate_chat(
                recipient=self.agent,
                message=prompt,
                max_turns=1,
                silent=False
            )
            
            # Extract response
            agent_response = None
            
            if hasattr(user_proxy, 'chat_messages'):
                if self.agent in user_proxy.chat_messages:
                    messages = user_proxy.chat_messages[self.agent]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            if not agent_response and hasattr(self.agent, 'chat_messages'):
                if user_proxy in self.agent.chat_messages:
                    messages = self.agent.chat_messages[user_proxy]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            if agent_response and agent_response.strip() != prompt.strip():
                return agent_response.strip()
            
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate candidate highlights: {e}")
        
        return ""
    
    def _generate_recommendations(self, job_analysis: dict, search_results: dict) -> str:
        """
        Use agentic behavior to generate actionable recommendations.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            search_results: Dictionary from SearchResults.to_dict()
            
        Returns:
            str: Formatted recommendations
        """
        prompt = f"""Based on the job requirements and search results, provide actionable recommendations for next steps in the hiring process.

JOB REQUIREMENTS:
{json.dumps(job_analysis, indent=2)}

SEARCH RESULTS:
- Found {len(search_results.get('developers', []))} candidates
- Found {search_results.get('total_repos_found', 0)} matching repositories

TOP CANDIDATES:
{json.dumps(search_results.get('developers', [])[:5], indent=2)}

Provide 3-5 specific, actionable recommendations in markdown format with bullet points. Focus on:
1. Which candidates to prioritize
2. What to look for in interviews
3. Any skill gaps to address
4. Next steps in the hiring process
"""

        try:
            # Clear agent history
            self.agent.clear_history()
            
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            chat_result = user_proxy.initiate_chat(
                recipient=self.agent,
                message=prompt,
                max_turns=1,
                silent=False
            )
            
            # Extract response
            agent_response = None
            
            if hasattr(user_proxy, 'chat_messages'):
                if self.agent in user_proxy.chat_messages:
                    messages = user_proxy.chat_messages[self.agent]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            if not agent_response and hasattr(self.agent, 'chat_messages'):
                if user_proxy in self.agent.chat_messages:
                    messages = self.agent.chat_messages[user_proxy]
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "assistant" and content and content.strip():
                                agent_response = content
                                break
            
            if agent_response and agent_response.strip() != prompt.strip():
                return agent_response.strip()
            
        except Exception as e:
            print(f"  [FormatterAgent] ⚠ Could not generate recommendations: {e}")
        
        return ""
    
    def _fallback_generate_summary(self, job_analysis: dict, search_results: dict) -> str:
        """
        Fallback method using direct LLM call if agentic approach fails.
        
        Args:
            job_analysis: Dictionary from JobAnalysis.to_dict()
            search_results: Dictionary from SearchResults.to_dict()
            
        Returns:
            str: Natural language summary
        """
        prompt = f"""Based on the following job requirements and candidate search results, 
provide a brief professional summary (2-3 paragraphs) with recommendations.

JOB REQUIREMENTS:
{json.dumps(job_analysis, indent=2)}

SEARCH RESULTS:
- Found {len(search_results.get('developers', []))} candidates
- Found {search_results.get('total_repos_found', 0)} matching repositories

TOP CANDIDATES:
{json.dumps(search_results.get('developers', [])[:5], indent=2)}

Please provide:
1. A summary of how well the search matched the requirements
2. Highlights of the top 2-3 candidates
3. A recommendation for next steps
"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {e}"


# ==============================================
# FACTORY FUNCTION
# ==============================================

def create_formatter_agent(
    model: str = "gpt-4o",
    temperature: float = 0.3
) -> FormatterAgent:
    """
    Factory function to create a FormatterAgent.
    
    Args:
        model: The LLM model to use
        temperature: Generation temperature
        
    Returns:
        FormatterAgent: Configured formatter agent
    """
    return FormatterAgent(model=model, temperature=temperature)


# ==============================================
# MODULE TEST
# ==============================================

if __name__ == "__main__":
    print("Testing FormatterAgent...")
    print("-" * 50)
    
    # Sample data
    job_analysis = {
        "title": "Senior Python Developer",
        "skills": ["Python", "Django", "PostgreSQL", "AWS"],
        "experience_years": 5,
        "experience_level": "senior",
        "tech_stack": ["Python", "Django", "Redis"],
        "frameworks": ["Django", "FastAPI"],
        "summary": "Looking for experienced Python developer"
    }
    
    search_results = {
        "developers": [
            {
                "username": "example_dev",
                "name": "Example Developer",
                "html_url": "https://github.com/example_dev",
                "followers": 1500,
                "public_repos": 45,
                "matching_skills": ["Python", "Django"],
                "relevance_score": 75.5
            }
        ],
        "repositories": [
            {
                "full_name": "django/django",
                "html_url": "https://github.com/django/django",
                "stars": 75000,
                "forks": 30000,
                "language": "Python"
            }
        ],
        "total_repos_found": 10000
    }
    
    agent = create_formatter_agent()
    
    # Test different formats
    print("\n=== MARKDOWN FORMAT ===")
    print(agent.format_results(job_analysis, search_results, "markdown")[:500])
    
    print("\n=== TEXT FORMAT ===")
    print(agent.format_results(job_analysis, search_results, "text")[:500])

