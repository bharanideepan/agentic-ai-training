#!/usr/bin/env python3
"""
Agentic GitHub Matcher - Main Application
==========================================

This is the main entry point for the Agentic GitHub Matcher workflow.
It orchestrates the multi-agent system to:
1. Analyze job descriptions for required skills
2. Search GitHub for matching developers and repositories
3. Format and present professional results

Usage:
    python app.py                          # Interactive mode
    python app.py --jd "path/to/jd.txt"   # File input mode
    python app.py --demo                   # Demo mode with sample JD

Author: Agentic AI Workflow
Version: 1.0.0
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Unicode support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Environment management
from dotenv import load_dotenv

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.markdown import Markdown

# Import our agents
from agents.analyst import AnalystAgent, create_analyst_agent
from agents.github_agent import GitHubSearchAgent, create_github_agent
from agents.formatter import FormatterAgent, create_formatter_agent


# LiteLLM for model gateway
import litellm


# ==============================================
# CONFIGURATION
# ==============================================

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console with UTF-8 support
console = Console(force_terminal=True, legacy_windows=False)

# Default model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Validate required environment variables
def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if all required variables are set
    """
    required_vars = ["OPENAI_API_KEY", "GITHUB_TOKEN"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        console.print(Panel(
            f"[red]Missing required environment variables:[/red]\n" +
            "\n".join(f"  • {var}" for var in missing) +
            "\n\n[yellow]Please copy env.template to .env and fill in your values.[/yellow]",
            title="⚠️ Configuration Error",
            border_style="red"
        ))
        return False
    
    return True


# ==============================================
# GUARDRAILS INITIALIZATION
# ==============================================

def initialize_guardrails():
    """
    Initialize Nemo Guardrails for input/output safety.
    
    This function is kept for backward compatibility but guardrails
    are now initialized automatically in guardrails/runner.py.
    """
    try:
        # Import to trigger initialization
        from guardrails.runner import _initialize_guardrails
        rails = _initialize_guardrails()
        if rails:
            console.print("[green]✓[/green] Guardrails initialized successfully")
        else:
            console.print("[yellow]⚠[/yellow] Guardrails not available, using basic validation")
        return rails
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Guardrails initialization failed: {e}")
        console.print("[dim]  Continuing with basic validation only[/dim]")
        return None


async def apply_input_guardrails(rails, text: str) -> tuple[bool, str]:
    """
    Apply input guardrails to validate user input.
    
    Uses guardrails/runner.py for validation.
    
    Args:
        rails: Not used (kept for backward compatibility)
        text: Input text to validate
        
    Returns:
        tuple: (is_safe, validated_message)
    """
    try:
        from guardrails.runner import validate_input, GuardrailsValidationError
        
        # Validate input using guardrails (async)
        validated_text = await validate_input(text)
        return True, validated_text
        
    except GuardrailsValidationError as e:
        # Validation failed - return error message
        return False, str(e)
    except Exception as e:
        # For other errors, log but allow (fail open for safety)
        console.print(f"[yellow]⚠ Guardrails check warning: {e}[/yellow]")
        return True, text


async def apply_output_guardrails(rails, output: str, context: str = "") -> tuple[bool, str]:
    """
    Apply output guardrails to validate agent responses.
    
    Uses guardrails/runner.py for validation.
    
    Args:
        rails: Not used (kept for backward compatibility)
        output: Agent output to validate
        context: Optional context (not used, kept for compatibility)
        
    Returns:
        tuple: (is_safe, validated_output)
    """
    try:
        from guardrails.runner import validate_output, GuardrailsValidationError
        
        # Validate output using guardrails (async)
        validated_output = await validate_output(output)
        return True, validated_output
        
    except GuardrailsValidationError as e:
        # Validation failed - return error message
        console.print(f"[yellow]⚠ Output validation failed: {e}[/yellow]")
        return False, str(e)
    except Exception as e:
        # For other errors, log but allow (fail open for safety)
        console.print(f"[yellow]⚠ Output guardrails validation failed: {e}[/yellow]")
        console.print("[dim]Allowing output (fail open)[/dim]")
        return True, output


# ==============================================
# WORKFLOW ORCHESTRATION
# ==============================================

class AgenticWorkflow:
    """
    Main workflow orchestrator that coordinates the three agents:
    1. AnalystAgent - Analyzes job descriptions
    2. GitHubSearchAgent - Searches GitHub for matches
    3. FormatterAgent - Formats the final output
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        guardrails=None
    ):
        """
        Initialize the workflow with all agents.
        
        Args:
            model: LLM model to use
            temperature: Generation temperature
            guardrails: Optional guardrails instance
        """
        self.model = model
        self.temperature = temperature
        self.guardrails = guardrails
        
        # Initialize agents
        console.print("\n[bold]Initializing Agents...[/bold]")
        
        # Check API key format
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            if not api_key.startswith("sk-"):
                console.print(f"[yellow]⚠ Warning: OpenAI API key doesn't start with 'sk-'. Current format: {api_key[:10]}...[/yellow]")
            else:
                console.print(f"[green]✓[/green] OpenAI API key format looks valid")
        else:
            console.print(f"[red]✗[/red] OPENAI_API_KEY not set!")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading AnalystAgent...", total=3)
            console.print(f"  [dim]Creating AnalystAgent with model: {model}[/dim]")
            self.analyst = create_analyst_agent(model=model, temperature=0.3)
            console.print(f"  [green]✓[/green] AnalystAgent initialized")
            progress.update(task, advance=1, description="Loading GitHubSearchAgent...")
            
            console.print(f"  [dim]Creating GitHubSearchAgent with model: {model}[/dim]")
            self.github_agent = create_github_agent(model=model, temperature=0.5)
            console.print(f"  [green]✓[/green] GitHubSearchAgent initialized")
            progress.update(task, advance=1, description="Loading FormatterAgent...")
            
            console.print(f"  [dim]Creating FormatterAgent with model: {model}[/dim]")
            self.formatter = create_formatter_agent(model=model, temperature=0.3)
            console.print(f"  [green]✓[/green] FormatterAgent initialized")
            progress.update(task, advance=1, description="All agents ready!")
        
        console.print("[green]✓[/green] All agents initialized successfully\n")
    
    async def run(self, job_description: str, output_format: str = "rich") -> str:
        """
        Execute the full workflow pipeline.
        
        Workflow Steps:
        1. Validate input with guardrails
        2. Analyze job description with AnalystAgent
        3. Search GitHub with GitHubSearchAgent
        4. Format results with FormatterAgent
        
        Args:
            job_description: The job description text to process
            output_format: Output format ("rich", "markdown", "text", "json")
            
        Returns:
            str: Formatted results
        """
        # Step 0: Apply input guardrails (before AnalystAgent)
        console.print(Panel("[bold]Starting Agentic Workflow[/bold]", border_style="blue"))
        
        is_safe, message = await apply_input_guardrails(self.guardrails, job_description)
        if not is_safe:
            console.print(f"[red]❌ Input validation failed: {message}[/red]")
            raise ValueError(f"Input validation failed: {message}")
        
        # Step 1: Analyze the job description
        console.print("\n[bold cyan]Step 1/3:[/bold cyan] Analyzing Job Description...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting skills and requirements...", total=None)
            try:
                job_analysis = self.analyst.analyze(job_description)
                progress.update(task, description="Analysis complete!")
            except Exception as e:
                console.print(f"  [red]❌ Analysis failed: {e}[/red]")
                raise
        
        # Display extracted skills
        searchable_skills = job_analysis.get_searchable_skills()
        console.print(f"  [green]✓[/green] Found {len(searchable_skills)} key skills")
        if searchable_skills:
            console.print(f"  [dim]Skills: {', '.join(searchable_skills[:8])}[/dim]")
        
        # Check if analysis was successful
        if not searchable_skills:
            console.print(f"  [yellow]⚠ Warning: No skills extracted from job description[/yellow]")
            console.print(f"  [yellow]⚠ This might indicate an API key or LLM issue[/yellow]")
        
        # Step 2: Search GitHub
        console.print("\n[bold cyan]Step 2/3:[/bold cyan] Searching GitHub...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Finding matching developers and repositories...", total=None)
            # Convert job_analysis to dict for passing to search
            analysis_dict = job_analysis.to_dict()
            search_results = await self.github_agent.search(
                skills=searchable_skills,  # Use all searchable skills for better matching
                job_analysis=analysis_dict,  # Pass job analysis for enhanced scoring
                max_candidates=10
            )
            progress.update(task, description="Search complete!")
        
        console.print(f"  [green]✓[/green] Found {len(search_results.developers)} candidates")
        console.print(f"  [green]✓[/green] Found {search_results.total_repos_found:,} related repositories")
        
        # Step 3: Format results
        console.print("\n[bold cyan]Step 3/3:[/bold cyan] Formatting Results...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating professional report...", total=None)
            
            # Convert to dictionaries for formatting (analysis_dict already created above)
            results_dict = search_results.to_dict()
            
            # Generate formatted output
            formatted_output = self.formatter.format_results(
                analysis_dict,
                results_dict,
                output_format=output_format
            )
            
            # Apply output guardrails to formatted output (after FormatterAgent)
            if formatted_output:
                is_safe, validated_output = await apply_output_guardrails(
                    self.guardrails,
                    formatted_output,
                    context="Formatted job matching report"
                )
                if not is_safe:
                    console.print(f"  [yellow]⚠ Formatted output flagged by guardrails[/yellow]")
                    # Raise exception to prevent unsafe output
                    raise ValueError(f"Output validation failed: {validated_output}")
                else:
                    formatted_output = validated_output
            
            progress.update(task, description="Report ready!")
        
        # Display results (for rich format, this prints to console)
        if output_format == "rich":
            self.formatter.print_report(analysis_dict, results_dict)
        
        console.print("\n[bold green]✓ Workflow Complete![/bold green]\n")
        
        return formatted_output
    
    async def run_async(self, job_description: str, output_format: str = "rich") -> str:
        """
        Async version of the workflow.
        
        Args:
            job_description: The job description text
            output_format: Output format
            
        Returns:
            str: Formatted results
        """
        # For now, delegate to sync version
        return self.run(job_description, output_format)


# ==============================================
# SAMPLE JOB DESCRIPTIONS FOR DEMO
# ==============================================

SAMPLE_JD_FULLSTACK = """
Senior Full-Stack Developer - AI/ML Platform

Company: TechCorp Innovation Labs
Location: Remote (US/Europe)
Type: Full-time

About the Role:
We are seeking an experienced Full-Stack Developer to join our AI/ML Platform team. 
You will be building cutting-edge tools that enable data scientists and ML engineers 
to deploy and manage machine learning models at scale.

Requirements:

Technical Skills:
- 5+ years of professional software development experience
- Strong proficiency in Python and TypeScript
- Experience with React or Vue.js for frontend development
- Backend experience with FastAPI, Django, or Node.js
- Solid understanding of PostgreSQL and Redis
- Experience with Docker and Kubernetes
- Familiarity with AWS services (EC2, S3, Lambda, SageMaker)
- Understanding of ML/AI concepts and MLOps practices

Nice to Have:
- Experience with PyTorch or TensorFlow
- Knowledge of Apache Kafka or similar streaming platforms
- AWS or GCP certifications
- Contributions to open-source projects

Soft Skills:
- Excellent communication and collaboration abilities
- Strong problem-solving skills
- Ability to work in a fast-paced environment
- Mentoring and leadership experience

What We Offer:
- Competitive salary and equity
- Flexible remote work
- Learning and development budget
- Health insurance and 401k matching
"""

SAMPLE_JD_FRONTEND = """
Senior Frontend Developer - React & Angular

Company: DigitalWave Solutions
Location: Hybrid (New York / Remote)
Type: Full-time

About the Role:
We are looking for a talented Senior Frontend Developer to lead the development 
of our next-generation web applications. You will work closely with UX designers 
and backend engineers to create beautiful, responsive, and performant user interfaces.

Requirements:

Technical Skills:
- 4+ years of professional frontend development experience
- Expert-level proficiency in React.js and React ecosystem (Redux, React Query, React Router)
- Strong experience with Angular (v12+) and RxJS
- Advanced TypeScript and JavaScript (ES6+) skills
- Proficiency in HTML5, CSS3, SASS/SCSS
- Experience with modern build tools (Webpack, Vite, esbuild)
- Knowledge of responsive design and cross-browser compatibility
- Familiarity with testing frameworks (Jest, Cypress, Playwright)
- Understanding of RESTful APIs and GraphQL

Nice to Have:
- Experience with Next.js or Nuxt.js
- Knowledge of micro-frontend architecture
- Familiarity with design systems (Material UI, Tailwind CSS, Chakra UI)
- Experience with Web Components and Shadow DOM
- Understanding of web accessibility (WCAG) standards
- CI/CD experience with GitHub Actions or GitLab CI

Soft Skills:
- Strong attention to detail and pixel-perfect implementation
- Excellent communication skills for cross-team collaboration
- Ability to mentor junior developers
- Passion for staying current with frontend trends

What We Offer:
- Competitive salary ($140k - $180k)
- Stock options
- Flexible work arrangements
- Annual conference budget
- Latest MacBook Pro and equipment
"""

# Default sample JD for --demo flag
SAMPLE_JD = SAMPLE_JD_FULLSTACK


# ==============================================
# MAIN ENTRY POINT
# ==============================================

def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Agentic GitHub Matcher - Find developers matching your job requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                              # Interactive mode
  python app.py --demo                       # Demo with Full-Stack Developer JD
  python app.py --demo --demo-type frontend  # Demo with Frontend Developer JD
  python app.py --jd job.txt                 # Process job description from file
  python app.py --format markdown            # Output in markdown format
        """
    )
    parser.add_argument(
        "--jd", 
        type=str, 
        help="Path to job description file"
    )
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run demo with sample job description"
    )
    parser.add_argument(
        "--demo-type",
        type=str,
        choices=["fullstack", "frontend"],
        default="fullstack",
        help="Sample JD type for demo mode: fullstack or frontend (default: fullstack)"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["rich", "markdown", "text", "json"],
        default="rich",
        help="Output format (default: rich)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Display welcome banner
    console.print(Panel.fit(
        "[bold blue]🎯 Agentic GitHub Matcher[/bold blue]\n"
        "[dim]Find the perfect developers for your job requirements[/dim]",
        border_style="blue"
    ))
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Initialize guardrails
    console.print("\n[bold]Initializing Safety Rails...[/bold]")
    guardrails = initialize_guardrails()
    
    # Get job description
    job_description = None
    
    if args.demo:
        # Select sample JD based on demo-type argument
        if args.demo_type == "frontend":
            job_description = SAMPLE_JD_FRONTEND
            console.print("\n[yellow]Running in DEMO mode with Frontend Developer job description[/yellow]")
        else:
            job_description = SAMPLE_JD_FULLSTACK
            console.print("\n[yellow]Running in DEMO mode with Full-Stack Developer job description[/yellow]")
        
    elif args.jd:
        # Read from file
        jd_path = Path(args.jd)
        if jd_path.exists():
            job_description = jd_path.read_text(encoding="utf-8")
            console.print(f"\n[green]Loaded job description from: {args.jd}[/green]")
        else:
            console.print(f"[red]Error: File not found: {args.jd}[/red]")
            sys.exit(1)
    
    else:
        # Interactive mode
        console.print("\n[bold]Enter your job description below.[/bold]")
        console.print("[dim]Type 'END' on a new line when finished, or 'DEMO' to use sample JD.[/dim]\n")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":
                    break
                elif line.strip().upper() == "DEMO":
                    job_description = SAMPLE_JD
                    console.print("[yellow]Using sample job description[/yellow]")
                    break
                lines.append(line)
            except EOFError:
                break
        
        if not job_description:
            job_description = "\n".join(lines)
    
    # Validate we have input
    if not job_description or not job_description.strip():
        console.print("[red]Error: No job description provided[/red]")
        sys.exit(1)
    
    # Initialize and run workflow
    try:
        workflow = AgenticWorkflow(
            model=args.model,
            guardrails=guardrails
        )
        
        result = asyncio.run(workflow.run(
            job_description=job_description,
            output_format=args.format
        ))
        
        # If not rich format, print the result
        if args.format != "rich":
            console.print("\n[bold]Output:[/bold]")
            if args.format == "markdown":
                console.print(Markdown(result))
            else:
                console.print(result)
        
        # Save to file option
        if args.format in ["markdown", "json"]:
            save_path = f"report.{args.format.replace('markdown', 'md')}"
            Path(save_path).write_text(result, encoding="utf-8")
            console.print(f"\n[green]Report saved to: {save_path}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Workflow interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error during workflow execution: {e}[/red]")
        raise


# ==============================================
# SCRIPT ENTRY
# ==============================================

if __name__ == "__main__":
    main()

