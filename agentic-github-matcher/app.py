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

# Configuration
from config import get_config

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
    
    Note: This function sets up guardrails configuration.
    In production, this would load and configure the rails.yaml file.
    """
    try:
        from nemoguardrails import LLMRails, RailsConfig
        
        config_path = Path(__file__).parent / "guardrails"
        
        if config_path.exists():
            config = RailsConfig.from_path(str(config_path))
            rails = LLMRails(config)
            console.print("[green]✓[/green] Guardrails initialized successfully")
            return rails
        else:
            console.print("[yellow]⚠[/yellow] Guardrails config not found, running without safety rails")
            return None
            
    except ImportError:
        console.print("[yellow]⚠[/yellow] Nemo Guardrails not installed, running without safety rails")
        return None
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Guardrails initialization failed: {e}")
        return None


def detect_and_mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Detect and mask PII (Personally Identifiable Information) in text.
    
    Detects:
    - Email addresses
    - Phone numbers (US format)
    - Credit card numbers
    - SSN (Social Security Numbers)
    - IP addresses
    - API keys (common patterns)
    
    Args:
        text: Input text to scan
        
    Returns:
        tuple: (masked_text, detected_pii_types)
    """
    import re
    
    detected_types = []
    masked_text = text
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, masked_text):
        detected_types.append("email")
        masked_text = re.sub(email_pattern, '[EMAIL_REDACTED]', masked_text)
    
    # Phone numbers (US format: (123) 456-7890, 123-456-7890, 123.456.7890, etc.)
    phone_pattern = r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    if re.search(phone_pattern, masked_text):
        detected_types.append("phone")
        masked_text = re.sub(phone_pattern, '[PHONE_REDACTED]', masked_text)
    
    # Credit card numbers (13-19 digits, with optional spaces/dashes)
    cc_pattern = r'\b(?:\d[ -]*?){13,19}\b'
    # Additional validation: check if it looks like a credit card
    cc_matches = re.finditer(cc_pattern, masked_text)
    for match in cc_matches:
        digits = re.sub(r'[^\d]', '', match.group())
        if 13 <= len(digits) <= 19:
            detected_types.append("credit_card")
            masked_text = masked_text[:match.start()] + '[CREDIT_CARD_REDACTED]' + masked_text[match.end():]
            break
    
    # SSN (Social Security Number: XXX-XX-XXXX)
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    if re.search(ssn_pattern, masked_text):
        detected_types.append("ssn")
        masked_text = re.sub(ssn_pattern, '[SSN_REDACTED]', masked_text)
    
    # IP addresses (IPv4)
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    if re.search(ip_pattern, masked_text):
        detected_types.append("ip_address")
        masked_text = re.sub(ip_pattern, '[IP_REDACTED]', masked_text)
    
    # API keys (common patterns: sk-, xoxb-, etc.)
    api_key_patterns = [
        r'\bsk-[a-zA-Z0-9]{32,}\b',  # OpenAI, Stripe
        r'\bxoxb-[a-zA-Z0-9-]+\b',   # Slack
        r'\bghp_[a-zA-Z0-9]{36}\b',  # GitHub
        r'\bAKIA[0-9A-Z]{16}\b',     # AWS Access Key
    ]
    for pattern in api_key_patterns:
        if re.search(pattern, masked_text, re.IGNORECASE):
            detected_types.append("api_key")
            masked_text = re.sub(pattern, '[API_KEY_REDACTED]', masked_text, flags=re.IGNORECASE)
    
    return masked_text, detected_types


def apply_input_guardrails(rails, text: str) -> tuple[bool, str]:
    """
    Apply input guardrails to validate user input.
    
    Performs:
    1. PII Detection & Masking - Detects and masks sensitive information
    2. Pattern-based validation - Blocks malicious/jailbreak patterns
    
    Args:
        rails: Initialized guardrails instance (optional, for future LLM-based validation)
        text: Input text to validate
        
    Returns:
        tuple: (is_safe, sanitized_message)
    """
    try:
        # Get configuration
        config = get_config()
        
        # Step 1: Detect and mask PII (if enabled)
        if config.guardrails.pii_detection_enabled:
            masked_text, detected_pii = detect_and_mask_pii(text)
            
            if detected_pii:
                if config.guardrails.pii_mask_mode == "block":
                    return False, f"Input blocked: PII detected ({', '.join(detected_pii)})"
                else:
                    console.print(f"[yellow]⚠ PII detected and masked: {', '.join(detected_pii)}[/yellow]")
        else:
            masked_text = text
            detected_pii = []
        
        # Step 2: Get blocked patterns from configuration
        blocked_patterns = config.guardrails.blocked_patterns
        
        # Step 3: Apply pattern-based validation on masked text
        text_lower = masked_text.lower()
        for pattern in blocked_patterns:
            if pattern in text_lower:
                return False, f"Input blocked: contains restricted pattern '{pattern}'"
        
        # In a full implementation, this would also use rails.generate() for LLM-based validation
        # For now, we do basic pattern-based validation with PII masking
        
        return True, masked_text
        
    except Exception as e:
        console.print(f"[yellow]Guardrails check warning: {e}[/yellow]")
        return True, text


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
    
    def run(self, job_description: str, output_format: str = "rich") -> str:
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
        # Step 0: Apply input guardrails
        console.print(Panel("[bold]Starting Agentic Workflow[/bold]", border_style="blue"))
        
        is_safe, message = apply_input_guardrails(self.guardrails, job_description)
        if not is_safe:
            console.print(f"[red]❌ {message}[/red]")
            return message
        
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
            search_results = self.github_agent.search(
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
        
        result = workflow.run(
            job_description=job_description,
            output_format=args.format
        )
        
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

