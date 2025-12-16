"""
Analyst Agent
=============

This module implements the AnalystAgent, responsible for analyzing
job descriptions and extracting key information such as:
- Required skills and technologies
- Experience requirements
- Preferred qualifications
- Tech stack details

The agent uses GPT-4o through LiteLLM to perform intelligent
analysis of job description text.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass, field

# AutoGen imports (pyautogen 0.2.x classic API)
from autogen import ConversableAgent, UserProxyAgent

# LiteLLM for model gateway
import litellm


# ==============================================
# DATA STRUCTURES
# ==============================================

@dataclass
class JobAnalysis:
    """
    Structured representation of analyzed job description.
    """
    title: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: Optional[int] = None
    experience_level: str = ""  # junior, mid, senior, lead, principal
    tech_stack: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    databases: list[str] = field(default_factory=list)
    cloud_platforms: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    domain_knowledge: list[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "experience_level": self.experience_level,
            "tech_stack": self.tech_stack,
            "frameworks": self.frameworks,
            "databases": self.databases,
            "cloud_platforms": self.cloud_platforms,
            "soft_skills": self.soft_skills,
            "certifications": self.certifications,
            "domain_knowledge": self.domain_knowledge,
            "summary": self.summary
        }
    
    def get_searchable_skills(self) -> list[str]:
        """
        Get a consolidated list of skills suitable for GitHub search.
        Combines technical skills, frameworks, and tech stack.
        """
        searchable = []
        searchable.extend(self.skills)
        searchable.extend(self.tech_stack)
        searchable.extend(self.frameworks)
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in searchable if not (x.lower() in seen or seen.add(x.lower()))]


# ==============================================
# ANALYST AGENT SYSTEM PROMPT
# ==============================================

ANALYST_SYSTEM_PROMPT = """You are an expert Job Description Analyst Agent.

Your role is to carefully analyze job descriptions and extract structured information about the requirements.

When given a job description, you must extract and return:

1. **Job Title**: The position title
2. **Skills**: Programming languages, tools, and technical skills
3. **Experience**: Years of experience and seniority level
4. **Tech Stack**: Specific technologies mentioned
5. **Frameworks**: Web frameworks, ML frameworks, etc.
6. **Databases**: SQL, NoSQL, data stores mentioned
7. **Cloud Platforms**: AWS, Azure, GCP, etc.
8. **Soft Skills**: Communication, leadership, collaboration skills
9. **Certifications**: Any required or preferred certifications
10. **Domain Knowledge**: Industry-specific knowledge required
11. **Summary**: A brief summary of the ideal candidate

CRITICAL: You MUST return ONLY a valid JSON object with these exact keys. Do not include any markdown formatting, code blocks, or explanatory text. Return ONLY the JSON:

{
    "title": "string",
    "skills": ["list", "of", "skills"],
    "experience_years": number or null,
    "experience_level": "junior|mid|senior|lead|principal",
    "tech_stack": ["list", "of", "technologies"],
    "frameworks": ["list", "of", "frameworks"],
    "databases": ["list", "of", "databases"],
    "cloud_platforms": ["list", "of", "platforms"],
    "soft_skills": ["list", "of", "soft_skills"],
    "certifications": ["list", "of", "certifications"],
    "domain_knowledge": ["list", "of", "domains"],
    "summary": "Brief summary of ideal candidate"
}

Be thorough but precise. Only extract information explicitly mentioned or strongly implied in the job description.
"""


# ==============================================
# ANALYST AGENT CLASS
# ==============================================

class AnalystAgent:
    """
    The AnalystAgent analyzes job descriptions and extracts
    structured skill and requirement information using agentic behavior.
    
    This agent uses AutoGen's ConversableAgent with LiteLLM to autonomously
    analyze job descriptions. The agent uses true agentic behavior where the
    LLM agent decides how to process the input and format the structured output,
    demonstrating true agentic AI behavior through UserProxyAgent.initiate_chat().
    
    LiteLLM Integration:
    - If LITELLM_PROXY_URL environment variable is set, all LLM calls route
      through the LiteLLM proxy/gateway, enabling:
      * Model routing via litellm.config.yaml
      * Automatic fallbacks
      * Cost tracking
      * Centralized model management
    - If LITELLM_PROXY_URL is not set, AutoGen uses OpenAI directly (bypasses LiteLLM)
    
    To use LiteLLM:
    1. Start LiteLLM proxy: litellm --config litellm.config.yaml --port 4000
    2. Set environment variable: export LITELLM_PROXY_URL=http://localhost:4000
    
    The agent makes autonomous decisions about:
    - How to analyze the job description
    - What information to extract
    - How to structure the JSON output
    
    No direct LLM calls are made - all LLM interactions go through the agent.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        name: str = "AnalystAgent",
        use_litellm_config: bool = True
    ):
        """
        Initialize the AnalystAgent.
        
        Args:
            model: The LLM model to use (must match model_name in litellm.config.yaml if using proxy)
            temperature: Temperature for generation (lower = more focused)
                       When using LiteLLM proxy, this can override config temperature if use_litellm_config=False
            name: Name identifier for the agent
            use_litellm_config: If True and using LiteLLM proxy, omit temperature to use config value.
                               If False, pass temperature to override config. Default: True
        """
        self.model = model
        self.temperature = temperature
        self.name = name
        self.use_litellm_config = use_litellm_config
        
        # Configure LiteLLM settings
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Check if LiteLLM proxy is configured
        # LiteLLM proxy allows routing through litellm.config.yaml
        litellm_proxy_url = os.getenv("LITELLM_PROXY_URL", "")
        
        # Configure LiteLLM to use the specified model
        # LiteLLM will handle routing, fallbacks, and model switching
        litellm.set_verbose = False  # Set to True for debugging
        
        # Log API key status (without exposing the key)
        if api_key:
            if api_key.startswith("sk-"):
                print(f"  [AnalystAgent] ✓ API key format valid")
            else:
                print(f"  [AnalystAgent] ⚠ API key format unusual: {api_key[:10]}...")
        else:
            print(f"  [AnalystAgent] ⚠ No API key found")
        
        # Configure AutoGen to use LiteLLM via OpenAI-compatible interface
        # If LiteLLM proxy is available, route through it
        # Otherwise, AutoGen will use OpenAI directly (bypasses LiteLLM)
        
        # If LiteLLM proxy is configured, route through it
        # LiteLLM will handle configuration from litellm.config.yaml
        if litellm_proxy_url:
            # When using LiteLLM proxy, minimal config is needed
            # LiteLLM proxy handles:
            # - Model routing (from litellm.config.yaml)
            # - API key management (from config or environment)
            # - Temperature and other params (from config, can be overridden per request)
            # - Fallbacks and retries (from config)
            llm_config_dict = {
                "model": model,  # Model name must match litellm.config.yaml model_name
                "api_type": "openai",
                "base_url": litellm_proxy_url.rstrip("/"),  # Route through LiteLLM proxy
                # API key: LiteLLM proxy can handle auth from config, but we pass it
                # in case proxy needs it for forwarding to OpenAI
                "api_key": api_key,
            }
            
            # Temperature handling:
            # - If use_litellm_config=True: Omit temperature to use value from litellm.config.yaml
            # - If use_litellm_config=False: Pass temperature to override config value
            if not use_litellm_config:
                llm_config_dict["temperature"] = temperature
                print(f"  [AnalystAgent] ✓ Using LiteLLM proxy: {litellm_proxy_url}")
                print(f"  [AnalystAgent] ✓ All LLM calls will route through LiteLLM gateway")
                print(f"  [AnalystAgent] ℹ Temperature ({temperature}) will override config value")
            else:
                print(f"  [AnalystAgent] ✓ Using LiteLLM proxy: {litellm_proxy_url}")
                print(f"  [AnalystAgent] ✓ All LLM calls will route through LiteLLM gateway")
                print(f"  [AnalystAgent] ℹ Using temperature from litellm.config.yaml (not passing explicit value)")
            
            print(f"  [AnalystAgent] ℹ Model routing, fallbacks, and other config from litellm.config.yaml")
        else:
            # Direct OpenAI configuration (bypasses LiteLLM)
            llm_config_dict = {
                "model": model,
                "api_type": "openai",
                "api_key": api_key,
                "temperature": temperature,
            }
            print(f"  [AnalystAgent] ⚠ LiteLLM proxy not configured (LITELLM_PROXY_URL not set)")
            print(f"  [AnalystAgent] ⚠ Will use OpenAI directly (bypasses LiteLLM)")
            print(f"  [AnalystAgent] ℹ To use LiteLLM, set LITELLM_PROXY_URL environment variable")
            print(f"  [AnalystAgent] ℹ Example: LITELLM_PROXY_URL=http://localhost:4000")
        
        self.llm_config = {
            "config_list": [llm_config_dict],
            "timeout": 120,
        }
        
        # Create the AutoGen agent for agentic behavior
        # The agent will autonomously analyze job descriptions and extract structured information
        self.agent = ConversableAgent(
            name=name,
            system_message=ANALYST_SYSTEM_PROMPT,
            llm_config=self.llm_config,
            human_input_mode="NEVER",  # Fully automated
            max_consecutive_auto_reply=1,
        )
        
        print(f"  [AnalystAgent] ✓ Using agentic behavior with AutoGen")
        if litellm_proxy_url:
            print(f"  [AnalystAgent] ✓ Model: {model} (via LiteLLM gateway)")
        else:
            print(f"  [AnalystAgent] ✓ Model: {model} (direct OpenAI - LiteLLM not configured)")
        
        # Store proxy URL for verification
        self.litellm_proxy_url = litellm_proxy_url
    
    def get_litellm_status(self) -> dict:
        """
        Get the LiteLLM configuration status for verification.
        
        Returns:
            dict: Status information including:
                - using_litellm: bool - Whether LiteLLM proxy is configured
                - proxy_url: str - LiteLLM proxy URL if configured
                - base_url: str - base_url in llm_config if set
        """
        status = {
            "using_litellm": bool(self.litellm_proxy_url),
            "proxy_url": self.litellm_proxy_url or "",
            "base_url": "",
        }
        
        if hasattr(self, 'llm_config'):
            config_list = self.llm_config.get("config_list", [])
            if config_list:
                config = config_list[0]
                status["base_url"] = config.get("base_url", "")
        
        return status
    
    def analyze(self, job_description: str) -> JobAnalysis:
        """
        Analyze a job description and extract structured information using agentic behavior.
        
        This method uses true agentic behavior - the LLM agent autonomously analyzes
        the job description and extracts structured information, demonstrating true
        agentic AI behavior through AutoGen's ConversableAgent.
        
        Args:
            job_description: The raw job description text
            
        Returns:
            JobAnalysis: Structured analysis results
        """
        print(f"  [AnalystAgent] 🤖 Starting agentic analysis (model: {self.model})")
        print(f"  [AnalystAgent] ✓ Using LLM agent for autonomous analysis")
        print(f"  [EXECUTION MODE] 🤖 AGENTIC MODE - Agent will analyze and extract information")
        
        # Build the analysis prompt for the agent
        analysis_prompt = f"""Please analyze the following job description and extract all relevant information.

JOB DESCRIPTION:
---
{job_description}
---

Return your analysis as a valid JSON object with these exact keys:
{{
    "title": "string",
    "skills": ["list", "of", "skills"],
    "experience_years": number or null,
    "experience_level": "junior|mid|senior|lead|principal",
    "tech_stack": ["list", "of", "technologies"],
    "frameworks": ["list", "of", "frameworks"],
    "databases": ["list", "of", "databases"],
    "cloud_platforms": ["list", "of", "platforms"],
    "soft_skills": ["list", "of", "soft_skills"],
    "certifications": ["list", "of", "certifications"],
    "domain_knowledge": ["list", "of", "domains"],
    "summary": "Brief summary of ideal candidate"
}}

Be thorough but precise. Only extract information explicitly mentioned or strongly implied in the job description.
CRITICAL: Return ONLY the JSON object, no markdown formatting or code blocks."""

        try:
            # Clear agent history for fresh start
            self.agent.clear_history()
            
            print("  🤖 Agent is analyzing the job description...")
            
            # Create a user proxy agent to initiate conversation
            # In AutoGen, when an agent responds, the UserProxyAgent handles the interaction
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,  # Allow one auto-reply from the agent
                code_execution_config=False,  # Don't execute code, just LLM responses
            )
            
            # Initiate chat - the agent will analyze and respond
            # The agent autonomously decides how to process and structure the analysis
            chat_result = user_proxy.initiate_chat(
                recipient=self.agent,
                message=analysis_prompt,
                max_turns=1,  # One turn: agent analyzes and responds
                silent=False
            )
            
            print(f"  [AnalystAgent] ✓ Agent analysis complete")
            
            # Extract JSON from agent's conversation history
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
            
            # Extract JSON from agent's response
            content = self._extract_json_from_messages(messages)
            
            if not content:
                # Also try to extract from chat result if available
                if hasattr(chat_result, 'chat_history'):
                    content = self._extract_json_from_messages(chat_result.chat_history)
            
            if not content:
                raise ValueError("Could not extract JSON from agent's response")
            
            # Parse JSON response
            print(f"  [AnalystAgent] Parsing JSON response...")
            data = json.loads(content)
            print(f"  [AnalystAgent] ✓ Successfully extracted: {len(data.get('skills', []))} skills")
            
            # Create JobAnalysis object
            return JobAnalysis(
                title=data.get("title", ""),
                skills=data.get("skills", []),
                experience_years=data.get("experience_years"),
                experience_level=data.get("experience_level", ""),
                tech_stack=data.get("tech_stack", []),
                frameworks=data.get("frameworks", []),
                databases=data.get("databases", []),
                cloud_platforms=data.get("cloud_platforms", []),
                soft_skills=data.get("soft_skills", []),
                certifications=data.get("certifications", []),
                domain_knowledge=data.get("domain_knowledge", []),
                summary=data.get("summary", "")
            )
        except json.JSONDecodeError as e:
            print(f"  [AnalystAgent] ❌ Error parsing JSON response: {e}")
            print(f"  [AnalystAgent] Response content preview: {content[:200] if 'content' in locals() else 'N/A'}")
            return JobAnalysis(summary=f"Error parsing analysis response: {e}")
        except Exception as e:
            import traceback
            print(f"  [AnalystAgent] ❌ Error during agentic analysis: {e}")
            print(f"  [AnalystAgent] Traceback: {traceback.format_exc()}")
            return JobAnalysis(summary=f"Error analyzing job description: {e}")
    
    def _extract_json_from_messages(self, messages: list) -> str:
        """
        Extract JSON content from agent's conversation messages.
        
        The agent may return JSON wrapped in markdown code blocks or as plain text.
        This method extracts the JSON content from the agent's response.
        
        Args:
            messages: List of messages from agent conversation
            
        Returns:
            str: Extracted JSON content, or empty string if not found
        """
        import re
        
        # Look for agent's response in messages
        for msg in reversed(messages):  # Start from most recent
            if not isinstance(msg, dict):
                continue
            
            # Get content from message
            content = msg.get("content", "")
            if not content:
                continue
            
            # If content is a string, try to extract JSON
            if isinstance(content, str):
                # Try to find JSON in markdown code blocks first
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    return json_match.group(1)
                
                # Try to find JSON object directly
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Validate it's valid JSON
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        continue
            
            # If content is already a dict, convert to JSON string
            elif isinstance(content, dict):
                try:
                    return json.dumps(content)
                except (TypeError, ValueError):
                    continue
        
        return ""
    
    async def analyze_async(self, job_description: str) -> JobAnalysis:
        """
        Async version of analyze method (delegates to sync version for now).
        
        Args:
            job_description: The raw job description text
            
        Returns:
            JobAnalysis: Structured analysis results
        """
        # For now, delegate to sync version
        # AutoGen's initiate_chat is sync, so we use the sync analyze method
        return self.analyze(job_description)


# ==============================================
# FACTORY FUNCTION
# ==============================================

def create_analyst_agent(
    model: str = "gpt-4o",
    temperature: float = 0.3,
    use_litellm_config: bool = True
) -> AnalystAgent:
    """
    Factory function to create an AnalystAgent.
    
    Args:
        model: The LLM model to use (must match model_name in litellm.config.yaml if using proxy)
        temperature: Generation temperature (only used if use_litellm_config=False when using proxy)
        use_litellm_config: If True and using LiteLLM proxy, use config values instead of explicit params.
                          Default: True (recommended to let LiteLLM handle config)
        
    Returns:
        AnalystAgent: Configured analyst agent
    """
    return AnalystAgent(
        model=model, 
        temperature=temperature,
        use_litellm_config=use_litellm_config
    )


# ==============================================
# MODULE TEST
# ==============================================

if __name__ == "__main__":
    # Test the analyst agent
    sample_jd = """
    Senior Python Developer
    
    We are looking for a Senior Python Developer with 5+ years of experience
    to join our AI/ML team.
    
    Requirements:
    - Strong proficiency in Python 3.x
    - Experience with Django or FastAPI
    - Knowledge of PostgreSQL and Redis
    - Familiarity with AWS services (EC2, S3, Lambda)
    - Experience with Docker and Kubernetes
    - Understanding of machine learning concepts
    - Excellent communication skills
    
    Nice to have:
    - PyTorch or TensorFlow experience
    - AWS certification
    - Experience in fintech domain
    """
    
    print("Testing AnalystAgent...")
    print("-" * 50)
    
    agent = create_analyst_agent()
    analysis = agent.analyze(sample_jd)
    
    print(f"Title: {analysis.title}")
    print(f"Skills: {analysis.skills}")
    print(f"Experience: {analysis.experience_years} years ({analysis.experience_level})")
    print(f"Tech Stack: {analysis.tech_stack}")
    print(f"Summary: {analysis.summary}")

