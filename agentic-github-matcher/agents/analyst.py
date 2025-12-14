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
    analyze job descriptions. The agent decides how to process the input
    and format the structured output, demonstrating true agentic AI behavior.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        name: str = "AnalystAgent"
    ):
        """
        Initialize the AnalystAgent.
        
        Args:
            model: The LLM model to use (via LiteLLM)
            temperature: Temperature for generation (lower = more focused)
            name: Name identifier for the agent
        """
        self.model = model
        self.temperature = temperature
        self.name = name
        
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
                    "response_format": {"type": "json_object"},  # Force JSON response
                }
            ],
            "timeout": 120,
        }
        
        # Log API key status (without exposing the key)
        if api_key:
            if api_key.startswith("sk-"):
                print(f"  [AnalystAgent] ✓ API key format valid (starts with 'sk-')")
            else:
                print(f"  [AnalystAgent] ⚠ API key format unusual (doesn't start with 'sk-'): {api_key[:10]}...")
        else:
            print(f"  [AnalystAgent] ⚠ No API key found in environment")
        
        # Create the AutoGen agent for agentic behavior
        self.agent = ConversableAgent(
            name=name,
            system_message=ANALYST_SYSTEM_PROMPT,
            llm_config=self.llm_config,
            human_input_mode="NEVER",  # Fully automated
            max_consecutive_auto_reply=1,
        )
        
        print(f"  [AnalystAgent] ✓ Agentic behavior enabled - using AutoGen agent")
    
    def analyze(self, job_description: str) -> JobAnalysis:
        """
        Analyze a job description and extract structured information using agentic behavior.
        
        This method uses the AutoGen agent to autonomously analyze the job description
        and extract structured information. The agent decides how to process the input
        and format the output.
        
        Args:
            job_description: The raw job description text
            
        Returns:
            JobAnalysis: Structured analysis results
        """
        # Prepare the analysis prompt for the agent
        prompt = f"""Please analyze the following job description and extract all relevant information.

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

Be thorough but precise. Only extract information explicitly mentioned or strongly implied in the job description."""

        try:
            print(f"  [AnalystAgent] 🤖 Starting agentic analysis with model: {self.model}")
            
            # Clear agent history for fresh start
            self.agent.clear_history()
            
            # Create a user proxy agent to initiate conversation with the agent
            # This enables agentic behavior where the agent processes the request
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,  # Don't execute code, just use agent
            )
            
            # Initiate chat - the agent will analyze and respond
            print(f"  [AnalystAgent] 🤖 Agent is analyzing job description...")
            chat_result = user_proxy.initiate_chat(
                recipient=self.agent,
                message=prompt,
                max_turns=1,  # Single turn for analysis
                silent=False
            )
            
            # Try to get response directly from chat_result first
            agent_response = None
            if hasattr(chat_result, 'summary'):
                agent_response = chat_result.summary
                print(f"  [AnalystAgent] Debug: Got response from chat_result.summary")
            elif hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                # Get the last message which should be from the agent
                last_msg = chat_result.chat_history[-1]
                if isinstance(last_msg, dict):
                    if last_msg.get("role") == "assistant" or last_msg.get("name") == self.agent.name:
                        agent_response = last_msg.get("content", "")
                        print(f"  [AnalystAgent] Debug: Got response from last message in chat_history")
            
            # If we didn't get it from chat_result, try other methods
            if not agent_response:
                # Method 1: Check user_proxy's chat_messages (most reliable)
                if hasattr(user_proxy, 'chat_messages'):
                    if self.agent in user_proxy.chat_messages:
                        messages = user_proxy.chat_messages[self.agent]
                        print(f"  [AnalystAgent] Debug: Found {len(messages)} messages in user_proxy.chat_messages")
                        # Look for assistant messages (from the agent)
                        for idx, msg in enumerate(reversed(messages)):  # Get most recent first
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                name = msg.get("name", "")
                                print(f"  [AnalystAgent] Debug: Message {idx}: role={role}, name={name}, content_len={len(str(content))}")
                                # Skip user messages, get assistant messages
                                if role == "assistant" and content and content.strip():
                                    # Make sure it's not the prompt
                                    if content.strip() != prompt.strip():
                                        agent_response = content
                                        print(f"  [AnalystAgent] Debug: Found assistant response (length: {len(agent_response)})")
                                        break
                
                # Method 2: Check agent's chat_messages
                if not agent_response and hasattr(self.agent, 'chat_messages'):
                    if user_proxy in self.agent.chat_messages:
                        messages = self.agent.chat_messages[user_proxy]
                        print(f"  [AnalystAgent] Debug: Found {len(messages)} messages in agent.chat_messages")
                        for idx, msg in enumerate(reversed(messages)):
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                name = msg.get("name", "")
                                print(f"  [AnalystAgent] Debug: Message {idx}: role={role}, name={name}, content_len={len(str(content))}")
                                if role == "assistant" and content and content.strip():
                                    if content.strip() != prompt.strip():
                                        agent_response = content
                                        print(f"  [AnalystAgent] Debug: Found assistant response from agent.chat_messages")
                                        break
                
                # Method 3: Check _oai_messages (OpenAI format)
                if not agent_response and hasattr(self.agent, '_oai_messages'):
                    if user_proxy in self.agent._oai_messages:
                        messages = self.agent._oai_messages[user_proxy]
                        print(f"  [AnalystAgent] Debug: Found {len(messages)} messages in agent._oai_messages")
                        for idx, msg in enumerate(reversed(messages)):
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                if role == "assistant" and content and content.strip():
                                    if content.strip() != prompt.strip():
                                        agent_response = content
                                        print(f"  [AnalystAgent] Debug: Found assistant response from _oai_messages")
                                        break
                
                # Method 4: Check chat_result.chat_history
                if not agent_response and hasattr(chat_result, 'chat_history'):
                    print(f"  [AnalystAgent] Debug: Checking chat_result.chat_history ({len(chat_result.chat_history)} messages)")
                    for idx, msg in enumerate(reversed(chat_result.chat_history)):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            print(f"  [AnalystAgent] Debug: chat_history[{idx}]: role={role}, content_len={len(str(content))}")
                            if role == "assistant" and content and content.strip() and content.strip() != prompt.strip():
                                agent_response = content
                                print(f"  [AnalystAgent] Debug: Found assistant response from chat_history")
                                break
                
                # Method 5: Check user_proxy's _oai_messages
                if not agent_response and hasattr(user_proxy, '_oai_messages'):
                    if self.agent in user_proxy._oai_messages:
                        messages = user_proxy._oai_messages[self.agent]
                        print(f"  [AnalystAgent] Debug: Found {len(messages)} messages in user_proxy._oai_messages")
                        for idx, msg in enumerate(reversed(messages)):
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                if role == "assistant" and content and content.strip() and content.strip() != prompt.strip():
                                    agent_response = content
                                    print(f"  [AnalystAgent] Debug: Found assistant response from user_proxy._oai_messages")
                                    break
            
            # Validate we got a real response (not the prompt)
            if not agent_response or agent_response.strip() == prompt.strip():
                print(f"  [AnalystAgent] ⚠ Warning: Could not extract agent response or got prompt back")
                print(f"  [AnalystAgent] Debug: agent_response is None: {agent_response is None}")
                if agent_response:
                    print(f"  [AnalystAgent] Debug: agent_response matches prompt: {agent_response.strip() == prompt.strip()}")
                    print(f"  [AnalystAgent] Debug: agent_response preview: {agent_response[:100] if len(agent_response) > 100 else agent_response}")
                # Debug: print what we found
                if hasattr(user_proxy, 'chat_messages'):
                    print(f"  [AnalystAgent] Debug: user_proxy.chat_messages keys: {list(user_proxy.chat_messages.keys())}")
                    if self.agent in user_proxy.chat_messages:
                        msgs = user_proxy.chat_messages[self.agent]
                        print(f"  [AnalystAgent] Debug: Messages from agent: {len(msgs)}")
                        for i, m in enumerate(msgs):
                            if isinstance(m, dict):
                                print(f"    [{i}] role={m.get('role')}, name={m.get('name')}, content_preview={str(m.get('content', ''))[:50]}")
                if hasattr(self.agent, 'chat_messages'):
                    print(f"  [AnalystAgent] Debug: agent.chat_messages keys: {list(self.agent.chat_messages.keys())}")
                    if user_proxy in self.agent.chat_messages:
                        msgs = self.agent.chat_messages[user_proxy]
                        print(f"  [AnalystAgent] Debug: Messages from user_proxy: {len(msgs)}")
                        for i, m in enumerate(msgs):
                            if isinstance(m, dict):
                                print(f"    [{i}] role={m.get('role')}, name={m.get('name')}, content_preview={str(m.get('content', ''))[:50]}")
                raise ValueError("No valid response received from agent (got prompt or empty response)")
            
            print(f"  [AnalystAgent] ✓ Agent response received")
            
            # Extract JSON from response (may contain markdown code blocks or plain JSON)
            content = agent_response.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            elif content.startswith("```"):
                content = content[3:]  # Remove ```
            
            if content.endswith("```"):
                content = content[:-3]  # Remove closing ```
            
            content = content.strip()
            
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
            print(f"  [AnalystAgent] Response content: {agent_response[:200] if 'agent_response' in locals() else 'N/A'}")
            # Fallback to direct LLM call if agentic approach fails
            print(f"  [AnalystAgent] ⚠ Falling back to direct LLM call...")
            return self._fallback_analyze(job_description)
        except Exception as e:
            import traceback
            print(f"  [AnalystAgent] ❌ Error during agentic analysis: {e}")
            print(f"  [AnalystAgent] Traceback: {traceback.format_exc()}")
            # Fallback to direct LLM call
            print(f"  [AnalystAgent] ⚠ Falling back to direct LLM call...")
            return self._fallback_analyze(job_description)
    
    def _fallback_analyze(self, job_description: str) -> JobAnalysis:
        """
        Fallback method using direct LLM call if agentic approach fails.
        
        Args:
            job_description: The raw job description text
            
        Returns:
            JobAnalysis: Structured analysis results
        """
        prompt = f"""Please analyze the following job description and extract all relevant information.

JOB DESCRIPTION:
---
{job_description}
---

Return your analysis as a JSON object."""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
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
        except Exception as e:
            print(f"  [AnalystAgent] ❌ Fallback analysis also failed: {e}")
            return JobAnalysis(summary=f"Error analyzing job description: {e}")
    
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
    temperature: float = 0.3
) -> AnalystAgent:
    """
    Factory function to create an AnalystAgent.
    
    Args:
        model: The LLM model to use
        temperature: Generation temperature
        
    Returns:
        AnalystAgent: Configured analyst agent
    """
    return AnalystAgent(model=model, temperature=temperature)


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

