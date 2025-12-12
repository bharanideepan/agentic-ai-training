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
from autogen import ConversableAgent, AssistantAgent

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

IMPORTANT: Return your analysis as a valid JSON object with these exact keys:
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
    structured skill and requirement information.
    
    This agent uses AutoGen's ConversableAgent with LiteLLM
    as the model gateway to process job descriptions.
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
        
        # Create the AutoGen agent
        self.agent = ConversableAgent(
            name=name,
            system_message=ANALYST_SYSTEM_PROMPT,
            llm_config=self.llm_config,
            human_input_mode="NEVER",  # Fully automated
            max_consecutive_auto_reply=1,
        )
    
    def analyze(self, job_description: str) -> JobAnalysis:
        """
        Analyze a job description and extract structured information.
        
        Args:
            job_description: The raw job description text
            
        Returns:
            JobAnalysis: Structured analysis results
        """
        # Prepare the analysis prompt
        prompt = f"""Please analyze the following job description and extract all relevant information.

JOB DESCRIPTION:
---
{job_description}
---

Return your analysis as a JSON object."""

        # Call LiteLLM directly for more control
        try:
            print(f"  [AnalystAgent] Calling LLM with model: {self.model}")
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            print(f"  [AnalystAgent] LLM response received")
            # Parse the response
            content = response.choices[0].message.content
            print(f"  [AnalystAgent] Parsing JSON response...")
            data = json.loads(content)
            print(f"  [AnalystAgent] Successfully extracted: {len(data.get('skills', []))} skills")
            
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
            print(f"  [AnalystAgent] Response content: {content[:200] if 'content' in locals() else 'N/A'}")
            # Return empty analysis on error
            return JobAnalysis(summary=f"Error analyzing job description: {e}")
        except Exception as e:
            import traceback
            print(f"  [AnalystAgent] ❌ Error during analysis: {e}")
            print(f"  [AnalystAgent] Traceback: {traceback.format_exc()}")
            return JobAnalysis(summary=f"Error analyzing job description: {e}")
    
    async def analyze_async(self, job_description: str) -> JobAnalysis:
        """
        Async version of analyze method.
        
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
            response = await litellm.acompletion(
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
            print(f"Error during async analysis: {e}")
            return JobAnalysis(summary=f"Error analyzing job description: {e}")


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

