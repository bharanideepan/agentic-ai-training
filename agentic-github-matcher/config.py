"""
Configuration Module
====================

Centralized configuration for the Agentic GitHub Matcher.
Loads settings from environment variables and configuration files.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==============================================
# PATH CONFIGURATION
# ==============================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Configuration file paths
LITELLM_CONFIG_PATH = PROJECT_ROOT / "litellm.config.yaml"
GUARDRAILS_CONFIG_PATH = PROJECT_ROOT / "guardrails"


# ==============================================
# MODEL CONFIGURATION
# ==============================================

@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    
    # Primary model
    model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o"))
    
    # Temperature settings per agent type
    analyst_temperature: float = 0.3  # Lower for more consistent analysis
    github_temperature: float = 0.5   # Moderate for tool calling
    formatter_temperature: float = 0.3  # Lower for consistent formatting
    
    # Token limits
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))
    
    # Timeout settings
    timeout: int = 120
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 5


# ==============================================
# API CONFIGURATION
# ==============================================

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    
    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # GitHub
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_api_base: str = "https://api.github.com"
    
    # Rate limiting
    github_rate_limit_buffer: int = 10  # Keep buffer from rate limit
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that required API keys are set.
        
        Returns:
            tuple: (is_valid, list of missing keys)
        """
        missing = []
        
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        
        if not self.github_token:
            missing.append("GITHUB_TOKEN")
        
        return len(missing) == 0, missing


# ==============================================
# WORKFLOW CONFIGURATION
# ==============================================

@dataclass
class WorkflowConfig:
    """Configuration for the workflow execution."""
    
    # Search settings
    max_candidates: int = 10
    max_repos_per_search: int = 15
    max_repos_per_user: int = 5
    
    # Skill extraction
    max_skills_for_search: int = 5
    
    # Output settings
    default_output_format: str = "rich"
    save_reports: bool = False
    report_directory: str = "reports"


# ==============================================
# GUARDRAILS CONFIGURATION
# ==============================================

@dataclass
class GuardrailsConfig:
    """Configuration for Nemo Guardrails."""
    
    enabled: bool = True
    config_path: Path = field(default_factory=lambda: GUARDRAILS_CONFIG_PATH)
    
    # PII detection and masking
    pii_detection_enabled: bool = True
    pii_mask_mode: str = "mask"  # "mask" (replace with [TYPE_REDACTED]) or "block" (reject input)
    
    # Blocked patterns for input validation
    blocked_patterns: list[str] = field(default_factory=lambda: [
        "ignore previous",
        "disregard instructions", 
        "pretend you are",
        "act as if",
        "hack",
        "exploit",
        "bypass",
        "inject"
    ])


# ==============================================
# MAIN CONFIGURATION CLASS
# ==============================================

@dataclass
class Config:
    """Main configuration container."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate all configuration."""
        return self.api.validate()


# ==============================================
# SINGLETON CONFIG INSTANCE
# ==============================================

# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: The configuration instance
    """
    global _config
    
    if _config is None:
        _config = Config()
    
    return _config


def reload_config() -> Config:
    """
    Reload configuration from environment.
    
    Returns:
        Config: Fresh configuration instance
    """
    global _config
    
    # Reload .env file
    load_dotenv(override=True)
    
    # Create new config
    _config = Config()
    
    return _config


# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def get_litellm_config() -> dict:
    """
    Get LiteLLM configuration as a dictionary.
    
    Returns:
        dict: LiteLLM compatible configuration
    """
    config = get_config()
    
    return {
        "config_list": [
            {
                "model": config.model.model,
                "api_type": "openai",
                "api_key": config.api.openai_api_key,
                "temperature": config.model.analyst_temperature,
            }
        ],
        "timeout": config.model.timeout,
    }


def print_config_status():
    """Print configuration status for debugging."""
    config = get_config()
    is_valid, missing = config.validate()
    
    print("=" * 50)
    print("Configuration Status")
    print("=" * 50)
    print(f"Model: {config.model.model}")
    print(f"Max Tokens: {config.model.max_tokens}")
    print(f"OpenAI Key: {'✓ Set' if config.api.openai_api_key else '✗ Missing'}")
    print(f"GitHub Token: {'✓ Set' if config.api.github_token else '✗ Missing'}")
    print(f"Guardrails: {'Enabled' if config.guardrails.enabled else 'Disabled'}")
    print(f"Valid: {'Yes' if is_valid else 'No - Missing: ' + ', '.join(missing)}")
    print("=" * 50)


# ==============================================
# MODULE TEST
# ==============================================

if __name__ == "__main__":
    print_config_status()

