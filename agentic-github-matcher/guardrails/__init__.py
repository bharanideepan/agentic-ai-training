# Guardrails Package Initialization
# ==================================
# This package contains Nemo Guardrails configuration.

from pathlib import Path

# Path to the rails configuration file
RAILS_CONFIG_PATH = Path(__file__).parent / "rails.yaml"

def get_rails_config_path() -> Path:
    """Get the path to the rails configuration file."""
    return RAILS_CONFIG_PATH

