"""
NeMo Guardrails Module
======================

This module provides input and output validation using NeMo Guardrails.

DO NOT use for:
- Tool calls
- GitHub MCP
- Agent reasoning
- Network or security layers

Only for:
- Validating user input (job description)
- Validating final output (formatted result)
"""

from guardrails.runner import validate_input, validate_output, GuardrailsValidationError

__all__ = ['validate_input', 'validate_output', 'GuardrailsValidationError']
