"""
NeMo Guardrails Runner
======================

This module provides input and output validation using NeMo Guardrails.
It initializes guardrails once and exposes two validation functions.

DO NOT use for:
- Tool calls
- GitHub MCP
- Agent reasoning
- Network or security layers
"""

import os
from pathlib import Path
from typing import Optional

# NeMo Guardrails imports
try:
    from nemoguardrails import LLMRails, RailsConfig
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    LLMRails = None
    RailsConfig = None


# ==============================================
# CUSTOM EXCEPTIONS
# ==============================================

class GuardrailsValidationError(Exception):
    """Raised when guardrails validation fails."""
    pass


# ==============================================
# GUARDRAILS INITIALIZATION
# ==============================================

_rails_instance: Optional[LLMRails] = None


def _initialize_guardrails() -> Optional[LLMRails]:
    """
    Initialize NeMo Guardrails once.
    
    Returns:
        LLMRails instance or None if initialization fails
    """
    global _rails_instance
    
    if _rails_instance is not None:
        return _rails_instance
    
    if not GUARDRAILS_AVAILABLE:
        return None
    
    try:
        # Get the guardrails directory path
        guardrails_dir = Path(__file__).parent
        
        # Load configuration from the guardrails directory
        config = RailsConfig.from_path(str(guardrails_dir))
        
        # Initialize LLMRails
        rails = LLMRails(config)
        
        _rails_instance = rails
        return rails
        
    except Exception as e:
        # Log error but don't raise - allow system to continue without guardrails
        print(f"[Guardrails] Warning: Failed to initialize guardrails: {e}")
        return None


# ==============================================
# VALIDATION FUNCTIONS
# ==============================================

async def validate_input(text: str) -> str:
    """
    Validate user input (job description) using guardrails.
    
    This function:
    - Passes text through input guardrails
    - Raises GuardrailsValidationError if validation fails
    - Returns the original text if validation passes
    
    Args:
        text: The job description text to validate
        
    Returns:
        str: The original text if validation passes
        
    Raises:
        GuardrailsValidationError: If validation fails
    """
    print(f"[Guardrails Runner] [INPUT] ===== Starting input validation =====")
    print(f"[Guardrails Runner] [INPUT] Input length: {len(text) if text else 0} characters")
    print(f"[Guardrails Runner] [INPUT] Input preview: {text[:100] if text else 'None'}...")
    
    if not text or not isinstance(text, str):
        print("[Guardrails Runner] [INPUT] Validation FAILED: Input must be a non-empty string")
        raise GuardrailsValidationError("Input must be a non-empty string")
    
    # Initialize guardrails if not already done
    print("[Guardrails Runner] [INPUT] Initializing guardrails...")
    rails = _initialize_guardrails()
    
    if rails is None:
        print("[Guardrails Runner] [INPUT] Guardrails not available, using basic validation")
        # If guardrails not available, perform basic validation
        if not text.strip():
            print("[Guardrails Runner] [INPUT] Validation FAILED: Empty input")
            raise GuardrailsValidationError("Job description cannot be empty")
        if len(text.strip()) < 10:
            print("[Guardrails Runner] [INPUT] Validation FAILED: Too short")
            raise GuardrailsValidationError("Job description is too short. Please provide a meaningful job description.")
        print("[Guardrails Runner] [INPUT] Basic validation PASSED")
        return text
    
    print("[Guardrails Runner] [INPUT] Guardrails initialized, running input rails...")
    
    try:
        # Use guardrails to validate input
        # The input rail flow will be triggered automatically
        print("[Guardrails Runner] [INPUT] Calling rails.generate_async() with input...")
        result = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        
        print(f"[Guardrails Runner] [INPUT] Rails.generate_async() returned: {type(result).__name__}")
        print(f"[Guardrails Runner] [INPUT] Result attributes: {dir(result)}")
        
        # Check if guardrails blocked the input
        if result is None:
            print("[Guardrails Runner] [INPUT] Validation FAILED: Result is None (blocked)")
            raise GuardrailsValidationError(
                "Input validation failed: Job description was rejected by guardrails"
            )
        
        # NeMo returns a GenerationResponse object, not a string
        # Extract the actual response text - check multiple possible locations
        response_text = None
        
        # Try different ways to extract the response
        if hasattr(result, 'content'):
            response_text = result.content
        elif hasattr(result, 'response'):
            # GenerationResponse.response might be a list or string
            resp = result.response
            if isinstance(resp, list) and len(resp) > 0:
                last_item = resp[-1]
                if isinstance(last_item, dict) and 'content' in last_item:
                    response_text = last_item['content']
                elif hasattr(last_item, 'content'):
                    response_text = last_item.content
                else:
                    response_text = str(last_item)
            elif isinstance(resp, str):
                response_text = resp
        elif hasattr(result, 'messages') and len(result.messages) > 0:
            # Get the last message content
            last_msg = result.messages[-1]
            if isinstance(last_msg, dict):
                response_text = last_msg.get('content', str(last_msg))
            elif hasattr(last_msg, 'content'):
                response_text = last_msg.content
        elif isinstance(result, str):
            response_text = result
        else:
            # Fallback: convert to string
            response_text = str(result)
        
        print(f"[Guardrails Runner] [INPUT] Response text: {response_text[:200] if response_text else 'None'}...")
        print(f"[Guardrails Runner] [INPUT] Response text type: {type(response_text).__name__}")
        
        # Check for exception events first (most reliable)
        # NeMo stores events in different places depending on the response type
        events_to_check = []
        if hasattr(result, 'events') and result.events:
            events_to_check.extend(result.events)
        if hasattr(result, 'log') and hasattr(result.log, 'events'):
            events_to_check.extend(result.log.events)
        if hasattr(result, 'messages'):
            # Check messages for exception indicators
            for msg in result.messages:
                if isinstance(msg, dict) and msg.get('type') == 'exception':
                    events_to_check.append(msg)
        
        print(f"[Guardrails Runner] [INPUT] Found {len(events_to_check)} events to check")
        for event in events_to_check:
            event_type = str(getattr(event, 'type', '') or event.get('type', '')).lower()
            print(f"[Guardrails Runner] [INPUT] Checking event type: {event_type}")
            if 'exception' in event_type or 'inputrail' in event_type or 'rail' in event_type:
                event_msg = getattr(event, 'message', None) or event.get('message', None) or str(event)
                print(f"[Guardrails Runner] [INPUT] Validation FAILED: Exception event detected: {event_msg}")
                raise GuardrailsValidationError(
                    f"Input validation failed: {event_msg}"
                )
        
        # Check if response text indicates blocking (refusal messages)
        if response_text:
            response_lower = str(response_text).lower()
            blocking_indicators = [
                "i'm sorry", "i cannot", "i can't", "i'm unable",
                "i cannot help", "i can't help", "i'm unable to help",
                "validation failed", "invalid:", "refuse", "reject", 
                "blocked", "cannot help", "unable to", "not allowed",
                "sorry, i can't", "sorry, i cannot"
            ]
            
            for indicator in blocking_indicators:
                if indicator in response_lower:
                    print(f"[Guardrails Runner] [INPUT] Validation FAILED: Blocking indicator '{indicator}' found in response")
                    # Extract reason if available
                    if "invalid:" in response_lower or "rejected:" in response_lower:
                        parts = str(response_text).split(":", 1)
                        if len(parts) > 1:
                            reason = parts[1].strip()
                            raise GuardrailsValidationError(f"Input validation failed: {reason}")
                    
                    # If it's a refusal message, input was blocked
                    raise GuardrailsValidationError(
                        "Input validation failed: Job description was rejected by guardrails"
                    )
        
        # Input passed validation - return original text
        print("[Guardrails Runner] [INPUT] ===== Validation PASSED =====")
        return text
        
    except GuardrailsValidationError as e:
        # Re-raise our custom exception
        print(f"[Guardrails Runner] [INPUT] ===== Validation FAILED: {str(e)} =====")
        raise
    except Exception as e:
        # Check if it's a blocking exception
        error_msg = str(e).lower()
        blocking_keywords = ["blocked", "refuse", "reject", "invalid", "exception"]
        
        for keyword in blocking_keywords:
            if keyword in error_msg:
                print(f"[Guardrails Runner] [INPUT] Validation FAILED: Blocking exception detected: {str(e)}")
                raise GuardrailsValidationError(
                    f"Input validation failed: {str(e)}"
                )
        
        # For other errors, log and allow (fail open for safety)
        print(f"[Guardrails Runner] [INPUT] Warning: Input validation error (non-blocking): {e}")
        print(f"[Guardrails Runner] [INPUT] Allowing input due to non-blocking error")
        return text


async def validate_output(text: str) -> str:
    """
    Validate final output (formatted result) using guardrails (async).
    
    This function:
    - Passes text through output guardrails
    - Raises GuardrailsValidationError if validation fails
    - Returns the original text if validation passes
    
    Args:
        text: The formatted output text to validate
        
    Returns:
        str: The original text if validation passes
        
    Raises:
        GuardrailsValidationError: If validation fails
    """
    print(f"[Guardrails Runner] [OUTPUT] ===== Starting output validation =====")
    print(f"[Guardrails Runner] [OUTPUT] Output length: {len(text) if text else 0} characters")
    print(f"[Guardrails Runner] [OUTPUT] Output preview: {text[:200] if text else 'None'}...")
    
    if not text or not isinstance(text, str):
        print("[Guardrails Runner] [OUTPUT] Validation FAILED: Output must be a non-empty string")
        raise GuardrailsValidationError("Output must be a non-empty string")
    
    # Initialize guardrails if not already done
    print("[Guardrails Runner] [OUTPUT] Initializing guardrails...")
    rails = _initialize_guardrails()
    
    if rails is None:
        print("[Guardrails Runner] [OUTPUT] Guardrails not available, using basic validation")
        # If guardrails not available, perform basic validation
        if not text.strip():
            print("[Guardrails Runner] [OUTPUT] Validation FAILED: Empty output")
            raise GuardrailsValidationError("Output cannot be empty")
        print("[Guardrails Runner] [OUTPUT] Basic validation PASSED")
        return text
    
    print("[Guardrails Runner] [OUTPUT] Guardrails initialized, running output rails...")
    
    try:
        # Use guardrails to validate output
        # The output rail flow will be triggered automatically
        # We simulate a conversation to trigger output rails
        print("[Guardrails Runner] [OUTPUT] Calling rails.generate_async() with output...")
        result = await rails.generate_async(
            messages=[
                {"role": "user", "content": "Validate this formatted report output"},
                {"role": "assistant", "content": text}
            ]
        )
        
        print(f"[Guardrails Runner] [OUTPUT] Rails.generate_async() returned: {type(result).__name__}")
        if isinstance(result, str):
            print(f"[Guardrails Runner] [OUTPUT] Result preview: {result[:200]}...")
        
        # Check if guardrails blocked the output
        if result is None:
            print("[Guardrails Runner] [OUTPUT] Validation FAILED: Result is None (blocked)")
            raise GuardrailsValidationError(
                "Output validation failed: Formatted output was rejected by guardrails"
            )
        
        # Check if result indicates blocking
        if isinstance(result, str):
            result_lower = result.lower()
            blocking_indicators = [
                "validation failed", "rejected", "refuse", "blocked",
                "cannot", "unable", "invalid", "not allowed", "unsafe"
            ]
            
            for indicator in blocking_indicators:
                if indicator in result_lower:
                    print(f"[Guardrails Runner] [OUTPUT] Validation FAILED: Blocking indicator '{indicator}' found in result")
                    # Extract reason if available
                    if ":" in result:
                        reason = result.split(":", 1)[1].strip()
                        raise GuardrailsValidationError(f"Output validation failed: {reason}")
                    else:
                        raise GuardrailsValidationError(
                            "Output validation failed: Formatted output was rejected by guardrails"
                        )
        
        # Check for exception events in the result
        if hasattr(result, 'events'):
            for event in result.events:
                if hasattr(event, 'type') and 'exception' in event.type.lower():
                    print(f"[Guardrails Runner] [OUTPUT] Validation FAILED: Exception event detected: {event}")
                    raise GuardrailsValidationError(
                        f"Output validation failed: {getattr(event, 'message', 'Output was rejected')}"
                    )
        
        # Output passed validation - return original text
        print("[Guardrails Runner] [OUTPUT] ===== Validation PASSED =====")
        return text
        
    except GuardrailsValidationError as e:
        # Re-raise our custom exception
        print(f"[Guardrails Runner] [OUTPUT] ===== Validation FAILED: {str(e)} =====")
        raise
    except Exception as e:
        # Check if it's a blocking exception
        error_msg = str(e).lower()
        blocking_keywords = ["blocked", "refuse", "reject", "invalid", "exception"]
        
        for keyword in blocking_keywords:
            if keyword in error_msg:
                print(f"[Guardrails Runner] [OUTPUT] Validation FAILED: Blocking exception detected: {str(e)}")
                raise GuardrailsValidationError(
                    f"Output validation failed: {str(e)}"
                )
        
        # For other errors, log and allow (fail open for safety)
        print(f"[Guardrails Runner] [OUTPUT] Warning: Output validation error (non-blocking): {e}")
        print(f"[Guardrails Runner] [OUTPUT] Allowing output due to non-blocking error")
        return text

