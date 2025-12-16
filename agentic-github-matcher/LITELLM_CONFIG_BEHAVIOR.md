# LiteLLM Configuration Behavior in Analyst Agent

## Overview

When using LiteLLM proxy, the agent can either:
1. **Use LiteLLM config values** (recommended) - Let `litellm.config.yaml` handle all configuration
2. **Override config values** - Pass explicit parameters that override the config

## Configuration Options

### Option 1: Use LiteLLM Config (Default, Recommended)

```python
from agents.analyst import create_analyst_agent

# This will use values from litellm.config.yaml
agent = create_analyst_agent(
    model="gpt-4o",
    use_litellm_config=True  # Default
)
```

**Behavior:**
- ✅ Uses `temperature: 0.7` from `litellm.config.yaml`
- ✅ Uses `api_key` from config/environment
- ✅ Uses model routing from config
- ✅ Uses fallbacks from config
- ✅ All other config values are respected

**What gets passed to AutoGen:**
```python
{
    "model": "gpt-4o",
    "api_type": "openai",
    "base_url": "http://localhost:4000",
    "api_key": "...",  # Still needed for proxy forwarding
    # temperature is NOT passed - uses config value
}
```

### Option 2: Override Config Values

```python
from agents.analyst import create_analyst_agent

# This will override temperature from config
agent = create_analyst_agent(
    model="gpt-4o",
    temperature=0.3,  # Overrides config value of 0.7
    use_litellm_config=False
)
```

**Behavior:**
- ✅ Uses `temperature: 0.3` (overrides config)
- ✅ Still uses LiteLLM routing and fallbacks
- ✅ Still uses other config values

**What gets passed to AutoGen:**
```python
{
    "model": "gpt-4o",
    "api_type": "openai",
    "base_url": "http://localhost:4000",
    "api_key": "...",
    "temperature": 0.3  # Overrides config
}
```

## Parameters in llm_config

### When Using LiteLLM Proxy

| Parameter | Required? | Source | Behavior |
|-----------|-----------|--------|----------|
| `model` | ✅ Yes | Passed explicitly | Must match `model_name` in `litellm.config.yaml` |
| `api_type` | ✅ Yes | Always `"openai"` | Tells AutoGen to use OpenAI-compatible API |
| `base_url` | ✅ Yes | From `LITELLM_PROXY_URL` | Routes requests through LiteLLM proxy |
| `api_key` | ✅ Yes | From environment | Needed for proxy to forward to OpenAI |
| `temperature` | ⚠️ Optional | Config or explicit | If `use_litellm_config=True`: Uses config value<br>If `use_litellm_config=False`: Overrides config |

### When NOT Using LiteLLM Proxy

| Parameter | Required? | Source | Behavior |
|-----------|-----------|--------|----------|
| `model` | ✅ Yes | Passed explicitly | Direct OpenAI model name |
| `api_type` | ✅ Yes | Always `"openai"` | Direct OpenAI API |
| `base_url` | ❌ No | Not set | Goes directly to `api.openai.com` |
| `api_key` | ✅ Yes | From environment | Direct OpenAI API key |
| `temperature` | ✅ Yes | Passed explicitly | Used directly |

## Why This Design?

### Benefits of Using LiteLLM Config (Default)

1. **Centralized Configuration**: All model settings in one place (`litellm.config.yaml`)
2. **Easy Model Switching**: Change model in config file, no code changes needed
3. **Consistent Behavior**: All agents use same config values
4. **Fallback Support**: Automatic fallbacks configured in YAML
5. **Cost Tracking**: LiteLLM can track costs across all requests

### When to Override Config

Override config when you need:
- **Per-agent temperature**: Different agents need different creativity levels
- **Per-request control**: Dynamic temperature based on context
- **Testing**: Test with different parameters without changing config

## Example: litellm.config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      temperature: 0.7  # ← This value is used when use_litellm_config=True
      max_tokens: 4096
```

## Verification

To verify which configuration is being used:

```python
from agents.analyst import create_analyst_agent

agent = create_analyst_agent()
status = agent.get_litellm_status()

print(f"Using LiteLLM: {status['using_litellm']}")
print(f"Proxy URL: {status['proxy_url']}")

# Check what's in llm_config
if hasattr(agent, 'llm_config'):
    config = agent.llm_config['config_list'][0]
    print(f"base_url: {config.get('base_url', 'Not set')}")
    print(f"temperature: {config.get('temperature', 'Using config value')}")
```

## Summary

✅ **Recommended**: Use `use_litellm_config=True` (default)
- Lets LiteLLM handle configuration from YAML
- Centralized model management
- Consistent behavior across agents

⚠️ **Override when needed**: Use `use_litellm_config=False`
- When you need per-agent or per-request control
- When testing different parameters
- Still uses LiteLLM routing and fallbacks

The key point: **Even when overriding parameters, you still get LiteLLM benefits** (routing, fallbacks, cost tracking) - you're just customizing specific values.

