# How to Verify LiteLLM Usage in Analyst Agent

This guide explains how to verify that the AnalystAgent is properly using LiteLLM for all LLM communications.

## Quick Verification Steps

### 1. Start LiteLLM Proxy Server

In a terminal, start the LiteLLM proxy:

```bash
cd agentic-github-matcher
litellm --config litellm.config.yaml --port 4000
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4000
```

### 2. Set Environment Variable

In another terminal (or the same one, in a new tab):

**Windows (PowerShell):**
```powershell
$env:LITELLM_PROXY_URL="http://localhost:4000"
```

**Windows (CMD):**
```cmd
set LITELLM_PROXY_URL=http://localhost:4000
```

**Linux/Mac:**
```bash
export LITELLM_PROXY_URL=http://localhost:4000
```

### 3. Run Verification Script

```bash
cd agentic-github-matcher
python verify_litellm_usage.py
```

The script will:
- ✅ Check if LiteLLM proxy is running
- ✅ Verify agent configuration
- ✅ Test the agent
- ✅ Provide verification instructions

## Manual Verification Methods

### Method 1: Check LiteLLM Proxy Logs

When you run the analyst agent, check the terminal where LiteLLM proxy is running. You should see log entries like:

```
INFO:     127.0.0.1:xxxxx - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     Model: gpt-4o
INFO:     Status: 200
```

**If you see these logs:** ✅ LiteLLM is being used

**If you don't see any logs:** ❌ Agent is bypassing LiteLLM

### Method 2: Check Agent Initialization Output

When you create an AnalystAgent, look for these messages:

**Using LiteLLM (Correct):**
```
[AnalystAgent] ✓ Using LiteLLM proxy: http://localhost:4000
[AnalystAgent] ✓ All LLM calls will route through LiteLLM gateway
[AnalystAgent] ✓ Model: gpt-4o (via LiteLLM gateway)
```

**Bypassing LiteLLM (Incorrect):**
```
[AnalystAgent] ⚠ LiteLLM proxy not configured (LITELLM_PROXY_URL not set)
[AnalystAgent] ⚠ Will use OpenAI directly (bypasses LiteLLM)
[AnalystAgent] ✓ Model: gpt-4o (direct OpenAI - LiteLLM not configured)
```

### Method 3: Programmatic Check

You can check the agent's LiteLLM status programmatically:

```python
from agents.analyst import create_analyst_agent

agent = create_analyst_agent()
status = agent.get_litellm_status()

print(f"Using LiteLLM: {status['using_litellm']}")
print(f"Proxy URL: {status['proxy_url']}")
print(f"base_url: {status['base_url']}")
```

### Method 4: Network Traffic Monitoring

Use a network monitoring tool to check where requests are going:

**Using LiteLLM:**
- Requests go to: `http://localhost:4000/v1/chat/completions`
- LiteLLM proxy then forwards to OpenAI

**Bypassing LiteLLM:**
- Requests go directly to: `https://api.openai.com/v1/chat/completions`

## Expected Behavior

### ✅ Correct (Using LiteLLM)

1. **Environment Variable Set:**
   ```bash
   echo $LITELLM_PROXY_URL  # Should show: http://localhost:4000
   ```

2. **Agent Initialization:**
   - Shows "Using LiteLLM proxy" message
   - Shows "via LiteLLM gateway" in model message

3. **Proxy Logs:**
   - Shows incoming requests
   - Shows model routing
   - Shows response status

4. **Network Traffic:**
   - Requests to `localhost:4000`
   - Not directly to `api.openai.com`

### ❌ Incorrect (Bypassing LiteLLM)

1. **Environment Variable Not Set:**
   ```bash
   echo $LITELLM_PROXY_URL  # Empty or not set
   ```

2. **Agent Initialization:**
   - Shows warning about LiteLLM not configured
   - Shows "direct OpenAI" in model message

3. **Proxy Logs:**
   - No requests appear in proxy logs

4. **Network Traffic:**
   - Requests go directly to `api.openai.com`
   - No requests to `localhost:4000`

## Troubleshooting

### Problem: Proxy not accessible

**Solution:**
1. Make sure LiteLLM proxy is running
2. Check the port (default: 4000)
3. Verify the URL matches: `http://localhost:4000`

### Problem: Agent still bypassing LiteLLM

**Check:**
1. Environment variable is set in the same terminal/session
2. Restart Python after setting environment variable
3. Verify `base_url` is in agent's `llm_config`

**Debug:**
```python
import os
print(f"LITELLM_PROXY_URL: {os.getenv('LITELLM_PROXY_URL')}")

from agents.analyst import create_analyst_agent
agent = create_analyst_agent()
print(agent.get_litellm_status())
```

### Problem: Proxy logs not showing requests

**Possible causes:**
1. Agent is using a different session/terminal
2. Environment variable not set in the right session
3. Agent was created before environment variable was set

**Solution:**
- Restart Python/application after setting environment variable
- Create agent in the same session where environment variable is set

## Summary

✅ **LiteLLM is working if:**
- `LITELLM_PROXY_URL` environment variable is set
- Agent initialization shows "Using LiteLLM proxy"
- Proxy logs show incoming requests
- Network traffic goes to `localhost:4000`

❌ **LiteLLM is NOT working if:**
- `LITELLM_PROXY_URL` is not set
- Agent shows "bypasses LiteLLM" warning
- No requests in proxy logs
- Network traffic goes directly to `api.openai.com`

