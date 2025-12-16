"""
Verification Script for LiteLLM Usage in Analyst Agent
======================================================

This script verifies that the AnalystAgent is properly using LiteLLM
for all LLM communications instead of bypassing it.

Usage:
    1. Start LiteLLM proxy: litellm --config litellm.config.yaml --port 4000
    2. Set environment: export LITELLM_PROXY_URL=http://localhost:4000
    3. Run this script: python verify_litellm_usage.py
"""

import os
import sys
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.analyst import create_analyst_agent


def check_litellm_proxy():
    """Check if LiteLLM proxy is running and accessible."""
    proxy_url = os.getenv("LITELLM_PROXY_URL", "")
    
    if not proxy_url:
        print("❌ LITELLM_PROXY_URL environment variable is not set")
        print("   Set it with: export LITELLM_PROXY_URL=http://localhost:4000")
        return False
    
    print(f"✓ LITELLM_PROXY_URL is set: {proxy_url}")
    
    # Try to connect to the proxy
    try:
        # LiteLLM proxy health endpoint
        health_url = f"{proxy_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✓ LiteLLM proxy is running and accessible at {proxy_url}")
            return True
        else:
            print(f"⚠ LiteLLM proxy responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to LiteLLM proxy at {proxy_url}")
        print("   Make sure the proxy is running:")
        print("   litellm --config litellm.config.yaml --port 4000")
        return False
    except Exception as e:
        print(f"⚠ Error checking LiteLLM proxy: {e}")
        return False


def check_agent_configuration():
    """Check if agent is configured to use LiteLLM."""
    print("\n" + "="*60)
    print("Checking Agent Configuration")
    print("="*60)
    
    proxy_url = os.getenv("LITELLM_PROXY_URL", "")
    
    if proxy_url:
        print(f"✓ Environment variable LITELLM_PROXY_URL is set")
        print(f"  Value: {proxy_url}")
    else:
        print("❌ Environment variable LITELLM_PROXY_URL is NOT set")
        print("  The agent will bypass LiteLLM and use OpenAI directly")
        return False
    
    # Create agent and check its configuration
    try:
        agent = create_analyst_agent()
        
        # Get LiteLLM status from agent
        status = agent.get_litellm_status()
        
        print(f"  Agent LiteLLM Status:")
        print(f"    - Using LiteLLM: {status['using_litellm']}")
        print(f"    - Proxy URL: {status['proxy_url'] or 'Not set'}")
        print(f"    - base_url in config: {status['base_url'] or 'Not set'}")
        
        if status['using_litellm'] and status['base_url']:
            if status['base_url'] == proxy_url.rstrip("/"):
                print("✓ base_url matches LITELLM_PROXY_URL")
                return True
            else:
                print(f"⚠ base_url ({status['base_url']}) doesn't match LITELLM_PROXY_URL ({proxy_url})")
                return False
        elif status['using_litellm']:
            print("✓ Agent is configured to use LiteLLM")
            return True
        else:
            print("❌ Agent is NOT configured to use LiteLLM")
            print("  This means the agent will bypass LiteLLM and use OpenAI directly")
            return False
            
    except Exception as e:
        print(f"❌ Error checking agent configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_litellm():
    """Test the agent and verify it uses LiteLLM."""
    print("\n" + "="*60)
    print("Testing Agent with LiteLLM")
    print("="*60)
    
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
    
    try:
        print("Creating AnalystAgent...")
        agent = create_analyst_agent()
        
        print("\nAnalyzing job description...")
        print("(Watch for LiteLLM proxy logs in the proxy server terminal)")
        print("-" * 60)
        
        analysis = agent.analyze(sample_jd)
        
        print("-" * 60)
        print("\n✓ Analysis completed successfully!")
        print(f"  Title: {analysis.title}")
        print(f"  Skills found: {len(analysis.skills)}")
        print(f"  Experience level: {analysis.experience_level}")
        
        # Check if we can verify LiteLLM usage
        proxy_url = os.getenv("LITELLM_PROXY_URL", "")
        if proxy_url:
            print(f"\n✓ If you see requests in the LiteLLM proxy logs, the agent is using LiteLLM")
            print(f"  Check the terminal where you started: litellm --config litellm.config.yaml --port 4000")
            print(f"  You should see log entries for the request")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during agent test: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_network_traffic():
    """Provide instructions for verifying network traffic."""
    print("\n" + "="*60)
    print("How to Verify LiteLLM is Being Used")
    print("="*60)
    
    proxy_url = os.getenv("LITELLM_PROXY_URL", "")
    
    if not proxy_url:
        print("⚠ LITELLM_PROXY_URL not set - cannot verify")
        return
    
    print("\n1. Check LiteLLM Proxy Logs:")
    print("   - Look at the terminal where you started the LiteLLM proxy")
    print("   - You should see log entries like:")
    print("     'POST /v1/chat/completions'")
    print("     'Model: gpt-4o'")
    print("     'Status: 200'")
    
    print("\n2. Check Network Traffic:")
    print("   - Use a network monitoring tool (Wireshark, tcpdump, etc.)")
    print(f"   - Filter for traffic to: {proxy_url}")
    print("   - You should see HTTP requests to the proxy")
    
    print("\n3. Check LiteLLM Dashboard (if enabled):")
    print("   - LiteLLM proxy may have a dashboard at http://localhost:4000/ui")
    print("   - Check for request logs and metrics")
    
    print("\n4. Monitor OpenAI API Usage:")
    print("   - If using LiteLLM, requests go to LiteLLM proxy first")
    print("   - If bypassing LiteLLM, requests go directly to api.openai.com")
    print("   - Check your OpenAI dashboard for API usage patterns")


def main():
    """Main verification function."""
    print("="*60)
    print("LiteLLM Usage Verification for Analyst Agent")
    print("="*60)
    
    # Step 1: Check if proxy is running
    print("\n[Step 1] Checking LiteLLM Proxy")
    print("-" * 60)
    proxy_running = check_litellm_proxy()
    
    if not proxy_running:
        print("\n⚠ LiteLLM proxy is not accessible")
        print("   Please start it with:")
        print("   litellm --config litellm.config.yaml --port 4000")
        print("\n   Then set the environment variable:")
        print("   export LITELLM_PROXY_URL=http://localhost:4000")
        return
    
    # Step 2: Check agent configuration
    print("\n[Step 2] Checking Agent Configuration")
    print("-" * 60)
    agent_configured = check_agent_configuration()
    
    if not agent_configured:
        print("\n❌ Agent is not properly configured to use LiteLLM")
        return
    
    # Step 3: Test the agent
    print("\n[Step 3] Testing Agent")
    print("-" * 60)
    test_passed = test_agent_with_litellm()
    
    # Step 4: Verification instructions
    verify_network_traffic()
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    if proxy_running and agent_configured and test_passed:
        print("✓ All checks passed!")
        print("✓ Agent appears to be configured to use LiteLLM")
        print("\n⚠ IMPORTANT: Verify by checking LiteLLM proxy logs")
        print("   If you see requests in the proxy logs, LiteLLM is being used")
        print("   If you don't see requests, the agent may still be bypassing LiteLLM")
    else:
        print("❌ Some checks failed")
        print("   Please review the output above and fix any issues")


if __name__ == "__main__":
    main()

