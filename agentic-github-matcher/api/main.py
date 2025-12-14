"""
FastAPI Backend for Agentic GitHub Matcher
===========================================

RESTful API with streaming support for job description analysis
and GitHub candidate matching.
"""

import os
import json
import asyncio
import queue
import threading
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import workflow components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    initialize_guardrails,
    AgenticWorkflow,
    DEFAULT_MODEL,
    TEMPERATURE
)

# ==============================================
# FASTAPI APP INITIALIZATION
# ==============================================

app = FastAPI(
    title="Agentic GitHub Matcher API",
    description="AI-powered job description analyzer and GitHub candidate matcher",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# GLOBAL STATE
# ==============================================

# Initialize guardrails once at startup
guardrails = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global guardrails
    print("🚀 Initializing API...")
    
    # Initialize guardrails
    print("  📋 Initializing guardrails...")
    guardrails = initialize_guardrails()
    print("  ✓ Guardrails initialized")
    
    # Note: MCP client is initialized on-demand by agents
    # No global initialization needed - each agent creates its own MCP session
    print("  ℹ GitHub MCP will be initialized by agents as needed")
    
    print("✅ API ready!")


# ==============================================
# REQUEST/RESPONSE MODELS
# ==============================================

class JobDescriptionRequest(BaseModel):
    """Request model for job description analysis."""
    job_description: str = Field(..., description="The job description text to analyze")
    model: Optional[str] = Field(default=DEFAULT_MODEL, description="LLM model to use")
    max_candidates: int = Field(default=10, ge=1, le=50, description="Maximum number of candidates to return")
    output_format: str = Field(default="json", description="Output format: json, markdown, or text")


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    type: str = Field(..., description="Type of chunk: status, progress, result, error")
    message: Optional[str] = Field(None, description="Status or error message")
    data: Optional[dict] = Field(None, description="Data payload")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")


# ==============================================
# API ENDPOINTS
# ==============================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic GitHub Matcher API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze (POST) - Analyze job description and find candidates",
            "health": "/api/health (GET) - Health check"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    # Check MCP client status
    mcp_status = "unknown"
    try:
        from tools.github_mcp import _mcp_client, _mcp_connection_failed, MCP_AVAILABLE
        if not MCP_AVAILABLE:
            mcp_status = "sdk_not_available"
        elif _mcp_connection_failed:
            mcp_status = "connection_failed"
        elif _mcp_client and _mcp_client._initialized:
            mcp_status = "connected"
        else:
            mcp_status = "not_initialized"
    except Exception:
        mcp_status = "error"
    
    return {
        "status": "healthy",
        "guardrails": "enabled" if guardrails else "disabled",
        "mcp_client": mcp_status
    }


async def process_workflow_stream(
    job_description: str,
    model: str,
    max_candidates: int,
    output_format: str
):
    """
    Process workflow and yield streaming chunks.
    
    Args:
        job_description: Job description text
        model: LLM model to use
        max_candidates: Maximum candidates
        output_format: Output format
    """
    global workflow
    
    try:
        # Helper to create JSON chunks
        def create_chunk(chunk_type: str, message: str, progress: float = None, data: dict = None):
            chunk = {"type": chunk_type, "message": message}
            if progress is not None:
                chunk["progress"] = progress
            if data is not None:
                chunk["data"] = data
            return json.dumps(chunk) + "\n"
        # Initialize workflow if not already done
        # Note: We create a new workflow instance for each request to avoid state issues
        current_workflow = AgenticWorkflow(
            model=model,
            temperature=TEMPERATURE,
            guardrails=guardrails
        )
        
        yield create_chunk("status", "Initializing agents...", 0)
        
        # Step 1: Guardrails validation
        yield create_chunk("status", "Validating input...", 5)
        
        from app import apply_input_guardrails
        is_safe, message = apply_input_guardrails(guardrails, job_description)
        if not is_safe:
            yield create_chunk("error", f"Input blocked: {message}")
            return
        
        # Step 2: Analyze job description
        yield create_chunk("status", "Analyzing job description...", 10)
        
        job_analysis = current_workflow.analyst.analyze(job_description)
        searchable_skills = job_analysis.get_searchable_skills()
        
        yield create_chunk(
            "progress",
            f"Found {len(searchable_skills)} key skills",
            30,
            {
                "skills": searchable_skills[:10],
                "analysis": job_analysis.to_dict()
            }
        )
        
        # Step 3: Search GitHub with real-time progress streaming
        # Use a shared list to capture progress updates (thread-safe)
        progress_updates = []
        progress_lock = threading.Lock()
        search_done = threading.Event()
        search_result_container = {"result": None, "error": None}
        
        def capture_progress(message: str, progress: float):
            """Capture progress updates for streaming."""
            with progress_lock:
                progress_updates.append((message, progress))
        
        async def run_search():
            """Run search asynchronously."""
            try:
                analysis_dict = job_analysis.to_dict()
                result = await current_workflow.github_agent.search(
                    skills=searchable_skills,
                    job_analysis=analysis_dict,
                    max_candidates=max_candidates,
                    progress_callback=capture_progress
                )
                search_result_container["result"] = result
            except Exception as e:
                search_result_container["error"] = str(e)
            finally:
                search_done.set()
        
        analysis_dict = job_analysis.to_dict()
        
        # Using GitHub MCP (async)
        github_strategy = "GitHub MCP"
        
        # Stream initial search status with strategy info
        yield create_chunk("status", f"Searching GitHub repositories (via {github_strategy})...", 35, {
            "strategy": github_strategy,
            "search_method": "mcp"
        })
        
        # Start search as async task
        search_task = asyncio.create_task(run_search())
        
        # Stream progress updates in real-time while search is running
        last_update_index = 0
        max_wait_time = 300  # 5 minutes timeout
        wait_count = 0
        max_wait_iterations = max_wait_time * 20  # Check every 0.05 seconds for better responsiveness
        
        while not search_done.is_set() and wait_count < max_wait_iterations:
            # Check for new progress updates
            with progress_lock:
                current_updates = progress_updates[last_update_index:]
                last_update_index = len(progress_updates)
            
            # Stream new progress updates immediately with strategy info
            if current_updates:
                for message, progress_val in current_updates:
                    yield create_chunk("status", message, progress_val, {
                        "strategy": github_strategy,
                        "search_method": "mcp"
                    })
            
            # Small delay to prevent busy waiting (reduced for better responsiveness)
            await asyncio.sleep(0.05)
            wait_count += 1
        
        # Wait for async task to complete
        try:
            await asyncio.wait_for(search_task, timeout=5.0)
        except asyncio.TimeoutError:
            pass  # Task may still be running, but we'll check results anyway
        
        # Check for errors
        if search_result_container["error"]:
            yield create_chunk("error", f"Search failed: {search_result_container['error']}")
            return
        
        if not search_result_container["result"]:
            yield create_chunk("error", "Search timed out or returned no results")
            return
        
        search_results_final = search_result_container["result"]
        
        # Stream any final progress updates with strategy info
        with progress_lock:
            final_updates = progress_updates[last_update_index:]
        for message, progress_val in final_updates:
            yield create_chunk("status", message, progress_val, {
                "strategy": github_strategy,
                "search_method": "direct"
            })
        
        yield create_chunk(
            "progress",
            f"Found {len(search_results_final.developers)} candidates",
            80,
            {
                "candidates_count": len(search_results_final.developers),
                "repos_count": search_results_final.total_repos_found,
                "strategy": github_strategy,
                "search_method": "direct"
            }
        )
        
        # Step 4: Format results using agentic behavior
        yield create_chunk("status", "Formatting results with AI insights...", 85)
        
        results_dict = search_results_final.to_dict()
        
        # Always use FormatterAgent for all formats (including JSON) to ensure agentic behavior
        formatted_output = current_workflow.formatter.format_results(
            analysis_dict,
            results_dict,
            output_format=output_format
        )
        
        # Send final result with strategy info
        # For JSON format, formatted_output contains the complete agentic JSON with AI-generated insights
        # For other formats, formatted_output contains the formatted string
        yield create_chunk(
            "result",
            "Analysis complete!",
            100,
            {
                "analysis": analysis_dict,
                "results": results_dict,
                "formatted_output": formatted_output,  # Always include formatted output (agentic for all formats)
                "strategy": github_strategy,
                "search_method": "direct"
            }
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_message = str(e)
        # Provide more helpful error messages
        if "name 'os' is not defined" in error_message:
            error_message = "Configuration error: Missing 'os' module import. Please check agent initialization."
        elif "OPENAI_API_KEY" in error_message or "api_key" in error_message.lower():
            error_message = f"API Key error: {error_message}. Please check your .env file."
        yield create_chunk(
            "error",
            error_message,
            data={"traceback": error_trace, "error_type": type(e).__name__}
        )


@app.post("/api/analyze")
async def analyze_job_description(request: JobDescriptionRequest):
    """
    Analyze job description and find matching GitHub candidates.
    
    This endpoint streams the results as they become available.
    
    Args:
        request: Job description request
        
    Returns:
        StreamingResponse: JSON stream of progress updates and results
    """
    if not request.job_description or not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    async def generate_stream():
        """Async generator wrapper to ensure proper streaming."""
        async for chunk in process_workflow_stream(
            job_description=request.job_description,
            model=request.model,
            max_candidates=request.max_candidates,
            output_format=request.output_format
        ):
            yield chunk
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",  # Newline-delimited JSON
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Transfer-Encoding": "chunked"  # Enable chunked transfer
        }
    )


# ==============================================
# RUN SERVER
# ==============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

