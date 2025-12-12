# Agentic GitHub Matcher - Project Analysis

## Executive Summary

The **Agentic GitHub Matcher** is an intelligent multi-agent system that analyzes job descriptions and automatically finds matching GitHub developers and repositories. Built using AutoGen, LiteLLM, and Nemo Guardrails, it demonstrates sophisticated agentic AI behavior through specialized agents that collaborate to solve complex recruitment tasks.

---

## Architecture Overview

### System Architecture

The system follows a **multi-agent orchestration pattern** with three specialized agents working in a sequential pipeline, coordinated by a central workflow orchestrator. Each agent has a specific role and communicates through structured data formats.

### Core Components

1. **Workflow Orchestrator** (`app.py`)

   - Coordinates the entire pipeline
   - Manages agent lifecycle
   - Handles input/output validation
   - Provides CLI and API interfaces

2. **AnalystAgent** (`agents/analyst.py`)

   - **Purpose**: Extracts structured information from job descriptions
   - **Input**: Raw job description text
   - **Output**: Structured `JobAnalysis` object with skills, experience, tech stack
   - **LLM Model**: GPT-4o via LiteLLM
   - **Temperature**: 0.3 (low for consistent analysis)

3. **GitHubSearchAgent** (`agents/github_agent.py`)

   - **Purpose**: Searches GitHub for matching developers and repositories
   - **Input**: Extracted skills and job analysis
   - **Output**: `SearchResults` with candidate profiles and repositories
   - **LLM Model**: GPT-4o via LiteLLM
   - **Temperature**: 0.5 (moderate for tool calling)
   - **Tools**: GitHub API functions (search_repositories_by_skills, fetch_user_profile, etc.)

4. **FormatterAgent** (`agents/formatter.py`)

   - **Purpose**: Formats results into professional reports
   - **Input**: Job analysis and search results
   - **Output**: Formatted reports (Rich, Markdown, JSON, Text)
   - **LLM Model**: GPT-4o via LiteLLM
   - **Temperature**: 0.3 (low for consistent formatting)

5. **GitHub Tools** (`tools/github_search.py` and `tools/github_mcp.py`)

   - **Primary**: GitHub MCP integration (`github_mcp.py`) - Uses Model Context Protocol for standardized GitHub API access
   - **Fallback**: Direct REST API calls (`github_search.py`) - Used when MCP is unavailable
   - Repository search functions
   - User profile fetching
   - Organization filtering
   - Rate limiting and error handling
   - **MCP Benefits**: Standardized interface, better abstraction, built-in rate limiting

6. **Guardrails** (`guardrails/rails.yaml`)

   - Input validation (PII detection, pattern blocking)
   - Output safety checks
   - Content filtering

7. **LiteLLM Gateway** (`litellm.config.yaml`)

   - Unified model routing
   - Fallback mechanisms
   - Cost management

8. **GitHub MCP Integration** (`tools/github_mcp.py`)

   - Model Context Protocol (MCP) client for GitHub
   - Connects to GitHub MCP server via stdio
   - Provides standardized tool interface
   - Automatic fallback to direct REST API if MCP unavailable

9. **FastAPI Backend** (`api/main.py`)
   - RESTful API with streaming support
   - Real-time progress updates
   - CORS-enabled for frontend integration

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   CLI (app)   │  │  FastAPI API │  │   Frontend   │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │   Workflow Orchestrator (app.py)    │
          │  - Input Validation                 │
          │  - Agent Coordination                │
          │  - Progress Tracking                │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      Nemo Guardrails                │
          │  - PII Detection & Masking          │
          │  - Pattern-based Validation         │
          │  - Content Safety Checks            │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      AnalystAgent                   │
          │  ┌───────────────────────────────┐ │
          │  │  System Prompt:               │ │
          │  │  "Extract skills, experience, │ │
          │  │   tech stack from JD"         │ │
          │  └───────────────────────────────┘ │
          │  ┌───────────────────────────────┐ │
          │  │      LiteLLM Gateway          │ │
          │  │      (GPT-4o API)             │ │
          │  └───────────────────────────────┘ │
          │  Output: JobAnalysis (structured)  │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │    GitHubSearchAgent                │
          │  ┌───────────────────────────────┐ │
          │  │  System Prompt:               │ │
          │  │  "Search GitHub for matching  │ │
          │  │   developers and repos"        │ │
          │  └───────────────────────────────┘ │
          │  ┌───────────────────────────────┐ │
          │  │      LiteLLM Gateway          │ │
          │  │      (GPT-4o with Tools)       │ │
          │  └───────────────────────────────┘ │
          │  ┌───────────────────────────────┐ │
          │  │    GitHub MCP Client          │ │
          │  │  (Model Context Protocol)     │ │
          │  └───────────────┬───────────────┘ │
          │                  │                 │
          │  ┌───────────────▼───────────────┐ │
          │  │    GitHub MCP Server          │ │
          │  │  (Standardized Tool Interface)│ │
          │  └───────────────┬───────────────┘ │
          │                  │                 │
          │  ┌───────────────▼───────────────┐ │
          │  │    GitHub REST API            │ │
          │  │    (api.github.com)           │ │
          │  └───────────────────────────────┘ │
          │  Note: Falls back to direct API    │
          │  calls if MCP is unavailable      │
          │  Output: SearchResults (candidates)│
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      FormatterAgent                 │
          │  ┌───────────────────────────────┐ │
          │  │  System Prompt:               │ │
          │  │  "Format results professionally│ │
          │  │   for recruitment use"        │ │
          │  └───────────────────────────────┘ │
          │  ┌───────────────────────────────┐ │
          │  │      LiteLLM Gateway          │ │
          │  │      (GPT-4o API)             │ │
          │  └───────────────────────────────┘ │
          │  ┌───────────────────────────────┐ │
          │  │      Rich Library             │ │
          │  │  (Beautiful Console Output)   │ │
          │  └───────────────────────────────┘ │
          │  Output: Formatted Report          │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      Nemo Guardrails                │
          │  - Output Validation                 │
          │  - Hallucination Detection          │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      Professional Report            │
          │  - Candidate Profiles               │
          │  - Repository Highlights            │
          │  - Skill Match Analysis             │
          └─────────────────────────────────────┘
```

### Data Flow

```
Job Description (Text)
    │
    ├─► [Guardrails] Input Validation
    │       ├─► PII Detection & Masking
    │       └─► Pattern-based Blocking
    │
    ├─► [AnalystAgent] Extract Structure
    │       ├─► LLM Analysis (GPT-4o)
    │       └─► JobAnalysis {
    │             title, skills, experience,
    │             tech_stack, frameworks, etc.
    │           }
    │
    ├─► [GitHubSearchAgent] Search & Match
    │       ├─► [GitHub MCP Client] Connect to MCP Server
    │       │     └─► Standardized tool interface
    │       │
    │       ├─► Multi-strategy Repository Search (via MCP)
    │       │     ├─► Language-based search
    │       │     └─► Topic-based search
    │       │
    │       ├─► Multi-strategy User Search (via MCP)
    │       │     ├─► Individual skill searches
    │       │     └─► Combined skill searches
    │       │
    │       ├─► Profile Fetching & Filtering (via MCP)
    │       │     ├─► Organization filtering
    │       │     └─► Skill extraction from repos
    │       │
    │       ├─► LLM-based Batch Scoring
    │       │     └─► Relevance scoring (0-100)
    │       │
    │       └─► SearchResults {
    │             developers: [DeveloperMatch],
    │             repositories: [Repository],
    │             total_count, etc.
    │           }
    │       Note: Falls back to direct REST API if MCP unavailable
    │
    ├─► [FormatterAgent] Format Output
    │       ├─► Rich Console Format
    │       ├─► Markdown Report
    │       ├─► JSON Data
    │       └─► Plain Text
    │
    └─► [Guardrails] Output Validation
            └─► Final Report
```

---

## Agentic Behavior

### What Makes This System "Agentic"

The system demonstrates true agentic AI behavior through:

#### 1. **Autonomous Decision-Making**

- **AnalystAgent** autonomously decides which information to extract from job descriptions
- **GitHubSearchAgent** makes intelligent decisions about:
  - Which search strategies to use (language-based vs. topic-based)
  - When to expand searches if insufficient candidates found
  - How to prioritize and rank candidates
- **FormatterAgent** autonomously structures reports based on data patterns

#### 2. **Tool Use and External API Integration**

- **GitHubSearchAgent** uses function calling to interact with GitHub API
- Agents can dynamically choose which tools to use based on context
- Tool results influence subsequent agent decisions

#### 3. **Adaptive Behavior**

- **Multi-Strategy Search**: If initial search yields few results, the agent automatically tries alternative strategies
- **Expansion Logic**: If fewer candidates than requested are found, the agent expands search parameters
- **Batch Processing**: Agents optimize API calls by batching operations (e.g., batch LLM scoring)

#### 4. **Context-Aware Processing**

- **Skill Matching**: Agents understand semantic relationships between skills (e.g., "Python" and "Django" are related)
- **Experience Level Matching**: Scoring considers job requirements (senior vs. junior roles)
- **Repository Quality Assessment**: Agents evaluate repository stars, forks, and activity

#### 5. **Intelligent Scoring and Ranking**

- **LLM-based Scoring**: Uses GPT-4o to evaluate candidate fit beyond keyword matching
- **Multi-factor Evaluation**: Considers skill overlap, repository quality, activity level, and job requirements
- **Exact vs. Partial Matching**: Distinguishes between candidates with 80%+ skill match vs. partial matches

#### 6. **Self-Correction and Error Handling**

- Agents handle API failures gracefully
- Fallback mechanisms (e.g., skill match percentage if LLM scoring fails)
- Retry logic for transient failures

### Agent Communication Pattern

The agents communicate through **structured data objects**:

1. **AnalystAgent → GitHubSearchAgent**

   - Passes `JobAnalysis` object with extracted skills
   - Provides context for better matching

2. **GitHubSearchAgent → FormatterAgent**

   - Passes `SearchResults` with candidate profiles
   - Includes relevance scores and matching skills

3. **Workflow Orchestrator**
   - Coordinates all agent interactions
   - Manages state and progress tracking
   - Handles error propagation

### Example Agentic Workflow

```
1. User provides job description: "Senior Python Developer with Django experience"

2. AnalystAgent analyzes:
   - Extracts: Python, Django, PostgreSQL, AWS
   - Determines: Senior level, 5+ years
   - Creates structured JobAnalysis

3. GitHubSearchAgent searches:
   - Strategy 1: Search repos by "python" language → finds 1000 repos
   - Strategy 2: Search repos by "django" topic → finds 500 repos
   - Strategy 3: Search users by "python" → finds 200 users
   - Strategy 4: Search users by "django" → finds 150 users
   - Fetches profiles for top candidates
   - Filters out organizations
   - Extracts matching skills from repos
   - Scores candidates with LLM (batch processing)
   - Ranks by relevance score

4. FormatterAgent formats:
   - Creates professional report
   - Highlights top 10 candidates
   - Includes repository highlights
   - Generates markdown/JSON/text output

5. System returns formatted report with actionable insights
```

---

## Technology Stack

### Core Frameworks

- **AutoGen (0.2.35)**: Multi-agent framework for agent orchestration
- **LiteLLM**: Unified model gateway for LLM routing
- **Nemo Guardrails**: Safety and validation framework
- **FastAPI**: Modern async web framework for API
- **Rich**: Beautiful terminal output library

### LLM Integration

- **Primary Model**: GPT-4o (via OpenAI API)
- **Fallback Model**: GPT-4o-mini
- **Temperature Settings**:
  - AnalystAgent: 0.3 (consistent analysis)
  - GitHubSearchAgent: 0.5 (balanced tool calling)
  - FormatterAgent: 0.3 (consistent formatting)

### External APIs

- **GitHub MCP Server**: Model Context Protocol server for GitHub (primary)
- **GitHub REST API**: Direct API access (fallback when MCP unavailable)
- **OpenAI API**: LLM inference

### Data Structures

- **JobAnalysis**: Structured job description analysis
- **DeveloperMatch**: Candidate profile with scoring
- **SearchResults**: Container for all search results

---

## Key Features

### 1. Multi-Agent Collaboration

- Three specialized agents working in sequence
- Each agent has a focused responsibility
- Agents communicate through structured data

### 2. Intelligent Search Strategies (via GitHub MCP)

- **MCP-based access**: Standardized GitHub API access through Model Context Protocol
- **Language-based search**: For programming languages
- **Topic-based search**: For frameworks and tools
- **User search**: Multiple query strategies
- **Organization filtering**: Excludes companies/orgs
- **Automatic fallback**: Falls back to direct REST API if MCP unavailable

### 3. Advanced Scoring

- **LLM-based batch scoring**: Efficient candidate evaluation
- **Multi-factor scoring**: Skill match, repo quality, activity
- **Exact vs. partial matching**: Prioritizes strong matches

### 4. Safety and Validation

- **PII Detection**: Automatically masks sensitive information
- **Input Validation**: Blocks malicious patterns
- **Output Validation**: Ensures professional output

### 5. Multiple Output Formats

- **Rich Console**: Beautiful terminal output
- **Markdown**: Professional reports
- **JSON**: Machine-readable data
- **Plain Text**: Simple text output

### 6. Real-time Progress Tracking

- Streaming API with progress updates
- Real-time status messages
- Progress percentage tracking

---

## Agent Responsibilities

### AnalystAgent

- **Input**: Raw job description text
- **Processing**: LLM-based extraction of structured information
- **Output**: JobAnalysis object with:
  - Job title
  - Required skills
  - Experience level and years
  - Tech stack, frameworks, databases
  - Cloud platforms, certifications
  - Domain knowledge
  - Summary

### GitHubSearchAgent

- **Input**: Extracted skills and job analysis
- **Processing**:
  - **GitHub MCP Integration**: Connects to GitHub MCP server for standardized API access
  - Multi-strategy repository search (via MCP tools)
  - Multi-strategy user search (via MCP tools)
  - Profile fetching and filtering (via MCP tools)
  - Skill extraction from repositories
  - LLM-based batch scoring
  - Intelligent ranking
  - **Fallback mechanism**: Uses direct REST API if MCP unavailable
- **Output**: SearchResults with:
  - Matched developers (with scores)
  - Relevant repositories
  - Match statistics

### FormatterAgent

- **Input**: Job analysis and search results
- **Processing**: Format data into professional reports
- **Output**: Formatted reports in multiple formats

---

## Configuration Management

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `GITHUB_TOKEN`: GitHub Personal Access Token (required for MCP)
- `MCP_SERVER_PATH`: (Optional) Path to GitHub MCP server executable (default: npx)
- `DEFAULT_MODEL`: Default LLM model (default: gpt-4o)
- `TEMPERATURE`: Generation temperature (default: 0.7)

### Configuration Files

- `litellm.config.yaml`: Model routing configuration
- `guardrails/rails.yaml`: Safety rail configuration
- `config.py`: Centralized configuration management

---

## API Endpoints

### FastAPI Backend (`/api/`)

- `POST /api/analyze`: Analyze job description and find candidates
  - **Request**: Job description, model, max_candidates, output_format
  - **Response**: Streaming JSON with progress updates and results
- `GET /api/health`: Health check endpoint

---

## Error Handling and Resilience

1. **API Failures**: Graceful degradation with fallback scores
2. **Rate Limiting**: Respects GitHub API rate limits
3. **Timeout Handling**: Configurable timeouts for all operations
4. **Input Validation**: Blocks invalid or malicious input
5. **Output Validation**: Ensures professional and accurate output

---

## Performance Optimizations

1. **Batch LLM Scoring**: Scores multiple candidates in a single API call
2. **Parallel Search Strategies**: Multiple search strategies run concurrently
3. **Caching**: Profile data cached during search
4. **Streaming**: Real-time progress updates via streaming API
5. **Organization Filtering**: Early filtering to reduce API calls

---

## Security Features

1. **PII Detection**: Automatically detects and masks:

   - Email addresses
   - Phone numbers
   - Credit card numbers
   - SSN
   - IP addresses
   - API keys

2. **Input Validation**: Blocks:

   - Injection attempts
   - Malicious patterns
   - Jailbreak attempts

3. **Output Validation**: Ensures:
   - Professional language
   - Factual accuracy
   - No hallucinated data

---

## Future Enhancements

Potential improvements:

- **Parallel Agent Execution**: Run agents in parallel where possible
- **Caching Layer**: Cache job analyses and search results
- **Advanced Matching**: Semantic similarity for skills
- **Multi-model Support**: Support for additional LLM providers
- **Analytics Dashboard**: Track search performance and success rates
- **Webhook Support**: Notify external systems of results

---

## GitHub MCP Integration

### What is Model Context Protocol (MCP)?

MCP is an open protocol designed to facilitate seamless integration between LLM applications and external data sources. For GitHub, MCP provides:

- **Standardized Interface**: Uniform command structure for GitHub operations
- **Better Abstraction**: Simplified interaction with GitHub API
- **Built-in Features**: Rate limiting, error handling, and authentication management
- **Interoperability**: Works with other MCP-compatible tools

### Implementation Details

The system uses GitHub MCP as the primary method for GitHub API access:

1. **MCP Client** (`tools/github_mcp.py`):

   - Connects to GitHub MCP server via stdio
   - Provides wrapper functions for repository search, user search, profile fetching
   - Handles MCP protocol communication

2. **GitHub MCP Server**:

   - Official GitHub MCP server (via `@modelcontextprotocol/server-github`)
   - Runs via npx or can be installed locally
   - Provides standardized tools for GitHub operations

3. **Fallback Mechanism**:
   - If MCP is unavailable, automatically falls back to direct REST API calls
   - Ensures system reliability even if MCP setup fails
   - Seamless transition between MCP and direct API

### Benefits of Using GitHub MCP

1. **Standardized Tool Interface**: Consistent interface reduces complexity
2. **Enhanced Abstraction**: Focus on functionality, not API details
3. **Built-in Rate Limiting**: MCP handles GitHub API rate limits automatically
4. **Error Handling**: Robust error handling and retry logic
5. **Future-Proof**: Easy to integrate additional MCP-compatible tools

## Conclusion

The Agentic GitHub Matcher demonstrates sophisticated agentic AI behavior through:

- **Autonomous decision-making** by specialized agents
- **Intelligent tool use** via GitHub MCP for standardized external API integration
- **Adaptive behavior** based on search results
- **Context-aware processing** for better matching
- **Self-correction** and error handling with automatic fallback mechanisms

The system successfully combines multiple AI agents with Model Context Protocol to solve a complex real-world problem (recruitment matching) with high accuracy, professional output, and standardized tool integration.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Project**: Agentic GitHub Matcher
