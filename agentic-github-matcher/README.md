# 🎯 Agentic GitHub Matcher

An intelligent multi-agent system that analyzes job descriptions and finds matching GitHub developers and repositories using AutoGen, LiteLLM, and Nemo Guardrails.

## 🌟 Features

- **Multi-Agent Architecture**: Three specialized agents working together

  - **AnalystAgent**: Extracts skills, requirements, and tech stack from job descriptions
  - **GitHubSearchAgent**: Searches GitHub for matching developers and repositories
  - **FormatterAgent**: Creates professional, formatted reports

- **LiteLLM Gateway**: All LLM calls routed through LiteLLM for flexibility and cost management

- **Safety Rails**: Nemo Guardrails for input validation and output safety

- **Multiple Output Formats**: Rich console, Markdown, plain text, or JSON

- **Beautiful CLI**: Interactive interface with progress indicators and colorful output

## 📁 Project Structure

```
agentic-github-matcher/
├── agents/
│   ├── __init__.py
│   ├── analyst.py        # Job description analyzer
│   ├── github_agent.py   # GitHub search agent (MCP-based)
│   └── formatter.py      # Report formatter
├── github_mcp/
│   ├── __init__.py
│   └── github_session.py # GitHub MCP session management
├── tools/
│   ├── __init__.py
│   └── github_mcp.py     # GitHub MCP integration (MCP-only)
├── guardrails/
│   ├── __init__.py
│   └── rails.yaml        # Safety configuration
├── litellm.config.yaml   # LiteLLM routing config
├── app.py                # Main application
├── env.template          # Environment template
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Prerequisites

**Node.js and npm** (required for GitHub MCP):

- Install Node.js from [https://nodejs.org/](https://nodejs.org/)
- This is required because GitHub MCP runs via `npx @modelcontextprotocol/server-github`
- Verify installation: `node --version` and `npm --version`

### 2. Clone and Setup

```bash
# Navigate to the project directory
cd agentic-github-matcher

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the environment template
cp env.template .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY: Your OpenAI API key
# - GITHUB_TOKEN: Your GitHub Personal Access Token
```

#### Getting API Keys

**OpenAI API Key:**

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy and paste into `.env`

**GitHub Token:**

1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens)
2. Generate new token (classic)
3. Select scopes: `public_repo`, `read:user`
4. Copy and paste into `.env`

### 4. Run the Application

```bash
# Interactive mode
python app.py

# Demo mode (uses sample job description)
python app.py --demo

# Process a job description file
python app.py --jd path/to/job_description.txt

# Specify output format
python app.py --demo --format markdown
```

## 📖 Usage Examples

### Interactive Mode

```bash
python app.py
```

Then paste your job description and type `END` on a new line when finished.

### Demo Mode

```bash
python app.py --demo
```

Runs with a sample Full-Stack Developer job description.

### File Input

```bash
python app.py --jd my_job_posting.txt
```

### Output Formats

```bash
# Rich console output (default)
python app.py --demo --format rich

# Markdown report
python app.py --demo --format markdown

# Plain text
python app.py --demo --format text

# JSON data
python app.py --demo --format json
```

### Specify Model

```bash
python app.py --demo --model gpt-4o-mini
```

## 🔧 Configuration

### LiteLLM Configuration

Edit `litellm.config.yaml` to:

- Add additional models
- Configure fallbacks
- Adjust timeouts and retries

### Guardrails Configuration

Edit `guardrails/rails.yaml` to:

- Add custom input validation rules
- Configure output safety checks
- Define blocked terms

### Environment Variables

| Variable         | Description                  | Required             |
| ---------------- | ---------------------------- | -------------------- |
| `OPENAI_API_KEY` | OpenAI API key               | Yes                  |
| `GITHUB_TOKEN`   | GitHub Personal Access Token | Yes                  |
| `DEFAULT_MODEL`  | Default LLM model            | No (default: gpt-4o) |
| `TEMPERATURE`    | Generation temperature       | No (default: 0.7)    |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                            │
│                    (Job Description)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Nemo Guardrails                           │
│                  (Input Validation)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     AnalystAgent                             │
│         Extracts: Skills, Tech Stack, Experience            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    LiteLLM                          │    │
│  │                  (GPT-4o API)                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  GitHubSearchAgent                           │
│       Searches: Repositories, Developers, Profiles          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            GitHub MCP (Model Context Protocol)      │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │  @modelcontextprotocol/server-github         │   │    │
│  │  │  (via npx stdio transport)                    │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │  Tools: search_users, get_user, search_repositories │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FormatterAgent                            │
│        Formats: Tables, Markdown, JSON Reports              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Rich Library                       │    │
│  │           (Beautiful Console Output)                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Nemo Guardrails                           │
│                  (Output Validation)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Professional Report                        │
│          (Candidates, Repositories, Analysis)               │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 GitHub MCP Integration

This project uses **GitHub MCP (Model Context Protocol)** exclusively for all GitHub operations. No REST API calls are made directly.

### What is MCP?

MCP (Model Context Protocol) is a standardized protocol for connecting AI applications to external data sources and tools. The GitHub MCP server provides a clean, standardized interface to GitHub's API.

### Why MCP Instead of REST?

- **Standardized Interface**: Consistent tool interface across different MCP servers
- **Better Abstraction**: Cleaner separation between application logic and API details
- **Built-in Rate Limiting**: MCP server handles rate limits automatically
- **Easier Integration**: Works seamlessly with agentic AI frameworks
- **Future-Proof**: Easy to swap or add new MCP servers

### MCP Server Setup

The GitHub MCP server runs automatically via:

```bash
npx @modelcontextprotocol/server-github
```

**Important Notes:**

- The server is started automatically by the application
- First connection may take 10-30 seconds (downloads package)
- Subsequent connections are faster (2-5 seconds, package cached)
- Requires Node.js and npm to be installed
- The server runs via stdio (standard input/output), not HTTP

### MCP Tools Available

The GitHub MCP server provides these tools (discovered dynamically):

- `search_users` - Search for GitHub users by query
- `get_user` - Get detailed user profile
- `get_user_repositories` - Get repositories for a user
- `search_repositories` - Search repositories by query
- `get_repository` - Get detailed repository information

### Environment Variables

| Variable       | Description                  | Required |
| -------------- | ---------------------------- | -------- |
| `GITHUB_TOKEN` | GitHub Personal Access Token | Yes      |

**Getting a GitHub Token:**

1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens)
2. Generate new token (classic)
3. Select scopes: `public_repo`, `read:user`
4. Copy and paste into `.env` as `GITHUB_TOKEN=your_token_here`

### MCP Session Management

MCP sessions are managed by `github_mcp/github_session.py`:

- Creates and manages connections to the GitHub MCP server
- Handles session lifecycle (connect, use, disconnect)
- Provides async context manager for clean resource management
- Automatically discovers available MCP tools at runtime

**Example Usage:**

```python
from github_mcp.github_session import create_github_mcp_session

async with create_github_mcp_session() as session:
    tools = await session.list_tools()
    result = await session.call_tool("search_users", {"query": "python developer"})
```

## 📊 Sample Output

```
╭───────────────────────────────────────────────────────────────╮
│              🎯 GitHub Talent Matcher Report                  │
╰───────────────────────────────────────────────────────────────╯

📋 JOB REQUIREMENTS ANALYSIS
──────────────────────────────────────────────
Position: Senior Full-Stack Developer
Experience: 5 years (senior)
Key Skills: Python, TypeScript, React, FastAPI, PostgreSQL
Tech Stack: Docker, Kubernetes, AWS

👥 MATCHED CANDIDATES
──────────────────────────────────────────────
┌──────┬──────────────────┬───────────┬───────┬─────────────────┬───────┐
│ Rank │ Developer        │ Followers │ Repos │ Skills Match    │ Score │
├──────┼──────────────────┼───────────┼───────┼─────────────────┼───────┤
│ 1    │ @example_dev     │    1,234  │    45 │ Python, React   │    85 │
│ 2    │ @another_dev     │      892  │    32 │ TypeScript, AWS │    72 │
└──────┴──────────────────┴───────────┴───────┴─────────────────┴───────┘

📦 TOP MATCHING REPOSITORIES
──────────────────────────────────────────────
┌────────────────────────────────┬─────────┬─────────┬────────────┐
│ Repository                     │ ⭐ Stars │ 🍴 Forks │ Language   │
├────────────────────────────────┼─────────┼─────────┼────────────┤
│ tiangolo/fastapi               │  65,000 │   5,400 │ Python     │
│ facebook/react                 │ 215,000 │  45,000 │ JavaScript │
└────────────────────────────────┴─────────┴─────────┴────────────┘
```

## 🛡️ Safety Features

- **Input Validation**: Blocks malicious prompts and injection attempts
- **Content Filtering**: Removes inappropriate or offensive content
- **Hallucination Detection**: Validates GitHub data against actual API responses
- **Rate Limiting**: MCP server handles GitHub API rate limits automatically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent framework
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM gateway
- [Nemo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Safety rails
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP framework
- [GitHub MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github) - Official GitHub MCP server

---

**Built with ❤️ using Agentic AI**
