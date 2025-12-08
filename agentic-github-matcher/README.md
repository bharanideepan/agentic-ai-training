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
│   ├── github_agent.py   # GitHub search agent
│   └── formatter.py      # Report formatter
├── tools/
│   ├── __init__.py
│   └── github_search.py  # GitHub API functions
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

### 1. Clone and Setup

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

### 2. Configure Environment

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

### 3. Run the Application

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
│  │              GitHub REST API                        │    │
│  │    (search_repos, fetch_profile, fetch_repos)      │    │
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
- **Rate Limiting**: Respects GitHub API rate limits

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

---

**Built with ❤️ using Agentic AI**
