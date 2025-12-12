# Agentic GitHub Matcher - API

FastAPI backend with streaming support for job description analysis and GitHub candidate matching.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token
API_PORT=8000
```

3. Run the API server:
```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Root endpoint with API information

### GET `/api/health`
Health check endpoint

### POST `/api/analyze`
Analyze job description and find matching GitHub candidates.

**Request Body:**
```json
{
  "job_description": "We are looking for a Senior Python Developer...",
  "model": "gpt-4o",
  "max_candidates": 10,
  "output_format": "json"
}
```

**Response:**
Streaming JSON (NDJSON format) with progress updates and results.

**Example Response Chunks:**
```json
{"type": "status", "message": "Analyzing job description...", "progress": 10}
{"type": "progress", "message": "Found 5 key skills", "data": {...}, "progress": 30}
{"type": "result", "message": "Analysis complete!", "data": {...}, "progress": 100}
```

## Testing

Test the API with curl:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python Developer with Django experience",
    "max_candidates": 5
  }'
```

