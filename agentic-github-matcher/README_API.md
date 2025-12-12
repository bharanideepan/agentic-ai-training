# Agentic GitHub Matcher - Full Stack Application

Complete guide to running the full-stack application with FastAPI backend and React frontend.

## Architecture

- **Backend**: FastAPI (Python) - `api/main.py`
- **Frontend**: React + Vite - `frontend/`
- **Communication**: RESTful API with streaming support

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env and add your OPENAI_API_KEY and GITHUB_TOKEN

# Run the API server
cd api
python main.py
```

The API will run on `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. Paste a job description in the form
3. Click "Analyze Job Description"
4. Watch the real-time progress and results stream in

## API Endpoints

- `GET /` - API information
- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze job description (streaming)

## Features

✅ Real-time streaming of analysis progress  
✅ Job description analysis with skill extraction  
✅ GitHub candidate matching with exact/partial match prioritization  
✅ PII detection and masking  
✅ Guardrails for input validation  
✅ Modern React UI with Tailwind CSS  
✅ Responsive design  

## Development

### Backend Development
```bash
# Run with auto-reload
uvicorn api.main:app --reload --port 8000
```

### Frontend Development
```bash
cd frontend
npm run dev
```

## Production Deployment

### Backend
```bash
# Build and run with production server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
cd frontend
npm run build
# Serve the dist/ directory with a web server
```

## Troubleshooting

1. **CORS errors**: Make sure the backend CORS settings include your frontend URL
2. **API connection failed**: Verify the backend is running on port 8000
3. **No candidates found**: Check that your GitHub token has proper permissions

