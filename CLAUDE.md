# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Plan & Review

### Before starting work
- Always in plan mode to make a plan
- After get the plan, make sure you Write the plan to .claude/tasks/TASK_NAME.md.
- The plan should be a detailed implementation plan and the reasoning behind them, as well as tasks broken down.
- If the task require external knowledge or certain package, also research to get latest knowledge (Use Task tool for research).
- Don't over plan it, always think MVP.
- Once you write the plan, firstly ask me to review it. Do not continue until I approve the plan.

### While implementing
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the changes you made, so following tasks can be easily hand over to other engineers.




## Development Commands

### Backend Development
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start backend with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run basic API test
python ../test_api.py
```

### Frontend Development  
```bash
# Navigate to frontend directory
cd frontend-standalone

# Serve with Python (simple)
python -m http.server 3000

# Or serve with live reload (if available)
npx live-server --port=3000
```

### Docker Development
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

### Testing
- Main test script: `python test_api.py` (tests backend API endpoints)
- Health check: `curl http://localhost:8000/health`
- API docs: http://localhost:8000/docs (when backend is running)

## Architecture Overview

### Separated Services Architecture
The application uses a **separated frontend/backend architecture**:

- **Backend**: FastAPI-based REST API (`backend/` directory)
  - Main entry: `backend/main.py` 
  - Core logic: `backend/fact_checker.py`
  - API models: `backend/models.py`
  - Multi-source checking: `backend/fact_check_apis.py`

- **Frontend**: Static web application (`frontend-standalone/` directory)
  - Communicates with backend via HTTP APIs
  - Can be deployed independently

### Key Backend Components

1. **Main FastAPI App** (`backend/main.py`)
   - Request handling and validation 
   - Audio processing with Whisper
   - Rate limiting (30 requests/minute for analysis)
   - CORS configuration for frontend communication
   - Professional vs. basic fact-checking modes

2. **Fact Checking Engine** (`backend/fact_checker.py`)
   - `AlternativeFactChecker`: Basic fact-checking with built-in knowledge base
   - `EnhancedFactChecker`: Advanced fact-checking with LangChain/FAISS (optional)
   - Audio prosody analysis for sarcasm detection
   - Sentiment analysis integration

3. **Multi-Source APIs** (`backend/fact_check_apis.py`)
   - Google Fact Check Tools API integration
   - OpenAI GPT integration for claim analysis
   - Multiple fact-checking source aggregation

4. **API Models** (`backend/models.py`)
   - Pydantic models for request/response validation
   - `AnalysisRequest`, `AnalysisResponse`, `ProsodyAnalysis` etc.

### Operating Modes

The backend operates in two modes based on API key configuration:

- **Basic Mode**: Uses built-in knowledge base for common conspiracy theories
- **Professional Mode**: Uses external APIs (Google Fact Check, OpenAI) when configured

## Environment Configuration

### Required Environment Variables
Create `backend/.env` from `backend/.env.example`:

```bash
# Required for basic functionality
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Optional for professional mode
GOOGLE_FACT_CHECK_API_KEY=your_google_key_here
NEWS_API_KEY=your_news_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Important Configuration Notes

- **API Keys**: The system gracefully degrades when professional APIs are unavailable
- **CORS**: Currently configured to allow all origins (`allow_origins=["*"]`)
- **Rate Limiting**: 30 requests/minute for analysis endpoints
- **Audio Support**: Supports WAV, MP3, M4A, OGG formats up to 50MB
- **Docker**: Uses multi-service setup with Redis caching and Nginx proxy options

## Key API Endpoints

- `POST /api/analyze` - Main fact-checking endpoint (text + optional audio)
- `POST /api/analyze-file` - File upload variant for audio analysis  
- `GET /health` - Health check
- `GET /api/status` - API configuration and recommendations
- `GET /docs` - Interactive API documentation (Swagger UI)

## Development Patterns

### Error Handling
- Graceful fallback from professional to basic fact-checking
- User-friendly error messages with suggestions
- Timeout handling for long-running audio processing

### Audio Processing
- Uses OpenAI Whisper for transcription
- Librosa for prosody analysis (pitch, energy, sarcasm detection)
- Base64 encoding for API transport

### Caching and Performance
- Request caching for repeated claims
- Optional Redis integration for distributed caching
- GPU support detection (falls back to CPU)

### Code Organization
- Separated concerns: main app, fact checking, API models
- Async/await throughout for performance
- Comprehensive logging with request tracking

## Security Considerations

- Input validation on all endpoints
- File size limits for audio uploads
- Rate limiting to prevent abuse
- Environment variable based configuration (no hardcoded secrets)
- CORS configuration for frontend communication