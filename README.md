# Enhanced Conspiracy Theory Debunker - Separated Architecture

A modern web application for fact-checking claims with advanced AI-powered analysis, audio processing, and real-time verification. Now with separated frontend and backend for independent deployment.

## 🏗️ Architecture Overview

The application is now split into two independent services:

- **Backend**: FastAPI-based REST API with AI processing capabilities
- **Frontend**: Static web application that communicates with the backend via HTTP APIs

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd Debunker
   ```

2. **Configure environment**:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your API keys
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Option 2: Manual Setup

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the backend**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend-standalone
   ```

2. **Serve static files** (using Python):
   ```bash
   python -m http.server 3000
   ```

   Or using Node.js:
   ```bash
   npx serve -p 3000
   ```

   Or using any web server of your choice.

3. **Access the application**: http://localhost:3000

## 📁 Project Structure

```
Debunker/
├── backend/                 # FastAPI backend
│   ├── main.py             # FastAPI application entry point
│   ├── fact_checker.py     # Core fact-checking logic
│   ├── models.py           # Pydantic models for API
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Backend container config
│   └── .env.example        # Environment variables template
├── frontend-standalone/     # Static frontend
│   ├── index.html          # Main HTML file
│   ├── script.js           # Frontend JavaScript
│   ├── styles.css          # CSS styles
│   ├── config.js           # Frontend configuration
│   ├── nginx.conf          # Nginx configuration
│   └── Dockerfile          # Frontend container config
├── docker-compose.yml       # Multi-service deployment
└── README.md               # This file
```

## 🔧 Configuration

### Backend Configuration

Key environment variables in `backend/.env`:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Optional
LOG_LEVEL=INFO
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
MAX_FILE_SIZE=52428800
```

### Frontend Configuration

The frontend automatically detects the environment and configures the backend URL:

- **Development**: `http://localhost:8000`
- **Production**: Configure in `config.js` or via the UI

## 🌐 API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/analyze` - Analyze text claim
- `POST /api/analyze-file` - Analyze with audio file
- `GET /api/stats` - Usage statistics

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔄 Development Workflow

### Backend Development

1. **Install in development mode**:
   ```bash
   cd backend
   pip install -e .
   ```

2. **Run with auto-reload**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/api/analyze" \
        -H "Content-Type: application/json" \
        -d '{"text_claim": "The Earth is flat", "enable_prosody": false}'
   ```

### Frontend Development

1. **Serve with live reload**:
   ```bash
   cd frontend-standalone
   npx live-server --port=3000
   ```

2. **Configure backend URL** in `config.js` if needed

## 🚢 Deployment

### Production Deployment with Docker

1. **Production compose file**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

2. **Environment-specific configuration**:
   ```bash
   # Use different .env files for different environments
   cp backend/.env.production backend/.env
   ```

### Cloud Deployment Options

#### Backend Deployment
- **Heroku**: Use `Procfile` with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
- **AWS ECS**: Use the provided Dockerfile
- **Google Cloud Run**: Deploy containerized backend
- **Railway/Render**: Direct deployment from repository

#### Frontend Deployment
- **Vercel/Netlify**: Deploy the `frontend-standalone` directory
- **AWS S3 + CloudFront**: Static site hosting
- **GitHub Pages**: For public repositories

### Reverse Proxy Setup

For production, use nginx or similar:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://frontend:80;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Considerations

### Backend Security
- API key management via environment variables
- CORS configuration for allowed origins
- Request size limits
- Rate limiting (implement as needed)
- Input validation and sanitization

### Frontend Security
- Content Security Policy headers
- XSS protection
- Secure communication with backend (HTTPS in production)

## 📊 Monitoring and Logging

### Backend Monitoring
- Health check endpoint: `/health`
- Structured logging with configurable levels
- Request/response logging
- Performance metrics (implement as needed)

### Error Handling
- Comprehensive error responses
- Client-friendly error messages
- Fallback mechanisms for service failures

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/
```

### Frontend Testing
```bash
cd frontend-standalone
# Use your preferred testing framework
```

### Integration Testing
```bash
# Test full stack with docker-compose
docker-compose up -d
# Run integration tests
```

## 🔧 Troubleshooting

### Common Issues

1. **Backend not starting**:
   - Check API keys in `.env` file
   - Verify port 8000 is available
   - Check logs: `docker-compose logs backend`

2. **Frontend can't connect to backend**:
   - Verify backend URL in frontend config
   - Check CORS settings in backend
   - Ensure both services are running

3. **Model loading issues**:
   - Increase Docker memory limits
   - Check available disk space
   - Verify internet connection for model downloads

### Debugging

```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper and GPT models
- Hugging Face for transformer models
- The open-source AI/ML community
- FastAPI and modern web development tools

---

**Note**: This application is for educational and research purposes. Always verify information through multiple reliable sources and use critical thinking when evaluating claims.