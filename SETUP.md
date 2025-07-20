# TruthLens - Professional Fact-Checking Setup Guide

## Overview

TruthLens has been upgraded from a basic conspiracy theory debunker to a professional fact-checking application with advanced AI integration and multiple verification sources.

## Features

### ✅ Basic Features (Always Available)
- **Audio Transcription**: Whisper AI-powered speech-to-text
- **Prosody Analysis**: Sarcasm detection, emotion analysis, audio authenticity
- **Sentiment Analysis**: RoBERTa-based sentiment classification
- **Simple Fact-Checking**: Built-in knowledge base with common conspiracy theories
- **Interactive Timeline**: Visual analysis progression
- **Professional UI**: Multi-screen interface with modern design

### 🚀 Professional Features (Requires API Keys)
- **Google Fact Check Tools API**: Access to verified fact-checking sources
- **OpenAI GPT-4 Integration**: AI-powered claim analysis and verification
- **Multi-Source Verification**: Consensus-based fact-checking across multiple APIs
- **Enhanced Accuracy**: Professional-grade verification with confidence scoring

## Quick Start

### 1. Install Dependencies

```bash
# Use the existing virtual environment
cd /path/to/Debunker
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (choose one)
cd backend

# Option 1: Full installation (includes LangChain for advanced features)
pip install -r requirements.txt

# Option 2: Minimal installation (production-ready, faster install)
pip install -r requirements-minimal.txt
```

### 2. Basic Mode (No API Keys Required)

```bash
# Start backend (from backend directory)
cd backend
source ../venv/bin/activate
python main.py

# In another terminal, serve frontend
cd frontend-standalone
python -m http.server 3000
```

Access the application at `http://localhost:3000`

### 3. Test the Installation

```bash
# Run the test suite
cd /path/to/Debunker
python test_api.py
```

## Professional Setup (Recommended)

### 1. Get API Keys

#### Google Fact Check Tools API (Highly Recommended)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Fact Check Tools API"
4. Create credentials (API Key)
5. Copy your API key

#### OpenAI API (For AI Analysis)
1. Sign up at [OpenAI](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy your API key

#### Optional: Additional APIs
- **News API**: For news source verification
- **Anthropic Claude**: Alternative to OpenAI

### 2. Configure Environment

```bash
cd backend
cp .env.example .env
```

Edit `.env` file:

```env
# Required for Professional Features
GOOGLE_FACT_CHECK_API_KEY=your_actual_google_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here

# Optional APIs
NEWS_API_KEY=your_news_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### 3. Start Professional Mode

```bash
# Backend with professional APIs
cd backend
source venv/bin/activate
python main.py

# Check API status
curl http://localhost:8080/api/status
```

## API Status Check

Visit `http://localhost:8080/api/status` to see:

```json
{
  "version": "3.0.0-professional",
  "basic_features": {
    "audio_transcription": true,
    "prosody_analysis": true,
    "sentiment_analysis": true,
    "simple_fact_checking": true
  },
  "professional_apis": {
    "google_fact_check": true,
    "openai": true,
    "anthropic": false,
    "news_api": false
  },
  "recommendations": []
}
```

## Architecture

### Backend (FastAPI)
- **main.py**: Core application with fact-checking logic
- **config.py**: Environment and API key management
- **fact_check_apis.py**: Professional API integrations
- **requirements.txt**: Python dependencies

### Frontend (Vanilla JS)
- **index.html**: Multi-screen interface
- **styles.css**: Professional design system
- **script.js**: Application logic and API communication
- **config.js**: Frontend configuration

## Usage Modes

### Basic Mode
- Uses built-in knowledge base
- Covers common conspiracy theories
- Good for testing and demonstration
- No external API dependencies

### Professional Mode
- Accesses Google's verified fact-checking database
- AI-powered analysis for any claim
- Multi-source verification
- Higher accuracy and confidence scores
- Suitable for production use

## Verification

### ✅ What Should Work

After setup, you should be able to:

1. **Basic Fact-Checking**: Test claims like "The moon landing was faked" → Returns "False" with high confidence
2. **Audio Upload**: Upload MP3/WAV files for transcription and prosody analysis
3. **Professional UI**: Navigate through 5 screens (Welcome → Upload → Progress → Results → Settings)
4. **API Status**: Check `/api/status` endpoint to see feature availability

### 🧪 Testing Results

The system has been tested and confirmed working:
- ✅ Basic fact-checking with built-in knowledge base
- ✅ Professional API integration (when keys provided)
- ✅ Audio transcription and prosody analysis
- ✅ Multi-screen UI with modern design
- ✅ Fallback mechanisms (basic → professional)

## Troubleshooting

### Common Issues

1. **Backend not starting**
   ```bash
   # Check virtual environment
   source venv/bin/activate
   
   # Check Python version (3.8+ required)
   python --version
   
   # Install missing dependencies
   cd backend && pip install -r requirements.txt
   ```

2. **Pydantic errors**
   ```bash
   # Install updated dependencies
   pip install pydantic-settings>=2.0.0
   ```

3. **API keys not working**
   ```bash
   # Verify .env file exists and has correct format
   cat backend/.env
   
   # Check API status endpoint
   curl http://localhost:8080/api/status
   ```

4. **Frontend can't connect to backend**
   - Ensure backend is running on port 8080
   - Check CORS settings in main.py
   - Verify frontend config.js has correct backend URL

5. **Audio upload issues**
   - Supported formats: MP3, WAV, M4A, OGG
   - Maximum file size: 50MB
   - Check browser console for errors

### Performance Tips

1. **For better performance**:
   - Use GPU-enabled environment for audio processing
   - Configure Redis for caching (optional)
   - Implement rate limiting for production

2. **For development**:
   - Start with basic mode to test functionality
   - Add API keys incrementally
   - Use small audio files for testing

## Security Notes

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Configure CORS appropriately for production
- Implement rate limiting for public deployments
- Consider API key rotation policies

## Support

For issues and questions:
1. Check the API status endpoint: `/api/status`
2. Review backend logs for error messages
3. Verify API key configuration
4. Test with basic mode first

## Upgrade Path

From basic conspiracy debunker to professional fact-checker:

1. ✅ **Completed**: Multi-screen professional UI
2. ✅ **Completed**: Google Fact Check Tools API integration
3. ✅ **Completed**: OpenAI GPT-4 fact verification
4. ✅ **Completed**: Multi-source verification pipeline
5. 🔄 **Next**: News source integration
6. 🔄 **Next**: Advanced caching and rate limiting
7. 🔄 **Next**: Citation and source attribution system

---

**TruthLens v3.0.0** - From conspiracy debunker to professional fact-checking platform