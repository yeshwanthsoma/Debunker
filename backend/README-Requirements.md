# TruthLens Requirements Guide

## Requirements Files Overview

### 📦 requirements.txt (Full Installation)
**Use for:** Development and full feature set
**Size:** ~70 packages
**Features:** 
- All TruthLens functionality
- LangChain integration (for future extensions)
- Complete development environment

```bash
pip install -r requirements.txt
```

### 🚀 requirements-minimal.txt (Production)
**Use for:** Production deployment and faster setup
**Size:** ~40 packages (43% smaller)
**Features:**
- Core fact-checking functionality
- Google Fact Check Tools API
- OpenAI integration
- Audio processing
- Professional UI support

```bash
pip install -r requirements-minimal.txt
```

### 📋 requirements-core.txt (Legacy)
**Status:** Deprecated - use minimal instead

## Tested Versions

The requirements files include exact versions that have been tested and confirmed working:

- **Python:** 3.12.11 (minimum 3.8+)
- **FastAPI:** 0.104.1
- **PyTorch:** 2.7.1
- **Transformers:** 4.53.2
- **OpenAI:** 1.97.0
- **Pydantic:** 2.11.7

## Installation Recommendations

### For New Users
```bash
pip install -r requirements-minimal.txt
```

### For Developers
```bash
pip install -r requirements.txt
```

### For Production
```bash
pip install -r requirements-minimal.txt
```

## Troubleshooting

### Common Issues

1. **PyTorch Installation**
   - If you get CUDA errors, PyTorch will fallback to CPU
   - For Apple Silicon Macs, MPS acceleration is automatically used

2. **Audio Dependencies**
   - `soundfile` requires system audio libraries
   - On Ubuntu: `sudo apt-get install libsndfile1`
   - On macOS: Usually works out of the box

3. **Large Downloads**
   - First run downloads ML models (~1-2GB)
   - Subsequent runs use cached models

4. **Memory Requirements**
   - Minimum: 4GB RAM
   - Recommended: 8GB+ RAM for smooth operation

## Dependency Explanation

### Core Dependencies
- **fastapi**: Web framework for API
- **transformers**: Hugging Face models (Whisper, sentiment analysis)
- **sentence-transformers**: Text embeddings
- **librosa**: Audio processing and analysis
- **openai**: GPT integration
- **aiohttp**: Async HTTP client for fact-checking APIs

### Optional Dependencies (Full Install Only)
- **langchain**: Advanced AI workflows (future features)
- **faiss-cpu**: Vector search (enhanced similarity matching)
- **datasets**: ML dataset utilities

Both requirement files ensure TruthLens works perfectly - choose based on your use case!