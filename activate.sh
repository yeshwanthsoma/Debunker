#!/bin/bash
# Activation script for Debunker project

echo "🕵️ Activating Debunker virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🐍 Python version: $(python --version)"
echo "📦 Installed packages:"
echo "  - FastAPI: $(pip show fastapi | grep Version | cut -d' ' -f2)"
echo "  - Transformers: $(pip show transformers | grep Version | cut -d' ' -f2)"
echo "  - LangChain: $(pip show langchain | grep Version | cut -d' ' -f2)"
echo "  - PyTorch: $(pip show torch | grep Version | cut -d' ' -f2)"

echo ""
echo "🚀 Ready to run:"
echo "  Backend: cd backend && python main.py"
echo "  Or with uvicorn: cd backend && uvicorn main:app --reload"
echo ""
echo "📋 Don't forget to set your API keys in backend/.env"