#!/bin/bash

echo "🚀 Starting Debunker Backend..."

# Set environment variables from secrets
export PYTHONPATH="/workspaces/Debunker/backend"

# Navigate to backend directory
cd /workspaces/Debunker/backend

# Start the application
python main.py