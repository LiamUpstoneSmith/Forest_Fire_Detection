#!/bin/bash
# run.sh - Start FastAPI server with uvicorn

# Exit if any command fails
set -e

# Go to project root (incase script is run from elsewhere)
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "Could not find virtual environment in .venv/"
    exit 1
fi

# Run FastAPI with hot reload
echo "Starting FastAPI server at http://127.0.0.1:8000"
uvicorn src.app:app --reload
