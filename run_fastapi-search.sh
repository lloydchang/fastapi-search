#!/bin/sh -x

# File: run_fastapi-serach.sh

set -euo pipefail

cd $(dirname $0)

BASE_DIR=$(pwd)

echo 'Starting FastAPI server...'

# Activate virtual environment or create one
if [ -d "venv" ]; then
    echo 'Activating existing virtual environment...'
    source venv/bin/activate
else
    echo 'Creating virtual environment...'
    python -m venv venv
    source venv/bin/activate
fi

# Install runtime requirements
if [ -f "requirements.txt" ]; then
    echo 'Installing runtime requirements...'
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo 'requirements.txt not found. Please ensure it exists.'
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Start Uvicorn server
uvicorn api.index:app --host localhost --port 8000 --reload
