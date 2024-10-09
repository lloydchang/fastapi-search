#!/bin/sh

# File: run_fastapi-search.sh

set -euo pipefail

cd $(dirname $0)

BASE_DIR=$(pwd)

# Start port range for FastAPI
START_PORT=8000
MAX_PORT=65535  # Set to the maximum allowable port number

echo "Starting FastAPI server..."

# Function to find the next available port starting from START_PORT
find_available_port() {
  port=$START_PORT
  while [ $port -le $MAX_PORT ]; do
    if ! lsof -i :$port > /dev/null; then
      echo $port
      return
    fi
    port=$((port + 1))
  done

  echo "No available ports found in the range $START_PORT to $MAX_PORT."
  exit 1
}

# Get the first available port starting from START_PORT
AVAILABLE_PORT=$(find_available_port)
echo "Using available port: $AVAILABLE_PORT"

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
    pip install --upgrade pip > /dev/null
    pip install -r requirements.txt > /dev/null
else
    echo 'requirements.txt not found. Please ensure it exists.'
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Start Uvicorn server on the available port
echo "Running Uvicorn server on localhost:$AVAILABLE_PORT..."
uvicorn api.index:app --host localhost --port $AVAILABLE_PORT --reload
