#!/bin/sh

# File: run_precompute.sh

set -euo pipefail

cd $(dirname $0)

BASE_DIR=$(pwd)

echo 'Starting precompute process...'

# Check if requirements_precompute.txt exists
if [ ! -f "requirements_precompute.txt" ]; then
    echo 'requirements_precompute.txt not found. Please ensure it exists.'
    exit 1
fi

# Activate virtual environment or create one
if [ -d "venv" ]; then
    echo 'Activating existing virtual environment...'
    source venv/bin/activate
else
    echo 'Creating virtual environment...'
    python -m venv venv
    source venv/bin/activate
fi

# Install precompute requirements
if [ -f "requirements_precompute.txt" ]; then
    echo 'Installing precompute requirements...'
    pip install --upgrade pip 2>&1 > /dev/null
    pip install -r requirements_precompute.txt 2>&1 > /dev/null
else
    echo 'requirements_precompute.txt not found. Please ensure it exists.'
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Run the precompute_cache.py. script
echo 'Running precompute script...'
python backend/fastapi/utils/precompute_cache.py

echo 'Precompute step completed successfully!'
