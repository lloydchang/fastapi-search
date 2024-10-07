#!/bin/sh -x

# Exit immediately if a command exits with a non-zero status
set -e

# Set the base directory to the script's parent directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project's root directory
cd "$BASE_DIR"

echo "Starting precompute process..."

# Check if requirements_precompute.txt exists
if [ ! -f "requirements_precompute.txt" ]; then
  echo "requirements_precompute.txt not found!"
  exit 1
fi

# Install precompute requirements
echo "Installing precompute requirements..."
pip install --upgrade pip
pip install -r requirements_precompute.txt

# Run the precompute script
echo "Running precompute script..."
python backend/fastapi/utils/precompute_cache.py

echo "Precompute step completed successfully!"
