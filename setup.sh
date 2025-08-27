#!/bin/bash

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    set -o allexport
    source .env
    set +o allexport
else
    echo "No .env file found. Skipping environment variable export."
fi

REPO_DIR="lm-evaluation-harness"
METRICS_DIR="$(pwd)"

# Check if lm-evaluation-harness is already installed
if pip show lm-eval > /dev/null 2>&1; then
    echo "lm-evaluation-harness appears to be installed."
    read -p "Do you want to reinstall it? [y/N]: " reinstall
    if [[ "$reinstall" =~ ^[Yy]$ ]]; then
        cd "$REPO_DIR" || { echo "Failed to enter $REPO_DIR"; exit 1; }
        pip install -e .
        pip install -e .[vllm]
    else
        echo "Skipping reinstallation."
    fi
else
      # Clone the repo if it doesn't exist
    if [ ! -d "$REPO_DIR" ]; then
        git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    else
        echo "$REPO_DIR already exists."
    fi
    cd "$REPO_DIR" || { echo "Failed to enter $REPO_DIR"; exit 1; }
    pip install -e .
fi

# Add the metrics folder to PYTHONPATH if it exists
if [ -d "$METRICS_DIR" ]; then
    export PYTHONPATH="$METRICS_DIR:$PYTHONPATH"
    echo "Added $METRICS_DIR to PYTHONPATH."
else
    echo "Warning: metrics directory not found at $METRICS_DIR."
fi
