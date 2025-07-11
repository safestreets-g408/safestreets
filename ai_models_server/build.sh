#!/bin/bash

# Exit on error
set -o errexit

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download model files if needed
if [ ! -d "models" ]; then
    mkdir -p models
fi

# You may need to add logic here to download your model files
# For example:
# wget -O models/yolov5s.pt https://path-to-your-model/yolov5s.pt

# Create directories for uploads and results
mkdir -p static/uploads
mkdir -p static/results

echo "Build completed successfully!"
