#!/bin/bash

# AI Models Server Startup Script
echo "🚀 Starting AI Models Server..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if models exist
echo "🔍 Checking for model files..."
missing_models=()

if [ ! -f "models/vit_model.pth" ]; then
    missing_models+=("vit_model.pth")
fi

if [ ! -f "models/yolo_model.pt" ]; then
    missing_models+=("yolo_model.pt")
fi

if [ ! -f "models/cnn_road_classifier_scripted.pt" ]; then
    missing_models+=("cnn_road_classifier_scripted.pt")
fi

if [ ! -f "models/class_names.txt" ]; then
    missing_models+=("class_names.txt")
fi

if [ ${#missing_models[@]} -gt 0 ]; then
    echo "⚠️  Missing model files:"
    for model in "${missing_models[@]}"; do
        echo "   - $model"
    done
    echo ""
    echo "💡 Run './scripts/download_models.sh' to download missing models"
    echo "   The server will start in fallback mode"
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📝 No .env file found. You may want to create one:"
    echo "   cp .env.example .env"
    echo "   # Edit .env with your configuration"
    echo ""
fi

# Start the server
echo "🌟 Starting server..."
echo "=================================="
python app.py
