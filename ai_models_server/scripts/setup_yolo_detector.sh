#!/bin/bash
# Script to install dependencies for YOLOv8 road damage detection

echo "📦 Installing dependencies for YOLOv8 road damage detection..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install Python and pip first."
    exit 1
fi

# Install required packages
echo "Installing ultralytics, torch, opencv-python, and pillow..."
pip install ultralytics torch torchvision opencv-python pillow

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Error installing dependencies."
    exit 1
fi

# Set up model directory
MODEL_DIR="models"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating models directory..."
    mkdir -p "$MODEL_DIR"
fi

# Check for model file
MODEL_PATH="$MODEL_DIR/yolo_model.pt"
if [ -f "$MODEL_PATH" ]; then
    echo "✅ YOLO model found at $MODEL_PATH"
else
    echo "⚠️ YOLO model not found at $MODEL_PATH"
    echo "Please make sure to place your trained model at this location."
fi

# Check for class names file
CLASS_NAMES_PATH="$MODEL_DIR/class_names.txt"
if [ -f "$CLASS_NAMES_PATH" ]; then
    echo "✅ Class names file found at $CLASS_NAMES_PATH"
else
    echo "⚠️ Class names file not found at $CLASS_NAMES_PATH"
    echo "Creating a default class names file..."
    echo -e "D00\nD01\nD0w0\nD10\nD11\nD20\nD40\nD43\nD44\nD50" > "$CLASS_NAMES_PATH"
    echo "✅ Default class names file created."
fi

echo ""
echo "🚀 Setup complete! You can now run the test script:"
echo "./scripts/test_yolo_detector.py"
