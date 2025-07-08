#!/bin/bash
# Script to prepare the CNN road classifier model

set -e

echo "🔄 Setting up CNN road classifier model..."

# Set paths
MODEL_DIR="models"
MODEL_PATH="$MODEL_DIR/cnn_road_classifier.pth"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if the model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ CNN road classifier model already exists at $MODEL_PATH"
else
    echo "⚠️ CNN road classifier model not found at $MODEL_PATH"
    echo "Please ensure you have the model file available"
    
    # Check if there's a scripted version and copy it
    if [ -f "$MODEL_DIR/cnn_road_classifier_scripted.pt" ]; then
        echo "Found scripted model, copying to correct location..."
        cp "$MODEL_DIR/cnn_road_classifier_scripted.pt" "$MODEL_PATH"
        echo "✅ Copied scripted model to $MODEL_PATH"
    else
        echo "❌ No model file found. Please manually place cnn_road_classifier.pth in the models directory."
        exit 1
    fi
fi

echo "✅ CNN road classifier setup complete!"
