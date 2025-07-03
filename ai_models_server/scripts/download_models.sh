#!/bin/bash

# Script to check and prepare model files for SafeStreets AI server
# This script ensures that all required models are in the correct location

echo "Checking model files for SafeStreets AI server..."

# Function to check if a model exists
check_model() {
  local model_name="$1"
  local model_path="$2"
  
  if [ -f "$model_path" ]; then
    echo "✓ $model_name found at $model_path"
    return 0
  else
    echo "✗ $model_name not found at $model_path"
    return 1
  fi
}

# Define the required models and their locations
VIT_MODEL_PATH="./vit_model.pth"
YOLO_MODEL_PATH="./yolo_model.pt"
ROAD_MODEL_PATH="./cnn_road_classifier_scripted.pt"

# Check Vision Transformer model
check_model "ViT model" "$VIT_MODEL_PATH"
vit_exists=$?

# Check YOLO model
check_model "YOLO model" "$YOLO_MODEL_PATH"
yolo_exists=$?

# Check Road Classifier model
check_model "Road classifier model" "$ROAD_MODEL_PATH"
road_exists=$?

echo ""
if [ $vit_exists -eq 0 ] && [ $yolo_exists -eq 0 ] && [ $road_exists -eq 0 ]; then
  echo "All required models are present."
  echo "You can start the AI server with 'python app.py'"
else
  echo "Some models are missing. The server will use fallback modes for missing models."
  echo "For best performance, ensure all model files are present."
  
  if [ $vit_exists -ne 0 ]; then
    echo "- ViT model missing: Place 'vit_model.pth' in the current directory."
  fi
  
  if [ $yolo_exists -ne 0 ]; then
    echo "- YOLO model missing: Place 'yolo_model.pt' in the current directory."
  fi
  
  if [ $road_exists -ne 0 ]; then
    echo "- Road classifier model missing: Place 'cnn_road_classifier_scripted.pt' in the current directory."
  fi
fi
echo ""
