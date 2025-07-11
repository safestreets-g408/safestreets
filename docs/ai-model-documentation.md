# AI Model Documentation

This document provides detailed information about the AI models used in the SafeStreets system for road damage detection and classification.

## Model Overview

SafeStreets employs multiple AI models working together to detect, classify, and analyze road damage. The system uses a combination of computer vision models and generative AI to provide comprehensive road damage assessment capabilities.

### Core AI Components

1. **Vision Transformer (ViT)** - Road damage classification
2. **YOLO Object Detector** - Damage area detection and localization
3. **CNN Road Classifier** - Road surface validation
4. **Google Gemini** - Natural language report generation

These components work together in a pipeline to process images, detect road damage, and generate detailed reports.

## Vision Transformer (ViT) Model

The Vision Transformer (ViT) model is used for classifying road damage types with high accuracy.

### Architecture

- **Base Model**: Vision Transformer (ViT-Base-16)
- **Input Resolution**: 224x224 pixels
- **Patch Size**: 16x16 pixels
- **Hidden Dimension**: 768
- **Implementation**: Supports both HuggingFace Transformers and PyTorch implementations
- **Output Classes**: Multiple road damage categories
- **File**: `vit_model.pt`

### Damage Classification System

The model classifies road damage into the following categories:

| Class Code | Damage Type | Description | Severity |
|------------|-------------|-------------|----------|
| **D00** | Longitudinal Cracks | Linear cracks parallel to road direction | Low to Medium |
| **D10** | Transverse Cracks | Linear cracks perpendicular to road direction | Low to Medium |
| **D20** | Alligator Cracks | Interconnected cracks forming polygon patterns | Medium to High |
| **D30** | Potholes | Circular/oval depressions in road surface | High to Critical |
| **D40** | Line Cracks | General linear crack patterns | Low to Medium |
| **D43** | Cross Walk Blur | Faded/damaged crosswalk markings | Low to Medium |
| **D44** | Whiteline Blur | Faded/damaged lane markings | Low to Medium |
| **D50** | Manhole Covers | Damaged or displaced utility covers | Medium to High |

## YOLO Object Detector

The system uses YOLO (You Only Look Once) object detection models to localize damage areas in road images.

### Architecture

- **Primary Implementation**: YOLOv8 with fallback to YOLOv5
- **Input**: Variable-sized images
- **Output**: Bounding boxes with damage classes and confidence scores
- **File**: `yolo_model.pt`
- **Features**:
  - Real-time detection capabilities
  - Precise bounding box coordinates
  - Multi-class damage detection
  - Confidence scoring

### Performance

- **Inference Time**: ~0.3s per image (on server hardware)
- **Default Confidence Threshold**: Configurable, typically 0.45
- **IOU Threshold**: 0.45

## CNN Road Classifier

A simple but effective CNN model is used to validate if uploaded images actually contain road surfaces, preventing irrelevant image processing.

### Architecture

- **Type**: Convolutional Neural Network
- **Input Size**: 128x128 pixels
- **Structure**:
  - Convolutional layers with ReLU activation
  - Max pooling layers
  - Fully connected output layer
  - Sigmoid activation for binary classification
- **Output**: Binary classification (road/non-road)
- **File**: `cnn_road_classifier.pth`

### Performance

- **Inference Time**: ~0.1s per image
- **Model Size**: Lightweight (<10MB)

## Google Gemini Integration

The SafeStreets system integrates Google's Gemini AI to generate natural language descriptions of detected road damage.

### Implementation

- **API**: Google Generative AI API (Gemini 1.5 Flash)
- **Input**: Structured damage data and optional image
- **Output**: Natural language damage descriptions and recommendations
- **Features**:
  - Generates professional damage reports
  - Includes severity assessments
  - Provides repair recommendations
  - Estimates priority levels

### Configuration

- Requires Google API key
- Includes fallback mechanisms when API is unavailable

## Model Server Implementation

The AI models are served via a Flask-based REST API with a robust architecture:

### Server Structure
- **Framework**: Flask with clean architecture
- **Model Management**: Centralized model manager for all AI components
- **Error Handling**: Comprehensive error handling with fallback mechanisms

### Processing Pipeline

1. **Image Reception**: Base64 encoded images via REST API
2. **Road Validation**: CNN Road Classifier validates road surface presence
3. **Damage Detection**: YOLO model detects and localizes damage areas
4. **Damage Classification**: ViT model classifies damage types
5. **Report Generation**: Gemini AI generates natural language descriptions
6. **Response**: Returns structured results with annotated images

### API Endpoints

```
# Health check endpoint
GET /health

# YOLO detection endpoint
POST /detect-yolo
{
  "image": "base64_encoded_image_string"
}

# Road classification endpoint
POST /validate-road
{
  "image": "base64_encoded_image_string"
}

# ViT classification endpoint
POST /predict
{
  "image": "base64_encoded_image_string"
}

# Report generation endpoint
POST /generate-report
{
  "damage_type": "D30",
  "confidence": 0.95,
  "image": "base64_encoded_image_string",
  "additional_context": {...}
}
```

## Model Deployment

### Hardware Requirements

#### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 1GB available space
- GPU: Not required, but recommended

#### Recommended Configuration
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 10GB+ SSD
- GPU: NVIDIA with 8GB+ VRAM
- CUDA: 11.0+

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- Flask 2.0+
- Pillow 9.0+
- NumPy 1.20+
- OpenCV 4.5+
- Ultralytics (for YOLOv8)

## Monitoring and Maintenance

### Performance Monitoring
The model server includes monitoring capabilities:
- Inference time tracking
- Error rate logging
- Confidence score distribution
- Resource utilization metrics

### Model Versioning
Model weights are versioned to enable rollbacks if needed.

### Fallback Mechanisms
The system implements robust fallback mechanisms:
- Model loading failure recovery
- Graceful degradation when models are unavailable
- Alternative prediction pathways when primary models fail

## Future Improvements

### Planned Enhancements
- **Severity Estimation**: More granular damage severity assessment
- **Instance Segmentation**: Pixel-level damage area identification
- **Multi-damage Detection**: Identifying multiple damage types in a single image
- **Edge Deployment**: Optimized models for on-device inference
- **Ensemble Methods**: Multiple model voting for improved accuracy

### Reference Materials

- **Notebooks**: Available in the `notebooks/` directory:
  - `image-classification-vit-pytorch.ipynb`
  - `road_detection_cnn.ipynb`
  - `vit_pytorch.ipynb`
  - `report_desc_generation_gemini.ipynb`
