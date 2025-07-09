# SafeStreets - AI Model Quick Reference

This document provides a quick reference guide for the AI models used in the SafeStreets system, including usage, performance metrics, and implementation details.

## Overview of AI Components

The SafeStreets system uses four key AI components:

1. **CNN Road Classifier**: Validates images contain road surfaces
2. **Vision Transformer (ViT)**: Classifies road damage types
3. **YOLO Object Detector**: Locates and bounds damage areas
4. **Google Gemini**: Generates natural language damage reports

## Model Specifications

### CNN Road Classifier

**Purpose**: Filter out non-road images before processing

**Architecture**:
- Base: Convolutional Neural Network
- Input: 224x224 RGB images
- Output: Binary classification (road/non-road)
- File: `cnn_road_classifier.pth`

**Performance**:
- Accuracy: 92% on validation set
- Inference Time: ~0.1s per image

**Usage**:
```python
from src.models.road_classifier import RoadClassifier

classifier = RoadClassifier("models/cnn_road_classifier.pth")
is_road = classifier.predict(image_path)

if is_road:
    # Process for damage detection
else:
    # Reject image
```

### Vision Transformer (ViT)

**Purpose**: Classify road damage types and severity

**Architecture**:
- Base: Vision Transformer (ViT-Base-16)
- Input: 224x224 RGB images
- Output: 8 damage categories
- File: `vit_model.pth`

**Damage Classes**:
| Class Code | Damage Type | Description |
|------------|-------------|-------------|
| **D00** | Longitudinal Cracks | Linear cracks parallel to road direction |
| **D10** | Transverse Cracks | Linear cracks perpendicular to road direction |
| **D20** | Alligator Cracks | Interconnected cracks forming polygon patterns |
| **D30** | Potholes | Circular/oval depressions in road surface |
| **D40** | Line Cracks | General linear crack patterns |
| **D43** | Cross Walk Blur | Faded/damaged crosswalk markings |
| **D44** | Whiteline Blur | Faded/damaged lane markings |
| **D50** | Manhole Covers | Damaged or displaced utility covers |

**Performance**:
- Accuracy: 68%+ across all classes
- Precision: 73% average across classes
- Recall: 69% average across classes
- Inference Time: ~1.2s per image

**Usage**:
```python
from src.models.vit_classifier import ViTClassifier

classifier = ViTClassifier("models/vit_model.pth")
damage_type, confidence = classifier.predict(image_path)

if confidence > 0.85:
    # Process damage type
```

### YOLO Object Detector

**Purpose**: Detect and localize damage areas in road images

**Architecture**:
- Base: YOLOv5s (Small)
- Input: Variable-sized images
- Output: Bounding boxes with damage classes
- File: `yolo_model.pt`

**Performance**:
- mAP@0.5: 76%
- Inference Time: ~0.3s per image
- IOU Threshold: 0.45

**Usage**:
```python
from src.models.yolo_detector import YoloDetector

detector = YoloDetector("models/yolo_model.pt")
detections = detector.detect(image_path)

for detection in detections:
    # Process each detection (box, class, confidence)
    box, class_id, confidence = detection
```

### Google Gemini Integration

**Purpose**: Generate professional damage report descriptions

**Implementation**:
- Model: Gemini 1.5 Flash
- Input: Structured damage data (type, location, severity)
- Output: Natural language report description
- API Integration: REST API calls to Google Gemini

**Example Usage**:
```python
from src.utils.gemini_client import GeminiClient

client = GeminiClient(api_key="YOUR_API_KEY")

report_data = {
    "damage_type": "D30",
    "severity": "High",
    "location": "Main Street, Downtown",
    "dimensions": "0.5m x 0.8m",
    "surface_type": "Asphalt"
}

report_summary = client.generate_report(report_data)
```

## Model Pipeline Integration

The AI components work together in the following sequence:

1. **Image Reception**:
   - Image is received via API from mobile app

2. **Road Validation**:
   - CNN Road Classifier validates image shows a road surface
   - Non-road images are rejected with appropriate message

3. **Damage Detection**:
   - YOLO model detects and localizes damage areas
   - Bounding boxes are created around damage regions

4. **Damage Classification**:
   - ViT model classifies the type of damage
   - Confidence scores determine classification reliability

5. **Report Generation**:
   - Damage data is structured with GPS location, timestamp
   - Gemini API generates natural language description
   - Complete report is stored in database

## Performance Optimization

- **Batch Processing**: When multiple images are uploaded
- **Image Preprocessing**: Standardization, normalization
- **Caching**: Common responses are cached
- **Model Quantization**: For faster inference

## Error Handling

The AI pipeline implements robust error handling:

- **Invalid Images**: Proper rejection and error messages
- **Low Confidence**: Warning flags for manual review
- **API Failures**: Graceful degradation with fallbacks
- **Timeout Management**: Preventing hanging requests

## Model Updates and Training

### Updating Models

1. Place new model files in the `models/` directory
2. Update class labels in `class_names.txt` if needed
3. Restart the AI Models Server

### Training New Models

To train updated models with new data:

1. Use notebooks in `notebooks/` directory
2. Follow training instructions in respective notebooks
3. Export trained models in appropriate format
4. Update the models in the deployment

## Monitoring AI Performance

The system logs key metrics for AI model performance:

- Inference time per image
- Classification accuracy
- Rejection rates
- API call success/failure rates

## Troubleshooting

### Common Issues:

1. **"Invalid tensor dimensions" error**:
   - Check image dimensions match model input requirements

2. **"CUDA out of memory" error**:
   - Reduce batch size or switch to CPU inference

3. **"Model file not found" error**:
   - Verify model files are correctly placed in `models/` directory

4. **Low classification confidence**:
   - Review training data quality
   - Consider fine-tuning with more examples

For more details on the AI models, refer to the comprehensive [AI Model Documentation](./ai-model-documentation.md).
