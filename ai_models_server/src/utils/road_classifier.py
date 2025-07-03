"""
Road Classifier Utilities
Handles road surface validation and classification
"""
import torch
import torch.nn as nn
import base64
import io
import os
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Union, Tuple
from torchvision import transforms

from ..core.config import CNN_ROAD_CLASSIFIER_PATH
from .model_compatibility import RobustModelLoader


class CNNRoadClassifier(nn.Module):
    """CNN model for road surface classification"""
    
    def __init__(self):
        super(CNNRoadClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of flattened features dynamically
        self.feature_size = None
        self.classifier = None
        
    def _initialize_classifier(self, x):
        """Initialize classifier based on actual feature size"""
        if self.classifier is None:
            # Get the actual feature size after convolutions
            with torch.no_grad():
                features = self.features(x)
                self.feature_size = features.view(features.size(0), -1).size(1)
                
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)  # road vs non-road
            )
        
    def forward(self, x):
        # Initialize classifier if needed
        if self.classifier is None:
            self._initialize_classifier(x)
            
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Global model variables
ROAD_CLASSIFIER = None
WORKING_INPUT_SIZE = (224, 224)  # Cache the working input size
ROAD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_road_classifier() -> bool:
    """Load the road classifier model"""
    global ROAD_CLASSIFIER
    
    try:
        print("Loading road classifier...")
        start_time = time.time()
        
        # Create fallback model
        fallback_model = CNNRoadClassifier()
        
        # Try to load pre-trained weights
        if CNN_ROAD_CLASSIFIER_PATH and os.path.exists(CNN_ROAD_CLASSIFIER_PATH):
            ROAD_CLASSIFIER, is_fallback = RobustModelLoader.load_with_fallback(
                CNN_ROAD_CLASSIFIER_PATH,
                fallback_model,
                model_type="cnn"
            )
            
            if ROAD_CLASSIFIER:
                ROAD_CLASSIFIER.eval()
                status = "fallback" if is_fallback else "loaded"
                print(f"Road classifier {status} successfully in {time.time() - start_time:.2f} seconds")
                return True
            else:
                print("Could not load any road classifier variant")
                # Use heuristic fallback
                ROAD_CLASSIFIER = None
                return True
        else:
            print("No model path provided or file not found, using heuristic-based road classification")
            ROAD_CLASSIFIER = None
            return True
            
    except Exception as e:
        print(f"Error loading road classifier: {e}")
        print("Falling back to heuristic-based road classification")
        ROAD_CLASSIFIER = None
        return True
            
    except Exception as e:
        print(f"Failed to load road classifier: {e}")
        ROAD_CLASSIFIER = None
        return True  # Still return True to use fallback
        return False


def validate_road_image(image_input: Union[str, Image.Image]) -> Dict[str, Any]:
    """
    Validate if an image contains a road surface
    
    Args:
        image_input: Can be a file path, PIL Image, or base64 string
        
    Returns:
        Dictionary containing validation results
    """
    try:
        # Handle different input types
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or len(image_input) > 100:
                # Assume base64 string
                clean_base64 = image_input.split(',')[1] if ',' in image_input else image_input
                img_bytes = base64.b64decode(clean_base64)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                # Assume file path
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            # Try to convert to PIL Image
            image = Image.fromarray(image_input).convert('RGB')
        
        # Always run heuristic analysis
        road_features = analyze_road_features(image)
        
        # Try model inference if available
        model_prediction = None
        model_confidence = 0.5
        
        if ROAD_CLASSIFIER is not None:
            try:
                # Use cached working input size if available
                global WORKING_INPUT_SIZE, ROAD_TRANSFORM
                
                # Create transform with the working input size
                if WORKING_INPUT_SIZE != (224, 224):
                    transform = transforms.Compose([
                        transforms.Resize(WORKING_INPUT_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(image).unsqueeze(0)
                else:
                    input_tensor = ROAD_TRANSFORM(image).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    outputs = ROAD_CLASSIFIER(input_tensor)
                    
                    # Handle different model output formats
                    if hasattr(outputs, 'shape') and len(outputs.shape) > 1:
                        # Regular model output with logits
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        model_prediction = predicted.item() == 1
                        model_confidence = confidence.item()
                    elif hasattr(outputs, 'item'):
                        # TorchScript or scalar output - assume sigmoid output
                        prob = torch.sigmoid(outputs).item()
                        model_prediction = prob > 0.5
                        model_confidence = prob if prob > 0.5 else (1.0 - prob)
                    else:
                        # Fallback for unknown output format
                        model_prediction = None
                        model_confidence = 0.5
                        
            except RuntimeError as shape_error:
                if "shape" in str(shape_error).lower():
                    print(f"Model shape mismatch: {shape_error}")
                    print("Trying with different input size...")
                    # Try different input sizes
                    for size in [(256, 256), (128, 128), (64, 64)]:
                        try:
                            alt_transform = transforms.Compose([
                                transforms.Resize(size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            alt_input = alt_transform(image).unsqueeze(0)
                            with torch.no_grad():
                                outputs = ROAD_CLASSIFIER(alt_input)
                                prob = torch.sigmoid(outputs).item() if hasattr(outputs, 'item') else 0.5
                                model_prediction = prob > 0.5
                                model_confidence = prob if prob > 0.5 else (1.0 - prob)
                                print(f"Success with input size {size}")
                                # Cache the working size
                                WORKING_INPUT_SIZE = size
                                # Update global transform
                                ROAD_TRANSFORM = alt_transform
                                break
                        except Exception:
                            continue
                    if model_prediction is None:
                        print("All input sizes failed, using heuristic only")
                        model_prediction = None
                        model_confidence = 0.5
                else:
                    raise shape_error
            except Exception as model_error:
                print(f"Error in road validation: {model_error}")
                # Fall back to heuristic only
                model_prediction = None
                model_confidence = 0.5
        
        # Combine model prediction with heuristics
        if model_prediction is not None:
            # Use weighted combination
            final_confidence = (model_confidence * 0.7) + (road_features['road_score'] * 0.3)
            final_is_road = final_confidence > 0.6
        else:
            # Use heuristic only
            final_confidence = road_features['road_score']
            final_is_road = final_confidence > 0.5
        
        # Ensure proper Python types for JSON serialization
        return {
            'is_road': bool(final_is_road),
            'confidence': float(final_confidence),
            'model_prediction': bool(model_prediction) if model_prediction is not None else None,
            'model_confidence': float(model_confidence),
            'heuristic_features': road_features,
            'fallback': model_prediction is None
        }
            
    except Exception as e:
        print(f"Error in road validation: {e}")
        return {
            'error': str(e),
            'is_road': True,  # Default to True for fallback
            'confidence': 0.5,
            'fallback': True
        }


def analyze_road_features(image: Image.Image) -> Dict[str, Any]:
    """
    Analyze image features to determine if it contains a road
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing feature analysis results
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic color analysis
        gray = np.mean(img_array, axis=2)
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Check for typical road colors (grays, darker tones)
        road_color_range = (mean_brightness > 30 and mean_brightness < 150)
        
        # Check for linear features (simplified edge detection)
        edges = detect_edges_simple(gray)
        edge_density = np.mean(edges)
        
        # Check for uniform texture (roads tend to have some uniformity)
        texture_score = 1.0 - (brightness_std / 255.0)
        
        # Calculate road score based on features
        road_score = 0.0
        
        if road_color_range:
            road_score += 0.3
        
        if edge_density > 0.1:  # Has some linear features
            road_score += 0.3
        
        if texture_score > 0.5:  # Reasonable texture uniformity
            road_score += 0.2
        
        # Aspect ratio check (roads are often horizontal)
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        if aspect_ratio > 1.2:  # Wider than tall
            road_score += 0.2
        
        return {
            'road_score': min(road_score, 1.0),
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'edge_density': edge_density,
            'texture_score': texture_score,
            'aspect_ratio': aspect_ratio,
            'road_color_range': road_color_range
        }
        
    except Exception as e:
        print(f"Error analyzing road features: {e}")
        return {
            'road_score': 0.5,
            'error': str(e)
        }


def detect_edges_simple(gray_image: np.ndarray) -> np.ndarray:
    """
    Simple edge detection using gradient
    
    Args:
        gray_image: Grayscale image as numpy array
        
    Returns:
        Edge map as numpy array
    """
    try:
        # Simple gradient-based edge detection
        grad_x = np.abs(np.gradient(gray_image, axis=1))
        grad_y = np.abs(np.gradient(gray_image, axis=0))
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1 range
        edges = edges / np.max(edges) if np.max(edges) > 0 else edges
        
        # Threshold to get binary edges
        edges = (edges > 0.1).astype(float)
        
        return edges
        
    except Exception as e:
        print(f"Error in edge detection: {e}")
        return np.zeros_like(gray_image)


def get_classifier_info() -> Dict[str, Any]:
    """Get information about the loaded road classifier"""
    if ROAD_CLASSIFIER is None:
        return {
            'loaded': False,
            'model_type': None,
            'classes': ['non-road', 'road']
        }
    
    try:
        return {
            'loaded': True,
            'model_type': 'CNN Road Classifier',
            'classes': ['non-road', 'road'],
            'input_size': (224, 224),
            'has_pretrained_weights': CNN_ROAD_CLASSIFIER_PATH is not None
        }
    except Exception as e:
        return {
            'loaded': False,
            'error': str(e),
            'classes': ['non-road', 'road']
        }


# Initialize model on module import
load_road_classifier()
