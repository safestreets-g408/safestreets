"""
Simple CNN Road Classifier Implementation
Provides the updated CNN model for road classification
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, Union, Tuple
from torchvision import transforms
from ..core.config import CNN_ROAD_CLASSIFIER_PATH

# Update to use the .pth file directly
ROAD_MODEL_PATH = os.path.join(os.path.dirname(CNN_ROAD_CLASSIFIER_PATH), "cnn_road_classifier.pth")

# STEP 2: Simple CNN Model from provided code
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Global variables
ROAD_CLASSIFIER = None
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_road_classifier() -> bool:
    """Load the road classifier model"""
    global ROAD_CLASSIFIER, TRANSFORM
    
    try:
        print(f"Loading road classifier from {ROAD_MODEL_PATH}...")
        start_time = time.time()
        
        # Create model
        model = SimpleCNN()
        
        # Check if model file exists
        if not os.path.exists(ROAD_MODEL_PATH):
            print(f"Road classifier model not found at {ROAD_MODEL_PATH}")
            ROAD_CLASSIFIER = None
            return False
        
        # Load the model
        try:
            checkpoint = torch.load(ROAD_MODEL_PATH, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            # Set model to evaluation mode
            model.eval()
            ROAD_CLASSIFIER = model
            print(f"Road classifier loaded successfully in {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error loading road classifier: {e}")
            ROAD_CLASSIFIER = None
            return False
            
    except Exception as e:
        print(f"Error loading road classifier: {e}")
        ROAD_CLASSIFIER = None
        return False

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
        
        # Roads typically have mid-range brightness
        road_color_range = (mean_brightness > 40 and mean_brightness < 180)
        
        # Calculate color balance
        r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
        color_balance = 1.0 - (max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean)) / 255.0)
        has_road_colors = color_balance > 0.85
        
        # Color consistency - roads often have consistent coloring
        r_std = np.std(img_array[:,:,0])
        g_std = np.std(img_array[:,:,1])
        b_std = np.std(img_array[:,:,2])
        color_consistency = 1.0 - min(1.0, (r_std + g_std + b_std) / 300)
        
        # Calculate road score based on features
        road_score = 0.0
        
        # Road color range
        if road_color_range:
            road_score += 0.3
        
        # Road color balance
        if has_road_colors:
            road_score += 0.4
        
        # Color consistency
        if color_consistency > 0.6:
            road_score += 0.3
        
        # Return features dictionary
        return {
            "road_score": road_score,
            "brightness": mean_brightness,
            "color_balance": color_balance,
            "color_consistency": color_consistency,
            "is_road_heuristic": road_score > 0.5
        }
        
    except Exception as e:
        print(f"Error in road feature analysis: {e}")
        return {
            "road_score": 0.0,
            "error": str(e),
            "is_road_heuristic": False
        }

def predict_road_image(image: Image.Image) -> Dict[str, Any]:
    """
    Use the CNN model to predict if an image contains a road
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing prediction results
    """
    global ROAD_CLASSIFIER, TRANSFORM
    
    if ROAD_CLASSIFIER is None:
        if not load_road_classifier():
            return {"is_road": False, "confidence": 0.0, "error": "Model not loaded"}
    
    try:
        # Preprocess the image
        img_tensor = TRANSFORM(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = ROAD_CLASSIFIER(img_tensor)
            confidence = output.item()  # Get confidence value (0-1)
        
        return {
            "is_road": confidence > 0.5,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error in road prediction: {e}")
        return {
            "is_road": False,
            "confidence": 0.0,
            "error": str(e)
        }

def validate_road_image(image_input: Union[str, Image.Image]) -> Dict[str, Any]:
    """
    Validate if an image contains a road surface
    
    Args:
        image_input: Can be a file path, PIL Image, or base64 string
        
    Returns:
        Dictionary containing validation results
    """
    # Handle different input types
    if isinstance(image_input, str):
        if image_input.startswith('data:image') or len(image_input) > 100:
            # Assume base64 string
            clean_base64 = image_input.split(',')[1] if ',' in image_input else image_input
            try:
                img_bytes = base64.b64decode(clean_base64)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
                return {"is_road": False, "confidence": 0.0, "error": str(e)}
        else:
            # Assume file path
            if not os.path.exists(image_input):
                return {"is_road": False, "confidence": 0.0, "error": f"File not found: {image_input}"}
            image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        # Try to convert to PIL Image
        try:
            image = Image.fromarray(image_input).convert('RGB')
        except Exception as e:
            return {"is_road": False, "confidence": 0.0, "error": str(e)}
    
    # Get heuristic features
    heuristic = analyze_road_features(image)
    
    # Get model prediction
    prediction = predict_road_image(image)
    
    # Combine results
    is_road = prediction.get("is_road", False)
    confidence = prediction.get("confidence", 0.0)
    
    # If model failed, use heuristic
    if "error" in prediction:
        is_road = heuristic["is_road_heuristic"]
        confidence = heuristic["road_score"]
    
    return {
        "is_road": is_road,
        "confidence": confidence,
        "model_prediction": is_road,
        "model_confidence": confidence,
        "heuristic_features": heuristic
    }

# Ensure model is loaded on module import
load_road_classifier()
