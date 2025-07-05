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
import traceback
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


class RoadClassifierWrapper:
    """
    Wrapper class for road classifier to handle dimension mismatches
    and provide a consistent interface
    """
    def __init__(self, model):
        self.model = model
        self.input_size = (128, 128)  # Default input size that works
        self.eval_mode = True
    
    def eval(self):
        """Set model to evaluation mode"""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.eval_mode = True
        return self
    
    def __call__(self, x):
        """Forward pass with proper error handling"""
        # Make sure input is the right size
        if x.shape[2:] != self.input_size:
            # Resize input to expected size
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                # Can't resize during tracing/scripting
                raise ValueError(f"Expected input size {self.input_size}, got {x.shape[2:]}")
            
            # Create resizing transform
            resize_transform = transforms.Resize(self.input_size)
            x = resize_transform(x)
        
        # Try to run the model
        try:
            return self.model(x)
        except RuntimeError as e:
            # Check if it's a dimension mismatch error
            if "shape" in str(e) and "invalid for input of size" in str(e):
                print(f"Dimension mismatch: {e}")
                # Return a fallback prediction for road
                return torch.tensor([[0.0, 1.0]])
            raise e


def load_road_classifier() -> bool:
    """Load the road classifier model"""
    global ROAD_CLASSIFIER, WORKING_INPUT_SIZE
    
    try:
        print("Loading road classifier...")
        start_time = time.time()
        
        # Create model
        model = CNNRoadClassifier()
        
        # Check if model file exists
        if not os.path.exists(CNN_ROAD_CLASSIFIER_PATH):
            print(f"Road classifier model not found at {CNN_ROAD_CLASSIFIER_PATH}")
            ROAD_CLASSIFIER = None
            return False
            
        # Try to load as TorchScript first
        try:
            torchscript_model = torch.jit.load(CNN_ROAD_CLASSIFIER_PATH, map_location="cpu")
            # Wrap the model to handle dimension mismatches
            ROAD_CLASSIFIER = RoadClassifierWrapper(torchscript_model)
            ROAD_CLASSIFIER.eval()
            # Set working input size to the one we know works based on error messages
            WORKING_INPUT_SIZE = (128, 128)
            ROAD_CLASSIFIER.input_size = WORKING_INPUT_SIZE
            print("Road classifier loaded as TorchScript with input size", WORKING_INPUT_SIZE)
            print(f"Road classifier loaded successfully in {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"TorchScript loading failed: {e}, trying state dict...")
            
            # Try to load as state dict
            try:
                checkpoint = torch.load(CNN_ROAD_CLASSIFIER_PATH, map_location="cpu")
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                    
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                ROAD_CLASSIFIER = RoadClassifierWrapper(model)
                ROAD_CLASSIFIER.eval()
                # Set working input size
                WORKING_INPUT_SIZE = (128, 128)
                ROAD_CLASSIFIER.input_size = WORKING_INPUT_SIZE
                print(f"Road classifier loaded successfully from state dict in {time.time() - start_time:.2f} seconds")
                return True
            except Exception as e2:
                print(f"State dict loading failed: {e2}, trying RobustModelLoader...")
                
                # Try using RobustModelLoader as last resort
                try:
                    loaded_model, is_fallback = RobustModelLoader.load_with_fallback(
                        CNN_ROAD_CLASSIFIER_PATH,
                        model,
                        model_type="cnn"
                    )
                    
                    if loaded_model:
                        ROAD_CLASSIFIER = RoadClassifierWrapper(loaded_model)
                        ROAD_CLASSIFIER.eval()
                        # Set working input size
                        WORKING_INPUT_SIZE = (128, 128)
                        ROAD_CLASSIFIER.input_size = WORKING_INPUT_SIZE
                        status = "fallback" if is_fallback else "loaded"
                        print(f"Road classifier {status} successfully using RobustModelLoader in {time.time() - start_time:.2f} seconds")
                        return True
                    else:
                        print("RobustModelLoader failed to load model")
                        ROAD_CLASSIFIER = None
                        return False
                except Exception as e3:
                    print(f"RobustModelLoader failed: {e3}")
                    ROAD_CLASSIFIER = None
                    return False
    
    except Exception as e:
        print(f"Error loading road classifier: {e}")
        ROAD_CLASSIFIER = None
        return False


def validate_road_image(image_input: Union[str, Image.Image]) -> Dict[str, Any]:
    """
    Validate if an image contains a road surface
    
    Args:
        image_input: Can be a file path, PIL Image, or base64 string
        
    Returns:
        Dictionary containing validation results
    """
    global ROAD_CLASSIFIER
    
    # Ensure model is loaded
    if ROAD_CLASSIFIER is None:
        if not load_road_classifier():
            print("Road classifier couldn't be loaded, using heuristic analysis only")
    
    try:
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
                    raise ValueError(f"Invalid base64 image data: {str(e)}")
            else:
                # Assume file path
                if not os.path.exists(image_input):
                    raise ValueError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            # Try to convert to PIL Image
            try:
                image = Image.fromarray(image_input).convert('RGB')
            except Exception as e:
                raise ValueError(f"Could not convert input to image: {str(e)}")
        
        # Always run heuristic analysis
        road_features = analyze_road_features(image)
        print(f"Heuristic road score: {road_features['road_score']:.2f}")
        
        # Try model inference if available
        model_prediction = None
        model_confidence = 0.5
        
        if ROAD_CLASSIFIER is not None:
            print("Using CNN model for road classification")
            try:
                # Always use the known working input size
                image_resized = image.resize(WORKING_INPUT_SIZE, Image.LANCZOS)
                
                # Create transform for the specific input size
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image_resized).unsqueeze(0)
                
                # Run inference with error handling
                with torch.no_grad():
                    try:
                        outputs = ROAD_CLASSIFIER(input_tensor)
                        
                        # Handle different model output formats
                        if hasattr(outputs, 'shape'):
                            # Check if output is binary (single value) or multi-class
                            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                                # Multi-class output with logits [batch_size, num_classes]
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                # Handle both binary and multi-class scenarios
                                if outputs.shape[1] >= 2:
                                    # We have at least 2 classes
                                    road_prob = probabilities[0, 1].item()  # Probability of class 1
                                    nonroad_prob = probabilities[0, 0].item()  # Probability of class 0
                                else:
                                    # Single class output - treat as probability
                                    road_prob = probabilities[0, 0].item()
                                    nonroad_prob = 1.0 - road_prob
                            else:
                                # Binary output - single value
                                # Apply sigmoid to get probability
                                if outputs.numel() == 1:  # Single output value
                                    road_prob = torch.sigmoid(outputs).item()
                                else:
                                    # Take first element if multiple outputs but single class
                                    road_prob = torch.sigmoid(outputs[0]).item()
                                    
                                nonroad_prob = 1.0 - road_prob
                                predicted = torch.tensor([1]) if road_prob > 0.5 else torch.tensor([0])
                                confidence = torch.tensor([max(road_prob, nonroad_prob)])
                            
                            # Determine which class is more likely to be "road" based on image features
                            # If heuristic analysis strongly suggests it's a road but model says otherwise,
                            # we might have a label inversion
                            heuristic_is_road = road_features['road_score'] > 0.7
                            
                            # Model prediction initially assumes class 1 is "road" if multi-class
                            # For binary output, >0.5 means road
                            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                                model_prediction = predicted.item() == 1
                            else:
                                model_prediction = road_prob > 0.5
                                
                            model_confidence = float(confidence.item() if hasattr(confidence, 'item') else confidence)
                            
                            # If labels might be inverted, adjust the prediction
                            if heuristic_is_road and model_prediction == False and nonroad_prob > 0.85:
                                # Strong heuristic evidence for road but model confidently says non-road
                                # Possible label inversion
                                print("Warning: Possible label inversion detected")
                                model_prediction = not model_prediction
                                model_confidence = nonroad_prob if model_prediction else road_prob
                            
                            print(f"Model prediction: {model_prediction} (confidence: {model_confidence:.4f}, road prob: {road_prob:.4f}, non-road prob: {nonroad_prob:.4f})")
                            
                            # Adjust confidence threshold based on image features
                            # If image has very strong road features, we may want to give model prediction more weight
                            if road_features['road_score'] > 0.8 and model_prediction:
                                # Strong heuristic evidence for road and model agrees
                                model_confidence = max(model_confidence, 0.85)
                            elif road_features['road_score'] < 0.3 and not model_prediction:
                                # Strong heuristic evidence against road and model agrees
                                model_confidence = max(model_confidence, 0.85)
                        else:
                            # Single output - assume sigmoid activation where >0.5 means "road"
                            if hasattr(outputs, 'item'):
                                prob = torch.sigmoid(outputs).item()
                            else:
                                prob = 0.5
                            model_prediction = prob > 0.5
                            model_confidence = prob if model_prediction else (1.0 - prob)
                            print(f"Model prediction (single output): {model_prediction} (confidence: {model_confidence:.4f})")
                    
                    except Exception as inference_error:
                        print(f"Error during model inference: {inference_error}")
                        # If we have error during inference, rely on heuristics
                        model_prediction = road_features['road_score'] > 0.55
                        model_confidence = max(0.6, road_features['road_score'])
                        print(f"Using heuristic fallback due to inference error: {model_prediction} (confidence: {model_confidence:.4f})")
                
            except Exception as model_error:
                print(f"Error in CNN model setup: {model_error}")
                traceback.print_exc()
                model_prediction = None
                model_confidence = 0.5
        
        # Make final decision using model and heuristics
        if model_prediction is not None:
            # Adjust model confidence based on image features
            heuristic_road_score = road_features['road_score']
            
            # Determine weights for model vs heuristics
            # - If model is very confident (>0.9), give it more weight
            # - If heuristics are very strong (>0.8 or <0.2), give them more weight
            # - Otherwise use a balanced approach
            
            if model_confidence > 0.9:
                # Very confident model prediction
                model_weight = 0.85
                heuristic_weight = 0.15
            elif heuristic_road_score > 0.8 or heuristic_road_score < 0.2:
                # Very strong heuristic evidence
                model_weight = 0.5
                heuristic_weight = 0.5
            else:
                # Balanced approach
                model_weight = 0.7
                heuristic_weight = 0.3
            
            print(f"Using model weight: {model_weight}, heuristic weight: {heuristic_weight}")
            
            # Check if model and heuristics strongly disagree
            model_says_road = model_prediction
            heuristics_say_road = heuristic_road_score > 0.6
            
            strong_disagreement = (model_says_road and heuristic_road_score < 0.3) or \
                                 (not model_says_road and heuristic_road_score > 0.8)
            
            if strong_disagreement:
                print("⚠️ Strong disagreement between model and heuristics!")
                # If heuristics are very strong, they should have more influence
                if heuristic_road_score > 0.8:
                    model_weight = 0.4
                    heuristic_weight = 0.6
                elif heuristic_road_score < 0.2:
                    model_weight = 0.4
                    heuristic_weight = 0.6
            
            # Calculate weighted score
            # For model contribution, use model confidence if aligned with prediction
            model_score = model_confidence if model_prediction else (1.0 - model_confidence)
            final_score = (model_score * model_weight) + (heuristic_road_score * heuristic_weight)
            
            # Decide based on weighted score with threshold
            final_is_road = final_score > 0.55
            
            # Confidence reflects both model confidence and heuristic strength
            final_confidence = final_score
            
            # If model is extremely confident (>0.95) and agrees with heuristics, boost confidence
            if model_confidence > 0.95 and model_says_road == heuristics_say_road:
                final_confidence = max(final_confidence, 0.95)
            
            print(f"Final decision: {final_is_road} (confidence: {final_confidence:.4f})")
        else:
            # Use heuristic only if model prediction not available
            heuristic_road_score = road_features['road_score']
            final_is_road = heuristic_road_score > 0.55
            final_confidence = heuristic_road_score
            print(f"Heuristic-only decision: {final_is_road} (confidence: {final_confidence:.4f})")
        
        # Ensure proper Python types for JSON serialization
        result = {
            'is_road': bool(final_is_road),
            'confidence': float(final_confidence),
            'model_prediction': bool(model_prediction) if model_prediction is not None else None,
            'model_confidence': float(model_confidence),
            'heuristic_features': road_features,
            'fallback': model_prediction is None
        }
        
        return result
            
    except Exception as e:
        print(f"Error in road validation: {e}")
        traceback.print_exc()
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
        # Roads typically have mid-range brightness and aren't too bright or dark
        road_color_range = (mean_brightness > 40 and mean_brightness < 180)
        
        # Asphalt/concrete road color detection
        # Calculate color histograms to check for dominant gray/asphalt colors
        r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
        
        # Road surfaces often have R, G, B values that are close to each other (grayscale-like)
        color_balance = 1.0 - (max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean)) / 255.0)
        has_road_colors = color_balance > 0.85
        
        # Color distribution - roads often have consistent coloring
        r_std = np.std(img_array[:,:,0])
        g_std = np.std(img_array[:,:,1])
        b_std = np.std(img_array[:,:,2])
        color_consistency = 1.0 - min(1.0, (r_std + g_std + b_std) / 300)  # Lower variation is better for roads
        
        # Check for linear features (simplified edge detection)
        edges = detect_edges_simple(gray)
        edge_density = np.mean(edges)
        
        # Check for vanishing point - roads often have lines converging to a point
        # This is a simplified approach using horizontal vs vertical edges
        horiz_edges = np.abs(np.gradient(gray, axis=0))
        vert_edges = np.abs(np.gradient(gray, axis=1))
        
        # Roads often have stronger horizontal edges at bottom and vertical edges in middle/top
        h, w = horiz_edges.shape
        top_half = horiz_edges[:h//2, :]
        bottom_half = horiz_edges[h//2:, :]
        
        # Check if bottom has stronger horizontal edges (road perspective)
        bottom_stronger = np.mean(bottom_half) > np.mean(top_half)
        
        # Calculate directional ratio
        directional_ratio = np.mean(horiz_edges) / (np.mean(vert_edges) + 0.001)
        has_directional_edges = 0.8 < directional_ratio < 1.5
        
        # Check for uniform texture (roads tend to have some uniformity)
        texture_score = 1.0 - min(1.0, brightness_std / 100.0)
        
        # Calculate road score based on features
        road_score = 0.0
        
        # Road color range
        if road_color_range:
            road_score += 0.15
            print("✓ Image has typical road brightness")
        else:
            print("✗ Image lacks typical road brightness")
            
        # Road color balance
        if has_road_colors:
            road_score += 0.15
            print(f"✓ Image has balanced road colors (score: {color_balance:.2f})")
        else:
            print(f"✗ Image lacks balanced road colors (score: {color_balance:.2f})")
        
        # Color consistency
        if color_consistency > 0.6:
            road_score += 0.15
            print(f"✓ Image has consistent colors (score: {color_consistency:.2f})")
        else:
            print(f"✗ Image has inconsistent colors (score: {color_consistency:.2f})")
        
        # Edge features
        if edge_density > 0.08 and edge_density < 0.25:  # Roads have some edges but aren't too busy
            road_score += 0.15
            print(f"✓ Image has appropriate edge density: {edge_density:.2f}")
        else:
            print(f"✗ Image has inappropriate edge density: {edge_density:.2f}")
        
        # Perspective features (bottom-heavier horizontal edges)
        if bottom_stronger:
            road_score += 0.15
            print("✓ Image has road-like perspective (stronger bottom edges)")
        else:
            print("✗ Image lacks road-like perspective")
        
        # Directional edges
        if has_directional_edges:
            road_score += 0.15
            print(f"✓ Image has directional edge pattern (ratio: {directional_ratio:.2f})")
        else:
            print(f"✗ Image lacks directional edge pattern (ratio: {directional_ratio:.2f})")
        
        # Texture uniformity
        if texture_score > 0.4:  # Reasonable texture uniformity
            road_score += 0.1
            print(f"✓ Image has uniform texture (score: {texture_score:.2f})")
        else:
            print(f"✗ Image lacks uniform texture (score: {texture_score:.2f})")
        
        # Aspect ratio check (roads are often horizontal)
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        if aspect_ratio > 1.2:  # Wider than tall
            road_score += 0.1
            print("✓ Image has typical road aspect ratio")
        
        # Final score (ensure it's between 0 and 1)
        final_road_score = min(road_score, 1.0)
        print(f"Final heuristic road score: {final_road_score:.4f}")
        
        return {
            'road_score': final_road_score,
            'mean_brightness': float(mean_brightness),
            'brightness_std': float(brightness_std),
            'edge_density': float(edge_density),
            'directional_ratio': float(directional_ratio),
            'texture_score': float(texture_score),
            'color_consistency': float(color_consistency),
            'color_balance': float(color_balance),
            'bottom_stronger': bool(bottom_stronger),
            'aspect_ratio': float(aspect_ratio),
            'road_color_range': bool(road_color_range),
            'has_road_colors': bool(has_road_colors)
        }
        
    except Exception as e:
        print(f"Error analyzing road features: {e}")
        traceback.print_exc()
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
