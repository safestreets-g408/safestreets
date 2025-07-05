"""
ViT Model Prediction Utilities
Handles Vision Transformer model inference for road damage classification
"""
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import time
import base64
import io
import traceback
import numpy as np
from typing import Tuple, Optional, Dict, Any

from ..core.config import VIT_MODEL_PATH, CLASS_NAMES_PATH
from .model_compatibility import RobustModelLoader


class ViTClassifier(torch.nn.Module):
    """Vision Transformer classifier for road damage detection"""
    
    def __init__(self, num_classes: int):
        super(ViTClassifier, self).__init__()
        from torchvision.models.vision_transformer import vit_b_16
        self.vit = vit_b_16(pretrained=True)
        # Get the original classifier input features
        original_in_features = self.vit.heads.head.in_features
        # Replace the original classifier with our custom one
        self.vit.heads.head = torch.nn.Linear(original_in_features, num_classes)
        self.bbox_regressor = torch.nn.Linear(original_in_features, 4)  # x, y, width, height

    def forward(self, x):
        # Use the standard forward pass
        batch_size = x.shape[0]
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Apply encoder
        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        cls_token = x[:, 0]
        
        # Get classification output
        class_output = self.vit.heads.head(cls_token)
        # Get bounding box output
        bbox_output = self.bbox_regressor(cls_token)
        
        return class_output, bbox_output


# Global model variables
MODEL = None
CLASS_NAMES = []
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_vit_model() -> bool:
    """Load the ViT model and class names"""
    global MODEL, CLASS_NAMES
    
    try:
        print("Loading ViT model...")
        start_time = time.time()
        
        # Load class names
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                CLASS_NAMES = f.read().splitlines()
        else:
            print(f"Warning: Class names file not found at {CLASS_NAMES_PATH}")
            CLASS_NAMES = ["D00", "D10", "D20", "D40", "D50", "D60"]  # Default classes
        
        # Create model instance
        model = ViTClassifier(num_classes=len(CLASS_NAMES))
        
        # Check if model file exists
        if not os.path.exists(VIT_MODEL_PATH):
            print(f"Error: ViT model file not found at {VIT_MODEL_PATH}")
            return False
            
        try:
            # Try to load as TorchScript first
            MODEL = torch.jit.load(VIT_MODEL_PATH)
            print("ViT model loaded as TorchScript")
        except Exception as e:
            print(f"TorchScript loading failed: {e}, trying state dict...")
            # Try to load as state dict
            checkpoint = torch.load(VIT_MODEL_PATH, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Try to load state dict with flexible key matching
            try:
                model.load_state_dict(state_dict, strict=False)
                MODEL = model
                print("ViT model loaded from state dict")
            except Exception as e2:
                print(f"State dict loading failed: {e2}")
                return False
        
        # Set to evaluation mode
        MODEL.eval()
        print(f"ViT model loaded successfully in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return False


def predict_from_base64(base64_string: str) -> Dict[str, Any]:
    """
    Predict road damage from base64 encoded image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Dictionary containing prediction results
    """
    global MODEL
    
    # Ensure model is loaded
    if MODEL is None:
        if not load_vit_model():
            print("ViT model couldn't be loaded, returning error")
            return {
                'error': 'ViT model not available',
                'prediction': 'Unknown',
                'confidence': 0.0,
                'bbox': [0, 0, 0, 0],
                'fallback': True
            }
    
    try:
        # Clean the base64 string if needed
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Decode and transform image
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size
        input_tensor = TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            try:
                outputs = MODEL(input_tensor)
                # Unpack outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    bbox_output = torch.zeros((1, 4))
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, bbox_output = outputs
                else:
                    logits = outputs
                    bbox_output = torch.zeros((1, 4))

                # Classification
                if logits.dim() > 1:
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                else:
                    prob = torch.sigmoid(logits).item()
                    probabilities = torch.tensor([[1 - prob, prob]])
                
                confidence, predicted = torch.max(probabilities, 1)
                pred_idx = predicted.item()
                pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"
                confidence_score = float(confidence.item())

                # Advanced bounding box processing
                # First, normalize model outputs for better bounding box prediction
                bbox_arr = bbox_output[0].cpu().numpy()
                w, h = original_size
                
                # Get raw coordinates from model output
                x_center, y_center, box_w, box_h = bbox_arr.tolist()
                
                # Apply sigmoid to center coordinates to constrain them to [0, 1]
                x_center = 1 / (1 + np.exp(-x_center))
                y_center = 1 / (1 + np.exp(-y_center))
                
                # Ensure positive box dimensions using exponential function for width/height
                # This gives a better spread of values and avoids very small boxes
                box_w = np.exp(min(box_w, 2))  # Limit max box size 
                box_h = np.exp(min(box_h, 2))  # Limit max box size
                
                # Scale box size based on confidence (higher confidence = larger box)
                # This helps for cases where the model is uncertain
                conf_scaling = max(0.5, min(1.5, confidence_score * 2))
                box_w = min(0.8, box_w * conf_scaling / 10)  # Limit to 80% of image width
                box_h = min(0.8, box_h * conf_scaling / 10)  # Limit to 80% of image height
                
                # Use damage class to adjust box size (certain damages tend to be larger/smaller)
                if pred_class in ["D40", "D43", "D44"]:  # Large damage classes
                    box_w *= 1.2
                    box_h *= 1.2
                elif pred_class in ["D00", "D10"]:  # Small damage classes
                    box_w *= 0.8
                    box_h *= 0.8
                
                # Ensure minimum box size relative to image dimensions
                min_box_w = w * 0.1  # At least 10% of image width
                min_box_h = h * 0.1  # At least 10% of image height
                
                # Calculate box coordinates with normalized values
                x1 = max(0, int((x_center - box_w / 2) * w))
                y1 = max(0, int((y_center - box_h / 2) * h))
                x2 = min(w, int((x_center + box_w / 2) * w))
                y2 = min(h, int((y_center + box_h / 2) * h))
                
                # Enforce minimum box size
                if x2 - x1 < min_box_w:
                    # Center the box while expanding to minimum width
                    center_x = (x1 + x2) // 2
                    half_width = int(min_box_w // 2)
                    x1 = max(0, center_x - half_width)
                    x2 = min(w, center_x + half_width)
                
                if y2 - y1 < min_box_h:
                    # Center the box while expanding to minimum height
                    center_y = (y1 + y2) // 2
                    half_height = int(min_box_h // 2)
                    y1 = max(0, center_y - half_height)
                    y2 = min(h, center_y + half_height)
                
                # Generate enhanced annotated image
                draw = ImageDraw.Draw(img)
                
                # Draw bounding box with improved visibility
                # Main rectangle (thicker border for better visibility)
                border_width = max(3, min(5, int(w/100)))  # Dynamic border width based on image size
                
                # Draw box with double border for better visibility
                # First: outer yellow border
                draw.rectangle([(x1-2, y1-2), (x2+2, y2+2)], outline=(255, 255, 0), width=border_width+2)
                # Second: inner green border
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=border_width)
                
                # Draw corner markers for better visibility
                corner_size = max(5, min(10, int(w/60)))  # Dynamic corner size
                
                # Top-left corner
                draw.line([(x1, y1), (x1 + corner_size, y1)], fill=(255, 0, 0), width=border_width)
                draw.line([(x1, y1), (x1, y1 + corner_size)], fill=(255, 0, 0), width=border_width)
                
                # Top-right corner
                draw.line([(x2, y1), (x2 - corner_size, y1)], fill=(255, 0, 0), width=border_width)
                draw.line([(x2, y1), (x2, y1 + corner_size)], fill=(255, 0, 0), width=border_width)
                
                # Bottom-left corner
                draw.line([(x1, y2), (x1 + corner_size, y2)], fill=(255, 0, 0), width=border_width)
                draw.line([(x1, y2), (x1, y2 - corner_size)], fill=(255, 0, 0), width=border_width)
                
                # Bottom-right corner
                draw.line([(x2, y2), (x2 - corner_size, y2)], fill=(255, 0, 0), width=border_width)
                draw.line([(x2, y2), (x2, y2 - corner_size)], fill=(255, 0, 0), width=border_width)
                
                # Add semi-transparent highlight on the damage area
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 0, 60))  # Semi-transparent yellow
                
                # Convert original image to RGBA if it isn't already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Composite the images
                img = Image.alpha_composite(img, overlay)
                img = img.convert('RGB')  # Convert back to RGB for JPEG saving
                
                # Add text label with better background for readability
                text = f"{pred_class} ({confidence_score:.2f})"
                text_y = max(0, y1 - 25)
                
                # Calculate approximate text dimensions
                # Using a simple estimation since we can't rely on font measurements
                text_width = len(text) * 8  # Approximate 8 pixels per character
                text_height = 15            # Default text height
                
                # Draw black background for text to improve readability
                draw.rectangle(
                    [(x1, text_y - 2), (x1 + text_width + 10, text_y + text_height + 2)],
                    fill=(0, 0, 0)
                )
                
                # Draw text with high contrast color
                draw.text((x1 + 5, text_y), text, fill=(255, 255, 0))
                
                # Convert annotated image to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                annotated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return {
                    'prediction': pred_class,
                    'confidence': confidence_score,
                    'bbox': [x1, y1, x2, y2],
                    'annotated_image': annotated_image,
                    'all_predictions': {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(min(len(CLASS_NAMES), probabilities.shape[1]))},
                    'fallback': False
                }
            except Exception as inference_error:
                print(f"Inference error: {inference_error}")
                # Fallback response
                return {
                    'prediction': CLASS_NAMES[0] if CLASS_NAMES else "Unknown",
                    'confidence': 0.5,
                    'bbox': [0, 0, original_size[0] // 4, original_size[1] // 4],
                    'all_predictions': {cls: 1.0 / len(CLASS_NAMES) for cls in CLASS_NAMES},
                    'fallback': True
                }
    except Exception as e:
        print(f"Error in ViT prediction: {e}")
        return {
            'error': str(e),
            'prediction': 'Unknown',
            'confidence': 0.0,
            'bbox': [0, 0, 0, 0],
            'fallback': True
        }


def predict_and_annotate(image_path: str) -> Tuple[str, Optional[str]]:
    """
    Predict road damage and return annotated image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (prediction_class, annotated_image_base64)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if MODEL is None:
        if not load_vit_model():
            raise RuntimeError("ViT model not available")

    try:
        # Load and process image
        img = Image.open(image_path).convert("RGB")
        
        # Convert to base64 for prediction
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Get prediction
        result = predict_from_base64(img_base64)
        
        if result.get('fallback', False):
            return result['prediction'], None
        
        # Create annotated image
        annotated_base64 = annotate_image_base64(img_base64, result['prediction'], result['bbox'])
        
        return result['prediction'], annotated_base64
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e


def annotate_image_base64(base64_string: str, pred_class: str, bbox_coords: list) -> str:
    """
    Annotate image with prediction and bounding box
    
    Args:
        base64_string: Base64 encoded image
        pred_class: Predicted class name
        bbox_coords: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Base64 encoded annotated image
    """
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(base64_string)
        
        # Use PIL for annotation (already imported at the top)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Validate bbox coordinates
        w, h = img.size
        x1, y1, x2, y2 = bbox_coords
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        # Ensure valid box dimensions
        if x2 <= x1:
            x2 = min(w, x1 + 10)
        if y2 <= y1:
            y2 = min(h, y1 + 10)
        
        # Draw rectangle and text
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        
        # Position text, ensure it's not out of bounds
        text_y = max(0, y1 - 20)
        draw.text((x1, text_y), pred_class, fill=(255, 0, 0))
        
        # Convert back to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        print(f"Error annotating image: {e}")
        traceback.print_exc()
        return base64_string  # Return original if annotation fails


# Initialize model on module import
load_vit_model()
