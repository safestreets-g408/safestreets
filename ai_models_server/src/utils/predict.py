"""
ViT Model Prediction Utilities
Handles Vision Transformer model inference for road damage classification
"""
import torch
from torchvision import transforms
from PIL import Image
import os
import time
import base64
import io
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
        
        # Create fallback model
        fallback_model = ViTClassifier(num_classes=len(CLASS_NAMES))
        
        # Try to load model with robust compatibility system
        if os.path.exists(VIT_MODEL_PATH):
            MODEL, is_fallback = RobustModelLoader.load_with_fallback(
                VIT_MODEL_PATH, 
                fallback_model,
                model_type="vit"
            )
            
            if MODEL:
                MODEL.eval()
                status = "fallback" if is_fallback else "loaded"
                print(f"ViT model {status} successfully in {time.time() - start_time:.2f} seconds")
                return True
            else:
                print("Failed to load any model variant")
                return False
        else:
            print(f"Warning: Model file not found at {VIT_MODEL_PATH}")
            # Use fallback model
            MODEL = fallback_model
            MODEL.eval()
            print(f"Using fallback ViT model in {time.time() - start_time:.2f} seconds")
            return True
            
    except Exception as e:
        print(f"Failed to load ViT model: {e}")
        # Last resort - create a minimal working model
        try:
            MODEL = ViTClassifier(num_classes=len(CLASS_NAMES))
            MODEL.eval()
            print("Created minimal ViT model as last resort")
            return True
        except Exception as final_e:
            print(f"Final fallback failed: {final_e}")
            MODEL = None
            return False


def predict_from_base64(base64_string: str) -> Dict[str, Any]:
    """
    Predict road damage from base64 encoded image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Dictionary containing prediction results
    """
    if MODEL is None:
        if not load_vit_model():
            return {
                'error': 'ViT model not available',
                'prediction': 'Unknown',
                'confidence': 0.0,
                'bbox': [0, 0, 0, 0],
                'fallback': True
            }
    
    try:
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

                # Bounding box conversion
                bbox_arr = bbox_output[0].cpu().numpy()
                w, h = original_size
                x_center, y_center, box_w, box_h = bbox_arr.tolist()
                x1 = max(0, int((x_center - box_w / 2) * w))
                y1 = max(0, int((y_center - box_h / 2) * h))
                x2 = min(w, int((x_center + box_w / 2) * w))
                y2 = min(h, int((y_center + box_h / 2) * h))

                return {
                    'prediction': pred_class,
                    'confidence': confidence_score,
                    'bbox': [x1, y1, x2, y2],
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
        import cv2
        
        # Decode base64 image
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Draw bounding box and class name
        x1, y1, x2, y2 = bbox_coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, pred_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        print(f"Error annotating image: {e}")
        return base64_string  # Return original if annotation fails


# Initialize model on module import
load_vit_model()
