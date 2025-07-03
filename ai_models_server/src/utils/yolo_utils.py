"""
YOLO Model Utilities
Handles YOLO object detection for road damage detection
"""
import torch
import base64
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

from ..core.config import YOLO_MODEL_PATH


# Global YOLO model
YOLO_MODEL = None


def load_yolo_model() -> bool:
    """Load the YOLO model"""
    global YOLO_MODEL
    
    try:
        print("Loading YOLO model...")
        
        # Try to load custom model first
        if YOLO_MODEL_PATH and torch.cuda.is_available():
            YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
        else:
            # Load pre-trained model as fallback
            YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # Set confidence threshold
        YOLO_MODEL.conf = 0.5
        print("YOLO model loaded successfully")
        return True
        
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        YOLO_MODEL = None
        return False


def process_image_from_base64(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image for YOLO processing
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    try:
        # Decode base64 string
        img_bytes = base64.b64decode(base64_string)
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except Exception as e:
        print(f"Error processing base64 image: {e}")
        raise


def detect_road_damage(image_input, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect road damage using YOLO model
    
    Args:
        image_input: Can be a file path, PIL Image, or base64 string
        save_path: Optional path to save annotated image
        
    Returns:
        Dictionary containing detection results
    """
    if YOLO_MODEL is None:
        if not load_yolo_model():
            return {
                'error': 'YOLO model not available',
                'detections': [],
                'annotated_image': None,
                'count': 0,
                'fallback': True
            }
    
    try:
        # Handle different input types
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or len(image_input) > 100:
                # Assume base64 string
                clean_base64 = image_input.split(',')[1] if ',' in image_input else image_input
                image = process_image_from_base64(clean_base64)
            else:
                # Assume file path
                image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            # Try to convert to PIL Image
            image = Image.fromarray(image_input)
        
        # Run inference
        results = YOLO_MODEL(image)
        
        # Process results
        detections = []
        if hasattr(results, 'pandas'):
            # YOLOv5 format
            df = results.pandas().xyxy[0]  # Get first image results
            for _, detection in df.iterrows():
                detections.append({
                    'class': detection['name'],
                    'confidence': float(detection['confidence']),
                    'bbox': [
                        float(detection['xmin']),
                        float(detection['ymin']),
                        float(detection['xmax']),
                        float(detection['ymax'])
                    ]
                })
        else:
            # Alternative format handling
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detections.append({
                            'class': result.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        })
        
        # Get annotated image
        annotated_image = None
        if hasattr(results, 'render'):
            # YOLOv5 format
            annotated_imgs = results.render()
            if annotated_imgs:
                import cv2
                annotated_img = annotated_imgs[0]
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', annotated_img)
                annotated_image = base64.b64encode(buffer).decode('utf-8')
        else:
            # Alternative format - convert PIL to base64
            try:
                result_img = results[0].plot()
                import cv2
                _, buffer = cv2.imencode('.jpg', result_img)
                annotated_image = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Error creating annotated image: {e}")
        
        # Save if path provided
        if save_path and annotated_image:
            try:
                import cv2
                img_data = base64.b64decode(annotated_image)
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                cv2.imwrite(save_path, img)
            except Exception as e:
                print(f"Error saving annotated image: {e}")
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'count': len(detections),
            'fallback': False
        }
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return {
            'error': str(e),
            'detections': [],
            'annotated_image': None,
            'count': 0,
            'fallback': True
        }


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded YOLO model"""
    if YOLO_MODEL is None:
        return {
            'loaded': False,
            'model_type': None,
            'classes': []
        }
    
    try:
        return {
            'loaded': True,
            'model_type': 'YOLOv5',
            'classes': list(YOLO_MODEL.names.values()) if hasattr(YOLO_MODEL, 'names') else [],
            'confidence_threshold': getattr(YOLO_MODEL, 'conf', 0.5)
        }
    except Exception as e:
        return {
            'loaded': False,
            'error': str(e),
            'classes': []
        }


# Initialize model on module import
load_yolo_model()
