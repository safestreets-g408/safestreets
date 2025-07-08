"""
Updated YOLO v8 Model Utilities
Handles YOLO object detection for road damage detection using YOLOv8
"""
import torch
import base64
import io
import os
import time
import traceback
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2

from ..core.config import YOLO_MODEL_PATH, CLASS_NAMES_PATH, DEFAULT_YOLO_CONFIDENCE

# Global YOLO model
YOLO_MODEL = None
CLASS_NAMES = []

# Load class names from file
def load_class_names():
    """Load class names from class_names.txt file"""
    global CLASS_NAMES
    
    try:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(CLASS_NAMES)} class names: {CLASS_NAMES}")
        else:
            # Fallback class names based on RDD2022 dataset as shown in the notebook
            CLASS_NAMES = ['D00', 'D01', 'D0w0', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D50']
            print(f"Class names file not found at {CLASS_NAMES_PATH}, using default road damage classes")
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to default class names
        CLASS_NAMES = ['D00', 'D01', 'D0w0', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D50']


def load_yolo_model() -> bool:
    """Load the YOLOv8 model"""
    global YOLO_MODEL, CLASS_NAMES
    
    try:
        print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        start_time = time.time()
        
        # Load class names first
        if not CLASS_NAMES:
            load_class_names()
            
        # Check if model file exists
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"YOLO model file not found at {YOLO_MODEL_PATH}")
            return False
            
        # Try to import YOLOv8
        try:
            from ultralytics import YOLO
            print("Using YOLOv8 from ultralytics")
            YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
            
            # Force the model to only detect road damage classes
            # This ensures we don't detect general objects like cars, persons, etc.
            if hasattr(YOLO_MODEL, 'names'):
                print(f"Model originally has {len(YOLO_MODEL.names)} classes")
                # Check if model classes match our road damage classes
                if not all(cls in CLASS_NAMES for cls in YOLO_MODEL.names.values()):
                    print("Model has non-road damage classes, configuring to only detect road damage")
                    
                    # If using general YOLOv8 model, restrict classes to road damage
                    if hasattr(YOLO_MODEL, 'overrides'):
                        print("Setting class overrides to focus on road damage")
                        # Only detect the road damage classes during prediction
                        YOLO_MODEL.overrides['classes'] = [i for i, name in YOLO_MODEL.names.items() 
                                                         if name.startswith('D') and name[1:].isdigit()]
                        print(f"Restricted detection to classes: {YOLO_MODEL.overrides.get('classes')}")
            
            print(f"YOLO model loaded successfully in {time.time() - start_time:.2f} seconds")
            return True
        except ImportError:
            print("Ultralytics not installed, falling back to YOLOv5")
            try:
                # Fall back to YOLOv5 if ultralytics not available
                YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, source='local')
                # For YOLOv5, filter classes during prediction in detect_road_damage function
                print(f"YOLOv5 model loaded successfully in {time.time() - start_time:.2f} seconds")
                return True
            except Exception as e2:
                print(f"Error loading YOLOv5 model: {e2}")
                return False
                
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        traceback.print_exc()
        YOLO_MODEL = None
        return False


def process_image_from_base64(base64_string: str, resize: bool = True) -> Image.Image:
    """
    Convert base64 string to PIL Image for YOLO processing
    
    Args:
        base64_string: Base64 encoded image string
        resize: Whether to resize large images for better performance
        
    Returns:
        PIL Image object
    """
    try:
        # Handle data URI format
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
            
        # Decode base64 string
        img_bytes = base64.b64decode(base64_string)
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Print image info for debugging
        print(f"Loaded image: {img.format}, size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            print(f"Converted image to RGB mode")
        
        # Resize large images for better performance
        # Road damage is typically visible at lower resolutions
        if resize and (img.width > 1280 or img.height > 1280):
            # Keep aspect ratio
            if img.width > img.height:
                new_width = 1280
                new_height = int(img.height * (1280 / img.width))
            else:
                new_height = 1280
                new_width = int(img.width * (1280 / img.height))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized image to {new_width}x{new_height} for better performance")
        
        return img
    except Exception as e:
        print(f"Error processing base64 image: {e}")
        traceback.print_exc()
        raise


def detect_road_damage(image_input: Union[str, Image.Image, np.ndarray], 
                       save_path: Optional[str] = None,
                       conf_threshold: float = None) -> Dict[str, Any]:
    """
    Detect road damage using YOLO model
    
    Args:
        image_input: Can be a file path, PIL Image, base64 string, or numpy array
        save_path: Optional path to save annotated image
        conf_threshold: Confidence threshold for detections (0.0 to 1.0)
        
    Returns:
        Dictionary containing detection results
    """
    global YOLO_MODEL, CLASS_NAMES
    
    # Set default confidence threshold
    if conf_threshold is None:
        conf_threshold = DEFAULT_YOLO_CONFIDENCE
    
    # Ensure model is loaded
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
        # Start timing
        start_time = time.time()
        
        # Handle different input types
        if isinstance(image_input, str):
            if image_input.startswith('data:image') or len(image_input) > 100:
                # Base64 string
                image = process_image_from_base64(image_input)
            else:
                # File path
                if not os.path.exists(image_input):
                    return {'error': f"File not found: {image_input}", 'fallback': True}
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_input)
        else:
            return {'error': f"Unsupported input type: {type(image_input)}", 'fallback': True}
        
        # Run inference
        print(f"Running YOLO inference with confidence threshold: {conf_threshold}")
        
        # Define road damage class patterns
        road_damage_patterns = ['D00', 'D01', 'D10', 'D11', 'D20', 'D30', 'D40', 'D43', 'D44', 'D50']
        
        if hasattr(YOLO_MODEL, 'predict'):  # YOLOv8 model
            # For YOLOv8, we can pass classes directly to the predict method
            # This is more efficient than filtering after prediction
            
            # Check if we need to specify classes (depends on model being used)
            classes = None
            if hasattr(YOLO_MODEL, 'names'):
                # Get class indices for road damage classes
                class_indices = []
                for idx, name in YOLO_MODEL.names.items():
                    # Check if this is a road damage class
                    is_road_damage = False
                    for pattern in road_damage_patterns:
                        if name.upper() == pattern or name.startswith(pattern):
                            is_road_damage = True
                            break
                    
                    if is_road_damage:
                        class_indices.append(idx)
                
                if class_indices:
                    classes = class_indices
                    print(f"Restricting detection to road damage classes: {[YOLO_MODEL.names[i] for i in classes]}")
            
            # Run prediction with specified classes
            results = YOLO_MODEL.predict(
                source=image, 
                conf=conf_threshold,
                classes=classes  # This will filter to only road damage classes if set
            )
            
            # Process YOLOv8 results
            detections = []
            for r in results:
                print(f"Got {len(r.boxes)} detections")
                if len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get class name
                        if cls_id < len(YOLO_MODEL.names):
                            class_name = YOLO_MODEL.names[cls_id]
                        else:
                            class_name = f"Class_{cls_id}"
                            
                        print(f"Detection {i}: Class={class_name}, Confidence={conf:.2f}")
                        
                        # Check if this is actually a road damage class
                        is_road_damage = any(class_name.upper() == pattern or class_name.startswith(pattern) 
                                             for pattern in road_damage_patterns)
                        
                        if not is_road_damage:
                            print(f"Skipping non-road damage detection: {class_name}")
                            continue
                            
                        # Get bounding box
                        if hasattr(box, 'xyxy'):
                            bbox = box.xyxy[0].tolist()  # Convert to list
                        elif hasattr(box, 'xywh'):
                            # Convert xywh to xyxy
                            x, y, w, h = box.xywh[0].tolist()
                            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]
                        else:
                            print("Warning: No bounding box format available")
                            continue
                            
                        # Map to standard damage class if needed
                        mapped_class = class_name
                        if not any(class_name.upper() == pattern for pattern in road_damage_patterns):
                            # Try to map to a standard class
                            for pattern in road_damage_patterns:
                                if class_name.startswith(pattern):
                                    mapped_class = pattern
                                    break
                        
                        detections.append({
                            'class': mapped_class,
                            'confidence': conf,
                            'bbox': bbox
                        })
            
            # Get annotated image
            annotated_image = None
            if len(results) > 0:
                # Plot the results
                result_img = results[0].plot()
                
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', result_img)
                annotated_image = base64.b64encode(buffer).decode('utf-8')
                
                # Save if path provided
                if save_path:
                    try:
                        cv2.imwrite(save_path, result_img)
                        print(f"Saved annotated image to: {save_path}")
                    except Exception as e:
                        print(f"Error saving annotated image: {e}")
                
        else:  # YOLOv5 model (fallback)
            results = YOLO_MODEL(image)
            
            # Process YOLOv5 results
            detections = []
            df = results.pandas().xyxy[0]  # Get first image results
            print(f"YOLOv5 detected {len(df)} objects")
            
            for _, detection in df.iterrows():
                class_name = detection['name']
                conf = float(detection['confidence'])
                
                print(f"YOLOv5 Detection: Class={class_name}, Confidence={conf:.2f}")
                
                # Only include detections above the threshold
                if conf < conf_threshold:
                    continue
                
                # Check if this is a road damage class
                is_road_damage = any(class_name.upper() == pattern or class_name.startswith(pattern) 
                                     for pattern in road_damage_patterns)
                
                if not is_road_damage:
                    print(f"Skipping non-road damage detection: {class_name}")
                    continue
                
                # Map to standard damage class if needed
                mapped_class = class_name
                if not any(class_name.upper() == pattern for pattern in road_damage_patterns):
                    # Try to map to a standard class
                    for pattern in road_damage_patterns:
                        if class_name.startswith(pattern):
                            mapped_class = pattern
                            break
                
                detections.append({
                    'class': mapped_class,
                    'confidence': conf,
                    'bbox': [
                        float(detection['xmin']),
                        float(detection['ymin']),
                        float(detection['xmax']),
                        float(detection['ymax'])
                    ]
                })
            
            # Get annotated image
            annotated_image = None
            annotated_imgs = results.render()
            if annotated_imgs:
                annotated_img = annotated_imgs[0]
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', annotated_img)
                annotated_image = base64.b64encode(buffer).decode('utf-8')
                
                # Save if path provided
                if save_path:
                    try:
                        cv2.imwrite(save_path, annotated_img)
                        print(f"Saved annotated image to: {save_path}")
                    except Exception as e:
                        print(f"Error saving annotated image: {e}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(f"Final road damage detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"  {i+1}. Class: {det['class']}, Confidence: {det['confidence']:.2f}")
            
        # If no road damage is detected, provide a message
        if len(detections) == 0:
            print("No road damage detected in the image")
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'count': len(detections),
            'inference_time': elapsed_time,
            'fallback': False
        }
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'detections': [],
            'annotated_image': None,
            'count': 0,
            'fallback': True
        }


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded YOLO model"""
    global YOLO_MODEL, CLASS_NAMES
    
    # Road damage class patterns for verification
    road_damage_patterns = ['D00', 'D01', 'D10', 'D11', 'D20', 'D30', 'D40', 'D43', 'D44', 'D50']
    
    if YOLO_MODEL is None:
        return {
            'loaded': False,
            'model_type': None,
            'classes': CLASS_NAMES,
            'confidence_threshold': DEFAULT_YOLO_CONFIDENCE,
            'is_yolov8': False,
            'is_road_damage_model': False
        }
    
    # Determine model type (YOLOv8 vs YOLOv5)
    is_yolov8 = hasattr(YOLO_MODEL, 'predict')
    model_type = 'YOLOv8' if is_yolov8 else 'YOLOv5'
    
    # Check if model has road damage classes
    road_damage_classes = []
    other_classes = []
    
    if hasattr(YOLO_MODEL, 'names'):
        if isinstance(YOLO_MODEL.names, dict):
            # YOLOv8 style
            class_names = list(YOLO_MODEL.names.values())
        else:
            # YOLOv5 style
            class_names = YOLO_MODEL.names
    else:
        # Use our loaded class names as fallback
        class_names = CLASS_NAMES
    
    # Categorize classes
    for cls in class_names:
        if isinstance(cls, str) and any(cls.upper() == pattern or cls.startswith(pattern) for pattern in road_damage_patterns):
            road_damage_classes.append(cls)
        else:
            other_classes.append(cls)
    
    is_road_damage_model = len(road_damage_classes) > 0
    
    return {
        'loaded': True,
        'model_type': model_type,
        'classes': CLASS_NAMES,
        'model_classes': class_names,
        'road_damage_classes': road_damage_classes,
        'other_classes': other_classes,
        'confidence_threshold': DEFAULT_YOLO_CONFIDENCE,
        'is_yolov8': is_yolov8,
        'is_road_damage_model': is_road_damage_model,
        'model_path': YOLO_MODEL_PATH
    }


def verify_road_damage_model() -> bool:
    """
    Verify if the current model is suitable for road damage detection
    
    Returns:
        bool: True if the model is suitable, False otherwise
    """
    global YOLO_MODEL, CLASS_NAMES
    
    # Road damage class patterns
    road_damage_patterns = ['D00', 'D01', 'D10', 'D11', 'D20', 'D30', 'D40', 'D43', 'D44', 'D50']
    
    # Ensure model is loaded
    if YOLO_MODEL is None:
        if not load_yolo_model():
            print("Could not load YOLO model for verification")
            return False
    
    try:
        # Check if the model has road damage classes
        has_road_damage_classes = False
        
        if hasattr(YOLO_MODEL, 'names'):
            # YOLOv8
            class_names = list(YOLO_MODEL.names.values())
        elif hasattr(YOLO_MODEL, 'names') and isinstance(YOLO_MODEL.names, list):
            # YOLOv5
            class_names = YOLO_MODEL.names
        else:
            # Use our loaded class names as a fallback
            class_names = CLASS_NAMES
            
        print(f"Model has the following classes: {class_names}")
        
        # Check if any class matches road damage patterns
        for cls in class_names:
            if isinstance(cls, str) and any(cls.upper() == pattern or cls.startswith(pattern) for pattern in road_damage_patterns):
                has_road_damage_classes = True
                print(f"Found road damage class: {cls}")
        
        if not has_road_damage_classes:
            print("WARNING: The model does not have specific road damage classes.")
            print("It may detect general objects like cars instead of road damage.")
            print("Consider using a model specifically trained for road damage detection.")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False


# Load model and class names on module import
if not CLASS_NAMES:
    load_class_names()

# Verify model on module import
verify_road_damage_model()
