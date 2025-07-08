"""
Model Manager for AI Models Server
Handles loading and status of different AI models
"""
import os
import sys
import traceback
from typing import Optional, Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from ..core.config import (
    VIT_MODEL_PATH,
    YOLO_MODEL_PATH, 
    CNN_ROAD_CLASSIFIER_PATH,
    CLASS_NAMES_PATH
)


class ModelManager:
    """Manages the loading and status of AI models"""
    
    def __init__(self):
        self.models = {}
        self.model_status = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        print("🔄 Loading AI models...")
        
        # Load ViT model
        self._load_vit_model()
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Load Road Classifier
        self._load_road_classifier()
        
        # Load Gemini utilities
        self._load_gemini_utils()
    
    def _load_vit_model(self):
        """Load ViT model"""
        try:
            from ..utils.predict import predict_from_base64, load_vit_model
            
            # Try to actually load the model
            model_loaded = load_vit_model()
            
            self.models['vit'] = predict_from_base64
            self.model_status['vit'] = {
                'loaded': model_loaded,
                'error': None if model_loaded else "Failed to load ViT model",
                'fallback': False
            }
            
            if model_loaded:
                print("✅ ViT model loaded successfully")
            else:
                print("⚠️ ViT model loading failed, prediction function available but will use fallback")
        except Exception as e:
            print(f"❌ ViT model failed to load: {e}")
            traceback.print_exc()
            self.models['vit'] = None
            self.model_status['vit'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True
            }
    
    def _load_yolo_model(self):
        """Load YOLO model"""
        try:
            # Try YOLOv8 first
            try:
                print("Attempting to load YOLOv8 model...")
                from ..utils.yolov8_utils import detect_road_damage, load_yolo_model, YOLO_MODEL, get_model_info
                
                # Attempt to load the YOLOv8 model
                model_loaded = load_yolo_model()
                
                if model_loaded and YOLO_MODEL is not None:
                    self.models['yolo'] = detect_road_damage
                    self.models['yolo_model'] = YOLO_MODEL
                    self.models['yolo_info'] = get_model_info
                    self.model_status['yolo'] = {
                        'loaded': True,
                        'error': None,
                        'fallback': False,
                        'version': 'v8'
                    }
                    print("✅ YOLOv8 model loaded successfully")
                    return
                else:
                    print("⚠️ YOLOv8 model loading failed, trying YOLOv5")
            except ImportError as e:
                print(f"YOLOv8 not available: {e}")
                print("Falling back to YOLOv5")
            except Exception as e:
                print(f"Error loading YOLOv8: {e}")
                print("Falling back to YOLOv5")
            
            # Fall back to YOLOv5
            from ..utils.yolo_utils import detect_road_damage, load_yolo_model, YOLO_MODEL
            
            # Attempt to load the YOLOv5 model
            model_loaded = load_yolo_model()
            
            self.models['yolo'] = detect_road_damage
            self.models['yolo_model'] = YOLO_MODEL
            self.model_status['yolo'] = {
                'loaded': model_loaded and YOLO_MODEL is not None,
                'error': None if (model_loaded and YOLO_MODEL is not None) else "Model loading failed",
                'fallback': not model_loaded or YOLO_MODEL is None,
                'version': 'v5'
            }
            
            if model_loaded and YOLO_MODEL is not None:
                print("✅ YOLOv5 model loaded successfully")
            else:
                print("⚠️ YOLO model loading failed, will use fallback detection")
        except Exception as e:
            print(f"❌ YOLO model failed to load: {e}")
            traceback.print_exc()
            self.models['yolo'] = None
            self.model_status['yolo'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True,
                'version': None
            }
    
    def _load_road_classifier(self):
        """Load Road Classifier"""
        try:
            # Try loading the simple CNN road classifier first
            try:
                from ..utils.simple_road_classifier import validate_road_image, load_road_classifier, ROAD_CLASSIFIER
                print("Using simple CNN road classifier")
                model_loaded = load_road_classifier()
            except ImportError:
                # Fall back to original classifier if simple one not available
                print("Simple road classifier not available, trying original")
                from ..utils.road_classifier import validate_road_image, load_road_classifier, ROAD_CLASSIFIER
                model_loaded = load_road_classifier()
            
            # Set up model in manager
            self.models['road_classifier'] = validate_road_image
            self.model_status['road_classifier'] = {
                'loaded': model_loaded and ROAD_CLASSIFIER is not None,
                'error': None if (model_loaded and ROAD_CLASSIFIER is not None) else "Model loading failed",
                'fallback': not model_loaded or ROAD_CLASSIFIER is None
            }
            print("✅ Road classifier loaded successfully")
        except Exception as e:
            print(f"❌ Road classifier failed to load: {e}")
            self.models['road_classifier'] = None
            self.model_status['road_classifier'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True
            }
    
    def _load_gemini_utils(self):
        """Load Gemini utilities"""
        try:
            from ..utils.gemini_utils import generate_road_damage_summary_from_params
            self.models['gemini'] = generate_road_damage_summary_from_params
            self.model_status['gemini'] = {
                'loaded': True,
                'error': None,
                'fallback': False
            }
            print("✅ Gemini utilities loaded successfully")
        except Exception as e:
            print(f"❌ Gemini utilities failed to load: {e}")
            self.models['gemini'] = None
            self.model_status['gemini'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True
            }
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        return self.model_status.get(model_name, {}).get('loaded', False)
    
    def get_model(self, model_name: str):
        """Get a model by name"""
        return self.models.get(model_name)
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get the status of a model"""
        return self.model_status.get(model_name, {
            'loaded': False,
            'error': 'Model not found',
            'fallback': True
        })
    
    def get_all_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        return self.model_status
    
    def get_missing_models(self) -> list:
        """Get list of missing models"""
        missing = []
        for model_name, status in self.model_status.items():
            if not status['loaded']:
                missing.append(f"{model_name}: {status['error']}")
        return missing


# Global model manager instance
model_manager = ModelManager()
