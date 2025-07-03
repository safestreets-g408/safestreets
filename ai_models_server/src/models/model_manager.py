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
            from ..utils.predict import predict_from_base64
            self.models['vit'] = predict_from_base64
            self.model_status['vit'] = {
                'loaded': True,
                'error': None,
                'fallback': False
            }
            print("✅ ViT model loaded successfully")
        except Exception as e:
            print(f"❌ ViT model failed to load: {e}")
            self.models['vit'] = None
            self.model_status['vit'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True
            }
    
    def _load_yolo_model(self):
        """Load YOLO model"""
        try:
            from ..utils.yolo_utils import detect_road_damage, YOLO_MODEL
            self.models['yolo'] = detect_road_damage
            self.models['yolo_model'] = YOLO_MODEL
            self.model_status['yolo'] = {
                'loaded': YOLO_MODEL is not None,
                'error': None if YOLO_MODEL is not None else "Model not found",
                'fallback': YOLO_MODEL is None
            }
            if YOLO_MODEL is not None:
                print("✅ YOLO model loaded successfully")
            else:
                print("❌ YOLO model not found, using fallback")
        except Exception as e:
            print(f"❌ YOLO model failed to load: {e}")
            self.models['yolo'] = None
            self.model_status['yolo'] = {
                'loaded': False,
                'error': str(e),
                'fallback': True
            }
    
    def _load_road_classifier(self):
        """Load Road Classifier"""
        try:
            from ..utils.road_classifier import validate_road_image
            self.models['road_classifier'] = validate_road_image
            self.model_status['road_classifier'] = {
                'loaded': True,
                'error': None,
                'fallback': False
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
