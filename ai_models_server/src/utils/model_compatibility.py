"""
Model Compatibility Utilities
Handles various model formats and architectures with robust fallback mechanisms
"""
import torch
import torch.nn as nn
import os
import time
from typing import Optional, Dict, Any, Union, List
from collections import OrderedDict


class ModelAdapter:
    """Adapter class to handle different model formats and architectures"""
    
    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        self.fallback_used = False
        
    def load_model(self, fallback_model: Optional[nn.Module] = None) -> bool:
        """
        Load model with multiple fallback strategies
        
        Args:
            fallback_model: A fallback model to use if loading fails
            
        Returns:
            bool: True if any model was loaded successfully
        """
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            if fallback_model:
                self.model = fallback_model
                self.fallback_used = True
                self.is_loaded = True
                return True
            return False
        
        start_time = time.time()
        
        # Strategy 1: Try TorchScript loading
        if self._try_torchscript_load():
            print(f"TorchScript model loaded in {time.time() - start_time:.2f}s")
            return True
            
        # Strategy 2: Try state dict loading with architecture matching
        if self._try_state_dict_load(fallback_model):
            print(f"State dict model loaded in {time.time() - start_time:.2f}s")
            return True
            
        # Strategy 3: Try Hugging Face format
        if self._try_huggingface_load():
            print(f"Hugging Face model loaded in {time.time() - start_time:.2f}s")
            return True
            
        # Strategy 4: Use fallback model
        if fallback_model:
            self.model = fallback_model
            self.fallback_used = True
            self.is_loaded = True
            print(f"Using fallback model in {time.time() - start_time:.2f}s")
            return True
            
        return False
    
    def _try_torchscript_load(self) -> bool:
        """Try to load as TorchScript model"""
        try:
            self.model = torch.jit.load(self.model_path, map_location="cpu")
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"TorchScript loading failed: {e}")
            return False
    
    def _try_state_dict_load(self, fallback_model: Optional[nn.Module]) -> bool:
        """Try to load as state dict with architecture matching"""
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            if fallback_model:
                # Try to match architectures
                if self._architectures_compatible(state_dict, fallback_model.state_dict()):
                    fallback_model.load_state_dict(state_dict, strict=False)
                    self.model = fallback_model
                    self.model.eval()
                    self.is_loaded = True
                    return True
                else:
                    print("Architecture mismatch - using fallback with random weights")
                    self.model = fallback_model
                    self.fallback_used = True
                    self.is_loaded = True
                    return True
            return False
            
        except Exception as e:
            print(f"State dict loading failed: {e}")
            return False
    
    def _try_huggingface_load(self) -> bool:
        """Try to load as Hugging Face model"""
        try:
            # This is a placeholder - would need specific HF model loading
            # based on the model type
            return False
        except Exception as e:
            print(f"Hugging Face loading failed: {e}")
            return False
    
    def _architectures_compatible(self, state_dict1: Dict, state_dict2: Dict) -> bool:
        """Check if two state dicts have compatible architectures"""
        try:
            # Check if keys match
            keys1 = set(state_dict1.keys())
            keys2 = set(state_dict2.keys())
            
            # Allow some flexibility in key matching
            common_keys = keys1.intersection(keys2)
            total_keys = keys1.union(keys2)
            
            # If more than 50% of keys match, consider compatible
            compatibility_ratio = len(common_keys) / len(total_keys)
            
            if compatibility_ratio > 0.5:
                # Check shapes of common keys
                for key in common_keys:
                    if state_dict1[key].shape != state_dict2[key].shape:
                        print(f"Shape mismatch for {key}: {state_dict1[key].shape} vs {state_dict2[key].shape}")
                        return False
                return True
            
            return False
            
        except Exception as e:
            print(f"Architecture compatibility check failed: {e}")
            return False
    
    def get_model(self) -> Optional[nn.Module]:
        """Get the loaded model"""
        return self.model if self.is_loaded else None
    
    def is_fallback(self) -> bool:
        """Check if fallback model is being used"""
        return self.fallback_used


class RobustModelLoader:
    """Robust model loader with multiple fallback strategies"""
    
    @staticmethod
    def load_with_fallback(model_path: str, 
                          fallback_model: Optional[nn.Module] = None,
                          model_type: str = "auto") -> tuple[Optional[nn.Module], bool]:
        """
        Load model with comprehensive fallback strategy
        
        Args:
            model_path: Path to model file
            fallback_model: Fallback model to use if loading fails
            model_type: Type hint for model loading
            
        Returns:
            Tuple of (model, is_fallback)
        """
        adapter = ModelAdapter(model_path, model_type)
        
        if adapter.load_model(fallback_model):
            return adapter.get_model(), adapter.is_fallback()
        
        return None, True
    
    @staticmethod
    def create_mock_outputs(num_classes: int, 
                           image_size: tuple = (224, 224),
                           include_bbox: bool = False) -> Dict[str, Any]:
        """
        Create mock outputs for testing/fallback
        
        Args:
            num_classes: Number of output classes
            image_size: Size of input image
            include_bbox: Whether to include bounding box
            
        Returns:
            Dictionary with mock prediction results
        """
        import random
        
        # Create mock probabilities
        probs = [random.random() for _ in range(num_classes)]
        prob_sum = sum(probs)
        probs = [p / prob_sum for p in probs]
        
        predicted_idx = probs.index(max(probs))
        confidence = max(probs)
        
        result = {
            'predicted_class_idx': predicted_idx,
            'confidence': confidence,
            'all_probabilities': probs,
            'fallback': True
        }
        
        if include_bbox:
            # Create a reasonable mock bounding box
            w, h = image_size
            box_w = w // 4
            box_h = h // 4
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            result['bbox'] = [x1, y1, x1 + box_w, y1 + box_h]
        
        return result
