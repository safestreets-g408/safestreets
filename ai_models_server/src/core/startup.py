"""
Startup utilities for the AI Models Server
"""
from .config import FLASK_HOST, FLASK_PORT, API_VERSION
from ..models.model_manager import ModelManager


def print_startup_info():
    """Print server startup information"""
    print("="*60)
    print(f"🚀 AI Models Server v{API_VERSION} Starting")
    print("="*60)
    print(f"Server URL: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"Health Check: http://{FLASK_HOST}:{FLASK_PORT}/health")
    print(f"Models Status: http://{FLASK_HOST}:{FLASK_PORT}/models-status")
    print("-"*60)
    print("Available Endpoints:")
    print(f"  📍 ViT Prediction: http://{FLASK_HOST}:{FLASK_PORT}/predict")
    print(f"  📍 YOLO Detection: http://{FLASK_HOST}:{FLASK_PORT}/detect-yolo")  
    print(f"  📍 Road Classification: http://{FLASK_HOST}:{FLASK_PORT}/classify-road")
    print(f"  📍 Summary Generation: http://{FLASK_HOST}:{FLASK_PORT}/generate-summary")
    print(f"  📍 YOLO Info: http://{FLASK_HOST}:{FLASK_PORT}/yolo-info")
    print("-"*60)
    
    # Initialize model manager and print status
    model_manager = ModelManager()
    all_status = model_manager.get_all_model_status()
    
    print("Model Status:")
    for model_name, status in all_status.items():
        status_icon = "✅" if status['loaded'] else "❌"
        fallback_text = " (Fallback Mode)" if status['fallback'] else ""
        print(f"  {status_icon} {model_name.title()}: {'Loaded' if status['loaded'] else 'Not Available'}{fallback_text}")
    
    missing_models = model_manager.get_missing_models()
    if missing_models:
        print("-"*60)
        print("⚠️  Missing Models:")
        for model_info in missing_models:
            print(f"  • {model_info}")
        print("\n💡 Run './scripts/download_models.sh' to download missing models")
        print("   Or place them manually in the models/ directory")
        print("   Services will use fallback modes until models are available")
    
    print("="*60)
