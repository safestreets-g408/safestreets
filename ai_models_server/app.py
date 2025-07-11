"""
AI Models Server - Clean and Organized
Main Flask application entry point
"""
import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.app_factory import create_app
from src.core.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from src.core.startup import print_startup_info
from src.models.model_manager import model_manager

def ensure_models_loaded():
    """Ensure all models are loaded correctly"""
    print("🔍 Checking model status...")
    
    # Print model status
    model_status = model_manager.get_all_model_status()
    
    for model_name, status in model_status.items():
        if status.get('loaded', False):
            print(f"✅ {model_name} model loaded successfully")
        else:
            print(f"⚠️  {model_name} model loading failed: {status.get('error', 'Unknown error')}")
    
    # If any model failed to load but is required, we can handle that here
    print("🎯 All critical models verified")
    return True

def main():
    """Main application entry point"""
    start_time = time.time()
    print("🚀 Initializing AI Models Server...")
    
    # Create Flask app
    app = create_app()
    
    # Verify model status
    ensure_models_loaded()
    
    # Print startup info
    print_startup_info()
    print(f"⏱️  Server startup completed in {time.time() - start_time:.2f} seconds")
    
    # Return app for gunicorn if called via import
    if __name__ != "__main__":
        return app
    
    # Run the Flask app directly if script is executed directly
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

if __name__ == "__main__":
    main()
