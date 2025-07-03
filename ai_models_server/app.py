"""
AI Models Server - Clean and Organized
Main Flask application entry point
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.app_factory import create_app
from src.core.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from src.core.startup import print_startup_info

def main():
    """Main application entry point"""
    app = create_app()
    print_startup_info()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

if __name__ == "__main__":
    main()
