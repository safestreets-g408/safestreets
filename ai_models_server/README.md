# AI Models Server - Clean Architecture

A well-organized Flask-based AI server for road damage detection and analysis.

## 🏗️ Project Structure

```
ai_models_server/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
│
├── src/                  # Source code
│   ├── core/            # Core application logic
│   │   ├── config.py    # Configuration management
│   │   ├── app_factory.py # Flask app factory
│   │   └── startup.py   # Startup utilities
│   ├── api/             # API layer
│   │   ├── routes.py    # Route definitions
│   │   ├── handlers.py  # Route handlers
│   │   └── error_handlers.py # Error handling
│   ├── models/          # AI models management
│   │   └── model_manager.py # Model loading and management
│   └── utils/           # Utility functions
│       ├── response_utils.py # Response formatting
│       ├── validation_utils.py # Request validation
│       └── image_utils.py    # Image processing utilities
│
├── models/              # AI model files
│   ├── vit_model.pth    # Vision Transformer model
│   ├── yolo_model.pt    # YOLO object detection model
│   ├── cnn_road_classifier_scripted.pt # Road classifier
│   └── class_names.txt  # Class labels
│
├── scripts/             # Utility scripts
│   ├── download_models.sh # Download model files
│   └── start_server.sh    # Server startup script
│
├── tests/               # Test files
│   ├── test_server.py   # Server tests
│   └── test_gemini.py   # Gemini API tests
│
├── static/              # Static files
│   └── uploads/         # Uploaded images
│
└── legacy/              # Legacy code (backup)
    └── ...              # Old files for reference
```

## 🚀 Features

- **Vision Transformer (ViT)**: Road damage classification
- **YOLO Object Detection**: Real-time damage detection  
- **Road Surface Classifier**: Validates road surface images
- **AI Summary Generation**: Automated report generation using Gemini AI
- **Fallback System**: Graceful degradation when models are unavailable
- **Clean Architecture**: Modular, maintainable codebase
- **Comprehensive Error Handling**: Robust error management
- **CORS Support**: Cross-origin resource sharing

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a .env file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the server:
```bash
python app.py
```

## Summary Generation

The `/generate-summary` endpoint accepts a JSON payload with the following parameters:
- `location`: Location of the damage (e.g., "Hyderabad, Telangana, India")
- `damageType`: Type of damage (e.g., "Linear Crack", "Pothole")
- `severity`: Severity level (e.g., "Low", "Medium", "High")
- `priority`: Priority level (e.g., "1" to "10")

Example response:
```json
{
  "summary": "A medium severity linear crack has been identified in Hyderabad, Telangana, India. This damage has been assigned a priority level of 5, indicating moderate urgency. Immediate assessment is recommended to prevent further deterioration of the road surface. Traffic in the affected area remains manageable, but the damage requires attention within the next maintenance cycle.",
  "success": true,
  "message": "Summary generated successfully"
}
```
