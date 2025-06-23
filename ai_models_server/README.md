# VIT Model Server

This server provides damage classification and reporting features for the SafeStreets platform.

## Features

- Road damage classification using the VIT model
- Image annotation with damage type and bounding boxes
- AI-powered damage report summary generation using Google Gemini 1.5 Flash model

## API Endpoints

### Health Check
- `GET /health` - Check if the server is running

### Damage Classification
- `POST /predict` - Send an image and get back classification results

### Report Summary Generation
- `POST /generate-summary` - Generate a professional summary of road damage based on report details

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
