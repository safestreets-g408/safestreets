"""
Gemini AI Utilities
Handles AI-powered road damage summary generation using Google's Gemini API
"""
import os
import base64
import json
from typing import Dict, Any, Optional, List
import requests
from PIL import Image
import io

from ..core.config import GOOGLE_API_KEY


# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def generate_road_damage_summary(
    image_base64: str,
    damage_type: str,
    confidence: float,
    bbox: Optional[List[int]] = None,
    additional_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive road damage summary using Gemini AI
    
    Args:
        image_base64: Base64 encoded image string
        damage_type: Type of damage detected
        confidence: Confidence score of the detection
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        additional_context: Additional context information
        
    Returns:
        Dictionary containing AI-generated summary
    """
    if not GOOGLE_API_KEY:
        return {
            'error': 'Google API key not configured',
            'summary': f'Road damage detected: {damage_type}',
            'priority': 'Medium',
            'severity': 'Unknown',
            'recommended_action': 'Manual inspection required',
            'fallback': True
        }
    
    try:
        # Prepare the prompt
        prompt = create_damage_analysis_prompt(
            damage_type, confidence, bbox, additional_context
        )
        
        # Prepare the request payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1000,
                "responseMimeType": "application/json"
            }
        }
        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Try to parse JSON response
                try:
                    analysis = json.loads(content)
                    return format_gemini_response(analysis)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    return {
                        'summary': content,
                        'priority': determine_priority(damage_type, confidence),
                        'severity': determine_severity(damage_type, confidence),
                        'recommended_action': get_recommended_action(damage_type),
                        'fallback': False
                    }
            else:
                return create_fallback_response(damage_type, confidence)
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return create_fallback_response(damage_type, confidence)
            
    except Exception as e:
        print(f"Error generating Gemini summary: {e}")
        return create_fallback_response(damage_type, confidence)


def create_damage_analysis_prompt(
    damage_type: str,
    confidence: float,
    bbox: Optional[List[int]] = None,
    additional_context: Optional[Dict] = None
) -> str:
    """
    Create a structured prompt for Gemini AI analysis
    
    Args:
        damage_type: Type of damage detected
        confidence: Confidence score of the detection
        bbox: Bounding box coordinates
        additional_context: Additional context information
        
    Returns:
        Formatted prompt string
    """
    # Ensure confidence is a float for formatting
    try:
        confidence_float = float(confidence)
    except (ValueError, TypeError):
        confidence_float = 0.7  # Default confidence
    
    prompt = f"""
Analyze this road damage image and provide a comprehensive assessment. The AI model has detected {damage_type} with {confidence_float:.2%} confidence.

Please provide your analysis in the following JSON format:

{{
    "damage_analysis": {{
        "primary_damage_type": "string",
        "secondary_damage_types": ["string"],
        "severity_level": "Low|Medium|High|Critical",
        "affected_area_size": "Small|Medium|Large|Extensive",
        "damage_description": "detailed description of the damage visible in the image"
    }},
    "risk_assessment": {{
        "safety_risk": "Low|Medium|High|Critical",
        "vehicle_damage_risk": "Low|Medium|High|Critical",
        "progression_risk": "Low|Medium|High|Critical",
        "weather_vulnerability": "Low|Medium|High|Critical"
    }},
    "priority_classification": {{
        "repair_priority": "Low|Medium|High|Urgent",
        "maintenance_urgency": "Routine|Scheduled|Immediate|Emergency",
        "cost_estimate": "Low|Medium|High|Very High"
    }},
    "recommendations": {{
        "immediate_actions": ["string"],
        "short_term_solutions": ["string"],
        "long_term_recommendations": ["string"],
        "monitoring_requirements": ["string"]
    }},
    "summary": "A concise 2-3 sentence summary of the damage and recommended actions"
}}

Additional context:
- Detection confidence: {confidence_float:.2%}
- Model prediction: {damage_type}
"""
    
    if bbox:
        prompt += f"\n- Damage location: Bounding box coordinates {bbox}"
    
    if additional_context:
        prompt += f"\n- Additional context: {additional_context}"
    
    prompt += """

Please analyze the image carefully and provide accurate, actionable insights based on what you can actually see in the image. Focus on practical road maintenance and safety considerations."""
    
    return prompt


def format_gemini_response(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format Gemini AI response into standardized structure
    
    Args:
        analysis: Raw response from Gemini API
        
    Returns:
        Formatted response dictionary
    """
    try:
        return {
            'summary': analysis.get('summary', 'Road damage detected'),
            'damage_type': analysis.get('damage_analysis', {}).get('primary_damage_type', 'Unknown'),
            'severity': analysis.get('damage_analysis', {}).get('severity_level', 'Medium'),
            'priority': analysis.get('priority_classification', {}).get('repair_priority', 'Medium'),
            'safety_risk': analysis.get('risk_assessment', {}).get('safety_risk', 'Medium'),
            'recommended_action': analysis.get('priority_classification', {}).get('maintenance_urgency', 'Scheduled'),
            'immediate_actions': analysis.get('recommendations', {}).get('immediate_actions', []),
            'long_term_recommendations': analysis.get('recommendations', {}).get('long_term_recommendations', []),
            'detailed_analysis': analysis,
            'fallback': False
        }
    except Exception as e:
        print(f"Error formatting Gemini response: {e}")
        return {
            'summary': 'Road damage detected - analysis incomplete',
            'severity': 'Medium',
            'priority': 'Medium',
            'recommended_action': 'Manual inspection required',
            'fallback': True
        }


def create_fallback_response(damage_type: str, confidence: float) -> Dict[str, Any]:
    """
    Create a fallback response when Gemini API is unavailable
    
    Args:
        damage_type: Type of damage detected
        confidence: Confidence score of the detection
        
    Returns:
        Fallback response dictionary
    """
    # Ensure confidence is a float
    try:
        confidence_float = float(confidence)
    except (ValueError, TypeError):
        confidence_float = 0.7  # Default confidence
    
    return {
        'summary': f'{damage_type} detected with {confidence_float:.1%} confidence. Professional inspection recommended.',
        'damage_type': damage_type,
        'severity': determine_severity(damage_type, confidence_float),
        'priority': determine_priority(damage_type, confidence_float),
        'safety_risk': determine_safety_risk(damage_type),
        'recommended_action': get_recommended_action(damage_type),
        'immediate_actions': get_immediate_actions(damage_type),
        'long_term_recommendations': get_long_term_recommendations(damage_type),
        'fallback': True
    }


def determine_severity(damage_type: str, confidence: float) -> str:
    """Determine damage severity based on type and confidence"""
    severity_map = {
        'pothole': 'High',
        'alligator_crack': 'High',
        'D40': 'High',  # Pothole
        'D20': 'High',  # Alligator crack
        'longitudinal_crack': 'Medium',
        'transverse_crack': 'Medium',
        'D00': 'Medium',  # Longitudinal crack
        'D10': 'Medium',  # Transverse crack
        'surface_crack': 'Low',
        'erosion': 'Medium',
        'D50': 'Low',   # Surface crack
        'D60': 'Medium'  # Erosion
    }
    
    base_severity = severity_map.get(damage_type.lower(), 'Medium')
    
    # Adjust based on confidence
    if confidence < 0.5:
        return 'Low'
    elif confidence > 0.85:
        severity_levels = ['Low', 'Medium', 'High', 'Critical']
        current_idx = severity_levels.index(base_severity)
        return severity_levels[min(current_idx + 1, len(severity_levels) - 1)]
    
    return base_severity


def determine_priority(damage_type: str, confidence: float) -> str:
    """Determine repair priority based on damage type"""
    priority_map = {
        'pothole': 'High',
        'alligator_crack': 'High',
        'D40': 'High',
        'D20': 'High',
        'longitudinal_crack': 'Medium',
        'transverse_crack': 'Medium',
        'D00': 'Medium',
        'D10': 'Medium',
        'surface_crack': 'Low',
        'erosion': 'Medium',
        'D50': 'Low',
        'D60': 'Medium'
    }
    
    return priority_map.get(damage_type.lower(), 'Medium')


def determine_safety_risk(damage_type: str) -> str:
    """Determine safety risk based on damage type"""
    risk_map = {
        'pothole': 'High',
        'alligator_crack': 'High',
        'D40': 'High',
        'D20': 'High',
        'longitudinal_crack': 'Medium',
        'transverse_crack': 'Medium',
        'D00': 'Medium',
        'D10': 'Medium',
        'surface_crack': 'Low',
        'erosion': 'Medium',
        'D50': 'Low',
        'D60': 'Medium'
    }
    
    return risk_map.get(damage_type.lower(), 'Medium')


def get_recommended_action(damage_type: str) -> str:
    """Get recommended action based on damage type"""
    action_map = {
        'pothole': 'Immediate repair required',
        'alligator_crack': 'Immediate repair required',
        'D40': 'Immediate repair required',
        'D20': 'Immediate repair required',
        'longitudinal_crack': 'Scheduled maintenance',
        'transverse_crack': 'Scheduled maintenance',
        'D00': 'Scheduled maintenance',
        'D10': 'Scheduled maintenance',
        'surface_crack': 'Routine maintenance',
        'erosion': 'Scheduled maintenance',
        'D50': 'Routine maintenance',
        'D60': 'Scheduled maintenance'
    }
    
    return action_map.get(damage_type.lower(), 'Professional inspection required')


def get_immediate_actions(damage_type: str) -> List[str]:
    """Get immediate actions based on damage type"""
    if damage_type.lower() in ['pothole', 'alligator_crack', 'd40', 'd20']:
        return [
            'Place warning signs around the area',
            'Restrict heavy vehicle access if possible',
            'Schedule emergency repair within 24-48 hours',
            'Monitor for safety hazards'
        ]
    else:
        return [
            'Document damage location and extent',
            'Add to maintenance schedule',
            'Monitor for progression'
        ]


def get_long_term_recommendations(damage_type: str) -> List[str]:
    """Get long-term recommendations based on damage type"""
    return [
        'Implement preventive maintenance program',
        'Consider road surface treatment',
        'Evaluate underlying structural issues',
        'Plan for comprehensive road rehabilitation if needed'
    ]


def test_gemini_connection() -> Dict[str, Any]:
    """Test the connection to Gemini API"""
    if not GOOGLE_API_KEY:
        return {
            'connected': False,
            'error': 'Google API key not configured'
        }
    
    try:
        # Simple test request
        payload = {
            "contents": [{
                "parts": [{"text": "Hello, respond with 'API working' if you can read this."}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 20
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                'connected': True,
                'status': 'API connection successful'
            }
        else:
            return {
                'connected': False,
                'error': f'API error: {response.status_code}'
            }
            
    except Exception as e:
        return {
            'connected': False,
            'error': f'Connection failed: {str(e)}'
        }


def generate_road_damage_summary_from_params(location: str, damage_type: str, severity: str, priority: str) -> str:
    """
    Generate road damage summary from parameters (wrapper for API compatibility)
    
    Args:
        location: Location of the damage
        damage_type: Type of damage detected
        severity: Severity level
        priority: Priority level
        
    Returns:
        Generated summary string
    """
    try:
        # Create a fallback summary without requiring image data
        confidence = 0.7  # Default confidence for text-based summaries
        
        # Map severity to confidence
        severity_confidence_map = {
            'Low': 0.6,
            'Medium': 0.7,
            'High': 0.8,
            'Critical': 0.9
        }
        
        confidence = severity_confidence_map.get(severity, 0.7)
        
        # Generate summary
        summary = f"Road damage report for {location}: {damage_type} detected with {severity} severity. "
        summary += f"Priority level: {priority}. "
        
        # Add recommendations based on damage type and severity
        if severity in ['High', 'Critical']:
            summary += "Immediate attention required. "
        elif severity == 'Medium':
            summary += "Scheduled maintenance recommended. "
        else:
            summary += "Monitor condition and plan routine maintenance. "
        
        # Add safety recommendations
        if damage_type.lower() in ['pothole', 'alligator_crack', 'd40', 'd20']:
            summary += "Safety hazard - consider temporary signage or barriers. "
        
        summary += f"Professional inspection and repair estimate recommended for {location}."
        
        return summary
        
    except Exception as e:
        print(f"Error generating parameter-based summary: {e}")
        return f"Road damage report for {location}: {damage_type} with {severity} severity. Priority: {priority}. Professional inspection recommended."
