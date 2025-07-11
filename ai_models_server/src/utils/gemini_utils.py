"""
Gemini AI Utilities
Handles AI-powered road damage summary generation using Google's Gemini API
"""
import os
import base64
import json
import traceback
from typing import Dict, Any, Optional, List
import requests
from PIL import Image
import io
import time
import hashlib
import functools

from ..core.config import GOOGLE_API_KEY


# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Simple cache for gemini responses
_gemini_cache = {}
_cache_ttl = 3600  # 1 hour cache time

def cache_result(func):
    """Cache decorator for expensive API calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function arguments
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
        
        current_time = time.time()
        # Check if result is in cache and not expired
        if cache_key in _gemini_cache and _gemini_cache[cache_key]['expire_time'] > current_time:
            print(f"Cache hit for {func.__name__}")
            return _gemini_cache[cache_key]['result']
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Cache the result
        _gemini_cache[cache_key] = {
            'result': result,
            'expire_time': current_time + _cache_ttl
        }
        
        # Clean up old cache entries
        for k in list(_gemini_cache.keys()):
            if _gemini_cache[k]['expire_time'] < current_time:
                del _gemini_cache[k]
                
        return result
    return wrapper


@cache_result
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
            timeout=60  # Increased timeout
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


@cache_result
def generate_road_damage_summary_from_params(location: str, damage_type: str, severity: str, priority: str) -> str:
    """
    Generate a comprehensive, professional road damage summary from parameters
    
    Args:
        location: Location of the damage
        damage_type: Type of damage detected
        severity: Severity level
        priority: Priority level
        
    Returns:
        Professionally formatted summary string
    """
    try:
        print(f"Generating professional summary for {location}, {damage_type}, {severity}, priority {priority}")
        start_time = time.time()
        
        # Create a professional summary without requiring image data
        confidence = 0.7  # Default confidence for text-based summaries
        
        # Map severity to confidence
        severity_confidence_map = {
            'Low': 0.6,
            'Medium': 0.7,
            'High': 0.8,
            'Critical': 0.9
        }
        
        confidence = severity_confidence_map.get(severity, 0.7)
        
        # Get detailed damage description based on type
        damage_descriptions = {
            'pothole': "depression in the road surface resulting from material failure and erosion, characterized by crumbling pavement and circular/irregular shape",
            'alligator_crack': "interconnected cracks forming a pattern resembling alligator skin, indicating structural failure in the pavement layers",
            'longitudinal_crack': "parallel cracks along the direction of travel, often resulting from poor joint construction or thermal cycling",
            'transverse_crack': "perpendicular cracks across the road surface, typically caused by thermal stress and pavement shrinkage",
            'surface_crack': "superficial fractures in the road surface that have not yet penetrated to deeper layers",
            'erosion': "deterioration of road material due to water infiltration and environmental factors, resulting in material loss",
            'D00': "longitudinal crack affecting surface integrity with potential for water infiltration",
            'D10': "transverse crack indicating thermal stress patterns in pavement structure",
            'D20': "alligator cracking demonstrating advanced fatigue in pavement layers",
            'D30': "edge cracking along road boundaries with potential for shoulder degradation",
            'D40': "pothole formation with complete material failure requiring prompt intervention",
            'D50': "surface deterioration with initial indications of pavement breakdown"
        }
        
        # Get implications based on damage type and severity
        implications = {
            'High': {
                'safety': "Presents significant safety hazards to road users, particularly motorcycles and smaller vehicles.",
                'progression': "Will rapidly deteriorate without intervention, especially during adverse weather conditions.",
                'cost': "Delay in repairs will likely result in substantially higher rehabilitation costs."
            },
            'Medium': {
                'safety': "Moderate safety concern that may impact vehicle handling and control under certain conditions.",
                'progression': "Expected to worsen gradually, accelerating with exposure to moisture and freeze-thaw cycles.",
                'cost': "Preventative maintenance now would be more cost-effective than future reactive repairs."
            },
            'Low': {
                'safety': "Minimal immediate safety concerns but warrants monitoring for progression.",
                'progression': "Likely to deteriorate slowly under normal conditions and usage patterns.",
                'cost': "Can be addressed during routine maintenance operations."
            }
        }
        
        # Customize repair approaches based on damage type
        repair_approaches = {
            'pothole': "mill and fill technique with proper compaction and edge sealing",
            'alligator_crack': "full-depth patching or asphalt overlay depending on structural assessment",
            'longitudinal_crack': "crack sealing with appropriate elastomeric material to prevent water infiltration",
            'transverse_crack': "routing and sealing to address thermal movement concerns",
            'surface_crack': "application of sealant treatment to prevent moisture penetration",
            'erosion': "stabilization of base material and surface restoration with drainage improvements",
            'D00': "crack sealing with performance-grade sealant",
            'D10': "crack filling and possible thermal movement accommodation",
            'D20': "mill and overlay or full-depth reclamation based on structural evaluation",
            'D30': "edge reinforcement and shoulder stabilization",
            'D40': "full-depth patching with proper compaction and material selection",
            'D50': "surface treatment and preventative maintenance application"
        }
        
        # Generate professional summary with sections
        summary = f"## Road Damage Assessment Report\n\n"
        summary += f"**Location:** {location}\n"
        summary += f"**Damage Type:** {damage_type}\n"
        summary += f"**Severity Level:** {severity}\n"
        summary += f"**Priority Classification:** {priority}\n\n"
        
        # Damage description
        summary += f"**Damage Analysis:**\n"
        damage_desc = damage_descriptions.get(damage_type.lower(), f"Identified road surface damage consistent with {damage_type} classification")
        summary += f"The inspection has identified {damage_desc}. "
        
        # Add severity-specific details
        severity_impl = implications.get(severity, implications.get('Medium'))
        summary += f"{severity_impl['safety']} {severity_impl['progression']}\n\n"
        
        # Add recommendations section
        summary += f"**Recommended Actions:**\n"
        
        if severity in ['High', 'Critical']:
            summary += f"1. **Immediate Intervention Required:** Schedule emergency repair within 24-72 hours.\n"
            summary += f"2. **Temporary Mitigation:** Implement hazard signage and possible traffic control measures.\n"
            if damage_type.lower() in ['pothole', 'alligator_crack', 'd40', 'd20']:
                summary += f"3. **Safety Measures:** Consider temporary road plates or cold patch application as interim solution.\n"
            summary += f"4. **Repair Methodology:** Utilize {repair_approaches.get(damage_type.lower(), 'appropriate repair techniques')} to ensure durability of repair.\n"
        elif severity == 'Medium':
            summary += f"1. **Scheduled Maintenance:** Include in upcoming maintenance cycle (recommended within 2-4 weeks).\n"
            summary += f"2. **Monitoring Protocol:** Establish regular inspection schedule to track progression.\n"
            summary += f"3. **Repair Methodology:** Apply {repair_approaches.get(damage_type.lower(), 'standard repair techniques')} to address damage effectively.\n"
            summary += f"4. **Preventative Measures:** Consider adjacent area treatment to prevent similar failures.\n"
        else:
            summary += f"1. **Planned Maintenance:** Incorporate into routine maintenance schedule.\n"
            summary += f"2. **Documentation:** Record in asset management system for future reference.\n"
            summary += f"3. **Monitoring Protocol:** Include in next scheduled inspection cycle.\n"
            summary += f"4. **Preventative Strategy:** Consider surface treatment during next scheduled maintenance.\n"
        
        # Budget and planning implications
        summary += f"\n**Planning Implications:**\n"
        summary += f"- {severity_impl['cost']}\n"
        summary += f"- Recommended repair approach using {repair_approaches.get(damage_type.lower(), 'standard techniques')} should be evaluated for cost-effectiveness.\n"
        summary += f"- Consider adjacent infrastructure inspection to identify potential related issues.\n\n"
        
        # Final note
        summary += f"This assessment is based on standardized classification protocols. Professional engineering inspection is recommended to confirm findings and finalize repair specifications for {location}."
        
        elapsed_time = time.time() - start_time
        print(f"Professional summary generated in {elapsed_time:.2f} seconds")
        
        return summary
        
    except Exception as e:
        print(f"Error generating professional summary: {e}")
        traceback.print_exc()
        return f"Road damage report for {location}: {damage_type} with {severity} severity. Priority: {priority}. Professional inspection recommended."
