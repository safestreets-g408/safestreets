import google.generativeai as genai
import os

# Configure the API key from environment variable or fallback to a default for development
API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyDc8TSD_GZFjMB1XkW310CoBFMT3qQFyss")

# Configure the API key
genai.configure(api_key=API_KEY)

# Load the Gemini 1.5 Flash model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def generate_road_damage_summary(location, damage_type, severity, priority):
    """
    Generate a summary for a road damage report using Google's Gemini 1.5 Flash model.
    
    Args:
        location (str): The location of the damage
        damage_type (str): The type of damage (e.g., Linear Crack, Pothole)
        severity (str): Severity level (e.g., Low, Medium, High)
        priority (str): Priority level (1-10)
        
    Returns:
        str: A generated summary for the road damage report
    """
    try:
        prompt = f"""
        Write a 100 word road damage report summary based on the following details:
        - Location: {location}
        - Damage Type: {damage_type}
        - Severity: {severity}
        - Priority: {priority}
        
        The summary should be professional, concise, and suitable for a municipal road maintenance report.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Road damage at {location}: {damage_type} with {severity} severity. Priority level: {priority}."
