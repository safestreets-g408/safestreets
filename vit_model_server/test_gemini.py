
from gemini_utils import generate_road_damage_summary

def test_summary_generator():
    print("Testing Road Damage Summary Generator\n")
    
    # Test cases
    test_cases = [
        {
            "location": "Hyderabad, Telangana, India",
            "damage_type": "Linear Crack",
            "severity": "Medium",
            "priority": "5"
        },
        {
            "location": "Downtown, Seattle, WA",
            "damage_type": "Pothole",
            "severity": "High",
            "priority": "8"
        },
        {
            "location": "Oxford Street, London, UK",
            "damage_type": "Road Settlement",
            "severity": "Low",
            "priority": "3"
        }
    ]
    
    # Run tests
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"- Location: {case['location']}")
        print(f"- Damage Type: {case['damage_type']}")
        print(f"- Severity: {case['severity']}")
        print(f"- Priority: {case['priority']}")
        
        summary = generate_road_damage_summary(
            case["location"], 
            case["damage_type"], 
            case["severity"], 
            case["priority"]
        )
        
        print("\nGenerated Summary:")
        print(summary)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    test_summary_generator()
