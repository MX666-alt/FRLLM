import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_runpod_api():
    """Test the RunPod API connection"""
    # Get configuration from environment
    api_url = os.getenv("RUNPOD_API_URL")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    print(f"API URL: {api_url}")
    print(f"API Key set: {bool(api_key)}")
    
    # Add Bearer prefix if not present
    if api_key and not api_key.startswith("Bearer "):
        api_key = f"Bearer {api_key}"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    
    # Simple test payload
    payload = {
        "input": {
            "prompt": "<s>[INST] Was ist die Hauptstadt von Deutschland? [/INST]",
            "max_new_tokens": 100,
            "temperature": 0.5
        }
    }
    
    print("Sending test request to RunPod API...")
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"Exception during API call: {e}")
        return False

if __name__ == "__main__":
    check_runpod_api()
