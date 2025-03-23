import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_runpod_api():
    """Test the RunPod API connection"""
    # Get configuration from environment
    api_url = os.getenv("RUNPOD_API_URL")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    print(f"API URL: {api_url}")
    print(f"API Key set: {bool(api_key)}")
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    
    # Simple test payload for DeepSeek
    payload = {
        "input": {
            "prompt": "Was ist die Hauptstadt von Deutschland?",
            "max_new_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["<|im_end|>"]
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
    test_runpod_api()
