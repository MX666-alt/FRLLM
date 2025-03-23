import os
import httpx
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

async def test_runpod_api():
    """Test the RunPod API connection asynchronously"""
    print(f"API URL: {RUNPOD_API_URL}")
    print(f"API Key set: {bool(RUNPOD_API_KEY)}")
    
    # Verschiedene Header-Formate testen
    headers_variations = [
        # Standard: API Key ohne Bearer Präfix
        {
            "Content-Type": "application/json",
            "Authorization": RUNPOD_API_KEY
        },
        # Mit Bearer Präfix
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
    ]
    
    # Verschiedene Payload-Formate für DeepSeek-Coder testen
    payload_variations = [
        # Format 1: Standardformat
        {
            "input": {
                "prompt": "Was ist die Hauptstadt von Deutschland?",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9
            }
        },
        # Format 2: Alternative Parameter
        {
            "input": {
                "prompt": "Was ist die Hauptstadt von Deutschland?",
                "max_new_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9
            }
        },
        # Format 3: Minimale Parameter
        {
            "input": {
                "prompt": "Was ist die Hauptstadt von Deutschland?"
            }
        }
    ]
    
    print("\nTesting various combinations...")
    
    for header_idx, headers in enumerate(headers_variations):
        for payload_idx, payload in enumerate(payload_variations):
            try:
                print(f"\nTest {header_idx+1}.{payload_idx+1}:")
                print(f"Headers: Authorization: {headers['Authorization'][:15]}...")
                print(f"Payload: {json.dumps(payload)[:100]}...")
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        RUNPOD_API_URL,
                        json=payload,
                        headers=headers,
                        timeout=60.0
                    )
                    
                    print(f"Response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("Success! Response structure:")
                        print(f"Keys: {list(result.keys())}")
                        
                        # Extract output if available
                        if "output" in result:
                            output = result["output"]
                            if isinstance(output, str):
                                print(f"Output (first 100 chars): {output[:100]}...")
                            elif isinstance(output, dict):
                                print(f"Output structure: {list(output.keys())}")
                        
                        return True
                    else:
                        print(f"Error response: {response.text}")
            except Exception as e:
                print(f"Exception during API call: {e}")
    
    # Zusätzliche Hilfestellung für API-Fehler
    print("\nAPI Troubleshooting:")
    print("1. Überprüfe die RunPod-Endpunkt-URL:")
    print(f"   - Aktuelle URL: {RUNPOD_API_URL}")
    print("   - Sollte in Format: https://api.runpod.ai/v2/[ENDPOINT-ID]/runsync sein")
    print("2. Überprüfe den API-Key:")
    print(f"   - Aktueller Key (Anfang): {RUNPOD_API_KEY[:10]}...")
    print(f"   - Länge: {len(RUNPOD_API_KEY)} Zeichen")
    print("3. Überprüfe auf RunPod.io, ob der Serverless-Endpunkt aktiv ist")
    
    return False

if __name__ == "__main__":
    result = asyncio.run(test_runpod_api())
    if result:
        print("\nRunPod API test successful!")
    else:
        print("\nAll RunPod API test variations failed. Please check your RunPod configuration.")
