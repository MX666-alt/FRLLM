import requests
import json

# RunPod API Konfiguration
API_URL = "https://api.runpod.ai/v2/cme6i62b6ovy2s/runsync"  # Korrigierter Endpunkt
API_KEY = "rpa_WNX3C1IIWUCLKNGK47DESSNOG7WGL024XSQSQB50nk5cni"  # Korrigierter API-Key

# Testfunktion
def test_api():
    # Standard Payload
    payload = {
        "input": {
            "prompt": "Was ist die Hauptstadt von Deutschland?",
            "max_tokens": 100,
            "temperature": 0.5
        }
    }
    
    # Header Varianten testen
    headers_variations = [
        # Variante 1: API Key ohne "Bearer"
        {
            "Content-Type": "application/json",
            "Authorization": API_KEY
        },
        # Variante 2: Mit "Bearer" Präfix
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    ]
    
    for i, headers in enumerate(headers_variations, 1):
        print(f"\nTest {i}: Mit Headers: {headers['Authorization'][:15]}...")
        
        try:
            response = requests.post(
                API_URL, 
                headers=headers,
                json=payload
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("Erfolg! Antwort:")
                result = response.json()
                print(json.dumps(result, indent=2))
                return True
            else:
                print(f"Fehler: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")
    
    return False

if __name__ == "__main__":
    print("RunPod API Test")
    print("===============")
    success = test_api()
    
    if success:
        print("\nTest erfolgreich!")
    else:
        print("\nTest fehlgeschlagen. Überprüfe die API-URL und den API-Key.")
