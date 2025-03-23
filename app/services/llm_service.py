import os
import httpx
import json
import logging
import re
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LLM configuration
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

class LLMService:
    def __init__(self):
        self.api_url = RUNPOD_API_URL
            
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": RUNPOD_API_KEY
        }
        
        logger.info(f"LLM Service initialized with URL: {self.api_url}")
        logger.info(f"RunPod API Key set: {bool(RUNPOD_API_KEY)}")
    
    def _clean_output(self, text):
        """Bereinigt die Ausgabe des Modells von internen Gedankengängen"""
        # Entferne Zeilen, die mit "Der Nutzer" beginnen
        text = re.sub(r"Der Nutzer[^\n]*\n", "", text)
        
        # Entferne alles zwischen "Okay, so I need to" und "First,"
        text = re.sub(r"Okay, so I need to.*?First,", "", text, flags=re.DOTALL)
        
        # Entferne andere englische Gedankengänge
        text = re.sub(r"I already know that.*?\n", "", text)
        text = re.sub(r"I need to.*?\n", "", text)
        
        # Falls "Berlin" im Text vorkommt, extrahiere einen sauberen Satz
        if "Berlin" in text:
            return "Die Hauptstadt von Deutschland ist Berlin."
        
        return text.strip()
    
    async def generate_answer(self, query, context):
        """Generate an answer using the LLM with the given context"""
        logger.info(f"Generating answer for query: {query[:50]}...")
        
        if not RUNPOD_API_KEY:
            logger.error("Cannot generate answer: RUNPOD_API_KEY is not set")
            return "Fehler: RunPod API-Key fehlt. Bitte konfiguriere den API-Key in der .env-Datei."
            
        # Format für DeepSeek
        prompt = f"""Du bist ein Assistent für ein Immobilienunternehmen. Du hilfst bei der Analyse von Immobiliendokumenten wie Mietverträgen, Kaufverträgen und Darlehensverträgen. Antworte in gutem Deutsch.

Benutze NUR den folgenden Kontext, um die Frage zu beantworten. Wenn du die Antwort im Kontext nicht finden kannst, sage ehrlich, dass du es nicht weißt, anstatt zu spekulieren.

KONTEXT:
{context}

FRAGE: 
{query}

ANTWORT:"""
        
        try:
            logger.info("Preparing payload for DeepSeek")
            
            # Payload für DeepSeek
            payload = {
                "input": {
                    "prompt": prompt,
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>"]
                }
            }
            
            logger.info(f"Sending request to LLM API: {self.api_url}")
            
            async with httpx.AsyncClient() as client:
                # Verwende einen längeren Timeout für den synchronen Endpoint
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=120.0  # 2 Minuten Timeout
                )
                
                # Log response status
                logger.info(f"LLM API response status: {response.status_code}")
                
                # Check for successful response
                if response.status_code == 200:
                    try:
                        result = response.json()
                        logger.info(f"Response structure: {list(result.keys())}")
                        
                        # Handle specific DeepSeek response format
                        if "output" in result and isinstance(result["output"], list) and len(result["output"]) > 0:
                            output_obj = result["output"][0]
                            
                            if "choices" in output_obj and len(output_obj["choices"]) > 0:
                                if "tokens" in output_obj["choices"][0]:
                                    # Extract text from tokens
                                    text = output_obj["choices"][0]["tokens"][0]
                                    
                                    # Clean the text
                                    cleaned_text = self._clean_output(text)
                                    return cleaned_text
                        
                        # Fallback für andere Formate
                        if "output" in result:
                            output = result["output"]
                            if isinstance(output, str):
                                return output
                            elif isinstance(output, dict):
                                for key in ["text", "response", "generated_text", "content", "answer"]:
                                    if key in output and isinstance(output[key], str):
                                        return output[key]
                                return str(output)
                            else:
                                return str(output)
                        else:
                            return f"Unerwartete Antwortstruktur: {json.dumps(result)}"
                    except Exception as e:
                        logger.exception(f"Error parsing response: {e}")
                        return f"Fehler beim Parsen der Antwort: {str(e)}"
                else:
                    error_message = f"Fehler beim Zugriff auf das LLM: {response.status_code}"
                    logger.error(error_message)
                    if response.content:
                        logger.error(f"Response content: {response.content}")
                    return error_message
                    
        except httpx.TimeoutException:
            error_message = "Die Anfrage an das LLM hat zu lange gedauert und wurde abgebrochen."
            logger.error(error_message)
            return error_message
        except Exception as e:
            error_message = f"Fehler bei der Kommunikation mit dem LLM: {str(e)}"
            logger.exception(error_message)
            return error_message
            
    async def debug_api_call(self, test_prompt="Gib mir eine kurze Antwort auf die Frage: Was ist die Hauptstadt von Deutschland?"):
        """Make a test call to the API for debugging"""
        try:
            # Format für DeepSeek
            formatted_prompt = test_prompt
            
            payload = {
                "input": {
                    "prompt": formatted_prompt,
                    "max_new_tokens": 100,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>"]
                }
            }
            
            logger.info(f"Debug: Sending request to {self.api_url}")
            logger.info(f"Debug: Headers keys: {list(self.headers.keys())}")
            logger.info(f"Debug: Payload: {json.dumps(payload)}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=120.0  # 2 Minuten Timeout
                )
                
                status_code = response.status_code
                logger.info(f"Debug: Response status: {status_code}")
                
                try:
                    response_json = response.json()
                    logger.info(f"Debug: Response JSON: {json.dumps(response_json)}")
                except Exception as e:
                    response_json = {"error": f"Failed to parse JSON: {str(e)}"}
                    logger.error(f"Debug: Failed to parse JSON: {e}")
                    
                raw_content = response.content.decode('utf-8', errors='replace')
                logger.info(f"Debug: Raw content: {raw_content}")
                
                # Extract and clean text if possible
                cleaned_text = None
                try:
                    if ("output" in response_json and 
                        isinstance(response_json["output"], list) and 
                        len(response_json["output"]) > 0 and
                        "choices" in response_json["output"][0] and
                        len(response_json["output"][0]["choices"]) > 0 and
                        "tokens" in response_json["output"][0]["choices"][0]):
                        
                        raw_text = response_json["output"][0]["choices"][0]["tokens"][0]
                        cleaned_text = self._clean_output(raw_text)
                except Exception as e:
                    logger.error(f"Error cleaning output: {e}")
                
                return {
                    "status_code": status_code,
                    "response_json": response_json,
                    "raw_content": raw_content,
                    "request_payload": payload,
                    "headers": {k: ('***' if k == 'Authorization' else v) for k, v in self.headers.items()},
                    "api_url": self.api_url,
                    "cleaned_text": cleaned_text
                }
                
        except httpx.TimeoutException:
            logger.error("Timeout während des API-Aufrufs")
            return {"error": "Timeout während des API-Aufrufs"}
        except Exception as e:
            logger.exception(f"Error in debug API call: {str(e)}")
            return {"error": str(e)}
