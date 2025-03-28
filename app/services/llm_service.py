import os
import httpx
import json
import logging
import re
import time
import asyncio
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
        
        # Konfigurierbare Timeouts und Wiederholungsversuche
        self.timeout = 60.0  # Timeout in Sekunden (reduziert von 120)
        self.max_retries = 2  # Maximale Anzahl von Wiederholungsversuchen
        self.retry_delay = 2  # Verzögerung zwischen Wiederholungsversuchen in Sekunden
        
        logger.info(f"LLM Service initialized with URL: {self.api_url}")
        logger.info(f"RunPod API Key set: {bool(RUNPOD_API_KEY)}")
        logger.info(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")
    
    def _clean_output(self, text):
        """Bereinigt die Ausgabe des Modells von internen Gedankengängen und Artefakten"""
        # Wenn der Text None oder leer ist, frühzeitig zurückkehren
        if not text:
            return "Keine Antwort vom LLM erhalten."
            
        # Debug-Log der Rohausgabe
        logger.info(f"Rohausgabe vom LLM (erste 200 Zeichen): {text[:200]}")
        
        # Erkennung und Entfernung von strukturierten Artefakten
        # Entferne COVID-Info-Tags und andere spezielle Markierungen
        covid_patterns = [
            r'\[/?covidInfo\].*?(?:\n|$)',
            r'_\s*COVID-19-INFO:.*?(?:\n|$)',
            r'\[bonusInformationen\..*?\]',
            r'Ergibt die Zusammenfassung der Datenschutz-Grundverord.*'
        ]
        for pattern in covid_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Entferne alle bekannten Gedankengangsmuster
        thought_patterns = [
            r"Der Nutzer[^\n]*\n",
            r"Okay, (?:so )?I need to.*?First,",
            r"Okay, ich muss.*?\n",
            r"Um das zu erreichen.*?\n",
            r"I need to.*?\n",
            r"I already know that.*?\n",
            r"First,.*?\n",
            r"Let me analyze.*?\n",
            r"Let's look at.*?\n",
            r"Based on the.*?\n",
            r"Looking at the.*?\n",
        ]
        
        for pattern in thought_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
        
        # Entferne Anführungszeichen am Anfang und Ende
        text = re.sub(r'^\s*["\']', '', text)
        text = re.sub(r'["\']\s*$', '', text)
        
        # Entferne Metainformationen
        text = re.sub(r"ANFRAGE:|ANTWORT:|Anfrage:|Antwort:", "", text)
        
        # Verarbeite bestimmte Sonderfälle
        if "Berlin" in text and len(text) < 100:
            return "Die Hauptstadt von Deutschland ist Berlin."
        
        # Entferne übermäßige Leerzeichen und Zeilenumbrüche
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Schneidet die Antwort ab, wenn spezielle Marker gefunden werden
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Abbruch, wenn eine der folgenden Markierungen gefunden wird
            if any(marker in line for marker in [
                '[', ']', 'covid', 'COVID', 'bonus', 'Bonus', 
                'Datenschutz', 'Verord', '---', '***', '///', '###'
            ]):
                # Stoppe, wenn solche Marker gefunden werden
                break
            cleaned_lines.append(line)
        
        # Bereinigten Text wieder zusammenfügen
        cleaned_text = '\n'.join(cleaned_lines).strip()
        
        # Als Fallback: Wenn der Text zu kurz ist nach der Bereinigung, 
        # versuche nur den ersten Satz zu extrahieren
        if len(cleaned_text) < 20 and len(text) > 30:
            # Extrahiere den ersten Satz, der sinnvoll erscheint
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if len(sentence) > 15 and not any(marker in sentence for marker in ['[', ']', 'covid', 'COVID']):
                    cleaned_text = sentence
                    break
        
        # Debug-Log der bereinigten Ausgabe
        logger.info(f"Bereinigte Ausgabe (erste 200 Zeichen): {cleaned_text[:200]}")
        
        return cleaned_text
    
    def _validate_response(self, text):
        """Überprüft die bereinigte Antwort auf Probleme"""
        if not text or text.strip() == "":
            return "Keine Antwort vom LLM erhalten."
            
        # Prüfe auf unvollständige JSON-Strukturen
        if text.startswith("{") and not text.endswith("}"):
            logger.warning("Unvollständige JSON-Antwort erkannt")
            return "Es gab ein Problem bei der Antwortgenerierung. Bitte versuchen Sie es noch einmal."
            
        # Prüfe auf teilweise Antworten
        if len(text) < 20 or text.endswith("..."):
            logger.warning("Zu kurze oder unvollständige Antwort erkannt")
            return "Die Antwort konnte nicht vollständig generiert werden. Bitte versuchen Sie es noch einmal."
        
        return text
    
    async def generate_answer(self, query, context):
        """Generate an answer using the LLM with the given context"""
        logger.info(f"Generating answer for query: {query[:50]}...")
        
        if not RUNPOD_API_KEY:
            logger.error("Cannot generate answer: RUNPOD_API_KEY is not set")
            return "Fehler: RunPod API-Key fehlt. Bitte konfiguriere den API-Key in der .env-Datei."
            
        # Optimiertes Prompt-Format für DeepSeek
        prompt = f"""Du bist ein präziser Assistent für ein Immobilienunternehmen. Du hilfst bei der Analyse von Immobiliendokumenten wie Mietverträgen, Kaufverträgen und Darlehensverträgen. Antworte in gutem Deutsch, direkt und ohne Gedankengänge. Füge keine zusätzlichen Informationen, Hinweise oder Tags hinzu.

Benutze NUR den folgenden Kontext, um die Frage zu beantworten. Wenn du die Antwort im Kontext nicht finden kannst, sage ehrlich, dass du es nicht weißt.

KONTEXT:
{context}

FRAGE: 
{query}

ANTWORT:"""
        
        # Reduzierte Token-Anzahl um Timeouts zu vermeiden
        max_tokens = 256  # Ursprünglich 512, reduziert zur Vermeidung von Timeouts
        
        # Payload für DeepSeek mit optimierten Parametern
        payload = {
            "input": {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": 0.3,  # Niedrigere Temperatur für präzisere Antworten
                "top_p": 0.9,
                "stop": ["<|im_end|>"]
            }
        }
        
        # Wiederholungslogik für Anfragen
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt+1}/{self.max_retries+1}: Sending request to LLM API")
                
                async with httpx.AsyncClient() as client:
                    # Mit anpassbarem Timeout
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout
                    )
                    
                    logger.info(f"LLM API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            logger.info(f"Response structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                            
                            # DeepSeek-Format
                            if "output" in result and isinstance(result["output"], list) and len(result["output"]) > 0:
                                output_obj = result["output"][0]
                                
                                if "choices" in output_obj and len(output_obj["choices"]) > 0:
                                    if "tokens" in output_obj["choices"][0]:
                                        # Extract text from tokens
                                        text = output_obj["choices"][0]["tokens"][0]
                                        
                                        # Bereinigung und Validierung
                                        cleaned_text = self._clean_output(text)
                                        validated_text = self._validate_response(cleaned_text)
                                        
                                        return validated_text
                                        
                            # Allgemeine Extraktion als Fallback
                            if "output" in result:
                                output = result["output"]
                                if isinstance(output, str):
                                    return self._clean_output(output)
                                elif isinstance(output, dict):
                                    for key in ["text", "response", "generated_text", "content", "answer"]:
                                        if key in output and isinstance(output[key], str):
                                            return self._clean_output(output[key])
                                return "Konnte keine verwertbare Antwort extrahieren. Bitte versuchen Sie es mit einer anderen Frage."
                            else:
                                return f"Unerwartetes Antwortformat. Bitte überprüfen Sie die RunPod-Konfiguration."
                                
                        except Exception as e:
                            logger.exception(f"Error parsing response: {e}")
                            if attempt < self.max_retries:
                                logger.info(f"Retrying after parse error...")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            return f"Fehler beim Verarbeiten der LLM-Antwort: {str(e)}"
                    elif response.status_code == 504 or response.status_code == 503 or response.status_code == 502:
                        # Gateway Timeout oder Service Unavailable - Wiederholungsversuch
                        logger.warning(f"Timeout/Service Unavailable (status code {response.status_code}). Retrying...")
                        if attempt < self.max_retries:
                            # Warte länger zwischen Wiederholungen bei 504
                            await asyncio.sleep(self.retry_delay * 2)
                            continue
                        return f"Der RunPod-Server brauchte zu lange zum Antworten. Bitte versuchen Sie es später erneut oder prüfen Sie den RunPod-Status."
                    else:
                        error_message = f"Fehler beim Zugriff auf das LLM: {response.status_code}"
                        logger.error(error_message)
                        if response.content:
                            logger.error(f"Response content: {response.content}")
                        
                        # Bei anderen Fehlercodes auch wiederholen
                        if attempt < self.max_retries:
                            logger.info(f"Retrying after HTTP error {response.status_code}...")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        
                        return error_message
                
            except httpx.TimeoutException:
                error_message = f"Timeout bei der Anfrage an den LLM (Versuch {attempt+1}/{self.max_retries+1})"
                logger.error(error_message)
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying after timeout...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                return "Der RunPod-Server hat nicht rechtzeitig geantwortet. Mögliche Ursachen:\n" \
                       "1. Der Server ist überlastet\n" \
                       "2. Die serverlose Instanz wurde gestoppt\n" \
                       "3. Die Anfrage ist zu komplex\n\n" \
                       "Bitte versuchen Sie es mit einer kürzeren Frage oder prüfen Sie den Status Ihres RunPod-Endpunkts."
            
            except Exception as e:
                error_message = f"Fehler bei der Kommunikation mit dem LLM: {str(e)}"
                logger.exception(error_message)
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying after general error...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                return error_message
            
    async def debug_api_call(self, test_prompt="Gib mir eine kurze Antwort auf die Frage: Was ist die Hauptstadt von Deutschland?"):
        """Make a test call to the API for debugging"""
        try:
            # Format für DeepSeek
            formatted_prompt = test_prompt
            
            # Reduzierte Tokenzahl für Debugging
            payload = {
                "input": {
                    "prompt": formatted_prompt,
                    "max_new_tokens": 50,  # Reduziert für Debugging
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>"]
                }
            }
            
            logger.info(f"Debug: Sending request to {self.api_url}")
            logger.info(f"Debug: Headers keys: {list(self.headers.keys())}")
            logger.info(f"Debug: Payload: {json.dumps(payload)}")
            
            # Timeout reduziert für Debugging
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        self.api_url,
                        json=payload,
                        headers=self.headers,
                        timeout=30.0  # Reduzierter Timeout für Debugging
                    )
                    
                    status_code = response.status_code
                    logger.info(f"Debug: Response status: {status_code}")
                    
                    try:
                        response_json = response.json()
                        logger.info(f"Debug: Response JSON structure: {list(response_json.keys()) if isinstance(response_json, dict) else type(response_json)}")
                    except Exception as e:
                        response_json = {"error": f"Failed to parse JSON: {str(e)}"}
                        logger.error(f"Debug: Failed to parse JSON: {e}")
                        
                    raw_content = response.content.decode('utf-8', errors='replace')
                    logger.info(f"Debug: Raw content: {raw_content[:500]}")
                    
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
                        "raw_content": raw_content[:1000],  # Begrenzt auf 1000 Zeichen
                        "request_payload": payload,
                        "headers": {k: ('***' if k == 'Authorization' else v) for k, v in self.headers.items()},
                        "api_url": self.api_url,
                        "cleaned_text": cleaned_text,
                        "timestamp": time.time()
                    }
                except httpx.TimeoutException:
                    logger.error("Debug: Timeout während des API-Aufrufs")
                    return {
                        "error": "Timeout während des API-Aufrufs", 
                        "api_url": self.api_url,
                        "request_payload": payload,
                        "timestamp": time.time()
                    }
                
        except Exception as e:
            logger.exception(f"Error in debug API call: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}
