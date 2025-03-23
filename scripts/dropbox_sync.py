#!/usr/bin/env python3
import os
import sys
import time
import logging
import json
import requests
from datetime import datetime
from pathlib import Path
import httpx

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/immobilien-rag/logs/dropbox_sync.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("dropbox_sync")

# Pfad zu einer Statusdatei, in der der letzte Sync gespeichert wird
STATUS_FILE = "/opt/immobilien-rag/data/sync_status.json"

def ensure_dirs_exist():
    """Stelle sicher, dass alle benötigten Verzeichnisse existieren"""
    Path("/opt/immobilien-rag/logs").mkdir(parents=True, exist_ok=True)
    Path("/opt/immobilien-rag/data").mkdir(parents=True, exist_ok=True)

def load_status():
    """Lade den letzten Sync-Status aus der Statusdatei"""
    if not os.path.exists(STATUS_FILE):
        return {
            "last_sync": None,
            "indexed_documents": [],
            "last_full_sync": None
        }
    
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden der Statusdatei: {e}")
        return {
            "last_sync": None,
            "indexed_documents": [],
            "last_full_sync": None
        }

def save_status(status):
    """Speichere den aktuellen Sync-Status in der Statusdatei"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Statusdatei: {e}")

def get_all_documents():
    """Hole eine Liste aller Dokumente aus Dropbox"""
    logger.info("Hole alle Dokumente aus Dropbox...")
    
    documents = []
    
    # Hole Authentifizierungs-Token
    token = get_auth_token()
    if not token:
        logger.error("Konnte kein Authentifizierungs-Token erhalten")
        return documents
    
    try:
        # Rekursive Funktion, um alle Dokumente aus Dropbox zu holen
        documents = get_documents_recursive("", token, [])
        logger.info(f"Insgesamt {len(documents)} Dokumente gefunden")
        return documents
    except Exception as e:
        logger.error(f"Fehler beim Holen der Dokumente: {e}")
        return documents

def get_documents_recursive(path, token, documents):
    """Rekursiv alle Dokumente in einem Pfad und seinen Unterordnern holen"""
    try:
        logger.info(f"Verarbeite Pfad: {path}")
        
        response = requests.get(
            f"http://localhost:8000/api/documents/list?path={path}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            logger.error(f"Fehler beim Holen der Dokumente für Pfad {path}: {response.status_code}")
            return documents
        
        path_documents = response.json()
        logger.info(f"{len(path_documents)} Dokumente/Ordner in Pfad {path} gefunden")
        
        for doc in path_documents:
            if doc['type'] == 'folder':
                # Rekursiv in Ordner vordringen
                documents = get_documents_recursive(doc['path'], token, documents)
            elif doc['type'] == 'file':
                # Datei zur Liste hinzufügen
                if doc['path'].lower().endswith(('.pdf', '.txt', '.md', '.docx', '.doc')):
                    documents.append(doc)
                    logger.debug(f"Dokument hinzugefügt: {doc['path']}")
                else:
                    logger.debug(f"Überspringe nicht unterstützten Dateityp: {doc['path']}")
        
        return documents
    except Exception as e:
        logger.error(f"Fehler bei get_documents_recursive für Pfad {path}: {e}")
        return documents

def get_indexed_documents():
    """Hole eine Liste aller indexierten Dokumente"""
    logger.info("Hole indexierte Dokumente...")
    
    token = get_auth_token()
    if not token:
        logger.error("Konnte kein Authentifizierungs-Token erhalten")
        return []
    
    try:
        response = requests.get(
            "http://localhost:8000/api/documents/check-indexes",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            logger.error(f"Fehler beim Holen der indexierten Dokumente: {response.status_code}")
            return []
        
        data = response.json()
        return data.get("indexes", [])
    except Exception as e:
        logger.error(f"Fehler beim Holen der indexierten Dokumente: {e}")
        return []

def get_auth_token():
    """Authentifizierung und Token-Erhalt"""
    try:
        # Login-Daten aus Umgebungsvariablen oder direkt
        username = "immobilien_admin"
        password = "S3cur3P@ssw0rd2025"
        
        response = requests.post(
            "http://localhost:8000/api/auth/token",
            data={"username": username, "password": password}
        )
        
        if response.status_code != 200:
            logger.error(f"Authentifizierung fehlgeschlagen: {response.status_code}")
            return None
        
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        logger.error(f"Fehler bei der Authentifizierung: {e}")
        return None

def index_document(doc_path, token):
    """Indiziere ein einzelnes Dokument"""
    try:
        # Entferne führenden Slash falls vorhanden
        if doc_path.startswith('/'):
            doc_path = doc_path[1:]
            
        logger.info(f"Indiziere Dokument: {doc_path}")
        
        # Realer API-Aufruf
        response = requests.post(
            f"http://localhost:8000/api/documents/index/{doc_path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={}
        )
        
        if response.status_code != 200:
            logger.error(f"Fehler beim Indizieren von {doc_path}: {response.status_code}")
            if response.status_code == 401:
                logger.error("Authentifizierungsfehler - Token ist möglicherweise abgelaufen")
            try:
                error_details = response.json()
                logger.error(f"Fehlerdetails: {error_details}")
            except:
                logger.error(f"Antworttext: {response.text}")
            return False
        
        logger.info(f"Dokument erfolgreich indiziert: {doc_path}")
        return True
    except Exception as e:
        logger.error(f"Exception beim Indizieren von {doc_path}: {e}")
        return False

def perform_sync():
    """Führe eine vollständige Synchronisierung durch"""
    logger.info("Starte Synchronisierung...")
    
    # Lade aktuellen Status
    status = load_status()
    
    # Hole Authentifizierungs-Token
    token = get_auth_token()
    if not token:
        logger.error("Konnte kein Authentifizierungs-Token erhalten, breche ab")
        return False
    
    # Hole alle Dokumente aus Dropbox
    all_documents = get_all_documents()
    dropbox_paths = [doc["path"] for doc in all_documents]
    logger.info(f"{len(dropbox_paths)} Dokumente in Dropbox gefunden")
    
    # Hole alle indexierten Dokumente
    indexed_docs = get_indexed_documents()
    logger.info(f"{len(indexed_docs)} Dokumente im Index gefunden")
    
    # Identifiziere neue und gelöschte Dokumente
    docs_to_index = [path for path in dropbox_paths if path not in indexed_docs]
    docs_to_remove = [path for path in indexed_docs if path not in dropbox_paths]
    
    logger.info(f"{len(docs_to_index)} neue Dokumente zu indizieren")
    logger.info(f"{len(docs_to_remove)} Dokumente zu entfernen")
    
    # Indiziere neue Dokumente
    newly_indexed = []
    for doc_path in docs_to_index:
        doc_details = next((doc for doc in all_documents if doc["path"] == doc_path), None)
        if doc_details:
            success = index_document(doc_path, token)
            if success:
                newly_indexed.append(doc_path)
                # Ein kurze Pause, um den Server nicht zu überlasten
                time.sleep(1)
    
    # Aktualisiere Status
    status["last_sync"] = datetime.now().isoformat()
    status["indexed_documents"] = indexed_docs + newly_indexed
    status["last_full_sync"] = datetime.now().isoformat()
    save_status(status)
    
    logger.info(f"Synchronisierung abgeschlossen. {len(newly_indexed)} Dokumente indiziert.")
    return True

if __name__ == "__main__":
    try:
        ensure_dirs_exist()
        logger.info("Starte Dropbox Synchronisierung...")
        success = perform_sync()
        if success:
            logger.info("Dropbox Synchronisierung erfolgreich abgeschlossen.")
            sys.exit(0)
        else:
            logger.error("Dropbox Synchronisierung fehlgeschlagen.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei der Synchronisierung: {e}")
        sys.exit(1)
