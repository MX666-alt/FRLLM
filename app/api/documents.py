from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.document import Document, DocumentQuery, SearchResponse, SearchResult
from app.services.dropbox_service import DropboxService
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService
from app.api.auth import get_current_user
from app.models.auth import User
from typing import List
import logging
from urllib.parse import unquote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
dropbox_service = DropboxService()
qdrant_service = QdrantService()
llm_service = LLMService()

@router.get("/list", response_model=List[Document])
async def list_documents(path: str = "", current_user: User = Depends(get_current_user)):
    """List all documents in the given path"""
    documents = dropbox_service.list_files(path)
    return [Document(**doc) for doc in documents]

@router.get("/get/{path:path}", response_model=Document)
async def get_document(path: str, current_user: User = Depends(get_current_user)):
    """Get document content by path"""
    # Decode the path if it's URL-encoded
    path = unquote(path)
    logger.info(f"Getting document content for path: {path}")
    
    content = dropbox_service.download_file(f"/{path}")
    if not content:
        logger.error(f"Document not found or could not be downloaded: {path}")
        raise HTTPException(status_code=404, detail="Document not found or could not be downloaded")
    
    # Get document information
    doc_info = dropbox_service.list_files(f"/{path.rsplit('/', 1)[0] if '/' in path else ''}")
    doc = next((d for d in doc_info if d["name"] == path.rsplit('/', 1)[-1]), None)
    
    if not doc:
        # Construct document info manually
        logger.info(f"Document info not found, constructing manually for: {path}")
        doc = {
            "id": path,
            "name": path.rsplit('/', 1)[-1],
            "path": f"/{path}",
            "type": "file"
        }
    
    # Return document with content
    document = Document(**doc)
    document.content = content
    logger.info(f"Successfully retrieved document: {document.name}")
    return document

@router.post("/index/{path:path}")
async def index_document(path: str, current_user: User = Depends(get_current_user)):
    """Index a document in the vector database"""
    try:
        # Decode the path if it's URL-encoded
        path = unquote(path)
        logger.info(f"Indexing document: {path}")
        
        # Get document content
        content = dropbox_service.download_file(f"/{path}")
        if not content:
            logger.error(f"Document not found or empty: {path}")
            raise HTTPException(status_code=404, detail="Document not found or could not be downloaded")
        
        # Get document information
        doc_name = path.rsplit('/', 1)[-1]
        logger.info(f"Document name: {doc_name}, content length: {len(content)} characters")
        
        # Special handling for very small or large documents
        if len(content) < 10:
            logger.warning(f"Document content too short to index: {path}, length: {len(content)}")
            raise HTTPException(status_code=400, detail="Document content too short to index")
        
        # Index document
        try:
            success = qdrant_service.index_document(
                document_id=path,
                document_path=f"/{path}",
                document_name=doc_name,
                document_content=content
            )
            
            if not success:
                logger.error(f"Failed to index document: {path}")
                raise HTTPException(status_code=500, detail="Failed to index document")
            
            logger.info(f"Document indexed successfully: {path}")
            return {"message": "Document indexed successfully", "document_id": path}
        except Exception as e:
            logger.exception(f"Error during document indexing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during document indexing: {str(e)}")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/check-indexes")
async def check_indexes(current_user: User = Depends(get_current_user)):
    """Check what documents are indexed in Qdrant"""
    try:
        logger.info("Checking indexed documents")
        indexes = qdrant_service.list_indexed_documents()
        logger.info(f"Found {len(indexes)} indexed documents")
        return {"indexes": indexes}
    except Exception as e:
        logger.exception(f"Error checking indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking indexes: {str(e)}")

@router.post("/search", response_model=SearchResponse)
async def search_documents(query: DocumentQuery, current_user: User = Depends(get_current_user)):
    """Search documents with semantic search"""
    logger.info(f"Searching documents for query: {query.query}")
    
    try:
        # First check if any documents are indexed
        logger.info("Checking if documents are indexed")
        indexed_docs = qdrant_service.list_indexed_documents()
        if not indexed_docs:
            logger.warning("No documents are indexed in Qdrant")
            return SearchResponse(
                results=[],
                answer="Es wurden keine Dokumente gefunden. Bitte indiziere zuerst einige Dokumente über die Dokumentenseite."
            )
        
        logger.info(f"Found {len(indexed_docs)} indexed documents: {indexed_docs}")
        
        # Search for relevant document chunks
        logger.info(f"Searching for query: {query.query}")
        search_results = qdrant_service.search(query.query, query.top_k)
        logger.info(f"Found {len(search_results)} search results")
        
        if not search_results:
            logger.warning("No search results found")
            return SearchResponse(
                results=[],
                answer=f"Es wurden keine relevanten Dokumente zu deiner Anfrage '{query.query}' gefunden. Bitte versuche eine andere Formulierung oder stelle sicher, dass relevante Dokumente indiziert wurden."
            )
        
        # Extract relevant context
        context = "\n\n".join([f"[Dokument: {result.payload['document_name']}]\n{result.payload['text']}" for result in search_results])
        logger.info(f"Context length: {len(context)} characters")
        
        # Log the context for debugging
        logger.info(f"Context for LLM: {context[:500]}...")
        
        # Try to generate answer with LLM (always attempt even with zero results)
        try:
            logger.info("Calling LLM to generate answer")
            answer = await llm_service.generate_answer(query.query, context if search_results else "")
            logger.info(f"LLM answer: {answer[:100]}...")
        except Exception as e:
            logger.exception(f"Error generating LLM answer: {str(e)}")
            # Detailliertere Fehlermeldung erstellen
            answer = f"Bei der Verarbeitung Ihrer Anfrage '{query.query}' ist ein Fehler aufgetreten.\n\n"
            answer += "Hier sind die gefundenen relevanten Informationen ohne KI-Analyse:\n\n"
            # Füge die rohen Suchergebnisse hinzu
            if search_results:
                for i, result in enumerate(search_results[:5], 1):
                    doc_name = result.payload.get("document_name", "Unbekanntes Dokument")
                    text = result.payload.get("text", "").strip()
                    answer += f"**Dokument {i}: {doc_name}**\n\n{text}\n\n"
            else:
                answer += "Es wurden keine relevanten Dokumente gefunden."
        
        # Prepare response with all search results
        results = []
        for result in search_results:
            document = Document(
                id=result.payload["document_id"],
                name=result.payload["document_name"],
                path=result.payload["document_path"],
                type="file",
                content=result.payload["text"]
            )
            results.append(SearchResult(document=document, score=result.score))
        
        return SearchResponse(results=results, answer=answer)
    except Exception as e:
        logger.exception(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Suche: {str(e)}")

@router.get("/debug-llm")
async def debug_llm(current_user: User = Depends(get_current_user), prompt: str = Query("Dies ist ein Testprompt. Bitte antworte mit 'Hallo Welt'.")):
    """Debug endpoint for LLM API"""
    logger.info("Debug LLM API call")
    result = await llm_service.debug_api_call(prompt)
    return result

@router.get("/system-check")
async def system_check(current_user: User = Depends(get_current_user)):
    """Führt einen vollständigen Systemcheck durch"""
    results = {}
    
    # Teste Dropbox-Verbindung
    try:
        dropbox_status = dropbox_service.get_token_info()
        results["dropbox"] = {"status": "ok", "details": dropbox_status}
    except Exception as e:
        results["dropbox"] = {"status": "error", "message": str(e)}
    
    # Teste Qdrant-Verbindung
    try:
        indexed_docs = qdrant_service.list_indexed_documents()
        results["qdrant"] = {"status": "ok", "indexed_documents": len(indexed_docs), "doc_ids": indexed_docs[:5] if indexed_docs else []}
    except Exception as e:
        results["qdrant"] = {"status": "error", "message": str(e)}
    
    # Teste LLM mit einfacher Anfrage
    try:
        llm_result = await llm_service.debug_api_call("Was ist 2+2?")
        raw_response = llm_result.get("cleaned_text", "Keine Antwort")
        results["llm"] = {
            "status": "ok", 
            "response": raw_response,
            "api_url": llm_result.get("api_url", ""),
            "status_code": llm_result.get("status_code", 0)
        }
    except Exception as e:
        results["llm"] = {"status": "error", "message": str(e)}
    
    return results
