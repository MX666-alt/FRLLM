from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import logging
from urllib.parse import unquote
from jose import jwt, JWTError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dropbox.exceptions import AuthError
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import API routers
from app.api import auth, documents
from app.services.auth import create_access_token, get_admin_user, ACCESS_TOKEN_EXPIRE_MINUTES
from app.services.dropbox_service import DropboxService
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService
from app.models.auth import User

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

# Create FastAPI app
app = FastAPI(title="Immobilien-Dokument-RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

# Create services
dropbox_service = DropboxService()
llm_service = LLMService()
qdrant_service = QdrantService()

# Helper functions
def get_path_parts(path):
    """Split a path into parts for breadcrumb navigation"""
    if not path:
        return []
    
    parts = []
    current = ""
    
    for part in path.strip("/").split("/"):
        if current:
            current = f"{current}/{part}"
        else:
            current = part
            
        parts.append({
            "name": part,
            "path": f"/{current}"
        })
    
    return parts

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Dashboard page requested")
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    logger.info("Login page requested")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request, 
    username: str = Form(...), 
    password: str = Form(...), 
    redirect_url: Optional[str] = Form(None)
):
    logger.info(f"Login attempt for user: {username}")
    admin_user = get_admin_user()
    
    # Verify user credentials
    if username != admin_user["username"] or password != admin_user["hashed_password"]:
        logger.warning(f"Invalid login attempt for user: {username}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Ungültiger Benutzername oder Passwort"
        })
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": admin_user["username"]}, expires_delta=access_token_expires
    )
    
    logger.info(f"Successful login for user: {username}")
    
    # Determine redirect URL (default to home page)
    redirect_to = redirect_url if redirect_url else "/"
    logger.info(f"Redirecting after login to: {redirect_to}")
    
    # Create redirect response
    response = RedirectResponse(url=redirect_to, status_code=303)
    
    # Set token in both cookie and localStorage via script
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=False,  # Set to false so JavaScript can access it
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    # Also set a script tag to store token in localStorage
    token_script = f"""
    <script>
    localStorage.setItem('access_token', '{access_token}');
    window.location.href = '{redirect_to}';
    </script>
    """
    
    return HTMLResponse(content=token_script)

@app.get("/logout")
async def logout():
    logger.info("User logged out")
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    # Also add script to clear localStorage
    token_script = f"""
    <script>
    localStorage.removeItem('access_token');
    localStorage.removeItem('redirect_after_login');
    window.location.href = '/login';
    </script>
    """
    
    return HTMLResponse(content=token_script)

@app.get("/dropbox-status")
async def dropbox_status():
    """Check Dropbox connection status"""
    try:
        status = dropbox_service.get_token_info()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error checking Dropbox status: {str(e)}")
        return JSONResponse(
            content={"valid": False, "error": str(e)},
            status_code=500
        )

@app.get("/debug-dropbox")
async def debug_dropbox():
    """Debug endpoint for Dropbox connection"""
    try:
        # Get token info
        token_info = dropbox_service.get_token_info()
        
        # Get root files directly
        root_files = dropbox_service.debug_list_root()
        
        # Get files using list_files method
        files_method = dropbox_service.list_files("")
        
        return JSONResponse({
            "token_info": token_info,
            "root_files_direct": root_files,
            "files_from_method": files_method,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.exception(f"Error in debug endpoint: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/debug-llm")
async def debug_llm():
    """Debug endpoint for LLM API"""
    try:
        result = await llm_service.debug_api_call()
        
        # Füge eine menschenlesbare Interpretation hinzu
        if "cleaned_text" in result and result["cleaned_text"]:
            result["human_readable_answer"] = f"Bereinigte Antwort: {result['cleaned_text']}"
        elif "response_json" in result and "output" in result["response_json"]:
            try:
                # Versuche die Antwort zu extrahieren
                output = result["response_json"]["output"]
                if isinstance(output, list) and len(output) > 0:
                    if "choices" in output[0] and len(output[0]["choices"]) > 0:
                        if "tokens" in output[0]["choices"][0]:
                            raw_text = output[0]["choices"][0]["tokens"][0]
                            result["human_readable_answer"] = f"Rohe Antwort: {raw_text}"
            except Exception as e:
                result["extraction_error"] = str(e)
        
        return JSONResponse(result)
    except Exception as e:
        logger.exception(f"Error in LLM debug endpoint: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/indexed-documents", response_class=HTMLResponse)
async def indexed_documents_page(request: Request):
    """UI page to view indexed documents"""
    logger.info("Indexed documents page requested")
    
    try:
        # Get indexed documents
        indexed_docs = qdrant_service.list_indexed_documents()
        logger.info(f"Found {len(indexed_docs)} indexed documents")
        
        # Format documents for display
        formatted_docs = []
        for doc_id in indexed_docs:
            doc_name = doc_id.split("/")[-1] if "/" in doc_id else doc_id
            formatted_docs.append({
                "id": doc_id,
                "name": doc_name,
                "path": doc_id
            })
        
        return templates.TemplateResponse("indexed_documents.html", {
            "request": request,
            "documents": formatted_docs,
            "documents_count": len(formatted_docs)
        })
    except Exception as e:
        logger.exception(f"Error loading indexed documents: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "status_code": 500,
            "detail": f"Fehler beim Laden der indizierten Dokumente: {str(e)}"
        })

@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, path: str = ""):
    logger.info(f"Documents page requested for path: '{path}'")
    
    try:
        # Decode the path if it's URL-encoded
        if path:
            path = unquote(path)
            logger.info(f"Decoded path: '{path}'")
        
        # Test Dropbox connection
        token_info = dropbox_service.get_token_info()
        if not token_info.get("valid", False):
            logger.error("Dropbox token is invalid")
            return templates.TemplateResponse("dropbox_error.html", {
                "request": request,
                "error_details": token_info.get("error", "Token is invalid")
            })
        
        # Get documents from Dropbox with proper path handling
        documents = dropbox_service.list_files(path)
        logger.info(f"Found {len(documents)} documents/folders in path '{path}'")
        
        # Debug output for each document
        for doc in documents:
            logger.info(f"Document: {doc['name']} ({doc['type']}), path: {doc['path']}")
        
        # Get path parts for breadcrumb navigation
        current_path_parts = get_path_parts(path)
        
        return templates.TemplateResponse("documents.html", {
            "request": request,
            "documents": documents,
            "current_path": path,
            "current_path_parts": current_path_parts
        })
    except AuthError as e:
        logger.error(f"Dropbox authentication error: {str(e)}")
        return templates.TemplateResponse("dropbox_error.html", {
            "request": request,
            "error_details": "Authentifizierungsfehler mit Dropbox. Das Token ist möglicherweise abgelaufen."
        })
    except Exception as e:
        logger.exception(f"Error loading documents: {str(e)}")
        return templates.TemplateResponse("dropbox_error.html", {
            "request": request,
            "error_details": f"Ein Fehler ist aufgetreten: {str(e)}"
        })

@app.get("/view-document", response_class=HTMLResponse)
async def view_document(request: Request, path: str):
    logger.info(f"View document requested for path: {path}")
    try:
        # Decode the path if it's URL-encoded
        path = unquote(path)
        logger.info(f"Decoded document path: '{path}'")
        
        # Get document content
        content = dropbox_service.download_file(path)
        
        if not content:
            logger.error(f"Document not found: {path}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create document object
        document = {
            "name": path.split("/")[-1],
            "path": path,
            "content": content
        }
        
        return templates.TemplateResponse("view_document.html", {
            "request": request,
            "document": document
        })
    except Exception as e:
        logger.exception(f"Error viewing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    logger.info("Search page requested")
    return templates.TemplateResponse("search.html", {"request": request})

# Check token validity
@app.get("/check-auth")
async def check_auth(request: Request):
    logger.info("Auth check requested")
    auth_header = request.headers.get("Authorization") 
    
    if not auth_header and "access_token" in request.cookies:
        auth_header = f"Bearer {request.cookies.get('access_token')}"
    
    if not auth_header:
        logger.warning("No auth token found in headers or cookies")
        return JSONResponse({"authenticated": False})
    
    try:
        # Extract the token from the Bearer header
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            logger.warning("Invalid token: no username")
            return JSONResponse({"authenticated": False})
        
        admin_user = get_admin_user()
        if username != admin_user["username"]:
            logger.warning(f"Invalid username in token: {username}")
            return JSONResponse({"authenticated": False})
        
        logger.info(f"Valid token for user: {username}")
        return JSONResponse({"authenticated": True})
    except JWTError:
        logger.warning("Invalid token: JWT error")
        return JSONResponse({"authenticated": False})

# Global exception handler for frontend routes
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        logger.warning("Unauthorized access attempt")
        return RedirectResponse(url="/login", status_code=303)
    
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return templates.TemplateResponse("error.html", {
        "request": request,
        "status_code": exc.status_code,
        "detail": exc.detail
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
