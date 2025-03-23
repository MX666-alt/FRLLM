import os
import tempfile
import logging
import time
from urllib.parse import unquote
import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from dropbox.exceptions import ApiError, AuthError
from dotenv import load_dotenv
import io
import re

# PDF-Parsing Bibliotheken
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    from pdfminer.pdfparser import PDFSyntaxError
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False
    logging.warning("pdfminer.six nicht installiert. PDF-Extraction wird eingeschränkt sein.")

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    logging.warning("PyPDF2/pypdf nicht installiert. Alternative PDF-Extraction wird nicht verfügbar sein.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Dropbox configuration
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

class DropboxService:
    def __init__(self):
        try:
            logger.info("Initializing Dropbox client")
            self.client = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
            # Test if the token is valid
            self.client.users_get_current_account()
            logger.info("Dropbox client initialized successfully")
            self._valid = True
        except AuthError as e:
            logger.error(f"Dropbox authentication error: {e}")
            self._valid = False
            # Still create the client, but log the error
            self.client = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        except Exception as e:
            logger.error(f"Error initializing Dropbox client: {e}")
            self._valid = False
            # Still create the client, but log the error
            self.client = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    
    def list_files(self, path=""):
        """List all files and folders in the given path"""
        try:
            # Check if token is valid
            if not self._valid:
                logger.error("Cannot list files: Dropbox token is invalid")
                return []
                
            # Ensure path is properly decoded and normalized
            path = unquote(path)
            # Remove trailing slash if present
            if path and path != "/" and path.endswith("/"):
                path = path[:-1]
                
            logger.info(f"Listing files in path: '{path}' (type: {type(path)})")
            
            # Special case for empty path or root path
            if not path or path == "":
                logger.info("Using empty path for root folder")
                result = self.client.files_list_folder("")
            else:
                logger.info(f"Using specified path: {path}")
                result = self.client.files_list_folder(path)
                
            logger.info(f"Got response with {len(result.entries)} entries")
            
            # Debug: print all entries from API
            for entry in result.entries:
                logger.info(f"API returned: {entry.name} ({entry.__class__.__name__}), path: {entry.path_display}")
            
            files = []
            
            for entry in result.entries:
                entry_info = {
                    "id": entry.id if hasattr(entry, 'id') else entry.name,
                    "name": entry.name,
                    "path": entry.path_display,
                    "type": "file" if isinstance(entry, FileMetadata) else "folder"
                }
                logger.info(f"Adding entry: {entry_info['name']} ({entry_info['type']}), path: {entry_info['path']}")
                files.append(entry_info)
            
            logger.info(f"Returning {len(files)} files/folders")
            return files
        except AuthError as e:
            logger.error(f"Dropbox authentication error listing files: {e}")
            self._valid = False
            return []
        except ApiError as e:
            logger.error(f"Dropbox API error listing files: {e}")
            # Provide a more user-friendly error
            if 'not_found' in str(e):
                logger.error(f"Path not found: {path}")
                return []
            else:
                logger.error(f"Detailed error: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def _extract_text_from_pdf_with_pdfminer(self, file_content):
        """Extract text from PDF using PDFMiner"""
        try:
            if not HAS_PDFMINER:
                return None
                
            logger.info("Extracting text from PDF using PDFMiner")
            pdf_file = io.BytesIO(file_content)
            text = pdf_extract_text(pdf_file)
            
            # Clean up the text a bit
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()
            
            logger.info(f"Extracted {len(text)} characters with PDFMiner")
            return text
        except PDFSyntaxError as e:
            logger.error(f"PDFMiner syntax error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text with PDFMiner: {e}")
            return None
    
    def _extract_text_from_pdf_with_pypdf(self, file_content):
        """Extract text from PDF using PyPDF"""
        try:
            if not HAS_PYPDF:
                return None
                
            logger.info("Extracting text from PDF using PyPDF")
            pdf_file = io.BytesIO(file_content)
            
            with pypdf.PdfReader(pdf_file) as pdf:
                text = ""
                for page_num in range(len(pdf.pages)):
                    page = pdf.pages[page_num]
                    text += page.extract_text() + "\n"
            
            # Clean up the text
            text = text.strip()
            
            logger.info(f"Extracted {len(text)} characters with PyPDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF: {e}")
            return None
    
    def _extract_text_from_pdf(self, file_content):
        """Extract text from PDF using available methods"""
        # Try with PDFMiner first (usually better quality)
        text = self._extract_text_from_pdf_with_pdfminer(file_content)
        
        # If PDFMiner failed, try with PyPDF
        if not text:
            text = self._extract_text_from_pdf_with_pypdf(file_content)
            
        # If we got any text, return it
        if text and len(text) > 10:  # Ensure we got meaningful text
            return text
            
        # If all methods failed, return a helpful message
        return "PDF konnte nicht analysiert werden. Möglicherweise ist es gescannt oder passwortgeschützt."
    
    def download_file(self, path):
        """Download a file and return its content as text"""
        try:
            # Check if token is valid
            if not self._valid:
                logger.error("Cannot download file: Dropbox token is invalid")
                return None
                
            # Ensure path is properly decoded
            path = unquote(path)
            logger.info(f"Downloading file: {path}")
            metadata, response = self.client.files_download(path)
            content = response.content
            
            # Check file extension
            is_pdf = path.lower().endswith('.pdf')
            
            # If it's a PDF, extract text
            if is_pdf:
                logger.info(f"File is a PDF, extracting text")
                extracted_text = self._extract_text_from_pdf(content)
                if extracted_text:
                    logger.info(f"Successfully extracted text from PDF, size: {len(extracted_text)} bytes")
                    return extracted_text
                else:
                    logger.warning(f"Failed to extract text from PDF")
            
            # Try to decode as text if not a PDF or if PDF extraction failed
            try:
                text_content = content.decode('utf-8')
                logger.info(f"File decoded as UTF-8, size: {len(text_content)} bytes")
                return text_content
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try other encodings
                logger.warning(f"UTF-8 decoding failed for file: {path}")
                
                # Try common encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text_content = content.decode(encoding)
                        logger.info(f"File decoded as {encoding}, size: {len(text_content)} bytes")
                        return text_content
                    except UnicodeDecodeError:
                        continue
                
                # If all decodings fail and it's not a PDF, return a useful message
                if not is_pdf:
                    logger.warning(f"All text decodings failed for non-PDF file")
                    return "Dieser Dateityp kann nicht als Text angezeigt werden."
                else:
                    # This should never happen as we already tried PDF extraction above
                    logger.warning(f"Both PDF extraction and text decoding failed")
                    return "PDF-Extraktion fehlgeschlagen. Die Datei ist möglicherweise beschädigt oder passwortgeschützt."
                    
        except AuthError as e:
            logger.error(f"Dropbox authentication error downloading file: {e}")
            self._valid = False
            return None
        except ApiError as e:
            logger.error(f"Dropbox API error downloading file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None

    def get_token_info(self):
        """Get information about the current access token"""
        try:
            account = self.client.users_get_current_account()
            self._valid = True
            return {
                "account_id": account.account_id,
                "display_name": account.name.display_name,
                "email": account.email,
                "valid": True
            }
        except AuthError:
            self._valid = False
            return {"valid": False, "error": "Authentication error"}
        except Exception as e:
            self._valid = False
            return {"valid": False, "error": str(e)}
            
    def debug_list_root(self):
        """Debug method to directly list root folder contents"""
        try:
            # Check if token is valid
            if not self._valid:
                logger.error("Cannot debug list root: Dropbox token is invalid")
                return []
                
            logger.info("DEBUG: Directly listing root folder")
            result = self.client.files_list_folder("")
            files = []
            
            logger.info(f"DEBUG: Found {len(result.entries)} entries")
            for entry in result.entries:
                logger.info(f"DEBUG: Entry - {entry.name} ({entry.__class__.__name__}), path: {entry.path_display}")
                files.append({
                    "name": entry.name,
                    "type": "file" if isinstance(entry, FileMetadata) else "folder",
                    "path": entry.path_display
                })
                
            return files
        except AuthError as e:
            logger.error(f"DEBUG: Dropbox authentication error: {e}")
            self._valid = False
            return []
        except Exception as e:
            logger.error(f"DEBUG: Error listing root: {e}")
            return []
