import os
import tempfile
import logging
import time
from urllib.parse import unquote
import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from dropbox.exceptions import ApiError, AuthError
from dotenv import load_dotenv

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
            content = response.content.decode('utf-8')
            logger.info(f"File downloaded successfully, size: {len(content)} bytes")
            return content
        except AuthError as e:
            logger.error(f"Dropbox authentication error downloading file: {e}")
            self._valid = False
            return None
        except ApiError as e:
            logger.error(f"Dropbox API error downloading file: {e}")
            return None
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try to use a temporary file
            logger.warning(f"UTF-8 decoding failed for file: {path}")
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    temp_path = temp.name
                    metadata, response = self.client.files_download(path)
                    temp.write(response.content)
                
                logger.info(f"File saved to temporary location: {temp_path}")
                return "Dieser Dateityp kann nicht als Text angezeigt werden."
            except Exception as e:
                logger.error(f"Error processing non-text file: {e}")
                return None
            finally:
                if 'temp_path' in locals():
                    os.remove(temp_path)
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
