import os
import logging
import sys
from dotenv import load_dotenv
import dropbox
from dropbox.exceptions import AuthError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

def test_dropbox_connection():
    try:
        logger.info("Initializing Dropbox client for testing")
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        
        # Test the connection by getting account info
        logger.info("Testing connection: Getting account info")
        account = dbx.users_get_current_account()
        logger.info(f"Connected to Dropbox account: {account.name.display_name} ({account.email})")
        
        # Test listing files in the root folder
        logger.info("Testing connection: Listing files in root folder")
        result = dbx.files_list_folder("")
        
        # Show all entries
        logger.info(f"Found {len(result.entries)} entries in root folder:")
        for entry in result.entries:
            logger.info(f"- {entry.name} ({entry.__class__.__name__})")
        
        return True
    except AuthError as e:
        logger.error(f"Authentication error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing Dropbox connection: {e}")
        return False

if __name__ == "__main__":
    print("Testing Dropbox connection...")
    result = test_dropbox_connection()
    if result:
        print("Dropbox connection successful!")
        sys.exit(0)
    else:
        print("Dropbox connection failed!")
        sys.exit(1)
