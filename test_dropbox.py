import os
import dropbox
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Dropbox token
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

def test_dropbox():
    print(f"Testing Dropbox connection with token: {DROPBOX_ACCESS_TOKEN[:10]}...")
    try:
        # Initialize Dropbox client
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        
        # Test the connection
        account = dbx.users_get_current_account()
        print(f"Successfully connected to Dropbox account: {account.name.display_name} ({account.email})")
        
        # Try to list files
        result = dbx.files_list_folder("")
        print(f"Found {len(result.entries)} items in root folder:")
        for entry in result.entries:
            print(f"- {entry.name} ({entry.__class__.__name__})")
            
        return True
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        return False

if __name__ == "__main__":
    test_dropbox()
