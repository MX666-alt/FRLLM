import logging
import asyncio
import sys
from dotenv import load_dotenv
from app.services.dropbox_service import DropboxService
from app.services.qdrant_service import QdrantService

# Configure detailed logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_index_document(document_path):
    """Test the document indexing process directly"""
    try:
        logger.info(f"Testing document indexing for path: {document_path}")
        
        # Initialize services
        logger.info("Initializing Dropbox service")
        dropbox_service = DropboxService()
        
        logger.info("Initializing Qdrant service")
        qdrant_service = QdrantService()
        
        # Test Qdrant connection
        logger.info("Testing Qdrant connection")
        collections = qdrant_service.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        logger.info(f"Available collections: {collection_names}")
        
        # Check if our collection exists
        logger.info(f"Our collection name: {qdrant_service.collection_name}")
        if qdrant_service.collection_name in collection_names:
            logger.info("Collection exists")
            collection_info = qdrant_service.client.get_collection(qdrant_service.collection_name)
            logger.info(f"Collection info: {collection_info}")
        else:
            logger.warning("Collection does not exist")
            qdrant_service._ensure_collection_exists()
        
        # Download document content from Dropbox
        logger.info(f"Downloading document from Dropbox: {document_path}")
        document_content = dropbox_service.download_file(document_path)
        
        if not document_content:
            logger.error(f"Failed to download document content from {document_path}")
            return False
        
        logger.info(f"Document content received. Length: {len(document_content)} characters")
        logger.info(f"Content preview: {document_content[:200]}...")
        
        # Extract document name from path
        document_name = document_path.split('/')[-1]
        logger.info(f"Document name: {document_name}")
        
        # Index the document
        logger.info("Starting document indexing")
        success = qdrant_service.index_document(
            document_id=document_path,
            document_path=document_path,
            document_name=document_name,
            document_content=document_content
        )
        
        logger.info(f"Indexing result: {success}")
        
        # Check if document was indexed
        logger.info("Checking indexed documents")
        indexed_docs = qdrant_service.list_indexed_documents()
        logger.info(f"Indexed documents after indexing: {indexed_docs}")
        
        # Show if our document is in the indexed list
        if document_path in indexed_docs:
            logger.info(f"SUCCESS: Document {document_path} is in the indexed list!")
        else:
            logger.warning(f"WARNING: Document {document_path} is NOT in the indexed list!")
        
        return success
    except Exception as e:
        logger.exception(f"Error in test_index_document: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_index_document.py <document_path>")
        print("Example: python test_index_document.py /test_document.txt")
        sys.exit(1)
    
    document_path = sys.argv[1]
    print(f"Testing indexing for document: {document_path}")
    
    result = asyncio.run(test_index_document(document_path))
    
    if result:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
