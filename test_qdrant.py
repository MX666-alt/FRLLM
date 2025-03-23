import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_db")

def test_qdrant_connection():
    """Test connection to Qdrant and check collection status"""
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        logger.info(f"Using collection: {QDRANT_COLLECTION_NAME}")
        
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Test the connection by getting collections list
        collections = client.get_collections()
        logger.info(f"Successfully connected to Qdrant")
        logger.info(f"Available collections: {[c.name for c in collections.collections]}")
        
        # Check if our collection exists
        if QDRANT_COLLECTION_NAME in [c.name for c in collections.collections]:
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} exists")
            
            # Get collection info
            collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
            logger.info(f"Collection info: {collection_info}")
            
            # Check points_count instead of vectors_count
            points_count = getattr(collection_info, 'points_count', 0)
            logger.info(f"Number of points: {points_count}")
            
            # Get a sample of points (if any exist)
            if points_count and points_count > 0:
                sample = client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=5,
                    with_payload=True,
                    with_vectors=False
                )
                
                logger.info(f"Sample of {len(sample[0])} points:")
                for point in sample[0]:
                    logger.info(f"Point ID: {point.id}")
                    logger.info(f"Payload: {point.payload}")
            else:
                logger.info("Collection exists but has no points yet")
        else:
            logger.warning(f"Collection {QDRANT_COLLECTION_NAME} does not exist")
            
            # Create the collection
            logger.info(f"Creating collection {QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} created successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if test_qdrant_connection():
        print("Qdrant connection test successful!")
    else:
        print("Qdrant connection test failed!")
