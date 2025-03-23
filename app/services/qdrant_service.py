import os
import logging
import traceback
import numpy as np
import hashlib
import uuid
import random
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_db")

class QdrantService:
    def __init__(self):
        try:
            logger.info(f"Initializing Qdrant client with URL: {QDRANT_URL}")
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info("Qdrant client initialized successfully")
            
            logger.info("Loading sentence transformer model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.collection_name = QDRANT_COLLECTION_NAME
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Vector size: {self.vector_size}")
            
            self._ensure_collection_exists()
            logger.info("Qdrant service initialization complete")
        except Exception as e:
            logger.error(f"Error initializing Qdrant service: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _ensure_collection_exists(self):
        """Make sure the collection exists, create it if it doesn't"""
        try:
            logger.info(f"Checking if collection exists: {self.collection_name}")
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Collection created successfully: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                # Get collection info to verify its configuration
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Collection info: {collection_info}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _generate_numeric_id(self, text, chunk_index=None):
        """Generate a numeric ID for Qdrant based on the document content"""
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Convert the first 12 chars of the hex to an integer
        # This should give a number between 0 and 2^48-1
        num_id = int(hash_hex[:12], 16)
        
        # If we have a chunk index, add it to the ID to make it unique per chunk
        if chunk_index is not None:
            num_id = num_id * 1000 + chunk_index % 1000
        
        return num_id
    
    def list_indexed_documents(self) -> List[str]:
        """Get a list of all document IDs that have been indexed"""
        try:
            logger.info(f"Listing indexed documents in collection: {self.collection_name}")
            
            # First, check if the collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"Collection {self.collection_name} does not exist yet")
                return []
            
            # Get collection info to check if it has points
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                logger.warning(f"Collection {self.collection_name} exists but has no points")
                return []
                
            # Get all points (up to 100)
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = scroll_result[0]  # First element is the list of points
                
                if not points:
                    logger.warning(f"No points found in collection {self.collection_name}")
                    return []
                    
                # Extract unique document IDs
                doc_ids = set()
                for point in points:
                    if 'document_id' in point.payload:
                        # Extract the base document ID from the point ID
                        doc_id = point.payload['document_id']
                        doc_ids.add(doc_id)
                
                logger.info(f"Found {len(doc_ids)} unique document IDs")
                return list(doc_ids)
            except Exception as e:
                logger.error(f"Error scrolling points: {e}")
                
                # Fallback: try search with a dummy vector
                try:
                    logger.info("Trying alternative method to get documents")
                    # Create a dummy vector of zeros
                    dummy_vector = [0.0] * self.vector_size
                    
                    # Search with a high limit
                    search_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=dummy_vector,
                        limit=100,
                        with_payload=True
                    )
                    
                    # Extract unique document IDs
                    doc_ids = set()
                    for result in search_results:
                        if 'document_id' in result.payload:
                            doc_id = result.payload['document_id']
                            doc_ids.add(doc_id)
                    
                    logger.info(f"Found {len(doc_ids)} unique document IDs using alternative method")
                    return list(doc_ids)
                except Exception as search_err:
                    logger.error(f"Error using alternative method: {search_err}")
                    return []
                
        except Exception as e:
            logger.error(f"Error listing indexed documents: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def index_document(self, document_id, document_path, document_name, document_content):
        """Index a document in Qdrant"""
        try:
            logger.info(f"Indexing document: {document_name} (ID: {document_id})")
            logger.info(f"Document content length: {len(document_content)} characters")
            logger.info(f"Document path: {document_path}")
            logger.info(f"Using collection: {self.collection_name}")
            
            # Clean document content
            document_content = document_content.strip()
            if not document_content:
                logger.warning(f"Empty document content for {document_name}")
                return False
            
            # Try to delete existing document chunks if they exist
            try:
                # We'll try to search for existing documents
                indexed_docs = self.list_indexed_documents()
                if document_id in indexed_docs:
                    logger.info(f"Document {document_id} already exists, need to delete")
                    
                    # Try to find all points with this document_id
                    dummy_vector = [0.0] * self.vector_size
                    
                    # This is a workaround for missing filter support in older versions
                    search_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=dummy_vector,
                        limit=100,  # Get up to 100 points
                        with_payload=True
                    )
                    
                    # Filter the results manually
                    point_ids = []
                    for result in search_results:
                        if 'document_id' in result.payload and result.payload['document_id'] == document_id:
                            point_ids.append(result.id)
                    
                    if point_ids:
                        logger.info(f"Deleting {len(point_ids)} existing points for document {document_id}")
                        for point_id in point_ids:
                            try:
                                self.client.delete(
                                    collection_name=self.collection_name,
                                    points_selector=models.PointIdsList(points=[point_id])
                                )
                            except Exception as del_err:
                                logger.warning(f"Error deleting point {point_id}: {del_err}")
                        
                        logger.info(f"Deleted existing chunks for document {document_id}")
            except Exception as e:
                logger.warning(f"Error checking/deleting existing document: {e}")
                # Continue with indexing even if checking fails
            
            # Split content into chunks
            chunks = self._chunk_text(document_content)
            logger.info(f"Created {len(chunks)} chunks from document")
            logger.info(f"Split document into {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"No valid chunks generated for {document_name}")
                return False
            
            # Create embeddings for each chunk
            logger.info(f"Creating embeddings for {len(chunks)} chunks")
            points = []
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                
                try:
                    # Create embedding
                    logger.info(f"Creating embedding for chunk {i} (length: {len(chunk)})")
                    embedding = self.model.encode(chunk)
                    
                    # Generate a numeric ID for this point
                    point_id = self._generate_numeric_id(document_id, i)
                    logger.info(f"Creating point with ID: {point_id}, chunk preview: {chunk[:50]}")
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={
                                "document_id": document_id,
                                "document_path": document_path,
                                "document_name": document_name,
                                "chunk_index": i,
                                "text": chunk
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk {i}: {str(e)}")
                    continue
            
            # Upload points to Qdrant
            if points:
                logger.info(f"Ready to upload {len(points)} points to Qdrant")
                
                # Upload in batches of 10 to avoid large requests
                batch_size = 10
                total_batches = (len(points) + batch_size - 1) // batch_size
                
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    logger.info(f"Uploading batch of {len(batch)} points to Qdrant (batch {batch_num}/{total_batches})")
                    
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch,
                            wait=True
                        )
                    except Exception as e:
                        logger.error(f"Error uploading batch: {e}")
                        # Try to upload points individually
                        logger.info("Trying to upload points individually...")
                        for point in batch:
                            try:
                                self.client.upsert(
                                    collection_name=self.collection_name,
                                    points=[point],
                                    wait=True
                                )
                            except Exception as e:
                                logger.error(f"Error uploading point {point.id}: {e}")
                
                # Verify the document was indexed
                indexed_docs = self.list_indexed_documents()
                logger.info(f"Post-indexing check: {indexed_docs}")
                
                if document_id in indexed_docs:
                    logger.info(f"Document {document_id} successfully indexed")
                else:
                    logger.warning(f"Document {document_id} not found in index after indexing attempt")
                
                logger.info(f"Successfully indexed document {document_name} with {len(points)} chunks")
                return True
            else:
                logger.warning(f"No valid points generated for {document_name}")
                return False
        except Exception as e:
            logger.error(f"Error indexing document {document_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        try:
            logger.info(f"Searching for query: '{query}' (top_k: {top_k})")
            
            # Check if collection exists and has points
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return []
                
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                logger.warning(f"Collection {self.collection_name} has no points")
                return []
            
            # Create query embedding
            logger.info("Creating embedding for search query")
            query_embedding = self.model.encode(query)
            
            # Search for similar chunks
            logger.info(f"Sending search request to Qdrant with top_k={top_k}")
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            logger.info(f"Found {len(search_result)} results")
            for i, result in enumerate(search_result):
                logger.info(f"Result {i}: score={result.score}, document={result.payload.get('document_name', 'unknown')}")
            
            return search_result
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _chunk_text(self, text, chunk_size=300, overlap=50):
        """Split text into chunks with overlap"""
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                logger.warning(f"Invalid text type: {type(text)}")
                return []
            
            # For very short texts, just return as a single chunk
            if len(text) < chunk_size:
                return [text]
                
            # Split text into sentences (simple approach)
            sentences = text.replace('\n', ' ').split('. ')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                # Approximate size by word count (rough estimate)
                sentence_size = len(sentence.split())
                
                if current_size + sentence_size > chunk_size and current_chunk:
                    # Chunk is full, add it to chunks
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(s.split()) for s in current_chunk)
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            # Ensure we return at least one chunk
            if not chunks and text:
                chunks = [text]
                
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback: return original text as a single chunk
            if text:
                return [text]
            return []
