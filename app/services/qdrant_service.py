import os
import logging
import traceback
import numpy as np
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
            logger.info(f"Using collection name: {QDRANT_COLLECTION_NAME}")
            
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
            
            # Check points_count instead of vectors_count
            points_count = getattr(collection_info, 'points_count', 0)
            if not points_count:
                logger.warning(f"Collection {self.collection_name} exists but has no points")
                return []
            
            # Get all document IDs using scroll pagination
            all_doc_ids = set()
            offset = None
            limit = 100  # Number of points to retrieve per request
            
            while True:
                # Get batch of points
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = scroll_result[0]  # First element is the list of points
                next_offset = scroll_result[1]  # Second element is the next offset
                
                if not points:
                    break
                    
                # Extract document IDs from points
                for point in points:
                    if 'document_id' in point.payload:
                        doc_id = point.payload['document_id']
                        all_doc_ids.add(doc_id)
                
                # Check if we've retrieved all points
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            logger.info(f"Found {len(all_doc_ids)} unique document IDs")
            return list(all_doc_ids)
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
            
            # Delete existing document chunks if they exist
            logger.info(f"Checking if document {document_id} already exists")
            
            try:
                # Create filter to find existing points for this document
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
                
                # Try to find existing points
                existing_points = self.client.scroll(
                    collection_name=self.collection_name,
                    filter=filter_condition,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                
                if existing_points[0]:  # If points exist
                    logger.info(f"Document {document_id} already exists, deleting existing chunks")
                    
                    # Sammle zuerst alle IDs (robuster als Filterlöschung)
                    all_point_ids = []
                    offset = None
                    
                    while True:
                        batch_result = self.client.scroll(
                            collection_name=self.collection_name,
                            filter=filter_condition,
                            limit=100,
                            offset=offset,
                            with_payload=False,
                            with_vectors=False
                        )
                        
                        points = batch_result[0]
                        next_offset = batch_result[1]
                        
                        if not points:
                            break
                            
                        all_point_ids.extend([point.id for point in points])
                        
                        if next_offset is None:
                            break
                            
                        offset = next_offset
                    
                    # Lösche die Punkte per ID (wenn welche gefunden wurden)
                    if all_point_ids:
                        logger.info(f"Deleting {len(all_point_ids)} existing points for document {document_id}")
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(
                                points=all_point_ids
                            )
                        )
                    
                    logger.info(f"Deleted existing chunks for document {document_id}")
            except Exception as e:
                logger.warning(f"Error checking/deleting existing document: {e}")
                # Continue with indexing even if checking fails
                
            # Split content into chunks
            chunks = self._chunk_text(document_content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"No valid chunks generated for {document_name}")
                return False
            
            # Create embeddings for each chunk
            points = []
            
            logger.info(f"Creating embeddings for {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                
                try:
                    # Create embedding
                    logger.info(f"Creating embedding for chunk {i} (length: {len(chunk)})")
                    embedding = self.model.encode(chunk)
                    
                    # Validate embedding
                    if not isinstance(embedding, np.ndarray):
                        logger.warning(f"Invalid embedding type for chunk {i}: {type(embedding)}")
                        continue
                    
                    if len(embedding) != self.vector_size:
                        logger.warning(f"Invalid embedding size for chunk {i}: {len(embedding)} instead of {self.vector_size}")
                        continue
                    
                    # Create point with a safe ID
                    safe_doc_id = document_id.replace('/', '_').replace(' ', '_').replace('.', '_')
                    point_id = f"{safe_doc_id}_{i}"
                    
                    # Create shorter preview of the chunk for logging
                    chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    logger.info(f"Creating point with ID: {point_id}, chunk preview: {chunk_preview}")
                    
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
                    logger.error(traceback.format_exc())
                    continue
            
            # Upload points to Qdrant
            if points:
                logger.info(f"Ready to upload {len(points)} points to Qdrant")
                
                # Kleine Stapelgröße für bessere Zuverlässigkeit
                batch_size = 50  # Process in batches to avoid timeouts
                for i in range(0, len(points), batch_size):
                    batch = points[i:i+batch_size]
                    logger.info(f"Uploading batch of {len(batch)} points to Qdrant (batch {i//batch_size + 1}/{(len(points)+batch_size-1)//batch_size})")
                    
                    try:
                        # Versuche den Stapel hochzuladen
                        result = self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch
                        )
                        logger.info(f"Batch upload successful: {result}")
                    except Exception as batch_error:
                        logger.error(f"Error uploading batch: {str(batch_error)}")
                        logger.error(traceback.format_exc())
                        
                        # Versuche jeden Punkt einzeln hochzuladen
                        logger.info("Trying to upload points individually...")
                        for point in batch:
                            try:
                                self.client.upsert(
                                    collection_name=self.collection_name,
                                    points=[point]
                                )
                                logger.info(f"Successfully uploaded point: {point.id}")
                            except Exception as point_error:
                                logger.error(f"Error uploading point {point.id}: {str(point_error)}")
                
                # Überprüfe, ob die Punkte tatsächlich hochgeladen wurden
                try:
                    # Prüfen, ob Dokument jetzt im Index ist
                    check_result = self.list_indexed_documents()
                    logger.info(f"Post-indexing check: {check_result}")
                    doc_indexed = document_id in check_result
                    
                    if doc_indexed:
                        logger.info(f"Document {document_id} successfully verified in index")
                    else:
                        logger.warning(f"Document {document_id} not found in index after indexing attempt")
                except Exception as check_error:
                    logger.error(f"Error checking indexing result: {str(check_error)}")
                
                logger.info(f"Successfully indexed document {document_name} with {len(points)} chunks")
                return True
            else:
                logger.warning(f"No valid points generated for {document_name}")
                return False
        except Exception as e:
            logger.error(f"Error indexing document {document_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False  # Return False instead of raising to prevent service crashes
    
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
            
            # Check points_count instead of vectors_count
            points_count = getattr(collection_info, 'points_count', 0)
            if not points_count:
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
                limit=top_k
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
                
            # Simple splitting by paragraphs first, then by sentences
            paragraphs = text.split('\n\n')
            chunks = []
            
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue
                    
                # Approximate size by character count
                para_size = len(paragraph)
                
                if current_size + para_size > chunk_size * 4 and current_chunk:  # Using character count instead of word count
                    # Current chunk is full, save it
                    chunks.append('\n\n'.join(current_chunk))
                    
                    # Start new chunk with overlap (keep the last 1-2 paragraphs)
                    overlap_start = max(0, len(current_chunk) - 2)  # Keep last 2 paragraphs for overlap
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(p) for p in current_chunk)
                
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_size += para_size
                
                # If a single paragraph is very large, split it further
                if para_size > chunk_size * 3:
                    # Split large paragraph into sentences
                    sentences = paragraph.replace('\n', ' ').split('. ')
                    sentences = [s.strip() + '.' for s in sentences if s.strip()]
                    
                    # Reset the current chunk with just the large paragraph
                    current_chunk = []
                    current_size = 0
                    
                    sentence_chunk = []
                    sentence_size = 0
                    
                    for sentence in sentences:
                        # Skip empty sentences
                        if not sentence.strip():
                            continue
                            
                        sent_size = len(sentence)
                        
                        if sentence_size + sent_size > chunk_size * 3 and sentence_chunk:
                            # Save the sentence chunk
                            chunks.append(' '.join(sentence_chunk))
                            
                            # Overlap for sentences
                            overlap_start = max(0, len(sentence_chunk) - 3)  # Keep last 3 sentences
                            sentence_chunk = sentence_chunk[overlap_start:]
                            sentence_size = sum(len(s) for s in sentence_chunk)
                        
                        sentence_chunk.append(sentence)
                        sentence_size += sent_size
                    
                    # Add the last sentence chunk if not empty
                    if sentence_chunk:
                        chunks.append(' '.join(sentence_chunk))
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            logger.info(f"Created {len(chunks)} chunks from document")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            logger.error(traceback.format_exc())
            return []
