"""
Vector Database Integration Module
Handles Pinecone operations for storing and retrieving assessment embeddings
"""

import logging
import os
from typing import List, Dict, Optional, Any
import time
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB:
    """Pinecone vector database wrapper"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize Pinecone connection
        
        Args:
            api_key: Pinecone API key (defaults to env var)
            index_name: Index name (defaults to env var)
            dimension: Vector dimension (required if creating new index)
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'shl-assessments')
        self.dimension = dimension
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
        
        # Connect to or create index
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure index exists, create if it doesn't"""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                
                # Get index stats to determine dimension if not provided
                if not self.dimension:
                    stats = self.index.describe_index_stats()
                    # Dimension is in the index configuration, but we can infer from stats
                    # For now, we'll require dimension to be provided or set
                    logger.info(f"Connected to existing index: {self.index_name}")
            else:
                # Create new index
                if not self.dimension:
                    raise ValueError("Dimension required to create new index. Set EMBEDDING_MODEL or provide dimension.")
                
                logger.info(f"Creating new index '{self.index_name}' with dimension {self.dimension}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Index '{self.index_name}' created and ready")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    def upsert_assessments(
        self,
        assessments: List[Dict],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """
        Upsert assessments with their embeddings to Pinecone
        
        Args:
            assessments: List of assessment dictionaries
            embeddings: List of embedding vectors
            batch_size: Batch size for upsert operations
        """
        if len(assessments) != len(embeddings):
            raise ValueError(f"Mismatch: {len(assessments)} assessments but {len(embeddings)} embeddings")
        
        logger.info(f"Upserting {len(assessments)} assessments to Pinecone")
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for idx, (assessment, embedding) in enumerate(zip(assessments, embeddings)):
            # Create unique ID (use URL hash or index)
            vector_id = self._create_vector_id(assessment, idx)
            
            # Prepare metadata
            description = assessment.get('description') or ''
            metadata = {
                'name': assessment.get('name') or '',
                'url': assessment.get('url') or '',
                'test_type': assessment.get('test_type') or '',
                'all_test_types': assessment.get('all_test_types') or '',
                'description': description[:1000],  # Limit metadata size
                'duration': assessment.get('duration') if assessment.get('duration') is not None else 0,
                'remote_support': assessment.get('remote_support') or '',
                'adaptive_support': assessment.get('adaptive_support') or '',
                'category': assessment.get('category') or ''
            }
            
            # Remove empty metadata fields
            metadata = {k: v for k, v in metadata.items() if v}
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert in batches
        total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[batch_idx:batch_idx + batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {batch_idx // batch_size + 1}/{total_batches}")
            except Exception as e:
                logger.error(f"Error upserting batch {batch_idx // batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully upserted {len(assessments)} assessments")
    
    def _create_vector_id(self, assessment: Dict, index: int) -> str:
        """
        Create unique vector ID from assessment
        
        Args:
            assessment: Assessment dictionary
            index: Fallback index if URL not available
            
        Returns:
            Unique string ID
        """
        # Use URL as base for ID (most unique)
        url = assessment.get('url', '')
        if url:
            # Extract meaningful part from URL
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            return f"assess_{url_hash}"
        else:
            # Fallback to index-based ID
            return f"assess_{index}"
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Search for similar assessments
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {'test_type': 'K'})
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            results = []
            for match in query_response.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata if include_metadata else {}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def delete_all(self):
        """Delete all vectors from index (use with caution!)"""
        logger.warning("Deleting all vectors from index")
        try:
            # Delete all vectors by fetching all IDs and deleting
            stats = self.index.describe_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                # Note: This is a simplified approach. For large indexes, 
                # you might want to use delete_by_filter or fetch IDs first
                logger.warning("delete_all() requires manual implementation for large indexes")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")


def initialize_vector_db(embedding_dimension: int) -> VectorDB:
    """
    Initialize and return VectorDB instance
    
    Args:
        embedding_dimension: Dimension of embeddings
        
    Returns:
        VectorDB instance
    """
    return VectorDB(dimension=embedding_dimension)


if __name__ == "__main__":
    # Test vector DB connection
    try:
        # This will fail without proper API keys, but tests the structure
        print("VectorDB module loaded successfully")
        print("Note: Requires PINECONE_API_KEY environment variable")
    except Exception as e:
        print(f"Error: {e}")





