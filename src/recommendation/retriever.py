"""
Retrieval Module
Implements semantic search and query preprocessing for assessment retrieval
"""

import logging
import re
from typing import List, Dict, Optional, Any
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Preprocess queries before embedding and retrieval"""
    
    @staticmethod
    def clean_query(query: str) -> str:
        """
        Clean and normalize query text
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        if not query:
            return ""
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove special characters that might interfere (keep alphanumeric, spaces, common punctuation)
        query = re.sub(r'[^\w\s\-.,!?]', ' ', query)
        
        # Normalize to lowercase (optional - depends on embedding model)
        # Some models work better with original case, so we'll keep it
        
        return query.strip()
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """
        Extract potential keywords from query
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = query.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    @staticmethod
    def detect_test_type_preference(query: str) -> Optional[str]:
        """
        Detect if query prefers a specific test type
        
        Args:
            query: Query string
            
        Returns:
            'K' for Knowledge/Skills, 'P' for Personality/Behavior, or None
        """
        query_lower = query.lower()
        
        # Knowledge/Skills indicators
        knowledge_keywords = [
            'programming', 'coding', 'technical', 'skill', 'knowledge',
            'java', 'python', 'sql', 'database', 'algorithm', 'software',
            'development', 'engineering', 'technical', 'expertise'
        ]
        
        # Personality/Behavior indicators
        personality_keywords = [
            'personality', 'behavior', 'collaboration', 'teamwork',
            'communication', 'leadership', 'soft skill', 'interpersonal',
            'attitude', 'trait', 'behavioral', 'social'
        ]
        
        knowledge_score = sum(1 for kw in knowledge_keywords if kw in query_lower)
        personality_score = sum(1 for kw in personality_keywords if kw in query_lower)
        
        if knowledge_score > personality_score and knowledge_score > 0:
            return 'K'
        elif personality_score > knowledge_score and personality_score > 0:
            return 'P'
        
        return None


class AssessmentRetriever:
    """Retrieve assessments using semantic search"""
    
    def __init__(self, vector_db: VectorDB, embedding_generator: EmbeddingGenerator):
        """
        Initialize retriever
        
        Args:
            vector_db: VectorDB instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.preprocessor = QueryPreprocessor()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        test_type_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve relevant assessments for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            test_type_filter: Optional filter by test type ('K' or 'P')
            min_score: Minimum similarity score threshold
            
        Returns:
            List of assessment results with scores and metadata
        """
        # Preprocess query
        cleaned_query = self.preprocessor.clean_query(query)
        if not cleaned_query:
            logger.warning("Empty query after preprocessing")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(cleaned_query)
        
        # Build filter
        filter_dict = None
        if test_type_filter:
            filter_dict = {'test_type': test_type_filter}
        
        # Search vector database
        results = self.vector_db.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k * 2,  # Get more results for filtering/reranking
            filter_dict=filter_dict,
            include_metadata=True
        )
        
        # Filter by minimum score
        filtered_results = [r for r in results if r['score'] >= min_score]
        
        # Limit to top_k
        filtered_results = filtered_results[:top_k]
        
        logger.info(f"Retrieved {len(filtered_results)} assessments for query: '{query[:50]}...'")
        
        return filtered_results
    
    def retrieve_balanced(
        self,
        query: str,
        top_k: int = 10,
        hard_skill_ratio: float = 0.6,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve balanced mix of hard skills (K) and soft skills (P) assessments
        
        Args:
            query: Search query
            top_k: Total number of results to return
            hard_skill_ratio: Ratio of hard skills (K) to soft skills (P)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of balanced assessment results
        """
        # Detect preference
        preferred_type = self.preprocessor.detect_test_type_preference(query)
        
        # Calculate counts
        k_count = int(top_k * hard_skill_ratio)
        p_count = top_k - k_count
        
        # Retrieve for each type
        k_results = self.retrieve(query, top_k=k_count, test_type_filter='K', min_score=min_score)
        p_results = self.retrieve(query, top_k=p_count, test_type_filter='P', min_score=min_score)
        
        # Combine and sort by score
        combined_results = k_results + p_results
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k
        balanced_results = combined_results[:top_k]
        
        logger.info(f"Retrieved balanced set: {len([r for r in balanced_results if r['metadata'].get('test_type') == 'K'])} K, "
                   f"{len([r for r in balanced_results if r['metadata'].get('test_type') == 'P'])} P")
        
        return balanced_results
    
    def format_results(self, results: List[Dict]) -> List[Dict]:
        """
        Format retrieval results for API/display
        
        Args:
            results: Raw retrieval results
            
        Returns:
            Formatted results with assessment details
        """
        formatted = []
        for result in results:
            metadata = result.get('metadata', {})
            formatted.append({
                'assessment_name': metadata.get('name', 'Unknown'),
                'assessment_url': metadata.get('url', ''),
                'test_type': metadata.get('test_type', ''),
                'description': metadata.get('description', ''),
                'category': metadata.get('category', ''),
                'similarity_score': result.get('score', 0.0)
            })
        
        return formatted


def create_retriever(vector_db: VectorDB, embedding_model: Optional[str] = None) -> AssessmentRetriever:
    """
    Create AssessmentRetriever instance
    
    Args:
        vector_db: VectorDB instance
        embedding_model: Optional embedding model name
        
    Returns:
        AssessmentRetriever instance
    """
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    return AssessmentRetriever(vector_db, embedding_generator)


if __name__ == "__main__":
    print("Retriever module loaded successfully")
    print("Note: Requires initialized VectorDB and EmbeddingGenerator")







