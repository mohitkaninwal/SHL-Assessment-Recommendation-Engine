"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines retrieval with LLM for intelligent recommendation
"""

import logging
from typing import List, Dict, Optional
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDB
from .retriever import AssessmentRetriever
from .llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for assessment recommendations"""
    
    def __init__(
        self,
        vector_db: VectorDB,
        embedding_generator: EmbeddingGenerator,
        llm_client: LLMClient,
        use_llm_reranking: bool = True,
        use_query_expansion: bool = False
    ):
        """
        Initialize RAG pipeline
        
        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator instance
            llm_client: LLM client instance
            use_llm_reranking: Whether to use LLM for re-ranking
            use_query_expansion: Whether to expand queries with LLM
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.retriever = AssessmentRetriever(vector_db, embedding_generator)
        self.use_llm_reranking = use_llm_reranking
        self.use_query_expansion = use_query_expansion
        
        logger.info("RAG Pipeline initialized")
    
    def recommend(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        balance_skills: bool = True,
        include_explanation: bool = False
    ) -> Dict:
        """
        Generate recommendations using RAG pipeline
        
        Args:
            query: User query
            top_k: Number of recommendations to return
            min_score: Minimum similarity score threshold
            balance_skills: Whether to balance K/P types
            include_explanation: Whether to include LLM explanation
            
        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(f"Processing query: '{query[:50]}...'")
        
        # Step 1: Query understanding with LLM
        query_analysis = self.llm_client.classify_query_intent(query)
        logger.info(f"Query analysis: {query_analysis['technical_weight']:.2f} technical, "
                   f"{query_analysis['behavioral_weight']:.2f} behavioral")
        
        # Step 2: Optional query expansion
        search_query = query
        if self.use_query_expansion:
            expanded_query = self.llm_client.expand_query(query)
            logger.info(f"Expanded query: {expanded_query[:100]}...")
            search_query = expanded_query
        
        # Step 3: Retrieve candidates from vector DB
        if balance_skills:
            # Use balanced retrieval based on LLM analysis
            candidates = self.retriever.retrieve_balanced(
                query=search_query,
                top_k=top_k * 2,  # Get more for re-ranking
                hard_skill_ratio=query_analysis['technical_weight'],
                min_score=min_score
            )
        else:
            # Simple retrieval
            candidates = self.retriever.retrieve(
                query=search_query,
                top_k=top_k * 2,
                min_score=min_score
            )
        
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Step 4: Format candidates
        formatted_candidates = self.retriever.format_results(candidates)
        
        # Step 5: Optional LLM re-ranking
        if self.use_llm_reranking and formatted_candidates:
            logger.info("Re-ranking with LLM...")
            reranked = self.llm_client.rerank_assessments(
                query=query,
                assessments=formatted_candidates,
                top_k=top_k
            )
            final_recommendations = reranked
        else:
            final_recommendations = formatted_candidates[:top_k]
        
        # Step 6: Generate explanation
        explanation = ""
        if include_explanation and final_recommendations:
            explanation = self.llm_client.generate_explanation(
                query=query,
                recommendations=final_recommendations
            )
        
        # Step 7: Compile results
        result = {
            'query': query,
            'recommendations': final_recommendations,
            'total_found': len(candidates),
            'returned': len(final_recommendations),
            'query_analysis': {
                'technical_weight': query_analysis['technical_weight'],
                'behavioral_weight': query_analysis['behavioral_weight'],
                'primary_skills': query_analysis['primary_skills']
            },
            'explanation': explanation if include_explanation else None
        }
        
        # Add statistics
        k_count = sum(1 for r in final_recommendations if r.get('test_type') == 'K')
        p_count = sum(1 for r in final_recommendations if r.get('test_type') == 'P')
        
        result['statistics'] = {
            'knowledge_skills_count': k_count,
            'personality_behavior_count': p_count,
            'other_count': len(final_recommendations) - k_count - p_count
        }
        
        logger.info(f"Returning {len(final_recommendations)} recommendations "
                   f"({k_count} K, {p_count} P)")
        
        return result
    
    def batch_recommend(
        self,
        queries: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[Dict]:
        """
        Generate recommendations for multiple queries
        
        Args:
            queries: List of queries
            top_k: Number of recommendations per query
            **kwargs: Additional arguments for recommend()
            
        Returns:
            List of recommendation results
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.recommend(query, top_k=top_k, **kwargs)
            results.append(result)
        
        return results


def create_rag_pipeline(
    vector_db: VectorDB,
    embedding_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    use_llm_reranking: bool = True,
    use_query_expansion: bool = False
) -> RAGPipeline:
    """
    Create RAG pipeline instance
    
    Args:
        vector_db: Vector database instance
        embedding_model: Optional embedding model name
        llm_api_key: Optional LLM API key
        llm_model: Optional LLM model name
        use_llm_reranking: Whether to use LLM re-ranking
        use_query_expansion: Whether to expand queries
        
    Returns:
        RAGPipeline instance
    """
    embedding_gen = EmbeddingGenerator(model_name=embedding_model)
    llm_client = LLMClient(api_key=llm_api_key, model=llm_model)
    
    return RAGPipeline(
        vector_db=vector_db,
        embedding_generator=embedding_gen,
        llm_client=llm_client,
        use_llm_reranking=use_llm_reranking,
        use_query_expansion=use_query_expansion
    )


if __name__ == "__main__":
    print("RAG Pipeline module loaded successfully")
    print("Note: Requires initialized VectorDB, EmbeddingGenerator, and LLMClient")
