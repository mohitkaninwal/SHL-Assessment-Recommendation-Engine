"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines retrieval with LLM for intelligent recommendation
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set, Tuple
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
        min_score: float = -1.0,
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

        # Step 2: Optional query expansion (only if enabled; otherwise search_query = query)
        search_query = query
        if self.use_query_expansion:
            expanded_query = self.llm_client.expand_query(query)
            logger.info(f"Expanded query: {expanded_query[:100]}...")
            search_query = expanded_query

        retrieval_pool_k = max(top_k * 10, 100)
        default_hard_ratio = 0.6

        # Run query classification and retrieval in parallel to cut latency
        def run_classify():
            return self.llm_client.classify_query_intent(query)

        def run_retrieve():
            if balance_skills:
                return self.retriever.retrieve_balanced(
                    query=search_query,
                    top_k=retrieval_pool_k,
                    hard_skill_ratio=default_hard_ratio,
                    min_score=min_score
                )
            return self.retriever.retrieve(
                query=search_query,
                top_k=retrieval_pool_k,
                min_score=min_score
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_classify = executor.submit(run_classify)
            future_retrieve = executor.submit(run_retrieve)
            query_analysis = future_classify.result()
            candidates = future_retrieve.result()

        logger.info(f"Query analysis: {query_analysis['technical_weight']:.2f} technical, "
                   f"{query_analysis['behavioral_weight']:.2f} behavioral")
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Step 4: Format candidates
        formatted_candidates = self.retriever.format_results(candidates)
        
        # Step 5: Optional LLM re-ranking (smaller pool = faster LLM response)
        if self.use_llm_reranking and formatted_candidates:
            logger.info("Re-ranking with LLM...")
            rerank_pool_k = min(len(formatted_candidates), max(top_k * 3, 30))
            reranked = self.llm_client.rerank_assessments(
                query=query,
                assessments=formatted_candidates[:rerank_pool_k],
                top_k=min(rerank_pool_k, top_k)
            )
            final_recommendations = reranked[:top_k]
        else:
            final_recommendations = formatted_candidates[:top_k]

        # Step 5b: Enforce K/P balance after reranking for mixed-domain queries.
        if (
            balance_skills
            and final_recommendations
            and not self._should_skip_balance_enforcement(final_recommendations)
        ):
            final_recommendations = self._enforce_post_rerank_balance(
                query_analysis=query_analysis,
                reranked_results=final_recommendations,
                candidate_pool=formatted_candidates,
                top_k=top_k
            )

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
        k_count = sum(1 for r in final_recommendations if 'K' in self._assessment_type_codes(r))
        p_count = sum(1 for r in final_recommendations if 'P' in self._assessment_type_codes(r))
        
        result['statistics'] = {
            'knowledge_skills_count': k_count,
            'personality_behavior_count': p_count,
            'other_count': len(final_recommendations) - k_count - p_count
        }
        
        logger.info(f"Returning {len(final_recommendations)} recommendations "
                   f"({k_count} K, {p_count} P)")
        
        return result

    def _assessment_type_codes(self, assessment: Dict) -> Set[str]:
        """Extract normalized test-type codes from assessment fields."""
        raw = assessment.get('all_test_types') or assessment.get('test_type') or ''
        if isinstance(raw, list):
            tokens = [str(token).upper().strip() for token in raw if str(token).strip()]
        else:
            tokens = [token for token in re.split(r"[,\s/|]+", str(raw).upper()) if token]
        return {token for token in tokens if re.fullmatch(r"[ABCDEKPS]", token)}

    def _is_labeled_prior(self, assessment: Dict) -> bool:
        """Whether assessment was injected/boosted by labeled query prior."""
        raw = (
            assessment.get("from_labeled_prior")
            or assessment.get("metadata", {}).get("from_labeled_prior")
        )
        return str(raw).lower() in {"yes", "true", "1"}

    def _should_skip_balance_enforcement(self, recommendations: List[Dict]) -> bool:
        """
        Preserve strong calibrated retrieval hits instead of over-enforcing type mix.
        """
        if not recommendations:
            return False
        prior_hits = sum(1 for item in recommendations if self._is_labeled_prior(item))
        return prior_hits >= 3

    def _is_mixed_domain_query(self, query_analysis: Dict) -> bool:
        """Detect whether query clearly spans technical and behavioral domains."""
        technical = float(query_analysis.get('technical_weight', 0.0) or 0.0)
        behavioral = float(query_analysis.get('behavioral_weight', 0.0) or 0.0)
        return technical >= 0.35 and behavioral >= 0.35 and abs(technical - behavioral) <= 0.25

    def _balance_targets(self, top_k: int, query_analysis: Dict) -> Tuple[int, int]:
        """
        Compute K/P targets for final list.
        - Mixed-domain: enforce near 50/50 split.
        - Otherwise: follow query technical/behavioral weights.
        """
        if top_k <= 1:
            return top_k, 0

        technical = float(query_analysis.get('technical_weight', 0.6) or 0.6)
        behavioral = float(query_analysis.get('behavioral_weight', 0.4) or 0.4)

        total = technical + behavioral
        if total <= 0:
            technical = 0.6
            total = 1.0

        if self._is_mixed_domain_query(query_analysis):
            k_ratio = 0.5
        else:
            # Preserve strong-domain intent while still allowing secondary coverage.
            k_ratio = technical / total
            k_ratio = max(0.1, min(0.9, k_ratio))

        k_target = int(round(top_k * k_ratio))
        k_target = max(1, min(top_k - 1, k_target))
        p_target = top_k - k_target

        return k_target, p_target

    def _enforce_post_rerank_balance(
        self,
        query_analysis: Dict,
        reranked_results: List[Dict],
        candidate_pool: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Preserve recall: start with top_k by rerank order, then only swap if we have 0 K or 0 P.
        """
        if top_k <= 0:
            return []

        selected = list(reranked_results[:top_k])
        seen_keys: Set[str] = {
            str(item.get('assessment_url') or item.get('url') or item.get('assessment_name') or id(item))
            for item in selected
        }
        k_target, p_target = self._balance_targets(top_k, query_analysis)

        def count(code: str) -> int:
            return sum(1 for item in selected if code in self._assessment_type_codes(item))

        def key_of(item: Dict) -> str:
            return str(item.get('assessment_url') or item.get('url') or item.get('assessment_name') or id(item))

        # Rest of pool (reranked after top_k + candidates) for optional swap.
        rest = [r for r in reranked_results[top_k:] + list(candidate_pool) if key_of(r) not in seen_keys]

        def best_of_type(ty: str):
            for item in rest:
                if ty in self._assessment_type_codes(item):
                    return item
            return None

        # Only swap when we have zero coverage of a type that we need.
        if count('K') == 0 and k_target >= 1:
            replacement = best_of_type('K')
            if replacement and len(selected) >= 1:
                selected[-1] = replacement
                seen_keys.add(key_of(replacement))
        if count('P') == 0 and p_target >= 1:
            replacement = best_of_type('P')
            if replacement and len(selected) >= 1:
                idx = -2 if count('K') >= 1 else -1
                if -idx <= len(selected):
                    selected[idx] = replacement

        return selected[:top_k]
    
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
