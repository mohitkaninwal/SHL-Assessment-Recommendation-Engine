"""
Recommendation Engine with RAG Pipeline
Main entry point for generating assessment recommendations using LLM
"""

import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDB
from .retriever import AssessmentRetriever
from .llm_client import LLMClient
from .rag_pipeline import RAGPipeline, create_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine"""
    
    def __init__(
        self,
        use_rag: bool = True,
        use_llm_reranking: bool = True,
        use_query_expansion: bool = False
    ):
        """
        Initialize recommendation engine
        
        Args:
            use_rag: Whether to use RAG pipeline (with LLM)
            use_llm_reranking: Whether to use LLM for re-ranking
            use_query_expansion: Whether to expand queries with LLM
        """
        logger.info("Initializing Recommendation Engine...")
        
        # Initialize components
        self.vector_db = VectorDB(dimension=384)
        self.embedding_gen = EmbeddingGenerator()
        
        self.use_rag = use_rag
        
        if use_rag:
            # Initialize with LLM
            self.llm_client = LLMClient()
            self.pipeline = RAGPipeline(
                vector_db=self.vector_db,
                embedding_generator=self.embedding_gen,
                llm_client=self.llm_client,
                use_llm_reranking=use_llm_reranking,
                use_query_expansion=use_query_expansion
            )
            logger.info("RAG Pipeline initialized")
        else:
            # Simple retrieval without LLM
            self.retriever = AssessmentRetriever(
                vector_db=self.vector_db,
                embedding_generator=self.embedding_gen
            )
            logger.info("Simple retrieval initialized (no LLM)")
    
    def recommend(
        self,
        query: str,
        top_k: int = 10,
        balance_skills: bool = True,
        include_explanation: bool = False
    ) -> Dict:
        """
        Generate recommendations for a query
        
        Args:
            query: User query
            top_k: Number of recommendations
            balance_skills: Whether to balance K/P types
            include_explanation: Whether to include explanation
            
        Returns:
            Recommendation results
        """
        if self.use_rag:
            return self.pipeline.recommend(
                query=query,
                top_k=top_k,
                balance_skills=balance_skills,
                include_explanation=include_explanation
            )
        else:
            # Simple retrieval
            if balance_skills:
                results = self.retriever.retrieve_balanced(query, top_k=top_k)
            else:
                results = self.retriever.retrieve(query, top_k=top_k)
            
            formatted = self.retriever.format_results(results)
            
            return {
                'query': query,
                'recommendations': formatted,
                'total_found': len(results),
                'returned': len(formatted)
            }
    
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
            **kwargs: Additional arguments
            
        Returns:
            List of results
        """
        if self.use_rag:
            return self.pipeline.batch_recommend(queries, top_k=top_k, **kwargs)
        else:
            results = []
            for query in queries:
                result = self.recommend(query, top_k=top_k, **kwargs)
                results.append(result)
            return results


def main():
    """CLI for recommendation engine"""
    parser = argparse.ArgumentParser(description='SHL Assessment Recommendation Engine')
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process'
    )
    parser.add_argument(
        '--queries-file',
        type=str,
        help='JSON file with list of queries'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of recommendations to return (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG pipeline (simple retrieval only)'
    )
    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable LLM re-ranking'
    )
    parser.add_argument(
        '--expand-query',
        action='store_true',
        help='Enable query expansion with LLM'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Include explanation for recommendations'
    )
    
    args = parser.parse_args()
    
    if not args.query and not args.queries_file:
        parser.error("Either --query or --queries-file must be provided")
    
    try:
        # Initialize engine
        engine = RecommendationEngine(
            use_rag=not args.no_rag,
            use_llm_reranking=not args.no_reranking,
            use_query_expansion=args.expand_query
        )
        
        # Process queries
        if args.query:
            # Single query
            result = engine.recommend(
                query=args.query,
                top_k=args.top_k,
                include_explanation=args.explain
            )
            
            print(f"\nQuery: {result['query']}")
            print(f"Found: {result['total_found']} assessments")
            print(f"\nTop {len(result['recommendations'])} Recommendations:")
            print("=" * 80)
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n{i}. {rec['assessment_name']}")
                print(f"   Type: {rec['test_type']}")
                print(f"   Score: {rec['similarity_score']:.3f}")
                print(f"   URL: {rec['assessment_url']}")
                if rec.get('description'):
                    print(f"   Description: {rec['description'][:100]}...")
            
            if result.get('explanation'):
                print(f"\nExplanation: {result['explanation']}")
            
            if result.get('query_analysis'):
                analysis = result['query_analysis']
                print(f"\nQuery Analysis:")
                print(f"  Technical: {analysis['technical_weight']:.2f}")
                print(f"  Behavioral: {analysis['behavioral_weight']:.2f}")
                print(f"  Skills: {', '.join(analysis['primary_skills'][:5])}")
            
            # Save to file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.output}")
        
        else:
            # Multiple queries from file
            with open(args.queries_file, 'r') as f:
                queries_data = json.load(f)
            
            if isinstance(queries_data, list):
                queries = queries_data
            elif isinstance(queries_data, dict) and 'queries' in queries_data:
                queries = queries_data['queries']
            else:
                raise ValueError("Invalid queries file format")
            
            print(f"Processing {len(queries)} queries...")
            results = engine.batch_recommend(
                queries=queries,
                top_k=args.top_k,
                include_explanation=args.explain
            )
            
            # Print summary
            print(f"\nProcessed {len(results)} queries")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['query'][:50]}... → {len(result['recommendations'])} recommendations")
            
            # Save results
            output_file = args.output or 'recommendations_output.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
