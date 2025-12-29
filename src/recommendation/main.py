"""
Main module for recommendation system
Orchestrates indexing and retrieval operations
"""

import logging
import argparse
import os
from dotenv import load_dotenv
from .indexer import index_catalog, AssessmentIndexer
from .retriever import create_retriever
from .vector_db import VectorDB, initialize_vector_db
from .embeddings import EmbeddingGenerator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SHL Assessment Recommendation System')
    parser.add_argument(
        'command',
        choices=['index', 'search', 'stats'],
        help='Command to execute'
    )
    parser.add_argument(
        '--catalog',
        type=str,
        default='data/processed_catalog.json',
        help='Path to catalog JSON file (for index command)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Search query (for search command)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Embedding model name (optional)'
    )
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['K', 'P'],
        help='Filter by test type (K or P)'
    )
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Return balanced mix of hard and soft skills'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'index':
            logger.info("=" * 60)
            logger.info("INDEXING ASSESSMENTS")
            logger.info("=" * 60)
            
            index_catalog(
                catalog_path=args.catalog,
                embedding_model=args.model
            )
            
            logger.info("=" * 60)
            logger.info("✓ Indexing completed successfully!")
            logger.info("=" * 60)
        
        elif args.command == 'search':
            if not args.query:
                logger.error("--query is required for search command")
                return
            
            logger.info("=" * 60)
            logger.info("SEARCHING ASSESSMENTS")
            logger.info("=" * 60)
            logger.info(f"Query: {args.query}")
            logger.info(f"Top K: {args.top_k}")
            
            # Initialize components
            embedding_model = args.model or os.getenv('EMBEDDING_MODEL')
            embedding_generator = EmbeddingGenerator(model_name=embedding_model)
            embedding_dim = embedding_generator.get_embedding_dimension()
            
            vector_db = initialize_vector_db(embedding_dimension=embedding_dim)
            retriever = create_retriever(vector_db, embedding_model=embedding_model)
            
            # Perform search
            if args.balanced:
                results = retriever.retrieve_balanced(
                    query=args.query,
                    top_k=args.top_k
                )
            else:
                results = retriever.retrieve(
                    query=args.query,
                    top_k=args.top_k,
                    test_type_filter=args.test_type
                )
            
            # Format and display results
            formatted_results = retriever.format_results(results)
            
            logger.info("=" * 60)
            logger.info(f"Found {len(formatted_results)} results:")
            logger.info("=" * 60)
            
            for idx, result in enumerate(formatted_results, 1):
                logger.info(f"\n{idx}. {result['assessment_name']}")
                logger.info(f"   URL: {result['assessment_url']}")
                logger.info(f"   Test Type: {result['test_type']}")
                logger.info(f"   Score: {result['similarity_score']:.4f}")
                if result.get('description'):
                    desc = result['description'][:100]
                    logger.info(f"   Description: {desc}...")
        
        elif args.command == 'stats':
            logger.info("=" * 60)
            logger.info("VECTOR DATABASE STATISTICS")
            logger.info("=" * 60)
            
            embedding_model = args.model or os.getenv('EMBEDDING_MODEL')
            embedding_generator = EmbeddingGenerator(model_name=embedding_model)
            embedding_dim = embedding_generator.get_embedding_dimension()
            
            vector_db = initialize_vector_db(embedding_dimension=embedding_dim)
            stats = vector_db.get_stats()
            
            logger.info(f"Total vectors: {stats.get('total_vectors', 0)}")
            logger.info(f"Dimension: {stats.get('dimension', 0)}")
            logger.info(f"Index fullness: {stats.get('index_fullness', 0)}")
            logger.info("=" * 60)
    
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()







