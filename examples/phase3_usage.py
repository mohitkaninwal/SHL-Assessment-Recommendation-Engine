#!/usr/bin/env python3
"""
Example usage of Phase 3: Embedding & Retrieval System
Demonstrates how to use the components for semantic search
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommendation.embeddings import EmbeddingGenerator
from src.recommendation.vector_db import VectorDB
from src.recommendation.retriever import AssessmentRetriever, QueryPreprocessor


def example_1_embedding_generation():
    """Example 1: Generate embeddings for text"""
    print("\n" + "="*60)
    print("Example 1: Embedding Generation")
    print("="*60)
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Generate single embedding
    text = "Java programming skills assessment"
    embedding = generator.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"Embedding dimension: {embedding.shape[0]}")
    print(f"First 5 values: {embedding[:5]}")


def example_2_query_preprocessing():
    """Example 2: Preprocess queries"""
    print("\n" + "="*60)
    print("Example 2: Query Preprocessing")
    print("="*60)
    
    preprocessor = QueryPreprocessor()
    
    # Test queries
    queries = [
        "Looking for Java developer with strong teamwork skills",
        "Python programmer needed for data analysis",
        "Leadership and communication assessment required"
    ]
    
    for query in queries:
        print(f"\nOriginal: {query}")
        
        # Clean query
        cleaned = preprocessor.clean_query(query)
        print(f"Cleaned: {cleaned}")
        
        # Extract keywords
        keywords = preprocessor.extract_keywords(query)
        print(f"Keywords: {keywords}")
        
        # Detect test type
        test_type = preprocessor.detect_test_type_preference(query)
        print(f"Detected type: {test_type} ({'Knowledge/Skills' if test_type == 'K' else 'Personality/Behavior' if test_type == 'P' else 'Mixed'})")


def example_3_semantic_search():
    """Example 3: Semantic search (requires Pinecone setup)"""
    print("\n" + "="*60)
    print("Example 3: Semantic Search")
    print("="*60)
    
    try:
        # Initialize components
        vector_db = VectorDB(dimension=384)
        embedding_gen = EmbeddingGenerator()
        retriever = AssessmentRetriever(vector_db, embedding_gen)
        
        # Test query
        query = "Java developer with good communication skills"
        print(f"Query: {query}")
        
        # Simple retrieval
        print("\n--- Simple Retrieval (top 5) ---")
        results = retriever.retrieve(query, top_k=5)
        
        formatted = retriever.format_results(results)
        for i, result in enumerate(formatted, 1):
            print(f"{i}. {result['assessment_name']}")
            print(f"   Score: {result['similarity_score']:.3f}")
            print(f"   Type: {result['test_type']}")
            print(f"   URL: {result['assessment_url']}")
            print()
        
        # Balanced retrieval
        print("\n--- Balanced Retrieval (60% K, 40% P) ---")
        balanced_results = retriever.retrieve_balanced(
            query,
            top_k=10,
            hard_skill_ratio=0.6
        )
        
        formatted_balanced = retriever.format_results(balanced_results)
        
        k_count = sum(1 for r in formatted_balanced if r['test_type'] == 'K')
        p_count = sum(1 for r in formatted_balanced if r['test_type'] == 'P')
        
        print(f"Retrieved: {k_count} Knowledge/Skills (K), {p_count} Personality/Behavior (P)")
        
        for i, result in enumerate(formatted_balanced, 1):
            print(f"{i}. [{result['test_type']}] {result['assessment_name']} ({result['similarity_score']:.3f})")
        
    except ValueError as e:
        if "PINECONE_API_KEY" in str(e):
            print("⚠ Pinecone not configured. Set PINECONE_API_KEY in .env file")
            print("  Get your key from: https://www.pinecone.io/")
        else:
            raise
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  Make sure you've indexed assessments first:")
        print("  python -m src.recommendation.indexer --catalog data/raw_catalog.json")


def example_4_filter_by_type():
    """Example 4: Filter results by test type"""
    print("\n" + "="*60)
    print("Example 4: Filter by Test Type")
    print("="*60)
    
    try:
        vector_db = VectorDB(dimension=384)
        embedding_gen = EmbeddingGenerator()
        retriever = AssessmentRetriever(vector_db, embedding_gen)
        
        query = "Software developer assessment"
        
        # Get only Knowledge/Skills assessments
        print(f"Query: {query}")
        print("\n--- Only Knowledge/Skills (K) ---")
        k_results = retriever.retrieve(query, top_k=5, test_type_filter='K')
        
        for i, result in enumerate(retriever.format_results(k_results), 1):
            print(f"{i}. {result['assessment_name']} ({result['similarity_score']:.3f})")
        
        # Get only Personality/Behavior assessments
        print("\n--- Only Personality/Behavior (P) ---")
        p_results = retriever.retrieve(query, top_k=5, test_type_filter='P')
        
        for i, result in enumerate(retriever.format_results(p_results), 1):
            print(f"{i}. {result['assessment_name']} ({result['similarity_score']:.3f})")
        
    except Exception as e:
        print(f"⚠ Error: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("PHASE 3: EMBEDDING & RETRIEVAL - USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Always works (no API key needed)
    example_1_embedding_generation()
    
    # Example 2: Always works (no API key needed)
    example_2_query_preprocessing()
    
    # Example 3: Requires Pinecone setup
    example_3_semantic_search()
    
    # Example 4: Requires Pinecone setup
    example_4_filter_by_type()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
