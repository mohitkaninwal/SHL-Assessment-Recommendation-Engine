#!/usr/bin/env python3
"""
Test script for Phase 4: LLM Integration & Recommendation Engine
Tests Groq API, RAG pipeline, and recommendation logic
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.recommendation.llm_client import LLMClient
from src.recommendation.embeddings import EmbeddingGenerator
from src.recommendation.vector_db import VectorDB
from src.recommendation.rag_pipeline import RAGPipeline
from src.recommendation.recommend import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_client():
    """Test 1: LLM Client (Groq API)"""
    print("\n" + "="*60)
    print("TEST 1: LLM Client (Groq API)")
    print("="*60)
    
    try:
        client = LLMClient()
        
        # Test simple generation
        prompt = "What is Python?"
        response = client.generate(prompt, max_tokens=50)
        print(f"✓ LLM Model: {client.model}")
        print(f"✓ Test prompt: {prompt}")
        print(f"✓ Response: {response[:100]}...")
        
        return True
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print(f"⚠ Groq API key not configured: {e}")
            print("⚠ Set GROQ_API_KEY in .env file to test LLM")
            print("⚠ Get your key from: https://console.groq.com/")
            return None
        raise
    except Exception as e:
        print(f"✗ LLM client test failed: {e}")
        return False


def test_query_classification():
    """Test 2: Query Intent Classification"""
    print("\n" + "="*60)
    print("TEST 2: Query Intent Classification")
    print("="*60)
    
    try:
        client = LLMClient()
        
        test_queries = [
            "Java programming skills assessment",
            "Team collaboration and leadership evaluation",
            "Python developer with good communication skills"
        ]
        
        for query in test_queries:
            result = client.classify_query_intent(query)
            print(f"\n✓ Query: '{query}'")
            print(f"  Technical: {result['technical_weight']:.2f}")
            print(f"  Behavioral: {result['behavioral_weight']:.2f}")
            print(f"  Skills: {', '.join(result['primary_skills'][:3])}")
        
        return True
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print("⚠ Groq API key not configured")
            return None
        raise
    except Exception as e:
        print(f"✗ Query classification failed: {e}")
        return False


def test_query_expansion():
    """Test 3: Query Expansion"""
    print("\n" + "="*60)
    print("TEST 3: Query Expansion")
    print("="*60)
    
    try:
        client = LLMClient()
        
        query = "Java developer"
        expanded = client.expand_query(query)
        
        print(f"✓ Original: {query}")
        print(f"✓ Expanded: {expanded[:200]}...")
        
        return True
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print("⚠ Groq API key not configured")
            return None
        raise
    except Exception as e:
        print(f"✗ Query expansion failed: {e}")
        return False


def test_rag_pipeline():
    """Test 4: RAG Pipeline"""
    print("\n" + "="*60)
    print("TEST 4: RAG Pipeline")
    print("="*60)
    
    try:
        # Check if vector DB is configured
        try:
            vector_db = VectorDB(dimension=384)
            stats = vector_db.get_stats()
            
            if stats.get('total_vectors', 0) == 0:
                print("⚠ Vector database is empty")
                print("⚠ Run indexer first: python -m src.recommendation.indexer --catalog data/raw_catalog.json")
                return None
        except ValueError as e:
            if "PINECONE_API_KEY" in str(e):
                print("⚠ Pinecone not configured")
                return None
            raise
        
        # Initialize RAG pipeline
        embedding_gen = EmbeddingGenerator()
        llm_client = LLMClient()
        
        pipeline = RAGPipeline(
            vector_db=vector_db,
            embedding_generator=embedding_gen,
            llm_client=llm_client,
            use_llm_reranking=True,
            use_query_expansion=False
        )
        
        # Test recommendation
        query = "Java developer with teamwork skills"
        result = pipeline.recommend(
            query=query,
            top_k=5,
            balance_skills=True,
            include_explanation=True
        )
        
        print(f"✓ Query: {query}")
        print(f"✓ Found: {result['total_found']} assessments")
        print(f"✓ Returned: {result['returned']} recommendations")
        print(f"✓ Technical weight: {result['query_analysis']['technical_weight']:.2f}")
        print(f"✓ Behavioral weight: {result['query_analysis']['behavioral_weight']:.2f}")
        
        print(f"\n✓ Top 3 Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. [{rec['test_type']}] {rec['assessment_name']} ({rec['similarity_score']:.3f})")
        
        if result.get('explanation'):
            print(f"\n✓ Explanation: {result['explanation'][:150]}...")
        
        return True
    except ValueError as e:
        if "GROQ_API_KEY" in str(e) or "PINECONE_API_KEY" in str(e):
            print("⚠ API keys not configured")
            return None
        raise
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recommendation_engine():
    """Test 5: Recommendation Engine"""
    print("\n" + "="*60)
    print("TEST 5: Recommendation Engine")
    print("="*60)
    
    try:
        engine = RecommendationEngine(
            use_rag=True,
            use_llm_reranking=True,
            use_query_expansion=False
        )
        
        query = "Python programmer with leadership skills"
        result = engine.recommend(
            query=query,
            top_k=5,
            balance_skills=True,
            include_explanation=True
        )
        
        print(f"✓ Engine initialized")
        print(f"✓ Query: {query}")
        print(f"✓ Recommendations: {len(result['recommendations'])}")
        
        k_count = sum(1 for r in result['recommendations'] if r.get('test_type') == 'K')
        p_count = sum(1 for r in result['recommendations'] if r.get('test_type') == 'P')
        print(f"✓ Balance: {k_count} Knowledge/Skills, {p_count} Personality/Behavior")
        
        return True
    except ValueError as e:
        if "GROQ_API_KEY" in str(e) or "PINECONE_API_KEY" in str(e):
            print("⚠ API keys not configured")
            return None
        raise
    except Exception as e:
        print(f"✗ Recommendation engine test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 4: LLM INTEGRATION & RECOMMENDATION ENGINE - TEST SUITE")
    print("="*60)
    
    results = {
        'LLM Client': test_llm_client(),
        'Query Classification': test_query_classification(),
        'Query Expansion': test_query_expansion(),
        'RAG Pipeline': test_rag_pipeline(),
        'Recommendation Engine': test_recommendation_engine()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP (not configured)"
        print(f"{status}: {test_name}")
    
    # Overall status
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1
    elif passed == 0:
        print("\n⚠ No tests passed. Check your configuration.")
        print("\nRequired:")
        print("1. GROQ_API_KEY in .env file (get from https://console.groq.com/)")
        print("2. PINECONE_API_KEY in .env file")
        print("3. Indexed assessments (run: python -m src.recommendation.indexer --catalog data/raw_catalog.json)")
        return 1
    else:
        print("\n✓ Phase 4 components are working!")
        print("\nNext steps:")
        print("1. Test with sample queries: python -m src.recommendation.recommend --query 'Java developer'")
        print("2. Move to Phase 5: Evaluation & Optimization")
        return 0


if __name__ == "__main__":
    sys.exit(main())
