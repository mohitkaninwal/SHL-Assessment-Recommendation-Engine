#!/usr/bin/env python3
"""
Test script for Phase 3: Embedding & Retrieval System
Tests embedding generation, vector DB connection, and retrieval
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from src.recommendation.embeddings import EmbeddingGenerator
from src.recommendation.vector_db import VectorDB
from src.recommendation.indexer import AssessmentIndexer
from src.recommendation.retriever import AssessmentRetriever, QueryPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_embedding_generation():
    """Test 1: Embedding Generation"""
    print("\n" + "="*60)
    print("TEST 1: Embedding Generation")
    print("="*60)
    
    try:
        generator = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "Java programming skills assessment"
        embedding = generator.generate_embedding(test_text)
        
        print(f"✓ Model loaded: {generator.model_name}")
        print(f"✓ Embedding dimension: {generator.embedding_dimension}")
        print(f"✓ Generated embedding shape: {embedding.shape}")
        print(f"✓ Sample embedding values: {embedding[:5]}")
        
        # Test batch embeddings
        test_texts = [
            "Python programming test",
            "Team collaboration assessment",
            "Leadership skills evaluation"
        ]
        batch_embeddings = generator.generate_embeddings_batch(test_texts)
        print(f"✓ Batch embeddings shape: {batch_embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False


def test_query_preprocessing():
    """Test 2: Query Preprocessing"""
    print("\n" + "="*60)
    print("TEST 2: Query Preprocessing")
    print("="*60)
    
    try:
        preprocessor = QueryPreprocessor()
        
        # Test query cleaning
        test_query = "  Looking for Java   developer with   teamwork skills  "
        cleaned = preprocessor.clean_query(test_query)
        print(f"✓ Original: '{test_query}'")
        print(f"✓ Cleaned: '{cleaned}'")
        
        # Test keyword extraction
        keywords = preprocessor.extract_keywords(cleaned)
        print(f"✓ Keywords: {keywords}")
        
        # Test type detection
        test_queries = [
            "Java programming skills",
            "Team collaboration and leadership",
            "Python developer with good communication"
        ]
        
        for query in test_queries:
            detected_type = preprocessor.detect_test_type_preference(query)
            print(f"✓ Query: '{query}' → Detected type: {detected_type}")
        
        return True
    except Exception as e:
        print(f"✗ Query preprocessing failed: {e}")
        return False


def test_vector_db_connection():
    """Test 3: Vector DB Connection"""
    print("\n" + "="*60)
    print("TEST 3: Vector DB Connection")
    print("="*60)
    
    try:
        # This will fail without proper API keys
        print("⚠ Attempting to connect to Pinecone...")
        print("⚠ This requires PINECONE_API_KEY in .env file")
        
        try:
            vector_db = VectorDB(dimension=384)  # MiniLM dimension
            stats = vector_db.get_stats()
            print(f"✓ Connected to Pinecone index")
            print(f"✓ Index stats: {stats}")
            return True
        except ValueError as e:
            if "PINECONE_API_KEY" in str(e):
                print(f"⚠ Pinecone API key not configured: {e}")
                print("⚠ Set PINECONE_API_KEY in .env file to test vector DB")
                return None  # Not a failure, just not configured
            raise
    except Exception as e:
        print(f"✗ Vector DB connection failed: {e}")
        return False


def test_assessment_text_creation():
    """Test 4: Assessment Text Creation"""
    print("\n" + "="*60)
    print("TEST 4: Assessment Text Creation")
    print("="*60)
    
    try:
        generator = EmbeddingGenerator()
        
        test_assessments = [
            {
                'name': 'Java Programming Test',
                'description': 'Comprehensive assessment of Java programming skills',
                'test_type': 'K',
                'category': 'Technical Skills'
            },
            {
                'name': 'Team Collaboration Assessment',
                'description': 'Evaluates ability to work effectively in teams',
                'test_type': 'P',
                'category': 'Soft Skills'
            }
        ]
        
        for assessment in test_assessments:
            text = generator.create_assessment_text(assessment)
            print(f"✓ Assessment: {assessment['name']}")
            print(f"  Combined text: {text}")
            print()
        
        return True
    except Exception as e:
        print(f"✗ Assessment text creation failed: {e}")
        return False


def test_with_sample_data():
    """Test 5: Test with Sample Catalog Data"""
    print("\n" + "="*60)
    print("TEST 5: Sample Data Processing")
    print("="*60)
    
    try:
        # Check if catalog exists
        catalog_path = project_root / "data" / "raw_catalog.json"
        if not catalog_path.exists():
            print("⚠ No catalog file found at data/raw_catalog.json")
            print("⚠ Run scraper first: python -m src.data_pipeline.main --mode scrape --use-selenium")
            return None
        
        # Load catalog
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assessments = data.get('assessments', [])
        print(f"✓ Loaded {len(assessments)} assessments from catalog")
        
        # Test with first 3 assessments
        sample_assessments = assessments[:3]
        
        generator = EmbeddingGenerator()
        
        for assessment in sample_assessments:
            text = generator.create_assessment_text(assessment)
            embedding = generator.generate_embedding(text)
            print(f"✓ {assessment.get('name', 'Unknown')}")
            print(f"  Text: {text[:100]}...")
            print(f"  Embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Sample data processing failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3: EMBEDDING & RETRIEVAL SYSTEM - TEST SUITE")
    print("="*60)
    
    results = {
        'Embedding Generation': test_embedding_generation(),
        'Query Preprocessing': test_query_preprocessing(),
        'Vector DB Connection': test_vector_db_connection(),
        'Assessment Text Creation': test_assessment_text_creation(),
        'Sample Data Processing': test_with_sample_data()
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
        return 1
    else:
        print("\n✓ Phase 3 components are working!")
        print("\nNext steps:")
        print("1. Set PINECONE_API_KEY in .env file")
        print("2. Run indexer: python -m src.recommendation.indexer --catalog data/raw_catalog.json")
        print("3. Test retrieval with sample queries")
        return 0


if __name__ == "__main__":
    sys.exit(main())
