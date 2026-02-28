#!/usr/bin/env python3
"""
Example usage of Phase 4: LLM Integration & Recommendation Engine
Demonstrates RAG pipeline and intelligent recommendations
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommendation.llm_client import LLMClient
from src.recommendation.recommend import RecommendationEngine


def example_1_query_classification():
    """Example 1: Classify query intent"""
    print("\n" + "="*60)
    print("Example 1: Query Intent Classification")
    print("="*60)
    
    try:
        client = LLMClient()
        
        queries = [
            "Senior Java developer with 5+ years experience",
            "Team leader with excellent communication skills",
            "Full-stack developer who can work in agile teams"
        ]
        
        for query in queries:
            result = client.classify_query_intent(query)
            print(f"\nQuery: {query}")
            print(f"Technical: {result['technical_weight']:.1%}")
            print(f"Behavioral: {result['behavioral_weight']:.1%}")
            print(f"Primary Skills: {', '.join(result['primary_skills'][:3])}")
            
    except ValueError as e:
        print(f"⚠ Error: {e}")
        print("Set GROQ_API_KEY in .env file")


def example_2_simple_recommendation():
    """Example 2: Simple recommendation"""
    print("\n" + "="*60)
    print("Example 2: Simple Recommendation")
    print("="*60)
    
    try:
        engine = RecommendationEngine(use_rag=True)
        
        query = "Java developer with problem-solving skills"
        result = engine.recommend(
            query=query,
            top_k=5,
            balance_skills=True
        )
        
        print(f"Query: {query}")
        print(f"Found: {result['total_found']} assessments")
        print(f"\nTop {len(result['recommendations'])} Recommendations:")
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. [{rec['test_type']}] {rec['assessment_name']}")
            print(f"   Score: {rec['similarity_score']:.3f}")
            print(f"   URL: {rec['assessment_url']}")
            
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_3_with_explanation():
    """Example 3: Recommendation with explanation"""
    print("\n" + "="*60)
    print("Example 3: Recommendation with Explanation")
    print("="*60)
    
    try:
        engine = RecommendationEngine(
            use_rag=True,
            use_llm_reranking=True
        )
        
        query = "Python data scientist with teamwork abilities"
        result = engine.recommend(
            query=query,
            top_k=5,
            balance_skills=True,
            include_explanation=True
        )
        
        print(f"Query: {query}")
        print(f"\nQuery Analysis:")
        analysis = result['query_analysis']
        print(f"  Technical: {analysis['technical_weight']:.1%}")
        print(f"  Behavioral: {analysis['behavioral_weight']:.1%}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"{i}. {rec['assessment_name']} ({rec['test_type']})")
        
        if result.get('explanation'):
            print(f"\nExplanation:")
            print(f"  {result['explanation']}")
            
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_4_batch_recommendations():
    """Example 4: Batch recommendations"""
    print("\n" + "="*60)
    print("Example 4: Batch Recommendations")
    print("="*60)
    
    try:
        engine = RecommendationEngine(use_rag=True)
        
        queries = [
            "Java backend developer",
            "Frontend React developer",
            "DevOps engineer with automation skills"
        ]
        
        results = engine.batch_recommend(
            queries=queries,
            top_k=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Query: {result['query']}")
            print(f"   Recommendations: {len(result['recommendations'])}")
            for j, rec in enumerate(result['recommendations'], 1):
                print(f"   {j}. {rec['assessment_name']} ({rec['test_type']})")
                
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_5_no_llm_comparison():
    """Example 5: Compare with and without LLM"""
    print("\n" + "="*60)
    print("Example 5: With vs Without LLM")
    print("="*60)
    
    try:
        query = "Software engineer with leadership potential"
        
        # Without LLM
        print("\n--- Without LLM (Simple Retrieval) ---")
        engine_simple = RecommendationEngine(use_rag=False)
        result_simple = engine_simple.recommend(query, top_k=5)
        
        print(f"Found: {result_simple['total_found']} assessments")
        for i, rec in enumerate(result_simple['recommendations'][:3], 1):
            print(f"{i}. {rec['assessment_name']} ({rec['test_type']})")
        
        # With LLM
        print("\n--- With LLM (RAG Pipeline) ---")
        engine_rag = RecommendationEngine(use_rag=True)
        result_rag = engine_rag.recommend(query, top_k=5, include_explanation=True)
        
        print(f"Found: {result_rag['total_found']} assessments")
        print(f"Technical: {result_rag['query_analysis']['technical_weight']:.1%}")
        print(f"Behavioral: {result_rag['query_analysis']['behavioral_weight']:.1%}")
        
        for i, rec in enumerate(result_rag['recommendations'][:3], 1):
            print(f"{i}. {rec['assessment_name']} ({rec['test_type']})")
        
        if result_rag.get('explanation'):
            print(f"\nExplanation: {result_rag['explanation'][:100]}...")
            
    except Exception as e:
        print(f"⚠ Error: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("PHASE 4: LLM INTEGRATION - USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Always works if Groq API key is set
    example_1_query_classification()
    
    # Example 2: Requires Pinecone + indexed data
    example_2_simple_recommendation()
    
    # Example 3: Full RAG with explanation
    example_3_with_explanation()
    
    # Example 4: Batch processing
    example_4_batch_recommendations()
    
    # Example 5: Comparison
    example_5_no_llm_comparison()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nTo run your own queries:")
    print("python -m src.recommendation.recommend --query 'Your query here' --explain")


if __name__ == "__main__":
    main()
