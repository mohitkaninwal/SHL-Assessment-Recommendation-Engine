#!/usr/bin/env python3
"""
Phase 5 Usage Examples
Demonstrates how to use the evaluation framework
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import Evaluator, load_dataset
from src.evaluation.metrics import mean_recall_at_k, calculate_metrics
from src.recommendation.recommend import RecommendationEngine


def example_1_load_dataset():
    """Example 1: Load and inspect dataset"""
    print("=" * 80)
    print("Example 1: Load Dataset")
    print("=" * 80)
    
    # Load train set
    queries, ground_truth = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    
    print(f"\nTrain Set:")
    print(f"  Queries: {len(queries)}")
    print(f"\nFirst query:")
    print(f"  Query: {queries[0][:80]}...")
    print(f"  Ground truth URLs: {len(ground_truth[0])}")
    for i, url in enumerate(ground_truth[0][:3], 1):
        print(f"    {i}. {url}")
    
    # Load test set
    test_queries, _ = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Test-Set')
    print(f"\nTest Set:")
    print(f"  Queries: {len(test_queries)}")
    print(f"\nFirst test query:")
    print(f"  {test_queries[0][:80]}...")


def example_2_evaluate_single_query():
    """Example 2: Evaluate a single query"""
    print("\n" + "=" * 80)
    print("Example 2: Evaluate Single Query")
    print("=" * 80)
    
    # Initialize engine
    engine = RecommendationEngine(use_rag=True)
    
    # Single query
    query = "I am hiring for Java developers who can also collaborate effectively"
    
    print(f"\nQuery: {query}")
    
    # Get recommendations
    result = engine.recommend(query=query, top_k=10, include_explanation=False)
    
    print(f"\nRecommendations: {len(result['recommendations'])}")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"{i}. {rec['assessment_name']}")
        print(f"   Type: {rec['test_type']}, Score: {rec['similarity_score']:.3f}")
        print(f"   URL: {rec['assessment_url']}")
    
    # Ground truth (example)
    ground_truth_urls = [
        'https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/',
        'https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/',
        'https://www.shl.com/solutions/products/product-catalog/view/java-8-new/'
    ]
    
    # Calculate recall
    pred_urls = [rec['assessment_url'] for rec in result['recommendations']]
    recall = mean_recall_at_k([pred_urls], [ground_truth_urls], k=10)
    
    print(f"\nRecall@10: {recall:.4f}")


def example_3_full_evaluation():
    """Example 3: Full evaluation on train set"""
    print("\n" + "=" * 80)
    print("Example 3: Full Evaluation")
    print("=" * 80)
    
    # Load dataset
    queries, ground_truth = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    
    # Initialize
    engine = RecommendationEngine(use_rag=True)
    evaluator = Evaluator(engine)
    
    print(f"\nEvaluating {len(queries)} queries...")
    
    # Run evaluation
    results = evaluator.evaluate(
        queries=queries,
        ground_truth=ground_truth,
        top_k=10,
        save_predictions=False
    )
    
    # Print results
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Show per-query results
    print("\nPer-Query Results:")
    for i, qm in enumerate(results['per_query_metrics'][:3], 1):
        print(f"\n{i}. {qm['query'][:60]}...")
        print(f"   Recall@10: {qm['recall@10']:.4f}")
        print(f"   Found: {qm['relevant_found']}/{qm['total_relevant']}")


def example_4_compare_configurations():
    """Example 4: Compare different configurations"""
    print("\n" + "=" * 80)
    print("Example 4: Compare Configurations")
    print("=" * 80)
    
    # Load dataset
    queries, ground_truth = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    
    # Test different configurations
    configs = [
        {'name': 'RAG + Re-ranking', 'use_rag': True, 'use_llm_reranking': True},
        {'name': 'RAG without Re-ranking', 'use_rag': True, 'use_llm_reranking': False},
        {'name': 'Simple Retrieval', 'use_rag': False, 'use_llm_reranking': False}
    ]
    
    results_comparison = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Initialize engine
        engine = RecommendationEngine(
            use_rag=config['use_rag'],
            use_llm_reranking=config['use_llm_reranking']
        )
        
        evaluator = Evaluator(engine)
        
        # Evaluate
        results = evaluator.evaluate(
            queries=queries,
            ground_truth=ground_truth,
            top_k=10,
            save_predictions=False
        )
        
        recall = results['metrics']['mean_recall@10']
        print(f"  Mean Recall@10: {recall:.4f}")
        
        results_comparison.append({
            'config': config['name'],
            'recall': recall
        })
    
    # Print comparison
    print("\n" + "-" * 80)
    print("Comparison:")
    for r in results_comparison:
        print(f"  {r['config']}: {r['recall']:.4f}")


def example_5_generate_test_predictions():
    """Example 5: Generate test set predictions"""
    print("\n" + "=" * 80)
    print("Example 5: Generate Test Predictions")
    print("=" * 80)
    
    # Load test queries
    queries, _ = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Test-Set')
    
    print(f"\nGenerating predictions for {len(queries)} test queries...")
    
    # Initialize
    engine = RecommendationEngine(use_rag=True)
    evaluator = Evaluator(engine)
    
    # Generate CSV
    df = evaluator.generate_csv_predictions(
        queries=queries,
        top_k=10,
        output_file='example_test_predictions.csv'
    )
    
    print(f"\nGenerated {len(df)} predictions")
    print(f"Unique queries: {df['query'].nunique()}")
    print(f"Avg per query: {len(df) / df['query'].nunique():.1f}")
    
    print("\nSample predictions:")
    print(df.head(10).to_string(index=False))


def example_6_error_analysis():
    """Example 6: Detailed error analysis"""
    print("\n" + "=" * 80)
    print("Example 6: Error Analysis")
    print("=" * 80)
    
    # Load dataset
    queries, ground_truth = load_dataset('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
    
    # Initialize and evaluate
    engine = RecommendationEngine(use_rag=True)
    evaluator = Evaluator(engine)
    
    results = evaluator.evaluate(
        queries=queries,
        ground_truth=ground_truth,
        top_k=10,
        save_predictions=False
    )
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors()
    
    print("\nError Analysis:")
    print(f"  Total queries: {error_analysis['total_queries']}")
    print(f"  Zero recall: {error_analysis['zero_recall_count']}")
    print(f"  Low recall (<0.5): {error_analysis['low_recall_count']}")
    print(f"  Perfect recall: {error_analysis['perfect_recall_count']}")
    
    # Show problematic queries
    if error_analysis['zero_recall_queries']:
        print("\nQueries with zero recall:")
        for q in error_analysis['zero_recall_queries'][:3]:
            print(f"  - {q['query'][:60]}...")
    
    if error_analysis['perfect_recall_queries']:
        print("\nQueries with perfect recall:")
        for q in error_analysis['perfect_recall_queries'][:3]:
            print(f"  - {q['query'][:60]}...")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("PHASE 5 USAGE EXAMPLES")
    print("=" * 80)
    
    try:
        # Run examples
        example_1_load_dataset()
        
        # Uncomment to run other examples (they require API keys and indexed data)
        # example_2_evaluate_single_query()
        # example_3_full_evaluation()
        # example_4_compare_configurations()
        # example_5_generate_test_predictions()
        # example_6_error_analysis()
        
        print("\n" + "=" * 80)
        print("Examples completed!")
        print("=" * 80)
        print("\nTo run full evaluation:")
        print("  python scripts/evaluate.py")
        print("\nTo generate test predictions:")
        print("  python scripts/generate_test_predictions.py")
        print("\nTo track iterations:")
        print("  python scripts/track_iterations.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
