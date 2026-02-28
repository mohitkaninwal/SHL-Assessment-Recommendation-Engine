#!/usr/bin/env python3
"""
Evaluation Script for Phase 5
Evaluates the recommendation system on the train set
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import Evaluator, load_dataset
from src.recommendation.recommend import RecommendationEngine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate recommendation system')
    parser.add_argument(
        '--dataset',
        type=str,
        default='Gen_AI Dataset.xlsx',
        help='Path to dataset Excel file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of recommendations to generate (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--iteration',
        type=str,
        help='Iteration name/number for tracking improvements'
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate iteration name if not provided
    if args.iteration:
        iteration_name = args.iteration
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        iteration_name = f'eval_{timestamp}'
    
    logger.info("=" * 80)
    logger.info(f"EVALUATION - {iteration_name}")
    logger.info("=" * 80)
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset}")
        queries, ground_truth = load_dataset(args.dataset, sheet_name='Train-Set')
        logger.info(f"Loaded {len(queries)} queries with ground truth")
        
        # Initialize recommendation engine
        logger.info("Initializing recommendation engine...")
        engine = RecommendationEngine(
            use_rag=not args.no_rag,
            use_llm_reranking=not args.no_reranking,
            use_query_expansion=args.expand_query
        )
        
        # Initialize evaluator
        evaluator = Evaluator(engine)
        
        # Run evaluation
        logger.info(f"Running evaluation with top_k={args.top_k}...")
        results = evaluator.evaluate(
            queries=queries,
            ground_truth=ground_truth,
            top_k=args.top_k,
            save_predictions=True,
            output_file=str(output_dir / f'{iteration_name}_results.json')
        )
        
        # Analyze errors
        logger.info("\nAnalyzing errors...")
        error_analysis = evaluator.analyze_errors()
        
        # Save error analysis
        with open(output_dir / f'{iteration_name}_error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Save configuration
        config = {
            'iteration': iteration_name,
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'top_k': args.top_k,
            'use_rag': not args.no_rag,
            'use_llm_reranking': not args.no_reranking,
            'use_query_expansion': args.expand_query,
            'num_queries': len(queries)
        }
        
        with open(output_dir / f'{iteration_name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Iteration: {iteration_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Queries: {len(queries)}")
        print(f"Top K: {args.top_k}")
        print(f"\nConfiguration:")
        print(f"  RAG Pipeline: {'Enabled' if not args.no_rag else 'Disabled'}")
        print(f"  LLM Re-ranking: {'Enabled' if not args.no_reranking else 'Disabled'}")
        print(f"  Query Expansion: {'Enabled' if args.expand_query else 'Disabled'}")
        print(f"\nMetrics:")
        print(f"  Mean Recall@10: {results['metrics']['mean_recall@10']:.4f}")
        print(f"  Mean Recall@5:  {results['metrics']['mean_recall@5']:.4f}")
        print(f"  Mean Precision@10: {results['metrics']['mean_precision@10']:.4f}")
        print(f"  F1@10: {results['metrics']['f1@10']:.4f}")
        print(f"\nError Analysis:")
        print(f"  Low Recall (<0.5): {error_analysis['low_recall_count']}/{error_analysis['total_queries']}")
        print(f"  Zero Recall: {error_analysis['zero_recall_count']}/{error_analysis['total_queries']}")
        print(f"  Perfect Recall: {error_analysis['perfect_recall_count']}/{error_analysis['total_queries']}")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
        # Save summary to file
        summary_file = output_dir / f'{iteration_name}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Iteration: {iteration_name}\n")
            f.write(f"Timestamp: {config['timestamp']}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Queries: {len(queries)}\n")
            f.write(f"Top K: {args.top_k}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  RAG Pipeline: {'Enabled' if not args.no_rag else 'Disabled'}\n")
            f.write(f"  LLM Re-ranking: {'Enabled' if not args.no_reranking else 'Disabled'}\n")
            f.write(f"  Query Expansion: {'Enabled' if args.expand_query else 'Disabled'}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  Mean Recall@10: {results['metrics']['mean_recall@10']:.4f}\n")
            f.write(f"  Mean Recall@5:  {results['metrics']['mean_recall@5']:.4f}\n")
            f.write(f"  Mean Precision@10: {results['metrics']['mean_precision@10']:.4f}\n")
            f.write(f"  F1@10: {results['metrics']['f1@10']:.4f}\n")
            f.write(f"\nError Analysis:\n")
            f.write(f"  Low Recall (<0.5): {error_analysis['low_recall_count']}/{error_analysis['total_queries']}\n")
            f.write(f"  Zero Recall: {error_analysis['zero_recall_count']}/{error_analysis['total_queries']}\n")
            f.write(f"  Perfect Recall: {error_analysis['perfect_recall_count']}/{error_analysis['total_queries']}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
