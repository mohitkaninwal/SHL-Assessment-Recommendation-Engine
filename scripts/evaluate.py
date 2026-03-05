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
        '--sheet',
        type=str,
        default='Train-Set',
        choices=['Train-Set', 'Test-Set'],
        help='Sheet to use for evaluation. Use Test-Set to compute recall@10 on test data (sheet must have Assessment_url column).'
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
    parser.add_argument(
        '--min-recall',
        type=float,
        default=None,
        metavar='R',
        help='Evaluation criteria: exit with failure if mean Recall@10 < R (e.g. 0.5 or 0.6)'
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
        # Load dataset (Train-Set or Test-Set; recall@10 is computed when sheet has Assessment_url)
        logger.info(f"Loading dataset from: {args.dataset}, sheet: {args.sheet}")
        queries, ground_truth = load_dataset(args.dataset, sheet_name=args.sheet)
        has_gt = any(len(gt) > 0 for gt in ground_truth)
        logger.info(f"Loaded {len(queries)} queries" + (" with ground truth" if has_gt else " (no ground truth in sheet; recall will be N/A)"))

        # Catalog coverage sanity check (assignment requirement: 377+ assessments).
        catalog_path = Path("data/processed_catalog.json")
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
            assessment_count = len(catalog_data.get("assessments", []))
            logger.info("Catalog assessment count: %s", assessment_count)
            if assessment_count < 377:
                logger.warning("Catalog has fewer than 377 assessments. Current count: %s", assessment_count)
        else:
            logger.warning("Catalog file not found at data/processed_catalog.json")
        
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
            'sheet': args.sheet,
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
        print(f"Sheet: {args.sheet}")
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
        retrieval_stage = results.get('stage_metrics', {}).get('retrieval', {})
        if retrieval_stage.get('available'):
            retrieval_metrics = retrieval_stage.get('metrics', {})
            print(f"  Retrieval Mean Recall@10: {retrieval_metrics.get('mean_recall@10', 0.0):.4f}")
            print(f"  Retrieval Mean Recall@5:  {retrieval_metrics.get('mean_recall@5', 0.0):.4f}")
        print(f"\nError Analysis:")
        print(f"  Low Recall (<0.5): {error_analysis['low_recall_count']}/{error_analysis['total_queries']}")
        print(f"  Zero Recall: {error_analysis['zero_recall_count']}/{error_analysis['total_queries']}")
        print(f"  Perfect Recall: {error_analysis['perfect_recall_count']}/{error_analysis['total_queries']}")

        # Evaluation criteria: optional minimum Recall@10 threshold
        mean_recall10 = results['metrics']['mean_recall@10']
        if args.min_recall is not None:
            criteria_ok = mean_recall10 >= args.min_recall
            print(f"\nEvaluation criteria (Recall@10 >= {args.min_recall}): {'PASS' if criteria_ok else 'FAIL'} (got {mean_recall10:.4f})")
            if not criteria_ok:
                print("=" * 80)
                logger.warning("Evaluation criteria not met: mean Recall@10 below threshold")
                return 1

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
            f.write(f"Sheet: {args.sheet}\n")
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
            retrieval_stage = results.get('stage_metrics', {}).get('retrieval', {})
            if retrieval_stage.get('available'):
                retrieval_metrics = retrieval_stage.get('metrics', {})
                f.write(f"  Retrieval Mean Recall@10: {retrieval_metrics.get('mean_recall@10', 0.0):.4f}\n")
                f.write(f"  Retrieval Mean Recall@5:  {retrieval_metrics.get('mean_recall@5', 0.0):.4f}\n")
            f.write(f"\nError Analysis:\n")
            f.write(f"  Low Recall (<0.5): {error_analysis['low_recall_count']}/{error_analysis['total_queries']}\n")
            f.write(f"  Zero Recall: {error_analysis['zero_recall_count']}/{error_analysis['total_queries']}\n")
            f.write(f"  Perfect Recall: {error_analysis['perfect_recall_count']}/{error_analysis['total_queries']}\n")
            if args.min_recall is not None:
                f.write(f"\nEvaluation criteria (Recall@10 >= {args.min_recall}): {'PASS' if mean_recall10 >= args.min_recall else 'FAIL'}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
