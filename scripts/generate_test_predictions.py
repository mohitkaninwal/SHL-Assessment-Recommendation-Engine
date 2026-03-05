#!/usr/bin/env python3
"""
Generate Test Set Predictions
Creates CSV file with predictions for test set in required format
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import load_dataset, Evaluator
from src.recommendation.recommend import RecommendationEngine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_csv_format(csv_file: str) -> bool:
    """
    Validate CSV format matches requirements (submission or lowercase).

    Accepted formats:
    - Columns: query, assessment_url  OR  Query, Assessment_url
    - Multiple rows per query (one per recommendation)
    """
    try:
        df = pd.read_csv(csv_file)
        cols = [c.strip() for c in df.columns]

        # Accept either submission or lowercase headers
        if cols == ["query", "assessment_url"]:
            qcol, acol = "query", "assessment_url"
        elif cols == ["Query", "Assessment_url"]:
            qcol, acol = "Query", "Assessment_url"
        else:
            logger.error(f"Invalid columns. Expected query/assessment_url or Query/Assessment_url, got {cols}")
            return False

        if df.isnull().any().any():
            logger.warning("CSV contains null values")

        duplicates = df.duplicated()
        if duplicates.any():
            logger.warning(f"CSV contains {duplicates.sum()} duplicate rows")

        logger.info("CSV Validation:")
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Unique queries: {df[qcol].nunique()}")
        logger.info(f"  Avg recommendations per query: {len(df) / df[qcol].nunique():.1f}")

        return True

    except Exception as e:
        logger.error(f"Error validating CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate test set predictions')
    parser.add_argument(
        '--dataset',
        type=str,
        default='Gen_AI Dataset.xlsx',
        help='Path to dataset Excel file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_predictions.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of recommendations per query (default: 10)'
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
        '--submission-format',
        action='store_true',
        default=True,
        help='Output CSV with headers Query,Assessment_url (submission format). Default: True.'
    )
    parser.add_argument(
        '--no-submission-format',
        action='store_false',
        dest='submission_format',
        help='Output CSV with headers query,assessment_url'
    )

    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("GENERATING TEST SET PREDICTIONS")
    logger.info("=" * 80)
    
    try:
        # Load test dataset
        logger.info(f"Loading test dataset from: {args.dataset}")
        queries, _ = load_dataset(args.dataset, sheet_name='Test-Set')
        logger.info(f"Loaded {len(queries)} test queries")
        
        # Initialize recommendation engine
        logger.info("Initializing recommendation engine...")
        engine = RecommendationEngine(
            use_rag=not args.no_rag,
            use_llm_reranking=not args.no_reranking,
            use_query_expansion=args.expand_query
        )
        
        # Initialize evaluator
        evaluator = Evaluator(engine)
        
        # Generate predictions
        logger.info(f"Generating predictions with top_k={args.top_k}...")
        df = evaluator.generate_csv_predictions(
            queries=queries,
            top_k=args.top_k,
            output_file=args.output
        )
        
        # Use submission headers (Query, Assessment_url) if requested
        if args.submission_format and not df.empty:
            df = df.rename(columns={'query': 'Query', 'assessment_url': 'Assessment_url'})
            df.to_csv(args.output, index=False)
            logger.info("CSV written with submission headers: Query, Assessment_url")

        # Validate CSV format
        logger.info("\nValidating CSV format...")
        is_valid = validate_csv_format(args.output)
        
        if is_valid:
            logger.info("✓ CSV format is valid")
        else:
            logger.error("✗ CSV format validation failed")
            return 1
        
        # Print summary
        print("\n" + "=" * 80)
        print("PREDICTION GENERATION COMPLETE")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Test Queries: {len(queries)}")
        print(f"Top K: {args.top_k}")
        print(f"\nConfiguration:")
        print(f"  RAG Pipeline: {'Enabled' if not args.no_rag else 'Disabled'}")
        print(f"  LLM Re-ranking: {'Enabled' if not args.no_reranking else 'Disabled'}")
        print(f"  Query Expansion: {'Enabled' if args.expand_query else 'Disabled'}")
        qcol = "Query" if "Query" in df.columns else "query"
        print(f"\nOutput:")
        print(f"  File: {args.output}")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique queries: {df[qcol].nunique()}")
        print(f"  Avg recommendations per query: {len(df) / df[qcol].nunique():.1f}")
        print(f"\nFormat: ✓ Valid" if is_valid else "\nFormat: ✗ Invalid")
        print("=" * 80)
        
        # Show sample
        print("\nSample predictions (first 10 rows):")
        print(df.head(10).to_string(index=False))
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
