#!/usr/bin/env python3
"""
Iteration Tracking Script
Compares evaluation results across iterations to track improvements
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_iteration_results(results_dir: str) -> list:
    """Load all iteration results from directory"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []
    
    iterations = []
    
    # Find all result files
    for config_file in sorted(results_dir.glob('*_config.json')):
        iteration_name = config_file.stem.replace('_config', '')
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load results
        results_file = results_dir / f'{iteration_name}_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            logger.warning(f"Results file not found for {iteration_name}")
            continue
        
        # Load error analysis
        error_file = results_dir / f'{iteration_name}_error_analysis.json'
        if error_file.exists():
            with open(error_file, 'r') as f:
                error_analysis = json.load(f)
        else:
            error_analysis = {}
        
        iterations.append({
            'name': iteration_name,
            'config': config,
            'metrics': results.get('metrics', {}),
            'summary': results.get('summary', {}),
            'error_analysis': error_analysis
        })
    
    return iterations


def compare_iterations(iterations: list) -> pd.DataFrame:
    """Create comparison table of iterations"""
    
    if not iterations:
        return pd.DataFrame()
    
    rows = []
    
    for it in iterations:
        row = {
            'Iteration': it['name'],
            'Timestamp': it['config'].get('timestamp', 'N/A'),
            'RAG': 'Yes' if it['config'].get('use_rag', True) else 'No',
            'Reranking': 'Yes' if it['config'].get('use_llm_reranking', True) else 'No',
            'Query Expansion': 'Yes' if it['config'].get('use_query_expansion', False) else 'No',
            'Recall@10': it['metrics'].get('mean_recall@10', 0.0),
            'Recall@5': it['metrics'].get('mean_recall@5', 0.0),
            'Precision@10': it['metrics'].get('mean_precision@10', 0.0),
            'F1@10': it['metrics'].get('f1@10', 0.0),
            'Zero Recall': it['error_analysis'].get('zero_recall_count', 0),
            'Perfect Recall': it['error_analysis'].get('perfect_recall_count', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table"""
    
    if df.empty:
        print("No iterations found")
        return
    
    print("\n" + "=" * 120)
    print("ITERATION COMPARISON")
    print("=" * 120)
    
    # Format numeric columns
    numeric_cols = ['Recall@10', 'Recall@5', 'Precision@10', 'F1@10']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    print(df.to_string(index=False))
    print("=" * 120)
    
    # Find best iteration
    df_numeric = df.copy()
    for col in numeric_cols:
        if col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col])
    
    best_idx = df_numeric['Recall@10'].idxmax()
    best_iteration = df.iloc[best_idx]
    
    print(f"\nBest Iteration: {best_iteration['Iteration']}")
    print(f"  Recall@10: {best_iteration['Recall@10']}")
    print(f"  Configuration: RAG={best_iteration['RAG']}, Reranking={best_iteration['Reranking']}, Query Expansion={best_iteration['Query Expansion']}")
    print("=" * 120)


def generate_improvement_report(iterations: list, output_file: str = None):
    """Generate detailed improvement report"""
    
    if not iterations:
        logger.warning("No iterations to report")
        return
    
    report = []
    report.append("=" * 80)
    report.append("ITERATION IMPROVEMENT REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Iterations: {len(iterations)}")
    report.append("")
    
    # Sort by timestamp
    iterations_sorted = sorted(iterations, key=lambda x: x['config'].get('timestamp', ''))
    
    for i, it in enumerate(iterations_sorted, 1):
        report.append(f"\nIteration {i}: {it['name']}")
        report.append("-" * 80)
        report.append(f"Timestamp: {it['config'].get('timestamp', 'N/A')}")
        report.append(f"\nConfiguration:")
        report.append(f"  RAG Pipeline: {'Enabled' if it['config'].get('use_rag', True) else 'Disabled'}")
        report.append(f"  LLM Re-ranking: {'Enabled' if it['config'].get('use_llm_reranking', True) else 'Disabled'}")
        report.append(f"  Query Expansion: {'Enabled' if it['config'].get('use_query_expansion', False) else 'Disabled'}")
        report.append(f"\nMetrics:")
        report.append(f"  Mean Recall@10: {it['metrics'].get('mean_recall@10', 0.0):.4f}")
        report.append(f"  Mean Recall@5:  {it['metrics'].get('mean_recall@5', 0.0):.4f}")
        report.append(f"  Mean Precision@10: {it['metrics'].get('mean_precision@10', 0.0):.4f}")
        report.append(f"  F1@10: {it['metrics'].get('f1@10', 0.0):.4f}")
        
        if it['error_analysis']:
            report.append(f"\nError Analysis:")
            report.append(f"  Zero Recall: {it['error_analysis'].get('zero_recall_count', 0)}")
            report.append(f"  Low Recall (<0.5): {it['error_analysis'].get('low_recall_count', 0)}")
            report.append(f"  Perfect Recall: {it['error_analysis'].get('perfect_recall_count', 0)}")
        
        # Calculate improvement from previous iteration
        if i > 1:
            prev_it = iterations_sorted[i-2]
            prev_recall = prev_it['metrics'].get('mean_recall@10', 0.0)
            curr_recall = it['metrics'].get('mean_recall@10', 0.0)
            improvement = curr_recall - prev_recall
            
            report.append(f"\nImprovement from previous:")
            report.append(f"  Recall@10 change: {improvement:+.4f} ({improvement/prev_recall*100:+.1f}%)" if prev_recall > 0 else "  Recall@10 change: N/A")
    
    report.append("\n" + "=" * 80)
    
    # Print report
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Track and compare evaluation iterations')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='evaluation_results',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for improvement report'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Export comparison table to CSV'
    )
    
    args = parser.parse_args()
    
    try:
        # Load all iterations
        logger.info(f"Loading iterations from: {args.results_dir}")
        iterations = load_iteration_results(args.results_dir)
        
        if not iterations:
            logger.error("No iterations found")
            return 1
        
        logger.info(f"Found {len(iterations)} iterations")
        
        # Create comparison table
        df = compare_iterations(iterations)
        
        # Print comparison
        print_comparison(df)
        
        # Export to CSV if requested
        if args.csv:
            df.to_csv(args.csv, index=False)
            logger.info(f"Comparison table saved to: {args.csv}")
        
        # Generate improvement report
        generate_improvement_report(iterations, output_file=args.output)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
