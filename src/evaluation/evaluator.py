"""
Evaluator Module
Handles dataset loading, prediction generation, and evaluation
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .metrics import mean_recall_at_k, calculate_metrics, calculate_per_query_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str, sheet_name: str = 'Train-Set') -> Tuple[List[str], List[List[str]]]:
    """
    Load dataset from Excel file
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name ('Train-Set' or 'Test-Set')
        
    Returns:
        Tuple of (queries, ground_truth_urls)
        - queries: List of query strings
        - ground_truth_urls: List of lists of assessment URLs (only for Train-Set)
    """
    logger.info(f"Loading dataset from {file_path}, sheet: {sheet_name}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Loaded {len(df)} rows from {sheet_name} sheet")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Extract queries - handle both 'query' and 'Query'
        if 'Query' in df.columns:
            queries = df['Query'].tolist()
        elif 'query' in df.columns:
            queries = df['query'].tolist()
        else:
            raise ValueError(f"No 'query' or 'Query' column found in {sheet_name} sheet")
        
        # Extract ground truth URLs (only for Train-Set)
        ground_truth_urls = []
        if sheet_name == 'Train-Set':
            # Handle different column name variations
            if 'Assessment_url' in df.columns:
                url_column = 'Assessment_url'
            elif 'assessment_url' in df.columns:
                url_column = 'assessment_url'
            elif 'Assessment URL' in df.columns:
                url_column = 'Assessment URL'
            else:
                raise ValueError(f"No assessment URL column found in {sheet_name} sheet. Columns: {df.columns.tolist()}")
            
            # Group by query to get all URLs for each query
            query_to_urls = {}
            for _, row in df.iterrows():
                query = row['Query'] if 'Query' in df.columns else row['query']
                url = row[url_column]
                
                if pd.notna(url):  # Skip NaN values
                    if query not in query_to_urls:
                        query_to_urls[query] = []
                    query_to_urls[query].append(str(url).strip())
            
            # Get unique queries in order
            unique_queries = []
            seen = set()
            for q in queries:
                if q not in seen:
                    unique_queries.append(q)
                    seen.add(q)
            
            # Build ground truth list
            queries = unique_queries
            ground_truth_urls = [query_to_urls.get(q, []) for q in queries]
            
            logger.info(f"Loaded {len(queries)} unique queries with ground truth")
            for i, (q, urls) in enumerate(zip(queries, ground_truth_urls)):
                logger.info(f"  Query {i+1}: {len(urls)} ground truth URLs")
        else:
            # Test set has no ground truth - get unique queries
            unique_queries = []
            seen = set()
            for q in queries:
                if q not in seen:
                    unique_queries.append(q)
                    seen.add(q)
            queries = unique_queries
            ground_truth_urls = [[] for _ in queries]
            logger.info(f"Loaded {len(queries)} unique test queries (no ground truth)")
        
        return queries, ground_truth_urls
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


class Evaluator:
    """Evaluator for recommendation system"""
    
    def __init__(self, recommendation_engine):
        """
        Initialize evaluator
        
        Args:
            recommendation_engine: Instance of RecommendationEngine
        """
        self.engine = recommendation_engine
        self.results = {}
    
    def evaluate(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        top_k: int = 10,
        save_predictions: bool = True,
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Evaluate recommendation engine on dataset
        
        Args:
            queries: List of query strings
            ground_truth: List of lists of ground truth URLs
            top_k: Number of recommendations to generate
            save_predictions: Whether to save predictions
            output_file: Optional file to save predictions
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {len(queries)} queries with top_k={top_k}")
        
        # Generate predictions
        predictions = []
        prediction_details = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                result = self.engine.recommend(
                    query=query,
                    top_k=top_k,
                    balance_skills=True,
                    include_explanation=False
                )
                
                # Extract URLs
                pred_urls = [rec['assessment_url'] for rec in result['recommendations']]
                predictions.append(pred_urls)
                
                # Save details
                prediction_details.append({
                    'query': query,
                    'predictions': result['recommendations'],
                    'query_analysis': result.get('query_analysis', {})
                })
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                predictions.append([])
                prediction_details.append({
                    'query': query,
                    'predictions': [],
                    'error': str(e)
                })
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth, k_values=[5, 10])
        
        # Calculate per-query metrics
        per_query_metrics = calculate_per_query_metrics(
            predictions, ground_truth, queries, k=10
        )
        
        # Compile results
        results = {
            'metrics': metrics,
            'per_query_metrics': per_query_metrics,
            'predictions': prediction_details,
            'summary': {
                'total_queries': len(queries),
                'mean_recall@10': metrics.get('mean_recall@10', 0.0),
                'mean_precision@10': metrics.get('mean_precision@10', 0.0),
                'f1@10': metrics.get('f1@10', 0.0)
            }
        }
        
        self.results = results
        
        # Save predictions if requested
        if save_predictions and output_file:
            self.save_results(output_file)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Queries: {len(queries)}")
        logger.info(f"Mean Recall@10: {metrics.get('mean_recall@10', 0.0):.4f}")
        logger.info(f"Mean Recall@5: {metrics.get('mean_recall@5', 0.0):.4f}")
        logger.info(f"Mean Precision@10: {metrics.get('mean_precision@10', 0.0):.4f}")
        logger.info(f"F1@10: {metrics.get('f1@10', 0.0):.4f}")
        logger.info("=" * 60)
        
        return results
    
    def save_results(self, output_file: str):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def generate_csv_predictions(
        self,
        queries: List[str],
        top_k: int = 10,
        output_file: str = 'predictions.csv'
    ):
        """
        Generate predictions in CSV format for submission
        
        Args:
            queries: List of query strings
            top_k: Number of recommendations per query
            output_file: Output CSV file path
        """
        logger.info(f"Generating CSV predictions for {len(queries)} queries")
        
        rows = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            
            try:
                result = self.engine.recommend(
                    query=query,
                    top_k=top_k,
                    balance_skills=True
                )
                
                # Add each recommendation as a row
                for rec in result['recommendations']:
                    rows.append({
                        'query': query,
                        'assessment_url': rec['assessment_url']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"CSV predictions saved to {output_file}")
        logger.info(f"Total rows: {len(df)}")
        
        return df
    
    def analyze_errors(self) -> Dict:
        """
        Analyze prediction errors
        
        Returns:
            Dictionary with error analysis
        """
        if not self.results:
            logger.warning("No results to analyze. Run evaluate() first.")
            return {}
        
        per_query = self.results['per_query_metrics']
        
        # Find queries with low recall
        low_recall_queries = [
            q for q in per_query
            if q['recall@10'] < 0.5
        ]
        
        # Find queries with zero recall
        zero_recall_queries = [
            q for q in per_query
            if q['recall@10'] == 0.0
        ]
        
        # Find queries with perfect recall
        perfect_recall_queries = [
            q for q in per_query
            if q['recall@10'] == 1.0
        ]
        
        analysis = {
            'total_queries': len(per_query),
            'low_recall_count': len(low_recall_queries),
            'zero_recall_count': len(zero_recall_queries),
            'perfect_recall_count': len(perfect_recall_queries),
            'low_recall_queries': low_recall_queries[:5],  # Top 5
            'zero_recall_queries': zero_recall_queries[:5],
            'perfect_recall_queries': perfect_recall_queries[:5]
        }
        
        logger.info("=" * 60)
        logger.info("ERROR ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total Queries: {analysis['total_queries']}")
        logger.info(f"Low Recall (<0.5): {analysis['low_recall_count']}")
        logger.info(f"Zero Recall: {analysis['zero_recall_count']}")
        logger.info(f"Perfect Recall: {analysis['perfect_recall_count']}")
        logger.info("=" * 60)
        
        return analysis


if __name__ == "__main__":
    # Test dataset loading
    dataset_path = "Gen_AI Dataset.xlsx"
    
    if Path(dataset_path).exists():
        queries, ground_truth = load_dataset(dataset_path, sheet_name='Train')
        print(f"Loaded {len(queries)} queries")
        print(f"First query: {queries[0]}")
        print(f"Ground truth URLs: {len(ground_truth[0])}")
    else:
        print(f"Dataset not found: {dataset_path}")
