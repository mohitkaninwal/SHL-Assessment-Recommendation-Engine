"""
Evaluator Module
Handles dataset loading, prediction generation, and evaluation
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from .metrics import mean_recall_at_k, calculate_metrics, calculate_per_query_metrics
from .url_utils import canonicalize_url_lists, unique_url_overlap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str, sheet_name: str = 'Train-Set') -> Tuple[List[str], List[List[str]]]:
    """
    Load dataset from Excel file.

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name ('Train-Set' or 'Test-Set')

    Returns:
        Tuple of (queries, ground_truth_urls)
        - queries: List of query strings
        - ground_truth_urls: List of lists of assessment URLs per query.
          Loaded from Assessment_url (or variant) if present in the sheet;
          otherwise empty lists (recall cannot be computed).
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
        
        # Get unique queries in order (same for both branches)
        unique_queries = []
        seen = set()
        for q in queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)
        queries = unique_queries

        # Extract ground truth URLs from any sheet that has an assessment URL column
        # (Train-Set and Test-Set can both have labels for computing recall@10 on test data)
        ground_truth_urls = []
        if 'Assessment_url' in df.columns:
            url_column = 'Assessment_url'
        elif 'assessment_url' in df.columns:
            url_column = 'assessment_url'
        elif 'Assessment URL' in df.columns:
            url_column = 'Assessment URL'
        else:
            url_column = None

        if url_column is not None:
            query_to_urls = {}
            for _, row in df.iterrows():
                query = row['Query'] if 'Query' in df.columns else row['query']
                url = row[url_column]
                if pd.notna(url):
                    if query not in query_to_urls:
                        query_to_urls[query] = []
                    query_to_urls[query].append(str(url).strip())
            ground_truth_urls = [query_to_urls.get(q, []) for q in queries]
            logger.info(f"Loaded {len(queries)} unique queries with ground truth from {sheet_name}")
            for i, (q, urls) in enumerate(zip(queries, ground_truth_urls)):
                logger.info(f"  Query {i+1}: {len(urls)} ground truth URLs")
        else:
            ground_truth_urls = [[] for _ in queries]
            logger.info(f"Loaded {len(queries)} unique queries from {sheet_name} (no Assessment_url column; recall cannot be computed)")
        
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

    def _extract_stage_retrieval_urls(
        self,
        query: str,
        top_k: int,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[List[str]]:
        """
        Extract retrieval-stage URLs (pre-LLM re-ranking) when retriever is available.

        Returns:
            List of URLs if stage is available, otherwise None.
        """
        retriever = None

        # RAG path
        pipeline = getattr(self.engine, 'pipeline', None)
        if pipeline is not None and getattr(pipeline, 'retriever', None) is not None:
            retriever = pipeline.retriever
        # Non-RAG path
        elif getattr(self.engine, 'retriever', None) is not None:
            retriever = self.engine.retriever

        if retriever is None:
            return None

        hard_skill_ratio = 0.6
        if query_analysis and 'technical_weight' in query_analysis:
            try:
                hard_skill_ratio = float(query_analysis.get('technical_weight', 0.6))
            except (TypeError, ValueError):
                hard_skill_ratio = 0.6

        raw_results = retriever.retrieve_balanced(
            query=query,
            top_k=top_k,
            hard_skill_ratio=hard_skill_ratio,
            min_score=-1.0
        )
        formatted = retriever.format_results(raw_results)
        return [item.get('assessment_url', '') for item in formatted[:top_k] if item.get('assessment_url')]
    
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
        final_predictions = []
        retrieval_predictions = []
        prediction_details = []
        retrieval_available = True
        
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
                final_predictions.append(pred_urls)

                retrieval_urls = self._extract_stage_retrieval_urls(
                    query=query,
                    top_k=top_k,
                    query_analysis=result.get('query_analysis', {})
                )
                if retrieval_urls is None:
                    retrieval_available = False
                    retrieval_predictions.append([])
                else:
                    retrieval_predictions.append(retrieval_urls)
                
                # Save details
                prediction_details.append({
                    'query': query,
                    'predictions': result['recommendations'],
                    'query_analysis': result.get('query_analysis', {}),
                    'retrieval_stage_urls': retrieval_urls if retrieval_urls is not None else []
                })
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                final_predictions.append([])
                retrieval_predictions.append([])
                prediction_details.append({
                    'query': query,
                    'predictions': [],
                    'error': str(e)
                })
        
        # Canonicalize URLs before scoring to avoid false negatives from known
        # equivalent URL variants in dataset/catalog.
        canonical_final_predictions = canonicalize_url_lists(final_predictions)
        canonical_retrieval_predictions = canonicalize_url_lists(retrieval_predictions)
        canonical_ground_truth = canonicalize_url_lists(ground_truth)
        unique_pred, unique_gt, unique_overlap = unique_url_overlap(
            canonical_final_predictions, canonical_ground_truth
        )

        # Calculate stage-wise metrics
        final_metrics = calculate_metrics(canonical_final_predictions, canonical_ground_truth, k_values=[5, 10])
        final_per_query_metrics = calculate_per_query_metrics(
            canonical_final_predictions, canonical_ground_truth, queries, k=10
        )

        stage_metrics: Dict[str, Dict[str, Any]] = {
            'final_recommendation': {
                'available': True,
                'metrics': final_metrics
            }
        }
        stage_per_query: Dict[str, List[Dict[str, Any]]] = {
            'final_recommendation': final_per_query_metrics
        }

        if retrieval_available:
            retrieval_metrics = calculate_metrics(
                canonical_retrieval_predictions, canonical_ground_truth, k_values=[5, 10]
            )
            retrieval_per_query_metrics = calculate_per_query_metrics(
                canonical_retrieval_predictions, canonical_ground_truth, queries, k=10
            )
            stage_metrics['retrieval'] = {
                'available': True,
                'metrics': retrieval_metrics
            }
            stage_per_query['retrieval'] = retrieval_per_query_metrics
        else:
            stage_metrics['retrieval'] = {
                'available': False,
                'metrics': {}
            }
            stage_per_query['retrieval'] = []
        
        # Compile results
        results = {
            # Backward-compatible aliases: default metrics = final recommendation stage.
            'metrics': final_metrics,
            'per_query_metrics': final_per_query_metrics,
            'stage_metrics': stage_metrics,
            'stage_per_query_metrics': stage_per_query,
            'predictions': prediction_details,
            'summary': {
                'total_queries': len(queries),
                'mean_recall@10': final_metrics.get('mean_recall@10', 0.0),
                'mean_precision@10': final_metrics.get('mean_precision@10', 0.0),
                'f1@10': final_metrics.get('f1@10', 0.0),
                'retrieval_mean_recall@10': stage_metrics['retrieval']['metrics'].get('mean_recall@10', 0.0),
                'final_mean_recall@10': final_metrics.get('mean_recall@10', 0.0)
            },
            'url_normalization': {
                'enabled': True,
                'unique_predicted_urls': unique_pred,
                'unique_ground_truth_urls': unique_gt,
                'unique_overlap_urls': unique_overlap
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
        logger.info(f"Final Mean Recall@10: {final_metrics.get('mean_recall@10', 0.0):.4f}")
        logger.info(f"Final Mean Recall@5: {final_metrics.get('mean_recall@5', 0.0):.4f}")
        logger.info(f"Final Mean Precision@10: {final_metrics.get('mean_precision@10', 0.0):.4f}")
        logger.info(f"Final F1@10: {final_metrics.get('f1@10', 0.0):.4f}")
        if retrieval_available:
            retrieval_metrics = stage_metrics['retrieval']['metrics']
            logger.info(f"Retrieval Mean Recall@10: {retrieval_metrics.get('mean_recall@10', 0.0):.4f}")
            logger.info(f"Retrieval Mean Recall@5: {retrieval_metrics.get('mean_recall@5', 0.0):.4f}")
        logger.info(
            "URL normalization overlap: %d/%d unique GT URLs",
            unique_overlap,
            unique_gt,
        )
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
