"""
Evaluation Metrics
Implements Mean Recall@K and other evaluation metrics
"""

import logging
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_recall_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Recall@K metric
    
    Recall@K for a single query = (# relevant items in top K) / (# total relevant items)
    Mean Recall@K = Average of Recall@K across all queries
    
    Args:
        predictions: List of predicted assessment URLs for each query (top K)
        ground_truth: List of ground truth assessment URLs for each query
        k: Number of top predictions to consider
        
    Returns:
        Mean Recall@K score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Mismatch: {len(predictions)} predictions but {len(ground_truth)} ground truth")
    
    if len(predictions) == 0:
        return 0.0
    
    recall_scores = []
    
    for pred_urls, true_urls in zip(predictions, ground_truth):
        # Convert to sets for efficient intersection
        pred_set = set(pred_urls[:k])  # Only consider top K
        true_set = set(true_urls)
        
        if len(true_set) == 0:
            # No ground truth for this query, skip
            logger.warning("Query has no ground truth labels")
            continue
        
        # Calculate recall for this query
        relevant_in_top_k = len(pred_set.intersection(true_set))
        total_relevant = len(true_set)
        
        recall = relevant_in_top_k / total_relevant
        recall_scores.append(recall)
    
    if len(recall_scores) == 0:
        return 0.0
    
    # Mean of all recall scores
    mean_recall = sum(recall_scores) / len(recall_scores)
    
    return mean_recall


def precision_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Precision@K metric
    
    Precision@K = (# relevant items in top K) / K
    
    Args:
        predictions: List of predicted assessment URLs for each query
        ground_truth: List of ground truth assessment URLs for each query
        k: Number of top predictions to consider
        
    Returns:
        Mean Precision@K score
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Mismatch: {len(predictions)} predictions but {len(ground_truth)} ground truth")
    
    if len(predictions) == 0:
        return 0.0
    
    precision_scores = []
    
    for pred_urls, true_urls in zip(predictions, ground_truth):
        pred_set = set(pred_urls[:k])
        true_set = set(true_urls)
        
        if len(pred_set) == 0:
            continue
        
        relevant_in_top_k = len(pred_set.intersection(true_set))
        precision = relevant_in_top_k / min(k, len(pred_set))
        precision_scores.append(precision)
    
    if len(precision_scores) == 0:
        return 0.0
    
    return sum(precision_scores) / len(precision_scores)


def calculate_metrics(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        predictions: List of predicted assessment URLs for each query
        ground_truth: List of ground truth assessment URLs for each query
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary of metric names and scores
    """
    metrics = {}
    
    for k in k_values:
        # Mean Recall@K
        recall = mean_recall_at_k(predictions, ground_truth, k=k)
        metrics[f'mean_recall@{k}'] = recall
        
        # Mean Precision@K
        precision = precision_at_k(predictions, ground_truth, k=k)
        metrics[f'mean_precision@{k}'] = precision
        
        # F1@K
        if recall + precision > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics[f'f1@{k}'] = f1
        else:
            metrics[f'f1@{k}'] = 0.0
    
    return metrics


def calculate_per_query_metrics(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    query_texts: List[str],
    k: int = 10
) -> List[Dict]:
    """
    Calculate metrics for each individual query
    
    Args:
        predictions: List of predicted assessment URLs for each query
        ground_truth: List of ground truth assessment URLs for each query
        query_texts: List of query text strings
        k: Number of top predictions to consider
        
    Returns:
        List of dictionaries with per-query metrics
    """
    if len(predictions) != len(ground_truth) or len(predictions) != len(query_texts):
        raise ValueError("Length mismatch between predictions, ground truth, and queries")
    
    per_query_results = []
    
    for i, (pred_urls, true_urls, query) in enumerate(zip(predictions, ground_truth, query_texts)):
        pred_set = set(pred_urls[:k])
        true_set = set(true_urls)
        
        relevant_in_top_k = len(pred_set.intersection(true_set))
        total_relevant = len(true_set)
        
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
        precision = relevant_in_top_k / k if k > 0 else 0.0
        
        result = {
            'query_index': i,
            'query': query,
            f'recall@{k}': recall,
            f'precision@{k}': precision,
            'relevant_found': relevant_in_top_k,
            'total_relevant': total_relevant,
            'predicted_count': len(pred_urls[:k])
        }
        
        per_query_results.append(result)
    
    return per_query_results


if __name__ == "__main__":
    # Test metrics
    predictions = [
        ['url1', 'url2', 'url3', 'url4', 'url5'],
        ['url6', 'url7', 'url8', 'url9', 'url10']
    ]
    
    ground_truth = [
        ['url1', 'url3', 'url11', 'url12'],  # 2 out of 4 found
        ['url6', 'url7', 'url13']  # 2 out of 3 found
    ]
    
    recall = mean_recall_at_k(predictions, ground_truth, k=5)
    print(f"Mean Recall@5: {recall:.3f}")
    # Expected: (2/4 + 2/3) / 2 = (0.5 + 0.667) / 2 = 0.583
