"""
URL normalization utilities for evaluation.
"""

from typing import List, Set, Tuple


def canonicalize_assessment_url(url: str) -> str:
    """
    Canonicalize SHL assessment URLs for fair matching.

    This handles known equivalent URL variants in the dataset/crawled catalog,
    especially `/solutions/products/...` vs `/products/...`.
    """
    if not url:
        return ""

    normalized = url.strip().rstrip("/")
    normalized = normalized.replace(
        "https://www.shl.com/solutions/products/product-catalog/view/",
        "https://www.shl.com/products/product-catalog/view/",
    )
    return normalized


def canonicalize_url_lists(url_lists: List[List[str]]) -> List[List[str]]:
    """Canonicalize nested URL lists."""
    return [[canonicalize_assessment_url(url) for url in urls] for urls in url_lists]


def unique_url_overlap(predictions: List[List[str]], ground_truth: List[List[str]]) -> Tuple[int, int, int]:
    """
    Compute overlap counts between unique predicted and ground-truth URLs.

    Returns:
        Tuple of (unique_predicted, unique_ground_truth, unique_overlap)
    """
    pred_set: Set[str] = {u for urls in predictions for u in urls if u}
    gt_set: Set[str] = {u for urls in ground_truth for u in urls if u}
    return len(pred_set), len(gt_set), len(pred_set.intersection(gt_set))
