"""
URL normalization utilities for evaluation.
"""

from typing import List, Set, Tuple


LEGACY_URL_ALIASES = {
    # Sales legacy variants
    "https://www.shl.com/products/product-catalog/view/entry-level-sales-7-1":
        "https://www.shl.com/products/product-catalog/view/entry-level-sales-solution",
    "https://www.shl.com/products/product-catalog/view/entry-level-sales-sift-out-7-1":
        "https://www.shl.com/products/product-catalog/view/entry-level-sales-solution",
    "https://www.shl.com/products/product-catalog/view/sales-representative-solution":
        "https://www.shl.com/products/product-catalog/view/entry-level-sales-solution",
    "https://www.shl.com/products/product-catalog/view/technical-sales-associate-solution":
        "https://www.shl.com/products/product-catalog/view/sales-and-service-phone-solution",
    # Admin/banking legacy variants
    "https://www.shl.com/products/product-catalog/view/administrative-professional-short-form":
        "https://www.shl.com/products/product-catalog/view/workplace-administration-skills-new",
    "https://www.shl.com/products/product-catalog/view/bank-administrative-assistant-short-form":
        "https://www.shl.com/products/product-catalog/view/workplace-administration-skills-new",
    "https://www.shl.com/products/product-catalog/view/financial-professional-short-form":
        "https://www.shl.com/products/product-catalog/view/financial-and-banking-services-new",
    "https://www.shl.com/products/product-catalog/view/general-entry-level-data-entry-7-0-solution":
        "https://www.shl.com/products/product-catalog/view/basic-computer-literacy-windows-10-new",
    # Legacy manager/professional solution variants
    "https://www.shl.com/products/product-catalog/view/manager-8-0-jfa-4310":
        "https://www.shl.com/products/product-catalog/view/sales-transformation-report-sales-manager",
    "https://www.shl.com/products/product-catalog/view/professional-7-0-solution-3958":
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment",
    "https://www.shl.com/products/product-catalog/view/professional-7-1-solution":
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment",
}


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
    normalized = LEGACY_URL_ALIASES.get(normalized, normalized)
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
