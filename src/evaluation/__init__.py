"""
Evaluation Module
Implements metrics and evaluation logic for recommendation system
"""

from .metrics import mean_recall_at_k, calculate_metrics
from .evaluator import Evaluator, load_dataset

__all__ = [
    'mean_recall_at_k',
    'calculate_metrics',
    'Evaluator',
    'load_dataset'
]
