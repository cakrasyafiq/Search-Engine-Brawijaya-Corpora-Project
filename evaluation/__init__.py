"""
Evaluation package for Information Retrieval search methods:
BM25, BGE-M3 (Semantic Search), and Hybrid Search.
"""

from evaluation.dataset import EvaluationDataset, load_and_split_dataset
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_query_metrics,
    aggregate_metrics,
)
from evaluation.resource_monitor import ResourceMonitor, StorageAuditor
from evaluation.alpha_optimizer import AlphaOptimizer
from evaluation.evaluator import SearchEvaluator

__all__ = [
    "EvaluationDataset",
    "load_and_split_dataset",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "compute_query_metrics",
    "aggregate_metrics",
    "ResourceMonitor",
    "StorageAuditor",
    "AlphaOptimizer",
    "SearchEvaluator",
]
