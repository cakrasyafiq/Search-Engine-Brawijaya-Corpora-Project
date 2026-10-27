import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


def precision_at_k(
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    k: int = 5,
    min_relevance: int = 1,
) -> float:
    """
    Calculate Precision@k: proportion of retrieved documents in the top-k
    that are relevant (relevance label >= min_relevance).

    Formula: Precision@k = |Retrieved@k ∩ Relevant| / k
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_terms[:k]
    relevant_retrieved = sum(
        1 for term in top_k if ground_truth.get(term, 0) >= min_relevance
    )
    return relevant_retrieved / float(k)


def recall_at_k(
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    k: int = 5,
    min_relevance: int = 1,
) -> float:
    """
    Calculate Recall@k: proportion of all known relevant documents
    that are retrieved in the top-k results.

    Formula: Recall@k = |Retrieved@k ∩ Relevant| / |Total Relevant|
    """
    total_relevant = sum(
        1 for label in ground_truth.values() if label >= min_relevance
    )
    if total_relevant == 0:
        return 0.0

    top_k = retrieved_terms[:k]
    relevant_retrieved = sum(
        1 for term in top_k if ground_truth.get(term, 0) >= min_relevance
    )
    return relevant_retrieved / float(total_relevant)


def reciprocal_rank(
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    k: Optional[int] = 5,
    min_relevance: int = 1,
) -> float:
    """
    Calculate Reciprocal Rank (RR): reciprocal of the rank position of the
    FIRST relevant document (1-indexed).

    If k is specified, only examines results up to rank k.
    If no relevant document is found in the examined range, returns 0.0.
    """
    candidates = retrieved_terms[:k] if k is not None else retrieved_terms
    for rank, term in enumerate(candidates, start=1):
        if ground_truth.get(term, 0) >= min_relevance:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    k: int = 5,
) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG@k) using standard exponential gain:
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    top_k = retrieved_terms[:k]
    dcg = 0.0
    for i, term in enumerate(top_k, start=1):
        rel = ground_truth.get(term, 0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1.0)
        dcg += gain / discount
    return dcg


def idcg_at_k(
    ground_truth: Dict[str, int],
    k: int = 5,
) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain (IDCG@k):
    DCG@k of an ideal ranking with the highest relevance labels at top positions.
    """
    sorted_rels = sorted(ground_truth.values(), reverse=True)
    ideal_top_k = sorted_rels[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_top_k, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1.0)
        idcg += gain / discount
    return idcg


def ndcg_at_k(
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    k: int = 5,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k):
    NDCG@k = DCG@k / IDCG@k
    Returns 0.0 if IDCG@k == 0.
    """
    idcg = idcg_at_k(ground_truth, k=k)
    if idcg <= 0.0:
        return 0.0
    dcg = dcg_at_k(retrieved_terms, ground_truth, k=k)
    return dcg / idcg


@dataclass
class QueryEvaluationResult:
    query_id: str
    query_type: str
    query_text: str
    retrieved_terms: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float
    k: int = 5


def compute_query_metrics(
    query_id: str,
    query_type: str,
    query_text: str,
    retrieved_terms: Sequence[str],
    ground_truth: Dict[str, int],
    latency_ms: float = 0.0,
    k: int = 5,
) -> QueryEvaluationResult:
    """Compute all evaluation metrics for a single query."""
    p_k = precision_at_k(retrieved_terms, ground_truth, k=k)
    r_k = recall_at_k(retrieved_terms, ground_truth, k=k)
    rr = reciprocal_rank(retrieved_terms, ground_truth, k=k)
    ndcg = ndcg_at_k(retrieved_terms, ground_truth, k=k)

    return QueryEvaluationResult(
        query_id=query_id,
        query_type=query_type,
        query_text=query_text,
        retrieved_terms=list(retrieved_terms[:k]),
        precision_at_k=round(p_k, 4),
        recall_at_k=round(r_k, 4),
        reciprocal_rank=round(rr, 4),
        ndcg_at_k=round(ndcg, 4),
        latency_ms=round(latency_ms, 3),
        k=k,
    )


@dataclass
class AggregatedMetrics:
    total_queries: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_latency_ms: float
    k: int = 5


def aggregate_metrics(results: Sequence[QueryEvaluationResult]) -> AggregatedMetrics:
    """Aggregate individual query evaluation metrics into macro-averages."""
    if not results:
        return AggregatedMetrics(
            total_queries=0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            mean_latency_ms=0.0,
            k=5,
        )

    n = len(results)
    k_val = results[0].k if results else 5
    avg_p = sum(r.precision_at_k for r in results) / n
    avg_r = sum(r.recall_at_k for r in results) / n
    avg_mrr = sum(r.reciprocal_rank for r in results) / n
    avg_ndcg = sum(r.ndcg_at_k for r in results) / n
    avg_latency = sum(r.latency_ms for r in results) / n

    return AggregatedMetrics(
        total_queries=n,
        precision_at_k=round(avg_p, 4),
        recall_at_k=round(avg_r, 4),
        mrr=round(avg_mrr, 4),
        ndcg_at_k=round(avg_ndcg, 4),
        mean_latency_ms=round(avg_latency, 3),
        k=k_val,
    )
