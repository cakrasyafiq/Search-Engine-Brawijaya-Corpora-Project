import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import hybrid_service
from evaluation.dataset import EvaluationDataset, EvaluationQuery
from evaluation.metrics import (
    compute_query_metrics,
    aggregate_metrics,
    QueryEvaluationResult,
    AggregatedMetrics,
)
from evaluation.resource_monitor import ResourceMonitor, ResourceProfile

logger = logging.getLogger(__name__)


@dataclass
class MethodEvaluationReport:
    method_name: str
    dataset_name: str
    total_queries: int
    k: int
    alpha: Optional[float]
    # Metrics
    overall_metrics: AggregatedMetrics
    conceptual_metrics: AggregatedMetrics
    exact_metrics: AggregatedMetrics
    # Resources
    resources: ResourceProfile
    # Detailed per-query results
    query_results: List[QueryEvaluationResult]


class SearchEvaluator:
    """
    Executes information retrieval evaluation and resource monitoring
    for BM25, BGE-M3, and Hybrid search methods.
    """

    def __init__(self, db: Session, k: int = 5):
        self.db = db
        self.k = k

    def _execute_bm25_query(self, query_text: str) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results = search_service.search(query=query_text, top_k=self.k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        terms = [r.document.term for r in results[: self.k]]
        return terms, latency_ms

    def _execute_bge_m3_query(self, query_text: str) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results = semantic_service.search(db=self.db, query=query_text, top_k=self.k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        terms = [r[0].term for r in results[: self.k]]
        return terms, latency_ms

    def _execute_hybrid_query(
        self, query_text: str, alpha: float
    ) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results, _, _ = hybrid_service.search(
            db=self.db, query=query_text, top_k=self.k, alpha=alpha
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        terms = [r.term for r in results[: self.k]]
        return terms, latency_ms

    def evaluate_method(
        self,
        method_name: str,
        dataset: EvaluationDataset,
        dataset_name: str = "Test Set",
        alpha: Optional[float] = None,
        warmup: bool = True,
    ) -> MethodEvaluationReport:
        """
        Evaluate a single search method across all queries in dataset,
        monitoring computational cost and calculating P@k, R@k, MRR, NDCG@k.
        """
        logger.info(f"Evaluating {method_name} on {dataset_name} ({len(dataset)} queries)...")

        # Ensure index & models are loaded
        if not search_service.is_ready:
            search_service.initialize(self.db)
        if "bge" in method_name.lower() or "hybrid" in method_name.lower():
            _ = semantic_service.get_model()

        # Optional warmup to eliminate cold-start cache anomalies
        if warmup:
            warmup_q = "warmup query"
            if method_name == "BM25":
                _ = self._execute_bm25_query(warmup_q)
            elif "bge" in method_name.lower() or "semantic" in method_name.lower():
                _ = self._execute_bge_m3_query(warmup_q)
            else:
                _ = self._execute_hybrid_query(warmup_q, alpha=alpha if alpha is not None else 0.5)

        # Initialize Resource Monitor
        monitor = ResourceMonitor(db=self.db)
        monitor.start_monitoring()

        query_results: List[QueryEvaluationResult] = []
        conceptual_results: List[QueryEvaluationResult] = []
        exact_results: List[QueryEvaluationResult] = []
        latencies_ms: List[float] = []

        for q in dataset.queries:
            if method_name == "BM25":
                terms, lat_ms = self._execute_bm25_query(q.query_text)
            elif "bge" in method_name.lower() or "semantic" in method_name.lower():
                terms, lat_ms = self._execute_bge_m3_query(q.query_text)
            elif "hybrid" in method_name.lower():
                eff_alpha = alpha if alpha is not None else 0.5
                terms, lat_ms = self._execute_hybrid_query(q.query_text, alpha=eff_alpha)
            else:
                raise ValueError(f"Unknown search method: {method_name}")

            latencies_ms.append(lat_ms)
            res = compute_query_metrics(
                query_id=q.query_id,
                query_type=q.query_type,
                query_text=q.query_text,
                retrieved_terms=terms,
                ground_truth=q.ground_truth_labels,
                latency_ms=lat_ms,
                k=self.k,
            )
            query_results.append(res)
            if q.query_type.lower() == "conceptual":
                conceptual_results.append(res)
            else:
                exact_results.append(res)

        resources = monitor.stop_monitoring(
            latencies_ms=latencies_ms,
            method_name=method_name,
        )

        overall_metrics = aggregate_metrics(query_results)
        conceptual_metrics = aggregate_metrics(conceptual_results)
        exact_metrics = aggregate_metrics(exact_results)

        return MethodEvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            total_queries=len(dataset),
            k=self.k,
            alpha=alpha,
            overall_metrics=overall_metrics,
            conceptual_metrics=conceptual_metrics,
            exact_metrics=exact_metrics,
            resources=resources,
            query_results=query_results,
        )

    def evaluate_all(
        self,
        dataset: EvaluationDataset,
        dataset_name: str = "Test Set",
        optimal_alpha: float = 0.5,
    ) -> Dict[str, MethodEvaluationReport]:
        """
        Evaluate all three methods (BM25, BGE-M3, Hybrid) on the dataset.
        """
        reports: Dict[str, MethodEvaluationReport] = {}

        # 1. BM25
        reports["BM25"] = self.evaluate_method(
            method_name="BM25",
            dataset=dataset,
            dataset_name=dataset_name,
        )

        # 2. BGE-M3
        reports["BGE-M3"] = self.evaluate_method(
            method_name="BGE-M3",
            dataset=dataset,
            dataset_name=dataset_name,
        )

        # 3. Hybrid Search
        hybrid_name = f"Hybrid Search (alpha={optimal_alpha})"
        reports[hybrid_name] = self.evaluate_method(
            method_name="Hybrid Search",
            dataset=dataset,
            dataset_name=dataset_name,
            alpha=optimal_alpha,
        )

        return reports
