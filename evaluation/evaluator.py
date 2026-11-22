import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.search.tokenizer import normalize_text
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import hybrid_service
from app.services.bge_sparse_service import bge_sparse_service
from app.services.bge_hybrid_service import bge_hybrid_service
from evaluation.dataset import EvaluationDataset, EvaluationQuery
from evaluation.metrics import (
    compute_query_metrics,
    aggregate_metrics,
    QueryEvaluationResult,
    AggregatedMetrics,
)
from evaluation.resource_monitor import ResourceMonitor, ResourceProfile, LatencySummary

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
    hybrid_metrics: AggregatedMetrics
    # Resources (None for external baselines like BCP Lexical Search)
    resources: Optional[ResourceProfile]
    # Detailed per-query results
    query_results: List[QueryEvaluationResult]
    # Optional metadata (e.g. strict vs normalized baseline details)
    metadata: Optional[Dict[str, Any]] = None


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

    def _execute_bge_sparse_query(self, query_text: str) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results = bge_sparse_service.search(
            query=query_text, top_k=self.k, db=self.db
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        terms = [r.document.term for r in results[: self.k]]
        return terms, latency_ms

    def _execute_bge_hybrid_query(
        self, query_text: str, alpha: float
    ) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results, _, _ = bge_hybrid_service.search(
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

        m_low = method_name.lower()

        # Ensure index & models are loaded
        if not search_service.is_ready:
            search_service.initialize(self.db)
        if "bge" in m_low or "hybrid" in m_low or "semantic" in m_low:
            _ = semantic_service.get_model()
        if ("sparse" in m_low and "bge" in m_low) or ("hybrid" in m_low and "bge" in m_low):
            if not bge_sparse_service.is_ready:
                bge_sparse_service.initialize(self.db)
            _ = semantic_service.get_sparse_linear()

        # Optional warmup to eliminate cold-start cache anomalies
        if warmup:
            warmup_q = "warmup query"
            if method_name == "BM25":
                _ = self._execute_bm25_query(warmup_q)
            elif "sparse" in m_low and "bge" in m_low:
                _ = self._execute_bge_sparse_query(warmup_q)
            elif "hybrid" in m_low and "bge" in m_low:
                _ = self._execute_bge_hybrid_query(warmup_q, alpha=alpha if alpha is not None else 0.5)
            elif "bge" in m_low or "semantic" in m_low:
                _ = self._execute_bge_m3_query(warmup_q)
            else:
                _ = self._execute_hybrid_query(warmup_q, alpha=alpha if alpha is not None else 0.5)

        # Initialize Resource Monitor
        monitor = ResourceMonitor(db=self.db)
        monitor.start_monitoring()

        query_results: List[QueryEvaluationResult] = []
        conceptual_results: List[QueryEvaluationResult] = []
        exact_results: List[QueryEvaluationResult] = []
        hybrid_results: List[QueryEvaluationResult] = []
        latencies_ms: List[float] = []

        for q in dataset.queries:
            if method_name == "BM25":
                terms, lat_ms = self._execute_bm25_query(q.query_text)
            elif "sparse" in m_low and "bge" in m_low:
                terms, lat_ms = self._execute_bge_sparse_query(q.query_text)
            elif "hybrid" in m_low and "bge" in m_low:
                eff_alpha = alpha if alpha is not None else 0.5
                terms, lat_ms = self._execute_bge_hybrid_query(q.query_text, alpha=eff_alpha)
            elif "bge" in m_low or "semantic" in m_low:
                terms, lat_ms = self._execute_bge_m3_query(q.query_text)
            elif "hybrid" in m_low:
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
            qt_low = q.query_type.lower()
            if qt_low == "conceptual":
                conceptual_results.append(res)
            elif qt_low == "hybrid":
                hybrid_results.append(res)
            else:
                exact_results.append(res)

        resources = monitor.stop_monitoring(
            latencies_ms=latencies_ms,
            method_name=method_name,
        )

        overall_metrics = aggregate_metrics(query_results)
        conceptual_metrics = aggregate_metrics(conceptual_results)
        exact_metrics = aggregate_metrics(exact_results)
        hybrid_metrics = aggregate_metrics(hybrid_results)

        return MethodEvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            total_queries=len(dataset),
            k=self.k,
            alpha=alpha,
            overall_metrics=overall_metrics,
            conceptual_metrics=conceptual_metrics,
            exact_metrics=exact_metrics,
            hybrid_metrics=hybrid_metrics,
            resources=resources,
            query_results=query_results,
        )

    def evaluate_external_bcp(
        self,
        bcp_csv_path: str,
        dataset: EvaluationDataset,
        dataset_name: str = "Test Set",
        method_name: str = "BCP Lexical Search (Baseline)",
    ) -> MethodEvaluationReport:
        """
        Evaluate external BCP baseline search results CSV (columns: query_id, response(k=5))
        against the given EvaluationDataset. Uses case and diacritic normalization
        for primary metrics and records strict character-exact metrics in metadata.
        """
        import csv

        bcp_map: Dict[str, List[str]] = {}
        with open(bcp_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["query_id"].strip()
                raw_resp = (row.get("response(k=5)") or "").strip()
                terms = [t.strip() for t in raw_resp.split(";") if t.strip()]
                bcp_map[qid] = terms[: self.k]

        query_results: List[QueryEvaluationResult] = []
        conceptual_results: List[QueryEvaluationResult] = []
        exact_results: List[QueryEvaluationResult] = []
        hybrid_results: List[QueryEvaluationResult] = []

        strict_results: List[QueryEvaluationResult] = []
        strict_conceptual: List[QueryEvaluationResult] = []
        strict_exact: List[QueryEvaluationResult] = []
        strict_hybrid: List[QueryEvaluationResult] = []
        normalization_diffs: List[Dict[str, str]] = []

        for q in dataset.queries:
            raw_terms = bcp_map.get(q.query_id, [])
            gt = q.ground_truth_labels
            norm_gt_lookup = {normalize_text(term): term for term in gt}

            mapped_terms: List[str] = []
            for t in raw_terms:
                if t in gt:
                    mapped_terms.append(t)
                else:
                    norm_t = normalize_text(t)
                    if norm_t in norm_gt_lookup:
                        canonical = norm_gt_lookup[norm_t]
                        mapped_terms.append(canonical)
                        normalization_diffs.append({
                            "query_id": q.query_id,
                            "bcp_term": t,
                            "ground_truth_term": canonical,
                        })
                    else:
                        mapped_terms.append(t)

            res = compute_query_metrics(
                query_id=q.query_id,
                query_type=q.query_type,
                query_text=q.query_text,
                retrieved_terms=mapped_terms,
                ground_truth=gt,
                latency_ms=0.0,
                k=self.k,
            )
            # Preserve raw BCP output strings in retrieved_terms (matching exp_1)
            res.retrieved_terms = list(raw_terms[: self.k])
            res.latency_ms = None  # type: ignore[assignment]

            s_res = compute_query_metrics(
                query_id=q.query_id,
                query_type=q.query_type,
                query_text=q.query_text,
                retrieved_terms=raw_terms,
                ground_truth=gt,
                latency_ms=0.0,
                k=self.k,
            )

            query_results.append(res)
            strict_results.append(s_res)
            qt_low = q.query_type.lower()
            if qt_low == "conceptual":
                conceptual_results.append(res)
                strict_conceptual.append(s_res)
            elif qt_low == "hybrid":
                hybrid_results.append(res)
                strict_hybrid.append(s_res)
            else:
                exact_results.append(res)
                strict_exact.append(s_res)

        strict_all = aggregate_metrics(strict_results)
        strict_conc = aggregate_metrics(strict_conceptual)
        strict_ex = aggregate_metrics(strict_exact)
        strict_hyb = aggregate_metrics(strict_hybrid)

        return MethodEvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            total_queries=len(dataset),
            k=self.k,
            alpha=None,
            overall_metrics=aggregate_metrics(query_results),
            conceptual_metrics=aggregate_metrics(conceptual_results),
            exact_metrics=aggregate_metrics(exact_results),
            hybrid_metrics=aggregate_metrics(hybrid_results),
            resources=None,
            query_results=query_results,
            metadata={
                "normalization_diffs": normalization_diffs,
                "strict_overall": {
                    "ndcg@5": strict_all.ndcg_at_k,
                    "mrr": strict_all.mrr,
                    "precision@5": strict_all.precision_at_k,
                    "recall@5": strict_all.recall_at_k,
                },
                "strict_conceptual": {
                    "ndcg@5": strict_conc.ndcg_at_k,
                    "mrr": strict_conc.mrr,
                },
                "strict_exact": {
                    "ndcg@5": strict_ex.ndcg_at_k,
                    "mrr": strict_ex.mrr,
                },
                "strict_hybrid": {
                    "ndcg@5": strict_hyb.ndcg_at_k,
                    "mrr": strict_hyb.mrr,
                },
            },
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
