import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import min_max_normalize
from evaluation.dataset import EvaluationDataset, EvaluationQuery
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    QueryEvaluationResult,
    AggregatedMetrics,
    aggregate_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class AlphaMetricRecord:
    alpha: float
    dense_weight: float
    sparse_weight: float
    # Overall metrics
    overall_ndcg: float
    overall_mrr: float
    overall_precision: float
    overall_recall: float
    # Conceptual queries metrics
    conceptual_ndcg: float
    conceptual_mrr: float
    conceptual_precision: float
    conceptual_recall: float
    # Exact term queries metrics
    exact_ndcg: float
    exact_mrr: float
    exact_precision: float
    exact_recall: float
    k: int = 5


class AlphaOptimizer:
    """
    Finds the optimal alpha weight for weighted score fusion in Hybrid Search
    by performing a grid search across alpha in [0.0, 1.0] on the validation dataset.
    """

    def __init__(
        self,
        db: Session,
        val_dataset: EvaluationDataset,
        k: int = 5,
        alpha_step: float = 0.05,
    ):
        self.db = db
        self.val_dataset = val_dataset
        self.k = k
        self.alpha_step = alpha_step

        # Candidate pool for fusion
        self.candidate_pool = max(k * 3, 50)

    def optimize(
        self,
        primary_metric: str = "ndcg",
        output_dir: Optional[str | Path] = None,
    ) -> Tuple[float, List[AlphaMetricRecord]]:
        """
        Run grid search for alpha from 0.0 to 1.0 (step alpha_step).
        Returns (optimal_alpha, records).
        """
        logger.info(
            f"Starting Alpha Optimization on {len(self.val_dataset)} validation queries..."
        )

        # 1. Pre-retrieve sparse and dense candidates for each validation query
        query_candidates: Dict[str, Dict[str, Any]] = {}

        # Warm up semantic service
        _ = semantic_service.get_model()

        for q in self.val_dataset.queries:
            # BM25 candidates
            bm25_res = search_service.search(
                query=q.query_text,
                top_k=self.candidate_pool,
                min_score=0.0,
            )
            bm25_scores = {r.document.id: float(r.score) for r in bm25_res}
            id_to_term = {r.document.id: r.document.term for r in bm25_res}

            # BGE-M3 candidates
            dense_scores: Dict[int, float] = {}
            try:
                sem_res = semantic_service.search(
                    db=self.db,
                    query=q.query_text,
                    top_k=self.candidate_pool,
                    min_similarity=0.0,
                )
                for lex, sim in sem_res:
                    dense_scores[lex.id] = float(sim)
                    id_to_term[lex.id] = lex.term
            except Exception as e:
                logger.warning(f"Semantic search failed for query '{q.query_text}': {e}")

            # Min-Max Normalized scores
            norm_bm25 = min_max_normalize(bm25_scores)
            norm_dense = min_max_normalize(dense_scores)

            all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

            query_candidates[q.query_id] = {
                "query": q,
                "all_ids": all_ids,
                "norm_bm25": norm_bm25,
                "norm_dense": norm_dense,
                "id_to_term": id_to_term,
            }

        # 2. Sweep alpha values from 0.0 to 1.0
        alphas = []
        curr = 0.0
        while curr <= 1.000001:
            alphas.append(round(curr, 4))
            curr += self.alpha_step

        records: List[AlphaMetricRecord] = []

        for alpha in alphas:
            w_dense = alpha
            w_sparse = round(1.0 - alpha, 4)

            overall_results: List[QueryEvaluationResult] = []
            conceptual_results: List[QueryEvaluationResult] = []
            exact_results: List[QueryEvaluationResult] = []

            for qid, data in query_candidates.items():
                q: EvaluationQuery = data["query"]
                all_ids = data["all_ids"]
                norm_bm25 = data["norm_bm25"]
                norm_dense = data["norm_dense"]
                id_to_term = data["id_to_term"]

                # Weighted score fusion
                fused: List[Tuple[str, float]] = []
                for doc_id in all_ids:
                    s_bm25 = norm_bm25.get(doc_id, 0.0)
                    s_dense = norm_dense.get(doc_id, 0.0)
                    score = (w_sparse * s_bm25) + (w_dense * s_dense)
                    term = id_to_term.get(doc_id, "")
                    if term:
                        fused.append((term, score))

                # Rank descending by fused score
                fused.sort(key=lambda x: x[1], reverse=True)
                retrieved_terms = [item[0] for item in fused[: self.k]]

                # Calculate metrics for query
                gt = q.ground_truth_labels
                p = precision_at_k(retrieved_terms, gt, k=self.k)
                r = recall_at_k(retrieved_terms, gt, k=self.k)
                rr = reciprocal_rank(retrieved_terms, gt, k=self.k)
                ndcg = ndcg_at_k(retrieved_terms, gt, k=self.k)

                q_res = QueryEvaluationResult(
                    query_id=q.query_id,
                    query_type=q.query_type,
                    query_text=q.query_text,
                    retrieved_terms=retrieved_terms,
                    precision_at_k=round(p, 4),
                    recall_at_k=round(r, 4),
                    reciprocal_rank=round(rr, 4),
                    ndcg_at_k=round(ndcg, 4),
                    latency_ms=0.0,
                    k=self.k,
                )

                overall_results.append(q_res)
                if q.query_type.lower() == "conceptual":
                    conceptual_results.append(q_res)
                else:
                    exact_results.append(q_res)

            # Macro aggregates
            agg_all = aggregate_metrics(overall_results)
            agg_conc = aggregate_metrics(conceptual_results)
            agg_exact = aggregate_metrics(exact_results)

            rec = AlphaMetricRecord(
                alpha=alpha,
                dense_weight=w_dense,
                sparse_weight=w_sparse,
                overall_ndcg=agg_all.ndcg_at_k,
                overall_mrr=agg_all.mrr,
                overall_precision=agg_all.precision_at_k,
                overall_recall=agg_all.recall_at_k,
                conceptual_ndcg=agg_conc.ndcg_at_k,
                conceptual_mrr=agg_conc.mrr,
                conceptual_precision=agg_conc.precision_at_k,
                conceptual_recall=agg_conc.recall_at_k,
                exact_ndcg=agg_exact.ndcg_at_k,
                exact_mrr=agg_exact.mrr,
                exact_precision=agg_exact.precision_at_k,
                exact_recall=agg_exact.recall_at_k,
                k=self.k,
            )
            records.append(rec)

        # 3. Select optimal alpha
        # Prioritize primary metric (NDCG@5), tiebreak with MRR
        best_rec = max(
            records,
            key=lambda r: (r.overall_ndcg, r.overall_mrr, r.overall_recall),
        )
        optimal_alpha = best_rec.alpha

        logger.info(
            f"Alpha tuning complete! Optimal alpha = {optimal_alpha} "
            f"(NDCG@{self.k}: {best_rec.overall_ndcg:.4f}, MRR: {best_rec.overall_mrr:.4f}, "
            f"P@{self.k}: {best_rec.overall_precision:.4f}, R@{self.k}: {best_rec.overall_recall:.4f})"
        )

        # 4. Save results if output_dir provided
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            self._save_results(records, optimal_alpha, out_p)

        return optimal_alpha, records

    @staticmethod
    def _save_results(
        records: List[AlphaMetricRecord], optimal_alpha: float, out_dir: Path
    ) -> None:
        csv_path = out_dir / "alpha_tuning_results.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

        json_path = out_dir / "alpha_tuning_results.json"
        data = {
            "optimal_alpha": optimal_alpha,
            "grid_results": [asdict(r) for r in records],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
