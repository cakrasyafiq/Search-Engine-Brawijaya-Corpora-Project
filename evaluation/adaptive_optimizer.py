import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
from sqlalchemy.orm import Session

from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import min_max_normalize
from app.services.adaptive_hybrid_service import compute_adaptive_alpha
from evaluation.dataset import EvaluationDataset, EvaluationQuery
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    QueryEvaluationResult,
    aggregate_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveGateRecord:
    tau: float
    scale: float
    alpha_min: float
    alpha_max: float
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
    # Hybrid queries metrics
    hybrid_ndcg: float
    hybrid_mrr: float
    hybrid_precision: float
    hybrid_recall: float
    # Mean adaptive alpha per segment
    mean_alpha_conceptual: float
    mean_alpha_exact: float
    mean_alpha_hybrid: float
    k: int = 5


class AdaptiveGateOptimizer:
    """
    Grid search optimizer for Smooth Sigmoid Lexical Confidence Gate parameters
    (tau, scale, alpha_min) on the validation dataset.
    """

    DEFAULT_TAU_GRID = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    DEFAULT_SCALE_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
    DEFAULT_ALPHA_MIN_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    def __init__(
        self,
        db: Session,
        val_dataset: EvaluationDataset,
        k: int = 5,
        tau_grid: Optional[Sequence[float]] = None,
        scale_grid: Optional[Sequence[float]] = None,
        alpha_min_grid: Optional[Sequence[float]] = None,
        alpha_max: float = 1.0,
    ):
        self.db = db
        self.val_dataset = val_dataset
        self.k = k
        self.tau_grid = list(tau_grid) if tau_grid is not None else list(self.DEFAULT_TAU_GRID)
        self.scale_grid = list(scale_grid) if scale_grid is not None else list(self.DEFAULT_SCALE_GRID)
        self.alpha_min_grid = (
            list(alpha_min_grid) if alpha_min_grid is not None else list(self.DEFAULT_ALPHA_MIN_GRID)
        )
        self.alpha_max = float(alpha_max)
        self.candidate_pool = max(k * 3, 50)

    def optimize(
        self,
        output_dir: Optional[str | Path] = None,
        filename_prefix: str = "adaptive_gate_tuning_results",
    ) -> Tuple[Dict[str, float], List[AdaptiveGateRecord]]:
        """
        Pre-retrieve sparse and dense candidates for validation queries and sweep
        (tau, scale, alpha_min) to maximize validation overall NDCG@k.
        """
        logger.info(
            f"Starting Adaptive Gate Optimization on {len(self.val_dataset)} validation queries "
            f"({len(self.tau_grid)}x{len(self.scale_grid)}x{len(self.alpha_min_grid)} = "
            f"{len(self.tau_grid) * len(self.scale_grid) * len(self.alpha_min_grid)} configs)..."
        )

        if not search_service.is_ready:
            search_service.initialize(self.db)
        _ = semantic_service.get_model()

        query_candidates: Dict[str, Dict[str, Any]] = {}
        for q in self.val_dataset.queries:
            sparse_res = search_service.search(
                query=q.query_text,
                top_k=self.candidate_pool,
                min_score=0.0,
            )
            bm25_scores = {r.document.id: float(r.score) for r in sparse_res}
            id_to_term = {r.document.id: r.document.term for r in sparse_res}
            max_bm25 = max(bm25_scores.values(), default=0.0)

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

            norm_bm25 = min_max_normalize(bm25_scores)
            norm_dense = min_max_normalize(dense_scores)
            all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

            query_candidates[q.query_id] = {
                "query": q,
                "max_bm25": max_bm25,
                "all_ids": all_ids,
                "norm_bm25": norm_bm25,
                "norm_dense": norm_dense,
                "id_to_term": id_to_term,
            }

        records: List[AdaptiveGateRecord] = []

        for tau in self.tau_grid:
            for scale in self.scale_grid:
                for alpha_min in self.alpha_min_grid:
                    overall_results: List[QueryEvaluationResult] = []
                    conceptual_results: List[QueryEvaluationResult] = []
                    exact_results: List[QueryEvaluationResult] = []
                    hybrid_results: List[QueryEvaluationResult] = []

                    conc_alphas: List[float] = []
                    exact_alphas: List[float] = []
                    hybrid_alphas: List[float] = []

                    for data in query_candidates.values():
                        q: EvaluationQuery = data["query"]
                        max_bm25: float = data["max_bm25"]
                        all_ids = data["all_ids"]
                        norm_bm25 = data["norm_bm25"]
                        norm_dense = data["norm_dense"]
                        id_to_term = data["id_to_term"]

                        alpha_q, _ = compute_adaptive_alpha(
                            max_bm25=max_bm25,
                            tau=tau,
                            scale=scale,
                            alpha_min=alpha_min,
                            alpha_max=self.alpha_max,
                        )
                        w_dense = round(alpha_q, 6)
                        w_sparse = round(1.0 - w_dense, 6)

                        fused: List[Tuple[str, float]] = []
                        for doc_id in all_ids:
                            s_bm25 = norm_bm25.get(doc_id, 0.0)
                            s_dense = norm_dense.get(doc_id, 0.0)
                            score = (w_sparse * s_bm25) + (w_dense * s_dense)
                            term = id_to_term.get(doc_id, "")
                            if term:
                                fused.append((term, score))

                        fused.sort(key=lambda x: x[1], reverse=True)
                        retrieved_terms = [item[0] for item in fused[: self.k]]

                        gt = q.ground_truth_labels
                        q_res = QueryEvaluationResult(
                            query_id=q.query_id,
                            query_type=q.query_type,
                            query_text=q.query_text,
                            retrieved_terms=retrieved_terms,
                            precision_at_k=round(precision_at_k(retrieved_terms, gt, k=self.k), 4),
                            recall_at_k=round(recall_at_k(retrieved_terms, gt, k=self.k), 4),
                            reciprocal_rank=round(reciprocal_rank(retrieved_terms, gt, k=self.k), 4),
                            ndcg_at_k=round(ndcg_at_k(retrieved_terms, gt, k=self.k), 4),
                            latency_ms=0.0,
                            k=self.k,
                        )

                        overall_results.append(q_res)
                        qt_low = q.query_type.lower()
                        if qt_low == "conceptual":
                            conceptual_results.append(q_res)
                            conc_alphas.append(alpha_q)
                        elif qt_low == "hybrid":
                            hybrid_results.append(q_res)
                            hybrid_alphas.append(alpha_q)
                        else:
                            exact_results.append(q_res)
                            exact_alphas.append(alpha_q)

                    agg_all = aggregate_metrics(overall_results)
                    agg_conc = aggregate_metrics(conceptual_results)
                    agg_exact = aggregate_metrics(exact_results)
                    agg_hybrid = aggregate_metrics(hybrid_results)

                    records.append(
                        AdaptiveGateRecord(
                            tau=float(tau),
                            scale=float(scale),
                            alpha_min=float(alpha_min),
                            alpha_max=float(self.alpha_max),
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
                            hybrid_ndcg=agg_hybrid.ndcg_at_k,
                            hybrid_mrr=agg_hybrid.mrr,
                            hybrid_precision=agg_hybrid.precision_at_k,
                            hybrid_recall=agg_hybrid.recall_at_k,
                            mean_alpha_conceptual=round(sum(conc_alphas) / len(conc_alphas), 4)
                            if conc_alphas
                            else 0.0,
                            mean_alpha_exact=round(sum(exact_alphas) / len(exact_alphas), 4)
                            if exact_alphas
                            else 0.0,
                            mean_alpha_hybrid=round(sum(hybrid_alphas) / len(hybrid_alphas), 4)
                            if hybrid_alphas
                            else 0.0,
                            k=self.k,
                        )
                    )

        best_rec = max(
            records,
            key=lambda r: (
                r.overall_ndcg,
                r.overall_mrr,
                r.overall_recall,
                -abs(r.scale - 1.0),
                -abs(r.tau - 8.0),
            ),
        )
        optimal_params = {
            "tau": best_rec.tau,
            "scale": best_rec.scale,
            "alpha_min": best_rec.alpha_min,
            "alpha_max": best_rec.alpha_max,
        }

        logger.info(
            f"Adaptive Gate tuning complete! Optimal params: tau={best_rec.tau}, "
            f"scale={best_rec.scale}, alpha_min={best_rec.alpha_min}, alpha_max={best_rec.alpha_max} "
            f"(NDCG@{self.k}: {best_rec.overall_ndcg:.4f}, MRR: {best_rec.overall_mrr:.4f}, "
            f"P@{self.k}: {best_rec.overall_precision:.4f}, R@{self.k}: {best_rec.overall_recall:.4f})"
        )

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            self._save_results(records, optimal_params, out_p, filename_prefix)

        return optimal_params, records

    @staticmethod
    def _save_results(
        records: List[AdaptiveGateRecord],
        optimal_params: Dict[str, float],
        out_dir: Path,
        filename_prefix: str = "adaptive_gate_tuning_results",
    ) -> None:
        csv_path = out_dir / f"{filename_prefix}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

        sorted_top = sorted(
            records,
            key=lambda r: (r.overall_ndcg, r.overall_mrr, r.overall_recall),
            reverse=True,
        )[:15]

        json_path = out_dir / f"{filename_prefix}.json"
        data = {
            "optimal_params": optimal_params,
            "top_15_configs": [asdict(r) for r in sorted_top],
            "grid_results": [asdict(r) for r in records],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
