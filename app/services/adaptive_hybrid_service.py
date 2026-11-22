import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Lexicon
from app.search.bm25_index import DocumentRecord
from app.schemas.lexicon import HybridLexiconCandidate
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import min_max_normalize

logger = logging.getLogger(__name__)


def compute_sigmoid_confidence(max_bm25: float, tau: float, scale: float) -> float:
    """
    Map raw top-1 BM25 score to bounded lexical confidence c(q) in [0.0, 1.0].
    Returns 0.0 directly when BM25 has zero matches (max_bm25 <= 0.0).
    """
    if scale <= 0.0:
        raise ValueError("Sigmoid scale parameter must be strictly positive.")
    if max_bm25 <= 0.0:
        return 0.0

    z = max(-60.0, min(60.0, (float(max_bm25) - float(tau)) / float(scale)))
    return 1.0 / (1.0 + math.exp(-z))


def compute_adaptive_alpha(
    max_bm25: float,
    tau: float,
    scale: float,
    alpha_min: float,
    alpha_max: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute query-adaptive semantic weight alpha(q) in [alpha_min, alpha_max]
    and corresponding lexical confidence c(q).

    When BM25 confidence is low (c(q) -> 0), alpha(q) -> alpha_max (1.0 semantic).
    When BM25 confidence is high (c(q) -> 1), alpha(q) -> alpha_min.
    """
    low = max(0.0, min(1.0, float(alpha_min)))
    high = max(low, min(1.0, float(alpha_max)))
    confidence = compute_sigmoid_confidence(max_bm25=max_bm25, tau=tau, scale=scale)
    alpha_q = high - (high - low) * confidence
    alpha_q = max(0.0, min(1.0, alpha_q))
    return alpha_q, confidence


class AdaptiveHybridService:
    """
    Adaptive Hybrid Search Service combining Okapi BM25 and BGE-M3 using
    a Smooth Sigmoid Lexical Confidence Gate over raw BM25 max scores,
    followed by Min-Max Normalization and Weighted Score Fusion.
    """

    @staticmethod
    def _extract_candidate_data(
        doc_id: int,
        bm25_docs: Dict[int, DocumentRecord],
        dense_models: Dict[int, Lexicon],
        db: Optional[Session] = None,
    ) -> Optional[Any]:
        """Retrieve document metadata from BM25 doc records, dense models, or database."""
        if doc_id in bm25_docs:
            return bm25_docs[doc_id]
        if doc_id in dense_models:
            return dense_models[doc_id]
        if db is not None:
            return db.query(Lexicon).filter(Lexicon.id == doc_id).first()
        return None

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        tau: Optional[float] = None,
        scale: Optional[float] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> Tuple[List[HybridLexiconCandidate], float, float, float, float]:
        """
        Execute adaptive hybrid search with Smooth Sigmoid Lexical Confidence Gating.

        Returns:
            (candidates, dense_weight, sparse_weight, max_bm25_score, lexical_confidence)
        """
        eff_tau = settings.ADAPTIVE_GATE_TAU if tau is None else float(tau)
        eff_scale = settings.ADAPTIVE_GATE_SCALE if scale is None else float(scale)
        eff_alpha_min = (
            settings.ADAPTIVE_GATE_ALPHA_MIN if alpha_min is None else float(alpha_min)
        )
        eff_alpha_max = (
            settings.ADAPTIVE_GATE_ALPHA_MAX if alpha_max is None else float(alpha_max)
        )

        if not search_service.is_ready and db is not None:
            search_service.initialize(db)

        candidate_pool = max(top_k * 3, 50)

        # 1. Sparse BM25 Search
        bm25_results = search_service.search(
            query=query,
            top_k=candidate_pool,
            min_score=0.0,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )
        bm25_scores: Dict[int, float] = {
            r.document.id: float(r.score) for r in bm25_results
        }
        bm25_matched: Dict[int, List[str]] = {
            r.document.id: r.matched_terms for r in bm25_results
        }
        bm25_docs: Dict[int, DocumentRecord] = {
            r.document.id: r.document for r in bm25_results
        }

        # 2. Compute query-adaptive alpha(q) from raw BM25 top-1 score
        max_bm25_score = max(bm25_scores.values(), default=0.0)
        alpha_q, lexical_confidence = compute_adaptive_alpha(
            max_bm25=max_bm25_score,
            tau=eff_tau,
            scale=eff_scale,
            alpha_min=eff_alpha_min,
            alpha_max=eff_alpha_max,
        )
        w_dense = round(alpha_q, 6)
        w_sparse = round(1.0 - w_dense, 6)

        # 3. Dense BGE-M3 Vector Search
        dense_scores: Dict[int, float] = {}
        dense_models: Dict[int, Lexicon] = {}
        try:
            semantic_results = semantic_service.search(
                db=db,
                query=query,
                top_k=candidate_pool,
                min_similarity=0.0,
                culture=culture,
                semantic_domain=semantic_domain,
                word_class=word_class,
            )
            for lex, sim_score in semantic_results:
                dense_scores[lex.id] = float(sim_score)
                dense_models[lex.id] = lex
        except Exception as e:
            logger.warning(
                f"Dense search encountered an error during adaptive hybrid search: {e}"
            )

        # 4. Min-Max Normalization on both sets of scores
        norm_bm25 = min_max_normalize(bm25_scores)
        norm_dense = min_max_normalize(dense_scores)

        # 5. Adaptive Weighted Score Fusion across candidate union
        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        if not all_ids:
            return (
                [],
                w_dense,
                w_sparse,
                round(max_bm25_score, 4),
                round(lexical_confidence, 6),
            )

        fused_items: List[Tuple[int, float]] = []
        for doc_id in all_ids:
            s_bm25_norm = norm_bm25.get(doc_id, 0.0)
            s_dense_norm = norm_dense.get(doc_id, 0.0)
            hybrid_score = round((w_sparse * s_bm25_norm) + (w_dense * s_dense_norm), 6)

            if hybrid_score >= min_score:
                fused_items.append((doc_id, hybrid_score))

        fused_items.sort(key=lambda x: x[1], reverse=True)
        top_items = fused_items[:top_k]

        # 6. Build candidate response objects
        candidates: List[HybridLexiconCandidate] = []
        for doc_id, h_score in top_items:
            doc_obj = self._extract_candidate_data(doc_id, bm25_docs, dense_models, db)
            if not doc_obj:
                continue

            candidate = HybridLexiconCandidate(
                id=doc_id,
                term=getattr(doc_obj, "term", ""),
                code=getattr(doc_obj, "code", None),
                culture=getattr(doc_obj, "culture", None),
                sub_culture=getattr(doc_obj, "sub_culture", None),
                word_class=getattr(doc_obj, "word_class", None),
                language=getattr(doc_obj, "language", None),
                dialect=getattr(doc_obj, "dialect", None),
                semantic_domain=getattr(doc_obj, "semantic_domain", None),
                usage_context=getattr(doc_obj, "usage_context", None),
                register=getattr(doc_obj, "register", None),
                phonetic=getattr(doc_obj, "phonetic", None),
                definition=getattr(doc_obj, "definition", None),
                variants=getattr(doc_obj, "variants", None),
                examples=getattr(doc_obj, "examples", None),
                categories=getattr(doc_obj, "categories", None),
                hybrid_score=round(h_score, 4),
                bm25_score=round(bm25_scores.get(doc_id, 0.0), 4),
                normalized_bm25_score=round(norm_bm25.get(doc_id, 0.0), 4),
                similarity_score=round(dense_scores.get(doc_id, 0.0), 4),
                normalized_similarity_score=round(norm_dense.get(doc_id, 0.0), 4),
                matched_terms=bm25_matched.get(doc_id, []),
            )
            candidates.append(candidate)

        return (
            candidates,
            w_dense,
            w_sparse,
            round(max_bm25_score, 4),
            round(lexical_confidence, 6),
        )


adaptive_hybrid_service = AdaptiveHybridService()
