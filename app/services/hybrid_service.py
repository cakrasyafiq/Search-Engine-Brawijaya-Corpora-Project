import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.db.models import Lexicon
from app.search.bm25_index import DocumentRecord
from app.schemas.lexicon import HybridLexiconCandidate
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service

logger = logging.getLogger(__name__)


def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Apply Min-Max normalization to a dictionary of {doc_id: raw_score}.
    Formula: S_norm = (S - min) / (max - min)
    
    Edge-cases:
    - Empty input -> returns empty dict.
    - All scores equal (max == min):
        - If score > 0 -> all normalized to 1.0
        - If score <= 0 -> all normalized to 0.0
    """
    if not scores:
        return {}

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        normalized_val = 1.0 if max_val > 0.0 else 0.0
        return {doc_id: normalized_val for doc_id in scores}

    range_val = max_val - min_val
    return {
        doc_id: round((score - min_val) / range_val, 6)
        for doc_id, score in scores.items()
    }


class HybridService:
    """
    Hybrid Search Service combining Okapi BM25 (sparse representation)
    and BGE-M3 (dense representation) using Min-Max Normalization
    and Weighted Score Fusion.
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
        alpha: Optional[float] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> Tuple[List[HybridLexiconCandidate], float, float]:
        """
        Execute hybrid search by querying both BM25 and BGE-M3,
        normalizing scores with min-max normalization, and fusing
        with weighted score fusion.

        Returns (candidates, effective_dense_weight, effective_sparse_weight).
        """
        # Resolve fusion weights
        if alpha is not None:
            w_dense = max(0.0, min(1.0, float(alpha)))
            w_sparse = round(1.0 - w_dense, 6)
        else:
            total_weight = dense_weight + sparse_weight
            if total_weight > 0:
                w_dense = round(dense_weight / total_weight, 6)
                w_sparse = round(sparse_weight / total_weight, 6)
            else:
                w_dense = 0.5
                w_sparse = 0.5

        # Query an expanded candidate pool from each retriever to maximize fusion recall
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
        bm25_scores: Dict[int, float] = {r.document.id: float(r.score) for r in bm25_results}
        bm25_matched: Dict[int, List[str]] = {r.document.id: r.matched_terms for r in bm25_results}
        bm25_docs: Dict[int, DocumentRecord] = {r.document.id: r.document for r in bm25_results}

        # 2. Dense BGE-M3 Vector Search
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
            logger.warning(f"Dense search encountered an error during hybrid search: {e}")

        # 3. Min-Max Normalization on both sets of scores
        norm_bm25 = min_max_normalize(bm25_scores)
        norm_dense = min_max_normalize(dense_scores)

        # 4. Score Fusion across candidate union
        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        if not all_ids:
            return [], w_dense, w_sparse

        fused_items: List[Tuple[int, float]] = []
        for doc_id in all_ids:
            s_bm25_norm = norm_bm25.get(doc_id, 0.0)
            s_dense_norm = norm_dense.get(doc_id, 0.0)
            hybrid_score = round((w_sparse * s_bm25_norm) + (w_dense * s_dense_norm), 6)

            if hybrid_score >= min_score:
                fused_items.append((doc_id, hybrid_score))

        # Sort descending by hybrid_score
        fused_items.sort(key=lambda x: x[1], reverse=True)
        top_items = fused_items[:top_k]

        # 5. Build candidate response objects
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

        return candidates, w_dense, w_sparse


hybrid_service = HybridService()
