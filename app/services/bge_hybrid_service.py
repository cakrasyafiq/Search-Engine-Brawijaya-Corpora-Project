import logging
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from app.db.models import Lexicon
from app.search.bm25_index import DocumentRecord
from app.schemas.lexicon import BGEHybridLexiconCandidate
from app.services.bge_sparse_service import bge_sparse_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import min_max_normalize

logger = logging.getLogger(__name__)


class BGEHybridService:
    """
    Full BGE-M3 Hybrid Search Service combining BAAI/bge-m3's native learned sparse
    representation and 1024-d dense representation using a single shared Transformer
    forward pass, Min-Max Normalization, and Weighted Score Fusion.
    """

    @staticmethod
    def _extract_candidate_data(
        doc_id: int,
        sparse_docs: Dict[int, DocumentRecord],
        dense_models: Dict[int, Lexicon],
        db: Optional[Session] = None,
    ) -> Optional[Any]:
        if doc_id in sparse_docs:
            return sparse_docs[doc_id]
        if doc_id in dense_models:
            return dense_models[doc_id]
        if db is not None and doc_id in bge_sparse_service.index.documents:
            return bge_sparse_service.index.documents[doc_id]
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
    ) -> Tuple[List[BGEHybridLexiconCandidate], float, float]:
        """
        Execute Full BGE-M3 Hybrid Search by jointly encoding the query into dense and
        sparse representations, querying both BGE-M3 Sparse and BGE-M3 Dense indexes,
        applying min-max normalization, and fusing via weighted score fusion.
        """
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

        if not bge_sparse_service.is_ready and db is not None:
            bge_sparse_service.initialize(db)

        candidate_pool = max(top_k * 3, 50)

        # Single-pass joint encoding of query into dense vector and sparse weights
        query_vector, query_sparse_weights = semantic_service.encode_query_both(query)

        # 1. Native BGE-M3 Sparse Search
        sparse_results = bge_sparse_service.search_by_weights(
            query_weights=query_sparse_weights,
            top_k=candidate_pool,
            min_score=0.0,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )
        sparse_scores: Dict[int, float] = {r.document.id: float(r.score) for r in sparse_results}
        sparse_matched: Dict[int, List[str]] = {
            r.document.id: r.matched_terms for r in sparse_results
        }
        sparse_docs: Dict[int, DocumentRecord] = {
            r.document.id: r.document for r in sparse_results
        }

        # 2. Dense BGE-M3 Vector Search
        dense_scores: Dict[int, float] = {}
        dense_models: Dict[int, Lexicon] = {}
        try:
            semantic_results = semantic_service.search_by_vector(
                db=db,
                query_vector=query_vector,
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
            logger.warning(f"Dense search encountered an error during BGE hybrid search: {e}")

        # 3. Min-Max Normalization on both sets of scores
        norm_sparse = min_max_normalize(sparse_scores)
        norm_dense = min_max_normalize(dense_scores)

        # 4. Score Fusion across candidate union
        all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        if not all_ids:
            return [], w_dense, w_sparse

        fused_items: List[Tuple[int, float]] = []
        for doc_id in all_ids:
            s_sparse_norm = norm_sparse.get(doc_id, 0.0)
            s_dense_norm = norm_dense.get(doc_id, 0.0)
            hybrid_score = round((w_sparse * s_sparse_norm) + (w_dense * s_dense_norm), 6)

            if hybrid_score >= min_score:
                fused_items.append((doc_id, hybrid_score))

        fused_items.sort(key=lambda x: x[1], reverse=True)
        top_items = fused_items[:top_k]

        # 5. Build candidate response objects
        candidates: List[BGEHybridLexiconCandidate] = []
        for doc_id, h_score in top_items:
            doc_obj = self._extract_candidate_data(doc_id, sparse_docs, dense_models, db)
            if not doc_obj:
                continue

            candidate = BGEHybridLexiconCandidate(
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
                sparse_score=round(sparse_scores.get(doc_id, 0.0), 4),
                normalized_sparse_score=round(norm_sparse.get(doc_id, 0.0), 4),
                similarity_score=round(dense_scores.get(doc_id, 0.0), 4),
                normalized_similarity_score=round(norm_dense.get(doc_id, 0.0), 4),
                matched_tokens=sparse_matched.get(doc_id, []),
            )
            candidates.append(candidate)

        return candidates, w_dense, w_sparse


bge_hybrid_service = BGEHybridService()

