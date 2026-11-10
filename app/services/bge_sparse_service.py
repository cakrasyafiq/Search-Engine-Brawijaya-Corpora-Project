import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Lexicon
from app.search.bm25_index import DocumentRecord, SearchResult
from app.services.search_service import model_to_document_record
from app.services.semantic_service import semantic_service, prepare_lexicon_text

logger = logging.getLogger(__name__)


class BGESparseInvertedIndex:
    """
    In-memory Inverted Index for BAAI/bge-m3's native learned sparse lexical weights
    (sparse_linear.pt + ReLU activation over XLM-RoBERTa token hidden states).
    """

    def __init__(self):
        self.documents: Dict[int, DocumentRecord] = {}
        # Inverted index: subword_token -> {doc_id: token_weight}
        self.postings: Dict[str, Dict[int, float]] = {}
        self.total_docs: int = 0

    def build_index(
        self,
        records: List[DocumentRecord],
        sparse_weights_list: List[Dict[str, float]],
    ) -> None:
        """Populate inverted index from DocumentRecords and their BGE-M3 sparse lexical weights."""
        self.documents.clear()
        self.postings.clear()
        self.total_docs = len(records)

        for rec, weights in zip(records, sparse_weights_list):
            self.documents[rec.id] = rec
            for token, weight in weights.items():
                if weight <= 0.0:
                    continue
                if token not in self.postings:
                    self.postings[token] = {}
                self.postings[token][rec.id] = float(weight)

    def search_by_weights(
        self,
        query_weights: Dict[str, float],
        top_k: int = 10,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Compute inner-product lexical matching score:
        S_sparse(q, d) = sum_{t in q ∩ d} w_q(t) * w_d(t)
        """
        if not query_weights or self.total_docs == 0:
            return []

        scores: Dict[int, float] = {}
        matched_tokens: Dict[int, Set[str]] = {}

        for token, q_weight in query_weights.items():
            if q_weight <= 0.0 or token not in self.postings:
                continue

            token_postings = self.postings[token]
            clean_token = token.lstrip("▁").strip() or token
            for doc_id, d_weight in token_postings.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + (q_weight * d_weight)
                if doc_id not in matched_tokens:
                    matched_tokens[doc_id] = set()
                matched_tokens[doc_id].add(clean_token)

        results: List[SearchResult] = []
        culture_filter = culture.lower().strip() if culture else None
        domain_filter = semantic_domain.lower().strip() if semantic_domain else None
        class_filter = word_class.lower().strip() if word_class else None

        for doc_id, raw_score in scores.items():
            if raw_score < min_score:
                continue
            doc = self.documents.get(doc_id)
            if not doc:
                continue

            if culture_filter and (not doc.culture or culture_filter not in doc.culture.lower()):
                continue
            if domain_filter and (
                not doc.semantic_domain or domain_filter not in doc.semantic_domain.lower()
            ):
                continue
            if class_filter and (not doc.word_class or class_filter not in doc.word_class.lower()):
                continue

            results.append(
                SearchResult(
                    document=doc,
                    score=round(raw_score, 4),
                    matched_terms=sorted(list(matched_tokens.get(doc_id, set()))),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def save_to_file(self, file_path: str) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "postings": self.postings,
                    "total_docs": self.total_docs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load_from_file(cls, file_path: str) -> "BGESparseInvertedIndex":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.documents = data["documents"]
        idx.postings = data["postings"]
        idx.total_docs = data["total_docs"]
        return idx


class BGESparseService:
    """
    Lifecycle manager and search service for BAAI/bge-m3's native sparse representation.
    """

    def __init__(self):
        self.index = BGESparseInvertedIndex()
        self.is_ready = False

    def initialize(self, db: Session, force_rebuild: bool = False) -> None:
        """
        Load BGE-M3 sparse index from disk cache if present and matching DB count;
        otherwise rebuild from PostgreSQL using BAAI/bge-m3 sparse_linear head.
        """
        cache_path = Path(settings.BGE_SPARSE_INDEX_CACHE_PATH)
        db_count = db.query(Lexicon).count()

        if not force_rebuild and cache_path.exists():
            try:
                loaded_index = BGESparseInvertedIndex.load_from_file(str(cache_path))
                if loaded_index.total_docs == db_count:
                    self.index = loaded_index
                    self.is_ready = True
                    logger.info(
                        f"BGE-M3 Sparse Inverted Index loaded from cache with {self.index.total_docs} documents."
                    )
                    return
                else:
                    logger.info(
                        f"Cached BGE-M3 sparse index count ({loaded_index.total_docs}) != DB count ({db_count}). Rebuilding..."
                    )
            except Exception as e:
                logger.warning(f"Could not load BGE-M3 sparse index cache: {e}. Rebuilding...")

        self.reindex(db)

    def reindex(self, db: Session) -> Tuple[int, float]:
        """
        Encode all Lexicon records using BGE-M3's native sparse representation,
        build the inverted index, and save to disk.
        """
        start_time = time.perf_counter()
        records = db.query(Lexicon).all()
        doc_records = [model_to_document_record(rec) for rec in records]
        texts = [prepare_lexicon_text(rec) for rec in records]

        sparse_weights_list = semantic_service.encode_sparse_batch(
            texts, batch_size=settings.EMBEDDING_BATCH_SIZE
        )

        new_index = BGESparseInvertedIndex()
        new_index.build_index(doc_records, sparse_weights_list)

        self.index = new_index
        self.is_ready = True

        try:
            self.index.save_to_file(settings.BGE_SPARSE_INDEX_CACHE_PATH)
        except Exception as e:
            logger.warning(f"Failed to persist BGE-M3 sparse index to cache: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Successfully built BGE-M3 Sparse index for {len(doc_records)} documents in {elapsed_ms:.2f}ms"
        )
        return len(doc_records), elapsed_ms

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> List[SearchResult]:
        """Encode user query with BGE-M3 sparse head and search the sparse inverted index."""
        if not query or not query.strip():
            return []
        if not self.is_ready and db is not None:
            self.initialize(db)
        query_weights = semantic_service.encode_sparse_query(query)
        return self.index.search_by_weights(
            query_weights=query_weights,
            top_k=top_k,
            min_score=min_score,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )

    def search_by_weights(
        self,
        query_weights: Dict[str, float],
        top_k: int = 10,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using pre-computed BGE-M3 sparse query weights."""
        return self.index.search_by_weights(
            query_weights=query_weights,
            top_k=top_k,
            min_score=min_score,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )


bge_sparse_service = BGESparseService()

