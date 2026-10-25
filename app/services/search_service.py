import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Lexicon
from app.search.bm25_index import BM25InvertedIndex, DocumentRecord, SearchResult

logger = logging.getLogger(__name__)


def model_to_document_record(model: Lexicon) -> DocumentRecord:
    """Convert an SQLAlchemy Lexicon model instance into a DocumentRecord for indexing."""
    return DocumentRecord(
        id=model.id,
        term=model.term,
        code=model.code,
        culture=model.culture,
        sub_culture=model.sub_culture,
        word_class=model.word_class,
        language=model.language,
        dialect=model.dialect,
        semantic_domain=model.semantic_domain,
        usage_context=model.usage_context,
        register=model.register,
        phonetic=model.phonetic,
        definition=model.definition,
        etymology=model.etymology,
        variants=model.variants,
        examples=model.examples,
        reference=model.reference,
        reference_url=model.reference_url,
        categories=model.categories,
        contributors=model.contributors,
        images_count=model.images_count or 0,
        audio_count=model.audio_count or 0,
        video_count=model.video_count or 0,
        three_d_models_count=model.three_d_models_count or 0,
        status=model.status,
        verification=model.verification,
    )


class SearchService:
    def __init__(self):
        self.index = BM25InvertedIndex(k1=settings.BM25_K1, b=settings.BM25_B)
        self.is_ready = False

    def initialize(self, db: Session, force_rebuild: bool = False) -> None:
        """
        Load index from disk cache if present and valid; otherwise,
        query PostgreSQL and build/cache the BM25 inverted index.
        """
        cache_path = Path(settings.INDEX_CACHE_PATH)
        db_count = db.query(Lexicon).count()

        if not force_rebuild and cache_path.exists():
            try:
                loaded_index = BM25InvertedIndex.load_from_file(str(cache_path))
                if loaded_index.total_docs == db_count:
                    self.index = loaded_index
                    self.is_ready = True
                    logger.info(
                        f"BM25 Inverted Index loaded from cache with {self.index.total_docs} documents."
                    )
                    return
                else:
                    logger.info(
                        f"Cached index count ({loaded_index.total_docs}) differs from DB count ({db_count}). Rebuilding..."
                    )
            except Exception as e:
                logger.warning(f"Could not load index cache: {e}. Rebuilding...")

        # Rebuild from database
        self.reindex(db)

    def reindex(self, db: Session) -> Tuple[int, float]:
        """
        Fetch all records from PostgreSQL, build the BM25 index,
        and persist to disk. Returns (indexed_count, execution_time_ms).
        """
        start_time = time.perf_counter()
        records = db.query(Lexicon).all()
        doc_records = [model_to_document_record(rec) for rec in records]

        new_index = BM25InvertedIndex(k1=settings.BM25_K1, b=settings.BM25_B)
        new_index.build_index(doc_records)

        # Atomic switch
        self.index = new_index
        self.is_ready = True

        # Save to cache
        try:
            self.index.save_to_file(settings.INDEX_CACHE_PATH)
        except Exception as e:
            logger.warning(f"Failed to persist index to cache: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Successfully built BM25 index for {len(doc_records)} documents in {elapsed_ms:.2f}ms"
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
    ) -> List[SearchResult]:
        """Query the in-memory BM25 index."""
        return self.index.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )


search_service = SearchService()
