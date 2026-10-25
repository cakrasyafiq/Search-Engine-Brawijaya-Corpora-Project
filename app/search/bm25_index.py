import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.search.tokenizer import tokenize, normalize_text


@dataclass
class DocumentRecord:
    id: int
    term: str
    code: Optional[str] = None
    culture: Optional[str] = None
    sub_culture: Optional[str] = None
    word_class: Optional[str] = None
    language: Optional[str] = None
    dialect: Optional[str] = None
    semantic_domain: Optional[str] = None
    usage_context: Optional[str] = None
    register: Optional[str] = None
    phonetic: Optional[str] = None
    definition: Optional[str] = None
    etymology: Optional[str] = None
    variants: Optional[str] = None
    examples: Optional[str] = None
    reference: Optional[str] = None
    reference_url: Optional[str] = None
    categories: Optional[str] = None
    contributors: Optional[str] = None
    images_count: int = 0
    audio_count: int = 0
    video_count: int = 0
    three_d_models_count: int = 0
    status: Optional[str] = None
    verification: Optional[str] = None


@dataclass
class SearchResult:
    document: DocumentRecord
    score: float
    matched_terms: List[str]


class BM25InvertedIndex:
    """
    In-memory BM25 Inverted Index with weighted multi-field representation,
    fast candidate lookups, and disk persistence.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[int, DocumentRecord] = {}
        # Inverted index: token -> {doc_id: weighted_tf}
        self.postings: Dict[str, Dict[int, float]] = {}
        # Document length: doc_id -> effective length
        self.doc_lengths: Dict[int, float] = {}
        # IDF values: token -> float
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.total_docs: int = 0

    # Note: Etymology is excluded to avoid cross-language dictionary noise (e.g. Dutch cross-references)
    FIELD_WEIGHTS = {
        "term": 3.0,
        "variants": 2.2,
        "semantic_domain": 1.5,
        "culture": 1.3,
        "usage_context": 1.3,
        "definition": 1.0,
        "examples": 1.0,
        "categories": 1.2,
    }

    def build_index(self, records: List[DocumentRecord]) -> None:
        """Construct the inverted index and compute IDF statistics for all documents."""
        self.documents.clear()
        self.postings.clear()
        self.doc_lengths.clear()
        self.idf.clear()

        self.total_docs = len(records)
        if self.total_docs == 0:
            self.avgdl = 0.0
            return

        total_length = 0.0

        for rec in records:
            self.documents[rec.id] = rec
            doc_tf: Dict[str, float] = {}
            doc_len = 0.0

            # Process fields with weights (etymology intentionally excluded to reduce noise)
            field_values = {
                "term": rec.term or "",
                "variants": rec.variants or "",
                "semantic_domain": rec.semantic_domain or "",
                "culture": rec.culture or "",
                "usage_context": rec.usage_context or "",
                "definition": rec.definition or "",
                "examples": rec.examples or "",
                "categories": rec.categories or "",
            }

            for field_name, weight in self.FIELD_WEIGHTS.items():
                content = field_values.get(field_name, "")
                if not content:
                    continue
                tokens = tokenize(content)
                for t in tokens:
                    doc_tf[t] = doc_tf.get(t, 0.0) + weight
                    doc_len += weight

            self.doc_lengths[rec.id] = max(doc_len, 1.0)
            total_length += self.doc_lengths[rec.id]

            # Populate inverted index postings
            for token, tf in doc_tf.items():
                if token not in self.postings:
                    self.postings[token] = {}
                self.postings[token][rec.id] = tf

        self.avgdl = total_length / self.total_docs if self.total_docs > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF with Okapi smoothing
        # IDF(q) = ln(1 + (N - df + 0.5) / (df + 0.5))
        for token, post in self.postings.items():
            df = len(post)
            self.idf[token] = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute BM25 search over the indexed corpus.
        Returns candidate SearchResult objects ranked by BM25 score descending.
        """
        if not query or not query.strip() or self.total_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = {}
        matched_tokens: Dict[int, Set[str]] = {}
        unique_tokens = list(set(query_tokens))
        normalized_q = normalize_text(query)

        # Inverted index lookup: calculate BM25 score for candidate documents
        for token in unique_tokens:
            if token not in self.postings:
                continue

            token_idf = self.idf.get(token, 0.0)
            token_postings = self.postings[token]

            for doc_id, tf in token_postings.items():
                dl = self.doc_lengths[doc_id]
                # BM25 formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                term_score = token_idf * (numerator / denominator)

                scores[doc_id] = scores.get(doc_id, 0.0) + term_score
                if doc_id not in matched_tokens:
                    matched_tokens[doc_id] = set()
                matched_tokens[doc_id].add(token)

        # Apply exact match bonuses and candidate filtering
        results: List[SearchResult] = []
        culture_filter = culture.lower().strip() if culture else None
        domain_filter = semantic_domain.lower().strip() if semantic_domain else None
        class_filter = word_class.lower().strip() if word_class else None

        for doc_id, base_score in scores.items():
            doc = self.documents[doc_id]

            # Filters
            if culture_filter and (not doc.culture or culture_filter not in doc.culture.lower()):
                continue
            if domain_filter and (not doc.semantic_domain or domain_filter not in doc.semantic_domain.lower()):
                continue
            if class_filter and (not doc.word_class or class_filter not in doc.word_class.lower()):
                continue

            # Exact match boost
            final_score = base_score
            doc_term_norm = normalize_text(doc.term)
            if doc_term_norm == normalized_q:
                final_score *= 1.4  # Exact term boost
            elif normalized_q in doc_term_norm:
                final_score *= 1.15

            if final_score < min_score:
                continue

            results.append(
                SearchResult(
                    document=doc,
                    score=round(final_score, 4),
                    matched_terms=sorted(list(matched_tokens.get(doc_id, set()))),
                )
            )

        # Sort descending by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def save_to_file(self, file_path: str) -> None:
        """Persist index state to disk."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "documents": self.documents,
                    "postings": self.postings,
                    "doc_lengths": self.doc_lengths,
                    "idf": self.idf,
                    "avgdl": self.avgdl,
                    "total_docs": self.total_docs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load_from_file(cls, file_path: str) -> "BM25InvertedIndex":
        """Load index state from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.documents = data["documents"]
        idx.postings = data["postings"]
        idx.doc_lengths = data["doc_lengths"]
        idx.idf = data["idf"]
        idx.avgdl = data["avgdl"]
        idx.total_docs = data["total_docs"]
        return idx
