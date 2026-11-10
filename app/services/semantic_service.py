import time
import logging
from typing import List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.db.models import Lexicon

logger = logging.getLogger(__name__)


def prepare_lexicon_text(lex: Lexicon) -> str:
    """Format lexicon fields into a descriptive text representation for BAAI/bge-m3."""
    parts = [
        f"Term: {lex.term}",
        f"Culture: {lex.culture}" if lex.culture else "",
        f"Semantic Domain: {lex.semantic_domain}" if lex.semantic_domain else "",
        f"Word Class: {lex.word_class}" if lex.word_class else "",
        f"Definition: {lex.definition}" if lex.definition else "",
        f"Variants: {lex.variants}" if lex.variants else "",
        f"Examples: {lex.examples}" if lex.examples else "",
    ]
    return "\n".join(p for p in parts if p)


class SemanticService:
    def __init__(self):
        self._model = None
        self._sparse_linear = None
        self._unused_token_ids = None

    def get_model(self):
        """Lazy load SentenceTransformer model on first usage."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device} for {settings.EMBEDDING_MODEL_NAME}")
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    device=device,
                    local_files_only=True,
                )
                logger.info(f"Model {settings.EMBEDDING_MODEL_NAME} loaded successfully.")
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)
                    logger.info(f"Model {settings.EMBEDDING_MODEL_NAME} loaded successfully.")
                except ImportError as e:
                    logger.error(
                        "sentence-transformers is not installed. Please install it via: pip install sentence-transformers"
                    )
                    raise RuntimeError(
                        "sentence-transformers is required for semantic search. Install via: pip install sentence-transformers"
                    ) from e
        return self._model

    def get_sparse_linear(self):
        """Lazy load BAAI/bge-m3 native sparse_linear.pt projection head."""
        if self._sparse_linear is None:
            import torch
            from huggingface_hub import hf_hub_download

            model = self.get_model()
            device = next(model[0].auto_model.parameters()).device

            try:
                sparse_path = hf_hub_download(
                    repo_id=settings.EMBEDDING_MODEL_NAME,
                    filename="sparse_linear.pt",
                    local_files_only=True,
                )
            except Exception:
                sparse_path = hf_hub_download(
                    repo_id=settings.EMBEDDING_MODEL_NAME,
                    filename="sparse_linear.pt",
                )

            linear = torch.nn.Linear(in_features=settings.EMBEDDING_DIMENSION, out_features=1)
            state_dict = torch.load(sparse_path, map_location=device)
            linear.load_state_dict(state_dict)
            linear.to(device)
            linear.eval()
            self._sparse_linear = linear

            tokenizer = model.tokenizer
            self._unused_token_ids = {
                tid
                for tid in [
                    tokenizer.cls_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.unk_token_id,
                ]
                if tid is not None
            }
            logger.info("Loaded BAAI/bge-m3 sparse_linear.pt head successfully.")
        return self._sparse_linear

    def _extract_sparse_weights(
        self, input_ids_tensor, token_weights_tensor, tokenizer
    ) -> List[dict[str, float]]:
        """Convert token IDs and ReLU sparse weights into {token_str: max_weight} dicts."""
        unused = self._unused_token_ids or set()
        input_ids_list = input_ids_tensor.detach().cpu().tolist()
        weights_list = token_weights_tensor.detach().cpu().tolist()

        batch_sparse: List[dict[str, float]] = []
        for seq_ids, seq_weights in zip(input_ids_list, weights_list):
            token_id_weights: dict[int, float] = {}
            for tid, w in zip(seq_ids, seq_weights):
                if tid in unused or w <= 0.0:
                    continue
                if w > token_id_weights.get(tid, 0.0):
                    token_id_weights[tid] = float(w)

            # Convert token IDs to readable subword tokens
            lexical_dict: dict[str, float] = {}
            for tid, w in token_id_weights.items():
                tok_str = tokenizer.convert_ids_to_tokens(tid)
                clean_tok = tok_str.lstrip("▁").strip() or tok_str
                # Use exact token string key to preserve 1-to-1 vocabulary fidelity,
                # or store clean token when unambiguous
                key = tok_str
                if w > lexical_dict.get(key, 0.0):
                    lexical_dict[key] = round(w, 6)
            batch_sparse.append(lexical_dict)
        return batch_sparse

    def encode_sparse_batch(self, texts: List[str], batch_size: int = 16) -> List[dict[str, float]]:
        """Encode a list of texts into BAAI/bge-m3 native sparse lexical weight dictionaries."""
        import torch

        if not texts:
            return []

        model = self.get_model()
        sparse_linear = self.get_sparse_linear()
        auto_model = model[0].auto_model
        tokenizer = model.tokenizer
        device = next(auto_model.parameters()).device
        max_len = getattr(model, "max_seq_length", 512) or 512

        all_sparse: List[dict[str, float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                outputs = auto_model(**encoded, return_dict=True)
                last_hidden = outputs.last_hidden_state
                token_weights = torch.relu(sparse_linear(last_hidden)).squeeze(-1)
                batch_dicts = self._extract_sparse_weights(
                    encoded["input_ids"], token_weights, tokenizer
                )
                all_sparse.extend(batch_dicts)

        return all_sparse

    def encode_sparse_query(self, query: str) -> dict[str, float]:
        """Encode a single query string into BAAI/bge-m3 native sparse lexical weights."""
        res = self.encode_sparse_batch([query], batch_size=1)
        return res[0] if res else {}

    def encode_query_both(self, query: str) -> Tuple[List[float], dict[str, float]]:
        """
        Single-pass joint encoding of a user query into both:
        1) 1024-d L2-normalized dense vector (CLS pooling)
        2) Native sparse lexical weights dictionary {token_str: weight}
        """
        import torch
        import torch.nn.functional as F

        model = self.get_model()
        sparse_linear = self.get_sparse_linear()
        auto_model = model[0].auto_model
        tokenizer = model.tokenizer
        device = next(auto_model.parameters()).device
        max_len = getattr(model, "max_seq_length", 512) or 512

        with torch.no_grad():
            encoded = tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            outputs = auto_model(**encoded, return_dict=True)
            last_hidden = outputs.last_hidden_state

            # 1. Dense CLS vector + L2 normalization
            cls_vec = F.normalize(last_hidden[:, 0], p=2, dim=-1)[0]
            dense_vec = cls_vec.detach().cpu().tolist()

            # 2. Sparse lexical weights
            token_weights = torch.relu(sparse_linear(last_hidden)).squeeze(-1)
            sparse_weights = self._extract_sparse_weights(
                encoded["input_ids"], token_weights, tokenizer
            )[0]

        return dense_vec, sparse_weights

    def encode_query(self, query: str) -> List[float]:
        """Encode a user query text into a 1024-d normalized vector."""
        model = self.get_model()
        vector = model.encode(query, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def generate_embeddings(self, db: Session, force: bool = False) -> Tuple[int, int, float]:
        """
        Generate BAAI/bge-m3 embeddings for database records missing vectors.
        Returns (generated_count, total_count_with_embeddings, execution_time_ms).
        """
        start_time = time.perf_counter()

        query = db.query(Lexicon)
        if not force:
            query = query.filter(Lexicon.embedding.is_(None))

        records_to_encode = query.all()
        if not records_to_encode:
            total_with_emb = db.query(Lexicon).filter(Lexicon.embedding.isnot(None)).count()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return 0, total_with_emb, elapsed_ms

        logger.info(f"Generating embeddings for {len(records_to_encode)} lexicon records...")
        model = self.get_model()

        texts = [prepare_lexicon_text(rec) for rec in records_to_encode]
        batch_size = settings.EMBEDDING_BATCH_SIZE

        for i in range(0, len(records_to_encode), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_recs = records_to_encode[i : i + batch_size]

            vectors = model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for rec, vec in zip(batch_recs, vectors):
                rec.embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            db.commit()

        # Attempt to create HNSW index in PostgreSQL for fast vector retrieval
        is_postgres = db.bind is not None and db.bind.dialect.name == "postgresql"
        if is_postgres:
            try:
                db.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS lexicon_embedding_hnsw_idx "
                        "ON lexicons USING hnsw (embedding vector_cosine_ops);"
                    )
                )
                db.commit()
            except Exception as e:
                logger.warning(f"Note: HNSW index creation skipped or not supported: {e}")

        total_with_emb = db.query(Lexicon).filter(Lexicon.embedding.isnot(None)).count()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return len(records_to_encode), total_with_emb, elapsed_ms

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> List[Tuple[Lexicon, float]]:
        """
        Execute semantic vector similarity search using pgvector.
        Returns list of (Lexicon, cosine_similarity_score).
        """
        query_vector = self.encode_query(query)
        return self.search_by_vector(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )

    def search_by_vector(
        self,
        db: Session,
        query_vector: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
        culture: Optional[str] = None,
        semantic_domain: Optional[str] = None,
        word_class: Optional[str] = None,
    ) -> List[Tuple[Lexicon, float]]:
        """
        Execute semantic vector similarity search using a pre-computed 1024-d query vector.
        Returns list of (Lexicon, cosine_similarity_score).
        """
        is_postgres = db.bind is not None and db.bind.dialect.name == "postgresql"
        # 1. Native PostgreSQL pgvector search
        if is_postgres:
            vec_str = "[" + ",".join(str(f) for f in query_vector) + "]"

            # Build query with optional filters
            sql_parts = [
                "SELECT id, (1.0 - (embedding <=> CAST(:query_vec AS vector))) AS similarity",
                "FROM lexicons",
                "WHERE embedding IS NOT NULL",
            ]
            params: dict[str, Any] = {
                "query_vec": vec_str,
                "top_k": top_k,
                "min_sim": min_similarity,
            }

            if culture:
                sql_parts.append("AND LOWER(culture) LIKE :culture")
                params["culture"] = f"%{culture.lower()}%"
            if semantic_domain:
                sql_parts.append("AND LOWER(semantic_domain) LIKE :domain")
                params["domain"] = f"%{semantic_domain.lower()}%"
            if word_class:
                sql_parts.append("AND LOWER(word_class) LIKE :word_class")
                params["word_class"] = f"%{word_class.lower()}%"

            sql_parts.append("AND (1.0 - (embedding <=> CAST(:query_vec AS vector))) >= :min_sim")
            sql_parts.append("ORDER BY embedding <=> CAST(:query_vec AS vector) ASC")
            sql_parts.append("LIMIT :top_k")

            raw_sql = " ".join(sql_parts)
            rows = db.execute(text(raw_sql), params).fetchall()

            if not rows:
                return []

            matched_ids = [r[0] for r in rows]
            id_to_sim = {r[0]: float(r[1]) for r in rows}

            lexicons = db.query(Lexicon).filter(Lexicon.id.in_(matched_ids)).all()
            id_to_lex = {l.id: l for l in lexicons}

            results = []
            for item_id in matched_ids:
                if item_id in id_to_lex:
                    results.append((id_to_lex[item_id], round(id_to_sim[item_id], 4)))
            return results

        # 2. In-memory fallback (for SQLite testing environments)
        else:
            lexicons = db.query(Lexicon).filter(Lexicon.embedding.isnot(None)).all()
            candidates = []

            for lex in lexicons:
                if culture and (not lex.culture or culture.lower() not in lex.culture.lower()):
                    continue
                if semantic_domain and (not lex.semantic_domain or semantic_domain.lower() not in lex.semantic_domain.lower()):
                    continue
                if word_class and (not lex.word_class or word_class.lower() not in lex.word_class.lower()):
                    continue

                emb = lex.embedding
                if not emb:
                    continue

                if isinstance(emb, str):
                    emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]

                # Cosine similarity between normalized vectors = dot product
                sim = sum(a * b for a, b in zip(query_vector, emb))
                if sim >= min_similarity:
                    candidates.append((lex, round(float(sim), 4)))

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]


semantic_service = SemanticService()
