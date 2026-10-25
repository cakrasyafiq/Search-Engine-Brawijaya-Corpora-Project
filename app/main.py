import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Depends, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import engine, Base, get_db, SessionLocal, init_db_extensions
from app.db.models import Lexicon
from app.db.seed import seed_database
from app.schemas.lexicon import (
    SearchResponse,
    LexiconCandidate,
    LexiconDetail,
    PaginatedLexicons,
    ReindexResponse,
    SemanticLexiconCandidate,
    SemanticSearchResponse,
    EmbeddingGenerationResponse,
)
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s"
)
logger = logging.getLogger("lexicon_search_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing database and search engine...")
    try:
        # Initialize PostgreSQL extensions (e.g. vector)
        init_db_extensions()

        # Create database tables if they do not exist
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified.")

        # Check if database has records; auto-seed from CSV if empty
        with SessionLocal() as db:
            count = db.query(Lexicon).count()
            if count == 0:
                logger.info("Database is empty. Automatically seeding from Lexicon.csv...")
                seed_database()
            else:
                logger.info(f"Database contains {count} lexicon entries.")

            # Initialize in-memory BM25 inverted index
            search_service.initialize(db)
            logger.info(f"BM25 Search Service ready with {search_service.index.total_docs} records indexed.")

    except Exception as e:
        logger.error(f"Startup initialization warning: {e}")
        logger.warning("FastAPI will start, but database operations may need connection verification.")

    yield
    logger.info("Shutting down Lexicon Search Engine API.")


app = FastAPI(
    title=settings.APP_NAME,
    description="High-performance Cultural Lexicon Search Engine API using BM25 Ranking and pgvector BAAI/bge-m3 Semantic Search",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Root & Status")
def root():
    return {
        "service": settings.APP_NAME,
        "version": "1.1.0",
        "docs_url": "/docs",
        "bm25_index_ready": search_service.is_ready,
        "bm25_indexed_documents": search_service.index.total_docs,
        "semantic_search_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
    }


@app.get("/health", summary="Service Health Check")
def health_check(db: Session = Depends(get_db)):
    db_connected = False
    db_count = 0
    records_with_embeddings = 0
    try:
        db_count = db.query(Lexicon).count()
        records_with_embeddings = db.query(Lexicon).filter(Lexicon.embedding.isnot(None)).count()
        db_connected = True
    except Exception as e:
        logger.warning(f"Health check DB ping failed: {e}")

    return {
        "status": "healthy" if (db_connected and search_service.is_ready) else "degraded",
        "database": {
            "connected": db_connected,
            "total_records": db_count,
            "records_with_embeddings": records_with_embeddings,
        },
        "search_index": {
            "ready": search_service.is_ready,
            "indexed_records": search_service.index.total_docs,
            "average_doc_length": round(search_service.index.avgdl, 2),
            "k1": settings.BM25_K1,
            "b": settings.BM25_B,
        },
        "semantic_vector_search": {
            "model": settings.EMBEDDING_MODEL_NAME,
            "dimension": settings.EMBEDDING_DIMENSION,
            "embedded_records": records_with_embeddings,
        },
    }


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Search Lexicon Database using BM25",
    description="Receive a user query and return candidate lexicons ranked by Okapi BM25 relevance score."
)
def search_lexicons(
    q: Optional[str] = Query(None, description="Search query string (alternative to 'query')"),
    query: Optional[str] = Query(None, description="Search query string"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of candidates to return"),
    min_score: float = Query(0.0, ge=0.0, description="Minimum BM25 score threshold"),
    culture: Optional[str] = Query(None, description="Filter candidates by culture (e.g. Tengger)"),
    semantic_domain: Optional[str] = Query(None, description="Filter candidates by semantic domain (e.g. Ritual)"),
    word_class: Optional[str] = Query(None, description="Filter candidates by word class (e.g. Noun)"),
):
    search_query = (query or q or "").strip()
    if not search_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' or 'query' must not be empty.",
        )

    start_time = time.perf_counter()

    results = search_service.search(
        query=search_query,
        top_k=top_k,
        min_score=min_score,
        culture=culture,
        semantic_domain=semantic_domain,
        word_class=word_class,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    candidates = [
        LexiconCandidate(
            id=res.document.id,
            term=res.document.term,
            code=res.document.code,
            culture=res.document.culture,
            sub_culture=res.document.sub_culture,
            word_class=res.document.word_class,
            language=res.document.language,
            dialect=res.document.dialect,
            semantic_domain=res.document.semantic_domain,
            usage_context=res.document.usage_context,
            register=res.document.register,
            phonetic=res.document.phonetic,
            definition=res.document.definition,
            variants=res.document.variants,
            examples=res.document.examples,
            categories=res.document.categories,
            bm25_score=res.score,
            matched_terms=res.matched_terms,
        )
        for res in results
    ]

    return SearchResponse(
        query=search_query,
        total_candidates=len(candidates),
        top_k=top_k,
        execution_time_ms=round(elapsed_ms, 3),
        results=candidates,
    )


@app.get(
    "/search/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic Vector Search using BAAI/bge-m3 and pgvector",
    description="Receive a user query, compute a 1024-d dense embedding via BAAI/bge-m3, and return candidate lexicons ranked by cosine similarity from pgvector."
)
def search_semantic(
    q: Optional[str] = Query(None, description="Search query string (alternative to 'query')"),
    query: Optional[str] = Query(None, description="Search query string"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of candidates to return"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff (0.0 to 1.0)"),
    culture: Optional[str] = Query(None, description="Filter candidates by culture (e.g. Tengger)"),
    semantic_domain: Optional[str] = Query(None, description="Filter candidates by semantic domain (e.g. Ritual)"),
    word_class: Optional[str] = Query(None, description="Filter candidates by word class (e.g. Noun)"),
    db: Session = Depends(get_db),
):
    search_query = (query or q or "").strip()
    if not search_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' or 'query' must not be empty.",
        )

    start_time = time.perf_counter()

    try:
        results = semantic_service.search(
            db=db,
            query=search_query,
            top_k=top_k,
            min_similarity=min_similarity,
            culture=culture,
            semantic_domain=semantic_domain,
            word_class=word_class,
        )
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    candidates = [
        SemanticLexiconCandidate(
            id=lex.id,
            term=lex.term,
            code=lex.code,
            culture=lex.culture,
            sub_culture=lex.sub_culture,
            word_class=lex.word_class,
            language=lex.language,
            dialect=lex.dialect,
            semantic_domain=lex.semantic_domain,
            usage_context=lex.usage_context,
            register=lex.register,
            phonetic=lex.phonetic,
            definition=lex.definition,
            variants=lex.variants,
            examples=lex.examples,
            categories=lex.categories,
            similarity_score=sim_score,
        )
        for lex, sim_score in results
    ]

    return SemanticSearchResponse(
        query=search_query,
        model=settings.EMBEDDING_MODEL_NAME,
        total_candidates=len(candidates),
        top_k=top_k,
        execution_time_ms=round(elapsed_ms, 3),
        results=candidates,
    )


@app.post(
    "/admin/generate-embeddings",
    response_model=EmbeddingGenerationResponse,
    summary="Compute BAAI/bge-m3 Embeddings for PostgreSQL records",
    description="Generate vector embeddings for all lexicon records missing vectors in PostgreSQL."
)
def trigger_generate_embeddings(
    force: bool = Query(False, description="Regenerate embeddings for all records even if already present"),
    db: Session = Depends(get_db),
):
    try:
        generated_count, total_count, elapsed_ms = semantic_service.generate_embeddings(db, force=force)
        return EmbeddingGenerationResponse(
            status="success",
            model=settings.EMBEDDING_MODEL_NAME,
            embeddings_generated=generated_count,
            total_with_embeddings=total_count,
            execution_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}",
        )


@app.get(
    "/lexicons",
    response_model=PaginatedLexicons,
    summary="List All Lexicons (Paginated)",
)
def list_lexicons(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.query(Lexicon).count()
    items = db.query(Lexicon).offset(skip).limit(limit).all()
    return PaginatedLexicons(
        total=total,
        skip=skip,
        limit=limit,
        items=items,
    )


@app.get(
    "/lexicons/{lexicon_id}",
    response_model=LexiconDetail,
    summary="Get Lexicon Details by ID",
)
def get_lexicon_by_id(lexicon_id: int, db: Session = Depends(get_db)):
    lexicon = db.query(Lexicon).filter(Lexicon.id == lexicon_id).first()
    if not lexicon:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lexicon entry with ID {lexicon_id} not found.",
        )
    return lexicon


@app.post(
    "/admin/reindex",
    response_model=ReindexResponse,
    summary="Rebuild BM25 Inverted Index from PostgreSQL",
)
def trigger_reindex(db: Session = Depends(get_db)):
    indexed_count, elapsed_ms = search_service.reindex(db)
    return ReindexResponse(
        status="success",
        indexed_count=indexed_count,
        execution_time_ms=round(elapsed_ms, 2),
    )
