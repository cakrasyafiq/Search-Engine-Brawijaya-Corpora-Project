from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LexiconCandidate(BaseModel):
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
    variants: Optional[str] = None
    examples: Optional[str] = None
    categories: Optional[str] = None
    bm25_score: float = Field(..., description="BM25 relevance score")
    matched_terms: List[str] = Field(default_factory=list, description="Query terms matched in document")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class SearchResponse(BaseModel):
    query: str
    total_candidates: int
    top_k: int
    execution_time_ms: float
    results: List[LexiconCandidate]


class LexiconDetail(BaseModel):
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
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class PaginatedLexicons(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[LexiconDetail]


class ReindexResponse(BaseModel):
    status: str
    indexed_count: int
    execution_time_ms: float


class SemanticLexiconCandidate(BaseModel):
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
    variants: Optional[str] = None
    examples: Optional[str] = None
    categories: Optional[str] = None
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class SemanticSearchResponse(BaseModel):
    query: str
    model: str = "BAAI/bge-m3"
    total_candidates: int
    top_k: int
    execution_time_ms: float
    results: List[SemanticLexiconCandidate]


class EmbeddingGenerationResponse(BaseModel):
    status: str
    model: str
    embeddings_generated: int
    total_with_embeddings: int
    execution_time_ms: float

