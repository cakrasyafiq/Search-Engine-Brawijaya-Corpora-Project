import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.db.models import Lexicon
from app.main import app
from app.search.bm25_index import DocumentRecord
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.bge_sparse_service import BGESparseInvertedIndex, bge_sparse_service

TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db

    session = TestingSessionLocal()
    dummy_vec_1 = [1.0 / (1024 ** 0.5)] * 1024
    dummy_vec_2 = [-1.0 / (1024 ** 0.5)] * 1024

    lex1 = Lexicon(
        id=1,
        term="Among-among",
        code="T/KK/1",
        culture="Tengger",
        word_class="Noun",
        semantic_domain="Ritual",
        definition="An Among-among is Tengger's protection rite for 44-day-old babies.",
        variants="selapanan\nslametan bayi",
        embedding=dummy_vec_1,
    )
    lex2 = Lexicon(
        id=2,
        term="Ancak",
        code="T/TT/S/9",
        culture="Tengger",
        word_class="Noun",
        semantic_domain="Material Culture",
        definition="A square-shaped offering container made of woven bamboo.",
        variants="",
        embedding=dummy_vec_2,
    )
    session.add_all([lex1, lex2])
    session.commit()

    search_service.reindex(session)
    # Build mock BGE sparse index in-memory
    doc1 = DocumentRecord(
        id=1,
        term="Among-among",
        code="T/KK/1",
        culture="Tengger",
        word_class="Noun",
        semantic_domain="Ritual",
        definition="An Among-among is Tengger's protection rite for 44-day-old babies.",
        variants="selapanan\nslametan bayi",
    )
    doc2 = DocumentRecord(
        id=2,
        term="Ancak",
        code="T/TT/S/9",
        culture="Tengger",
        word_class="Noun",
        semantic_domain="Material Culture",
        definition="A square-shaped offering container made of woven bamboo.",
        variants="",
    )
    bge_sparse_service.index.build_index(
        [doc1, doc2],
        [
            {"▁Among": 0.32, "▁protection": 0.25, "▁rite": 0.28, "▁babies": 0.22},
            {"▁Ancak": 0.35, "▁bamboo": 0.30, "▁offering": 0.24},
        ],
    )
    bge_sparse_service.is_ready = True
    session.close()

    with TestClient(app) as c:
        yield c

    Base.metadata.drop_all(bind=test_engine)


# -------------------------------------------------------------
# Unit Tests for BGESparseInvertedIndex
# -------------------------------------------------------------

def test_bge_sparse_index_inner_product():
    idx = BGESparseInvertedIndex()
    d1 = DocumentRecord(id=1, term="Barikan", culture="Tengger", semantic_domain="Ritual")
    d2 = DocumentRecord(id=2, term="Saron", culture="Javanese", semantic_domain="Music")
    idx.build_index(
        [d1, d2],
        [
            {"▁Bar": 0.4, "ikan": 0.3, "▁ritual": 0.2},
            {"▁Saron": 0.5, "▁gamelan": 0.3},
        ],
    )

    res = idx.search_by_weights({"▁Bar": 0.5, "▁ritual": 0.5}, top_k=5)
    assert len(res) == 1
    assert res[0].document.term == "Barikan"
    # Expected inner product: 0.4*0.5 + 0.2*0.5 = 0.3
    assert abs(res[0].score - 0.3) < 1e-4
    assert "Bar" in res[0].matched_terms
    assert "ritual" in res[0].matched_terms


# -------------------------------------------------------------
# Integration Tests for /search/sparse-bge Endpoint
# -------------------------------------------------------------

def test_search_sparse_bge_success(client, monkeypatch):
    monkeypatch.setattr(
        semantic_service,
        "encode_sparse_query",
        lambda q: {"▁Among": 0.4, "▁rite": 0.3},
    )

    response = client.get("/search/sparse-bge?query=Among-among%20rite")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Among-among rite"
    assert data["model"] == "BAAI/bge-m3"
    assert data["total_candidates"] == 1
    first = data["results"][0]
    assert first["term"] == "Among-among"
    assert first["sparse_score"] > 0
    assert "Among" in first["matched_tokens"]


def test_search_sparse_bge_empty_query(client):
    response = client.get("/search/sparse-bge?query=")
    assert response.status_code == 400


# -------------------------------------------------------------
# Integration Tests for /search/hybrid-bge Endpoint
# -------------------------------------------------------------

def test_search_hybrid_bge_success(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(
        semantic_service,
        "encode_query_both",
        lambda q: (dummy_vec, {"▁Among": 0.4, "▁rite": 0.3}),
    )

    response = client.get("/search/hybrid-bge?query=protection%20rite%20Among-among")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "protection rite Among-among"
    assert data["dense_weight"] == 0.5
    assert data["sparse_weight"] == 0.5
    assert data["total_candidates"] >= 1

    first = data["results"][0]
    assert first["term"] == "Among-among"
    assert "hybrid_score" in first
    assert "sparse_score" in first
    assert "normalized_sparse_score" in first
    assert "similarity_score" in first
    assert "normalized_similarity_score" in first
    assert first["hybrid_score"] == 1.0


def test_search_hybrid_bge_alpha_weights(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(
        semantic_service,
        "encode_query_both",
        lambda q: (dummy_vec, {"▁bamboo": 0.4}),
    )

    res_dense = client.get("/search/hybrid-bge?query=bamboo&alpha=1.0")
    assert res_dense.status_code == 200
    assert res_dense.json()["dense_weight"] == 1.0
    assert res_dense.json()["sparse_weight"] == 0.0
    assert res_dense.json()["results"][0]["term"] == "Among-among"

    res_sparse = client.get("/search/hybrid-bge?query=bamboo&alpha=0.0")
    assert res_sparse.status_code == 200
    assert res_sparse.json()["dense_weight"] == 0.0
    assert res_sparse.json()["sparse_weight"] == 1.0
    assert res_sparse.json()["results"][0]["term"] == "Ancak"

