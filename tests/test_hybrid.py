import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.db.models import Lexicon
from app.main import app
from app.services.search_service import search_service
from app.services.semantic_service import semantic_service
from app.services.hybrid_service import min_max_normalize, hybrid_service

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
    session.close()

    with TestClient(app) as c:
        yield c

    Base.metadata.drop_all(bind=test_engine)


# -------------------------------------------------------------
# Unit Tests for min_max_normalize
# -------------------------------------------------------------

def test_min_max_normalize_empty():
    assert min_max_normalize({}) == {}


def test_min_max_normalize_single():
    # Single positive score -> 1.0
    assert min_max_normalize({1: 5.2}) == {1: 1.0}
    # Single zero score -> 0.0
    assert min_max_normalize({1: 0.0}) == {1: 0.0}


def test_min_max_normalize_identical():
    # Multiple identical positive scores -> 1.0
    result = min_max_normalize({1: 4.0, 2: 4.0})
    assert result == {1: 1.0, 2: 1.0}


def test_min_max_normalize_range():
    # Normal range [10, 20, 30] -> [0.0, 0.5, 1.0]
    scores = {1: 10.0, 2: 20.0, 3: 30.0}
    norm = min_max_normalize(scores)
    assert norm[1] == 0.0
    assert norm[2] == 0.5
    assert norm[3] == 1.0


# -------------------------------------------------------------
# Integration Tests for /search/hybrid Endpoint
# -------------------------------------------------------------

def test_search_hybrid_success(client, monkeypatch):
    # Mock semantic encode query to match lex1's embedding
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    response = client.get("/search/hybrid?query=protection rite Among-among")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "protection rite Among-among"
    assert data["total_candidates"] >= 1
    assert data["dense_weight"] == 0.5
    assert data["sparse_weight"] == 0.5

    first_result = data["results"][0]
    assert first_result["term"] == "Among-among"
    assert "hybrid_score" in first_result
    assert "bm25_score" in first_result
    assert "normalized_bm25_score" in first_result
    assert "similarity_score" in first_result
    assert "normalized_similarity_score" in first_result
    assert first_result["hybrid_score"] > 0


def test_search_hybrid_alpha_weight(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    # alpha = 1.0 means dense only
    response = client.get("/search/hybrid?query=protection rite&alpha=1.0")
    assert response.status_code == 200
    data = response.json()
    assert data["dense_weight"] == 1.0
    assert data["sparse_weight"] == 0.0

    # alpha = 0.0 means sparse only
    response_bm25 = client.get("/search/hybrid?query=protection rite&alpha=0.0")
    assert response_bm25.status_code == 200
    data_bm25 = response_bm25.json()
    assert data_bm25["dense_weight"] == 0.0
    assert data_bm25["sparse_weight"] == 1.0


def test_search_hybrid_custom_weights(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    # dense_weight=3.0, sparse_weight=1.0 -> 0.75 and 0.25
    response = client.get("/search/hybrid?query=bamboo&dense_weight=3.0&sparse_weight=1.0")
    assert response.status_code == 200
    data = response.json()
    assert data["dense_weight"] == 0.75
    assert data["sparse_weight"] == 0.25


def test_search_hybrid_empty_query(client):
    response = client.get("/search/hybrid?query=")
    assert response.status_code == 400
    assert "must not be empty" in response.json()["detail"]


def test_search_hybrid_filtering(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    # Matching domain "Ritual"
    res1 = client.get("/search/hybrid?query=rite&semantic_domain=Ritual")
    assert res1.status_code == 200
    data1 = res1.json()
    for item in data1["results"]:
        assert item["semantic_domain"] == "Ritual"

    # Non-existent domain
    res2 = client.get("/search/hybrid?query=rite&semantic_domain=NonExistentDomain")
    assert res2.status_code == 200
    assert res2.json()["total_candidates"] == 0


def test_health_and_root_include_hybrid(client):
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.json()["hybrid_search_ready"] is True

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert "hybrid_search" in health_res.json()
    assert health_res.json()["hybrid_search"]["ready"] is True
