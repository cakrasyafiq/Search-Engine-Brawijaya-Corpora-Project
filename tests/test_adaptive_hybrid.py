import math
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
from app.services.adaptive_hybrid_service import (
    compute_sigmoid_confidence,
    compute_adaptive_alpha,
)

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
# Unit Tests for Sigmoid Confidence Gate Math
# -------------------------------------------------------------

def test_sigmoid_confidence_zero_bm25():
    assert compute_sigmoid_confidence(0.0, tau=8.0, scale=1.0) == 0.0
    assert compute_sigmoid_confidence(-1.0, tau=8.0, scale=1.0) == 0.0

    alpha_q, conf = compute_adaptive_alpha(
        max_bm25=0.0, tau=8.0, scale=1.0, alpha_min=0.4, alpha_max=1.0
    )
    assert conf == 0.0
    assert alpha_q == 1.0


def test_sigmoid_confidence_midpoint():
    conf = compute_sigmoid_confidence(8.0, tau=8.0, scale=1.0)
    assert math.isclose(conf, 0.5, rel_tol=1e-9)

    alpha_q, conf_a = compute_adaptive_alpha(
        max_bm25=8.0, tau=8.0, scale=1.0, alpha_min=0.4, alpha_max=1.0
    )
    assert math.isclose(conf_a, 0.5, rel_tol=1e-9)
    assert math.isclose(alpha_q, 0.7, rel_tol=1e-9)


def test_adaptive_alpha_monotonic_decrease():
    scores = [0.0, 2.0, 5.0, 8.0, 12.0, 20.0]
    alphas = [
        compute_adaptive_alpha(s, tau=8.0, scale=1.5, alpha_min=0.3, alpha_max=1.0)[0]
        for s in scores
    ]
    for prev, curr in zip(alphas, alphas[1:]):
        assert curr <= prev
    assert alphas[0] == 1.0
    assert alphas[-1] < 0.31


def test_sigmoid_confidence_invalid_scale():
    with pytest.raises(ValueError):
        compute_sigmoid_confidence(5.0, tau=8.0, scale=0.0)


# -------------------------------------------------------------
# Integration Tests for /search/adaptive-hybrid Endpoint
# -------------------------------------------------------------

def test_adaptive_hybrid_endpoint_success(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    response = client.get("/search/adaptive-hybrid?query=protection rite Among-among")
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "protection rite Among-among"
    assert data["total_candidates"] >= 1
    assert 0.0 <= data["adaptive_alpha"] <= 1.0
    assert math.isclose(data["dense_weight"] + data["sparse_weight"], 1.0, abs_tol=1e-5)
    assert data["max_bm25_score"] > 0.0
    assert 0.0 < data["lexical_confidence"] <= 1.0
    assert data["results"][0]["term"] == "Among-among"


def test_adaptive_hybrid_zero_bm25_falls_back_to_full_semantic(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    response = client.get("/search/adaptive-hybrid?query=nonexistentconceptualtermxyz")
    assert response.status_code == 200
    data = response.json()

    assert data["max_bm25_score"] == 0.0
    assert data["lexical_confidence"] == 0.0
    assert data["adaptive_alpha"] == 1.0
    assert data["dense_weight"] == 1.0
    assert data["sparse_weight"] == 0.0
    assert data["results"][0]["term"] == "Among-among"


def test_adaptive_hybrid_custom_gate_params(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    response = client.get(
        "/search/adaptive-hybrid?query=bamboo&tau=100.0&scale=0.5&alpha_min=0.2&alpha_max=0.95"
    )
    assert response.status_code == 200
    data = response.json()

    assert data["gate_tau"] == 100.0
    assert data["gate_scale"] == 0.5
    assert data["gate_alpha_min"] == 0.2
    assert data["gate_alpha_max"] == 0.95
    assert math.isclose(data["adaptive_alpha"], 0.95, abs_tol=1e-4)


def test_adaptive_hybrid_empty_query_returns_400(client):
    response = client.get("/search/adaptive-hybrid?query=   ")
    assert response.status_code == 400


def test_existing_hybrid_endpoint_preserved(client, monkeypatch):
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    response = client.get("/search/hybrid?query=bamboo&alpha=0.6")
    assert response.status_code == 200
    data = response.json()
    assert data["dense_weight"] == 0.6
    assert data["sparse_weight"] == 0.4
