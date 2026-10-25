import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.session import Base, get_db
from app.db.models import Lexicon
from app.main import app
from app.services.search_service import search_service

from sqlalchemy.pool import StaticPool

# Use in-memory SQLite database with StaticPool for testing API endpoints
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

    # Seed test sample
    session = TestingSessionLocal()
    sample_lex = Lexicon(
        term="Among-among",
        code="T/KK/1",
        culture="Tengger",
        word_class="Noun",
        semantic_domain="Ritual",
        definition="An Among-among is Tengger's protection rite for 44-day-old babies.",
        variants="selapanan\nslametan bayi",
    )
    session.add(sample_lex)
    session.commit()

    # Rebuild search service with test database
    search_service.reindex(session)
    session.close()

    with TestClient(app) as c:
        yield c

    Base.metadata.drop_all(bind=test_engine)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["bm25_index_ready"] is True


def test_search_endpoint(client):
    response = client.get("/search?query=Among-among")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Among-among"
    assert data["total_candidates"] >= 1
    assert data["results"][0]["term"] == "Among-among"
    assert data["results"][0]["bm25_score"] > 0


def test_search_endpoint_variants(client):
    response = client.get("/search?query=slametan%20bayi")
    assert response.status_code == 200
    data = response.json()
    assert data["total_candidates"] >= 1
    assert data["results"][0]["term"] == "Among-among"


def test_search_endpoint_empty_query(client):
    response = client.get("/search?query=")
    assert response.status_code == 400


def test_semantic_search_endpoint(client, monkeypatch):
    from app.services.semantic_service import semantic_service

    # Mock encode_query to return a fixed 1024-d unit vector
    dummy_vec = [1.0 / (1024 ** 0.5)] * 1024
    monkeypatch.setattr(semantic_service, "encode_query", lambda q: dummy_vec)

    # Assign embedding to the test record in DB
    session = TestingSessionLocal()
    rec = session.query(Lexicon).filter(Lexicon.term == "Among-among").first()
    rec.embedding = dummy_vec
    session.commit()
    session.close()

    response = client.get("/search/semantic?query=protection%20rite")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "protection rite"
    assert data["model"] == "BAAI/bge-m3"
    assert data["total_candidates"] >= 1
    assert data["results"][0]["term"] == "Among-among"
    assert "similarity_score" in data["results"][0]
    assert data["results"][0]["similarity_score"] > 0.99
