# Lexicon Search Engine Backend (FastAPI + BM25 + PostgreSQL)

A high-performance cultural lexicon search engine built with **FastAPI**, **PostgreSQL**, and an in-memory **BM25 (Best Matching 25)** inverted index engine with persistence and field-weighted relevance.

---

## 📁 Project Structure

```
Local_Experiment/
├── app/
│   ├── __init__.py
│   ├── config.py             # Environment settings & hyperparameter configs
│   ├── main.py               # FastAPI application with search & lexicon endpoints
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py        # SQLAlchemy engine, session maker & DB dependency
│   │   ├── models.py         # Lexicon PostgreSQL table schema (29 columns)
│   │   └── seed.py           # Database seeder script for Lexicon.csv
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── lexicon.py        # Pydantic schemas for requests, responses & search candidates
│   ├── search/
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # Unicode normalizer, accent stripper & compound tokenizer
│   │   └── bm25_index.py     # Inverted index data structure & Okapi BM25 ranking engine
│   └── services/
│       ├── __init__.py
│       └── search_service.py # BM25 lifecycle manager & cache synchronization
├── Dataset/
│   └── Lexicon/
│       └── Lexicon.csv       # Source lexicon dataset
├── tests/
│   ├── __init__.py
│   ├── test_bm25.py          # Unit tests for BM25 ranking & tokenization
│   └── test_api.py           # FastAPI endpoint tests
├── docker-compose.yml        # PostgreSQL container setup
├── .env.example              # Environment variables template
├── .env                      # Local development environment configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Setup & user guide
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup & Dependency Installation

Activate your Python environment and install the required dependencies:

```powershell
# Using the existing virtual environment in PowerShell:
.\.venv\Scripts\Activate.ps1

# Or create a new virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies:
pip install -r requirements.txt
```

---

### 2. PostgreSQL Database Setup

#### Option A: Quickstart with Docker Compose
If you have Docker running, launch PostgreSQL with a single command:
```powershell
docker compose up -d
```
This starts PostgreSQL 16 on port `5432` with user `postgres`, password `postgres`, and database `lexicon_db`.

#### Option B: Existing PostgreSQL Server
If using an existing PostgreSQL installation, create the database:
```sql
CREATE DATABASE lexicon_db;
```

#### Configure `.env`
Ensure `.env` matches your PostgreSQL connection string:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lexicon_db
BM25_K1=1.5
BM25_B=0.75
INDEX_CACHE_PATH=bm25_index.pkl
APP_NAME="Lexicon BM25 Search Engine"
DEBUG=True
```

---

### 3. Seed the Database from `Lexicon.csv`

Run the seeder script to create the table schema and populate all 169 entries from `Dataset/Lexicon/Lexicon.csv`:

```powershell
python -m app.db.seed
```

> **Note**: To force a fresh truncate and re-import, add `--truncate`:
> ```powershell
> python -m app.db.seed --truncate
> ```

---

### 4. Running the Tests

Verify that BM25 calculations, tokenization, and API endpoints work as expected:

```powershell
pytest -v
```

---

### 5. Start the FastAPI Development Server

Start the API with hot-reload enabled:

```powershell
uvicorn app.main:app --reload --port 8000
```

The API will now be live at:
- **Root URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## 🔍 API Usage Examples

### 1. BM25 Search Endpoint (`GET /search`)

Search candidate lexicons by query text:

```http
GET /search?query=Among-among
```
or using `q`:
```http
GET /search?q=slametan%20bayi&top_k=5
```

#### Optional Query Filters:
- `query` / `q`: The search term (e.g., `"slametan"`, `"offering"`, `"bamboo"`, `"Among-among"`)
- `top_k`: Maximum candidate results to return (default: `10`, range: `1` to `100`)
- `min_score`: Minimum BM25 score cutoff (default: `0.0`)
- `culture`: Filter by culture name (e.g., `Tengger`)
- `semantic_domain`: Filter by semantic domain (e.g., `Ritual`, `Material Culture`)
- `word_class`: Filter by word class (e.g., `Noun`)

#### Example Response:
```json
{
  "query": "slametan bayi",
  "total_candidates": 1,
  "top_k": 5,
  "execution_time_ms": 0.42,
  "results": [
    {
      "id": 1,
      "term": "Among-among",
      "code": "T/KK/1",
      "culture": "Tengger",
      "sub_culture": "Tengger — General",
      "word_class": "Noun",
      "language": "Tenggerese",
      "dialect": "Tengger",
      "semantic_domain": "Ritual",
      "usage_context": "Ceremony",
      "register": "Ritual",
      "phonetic": "[amɔŋ-amɔŋ]",
      "definition": "An Among-among is Tengger's protection rite for 44-day-old babies...",
      "variants": "selapanan\nslametan bayi\nmitoni cilik\nbancakan anak",
      "examples": "The Tengger people observe the Among-among tradition...",
      "categories": "Ritual & Belief",
      "bm25_score": 4.8214,
      "matched_terms": ["slametan", "bayi"]
    }
  ]
}
```

---

### 2. Semantic Vector Search Endpoint (`GET /search/semantic`)

Search candidate lexicons by semantic meaning using **BAAI/bge-m3** (1024-d dense vectors) and **pgvector**:

```http
GET /search/semantic?query=protective%20ceremony%20for%20infant&top_k=5
```

#### Optional Query Filters:
- `query` / `q`: Natural language search query
- `top_k`: Maximum candidate results to return (default: `10`, range: `1` to `100`)
- `min_similarity`: Minimum cosine similarity score cutoff (`0.0` to `1.0`, default: `0.0`)
- `culture`: Filter by culture name (e.g., `Tengger`)
- `semantic_domain`: Filter by semantic domain (e.g., `Ritual`, `Material Culture`)
- `word_class`: Filter by word class (e.g., `Noun`)

#### Example Response:
```json
{
  "query": "protective ceremony for infant",
  "model": "BAAI/bge-m3",
  "total_candidates": 2,
  "top_k": 5,
  "execution_time_ms": 28.5,
  "results": [
    {
      "id": 1,
      "term": "Among-among",
      "code": "T/KK/1",
      "culture": "Tengger",
      "semantic_domain": "Ritual",
      "definition": "An Among-among is Tengger's protection rite for 44-day-old babies...",
      "variants": "selapanan\nslametan bayi\nmitoni cilik\nbancakan anak",
      "similarity_score": 0.8412
    }
  ]
}
```

---

### 3. All Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?query=...` | Keyword & phrase search using Okapi BM25 |
| `GET` | `/search/semantic?query=...` | Dense vector semantic search using BAAI/bge-m3 & pgvector |
| `GET` | `/lexicons` | List all lexicons paginated (`skip`, `limit`) |
| `GET` | `/lexicons/{id}` | Retrieve full details of a single lexicon entry |
| `POST` | `/admin/reindex` | Force rebuild the BM25 inverted index from PostgreSQL |
| `POST` | `/admin/generate-embeddings` | Generate BAAI/bge-m3 vector embeddings for PostgreSQL records |
| `GET` | `/health` | Check database connection, BM25 index status, and vector counts |

---

## 🧬 Generating Embeddings (CLI)

To compute embeddings for all records using `BAAI/bge-m3` directly from the command line:

```powershell
python -m app.db.embed
```

To force recompute all embeddings:
```powershell
python -m app.db.embed --force
```
