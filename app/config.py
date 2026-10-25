import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "Lexicon BM25 Search Engine"
    DEBUG: bool = True
    DATABASE_URL: str = "postgresql+pg8000://postgres:postgres@localhost:5432/lexicon_db"
    
    # BM25 Hyperparameters
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    
    # Path to persist the compiled BM25 inverted index
    INDEX_CACHE_PATH: str = "bm25_index.pkl"

    # Semantic Vector Search Settings (BAAI/bge-m3)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 16
    
    # Default CSV path for database seeding
    LEXICON_CSV_PATH: str = str(
        Path(__file__).resolve().parent.parent / "Dataset" / "Lexicon" / "Lexicon.csv"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
