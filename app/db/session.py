from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

# Normalize postgresql:// to a working driver
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    # Check available drivers in order of reliability for Windows policy: pg8000 -> psycopg2 -> psycopg
    try:
        import pg8000  # type: ignore # noqa: F401
        db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)
    except ImportError:
        try:
            import psycopg2  # type: ignore # noqa: F401
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            pass

# Configure engine kwargs depending on dialect
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db_extensions():
    """Ensure vector extension is enabled and embedding column exists in PostgreSQL."""
    if "postgresql" in db_url:
        from sqlalchemy import text
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
        except Exception as e:
            print(f"Note on vector extension creation: {e}")

        try:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE lexicons ADD COLUMN IF NOT EXISTS embedding vector({settings.EMBEDDING_DIMENSION});"))
                conn.commit()
        except Exception:
            pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
