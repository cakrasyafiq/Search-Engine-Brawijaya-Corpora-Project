from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.types import UserDefinedType
from app.db.session import Base
from app.config import settings

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    class Vector(UserDefinedType):
        """Fallback Vector user-defined type when pgvector package is not yet installed."""
        cache_ok = True

        def __init__(self, dim: int = 1024):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"vector({self.dim})"

        def bind_processor(self, dialect):
            def process(value):
                if value is None:
                    return None
                if isinstance(value, (list, tuple)):
                    return "[" + ",".join(str(float(x)) for x in value) + "]"
                return str(value)
            return process

        def result_processor(self, dialect, coltype):
            def process(value):
                if value is None:
                    return None
                if isinstance(value, str):
                    val = value.strip("[]")
                    return [float(x) for x in val.split(",") if x.strip()]
                return value
            return process


class Lexicon(Base):
    __tablename__ = "lexicons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    term = Column(String(255), nullable=False, index=True)
    code = Column(String(100), nullable=True, index=True)
    culture = Column(String(100), nullable=True, index=True)
    sub_culture = Column(String(150), nullable=True)
    word_class = Column(String(100), nullable=True, index=True)
    language = Column(String(100), nullable=True)
    dialect = Column(String(100), nullable=True)
    semantic_domain = Column(String(150), nullable=True, index=True)
    usage_context = Column(String(150), nullable=True)
    register = Column(String(100), nullable=True)
    phonetic = Column(String(150), nullable=True)
    definition = Column(Text, nullable=True)
    etymology = Column(Text, nullable=True)
    variants = Column(Text, nullable=True)
    examples = Column(Text, nullable=True)
    reference = Column(Text, nullable=True)
    reference_url = Column(Text, nullable=True)
    categories = Column(String(255), nullable=True)
    contributors = Column(String(255), nullable=True)
    images_count = Column(Integer, default=0)
    audio_count = Column(Integer, default=0)
    video_count = Column(Integer, default=0)
    three_d_models_count = Column(Integer, default=0)
    status = Column(String(50), nullable=True)
    verification = Column(String(50), nullable=True)
    verified_by = Column(String(150), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    
    # 1024-dimensional embedding for BAAI/bge-m3 semantic search
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)

    def __repr__(self) -> str:
        return f"<Lexicon(id={self.id}, term='{self.term}', culture='{self.culture}')>"
