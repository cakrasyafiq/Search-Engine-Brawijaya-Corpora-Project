import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings
from app.db.session import engine, SessionLocal, Base
from app.db.models import Lexicon


def parse_int_safe(val: Optional[str], default: int = 0) -> int:
    if not val:
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


def parse_datetime_safe(val: Optional[str]) -> Optional[datetime]:
    if not val or not val.strip():
        return None
    val = val.strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def seed_database(csv_path: Optional[str] = None, truncate: bool = False) -> int:
    """Read Lexicon.csv and seed records into the PostgreSQL database."""
    target_path = Path(csv_path or settings.LEXICON_CSV_PATH)
    if not target_path.exists():
        raise FileNotFoundError(f"Lexicon CSV not found at: {target_path}")

    print(f"Connecting to database at {settings.DATABASE_URL}...")
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        if truncate:
            print("Truncating existing lexicons table...")
            session.query(Lexicon).delete()
            session.commit()

        # Check existing count
        existing_count = session.query(Lexicon).count()
        if existing_count > 0 and not truncate:
            print(f"Table 'lexicons' already has {existing_count} records. Seeding will skip duplicates.")
            existing_terms = set(
                row[0] for row in session.query(Lexicon.term).all()
            )
        else:
            existing_terms = set()

        records_to_insert = []
        with open(target_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row.get("Term", "").strip()
                if not term:
                    continue

                if term in existing_terms:
                    continue

                lex = Lexicon(
                    term=term,
                    code=row.get("Code", "").strip() or None,
                    culture=row.get("Culture", "").strip() or None,
                    sub_culture=row.get("Sub-culture", "").strip() or None,
                    word_class=row.get("Word class", "").strip() or None,
                    language=row.get("Language", "").strip() or None,
                    dialect=row.get("Dialect", "").strip() or None,
                    semantic_domain=row.get("Semantic domain", "").strip() or None,
                    usage_context=row.get("Usage context", "").strip() or None,
                    register=row.get("Register", "").strip() or None,
                    phonetic=row.get("Phonetic", "").strip() or None,
                    definition=row.get("Definition", "").strip() or None,
                    etymology=row.get("Etymology", "").strip() or None,
                    variants=row.get("Variants", "").strip() or None,
                    examples=row.get("Examples", "").strip() or None,
                    reference=row.get("Reference", "").strip() or None,
                    reference_url=row.get("Reference URL", "").strip() or None,
                    categories=row.get("Categories", "").strip() or None,
                    contributors=row.get("Contributors", "").strip() or None,
                    images_count=parse_int_safe(row.get("Images")),
                    audio_count=parse_int_safe(row.get("Audio")),
                    video_count=parse_int_safe(row.get("Video")),
                    three_d_models_count=parse_int_safe(row.get("3D models")),
                    status=row.get("Status", "").strip() or None,
                    verification=row.get("Verification", "").strip() or None,
                    verified_by=row.get("Verified by", "").strip() or None,
                    verified_at=parse_datetime_safe(row.get("Verified at")),
                    created_at=parse_datetime_safe(row.get("Created")),
                    updated_at=parse_datetime_safe(row.get("Updated")),
                )
                records_to_insert.append(lex)
                existing_terms.add(term)

        if records_to_insert:
            session.bulk_save_objects(records_to_insert)
            session.commit()
            print(f"Successfully seeded {len(records_to_insert)} new lexicon records into PostgreSQL.")
        else:
            print("No new records to insert.")

        total = session.query(Lexicon).count()
        print(f"Total lexicon records in database: {total}")
        return len(records_to_insert)

    except Exception as e:
        session.rollback()
        print(f"Error during seeding: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    truncate_flag = "--truncate" in sys.argv
    seed_database(truncate=truncate_flag)
