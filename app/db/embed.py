import sys
from app.db.session import SessionLocal, init_db_extensions
from app.services.semantic_service import semantic_service


def main():
    force_flag = "--force" in sys.argv
    print("Initializing database vector extensions...")
    init_db_extensions()

    session = SessionLocal()
    try:
        print("Generating embeddings using BAAI/bge-m3...")
        encoded, total, ms = semantic_service.generate_embeddings(session, force=force_flag)
        print(
            f"Successfully generated embeddings for {encoded} records in {ms:.2f}ms."
        )
        print(f"Total records with vector embeddings in PostgreSQL: {total}")
    except Exception as e:
        print(f"Error during embedding generation: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
