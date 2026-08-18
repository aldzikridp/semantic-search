"""PGEngine/PGVectorStore construction and schema initialization (spec §7.5)."""

from __future__ import annotations

from sqlalchemy import text

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from semsearch.config import Settings
from semsearch.errors import SchemaMismatchError


# Custom top-level columns (REAL Postgres columns, not JSONB keys).
# Defined here so both init_vectorstore_table and PGVectorStore.create_sync
# reference the same definitions.
CUSTOM_COLUMNS = [
    Column("source", "TEXT", nullable=False),
    Column("chunk_index", "INTEGER", nullable=False),
    Column("document_hash", "CHAR(64)"),
]

# List of column NAMES (strings) — what PGVectorStore.create_sync expects.
CUSTOM_COLUMN_NAMES = [c.name for c in CUSTOM_COLUMNS]

# Explicit column name overrides.
# LC_CONTENT_COLUMN: PGVectorStore's actual default is "page_content"; we override.
# LC_METADATA_COLUMN: this IS LangChain's default name.
# LC_ID_COLUMN: this IS the default name; we override its TYPE from UUID to TEXT.
LC_CONTENT_COLUMN = "content"
LC_METADATA_COLUMN = "langchain_metadata"
LC_ID_COLUMN = "langchain_id"


def build_engine(settings: Settings) -> PGEngine:
    """Construct a PGEngine bound to settings.database_url.

    The PGEngine holds a connection pool shared across all PGVectorStore
    instances derived from it. Construct ONCE per process; pass to build_store().
    """
    return PGEngine.from_connection_string(url=settings.database_url)


def build_store(
    settings: Settings,
    engine: PGEngine,
    embedder: Embeddings,
) -> PGVectorStore:
    """Construct a PGVectorStore bound to settings.collection_name.

    Uses PGVectorStore.create_sync (NOT the deprecated PGVector constructor).

    Critical kwargs:
      - id_column="langchain_id" with TEXT type — we override the default UUID PK
        so we can pass deterministic string IDs (f"{source}::{chunk_index}").
      - content_column="content" — our override (PGVectorStore default is "page_content").
      - metadata_json_column="langchain_metadata" — this IS LangChain's default.
      - metadata_columns=CUSTOM_COLUMN_NAMES — list of strings (NOT Column objects).
    """
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.collection_name,
        embedding_service=embedder,
        id_column=LC_ID_COLUMN,
        content_column=LC_CONTENT_COLUMN,
        metadata_json_column=LC_METADATA_COLUMN,
        metadata_columns=CUSTOM_COLUMN_NAMES,
    )


def init_schema(
    settings: Settings,
    engine: PGEngine,
    vector_size: int,
    *,
    recreate: bool = False,
) -> None:
    """Idempotently create the chunks table with the right vector dim.

    Implementation notes (subtleties of langchain-postgres 0.0.17):
      1. `init_vectorstore_table` uses `CREATE TABLE` (NOT `CREATE TABLE IF NOT EXISTS`),
         so calling it twice raises an error. We check `information_schema.tables`
         first and skip if the table already exists (unless recreate=True).
      2. `init_vectorstore_table` does NOT create the UNIQUE constraint or HNSW/GIN
         indexes. We add them via explicit `CREATE [UNIQUE] INDEX IF NOT EXISTS`.
      3. `Column("langchain_id", "TEXT")` without `primary_key=True` — the installed
         Column dataclass doesn't support primary_key; the id column is implicitly
         PK by role.

    Args:
        settings: App settings (reads collection_name).
        engine: PGEngine instance.
        vector_size: Dimension of the active embedding provider.
        recreate: If True, DROP TABLE before re-creating.

    Raises:
        SchemaMismatchError: If the table exists but its vector_size doesn't match
                             the active provider and recreate=False.
    """
    table = settings.collection_name

    with engine._async_engine.begin() as conn:
        # Step 1: Check if table exists
        result = conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_name = :table_name"
                    ")"
                ),
                {"table_name": table},
            ).scalar()
        )
        table_exists = bool(result)

        if table_exists and recreate:
            # Step 2a: Drop table if recreate=True
            conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(f"DROP TABLE IF EXISTS {table} CASCADE")
                )
            )
            table_exists = False

        elif table_exists:
            # Step 2b: Check vector dimension matches
            existing_dim = conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(
                        "SELECT atttypmod - 4 AS dim "
                        "FROM pg_attribute "
                        "WHERE attrelid = :table::regclass "
                        "AND attname = 'embedding'"
                    ),
                    {"table": table},
                ).scalar()
            )
            if existing_dim is not None and existing_dim != vector_size:
                raise SchemaMismatchError(
                    f"Table '{table}' has vector({existing_dim}) but the active "
                    f"provider produces vector({vector_size}). "
                    f"Use recreate=True to drop and re-create."
                )

    if not table_exists:
        # Step 3: Create table via langchain-postgres
        engine.init_vectorstore_table(
            table_name=table,
            vector_size=vector_size,
            id_column=Column(LC_ID_COLUMN, "TEXT"),
            content_column=LC_CONTENT_COLUMN,
            metadata_json_column=LC_METADATA_COLUMN,
            metadata_columns=CUSTOM_COLUMNS,
        )

        # Step 4: Create indexes (init_vectorstore_table does NOT create them)
        with engine._async_engine.begin() as conn:
            # HNSW cosine similarity index
            conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
                        f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                        f"WITH (m = 16, ef_construction = 64)"
                    )
                )
            )

            # JSONB GIN index for filter performance
            conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx "
                        f"ON {table} USING gin (langchain_metadata jsonb_path_ops)"
                    )
                )
            )

            # Composite index for re-ingest lookup
            conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {table}_source_chunk_idx "
                        f"ON {table} (source, chunk_index)"
                    )
                )
            )

            # UNIQUE constraint for chunk identity
            conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    text(
                        f"ALTER TABLE {table} "
                        f"ADD CONSTRAINT {table}_source_chunk_unique "
                        f"UNIQUE (source, chunk_index)"
                    )
                )
            )
