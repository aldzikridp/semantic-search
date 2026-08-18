"""PGEngine/PGVectorStore construction and schema initialization (spec §7.5)."""

from __future__ import annotations

import psycopg
from sqlalchemy import create_engine, text

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from semsearch.config import Settings
from semsearch.errors import SchemaMismatchError


# Custom top-level columns (REAL Postgres columns, not JSONB keys).
CUSTOM_COLUMNS = [
    Column("source", "TEXT", nullable=False),
    Column("chunk_index", "INTEGER", nullable=False),
    Column("document_hash", "CHAR(64)"),
]

# List of column NAMES (strings) — what PGVectorStore.create_sync expects.
CUSTOM_COLUMN_NAMES = [c.name for c in CUSTOM_COLUMNS]

# Explicit column name overrides.
LC_CONTENT_COLUMN = "content"
LC_METADATA_COLUMN = "langchain_metadata"
LC_ID_COLUMN = "langchain_id"


def build_engine(settings: Settings) -> PGEngine:
    """Construct a PGEngine bound to settings.database_url."""
    # PGEngine uses asyncpg internally; convert psycopg URL if needed
    url = settings.database_url
    if "+psycopg" in url:
        url = url.replace("+psycopg", "+asyncpg")
    return PGEngine.from_connection_string(url=url)


def build_store(
    settings: Settings,
    engine: PGEngine,
    embedder: Embeddings,
) -> PGVectorStore:
    """Construct a PGVectorStore bound to settings.collection_name."""
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

    Args:
        settings: App settings (reads collection_name).
        engine: PGEngine instance.
        vector_size: Dimension of the active embedding provider.
        recreate: If True, DROP TABLE before re-creating.

    Raises:
        SchemaMismatchError: If the table exists but its vector_size doesn't match.
    """
    table = settings.collection_name
    db_url = settings.database_url

    # Use raw psycopg for schema operations
    conn = psycopg.connect(db_url.replace("+psycopg", ""))
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            # Step 1: Check if table exists
            cur.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_name = %s"
                ")",
                (table,),
            )
            table_exists = cur.fetchone()[0]

            if table_exists and recreate:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                table_exists = False

            elif table_exists:
                # Check vector dimension matches
                cur.execute(
                    "SELECT atttypmod - 4 AS dim "
                    "FROM pg_attribute "
                    "WHERE attrelid = %s::regclass "
                    "AND attname = 'embedding'",
                    (table,),
                )
                row = cur.fetchone()
                existing_dim = row[0] if row else None
                if existing_dim is not None and existing_dim != vector_size:
                    raise SchemaMismatchError(
                        f"Table '{table}' has vector({existing_dim}) but the active "
                        f"provider produces vector({vector_size}). "
                        f"Use recreate=True to drop and re-create."
                    )

    finally:
        conn.close()

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
        conn = psycopg.connect(db_url.replace("+psycopg", ""))
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
                    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = 16, ef_construction = 64)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx "
                    f"ON {table} USING gin ((langchain_metadata::jsonb) jsonb_path_ops)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_source_chunk_idx "
                    f"ON {table} (source, chunk_index)"
                )
                cur.execute(
                    f"ALTER TABLE {table} "
                    f"ADD CONSTRAINT {table}_source_chunk_unique "
                    f"UNIQUE (source, chunk_index)"
                )
        finally:
            conn.close()
