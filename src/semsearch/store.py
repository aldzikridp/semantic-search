"""PGEngine/PGVectorStore construction and schema initialization (spec §7.5)."""

import logging

import psycopg
from sqlalchemy import create_engine, text

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from semsearch.config import Settings
from semsearch.errors import SchemaMismatchError

logger = logging.getLogger(__name__)


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
    # Normalize URL for SQLAlchemy/asyncpg
    url = settings.database_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if "+psycopg" in url:
        url = url.replace("+psycopg", "+asyncpg")
    elif "+asyncpg" not in url and "+psycopg" not in url:
        # Add asyncpg driver if no driver specified
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Pool settings to prevent stale connections:
    # pool_recycle: recreate connections before server closes them
    # pool_pre_ping: test connection health before use
    # connect_args: TCP keep-alive settings to detect dead connections
    tc = settings.timeout
    connect_args = {}
    if tc.db_keepalive_idle > 0:
        connect_args["server_settings"] = {
            "tcp_keepalives_idle": str(tc.db_keepalive_idle),
            "tcp_keepalives_interval": str(tc.db_keepalive_interval),
            "tcp_keepalives_count": str(tc.db_keepalive_count),
        }
    return PGEngine.from_connection_string(
        url=url,
        pool_recycle=tc.db_pool_recycle,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )


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


def _has_vectorscale(cur: psycopg.Cursor) -> bool:
    """Check if pgvectorscale extension is installed in the database."""
    cur.execute(
        "SELECT EXISTS ("
        "  SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'"
        ")"
    )
    return cur.fetchone()[0]


def _index_type_for_vector(cur: psycopg.Cursor, table: str) -> str:
    """Return the vector index type on the table: 'diskann', 'hnsw', or 'none'."""
    cur.execute(
        "SELECT am.amname "
        "FROM pg_class c "
        "JOIN pg_index i ON c.oid = i.indexrelid "
        "JOIN pg_am am ON c.relam = am.oid "
        "WHERE i.indrelid = %s::regclass "
        "AND c.relname = %s || '_diskann_idx'",
        (table, table),
    )
    if cur.fetchone():
        return "diskann"
    cur.execute(
        "SELECT am.amname "
        "FROM pg_class c "
        "JOIN pg_index i ON c.oid = i.indexrelid "
        "JOIN pg_am am ON c.relam = am.oid "
        "WHERE i.indrelid = %s::regclass "
        "AND c.relname = %s || '_hnsw_idx'",
        (table, table),
    )
    if cur.fetchone():
        return "hnsw"
    return "none"


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
                # Use format_type() for reliable dim detection across pgvector versions.
                # atttypmod varies: some versions store N+4, others store N directly.
                cur.execute(
                    "SELECT (regexp_match(format_type(atttypid, atttypmod), '\\((\\d+)\\)'))[1]::int AS dim "
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

                # Auto-upgrade: HNSW → DiskANN if vectorscale is available
                index_type = _index_type_for_vector(cur, table)
                if index_type == "hnsw" and _has_vectorscale(cur):
                    logger.info("Upgrading HNSW index to DiskANN on %s", table)
                    cur.execute(f"DROP INDEX IF EXISTS {table}_hnsw_idx")

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

    # Step 4: Create/upgrade indexes (idempotent)
    # Runs for both new tables and existing tables (to create missing indexes
    # or upgrade HNSW → DiskANN).
    conn = psycopg.connect(db_url.replace("+psycopg", ""))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            has_vectorscale = _has_vectorscale(cur)
            index_type = _index_type_for_vector(cur, table)

            if has_vectorscale and index_type != "diskann":
                # Create DiskANN (new table or upgrade from HNSW)
                if index_type == "hnsw":
                    cur.execute(f"DROP INDEX IF EXISTS {table}_hnsw_idx")

                cfg = settings.diskann  # DiskANNConfig or None
                storage = cfg.storage_layout if cfg else "memory_optimized"
                bits = cfg.num_bits_per_dimension if cfg else 2
                num_neighbors = cfg.num_neighbors if cfg else 50
                search_list = cfg.search_list_size if cfg else 100
                alpha = cfg.max_alpha if cfg else 1.2
                dims = cfg.num_dimensions if cfg else 0

                # num_bits_per_dimension > 1 only valid for ≤900 dims
                if bits > 1 and vector_size > 900:
                    bits = 1

                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_diskann_idx "
                    f"ON {table} USING diskann (embedding vector_cosine_ops) "
                    f"WITH ("
                    f"  storage_layout = '{storage}',"
                    f"  num_bits_per_dimension = {bits},"
                    f"  num_neighbors = {num_neighbors},"
                    f"  search_list_size = {search_list},"
                    f"  max_alpha = {alpha},"
                    f"  num_dimensions = {dims}"
                    f")"
                )
                logger.info(
                    "Created DiskANN index on %s (dims=%d, storage=%s, bits=%d)",
                    table, vector_size, storage, bits,
                )

            elif not has_vectorscale and index_type == "none" and vector_size <= 2000:
                # HNSW fallback when vectorscale is not installed
                hnsw = settings.hnsw
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx "
                    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {hnsw.m}, ef_construction = {hnsw.ef_construction})"
                )

            # Set table-level ef_search default (idempotent, works for existing indexes too)
            if not has_vectorscale:
                hnsw = settings.hnsw
                try:
                    cur.execute(
                        f"ALTER TABLE {table} SET (hnsw.ef_search = {hnsw.ef_search})"
                    )
                except Exception as e:
                    # ef_search not supported by this pgvector version — safe to ignore
                    logger.debug("Could not set hnsw.ef_search on %s: %s", table, e)

            # Non-vector indexes (always created, idempotent)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx "
                f"ON {table} USING gin ((langchain_metadata::jsonb) jsonb_path_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_source_chunk_idx "
                f"ON {table} (source, chunk_index)"
            )
            # UNIQUE constraint — use IF NOT EXISTS equivalent
            cur.execute(
                f"DO $$ "
                f"BEGIN "
                f"  IF NOT EXISTS ("
                f"    SELECT 1 FROM pg_constraint "
                f"    WHERE conname = '{table}_source_chunk_unique'"
                f"  ) THEN "
                f"    ALTER TABLE {table} "
                f"    ADD CONSTRAINT {table}_source_chunk_unique "
                f"    UNIQUE (source, chunk_index); "
                f"  END IF; "
                f"END $$;"
            )
    finally:
        conn.close()
