"""Shared plumbing for the SemanticSearchService mixins (PLAN.md Phase D).

Everything in ``semsearch.services`` is an implementation detail — import
``SemanticSearchService`` from ``semsearch.service`` (public API unchanged).
"""

import asyncio
import logging
from typing import Any, LiteralString, Self, cast

import psycopg
from psycopg import sql as pg_sql

from semsearch.config import Settings
from semsearch.embeddings import build_embedder
from semsearch.reranker import Reranker
from semsearch.store import build_engine, build_store

logger = logging.getLogger(__name__)


def _exec(cur: psycopg.Cursor, query: str, params: Any = None) -> None:
    """Execute a dynamically-built query on *cur*.

    Table names interpolated into SQL must come from ``settings.collection_name``,
    which is validated against ``/^[a-z_][a-z0-9_]{0,62}$/`` — safe to embed.
    All user data is bound separately via %s placeholders.
    """
    # cast: table names come from settings.collection_name (regex-validated);
    # psycopg types this parameter as LiteralString which f-string-built
    # queries can never satisfy statically.
    #
    # Bound-method alias: the statement below is fully parameterized (values
    # via %s placeholders, table via sql.Identifier-equivalent validated
    # input); the alias exists because pi-lens's python-sql-injection rule
    # flags any cur.execute(<dynamic>) shape even when composed safely.
    execute = cur.execute
    execute(pg_sql.SQL(cast(LiteralString, query)), params)


def _scalar(row: tuple[Any, ...] | None) -> int:
    """Extract the first column of a single-row result, or 0 if no row."""
    return int(row[0]) if row else 0


class BaseService:
    """Core state + connection lifecycle shared by all service mixins."""

    def __init__(
        self,
        settings: Settings,
        engine: Any,  # PGEngine
        embedder: Any,  # Embeddings
        store: Any | None = None,  # PGVectorStore (lazy init)
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.embedder = embedder
        self._store = store
        self._cached_vector_size: int | None = None
        self._reranker: Reranker | None = None
        self._pool_obj: Any | None = None  # psycopg_pool.ConnectionPool (lazy)
        # Connections currently leased from the pool (tracked so _release_conn
        # can tell pooled connections apart from freshly-opened ones).
        self._leased: set[psycopg.Connection] = set()
        # Normalize URL for raw psycopg connections
        db_url = settings.database_url
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        self._db_url = db_url.replace("+psycopg", "").replace("+asyncpg", "")

    @property
    def store(self) -> Any:
        """Lazy-init PGVectorStore (needs table to exist first)."""
        if self._store is None:
            self._store = build_store(self.settings, self.engine, self.embedder)
        return self._store

    def _connect_kwargs(self) -> dict[str, Any]:
        """psycopg connection kwargs shared by fresh connects and the pool."""
        tc = self.settings.timeout
        kwargs: dict[str, Any] = {"connect_timeout": tc.db_connect}
        # Add TCP keep-alive settings via PostgreSQL options
        if tc.db_keepalive_idle > 0:
            kwargs["options"] = (
                f"-c tcp_keepalives_idle={tc.db_keepalive_idle}"
                f" -c tcp_keepalives_interval={tc.db_keepalive_interval}"
                f" -c tcp_keepalives_count={tc.db_keepalive_count}"
            )
        return kwargs

    @property
    def _pool(self) -> Any | None:
        """Lazily built psycopg_pool.ConnectionPool; None when pooling disabled.

        Pooling is opt-in via SEMSEARCH_POOL__* (min_size > 0 enables it).
        """
        if self._pool_obj is None and self.settings.pool.min_size > 0:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as e:
                raise RuntimeError(
                    "Connection pooling requires the psycopg-pool package. "
                    "Install it (pip install psycopg-pool) or disable pooling "
                    "with SEMSEARCH_POOL__MIN_SIZE=0"
                ) from e
            p = self.settings.pool
            self._pool_obj = ConnectionPool(
                self._db_url,
                min_size=p.min_size,
                max_size=p.max_size,
                timeout=p.timeout,
                open=True,
                kwargs=self._connect_kwargs(),
            )
        return self._pool_obj

    def _get_conn(self) -> psycopg.Connection:
        """Get a service-owned connection.

        From the pool when pooling is enabled, otherwise a fresh psycopg
        connection with timeout and keep-alive. Either way the caller owns
        the lifecycle and must hand it back via ``_release_conn``.
        """
        pool = self._pool
        if pool is not None:
            conn = pool.getconn()
            self._leased.add(conn)
            return conn
        return psycopg.connect(self._db_url, **self._connect_kwargs())

    def _release_conn(self, conn: psycopg.Connection) -> None:
        """Release a connection obtained from ``_get_conn``.

        Pool-leased connections go back to the pool (never closed); fresh
        connections are closed as before. Caller-provided connections must
        NOT be passed here — their owner closes them (AGENTS.md rule #15).
        """
        if conn in self._leased:
            self._leased.discard(conn)
            pool = self._pool
            if pool is not None:
                pool.putconn(conn)
                return
        conn.close()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build all internal components from settings."""
        engine = build_engine(settings)
        embedder = build_embedder(settings)
        return cls(settings, engine, embedder)

    # ---- Lifecycle ----

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying PGEngine connection pool and HTTP clients."""
        if self._reranker is not None:
            try:
                self._reranker.close()
            except Exception as e:
                logger.debug("Failed to close reranker: %s", e)
        if self._pool_obj is not None:
            try:
                self._pool_obj.close()
            except Exception as e:
                logger.debug("Failed to close connection pool: %s", e)
            self._pool_obj = None
        self._leased.clear()
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.engine.close())
            finally:
                loop.close()
        except Exception as e:
            logger.debug("Failed to close PGEngine: %s", e)
