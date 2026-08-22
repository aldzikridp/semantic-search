"""Admin mixin: schema init, dimension probing, delete, and stats.

Part of ``semsearch.services`` (PLAN.md Phase D) — implementation detail of
``semsearch.service.SemanticSearchService``.
"""

import logging
from typing import TYPE_CHECKING, Any

import psycopg

from semsearch.errors import DeleteError
from semsearch.models import DeleteResult
from semsearch.store import init_schema

from .base import BaseService, _exec, _scalar

if TYPE_CHECKING:
    from semsearch.reranker import Reranker

logger = logging.getLogger(__name__)


class AdminMixin(BaseService):
    """Schema lifecycle, dimension probing, filtered delete, and stats."""

    if TYPE_CHECKING:
        # Cross-mixin dependency: ``reranker`` is provided by SearchMixin in
        # the composed SemanticSearchService facade. Declared here (types
        # only, erased at runtime) so warmup() type-checks standalone.
        @property
        def reranker(self) -> "Reranker | None": ...

    # ---- Schema ----

    def init_schema(self, *, recreate: bool = False) -> None:
        """Idempotently create the chunks table with the active provider's dim."""
        vector_size = self._get_vector_size()
        init_schema(self.settings, self.engine, vector_size, recreate=recreate)

    def _get_vector_size(self, *, force_probe: bool = False) -> int:
        """Return the embedding dimension. Cached after first probe.

        Args:
            force_probe: If True, ignore the cache and re-embed the probe string.
        """
        if not force_probe and self._cached_vector_size is not None:
            return self._cached_vector_size
        size = len(self.embedder.embed_query("dimension probe"))
        self._cached_vector_size = size
        return size

    def _get_vector_size_from_db(
        self, conn: psycopg.Connection | None = None
    ) -> int | None:
        """Read the vector dimension straight from pg_attribute — no API call.

        Args:
            conn: Optional psycopg connection to reuse. If None, a new
                connection is created and closed internally.
        """
        table = self.settings.collection_name
        owns_conn = conn is None
        if conn is None:
            conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT (regexp_match(format_type(atttypid, atttypmod), "
                    "'\\((\\d+)\\)'))[1]::int "
                    "FROM pg_attribute "
                    "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                    (table,),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            # Table doesn't exist or other DB error — fall back to probe.
            logger.debug("_get_vector_size_from_db failed: %s", e)
            return None
        finally:
            if owns_conn:
                self._release_conn(conn)

    # ---- Delete ----

    def delete(
        self, filter: dict[str, Any], conn: psycopg.Connection | None = None
    ) -> DeleteResult:
        """Delete every chunk matching the filter.

        Args:
            filter: Dict of key-value pairs to match. Empty dict deletes all.
            conn: Optional psycopg connection to reuse. If None, a new
                connection is created and closed internally.

        Returns:
            DeleteResult with the count of deleted chunks.
        """
        table = self.settings.collection_name
        owns_conn = conn is None
        if conn is None:
            conn = self._get_conn()

        try:
            with conn.cursor() as cur:
                if not filter:
                    _exec(cur, f"SELECT COUNT(*) FROM {table}")
                    count = _scalar(cur.fetchone())
                    _exec(cur, f"DELETE FROM {table}")
                else:
                    # Build WHERE clause from filter dict.
                    # 'source' is a top-level column; other keys live inside
                    # langchain_metadata JSON.
                    conditions: list[str] = []
                    params: list[Any] = []
                    for key, value in filter.items():
                        if key == "source":
                            conditions.append("source = %s")
                        else:
                            conditions.append(f"langchain_metadata->>'{key}' = %s")
                        params.append(str(value))
                    where_clause = " AND ".join(conditions)

                    _exec(
                        cur,
                        f"SELECT COUNT(*) FROM {table} WHERE {where_clause}",
                        params,
                    )
                    count = _scalar(cur.fetchone())
                    _exec(
                        cur,
                        f"DELETE FROM {table} WHERE {where_clause}",
                        params,
                    )

                conn.commit()

            return DeleteResult(deleted_count=count, filter=filter)

        except Exception as exc:
            conn.rollback()
            raise DeleteError(f"Delete failed: {exc}") from exc
        finally:
            if owns_conn:
                self._release_conn(conn)

    # ---- Warmup ----

    def warmup(self) -> dict[str, bool]:
        """Pre-build lazily-initialized local resources.

        Safe at server startup: touches ONLY local resources (PGVectorStore,
        one DB round-trip, reranker client construction, SQLAlchemy pool
        pre-fill). Never contacts the embedding or reranker APIs. Fail-open:
        logs and continues on error so startup never depends on DB state or
        providers (PLAN.md Phases W and F).
        """
        result = {"store": False, "db": False, "reranker": False, "pool_prefill": False}
        try:
            _ = self.store  # build PGVectorStore
            result["store"] = True
        except Exception as e:
            logger.warning("Warmup: store init deferred (%s)", e)
        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                result["db"] = True
            finally:
                self._release_conn(conn)
        except Exception as e:
            logger.warning("Warmup: DB round-trip deferred (%s)", e)
        if self.settings.reranker is not None:
            try:
                _ = self.reranker  # construct client (no request sent)
                result["reranker"] = True
            except Exception as e:
                logger.warning("Warmup: reranker init deferred (%s)", e)
        # Pool pre-fill (PLAN.md Phase F): open up to pool_size connections
        # through PGEngine's underlying SQLAlchemy engine and return them, so
        # the first concurrent burst doesn't serialize on connection setup.
        # Must run on PGEngine's own event loop (via _run_as_sync): asyncpg
        # connections are loop-affine, and sync_engine IO outside greenlet
        # context fails. Fail-open like every other step.
        try:
            async_engine = getattr(self.engine, "_pool", None)  # AsyncEngine (private)
            if async_engine is not None:
                target = min(async_engine.sync_engine.pool.size(), 5)

                async def _prefill() -> None:
                    conns = []
                    try:
                        for _ in range(target):
                            sa_conn = await async_engine.connect()
                            await sa_conn.exec_driver_sql("SELECT 1")
                            conns.append(sa_conn)
                    finally:
                        for sa_conn in conns:
                            await sa_conn.close()  # returned to the pool, not closed

                self.engine._run_as_sync(_prefill())
                result["pool_prefill"] = True
        except Exception as e:
            logger.warning("Warmup: pool pre-fill deferred (%s)", e)
        return result

    # ---- Stats ----

    def stats(self, conn: psycopg.Connection | None = None) -> dict[str, Any]:
        """Return statistics about the chunks table.

        Args:
            conn: Optional psycopg connection to reuse. If None, a new
                connection is created and closed internally.
        """
        table = self.settings.collection_name
        owns_conn = conn is None
        if conn is None:
            conn = self._get_conn()

        try:
            with conn.cursor() as cur:
                _exec(cur, f"SELECT COUNT(*) FROM {table}")
                chunk_count = _scalar(cur.fetchone())

                _exec(cur, f"SELECT COUNT(DISTINCT source) FROM {table}")
                source_count = _scalar(cur.fetchone())

                _exec(
                    cur,
                    f"SELECT source, COUNT(*) AS cnt "
                    f"FROM {table} "
                    f"GROUP BY source "
                    f"ORDER BY cnt DESC "
                    f"LIMIT 20",
                )
                sources_by_count = [(row[0], row[1]) for row in cur.fetchall()]

            # Read vector dim from DB while connection is still open
            embedding_dim = self._get_vector_size_from_db(conn) or self._get_vector_size()
        finally:
            if owns_conn:
                self._release_conn(conn)

        return {
            "table": table,
            "embedding_provider": self.settings.embedding_provider.type,
            "embedding_dim": embedding_dim,
            "chunk_count": chunk_count,
            "source_count": source_count,
            "sources_by_count": sources_by_count,
        }
