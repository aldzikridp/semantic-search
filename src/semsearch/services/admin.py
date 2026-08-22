"""Admin mixin: schema init, dimension probing, delete, and stats.

Part of ``semsearch.services`` (PLAN.md Phase D) — implementation detail of
``semsearch.service.SemanticSearchService``.
"""

import logging
from typing import Any

import psycopg

from semsearch.errors import DeleteError
from semsearch.models import DeleteResult
from semsearch.store import init_schema

from .base import BaseService, _exec, _scalar

logger = logging.getLogger(__name__)


class AdminMixin(BaseService):
    """Schema lifecycle, dimension probing, filtered delete, and stats."""

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
