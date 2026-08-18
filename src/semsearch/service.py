"""Core service — SemanticSearchService facade (spec §7.6)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy import text

from semsearch.config import Settings
from semsearch.embeddings import build_embedder
from semsearch.store import build_engine, build_store, init_schema


class SemanticSearchService:
    """High-level facade over loader + splitter + embedder + PGVectorStore.

    Lifecycle::

        with SemanticSearchService.from_settings(settings) as svc:
            svc.init_schema()
            svc.ingest(...)
            svc.search(...)
            svc.delete(filter={...})
    """

    def __init__(
        self,
        settings: Settings,
        engine: Any,  # PGEngine
        embedder: Any,  # Embeddings
        store: Any,  # PGVectorStore
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_settings(cls, settings: Settings) -> SemanticSearchService:
        """Build all internal components from settings.

        Does NOT call init_schema — caller must explicitly invoke
        ``svc.init_schema()`` if the table doesn't exist yet.
        """
        engine = build_engine(settings)
        embedder = build_embedder(settings)
        store = build_store(settings, engine, embedder)
        return cls(settings, engine, embedder, store)

    # ---- Lifecycle ----

    def __enter__(self) -> SemanticSearchService:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying PGEngine connection pool.

        PGEngine.close() is async — it disposes an underlying async
        SQLAlchemy pool. For v1 (sync API), we submit the coroutine
        and wait for completion.
        """
        try:
            loop = self.engine._async_engine.pool._loop
            asyncio.run_coroutine_threadsafe(
                self.engine._async_engine.dispose(), loop
            ).result(timeout=10)
        except Exception:
            # Fallback: if the above fails, try asyncio.run
            try:
                asyncio.run(self.engine._async_engine.dispose())
            except Exception:
                pass  # Best-effort cleanup

    # ---- Schema ----

    def init_schema(self, *, recreate: bool = False) -> None:
        """Idempotently create the chunks table with the active provider's dim.

        Args:
            recreate: If True, DROP TABLE before re-creating.
        """
        vector_size = self._get_vector_size()
        init_schema(self.settings, self.engine, vector_size, recreate=recreate)

    def _get_vector_size(self) -> int:
        """Embed a dummy query to determine the vector dimension.

        This is a one-time cost on service initialization.
        """
        return len(self.embedder.embed_query("dimension probe"))

    # ---- Stats ----

    def stats(self) -> dict[str, Any]:
        """Return statistics about the chunks table.

        Returns:
            Dict with keys: table, embedding_provider, embedding_dim,
            chunk_count, source_count, sources_by_count (top 20).
        """
        table = self.settings.collection_name

        with self.engine.begin() as conn:
            # Chunk count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            chunk_count = result.scalar_one()

            # Distinct source count
            result = conn.execute(
                text(f"SELECT COUNT(DISTINCT source) FROM {table}")
            )
            source_count = result.scalar_one()

            # Top 20 sources by chunk count
            result = conn.execute(
                text(
                    f"SELECT source, COUNT(*) AS cnt "
                    f"FROM {table} "
                    f"GROUP BY source "
                    f"ORDER BY cnt DESC "
                    f"LIMIT 20"
                )
            )
            sources_by_count = [(row[0], row[1]) for row in result.fetchall()]

        return {
            "table": table,
            "embedding_provider": self.settings.embedding_provider.type,
            "embedding_dim": self._get_vector_size(),
            "chunk_count": chunk_count,
            "source_count": source_count,
            "sources_by_count": sources_by_count,
        }
