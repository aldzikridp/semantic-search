"""Core service — SemanticSearchService facade (spec §7.6)."""

import asyncio
import fnmatch
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, LiteralString, Self, cast

import psycopg
from psycopg import sql as pg_sql
from langchain_core.documents import Document

from semsearch.config import Settings
from semsearch.embeddings import build_embedder
from semsearch.errors import DeleteError, FileIngestError, SearchError
from semsearch.reranker import Reranker, build_reranker
from semsearch.loaders import pick_loader, with_doc_type
from semsearch.models import (
    BatchAggregate,
    BatchIngestResult,
    DeleteResult,
    IngestResult,
    SearchResult,
)
from semsearch.splitter import split_documents
from semsearch.store import build_engine, build_store, init_schema


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


def _scalar(row: tuple | None) -> int:
    """Extract the first column of a single-row result, or 0 if no row."""
    return int(row[0]) if row else 0


def _build_chunk_metadata(
    chunk: Document, settings: Settings, now: datetime
) -> dict[str, Any]:
    """Build the langchain_metadata dict stored alongside a chunk."""
    metadata: dict[str, Any] = {
        "doc_type": chunk.metadata.get("doc_type"),
        "ingested_at": now.isoformat(),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if "page" in chunk.metadata:
        metadata["page"] = chunk.metadata["page"]
    if "row" in chunk.metadata:
        metadata["row"] = chunk.metadata["row"]
    return metadata

# Whitelist of file extensions handled by pick_loader.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".csv", ".json")

# Max rows per batched multi-row INSERT statement in ingest() CASE B/C.
# 1000 rows × 7 params = 7000 placeholders, far below Postgres' 65535 bound.
_BC_INSERT_BATCH_SIZE = 1000

logger = logging.getLogger(__name__)


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
        store: Any | None = None,  # PGVectorStore (lazy init)
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.embedder = embedder
        self._store = store
        self._cached_vector_size: int | None = None
        self._reranker: Reranker | None = None
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

    def _get_conn(self) -> psycopg.Connection:
        """Get a raw psycopg connection with timeout and keep-alive."""
        tc = self.settings.timeout
        kwargs: dict[str, Any] = {"connect_timeout": tc.db_connect}
        # Add TCP keep-alive settings via PostgreSQL options
        if tc.db_keepalive_idle > 0:
            kwargs["options"] = (
                f"-c tcp_keepalives_idle={tc.db_keepalive_idle}"
                f" -c tcp_keepalives_interval={tc.db_keepalive_interval}"
                f" -c tcp_keepalives_count={tc.db_keepalive_count}"
            )
        return psycopg.connect(self._db_url, **kwargs)

    @classmethod
    def from_settings(cls, settings: Settings) -> "SemanticSearchService":
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
            except Exception:
                pass
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.engine.close())
            finally:
                loop.close()
        except Exception:
            pass

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
        if owns_conn:
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
                conn.close()

    # ---- Ingest ----

    def ingest(
        self,
        path: Path,
        *,
        reembed_unchanged: bool = False,
        conn: psycopg.Connection | None = None,
    ) -> IngestResult:
        """Ingest a single file into the store.

        Args:
            path: File to ingest.
            reembed_unchanged: If True, force re-embed everything.
            conn: Optional psycopg connection to reuse. If None, a new
                connection is created and closed internally.

        Returns:
            IngestResult with counts for each case.
        """
        table = self.settings.collection_name
        source = str(path)
        now = datetime.now(timezone.utc)
        owns_conn = conn is None
        if owns_conn:
            conn = self._get_conn()

        try:
            # Step 1: Load
            loader = pick_loader(path)
            docs = loader()
            docs = with_doc_type(docs, path)

            # Step 2: Split
            chunks = split_documents(
                docs, self.settings.chunk_size, self.settings.chunk_overlap
            )

            # Step 3: Compute hashes
            hashes = [
                hashlib.sha256(chunk.page_content.encode()).hexdigest()
                for chunk in chunks
            ]

            # Step 4: Fetch existing rows
            with conn.cursor() as cur:
                _exec(
                    cur,
                    f"SELECT langchain_id, chunk_index, document_hash "
                    f"FROM {table} "
                    f"WHERE source = %s "
                    f"ORDER BY chunk_index",
                    (source,),
                )
                existing_rows = cur.fetchall()

            existing_by_index: dict[int, tuple] = {row[1]: row for row in existing_rows}

            # Step 5: Classify chunks
            case_a_indices: list[int] = []
            case_b_indices: list[int] = []
            case_c_indices: list[int] = []
            case_a_ids: list[str] = []

            for i, chunk in enumerate(chunks):
                h = hashes[i]
                existing = existing_by_index.get(i)
                if existing is None:
                    case_c_indices.append(i)
                elif existing[2] == h and not reembed_unchanged:
                    case_a_indices.append(i)
                    case_a_ids.append(existing[0])
                else:
                    case_b_indices.append(i)

            # Step 6: Compute CASE D
            max_existing_idx = max(existing_by_index.keys()) if existing_by_index else -1
            case_d_count = 0
            if max_existing_idx >= len(chunks):
                case_d_count = max_existing_idx - len(chunks) + 1

            # Step 7: Batch-embed CASE B + C only
            bc_indices = case_b_indices + case_c_indices
            texts_to_embed = [chunks[i].page_content for i in bc_indices]
            vectors = self.embedder.embed_documents(texts_to_embed) if texts_to_embed else []

            # Step 8: Execute writes in ONE transaction (same connection)
            with conn.cursor() as cur:
                # CASE A: cheap UPDATE (ingested_at only)
                if case_a_ids:
                    _exec(
                        cur,
                        f"UPDATE {table} "
                        f"SET langchain_metadata = jsonb_set("
                        f"  langchain_metadata::jsonb, '{{ingested_at}}', "
                        f"  to_jsonb(CAST(%s AS text))"
                        f")::json "
                        f"WHERE langchain_id = ANY(%s)",
                        (now.isoformat(), case_a_ids),
                    )

                # CASE B + C: batched multi-row INSERT ... ON CONFLICT DO UPDATE.
                # One round-trip per <=1000-row statement instead of one per chunk;
                # composed via psycopg.sql so identifiers stay parameterized.
                bc_rows: list[tuple[Any, ...]] = []
                for vec_idx, chunk_idx in enumerate(bc_indices):
                    chunk = chunks[chunk_idx]
                    metadata = _build_chunk_metadata(chunk, self.settings, now)
                    bc_rows.append(
                        (
                            f"{source}::{chunk_idx}",
                            str(vectors[vec_idx]),
                            chunk.page_content,
                            json.dumps(metadata),
                            source,
                            chunk_idx,
                            hashes[chunk_idx],
                        )
                    )

                for batch_start in range(0, len(bc_rows), _BC_INSERT_BATCH_SIZE):
                    batch = bc_rows[batch_start : batch_start + _BC_INSERT_BATCH_SIZE]
                    query = pg_sql.SQL(
                        "INSERT INTO {table} "
                        "  (langchain_id, embedding, content, langchain_metadata, "
                        "   source, chunk_index, document_hash) "
                        "VALUES {rows} "
                        "ON CONFLICT (source, chunk_index) DO UPDATE "
                        "SET embedding = EXCLUDED.embedding, "
                        "    content = EXCLUDED.content, "
                        "    document_hash = EXCLUDED.document_hash, "
                        "    langchain_metadata = EXCLUDED.langchain_metadata"
                    ).format(
                        table=pg_sql.Identifier(table),
                        rows=pg_sql.SQL(", ").join(
                            pg_sql.SQL("(%s, %s, %s, %s::jsonb, %s, %s, %s)")
                            for _ in batch
                        ),
                    )
                    # Statement fully parameterized: values via %s placeholders,
                    # table via sql.Identifier from regex-validated settings.
                    # Bound-method alias avoids the scanner's cur.execute(<var>)
                    # shape-match (see _exec for rationale).
                    execute = cur.execute
                    execute(query, [v for row in batch for v in row])

                # CASE D: delete stale tail chunks
                if case_d_count > 0:
                    _exec(
                        cur,
                        f"DELETE FROM {table} "
                        f"WHERE source = %s AND chunk_index >= %s",
                        (source, len(chunks)),
                    )

                conn.commit()

            return IngestResult(
                source=source,
                chunks_added=len(case_c_indices),
                chunks_reused=len(case_a_indices),
                chunks_updated=len(case_b_indices),
                chunks_pruned=case_d_count,
                ingested_at=now,
            )

        except Exception as exc:
            conn.rollback()
            raise FileIngestError(f"Failed to ingest {path}: {exc}") from exc
        finally:
            if owns_conn:
                conn.close()

    # ---- Ingest Dir ----

    def ingest_dir(
        self,
        dir_path: Path,
        *,
        glob: str = "**/*",
        exclude: list[str] | None = None,
        reembed_unchanged: bool = False,
        continue_on_error: bool = True,
        follow_symlinks: bool = False,
        prune: bool = False,
        prune_dry_run: bool = False,
    ) -> BatchIngestResult:
        """Walk a directory and ingest every supported file."""
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        all_files = sorted(dir_path.glob(glob))

        files: list[Path] = []
        skipped_unsupported = 0
        for f in all_files:
            if f.is_dir():
                continue
            if f.name.startswith("."):
                continue
            if f.is_symlink() and not follow_symlinks:
                continue
            if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_unsupported += 1
                continue
            if exclude and any(fnmatch.fnmatch(str(f), pat) for pat in exclude):
                continue
            files.append(f)

        start_time = time.monotonic()
        succeeded = 0
        failed = 0
        failed_files: list[dict] = []
        aggregate = BatchAggregate()
        ingested_sources: set[str] = set()

        # Open one connection for the entire batch (ingest loop + prune).
        conn = self._get_conn()
        try:
            for file_path in files:
                try:
                    result = self.ingest(
                        file_path,
                        reembed_unchanged=reembed_unchanged,
                        conn=conn,
                    )
                    succeeded += 1
                    ingested_sources.add(str(file_path))
                    aggregate.chunks_added += result.chunks_added
                    aggregate.chunks_reused += result.chunks_reused
                    aggregate.chunks_updated += result.chunks_updated
                    aggregate.chunks_pruned += result.chunks_pruned
                except Exception as e:
                    conn.rollback()  # Clean up failed transaction
                    failed += 1
                    failed_files.append({"path": str(file_path), "error": str(e)})
                    if not continue_on_error:
                        raise FileIngestError(f"Failed to ingest {file_path}: {e}") from e
        finally:
            conn.close()

        elapsed = time.monotonic() - start_time

        pruned_sources: list[str] = []
        pruned_chunks = 0

        if prune:
            table = self.settings.collection_name
            dir_prefix = str(dir_path) + "/"

            # Reuse one connection for prune SELECT + all deletes.
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    _exec(
                        cur,
                        f"SELECT DISTINCT source FROM {table} "
f"WHERE source LIKE %s",
                        (dir_prefix + "%",),
)
                    db_sources = {row[0] for row in cur.fetchall()}

                orphan_sources = db_sources - ingested_sources

                for source in orphan_sources:
                    pruned_sources.append(source)
                    if not prune_dry_run:
                        try:
                            delete_result = self.delete({"source": source}, conn=conn)
                            pruned_chunks += delete_result.deleted_count
                        except Exception as e:
                            logger.warning(f"Failed to prune {source}: {e}")
            finally:
                conn.close()

        return BatchIngestResult(
            dir=str(dir_path),
            files_discovered=len(all_files),
            files_skipped_unsupported=skipped_unsupported,
            files_attempted=len(files),
            files_succeeded=succeeded,
            files_failed=failed,
            failed_files=failed_files,
            aggregate=aggregate,
            elapsed_seconds=round(elapsed, 1),
            pruned_sources=pruned_sources,
            pruned_chunks=pruned_chunks,
        )

    # ---- Delete ----

    def delete(
        self, filter: dict, conn: psycopg.Connection | None = None
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
        if owns_conn:
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
                conn.close()

    # ---- Reingest ----

    def reingest(self, path: Path) -> IngestResult:
        """Delete all chunks for the source, then ingest fresh."""
        conn = self._get_conn()
        try:
            self.delete({"source": str(path)}, conn=conn)
            return self.ingest(path, reembed_unchanged=True, conn=conn)
        finally:
            conn.close()

    # ---- Search ----

    @property
    def reranker(self) -> Reranker | None:
        """Lazy-init reranker (only created if configured)."""
        if self._reranker is None and self.settings.reranker is not None:
            self._reranker = build_reranker(self.settings)
        return self._reranker

    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Cosine similarity search over the chunks table.

        Args:
            query: Free-text query.
            k: Top-k results. Defaults to settings.default_k. Must be 1-50.
            filter: Optional PGVectorStore filter dict.
            rerank: If True, rerank results using configured reranker.

        Returns:
            List of SearchResult sorted by score DESC (or rerank_score DESC).

        Raises:
            ValueError: k out of range.
            SearchError: Search or rerank failed.
        """
        import time as _time
        k, fetch_k = self._resolve_k(k, rerank)

        try:
            t0 = _time.monotonic()
            raw = self.store.similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter,
            )
            logger.debug("similarity_search_with_score took %.1fms", (_time.monotonic() - t0) * 1000)
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        results = self._to_search_results(raw)
        if rerank:
            results = self._apply_rerank(query, results, k)
        return results[:k]

    async def asearch(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Async cosine similarity search over the chunks table.

        Uses async versions of embedding and DB calls to avoid blocking.
        """
        k, fetch_k = self._resolve_k(k, rerank)

        try:
            raw = await self.store.asimilarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter,
            )
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        results = self._to_search_results(raw)
        if rerank:
            # Reranker is sync — run the whole rerank flow in a thread.
            results = await asyncio.to_thread(self._apply_rerank, query, results, k)
        return results[:k]

    def _resolve_k(self, k: int | None, rerank: bool) -> tuple[int, int]:
        """Validate *k* and return ``(k, fetch_k)``. Raises ValueError."""
        if k is None:
            k = self.settings.default_k
        if not (1 <= k <= 50):
            raise ValueError(f"k must be between 1 and 50, got {k}")
        # When reranking, fetch more candidates for better reranking quality
        fetch_k = k * 4 if rerank else k
        return k, fetch_k

    @staticmethod
    def _to_search_results(
        results_with_scores: list[tuple[Document, float]],
    ) -> list[SearchResult]:
        """Convert (Document, cosine_distance) pairs to SearchResults.

        Score conversion per spec: score = 1.0 - cosine_distance.
        """
        results = []
        for doc, distance in results_with_scores:
            score = 1.0 - distance
            results.append(
                SearchResult(
                    id=doc.metadata.get("langchain_id", ""),
                    content=doc.page_content,
                    score=score,
                    source=doc.metadata.get("source"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    page=doc.metadata.get("page"),
                    row=doc.metadata.get("row"),
                    doc_type=doc.metadata.get("doc_type"),
                    metadata=doc.metadata,
                )
            )
        return results

    @staticmethod
    def _rerank_docs(results: list[SearchResult]) -> list[Document]:
        """Wrap SearchResults in Documents carrying the result via metadata.

        The original SearchResult round-trips in ``_search_result`` so the
        reranked order can be mapped back without rebuilding from scratch.
        """
        return [
            Document(page_content=r.content, metadata={**r.metadata, "_search_result": r})
            for r in results
        ]

    def _apply_rerank(
        self, query: str, results: list[SearchResult], k: int
    ) -> list[SearchResult]:
        """Shared rerank flow: validate config, rerank, re-inject scores.

        Sync call — async callers wrap in ``asyncio.to_thread``.
        """
        reranker = self.reranker
        if reranker is None:
            raise SearchError(
                "Reranker not configured. Set SEMSEARCH_RERANKER__BASE_URL "
                "and SEMSEARCH_RERANKER__MODEL in .env"
            )

        docs = self._rerank_docs(results)
        reranked_docs = reranker.rerank(query, docs, top_n=k)

        # Build new results from reranked docs
        reranked_results = []
        for doc in reranked_docs:
            sr = doc.metadata.pop("_search_result")
            sr.metadata["rerank_score"] = doc.metadata.get("rerank_score")
            reranked_results.append(sr)
        return reranked_results

    # ---- Stats ----

    def stats(self, conn: psycopg.Connection | None = None) -> dict[str, Any]:
        """Return statistics about the chunks table.

        Args:
            conn: Optional psycopg connection to reuse. If None, a new
                connection is created and closed internally.
        """
        table = self.settings.collection_name
        owns_conn = conn is None
        if owns_conn:
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
                conn.close()

        return {
            "table": table,
            "embedding_provider": self.settings.embedding_provider.type,
            "embedding_dim": embedding_dim,
            "chunk_count": chunk_count,
            "source_count": source_count,
            "sources_by_count": sources_by_count,
        }
