"""Core service — SemanticSearchService facade (spec §7.6)."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
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

# Whitelist of file extensions handled by pick_loader.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".csv", ".json")

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
        """Get a raw psycopg connection."""
        return psycopg.connect(self._db_url)

    @classmethod
    def from_settings(cls, settings: Settings) -> SemanticSearchService:
        """Build all internal components from settings."""
        engine = build_engine(settings)
        embedder = build_embedder(settings)
        return cls(settings, engine, embedder)

    # ---- Lifecycle ----

    def __enter__(self) -> SemanticSearchService:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying PGEngine connection pool."""
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

    def _get_vector_size(self) -> int:
        """Embed a dummy query to determine the vector dimension."""
        return len(self.embedder.embed_query("dimension probe"))

    # ---- Ingest ----

    def ingest(self, path: Path, *, reembed_unchanged: bool = False) -> IngestResult:
        """Ingest a single file into the store.

        Args:
            path: File to ingest.
            reembed_unchanged: If True, force re-embed everything.

        Returns:
            IngestResult with counts for each case.
        """
        table = self.settings.collection_name
        source = str(path)
        now = datetime.now(timezone.utc)

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
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT langchain_id, chunk_index, document_hash "
                        f"FROM {table} "
                        f"WHERE source = %s "
                        f"ORDER BY chunk_index",
                        (source,),
                    )
                    existing_rows = cur.fetchall()
            finally:
                conn.close()

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

            # Step 8: Execute writes in ONE transaction
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    # CASE A: cheap UPDATE (ingested_at only)
                    if case_a_ids:
                        cur.execute(
                            f"UPDATE {table} "
                            f"SET langchain_metadata = jsonb_set("
                            f"  langchain_metadata::jsonb, '{{ingested_at}}', "
                            f"  to_jsonb(CAST(%s AS text))"
                            f")::json "
                            f"WHERE langchain_id = ANY(%s)",
                            (now.isoformat(), case_a_ids),
                        )

                    # CASE B + C: INSERT ... ON CONFLICT DO UPDATE
                    for vec_idx, chunk_idx in enumerate(bc_indices):
                        chunk = chunks[chunk_idx]
                        h = hashes[chunk_idx]
                        vec = vectors[vec_idx]
                        chunk_id = f"{source}::{chunk_idx}"

                        metadata: dict[str, Any] = {
                            "doc_type": chunk.metadata.get("doc_type"),
                            "ingested_at": now.isoformat(),
                            "chunk_size": self.settings.chunk_size,
                            "chunk_overlap": self.settings.chunk_overlap,
                        }
                        if "page" in chunk.metadata:
                            metadata["page"] = chunk.metadata["page"]
                        if "row" in chunk.metadata:
                            metadata["row"] = chunk.metadata["row"]

                        cur.execute(
                            f"INSERT INTO {table} "
                            f"  (langchain_id, embedding, content, langchain_metadata, "
                            f"   source, chunk_index, document_hash) "
                            f"VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s) "
                            f"ON CONFLICT (source, chunk_index) DO UPDATE "
                            f"SET embedding = EXCLUDED.embedding, "
                            f"    content = EXCLUDED.content, "
                            f"    document_hash = EXCLUDED.document_hash, "
                            f"    langchain_metadata = EXCLUDED.langchain_metadata",
                            (
                                chunk_id,
                                str(vec),
                                chunk.page_content,
                                json.dumps(metadata),
                                source,
                                chunk_idx,
                                h,
                            ),
                        )

                    # CASE D: delete stale tail chunks
                    if case_d_count > 0:
                        cur.execute(
                            f"DELETE FROM {table} "
                            f"WHERE source = %s AND chunk_index >= %s",
                            (source, len(chunks)),
                        )

                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

            return IngestResult(
                source=source,
                chunks_added=len(case_c_indices),
                chunks_reused=len(case_a_indices),
                chunks_updated=len(case_b_indices),
                chunks_pruned=case_d_count,
                ingested_at=now,
            )

        except Exception as e:
            raise FileIngestError(f"Failed to ingest {path}: {e}") from e

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

        for file_path in files:
            try:
                result = self.ingest(file_path, reembed_unchanged=reembed_unchanged)
                succeeded += 1
                ingested_sources.add(str(file_path))
                aggregate.chunks_added += result.chunks_added
                aggregate.chunks_reused += result.chunks_reused
                aggregate.chunks_updated += result.chunks_updated
                aggregate.chunks_pruned += result.chunks_pruned
            except Exception as e:
                failed += 1
                failed_files.append({"path": str(file_path), "error": str(e)})
                if not continue_on_error:
                    raise FileIngestError(f"Failed to ingest {file_path}: {e}") from e

        elapsed = time.monotonic() - start_time

        pruned_sources: list[str] = []
        pruned_chunks = 0

        if prune:
            table = self.settings.collection_name
            dir_prefix = str(dir_path) + "/"

            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT DISTINCT source FROM {table} "
                        f"WHERE source LIKE %s",
                        (dir_prefix + "%",),
                    )
                    db_sources = {row[0] for row in cur.fetchall()}
            finally:
                conn.close()

            orphan_sources = db_sources - ingested_sources

            for source in orphan_sources:
                pruned_sources.append(source)
                if not prune_dry_run:
                    try:
                        delete_result = self.delete({"source": source})
                        pruned_chunks += delete_result.deleted_count
                    except Exception as e:
                        logger.warning(f"Failed to prune {source}: {e}")

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

    def delete(self, filter: dict) -> DeleteResult:
        """Delete every chunk matching the filter."""
        table = self.settings.collection_name

        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    if not filter:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        cur.execute(f"DELETE FROM {table}")
                    else:
                        # Use PGVectorStore for filtered delete
                        # First count using the store
                        # For simplicity, count before and after
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count_before = cur.fetchone()[0]

                        # Use PGVectorStore delete (it manages its own connection)
                        cur.close()
                        conn.close()
                        conn = None

                        self.store.delete(filter=filter)

                        conn = self._get_conn()
                        cur = conn.cursor()
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count_after = cur.fetchone()[0]
                        count = count_before - count_after

                    conn.commit()
            except Exception:
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()

            return DeleteResult(deleted_count=count, filter=filter)

        except Exception as e:
            raise DeleteError(f"Delete failed: {e}") from e

    # ---- Reingest ----

    def reingest(self, path: Path) -> IngestResult:
        """Delete all chunks for the source, then ingest fresh."""
        source = str(path)
        self.delete({"source": source})
        return self.ingest(path, reembed_unchanged=True)

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
        if k is None:
            k = self.settings.default_k
        if not (1 <= k <= 50):
            raise ValueError(f"k must be between 1 and 50, got {k}")

        # When reranking, fetch more candidates for better reranking quality
        fetch_k = k * 4 if rerank else k

        try:
            results_with_scores = self.store.similarity_search_with_score(
                query,
                k=fetch_k,
                filter=filter,
            )
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        # Convert to SearchResult
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

        # Rerank if requested
        if rerank:
            reranker = self.reranker
            if reranker is None:
                raise SearchError(
                    "Reranker not configured. Set SEMSEARCH_RERANKER__BASE_URL "
                    "and SEMSEARCH_RERANKER__MODEL in .env"
                )
            # Extract documents from results for reranking
            docs = []
            for r in results:
                doc = Document(
                    page_content=r.content,
                    metadata={**r.metadata, "_search_result": r},
                )
                docs.append(doc)

            reranked_docs = reranker.rerank(query, docs, top_n=k)

            # Build new results from reranked docs
            results = []
            for doc in reranked_docs:
                sr = doc.metadata.pop("_search_result")
                sr.metadata["rerank_score"] = doc.metadata.get("rerank_score")
                results.append(sr)

        return results[:k]

    # ---- Stats ----

    def stats(self) -> dict[str, Any]:
        """Return statistics about the chunks table."""
        table = self.settings.collection_name

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                chunk_count = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(DISTINCT source) FROM {table}")
                source_count = cur.fetchone()[0]

                cur.execute(
                    f"SELECT source, COUNT(*) AS cnt "
                    f"FROM {table} "
                    f"GROUP BY source "
                    f"ORDER BY cnt DESC "
                    f"LIMIT 20"
                )
                sources_by_count = [(row[0], row[1]) for row in cur.fetchall()]
        finally:
            conn.close()

        return {
            "table": table,
            "embedding_provider": self.settings.embedding_provider.type,
            "embedding_dim": self._get_vector_size(),
            "chunk_count": chunk_count,
            "source_count": source_count,
            "sources_by_count": sources_by_count,
        }
