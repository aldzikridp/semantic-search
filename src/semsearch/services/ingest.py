"""Ingest mixin: file loading, chunking, embedding and write-path SQL.

Part of ``semsearch.services`` (PLAN.md Phase D) — implementation detail of
``semsearch.service.SemanticSearchService``. The write path stays
service-owned raw psycopg per AGENTS.md decision #1.
"""

import fnmatch
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
from langchain_core.documents import Document
from psycopg import sql as pg_sql

from semsearch.config import Settings
from semsearch.errors import FileIngestError
from semsearch.loaders import pick_loader, with_doc_type
from semsearch.models import BatchAggregate, BatchIngestResult, IngestResult
from semsearch.splitter import split_documents

from .base import BaseService, _exec

if TYPE_CHECKING:
    from semsearch.models import DeleteResult

logger = logging.getLogger(__name__)


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


class IngestMixin(BaseService):
    """Single-file and directory ingestion plus reingest."""

    if TYPE_CHECKING:
        # Cross-mixin dependency: ``delete`` is provided by AdminMixin in the
        # composed SemanticSearchService facade. Declared here (types only,
        # erased at runtime) so mixin bodies type-check standalone.
        def delete(
            self, filter: dict[str, Any], conn: psycopg.Connection | None = None
        ) -> "DeleteResult": ...

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
        if conn is None:
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

            existing_by_index: dict[int, tuple[Any, ...]] = {
                row[1]: row for row in existing_rows
            }

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
                    # shape-match (see base._exec for rationale).
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
                self._release_conn(conn)

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
        failed_files: list[dict[str, Any]] = []
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
            self._release_conn(conn)

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
                self._release_conn(conn)

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

    # ---- Reingest ----

    def reingest(self, path: Path) -> IngestResult:
        """Delete all chunks for the source, then ingest fresh."""
        conn = self._get_conn()
        try:
            self.delete({"source": str(path)}, conn=conn)
            return self.ingest(path, reembed_unchanged=True, conn=conn)
        finally:
            self._release_conn(conn)
