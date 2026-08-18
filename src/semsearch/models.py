"""Pydantic models for the public API contract (spec §7.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result returned by the service.

    score is SIMILARITY (higher = better), converted from LangChain's
    cosine distance via: score = 1.0 - distance.
    """

    id: str  # langchain_id (TEXT, deterministic "{source}::{chunk_index}")
    content: str  # chunk text
    score: float = Field(
        ...,
        description="Cosine similarity in [-1, 1]; higher is better. "
        "Computed as 1.0 - cosine_distance.",
    )
    source: str | None = None  # top-level column (str(path))
    chunk_index: int | None = None  # top-level column (0-based position)
    page: int | None = None  # langchain_metadata.page (PDF only)
    row: int | None = None  # langchain_metadata.row (CSV only)
    doc_type: str | None = None  # langchain_metadata.doc_type
    metadata: dict[str, Any] = Field(default_factory=dict)  # full langchain_metadata blob


class DeleteResult(BaseModel):
    """Result of a delete operation.

    deleted_count is computed BEFORE the actual delete via SELECT COUNT(*)
    in the same transaction.
    """

    deleted_count: int
    filter: dict[str, Any]


class IngestResult(BaseModel):
    """Result of ingesting a single file."""

    source: str
    chunks_added: int  # CASE C: newly embedded (no prior row)
    chunks_reused: int  # CASE A: content unchanged; existing embedding reused
    chunks_updated: int  # CASE B: content changed; re-embedded
    chunks_pruned: int = 0  # CASE D: stale tail chunks deleted (file shortened)
    ingested_at: datetime


class BatchAggregate(BaseModel):
    """Aggregate counts across multiple files in a batch.

    Separate from IngestResult because `source` doesn't apply at the batch level.
    """

    chunks_added: int = 0
    chunks_reused: int = 0
    chunks_updated: int = 0
    chunks_pruned: int = 0


class BatchIngestResult(BaseModel):
    """Result of ingest_dir(). Aggregates per-file outcomes + optional prune info."""

    dir: str
    files_discovered: int  # all files matching glob/exclude (incl. unsupported)
    files_skipped_unsupported: int  # subset with unsupported extensions
    files_attempted: int  # files_discovered - files_skipped_unsupported
    files_succeeded: int
    files_failed: int
    failed_files: list[dict] = Field(
        default_factory=list,
        description='List of {"path": str, "error": str} dicts for each failed file.',
    )
    aggregate: BatchAggregate  # sum of chunks_added/reused/updated/pruned
    elapsed_seconds: float
    pruned_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose chunks were deleted. Empty unless prune=True.",
    )
    pruned_chunks: int = Field(
        default=0,
        description="Total chunks deleted by prune. 0 if prune_dry_run=True.",
    )
