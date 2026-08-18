"""Thin wrapper around RecursiveCharacterTextSplitter (spec §7.4)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with sensible defaults.

    Separators try paragraph boundaries first, then sentences, then words:
    ``["\\n\\n", "\\n", ". ", " ", ""]``

    Args:
        chunk_size: Maximum chunk length in characters (default 1000).
        chunk_overlap: Overlap between consecutive chunks (default 200).

    Returns:
        A configured ``RecursiveCharacterTextSplitter`` instance.
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into chunks, preserving all metadata.

    Every output chunk inherits the source Document's metadata (``source``,
    ``page``, ``row``, ``doc_type``, etc.). The splitter adds nothing new —
    the service later promotes ``source``, ``chunk_index``, and
    ``document_hash`` to top-level table columns.

    Args:
        docs: Documents to split (e.g. from ``pick_loader`` + ``with_doc_type``).
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of chunk Documents with ``len(result) >= len(docs)``.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
