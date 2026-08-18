"""File-type dispatch and metadata injection (spec §7.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document


# Extension → doc_type mapping. Loaders don't set doc_type natively —
# the service injects it after load() based on this map.
_DOC_TYPE_BY_EXT: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
}

# Whitelist of supported extensions for quick membership checks.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_DOC_TYPE_BY_EXT)


def pick_loader(path: Path) -> Callable[[], list[Document]]:
    """Dispatch on file extension and return a lazy loader callable.

    Dispatch rules:
        .txt, .md  → TextLoader (single document)
        .pdf       → PyMuPDFLoader (one Document per page)
        .csv       → CSVLoader (one Document per row; first row = headers)
        .json      → JSONLoader (jq_schema='.[].content')

    Args:
        path: File path. Must exist on disk.

    Returns:
        A callable that, when invoked, returns ``list[Document]``. Each
        Document's metadata will include ``source`` and ``doc_type`` after
        ``with_doc_type()`` is called by the service.

    Raises:
        ValueError: If the file extension is not in ``_DOC_TYPE_BY_EXT``.
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _DOC_TYPE_BY_EXT:
        supported = ", ".join(sorted(_DOC_TYPE_BY_EXT))
        raise ValueError(
            f"Unsupported file extension {suffix!r}. "
            f"Supported: {supported}"
        )

    # Lazy import so import-time doesn't pull in heavy deps (pymupdf, jq).
    if suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")

    elif suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(str(path))

    elif suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader

        loader = CSVLoader(str(path))

    elif suffix == ".json":
        from langchain_community.document_loaders import JSONLoader

        loader = JSONLoader(
            str(path),
            jq_schema=".[].content",
            text_content=False,
        )
    else:  # pragma: no cover — unreachable due to guard above
        raise ValueError(f"Unsupported file extension: {suffix}")

    return lambda: loader.load()


def with_doc_type(docs: list[Document], path: Path) -> list[Document]:
    """Inject ``source`` and ``doc_type`` into each Document's metadata.

    Mutates the documents in place and returns the same list. Called by the
    service after ``loader.load()`` because LangChain's stock loaders don't
    set ``doc_type`` natively.

    Args:
        docs: Documents returned by a LangChain loader.
        path: Original file path (used for both ``source`` and ``doc_type``).

    Returns:
        The same ``docs`` list, now with metadata enriched.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    doc_type = _DOC_TYPE_BY_EXT.get(suffix)
    if doc_type is None:
        supported = ", ".join(sorted(_DOC_TYPE_BY_EXT))
        raise ValueError(
            f"Unsupported file extension {suffix!r}. "
            f"Supported: {supported}"
        )
    source = str(path)
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["doc_type"] = doc_type
    return docs
