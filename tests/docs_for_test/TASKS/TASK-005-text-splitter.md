# TASK-005: Text Splitter

> **Phase**: 5 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-001 (langchain-text-splitters)
> **Blocks**: TASK-008

## Objective

Implement a thin wrapper around `RecursiveCharacterTextSplitter` for chunking documents.

## File to Create

### `src/semsearch/splitter.py`

## Implementation

### 1. `build_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter`

```python
def build_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
```

### 2. `split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]`

```python
def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = build_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
```

## Chunking Behavior by Source Type

| Source | Special handling |
|--------|------------------|
| Text / Markdown | None — feed raw text to splitter |
| PDF | PyMuPDFLoader returns one Document per page. Splitter further splits each page. `metadata.page` preserved. |
| CSV | CSVLoader returns one Document per row. Splitter skipped if row content < chunk_size. |
| JSON | JSONLoader produces one Document per element. Each element is its own chunk if `len(content) <= chunk_size`. |

## Metadata Propagation

Every chunk inherits the source Document's metadata. The splitter preserves all existing metadata (`source`, `page`, `row`, `doc_type`) and adds nothing new.

The service later promotes `source`, `chunk_index`, and `document_hash` to top-level table columns.

## Critical Notes

1. **Separators order**: `["\n\n", "\n", ". ", " ", ""]` — Tries to split on paragraph boundaries first, then sentences, then words.

2. **`chunk_index` stability**: `RecursiveCharacterTextSplitter` is deterministic given the same `chunk_size`, `chunk_overlap`, and separators. Same file + same settings = same chunk assignments.

3. **`chunk_size` default is 1000 characters** — Not tokens. For most English text, ~750 tokens.

4. **`chunk_overlap` default is 200 characters** — Provides context continuity between chunks.

## Verification

- [ ] `split_documents` returns ≥ len(input_docs) chunks
- [ ] All metadata preserved in output chunks
- [ ] Same input + same settings = deterministic output
- [ ] Changing `chunk_size` changes chunk boundaries
