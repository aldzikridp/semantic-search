# TASK-004: Document Loaders

> **Phase**: 4 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-001 (langchain-core)
> **Blocks**: TASK-008

## Objective

Implement file-type dispatch (`pick_loader`) and metadata injection (`with_doc_type`) for loading heterogeneous document sources.

## File to Create

### `src/semsearch/loaders.py`

## Implementation

### 1. Extension → doc_type mapping

```python
_DOC_TYPE_BY_EXT = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "json",
}
```

### 2. `pick_loader(path: Path) -> Callable[[], list[Document]]`

Dispatch on file extension:
- `.txt`, `.md` → `TextLoader(path, encoding="utf-8")`
- `.pdf` → `PyMuPDFLoader(str(path))`
- `.csv` → `CSVLoader(str(path))`
- `.json` → `JSONLoader(str(path), jq_schema=".[].content", text_content=False)`

Raises:
- `ValueError` for unsupported extensions
- `FileNotFoundError` if path doesn't exist

Returns a **callable** (not the loaded docs directly) — lazy loading.

### 3. `with_doc_type(docs: list[Document], path: Path) -> list[Document]`

Injects `source` (as `str(path)`) and `doc_type` (from extension) into each Document's metadata. Mutates in place and returns the same list.

**Key insight**: LangChain loaders do NOT set `doc_type` natively. The service wraps loader output to inject it based on file extension.

### Loader Output Metadata

Each loader produces different metadata:
- **TextLoader**: `{source: str}`
- **PyMuPDFLoader**: `{source: str, page: int}` (one doc per page)
- **CSVLoader**: `{source: str, row: int}` (one doc per row)
- **JSONLoader**: `{source: str, id: Any}` (one doc per element)

After `with_doc_type()`, all have `doc_type` added.

## Critical Notes

1. **`JSONLoader` uses `jq_schema='.[].content'`** — Requires `jq` Python package (pinned in requirements.txt)

2. **`text_content=False`** for JSONLoader — Tells it NOT to treat the extracted value as plain text (it may be structured)

3. **Loaders return a callable** — `pick_loader` returns `lambda: loader.load()`, not the loaded docs. This allows lazy loading.

4. **Path is stored as `str(path)`** — No normalization. `docs/handbook.pdf` and `./docs/handbook.pdf` are different sources.

## Verification (Unit Tests U-1 to U-4)

- [ ] U-1: `pick_loader(Path("a.txt"))` returns TextLoader-backed callable
- [ ] U-2: `pick_loader(Path("a.pdf"))` returns PyMuPDFLoader-backed callable
- [ ] U-3: `pick_loader(Path("a.docx"))` raises `ValueError`
- [ ] U-4: `pick_loader(Path("missing.txt"))` raises `FileNotFoundError`
- [ ] `with_doc_type` adds `source` and `doc_type` to each doc
- [ ] Metadata from loaders (page, row) preserved after `with_doc_type`
