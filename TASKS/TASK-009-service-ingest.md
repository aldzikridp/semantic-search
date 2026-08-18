# TASK-009: Core Service — Ingest Method

> **Phase**: 8.3 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-008
> **Blocks**: TASK-012, TASK-013, TASK-014, TASK-016

## Objective

Implement the `ingest()` method — the most complex method in the service. This is the **write path** that uses service-owned SQLAlchemy transactions (NOT `PGVectorStore.add_documents`).

## File to Modify

### `src/semsearch/service.py` (add `ingest` method)

## Implementation

### Method Signature

```python
def ingest(self, path: Path, *, reembed_unchanged: bool = False) -> IngestResult:
```

### The Four Cases (SPEC §10.1)

| Case | Existing row? | Hash matches? | Action | Embedding API call? |
|------|---------------|---------------|--------|---------------------|
| **A** | yes | yes | Reuse embedding; UPDATE `ingested_at` | ❌ No |
| **B** | yes | no | Re-embed, UPSERT | ✅ Yes |
| **C** | no | n/a | Embed, INSERT | ✅ Yes |
| **D** | yes (stale tail) | n/a | DELETE | ❌ No |

### Step-by-Step Flow

```python
def ingest(self, path: Path, *, reembed_unchanged: bool = False) -> IngestResult:
    source = str(path)
    now = datetime.utcnow()

    # Step 1: Load
    loader = pick_loader(path)
    docs = loader()
    docs = with_doc_type(docs, path)

    # Step 2: Split
    chunks = self.splitter.split_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)

    # Step 3: Compute hashes
    hashes = [sha256(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    # Step 4: Fetch existing rows for this source
    with self.engine.begin() as conn:
        existing_rows = conn.execute(
            text(f"SELECT langchain_id, chunk_index, document_hash FROM {self.settings.collection_name} WHERE source = :source ORDER BY chunk_index"),
            {"source": source}
        ).fetchall()
    existing_by_index = {row.chunk_index: row for row in existing_rows}

    # Step 5: Classify chunks
    case_a = []  # (chunk, index, hash, langchain_id) — reuse
    case_b = []  # (chunk, index, hash, langchain_id) — re-embed
    case_c = []  # (chunk, index, hash) — new
    for i, chunk in enumerate(chunks):
        h = hashes[i]
        existing = existing_by_index.get(i)
        if existing is None:
            case_c.append((chunk, i, h))
        elif existing.document_hash == h and not reembed_unchanged:
            case_a.append((chunk, i, h, existing.langchain_id))
        else:
            case_b.append((chunk, i, h, existing.langchain_id))

    # Step 6: Compute CASE D
    max_existing_idx = max(existing_by_index.keys()) if existing_by_index else -1
    case_d_count = max(0, max_existing_idx + 1 - len(chunks)) if max_existing_idx >= len(chunks) else 0

    # Step 7: Batch-embed CASE B + C only
    texts_to_embed = [c[0].page_content for c in case_b] + [c[0].page_content for c in case_c]
    vectors = self.embedder.embed_documents(texts_to_embed) if texts_to_embed else []
    vectors_b = vectors[:len(case_b)]
    vectors_c = vectors[len(case_b):]

    # Step 8: Execute writes in ONE transaction
    with self.engine.begin() as conn:
        # CASE A: cheap UPDATE (ingested_at only)
        if case_a:
            case_a_ids = [a[3] for a in case_a]
            conn.execute(
                text(f"""
                    UPDATE {self.settings.collection_name}
                    SET langchain_metadata = jsonb_set(
                        langchain_metadata, '{{ingested_at}}', to_jsonb(:now::text))
                    WHERE langchain_id = ANY(:ids)
                """),
                {"ids": case_a_ids, "now": now.isoformat()}
            )

        # CASE B + C: INSERT ... ON CONFLICT DO UPDATE
        for (chunk, idx, h), vec in zip(case_b + case_c, vectors_b + vectors_c):
            chunk_id = f"{source}::{idx}"
            metadata = {
                "doc_type": chunk.metadata.get("doc_type"),
                "ingested_at": now.isoformat(),
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            }
            # Add type-specific metadata
            if "page" in chunk.metadata:
                metadata["page"] = chunk.metadata["page"]
            if "row" in chunk.metadata:
                metadata["row"] = chunk.metadata["row"]

            conn.execute(
                text(f"""
                    INSERT INTO {self.settings.collection_name}
                        (langchain_id, embedding, content, langchain_metadata,
                         source, chunk_index, document_hash)
                    VALUES (:id, :vec, :text, :meta::jsonb, :source, :chunk_index, :hash)
                    ON CONFLICT (source, chunk_index) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        document_hash = EXCLUDED.document_hash,
                        langchain_metadata = EXCLUDED.langchain_metadata
                """),
                {
                    "id": chunk_id,
                    "vec": str(vec),  # pgvector accepts string representation
                    "text": chunk.page_content,
                    "meta": json.dumps(metadata),
                    "source": source,
                    "chunk_index": idx,
                    "hash": h,
                }
            )

        # CASE D: delete stale tail chunks
        if case_d_count > 0:
            conn.execute(
                text(f"""
                    DELETE FROM {self.settings.collection_name}
                    WHERE source = :source AND chunk_index >= :new_len
                """),
                {"source": source, "new_len": len(chunks)}
            )

    return IngestResult(
        source=source,
        chunks_added=len(case_c),
        chunks_reused=len(case_a),
        chunks_updated=len(case_b),
        chunks_pruned=case_d_count,
        ingested_at=now,
    )
```

## Critical Implementation Details

### 1. Service-owned SQL, NOT PGVectorStore.add_documents()

Three reasons:
- **Atomicity**: `add_documents()` does row-by-row commits — mid-loop failure leaves partial state
- **No double embedding**: We precompute embeddings for B/C; `add_documents` would re-embed them
- **UPSERT**: `ON CONFLICT (source, chunk_index) DO UPDATE` not exposed by `add_documents`

### 2. Deterministic IDs

```python
chunk_id = f"{source}::{chunk_index}"
```

This makes `langchain_id` human-readable and matches the natural key.

### 3. Vector format for pgvector

pgvector accepts vectors as strings: `str([0.1, 0.2, ...])` or as list. Check psycopg3 + pgvector integration.

### 4. SHA256 hash computation

```python
import hashlib
hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
```

Returns 64-char hex string, stored in `CHAR(64)` column.

### 5. Transaction atomicity

ALL writes (CASE A, B, C, D) in ONE `engine.begin()` block. On exception, automatic ROLLBACK.

## Verification (Integration Tests)

- [ ] I-1: Ingest text file → `chunks_added` correct, rows in DB
- [ ] I-2: Ingest PDF → `chunks_added >= 5`, each has `metadata.page`
- [ ] I-3: Ingest CSV → `chunks_added == 10`, each has `metadata.row`
- [ ] I-9: Re-ingest unchanged → `chunks_reused == N`, `chunks_added == 0`
- [ ] I-9b: Re-ingest unchanged → zero `embed_documents` calls
- [ ] I-10: Re-ingest with 1 chunk changed → `chunks_reused == N-1`, `chunks_updated == 1`
- [ ] I-10b: Partial change → `embed_documents` called once with list of length 1
- [ ] I-10c: DB state after partial change → only changed chunk has new content/hash
- [ ] I-10d: CASE D → `chunks_pruned == 2` when file shortened
- [ ] I-10e: File grows → `chunks_pruned == 0`
- [ ] I-11: `--force` → `chunks_reused == 0`, `chunks_updated == N`
