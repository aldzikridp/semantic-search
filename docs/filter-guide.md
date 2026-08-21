# Filter Parameter Reference

The `filter` parameter is a JSON object (dict) used to narrow search results or target specific chunks for deletion.

---

## Important: Search vs Delete

| Operation | Filter Support |
|-----------|----------------|
| `search` | **Full operator support** — all comparison, logical, and text operators |
| `delete` | **Exact match only** — simple key-value equality, no operators |

---

## Search Filters (Full Operator Support)

Search filters are processed by langchain-postgres and support rich query expressions.

### Basic Exact Match

```json
{"doc_type": "pdf"}
```

Multiple keys at the top level are automatically combined with AND:

```json
{"doc_type": "pdf", "source": "/docs/readme.md"}
```

### Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equal to (default) | `{"page": {"$eq": 5}}` |
| `$ne` | Not equal to | `{"doc_type": {"$ne": "csv"}}` |
| `$gt` | Greater than | `{"page": {"$gt": 5}}` |
| `$gte` | Greater than or equal | `{"page": {"$gte": 1}}` |
| `$lt` | Less than | `{"row": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"row": {"$lte": 50}}` |

### Special Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$in` | Value in list | `{"doc_type": {"$in": ["pdf", "csv"]}}` |
| `$nin` | Value not in list | `{"doc_type": {"$nin": ["json"]}}` |
| `$between` | Between range (inclusive) | `{"page": {"$between": [1, 10]}}` |
| `$exists` | Field exists / is null | `{"page": {"$exists": true}}` |

### Text Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$like` | SQL LIKE (case-sensitive) | `{"source": {"$like": "docs/%"}}` |
| `$ilike` | SQL ILIKE (case-insensitive) | `{"source": {"$ilike": "%readme%"}}` |

Use `%` as wildcard, `_` for single character.

### Logical Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$and` | All conditions must match | `{"$and": [{"doc_type": "pdf"}, {"page": {"$gt": 5}}]}` |
| `$or` | Any condition must match | `{"$or": [{"doc_type": "pdf"}, {"doc_type": "csv"}]}` |
| `$not` | Negate a condition | `{"$not": {"doc_type": "pdf"}}` |

### Search Filter Examples

```bash
# PDFs only
semsearch search "query" --filter '{"doc_type": "pdf"}'

# PDFs with page > 5
semsearch search "query" --filter '{"$and": [{"doc_type": "pdf"}, {"page": {"$gt": 5}}]}'

# PDFs or CSVs
semsearch search "query" --filter '{"$or": [{"doc_type": "pdf"}, {"doc_type": "csv"}]}'

# Files in docs/ directory (case-insensitive)
semsearch search "query" --filter '{"source": {"$ilike": "docs/%"}}'

# First 10 rows of CSVs
semsearch search "query" --filter '{"$and": [{"doc_type": "csv"}, {"row": {"$lte": 10}}]}'

# Multiple file types
semsearch search "query" --filter '{"doc_type": {"$in": ["pdf", "csv", "json"]}}'

# Exclude JSON files
semsearch search "query" --filter '{"doc_type": {"$ne": "json"}}'

# Field exists check
semsearch search "query" --filter '{"page": {"$exists": true}}'
```

---

## Delete Filters (Exact Match Only)

Delete filters use raw SQL with simple equality matching. **No operators are supported.**

### How It Works

```json
{"source": "/docs/readme.md", "doc_type": "text"}
```

Becomes:

```sql
WHERE source = '/docs/readme.md' AND langchain_metadata->>'doc_type' = 'text'
```

### Filterable Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | File path (top-level column, indexed) |
| `doc_type` | string | `"text"`, `"pdf"`, `"csv"`, `"json"` |
| `page` | string | Page number (PDF only, compared as string) |
| `row` | string | Row number (CSV only, compared as string) |

### Delete Filter Examples

```bash
# Delete by source
semsearch delete --filter '{"source": "/docs/old-file.md"}'

# Delete by doc type
semsearch delete --filter '{"doc_type": "pdf"}'

# Delete by source AND doc type
semsearch delete --filter '{"source": "/data/sales.csv", "doc_type": "csv"}'

# Delete ALL chunks (dangerous!)
semsearch delete --all --yes
```

### Limitations

- No `$gt`, `$lt`, `$between` — only exact equality
- No `$in`, `$nin` — can't match multiple values
- No `$like`, `$ilike` — no pattern matching
- No `$and`, `$or`, `$not` — no logical operators
- No `$exists` — can't check for null fields
- Values are compared as strings (even numbers)

---

## Filterable Fields Reference

| Field | Type | Available On | In Search | In Delete |
|-------|------|--------------|-----------|-----------|
| `source` | string | All files | ✅ Full operators | ✅ Exact match |
| `doc_type` | string | All files | ✅ Full operators | ✅ Exact match |
| `page` | integer | PDF only | ✅ Full operators | ✅ Exact match (as string) |
| `row` | integer | CSV only | ✅ Full operators | ✅ Exact match (as string) |
| `chunk_index` | integer | All files | ✅ Full operators | ✅ Exact match (as string) |
| `document_hash` | string | All files | ✅ Full operators | ✅ Exact match |

---

## API Examples

### Search with operators

```bash
curl -X POST http://localhost:8383/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "python tutorial",
    "k": 5,
    "filter": {"$and": [{"doc_type": "pdf"}, {"page": {"$gt": 2}}]}
  }'
```

### Delete with exact match

```bash
curl -X DELETE http://localhost:8383/delete \
  -H "Content-Type: application/json" \
  -d '{"filter": {"source": "/docs/old-file.md"}}'
```

---

## Schema for AI Agents

### Search Filter Schema

```json
{
  "type": "object",
  "description": "Filter dict with full operator support. Multiple top-level keys are AND-ed.",
  "additionalProperties": {
    "oneOf": [
      {"type": "string", "description": "Exact match value"},
      {"type": "integer"},
      {"type": "boolean"},
      {
        "type": "object",
        "description": "Operator expression",
        "properties": {
          "$eq": {"description": "Equal to"},
          "$ne": {"description": "Not equal to"},
          "$gt": {"description": "Greater than"},
          "$gte": {"description": "Greater than or equal"},
          "$lt": {"description": "Less than"},
          "$lte": {"description": "Less than or equal"},
          "$in": {"type": "array", "description": "Value in list"},
          "$nin": {"type": "array", "description": "Value not in list"},
          "$between": {"type": "array", "items": 2, "description": "[low, high] inclusive"},
          "$like": {"type": "string", "description": "SQL LIKE pattern (% wildcard)"},
          "$ilike": {"type": "string", "description": "SQL ILIKE pattern (case-insensitive)"},
          "$exists": {"type": "boolean", "description": "Field exists / is not null"}
        }
      }
    ]
  },
  "properties": {
    "$and": {"type": "array", "description": "All conditions must match"},
    "$or": {"type": "array", "description": "Any condition must match"},
    "$not": {"description": "Negate a condition"}
  },
  "examples": [
    {"doc_type": "pdf"},
    {"source": {"$ilike": "docs/%"}},
    {"$and": [{"doc_type": "pdf"}, {"page": {"$gt": 5}}]},
    {"$or": [{"doc_type": "pdf"}, {"doc_type": "csv"}]},
    {"doc_type": {"$in": ["pdf", "csv", "json"]}}
  ]
}
```

### Delete Filter Schema

```json
{
  "type": "object",
  "description": "Filter dict with exact match only. All conditions are AND-ed. No operators supported.",
  "additionalProperties": {
    "type": "string",
    "description": "Exact value to match (compared as string)"
  },
  "examples": [
    {"source": "/docs/readme.md"},
    {"doc_type": "pdf"},
    {"source": "/data/sales.csv", "doc_type": "csv"}
  ]
}
```
