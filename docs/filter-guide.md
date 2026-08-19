# Filter Guide

The `--filter` option lets you narrow search results by file metadata. Filters are JSON dictionaries passed to the `search`, `delete`, and `ingest-dir` commands.

## Basic Syntax

```bash
semsearch search "query" --filter '{"key": "value"}'
semsearch delete --filter '{"key": "value"}'
```

## Simple Filters

### Exact Match

```bash
# Search only in a specific file
semsearch search "database" --filter '{"source": "TASKS/TASK-007-database-store.md"}'

# Search only PDFs
semsearch search "configuration" --filter '{"doc_type": "pdf"}'

# Search only CSVs
semsearch search "data" --filter '{"doc_type": "csv"}'
```

### Filter by File Type

The `doc_type` field corresponds to file extension:

| Extension | doc_type |
|-----------|----------|
| `.txt` | `text` |
| `.md` | `text` |
| `.pdf` | `pdf` |
| `.csv` | `csv` |
| `.json` | `json` |

```bash
# Only text files
semsearch search "notes" --filter '{"doc_type": "text"}'

# Only PDFs
semsearch search "manual" --filter '{"doc_type": "pdf"}'

# Only CSVs
semsearch search "revenue" --filter '{"doc_type": "csv"}'
```

### Filter by Source Path

```bash
# Exact file
semsearch search "setup" --filter '{"source": "docs/readme.md"}'

# Specific directory files
semsearch search "api" --filter '{"source": "src/semsearch/service.py"}'
```

## Pattern Matching

### Prefix Match ($ilike)

Use `$ilike` for pattern matching with `%` as wildcard:

```bash
# All files in docs/ directory
semsearch search "guide" --filter '{"source": {"$ilike": "docs/%"}}'

# All files starting with "task"
semsearch search "spec" --filter '{"source": {"$ilike": "TASKS/task%"}}'

# All markdown files
semsearch search "notes" --filter '{"source": {"$ilike": "%.md"}}'

# All files in subdirectory
semsearch search "test" --filter '{"source": {"$ilike": "tests/%"}}'
```

### Case-Insensitive Match

`$ilike` is case-insensitive:

```bash
# Matches README.md, readme.md, Readme.md, etc.
semsearch search "intro" --filter '{"source": {"$ilike": "%readme%"}}'
```

## Comparison Operators

### Numeric Comparisons

For fields like `page` (PDF) or `row` (CSV):

```bash
# PDF pages > 5
semsearch search "summary" --filter '{"page": {"$gt": 5}}'

# CSV rows 1-10
semsearch search "data" --filter '{"row": {"$between": [1, 10]}}'

# First page only
semsearch search "title" --filter '{"page": {"$eq": 1}}'
```

### Available Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equal to | `{"doc_type": {"$eq": "pdf"}}` |
| `$ne` | Not equal to | `{"doc_type": {"$ne": "csv"}}` |
| `$gt` | Greater than | `{"page": {"$gt": 5}}` |
| `$gte` | Greater than or equal | `{"page": {"$gte": 1}}` |
| `$lt` | Less than | `{"row": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"row": {"$lte": 50}}` |
| `$in` | In list | `{"doc_type": {"$in": ["pdf", "csv"]}}` |
| `$nin` | Not in list | `{"doc_type": {"$nin": ["json"]}}` |
| `$between` | Between range | `{"page": {"$between": [1, 10]}}` |
| `$like` | SQL LIKE (case-sensitive) | `{"source": {"$like": "docs/%"}}` |
| `$ilike` | SQL ILIKE (case-insensitive) | `{"source": {"$ilike": "docs/%"}}` |

## Logical Operators

### AND

All conditions must match:

```bash
# PDFs in docs/ directory
semsearch search "guide" --filter '{"$and": [{"doc_type": "pdf"}, {"source": {"$ilike": "docs/%"}}]}'

# First page of any PDF
semsearch search "intro" --filter '{"$and": [{"doc_type": "pdf"}, {"page": 1}]}'
```

### OR

Any condition must match:

```bash
# PDFs or CSVs
semsearch search "data" --filter '{"$or": [{"doc_type": "pdf"}, {"doc_type": "csv"}]}'

# Files in docs/ or tests/
semsearch search "code" --filter '{"$or": [{"source": {"$ilike": "docs/%"}}, {"source": {"$ilike": "tests/%"}}]}'
```

### Combining AND + OR

```bash
# (PDFs or CSVs) in docs/ directory
semsearch search "report" --filter '{"$and": [{"$or": [{"doc_type": "pdf"}, {"doc_type": "csv"}]}, {"source": {"$ilike": "docs/%"}}]}'
```

## Special Operators

### $exists

Check if a field exists:

```bash
# Only PDFs (which have page metadata)
semsearch search "content" --filter '{"page": {"$exists": true}}'

# Only CSVs (which have row metadata)
semsearch search "records" --filter '{"row": {"$exists": true}}'
```

### $not

Negate a condition:

```bash
# Not PDFs
semsearch search "text" --filter '{"$not": {"doc_type": "pdf"}}'

# Not in docs/ directory
semsearch search "code" --filter '{"$not": {"source": {"$ilike": "docs/%"}}}'
```

## Practical Examples

### Search by File Type

```bash
# Only PDFs
semsearch search "user manual" --filter '{"doc_type": "pdf"}'

# Only markdown files
semsearch search "documentation" --filter '{"doc_type": "text"}'

# Only CSVs
semsearch search "sales data" --filter '{"doc_type": "csv"}'

# Only JSON files
semsearch search "api response" --filter '{"doc_type": "json"}'
```

### Search by Location

```bash
# Files in docs/ folder
semsearch search "getting started" --filter '{"source": {"$ilike": "docs/%"}}'

# Files in src/ folder
semsearch search "implementation" --filter '{"source": {"$ilike": "src/%"}}'

# Files in tests/ folder
semsearch search "test case" --filter '{"source": {"$ilike": "tests/%"}}'

# Files in TASKS/ folder
semsearch search "task description" --filter '{"source": {"$ilike": "TASKS/%"}}'
```

### Search by Page (PDFs)

```bash
# First page of PDFs
semsearch search "title" --filter '{"$and": [{"doc_type": "pdf"}, {"page": 1}]}'

# Pages 5-10
semsearch search "chapter 2" --filter '{"$and": [{"doc_type": "pdf"}, {"page": {"$between": [5, 10]}}]}'

# Last pages
semsearch search "conclusion" --filter '{"$and": [{"doc_type": "pdf"}, {"page": {"$gt": 50}}]}'
```

### Search by Row (CSVs)

```bash
# First 10 rows
semsearch search "header data" --filter '{"$and": [{"doc_type": "csv"}, {"row": {"$lt": 10}}]}'

# Rows 50-100
semsearch search "mid data" --filter '{"$and": [{"doc_type": "csv"}, {"row": {"$between": [50, 100]}}]}'
```

### Exclude Files

```bash
# Everything except PDFs
semsearch search "notes" --filter '{"doc_type": {"$ne": "pdf"}}'

# Everything except test files
semsearch search "production" --filter '{"source": {"$not": {"$ilike": "tests/%"}}}'
```

### Multiple File Types

```bash
# PDFs and docs
semsearch search "guide" --filter '{"doc_type": {"$in": ["pdf", "text"]}}'

# Everything except CSVs and JSON
semsearch search "text content" --filter '{"doc_type": {"$nin": ["csv", "json"]}}'
```

## Delete Filters

### Delete by File

```bash
# Delete all chunks from a specific file
semsearch delete --filter '{"source": "docs/old-readme.md"}'
```

### Delete by Directory

```bash
# Delete all chunks from docs/ directory
semsearch delete --filter '{"source": {"$ilike": "docs/%"}}'

# Delete all chunks from old/ subdirectory
semsearch delete --filter '{"source": {"$ilike": "%/old/%"}}'
```

### Delete by File Type

```bash
# Delete all PDF chunks
semsearch delete --filter '{"doc_type": "pdf"}'

# Delete all CSV chunks
semsearch delete --filter '{"doc_type": "csv"}'
```

### Delete by Pattern

```bash
# Delete temporary files
semsearch delete --filter '{"source": {"$ilike": "%tmp%"}}'

# Delete backup files
semsearch delete --filter '{"source": {"$ilike": "%.bak"}}'
```

## Prune with ingest-dir

Use `--prune` to delete chunks from files that no longer exist on disk:

```bash
# Preview what would be pruned
semsearch ingest-dir docs/ --prune --dry-run

# Actually prune
semsearch ingest-dir docs/ --prune
```

## Query Syntax Tips

### Escaping Quotes

Use single quotes for the outer shell, double quotes inside:

```bash
# Correct
semsearch search 'query' --filter '{"doc_type": "pdf"}'

# Also correct
semsearch search "query" --filter '{"doc_type": "pdf"}'
```

### Complex Filters

For complex filters, save to a file and use variable:

```bash
# Save filter to file
echo '{"$and": [{"doc_type": "pdf"}, {"source": {"$ilike": "docs/%"}}, {"page": {"$gt": 5}}]}' > filter.json

# Use variable
FILTER='{"$and": [{"doc_type": "pdf"}, {"source": {"$ilike": "docs/%"}}, {"page": {"$gt": 5}}]}'
semsearch search "query" --filter "$FILTER"
```

### Multi-line Filters

Use jq to format complex filters:

```bash
# Create filter with jq
FILTER=$(jq -n '{
  "$and": [
    {"doc_type": "pdf"},
    {"source": {"$ilike": "docs/%"}},
    {"page": {"$gt": 5}}
  ]
}')

semsearch search "query" --filter "$FILTER"
```

## Troubleshooting

### "Invalid filter" error

Check your JSON syntax:

```bash
# Wrong: missing quotes
semsearch search 'query' --filter '{doc_type: pdf}'

# Correct
semsearch search 'query' --filter '{"doc_type": "pdf"}'
```

### No results with filter

Verify the filter matches existing data:

```bash
# Check what's in the database
semsearch stats

# Try broader filter
semsearch search "query" --filter '{"doc_type": "text"}'
```

### Filter not working as expected

Check field values in metadata:

```bash
# Search without filter first
semsearch search "query" --k 1

# Look at the metadata in results
# The "source", "doc_type", "page", "row" fields are available
```

### Special characters in paths

URL-encode or use `$ilike` with wildcards:

```bash
# Files with spaces (use $ilike)
semsearch search "notes" --filter '{"source": {"$ilike": "%my file%"}}'

# Files with special characters
semsearch search "data" --filter '{"source": {"$ilike": "%data (1)%"}}'
```

## Quick Reference

| Task | Filter |
|------|--------|
| Specific file | `{"source": "path/to/file.md"}` |
| File type | `{"doc_type": "pdf"}` |
| Directory | `{"source": {"$ilike": "docs/%"}}` |
| Multiple types | `{"doc_type": {"$in": ["pdf", "csv"]}}` |
| Exclude type | `{"doc_type": {"$ne": "json"}}` |
| Page range | `{"page": {"$between": [1, 10]}}` |
| Combine AND | `{"$and": [{...}, {...}]}` |
| Combine OR | `{"$or": [{...}, {...}]}` |
| Negate | `{"$not": {...}}` |
| Field exists | `{"page": {"$exists": true}}` |
