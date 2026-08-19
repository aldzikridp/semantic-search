# semsearch Documentation

Semantic search over local documents using LangChain + PostgreSQL + pgvector.

## Overview

semsearch is a Python CLI tool that ingests documents (PDF, CSV, JSON, TXT, MD), splits them into chunks, generates embeddings via API providers, and stores them in PostgreSQL with pgvector for similarity search.

## Features

- **Multiple file types**: PDF, CSV, JSON, TXT, Markdown
- **API-based embeddings**: OpenAI, OpenRouter, OpenAI-compatible, Ollama
- **Idempotent ingest**: Re-ingesting unchanged files makes zero API calls
- **Batch operations**: Ingest entire directories with glob/exclude filters
- **Prune support**: Clean up orphaned chunks from deleted files
- **Filter search**: Search by source, doc_type, or custom metadata
- **JSON output**: All commands output structured JSON

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, setup, first steps |
| [Configuration](configuration.md) | Environment variables, providers |
| [CLI Reference](cli-reference.md) | All commands with examples |
| [Architecture](architecture.md) | Code structure, design decisions |
| [API Reference](api-reference.md) | Python API for programmatic use |
| [Database](database.md) | Schema, setup, migrations |
| [Providers](providers.md) | Embedding provider details |
| [Development](development.md) | Contributing, testing, debugging |
| [Filter Guide](filter-guide.md) | Search and delete filters |

## Quick Links

- **First time?** → [Getting Started](getting-started.md)
- **Need to configure?** → [Configuration](configuration.md)
- **CLI usage?** → [CLI Reference](cli-reference.md)
- **Contributing?** → [Development](development.md)

| [Filter Guide](filter-guide.md) | Search and delete filters |
