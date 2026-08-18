# Configuration

semsearch is configured via environment variables or a `.env` file.

## Environment Variables

All variables use the `SEMSEARCH_` prefix. Nested fields use **double underscore** (`__`).

### Database

```bash
# PostgreSQL connection string
SEMSEARCH_DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Table name (must match /^[a-z_][a-z0-9_]{0,62}$/)
SEMSEARCH_COLLECTION_NAME=semsearch_chunks
```

### Embedding Provider

```bash
# Provider type
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai

# Model name
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small

# API key (required for openai, openrouter)
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# Base URL (required for openai_compatible, optional for ollama)
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434
```

### Chunking

```bash
# Chunk size in characters (default: 1000)
SEMSEARCH_CHUNK_SIZE=1000

# Overlap between chunks (default: 200)
SEMSEARCH_CHUNK_OVERLAP=200

# Default k for search (default: 5)
SEMSEARCH_DEFAULT_K=5
```

## Provider Configuration

### OpenAI

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### OpenRouter

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-or-v1-...

# Optional routing
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["deepinfra","together"]
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["bad-provider"]
```

### Ollama

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=nomic-embed-text
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434
```

### OpenAI-compatible

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai_compatible
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=bge-small-en-v1.5
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:1234/v1
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=not-needed-but-required
```

## OpenRouter Routing

OpenRouter supports advanced routing options:

```bash
# Provider order (lowercase slugs)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER=["deepinfra","together"]

# Allow fallbacks (default: null = OpenRouter default)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ALLOW_FALLBACKS=true

# Ignore specific providers
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_IGNORE=["deepseek"]

# Only use specific providers
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ONLY=["deepinfra","azure"]

# Data collection policy
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_DATA_COLLECTION=deny

# Max price (USD per 1M tokens)
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_MAX_PRICE={"prompt":1}
```

## .env.example

```bash
# Database
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch
SEMSEARCH_COLLECTION_NAME=semsearch_chunks

# Provider (pick one)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...

# Chunking
SEMSEARCH_CHUNK_SIZE=1000
SEMSEARCH_CHUNK_OVERLAP=200
SEMSEARCH_DEFAULT_K=5
```

## Environment-specific Config

### Development

```bash
# Use local PostgreSQL
SEMSEARCH_DATABASE_URL=postgresql://semsearch:change_me@localhost:5432/semsearch

# Use cheaper/faster model
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
```

### Production

```bash
# Use managed PostgreSQL
SEMSEARCH_DATABASE_URL=postgresql://user:pass@db.example.com:5432/semsearch

# Use higher quality model
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-large
```

### CI/Testing

```bash
# Use testcontainers (no external DB needed)
SEMSEARCH_DATABASE_URL=postgresql://test:test@localhost:5432/test

# Use mock embeddings (no API calls)
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=test-key
```

## Validation

The configuration is validated at startup:

- `collection_name` must match `/^[a-z_][a-z0-9_]{0,62}$/`
- `provider.type` must be one of: `openai`, `ollama`, `openai_compatible`, `openrouter`
- API keys are validated for providers that require them

## Troubleshooting

### "provider 'openai' requires api_key"

Set the API key:
```bash
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### "collection_name must match..."

Use only lowercase letters, numbers, and underscores:
```bash
SEMSEARCH_COLLECTION_NAME=semsearch_chunks  # ✅
SEMSEARCH_COLLECTION_NAME=My-Table          # ❌
```

### "Id column does not exist"

The table was created with a different schema. Recreate:
```bash
semsearch init --recreate --yes
```
