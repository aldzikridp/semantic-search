# Embedding Providers

semsearch supports multiple embedding providers via lazy-loaded LangChain integrations.

## Supported Providers

| Provider | Type | API Key | Default Model | Dimensions |
|----------|------|---------|---------------|------------|
| OpenAI | `openai` | Required | text-embedding-3-small | 1536 |
| OpenRouter | `openrouter` | Required | — | Varies |
| OpenAI-compatible | `openai_compatible` | Optional | — | Varies |
| Ollama | `ollama` | No | — | Varies |

## Configuration

All providers use the same configuration pattern:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

## OpenAI

Standard OpenAI embeddings API.

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

**Models:**

| Model | Dimensions | Price |
|-------|------------|-------|
| text-embedding-3-small | 1536 | $0.02/1M tokens |
| text-embedding-3-large | 3072 | $0.13/1M tokens |
| text-embedding-ada-002 | 1536 | $0.10/1M tokens |

**Installation:**

```bash
pip install -e ".[openai]"
```

## OpenRouter

OpenRouter provides access to multiple embedding providers with routing and fallbacks.

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-or-v1-...
```

**Default base URL:** `https://openrouter.ai/api/v1`

### Routing Options

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

**Installation:**

```bash
pip install -e ".[openai]"
```

**Models:**

OpenRouter supports any model from its providers:

```bash
# OpenAI models
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small

# Qwen models
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=qwen/qwen3-embedding-8b

# Other providers
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=deepinfra/bge-small-en-v1.5
```

## OpenAI-compatible

For any OpenAI-compatible endpoint (LM Studio, vLLM, Ollama's OpenAI shim, etc.).

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openai_compatible
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=bge-small-en-v1.5
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:1234/v1
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=not-needed-but-required
```

**Common endpoints:**

| Service | Default URL |
|---------|-------------|
| LM Studio | http://localhost:1234/v1 |
| vLLM | http://localhost:8000/v1 |
| Ollama (OpenAI shim) | http://localhost:11434/v1 |

**Installation:**

```bash
pip install -e ".[openai]"
```

## Ollama

Local embedding models via Ollama daemon.

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=ollama
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=nomic-embed-text
SEMSEARCH_EMBEDDING_PROVIDER__BASE_URL=http://localhost:11434
```

**Default base URL:** `http://localhost:11434`

**Popular models:**

| Model | Dimensions | Size |
|-------|------------|------|
| nomic-embed-text | 768 | 274MB |
| all-minilm | 384 | 46MB |
| bge-small | 384 | 133MB |
| mxbai-embed-large | 1024 | 670MB |

**Installation:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull nomic-embed-text

# Install Python package
pip install -e ".[ollama]"
```

## Provider Comparison

| Provider | Local | Free | Quality | Speed |
|----------|-------|------|---------|-------|
| OpenAI | No | No | High | Fast |
| OpenRouter | No | Varies | Varies | Fast |
| OpenAI-compatible | Depends | Depends | Varies | Varies |
| Ollama | Yes | Yes | Good | Slow |

## Choosing a Provider

**For development:**
- Ollama (free, local, no API key)
- OpenAI text-embedding-3-small (cheap, fast)

**For production:**
- OpenAI text-embedding-3-small (good balance)
- OpenAI text-embedding-3-large (best quality)
- OpenRouter (flexibility, fallbacks)

**For cost optimization:**
- OpenRouter with `provider_max_price`
- Ollama (free)
- OpenAI-compatible with local models

## Troubleshooting

### "provider 'openai' requires api_key"

Set the API key:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__API_KEY=sk-...
```

### "ImportError: No module named 'langchain_openai'"

Install the provider package:

```bash
pip install -e ".[openai]"
```

### "Connection refused" (Ollama)

Start the Ollama daemon:

```bash
ollama serve
```

### Slow embeddings (Ollama)

Use a smaller model:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=all-minilm
```

### High costs (OpenAI)

Use a cheaper model:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=text-embedding-3-small
```

Or use OpenRouter with price cap:

```bash
SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
SEMSEARCH_EMBEDDING_PROVIDER__MODEL=openai/text-embedding-3-small
SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_MAX_PRICE={"prompt":0.05}
```

## Reranker

Reranking improves search precision by re-scoring results with a cross-encoder model. The reranker uses a generic HTTP API compatible with OpenRouter, Jina, and other providers.

### Configuration

```bash
# OpenRouter (reuse existing API key)
SEMSEARCH_RERANKER__BASE_URL=https://openrouter.ai/api/v1/rerank
SEMSEARCH_RERANKER__MODEL=cohere/rerank-v3.5

# Jina (separate API key, free tier: 1M tokens/month)
SEMSEARCH_RERANKER__BASE_URL=https://api.jina.ai/v1/rerank
SEMSEARCH_RERANKER__MODEL=jina-reranker-v2-base-multilingual
SEMSEARCH_RERANKER__API_KEY=jina_...
```

### Available Models

| Provider | Model | Quality | Price |
|----------|-------|---------|-------|
| OpenRouter | `cohere/rerank-v3.5` | Excellent | Pay-per-use |
| OpenRouter | `cohere/rerank-english-v3.0` | Good | Pay-per-use |
| Jina | `jina-reranker-v2-base-multilingual` | Good | 1M tokens/month free |

### Usage

```bash
semsearch search "query" --rerank --k 5
```

### How it works

1. Vector search retrieves `k * 4` candidates
2. Reranker re-scores candidates against the query
3. Top-k results returned with `rerank_score` in metadata
4. File names and metadata are preserved
