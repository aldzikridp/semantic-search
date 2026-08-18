# TASK-006: Embeddings Factory

> **Phase**: 6 | **Priority**: Critical | **Status**: ✅ Done
> **Depends on**: TASK-002 (config, errors)
> **Blocks**: TASK-007, TASK-008

## Objective

Implement the provider-dispatch factory that creates LangChain `Embeddings` instances based on configuration, including OpenRouter routing translation.

**Note**: HuggingFace local embeddings removed — rely entirely on API providers.

## File to Create

### `src/semsearch/embeddings.py`

## Implementation

### 1. `build_embedder(settings: Settings) -> Embeddings`

Dispatch based on `settings.embedding_provider.type`:

| Type | Class | Required Config | Notes |
|------|-------|-----------------|-------|
| `openai` | `OpenAIEmbeddings` | `api_key` | Standard OpenAI API |
| `ollama` | `OllamaEmbeddings` | Ollama daemon running | Default base_url: `http://localhost:11434` |
| `openrouter` | `OpenAIEmbeddings` | `api_key` | Default base_url: `https://openrouter.ai/api/v1` |
| `openai_compatible` | `OpenAIEmbeddings` | `base_url` | Any OpenAI-compatible endpoint |

### 2. `_build_openrouter_routing(cfg: EmbeddingProviderConfig) -> dict`

Builds the `extra_body` dict for OpenRouter routing. Returns the **inner** dict — caller wraps as `model_kwargs={"extra_body": <this>}`.

**Field mapping** (spec field → OpenRouter API field):
- `provider_order` → `provider.order` (lowercased)
- `provider_allow_fallbacks` → `provider.allow_fallbacks` (**ONLY if explicitly set; None = OMIT**)
- `provider_ignore` → `provider.ignore` (lowercased)
- `provider_only` → `provider.only` (lowercased)
- `provider_require_parameters` → `provider.require_parameters`
- `provider_data_collection` → `provider.data_collection`
- `provider_max_price` → `provider.max_price`

**Returns**: `{"provider": {...}}` or `{}` if no routing fields set.

### 3. `_require(secret: SecretStr | None, name: str) -> None`

Raises `ProviderConfigError` if secret is None or empty.

## Critical Implementation Details

### 1. OpenRouter routing goes through `model_kwargs`

```python
# CORRECT ✅
OpenAIEmbeddings(
    api_key=...,
    base_url="https://openrouter.ai/api/v1",
    model=...,
    model_kwargs={"extra_body": {"provider": {"order": ["openai", "together"]}}},
)

# WRONG ❌ — extra_body is NOT a direct kwarg on OpenAIEmbeddings
OpenAIEmbeddings(
    api_key=...,
    base_url="https://openrouter.ai/api/v1",
    model=...,
    extra_body={"provider": {"order": ["openai", "together"]}},  # DOES NOT EXIST
)
```

### 2. `allow_fallbacks=None` MUST be omitted

```python
# CORRECT ✅ — When allow_fallbacks is None, omit from dict
if cfg.provider_allow_fallbacks is not None:
    provider_body["allow_fallbacks"] = cfg.provider_allow_fallbacks

# WRONG ❌ — This always emits allow_fallbacks=True when user didn't set it
provider_body["allow_fallbacks"] = cfg.provider_allow_fallbacks or True  # DON'T DO THIS
```

### 3. Provider slugs are LOWERCASED

```python
# CORRECT ✅
provider_body["order"] = [s.lower() for s in cfg.provider_order]

# WRONG ❌ — Preserves user's casing which may not match OpenRouter's expectations
provider_body["order"] = cfg.provider_order
```

### 4. Lazy imports

Each provider's langchain package is imported only when needed. This means:
- `langchain-openai` is only imported for `openai`, `openrouter`, `openai_compatible`
- `langchain-ollama` is only imported for `ollama`

If a package isn't installed, the import fails with `ImportError`.

## Verification (Unit Tests U-5 to U-9)

- [ ] U-5: OpenAI without API key raises `ProviderConfigError`
- [ ] U-7: Ollama with no server raises on first embed call (connection refused)
- [ ] U-8: OpenRouter routing uses `model_kwargs["extra_body"]`, NOT direct `extra_body=` kwarg
- [ ] U-9: OpenRouter routing fields ignored for non-openrouter types (`model_kwargs` is `{}`)
- [ ] `allow_fallbacks=None` is OMITTED from output (not emitted as True)
- [ ] Provider slugs are lowercased: `["OpenAI", "Together"]` → `["openai", "together"]`
- [ ] `_build_openrouter_routing` returns `{}` when no routing fields set
