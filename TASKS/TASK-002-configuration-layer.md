# TASK-002: Configuration Layer

> **Phase**: 2 | **Priority**: Critical | **Status**: Not Started
> **Depends on**: TASK-001
> **Blocks**: TASK-006, TASK-007, TASK-008, TASK-009

## Objective

Implement `config.py` with typed environment variable loading using pydantic-settings, and `errors.py` with the exception hierarchy.

## Files to Create

### 1. `src/semsearch/errors.py`

Exception hierarchy per SPEC §7.7:

```python
class SemSearchError(Exception):
    """Base class for all semsearch errors."""

class FileIngestError(SemSearchError): ...
class SearchError(SemSearchError): ...
class DeleteError(SemSearchError): ...
class SchemaMismatchError(SemSearchError): ...
class ProviderConfigError(SemSearchError): ...
```

### 2. `src/semsearch/config.py`

#### Constants

```python
import re
_TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
```

#### `EmbeddingProviderConfig(BaseModel)`

Fields per SPEC §6.1:
- `type: Literal["openai", "huggingface", "ollama", "openai_compatible", "openrouter"]`
- `model: str`
- `api_key: SecretStr | None = None`
- `base_url: str | None = None`
- `device: str = "cpu"`
- `provider_order: list[str] | None = None`
- `provider_allow_fallbacks: bool | None = None` — **MUST default to None, not True**
- `provider_ignore: list[str] | None = None`
- `provider_only: list[str] | None = None`
- `provider_require_parameters: bool = False`
- `provider_data_collection: Literal["allow", "deny"] | None = None`
- `provider_max_price: dict | None = None`

#### `Settings(BaseSettings)`

```python
model_config = SettingsConfigDict(
    env_prefix="SEMSEARCH_",
    env_file=".env",
    extra="ignore",
    env_nested_delimiter="__",  # CRITICAL: double underscore
)
```

Fields:
- `database_url: str` — default connection string
- `collection_name: str` — default `"semsearch_chunks"`, validated by `_TABLE_NAME_RE`
- `embedding_provider: EmbeddingProviderConfig` — default: HuggingFace with `all-MiniLM-L6-v2`
- `chunk_size: int = 1000`
- `chunk_overlap: int = 200`
- `default_k: int = 5`
- `recreate_collection_on_init: bool = False`

#### Validator for `collection_name`

```python
@field_validator("collection_name")
@classmethod
def _validate_table_name(cls, v: str) -> str:
    if not _TABLE_NAME_RE.match(v):
        raise ValueError(...)
    return v
```

## Critical Gotchas

1. **`env_nested_delimiter="__"`** — This allows `SEMSEARCH_EMBEDDING_PROVIDER__TYPE` to map to `settings.embedding_provider.type`. A single underscore (`SEMSEARCH_EMBEDDING_PROVIDER_TYPE`) will NOT work.

2. **`provider_allow_fallbacks` defaults to `None`** — NOT `True`. When None, the OpenRouter routing builder must OMIT the field entirely (not emit `{"allow_fallbacks": True}`).

3. **`SecretStr` for API keys** — Prevents leakage in `repr()` and logs.

4. **`extra="ignore"`** — Prevents errors from unexpected env vars in `.env`.

## Verification

- [ ] `Settings()` loads from `.env` correctly
- [ ] `SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter` sets `settings.embedding_provider.type`
- [ ] Single underscore `SEMSEARCH_EMBEDDING_PROVIDER_TYPE` does NOT set the nested field
- [ ] `collection_name` validation rejects `"DROP TABLE;"`
- [ ] `SecretStr` fields don't leak in `repr(settings)`
- [ ] All default values match spec
