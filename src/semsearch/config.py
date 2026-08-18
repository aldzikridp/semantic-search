"""Typed configuration via pydantic-settings (spec §6.1)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Validates that collection_name is a safe SQL identifier — it becomes a table
# name in init_vectorstore_table(table_name=...) and is interpolated into raw
# SQL statements (the service-owned write path).
_TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class EmbeddingProviderConfig(BaseModel):
    """Single embedding provider configuration.

    One provider is active at a time. To switch providers, edit
    SEMSEARCH_EMBEDDING_PROVIDER__TYPE in .env — the program does NOT
    cascade across providers at runtime. Fallback only happens inside
    OpenRouter (provider.order + provider.allow_fallbacks), not across.

    Env var mapping uses double-underscore delimiter (see Settings.model_config):
        SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
        SEMSEARCH_EMBEDDING_PROVIDER__PROVIDER_ORDER='["openai","together"]'
    """

    type: Literal["openai", "huggingface", "ollama", "openai_compatible", "openrouter"]
    model: str

    # API config — meaning depends on `type`:
    #   openai             — OpenAI's hosted API (api_key required, base_url optional)
    #   openrouter         — OpenRouter (api_key required, base_url defaults to https://openrouter.ai/api/v1)
    #   openai_compatible  — any OpenAI-compatible endpoint (base_url required, api_key optional)
    #   huggingface        — local SentenceTransformers (api_key ignored)
    #   ollama             — local Ollama daemon (base_url optional, defaults to http://localhost:11434)
    api_key: SecretStr | None = None
    base_url: str | None = None
    device: str = "cpu"  # huggingface only

    # ---- OpenRouter routing — ignored by all other types ----
    # Canonical reference: https://openrouter.ai/docs/guides/routing/provider-selection
    #
    # IMPORTANT: provider slugs are LOWERCASE identifiers, not display names.
    provider_order: list[str] | None = None  # e.g. ["openai", "together"]
    provider_allow_fallbacks: bool | None = None  # None = OpenRouter default (true).
    # Must be None default so _build_openrouter_routing can omit when unset.
    provider_ignore: list[str] | None = None  # provider slugs to skip
    provider_only: list[str] | None = None  # provider slugs to allow (whitelist)
    provider_require_parameters: bool = False
    provider_data_collection: Literal["allow", "deny"] | None = None
    # "deny" = avoid providers that may store data
    provider_max_price: dict | None = None  # e.g. {"prompt": 1} (USD per 1M tokens)
    # For embeddings, only "prompt" is meaningful


class Settings(BaseSettings):
    """Top-level settings loaded from environment variables / .env file.

    Uses double-underscore delimiter for nested fields:
        SEMSEARCH_EMBEDDING_PROVIDER__TYPE=openrouter
        → settings.embedding_provider.type = "openrouter"
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMSEARCH_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ---- Database ----
    database_url: str = Field(
        default="postgresql+psycopg://semsearch_app:change_me_in_prod@localhost:5432/semsearch",
    )
    collection_name: str = Field(default="semsearch_chunks")

    @field_validator("collection_name")
    @classmethod
    def _validate_table_name(cls, v: str) -> str:
        if not _TABLE_NAME_RE.match(v):
            raise ValueError(
                f"collection_name {v!r} must match /^[a-z_][a-z0-9_]{{0,62}}$/ "
                f"(lowercase, alphanumeric + underscore, 1-63 chars, must start with "
                f"a letter or underscore)"
            )
        return v

    # ---- Embedding provider (single, no cascade) ----
    embedding_provider: EmbeddingProviderConfig = Field(
        default=EmbeddingProviderConfig(
            type="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    # ---- Chunking ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Search defaults ----
    default_k: int = 5

    # ---- Lifecycle ----
    recreate_collection_on_init: bool = False  # safety: must be opt-in
