"""Provider-dispatch factory for LangChain Embeddings (spec §7.2)."""

from __future__ import annotations

from pydantic import SecretStr

from semsearch.config import EmbeddingProviderConfig, Settings
from semsearch.errors import ProviderConfigError


def build_embedder(settings: Settings) -> "Embeddings":
    """Return a LangChain Embeddings instance based on settings.embedding_provider.

    Single-provider design: no runtime cascade across providers. Fallback only
    happens INSIDE OpenRouter (via provider.order + provider.allow_fallbacks).
    If the active provider is unreachable, embed calls raise and the caller
    must catch / propagate.

    Raises:
        ProviderConfigError: required credentials missing for the selected type.
        ImportError: if the corresponding langchain-* package is not installed.
    """
    cfg = settings.embedding_provider

    if cfg.type == "openai":
        _require(cfg.api_key, "openai")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=cfg.api_key.get_secret_value(),
            model=cfg.model,
        )

    if cfg.type == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=cfg.base_url or "http://localhost:11434",
            model=cfg.model,
        )

    if cfg.type in ("openai_compatible", "openrouter"):
        from langchain_openai import OpenAIEmbeddings

        if cfg.type == "openrouter":
            base_url = cfg.base_url or "https://openrouter.ai/api/v1"
            # CRITICAL: OpenAIEmbeddings does NOT accept `extra_body` as a direct kwarg.
            # It accepts `model_kwargs: dict[str, Any]` which is unpacked into the
            # underlying `client.create(**model_kwargs)` call. The OpenAI SDK accepts
            # `extra_body` as a top-level kwarg on `.create()`, so we nest it:
            #     model_kwargs = {"extra_body": {"provider": {...}}}
            extra_body = _build_openrouter_routing(cfg)
            model_kwargs: dict = {"extra_body": extra_body} if extra_body else {}
        else:
            base_url = cfg.base_url or "http://localhost:1234/v1"
            model_kwargs = {}

        return OpenAIEmbeddings(
            api_key=(cfg.api_key or SecretStr("not-needed")).get_secret_value(),
            base_url=base_url,
            model=cfg.model,
            model_kwargs=model_kwargs,
        )

    raise ProviderConfigError(f"unknown embedding provider type: {cfg.type}")


def _build_openrouter_routing(cfg: EmbeddingProviderConfig) -> dict:
    """Build the OpenRouter ``extra_body`` dict (nested under ``model_kwargs``).

    Returns the inner dict — caller wraps it as
    ``model_kwargs={"extra_body": <this>}``.

    Field mapping (spec → OpenRouter):
        provider_order              → provider.order
        provider_allow_fallbacks   → provider.allow_fallbacks (ONLY if set; None = omit)
        provider_ignore            → provider.ignore
        provider_only              → provider.only
        provider_require_parameters→ provider.require_parameters
        provider_data_collection   → provider.data_collection
        provider_max_price         → provider.max_price

    Provider slugs are LOWERCASED before sending.

    Returns ``{}`` when no routing fields are set.
    """
    provider_body: dict = {}

    if cfg.provider_order is not None:
        provider_body["order"] = [s.lower() for s in cfg.provider_order]

    # Only emit allow_fallbacks if the user explicitly set it (None = OR default).
    if cfg.provider_allow_fallbacks is not None:
        provider_body["allow_fallbacks"] = cfg.provider_allow_fallbacks

    if cfg.provider_ignore is not None:
        provider_body["ignore"] = [s.lower() for s in cfg.provider_ignore]

    if cfg.provider_only is not None:
        provider_body["only"] = [s.lower() for s in cfg.provider_only]

    if cfg.provider_require_parameters:
        provider_body["require_parameters"] = True

    if cfg.provider_data_collection is not None:
        provider_body["data_collection"] = cfg.provider_data_collection

    if cfg.provider_max_price is not None:
        provider_body["max_price"] = cfg.provider_max_price

    return {"provider": provider_body} if provider_body else {}


def _require(secret: SecretStr | None, name: str) -> None:
    """Raise ProviderConfigError if secret is None or empty."""
    if not secret or not secret.get_secret_value():
        raise ProviderConfigError(f"provider {name!r} requires api_key")
