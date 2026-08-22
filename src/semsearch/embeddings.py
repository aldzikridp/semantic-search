"""Provider-dispatch factory for LangChain Embeddings (spec §7.2)."""

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
# pi-lens-ignore: import-not-found
from pydantic import SecretStr

from semsearch.config import EmbeddingProviderConfig, Settings
from semsearch.errors import ProviderConfigError

logger = logging.getLogger(__name__)


def _detect_httpx_version() -> tuple[Any, Any, Any]:
    """Detect which httpx version the OpenAI SDK expects.

    OpenAI SDK v2.x uses httpx, v3.x uses httpx2.

    Returns (httpx_module, timeout_class, limits_class).
    """
    try:
        import openai
        # Check OpenAI SDK version
        major_version = int(openai.__version__.split(".")[0])
        if major_version >= 3:
            import httpx2
            return httpx2, httpx2.Timeout, httpx2.Limits
    except (ImportError, AttributeError, ValueError):
        pass

    # Default to httpx (v2 SDK or unknown)
    import httpx
    return httpx, httpx.Timeout, httpx.Limits


# Detect at module load time
_httpx_module, _Timeout, _Limits = _detect_httpx_version()
logger.debug("Using %s for HTTP client", _httpx_module.__name__)

# Shared client settings
_HTTPX_TIMEOUT = _Timeout(connect=5.0, read=10.0, write=5.0, pool=2.0)
_HTTPX_LIMITS = _Limits(
    max_connections=10,
    max_keepalive_connections=5,
    # 5-minute keep-alive: agent queries arrive seconds-to-minutes apart, so a
    # short expiry forced a TCP+TLS re-handshake (~50–150 ms) on nearly every
    # request. Idle sockets closed cleanly by the provider (LB idle-timeout,
    # typically 60–120 s) are detected at checkout and replaced transparently;
    # half-open connections are covered by the OpenAI SDK's max_retries=2.
    keepalive_expiry=300.0,
)


def _make_openai_clients(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
) -> tuple[Any, Any]:
    """Create OpenAI sync and async clients with proper httpx settings.

    Args:
        api_key: OpenAI API key.
        base_url: Optional base URL for OpenAI-compatible endpoints.
        timeout: Request timeout in seconds.

    Returns (sync_client.embeddings, async_client.embeddings) for use with
    OpenAIEmbeddings.
    """
    from openai import OpenAI, AsyncOpenAI

    http_timeout = _Timeout(connect=5.0, read=timeout, write=5.0, pool=2.0)

    http_client = _httpx_module.Client(
        timeout=http_timeout,
        limits=_HTTPX_LIMITS,
    )
    http_async_client = _httpx_module.AsyncClient(
        timeout=http_timeout,
        limits=_HTTPX_LIMITS,
    )

    kwargs = {
        "api_key": api_key,
        "timeout": http_timeout,
        "max_retries": 2,
        "http_client": http_client,
    }
    async_kwargs = {
        "api_key": api_key,
        "timeout": http_timeout,
        "max_retries": 2,
        "http_client": http_async_client,
    }
    if base_url:
        kwargs["base_url"] = base_url
        async_kwargs["base_url"] = base_url

    sync_client = OpenAI(**kwargs)
    async_client = AsyncOpenAI(**async_kwargs)
    return sync_client.embeddings, async_client.embeddings


def build_embedder(settings: Settings) -> Embeddings:
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
    timeout = settings.timeout.embedding

    if cfg.type == "openai":
        api_key = _require(cfg.api_key, "openai")
        from langchain_openai import OpenAIEmbeddings

        sync_client, async_client = _make_openai_clients(
            api_key=api_key.get_secret_value(),
            timeout=timeout,
        )
        return OpenAIEmbeddings(
            client=sync_client,
            async_client=async_client,
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
            model_kwargs: dict[str, Any] = {"extra_body": extra_body} if extra_body else {}
        else:
            base_url = cfg.base_url or "http://localhost:1234/v1"
            model_kwargs = {}

        sync_client, async_client = _make_openai_clients(
            api_key=(cfg.api_key or SecretStr("not-needed")).get_secret_value(),
            base_url=base_url,
            timeout=timeout,
        )
        return OpenAIEmbeddings(
            client=sync_client,
            async_client=async_client,
            model=cfg.model,
            model_kwargs=model_kwargs,
            check_embedding_ctx_length=False,
        )

    raise ProviderConfigError(f"unknown embedding provider type: {cfg.type}")


def _build_openrouter_routing(cfg: EmbeddingProviderConfig) -> dict[str, Any]:
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
    provider_body: dict[str, Any] = {}

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


def _require(secret: SecretStr | None, name: str) -> SecretStr:
    """Return the secret if present and non-empty, else raise ProviderConfigError."""
    if not secret or not secret.get_secret_value():
        raise ProviderConfigError(f"provider {name!r} requires api_key")
    return secret
