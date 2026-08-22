"""Unit tests for embeddings factory (U-5 through U-9)."""

# pi-lens-ignore: import-not-found
import pytest

from semsearch.config import Settings, EmbeddingProviderConfig
from semsearch.embeddings import _build_openrouter_routing, build_embedder
from semsearch.errors import ProviderConfigError


class TestBuildEmbedder:
    def test_openai_without_key_raises(self) -> None:
        """U-5: OpenAI provider without API key raises ProviderConfigError"""
        settings = Settings(
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
            )
        )
        with pytest.raises(ProviderConfigError, match="api_key"):
            build_embedder(settings)

    def test_ollama_builds_embedder(self) -> None:
        """U-7: Ollama builds OllamaEmbeddings (or raises ImportError)"""
        settings = Settings(
            embedding_provider=EmbeddingProviderConfig(
                type="ollama",
                model="nomic-embed-text",
            )
        )
        try:
            embedder = build_embedder(settings)
            # If langchain-ollama is installed, verify the type
            from langchain_ollama import OllamaEmbeddings
            assert isinstance(embedder, OllamaEmbeddings)
        except ImportError:
            # langchain-ollama not installed — acceptable
            pass


class TestOpenRouterRouting:
    def test_routing_uses_model_kwargs(self) -> None:
        """U-8: OpenRouter routing returns inner dict (caller wraps in model_kwargs)"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="openai/text-embedding-3-small",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        # _build returns inner dict, caller wraps as model_kwargs={"extra_body": routing}
        assert "extra_body" not in routing
        assert routing == {"provider": {"order": ["openai", "together"]}}

    def test_routing_fields_present_for_non_openrouter(self) -> None:
        """U-9: _build_openrouter_routing builds dict regardless of type (caller decides)"""
        cfg = EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        # The function builds the dict based on fields; build_embedder decides whether to use it
        assert routing == {"provider": {"order": ["openai", "together"]}}

    def test_allow_fallbacks_none_omitted(self) -> None:
        """allow_fallbacks=None is OMITTED, not emitted as True"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_allow_fallbacks=None,
        )
        routing = _build_openrouter_routing(cfg)
        if "provider" in routing:
            assert "allow_fallbacks" not in routing["provider"]

    def test_slugs_lowercased(self) -> None:
        """Provider slugs normalized to lowercase"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_order=["OpenAI", "Together"],
            provider_ignore=["DeepInfra"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["order"] == ["openai", "together"]
        assert routing["provider"]["ignore"] == ["deepinfra"]


class TestHttpLimits:
    """Phase E: HTTP keep-alive tuned for agent workloads."""

    def test_keepalive_expiry_is_300s(self) -> None:
        """Pooled connections must survive idle gaps of minutes, not seconds.

        With the old 10 s expiry nearly every request re-paid TCP+TLS
        (~50–150 ms). Cleanly-closed idle sockets are still detected and
        replaced transparently at checkout.
        """
        from semsearch.embeddings import _HTTPX_LIMITS

        assert _HTTPX_LIMITS.keepalive_expiry == 300.0
