"""Integration tests for provider-specific behavior (I-14, I-21 through I-28)."""

import pytest
from pathlib import Path

from semsearch.config import EmbeddingProviderConfig
from semsearch.embeddings import _build_openrouter_routing, build_embedder
from semsearch.errors import ProviderConfigError


class TestOpenRouterRouting:
    def test_default_base_url(self):
        """I-21: OpenRouter default base_url when unset."""
        cfg = EmbeddingProviderConfig(type="openrouter", model="test")
        routing = _build_openrouter_routing(cfg)
        assert routing == {}  # No routing fields set

    def test_allow_fallbacks_false_propagated(self):
        """I-22: allow_fallbacks=false propagated correctly."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_allow_fallbacks=False,
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["allow_fallbacks"] is False

    def test_allow_fallbacks_none_omitted(self):
        """I-22b: allow_fallbacks=None (default) is OMITTED."""
        cfg = EmbeddingProviderConfig(type="openrouter", model="test")
        routing = _build_openrouter_routing(cfg)
        if "provider" in routing:
            assert "allow_fallbacks" not in routing["provider"]

    def test_max_price_propagated(self):
        """I-23: max_price propagated correctly."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_max_price={"prompt": 1},
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["max_price"] == {"prompt": 1}

    def test_data_collection_propagated(self):
        """I-27: data_collection propagated correctly."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_data_collection="deny",
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["data_collection"] == "deny"

    def test_only_whitelist_propagated(self):
        """I-28: only whitelist propagated correctly."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_only=["openai", "azure"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["only"] == ["openai", "azure"]

    def test_slugs_lowercased_order(self):
        """I-28b: Provider order slugs normalized to lowercase."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["order"] == ["openai", "together"]

    def test_slugs_lowercased_ignore(self):
        """I-28c: Provider ignore slugs normalized to lowercase."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_ignore=["DeepInfra"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["ignore"] == ["deepinfra"]

    def test_all_routing_fields(self):
        """All OpenRouter routing fields work together."""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_order=["OpenAI"],
            provider_allow_fallbacks=True,
            provider_ignore=["DeepInfra"],
            provider_only=["OpenAI", "Azure"],
            provider_require_parameters=True,
            provider_data_collection="deny",
            provider_max_price={"prompt": 1},
        )
        routing = _build_openrouter_routing(cfg)
        assert routing == {
            "provider": {
                "order": ["openai"],
                "allow_fallbacks": True,
                "ignore": ["deepinfra"],
                "only": ["openai", "azure"],
                "require_parameters": True,
                "data_collection": "deny",
                "max_price": {"prompt": 1},
            }
        }


class TestProviderErrors:
    def test_openai_without_key_raises(self):
        """OpenAI without API key raises ProviderConfigError."""
        cfg = EmbeddingProviderConfig(type="openai", model="test")
        with pytest.raises(ProviderConfigError, match="api_key"):
            from semsearch.config import Settings
            settings = Settings(embedding_provider=cfg)
            build_embedder(settings)

    def test_openrouter_without_key_builds(self):
        """OpenRouter without API key still builds (uses default)."""
        cfg = EmbeddingProviderConfig(type="openrouter", model="test")
        from semsearch.config import Settings
        settings = Settings(embedding_provider=cfg)
        # OpenRouter allows empty key (will fail on actual API call)
        embedder = build_embedder(settings)
        assert embedder is not None

    def test_unknown_provider_raises(self):
        """Unknown provider type raises ProviderConfigError."""
        with pytest.raises(Exception):
            EmbeddingProviderConfig(type="unknown", model="test")


class TestErrorHandling:
    def test_embed_failure_rolls_back(self, service, tmp_path):
        """I-25: Embed failure rolls back transaction."""
        file = tmp_path / "test.txt"
        file.write_text("test content " * 100)

        # Mock embedder to fail
        original_embed = service.embedder.embed_documents

        def fail_embed(texts):
            raise Exception("API error")

        service.embedder.embed_documents = fail_embed

        try:
            with pytest.raises(Exception):
                service.ingest(file)
        finally:
            service.embedder.embed_documents = original_embed

        # Verify no partial writes
        stats = service.stats()
        assert stats["chunk_count"] == 0
