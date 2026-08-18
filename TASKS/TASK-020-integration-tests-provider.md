# TASK-020: Integration Tests — Provider & Edge Cases

> **Phase**: 10.7 | **Priority**: Medium | **Status**: Not Started
> **Depends on**: TASK-008, TASK-006
> **Blocks**: TASK-022

## Objective

Write integration tests for provider-specific behavior, OpenRouter routing, and error handling.

## File to Create

### `tests/test_service_provider.py`

```python
import pytest
from unittest.mock import patch, MagicMock

class TestProviderSwap:
    def test_provider_swap(self, settings):  # I-14
        """Switching providers requires only config change (after --recreate if dims differ)"""
        # Start with HuggingFace (384 dim)
        # Ingest, search
        # Switch to OpenAI (1536 dim) - would need --recreate
        # Verify both work with their respective dims
        pass

class TestOpenRouter:
    def test_default_base_url(self):  # I-21
        """OpenRouter default base_url when unset"""
        cfg = EmbeddingProviderConfig(type="openrouter", model="test")
        routing = _build_openrouter_routing(cfg)
        # Verify base_url defaults to https://openrouter.ai/api/v1
        pass

    def test_allow_fallbacks_false_propagated(self):  # I-22
        """allow_fallbacks=false propagated correctly"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_allow_fallbacks=False,
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["allow_fallbacks"] == False

    def test_allow_fallbacks_none_omitted(self):  # I-22b
        """allow_fallbacks=None (default) is OMITTED"""
        cfg = EmbeddingProviderConfig(type="openrouter", model="test")
        routing = _build_openrouter_routing(cfg)
        if "provider" in routing:
            assert "allow_fallbacks" not in routing["provider"]

    def test_max_price_propagated(self):  # I-23
        """max_price propagated correctly"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_max_price={"prompt": 1},
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["max_price"] == {"prompt": 1}

    def test_data_collection_propagated(self):  # I-27
        """data_collection propagated correctly"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_data_collection="deny",
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["data_collection"] == "deny"

    def test_only_whitelist_propagated(self):  # I-28
        """only whitelist propagated correctly"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_only=["openai", "azure"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["only"] == ["openai", "azure"]

    def test_slugs_lowercased_order(self):  # I-28b
        """Provider order slugs normalized to lowercase"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["order"] == ["openai", "together"]

    def test_slugs_lowercased_ignore(self):  # I-28c
        """Provider ignore slugs normalized to lowercase"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_ignore=["DeepInfra"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing["provider"]["ignore"] == ["deepinfra"]

class TestOpenAICompatible:
    def test_custom_base_url(self):  # I-24
        """openai_compatible hits custom base_url"""
        # Mock HTTP requests
        # Verify requests go to custom URL, not api.openai.com
        pass

class TestErrorHandling:
    def test_openrouter_down_propagates_error(self, service):  # I-25
        """OpenRouter down → FileIngestError, transaction rolled back"""
        with patch.object(service.embedder, "embed_documents", side_effect=Exception("API error")):
            with pytest.raises(Exception):
                service.ingest(Path("test.txt"))
        # Verify no DB writes (transaction rolled back)
        stats = service.stats()
        assert stats["chunk_count"] == 0

class TestCLIProviderOverride:
    def test_provider_flag_overrides_env(self):  # I-26
        """--provider flag overrides env setting"""
        # Use CliRunner with --provider openrouter
        # Verify embedder built with openrouter, not huggingface
        pass
```

## Critical Notes

1. **Provider swap test (I-14)** — May require --recreate if dimensions differ
2. **OpenRouter tests** — Pure unit tests on `_build_openrouter_routing`
3. **Error propagation test** — Verify transaction rollback on failure
4. **CLI override test** — Use CliRunner

## Verification

- [ ] All provider tests pass
- [ ] OpenRouter routing fields correctly propagated
- [ ] Slugs normalized to lowercase
- [ ] Errors propagate correctly
- [ ] CLI overrides work
