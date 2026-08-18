# TASK-015: Unit Tests — Loaders & Embeddings

> **Phase**: 10.1-10.2 | **Priority**: High | **Status**: Not Started
> **Depends on**: TASK-004, TASK-006
> **Blocks**: TASK-022

## Objective

Write unit tests for loader dispatch and embedding factory (no DB required).

## Files to Create

### 1. `tests/test_loaders.py`

Tests U-1 through U-4:

```python
import pytest
from pathlib import Path
from semsearch.loaders import pick_loader, with_doc_type

class TestPickLoader:
    def test_txt_loader(self):  # U-1
        """pick_loader dispatches .txt to TextLoader"""
        loader = pick_loader(Path("test.txt"))
        assert callable(loader)

    def test_pdf_loader(self):  # U-2
        """pick_loader dispatches .pdf to PyMuPDFLoader"""
        loader = pick_loader(Path("test.pdf"))
        assert callable(loader)

    def test_rejects_docx(self):  # U-3
        """pick_loader raises ValueError for unsupported extension"""
        with pytest.raises(ValueError, match="unsupported"):
            pick_loader(Path("test.docx"))

    def test_rejects_missing_file(self):  # U-4
        """pick_loader raises FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            pick_loader(Path("/nonexistent/file.txt"))

class TestWithDocType:
    def test_injects_source_and_doc_type(self):
        """with_doc_type adds source and doc_type to metadata"""
        from langchain_core.documents import Document
        docs = [Document(page_content="test")]
        result = with_doc_type(docs, Path("test.txt"))
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["doc_type"] == "text"

    def test_preserves_existing_metadata(self):
        """with_doc_type preserves existing metadata"""
        from langchain_core.documents import Document
        docs = [Document(page_content="test", metadata={"page": 5})]
        result = with_doc_type(docs, Path("test.pdf"))
        assert result[0].metadata["page"] == 5
        assert result[0].metadata["doc_type"] == "pdf"
```

### 2. `tests/test_embeddings.py`

Tests U-5 through U-9:

```python
import pytest
from unittest.mock import patch, MagicMock
from semsearch.config import Settings, EmbeddingProviderConfig
from semsearch.embeddings import build_embedder, _build_openrouter_routing
from semsearch.errors import ProviderConfigError

class TestBuildEmbedder:
    def test_openai_without_key_raises(self):  # U-5
        """OpenAI provider without API key raises ProviderConfigError"""
        settings = Settings(
            embedding_provider=EmbeddingProviderConfig(
                type="openai",
                model="text-embedding-3-small",
            )
        )
        with pytest.raises(ProviderConfigError, match="api_key"):
            build_embedder(settings)

    def test_ollama_unreachable_raises_on_embed(self):  # U-7
        """Ollama with no server raises on first embed call"""
        settings = Settings(
            embedding_provider=EmbeddingProviderConfig(
                type="ollama",
                model="nomic-embed-text",
                base_url="http://localhost:19999",  # Non-existent
            )
        )
        embedder = build_embedder(settings)
        with pytest.raises(Exception):  # Connection refused
            embedder.embed_query("test")

class TestOpenRouterRouting:
    def test_routing_uses_model_kwargs(self):  # U-8
        """OpenRouter routing uses model_kwargs['extra_body'], NOT direct extra_body="""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="openai/text-embedding-3-small",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        assert "extra_body" not in routing  # _build returns inner dict
        # The caller wraps: model_kwargs={"extra_body": routing}
        assert routing == {"provider": {"order": ["openai", "together"]}}

    def test_routing_ignored_for_non_openrouter(self):  # U-9
        """OpenRouter routing fields ignored for non-openrouter types"""
        cfg = EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            provider_order=["OpenAI", "Together"],
        )
        routing = _build_openrouter_routing(cfg)
        assert routing == {}  # Empty for non-openrouter

    def test_allow_fallbacks_none_omitted(self):
        """allow_fallbacks=None is OMITTED, not emitted as True"""
        cfg = EmbeddingProviderConfig(
            type="openrouter",
            model="test",
            provider_allow_fallbacks=None,
        )
        routing = _build_openrouter_routing(cfg)
        if "provider" in routing:
            assert "allow_fallbacks" not in routing["provider"]

    def test_slugs_lowercased(self):
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
```

## Critical Notes

1. **U-8 is the most important test** — Verifies the `model_kwargs` vs `extra_body` distinction
2. **Mock `OpenAIEmbeddings` for U-8** — Don't actually call the API
3. **Ollama test (U-7) requires no server** — Connection refused is the expected behavior

## Verification

- [ ] All 9 unit tests pass
- [ ] No DB required (pure unit tests)
- [ ] U-8 specifically verifies `model_kwargs` nesting
- [ ] `allow_fallbacks=None` test confirms omission behavior
