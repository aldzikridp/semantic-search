"""Unit tests for document loaders (U-1 through U-4)."""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from semsearch.loaders import pick_loader, with_doc_type


class TestPickLoader:
    def test_txt_loader_returns_callable(self, tmp_path):
        """U-1: pick_loader dispatches .txt to TextLoader"""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        loader = pick_loader(f)
        assert callable(loader)

    def test_pdf_loader_returns_callable(self, tmp_path):
        """U-2: pick_loader dispatches .pdf to PyMuPDFLoader"""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
        loader = pick_loader(f)
        assert callable(loader)

    def test_rejects_docx(self, tmp_path):
        """U-3: pick_loader raises ValueError for unsupported extension"""
        f = tmp_path / "test.docx"
        f.write_text("test")
        with pytest.raises(ValueError, match="(?i)unsupported"):
            pick_loader(f)

    def test_rejects_missing_file(self):
        """U-4: pick_loader raises FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            pick_loader(Path("/nonexistent/file.txt"))


class TestWithDocType:
    def test_injects_source_and_doc_type(self):
        """with_doc_type adds source and doc_type to metadata"""
        docs = [Document(page_content="test")]
        result = with_doc_type(docs, Path("test.txt"))
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["doc_type"] == "text"

    def test_preserves_existing_metadata(self):
        """with_doc_type preserves existing metadata (page, row)"""
        docs = [Document(page_content="test", metadata={"page": 5, "row": 3})]
        result = with_doc_type(docs, Path("test.pdf"))
        assert result[0].metadata["page"] == 5
        assert result[0].metadata["row"] == 3
        assert result[0].metadata["doc_type"] == "pdf"

    def test_rejects_unsupported_extension(self):
        """with_doc_type raises ValueError for unsupported extension"""
        docs = [Document(page_content="test")]
        with pytest.raises(ValueError, match="(?i)unsupported"):
            with_doc_type(docs, Path("test.xyz"))
