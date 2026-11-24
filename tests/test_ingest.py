"""Tests for PDF and text ingestion and chunk metadata."""

from pathlib import Path

import pytest

from documind.ingest import chunk_documents, load_document_bytes, load_document_path
from documind.sample_data import generate_handbook_pdf


def test_load_text_and_chunk_metadata() -> None:
    content = ("Paragraph one about FAISS vector indexing.\n\n" * 20).encode("utf-8")
    docs = load_document_bytes("notes.md", content)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.md"
    assert docs[0].metadata["page"] == 1

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_id"] == "notes.md:p1:c0"
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_unsupported_extension_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document_bytes("archive.zip", b"123")


def test_generate_and_load_56_page_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "handbook_56p.pdf"
    generate_handbook_pdf(pdf_path, target_pages=56)
    docs = load_document_path(pdf_path)
    assert len(docs) == 56
    assert docs[0].metadata["page"] == 1
    assert docs[-1].metadata["page"] == 56
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)
    assert len(chunks) >= 56
