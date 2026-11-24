"""Document loading (PDF, TXT, Markdown) and overlapping text chunking."""

from __future__ import annotations

import io
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document_bytes(filename: str, content: bytes) -> list[Document]:
    """Load PDF, TXT, or Markdown bytes into page-level LangChain Documents."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{filename}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        total_pages = len(reader.pages)
        docs: list[Document] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": Path(filename).name,
                            "page": idx,
                            "total_pages": total_pages,
                        },
                    )
                )
        return docs

    decoded = content.decode("utf-8", errors="replace").strip()
    if not decoded:
        return []
    return [
        Document(
            page_content=decoded,
            metadata={
                "source": Path(filename).name,
                "page": 1,
                "total_pages": 1,
            },
        )
    ]


def load_document_path(path: Path) -> list[Document]:
    """Read a document from disk into page-level Documents."""
    return load_document_bytes(path.name, path.read_bytes())


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split page-level documents into overlapping chunks with exact citation metadata."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "document")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}:p{page}:c{idx}"
    return chunks
