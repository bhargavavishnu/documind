"""FAISS vector index management and naive keyword (BM25/TF) baseline retriever."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from documind.config import Settings

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for the naive keyword baseline."""
    return _WORD_RE.findall(text.lower())


def build_embeddings(settings: Settings) -> Embeddings:
    """Instantiate OpenAIEmbeddings or local HuggingFace sentence-transformer embeddings."""
    provider = settings.resolve_provider()
    if provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorIndexManager:
    """Manages FAISS vector index creation, persistence, and similarity retrieval."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.store: FAISS | None = None
        self.chunks: list[Document] = []

    def build_from_documents(self, chunks: list[Document]) -> FAISS:
        """Create an in-memory FAISS index from chunked Documents."""
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty document list.")
        self.chunks = list(chunks)
        self.store = FAISS.from_documents(self.chunks, self.embeddings)
        return self.store

    def save(self, directory: Path) -> None:
        """Persist the FAISS index to disk."""
        if self.store is None:
            raise RuntimeError("No FAISS index built yet.")
        directory.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(directory))

    def load(self, directory: Path) -> FAISS:
        """Load a previously saved FAISS index from disk."""
        self.store = FAISS.load_local(
            str(directory),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.store

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return top-k (Document, similarity_or_distance_score) pairs from FAISS."""
        if self.store is None:
            raise RuntimeError("FAISS index is not initialized.")
        return self.store.similarity_search_with_score(query, k=k)


class KeywordRetriever:
    """Naive lexical baseline (BM25 + exact token matching) for RAG evaluation comparison."""

    def __init__(self, chunks: Iterable[Document]) -> None:
        self.chunks = list(chunks)
        corpus = [tokenize(doc.page_content) for doc in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Retrieve top-k documents using lexical keyword overlap / BM25."""
        if not self.chunks or self._bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:k]]
