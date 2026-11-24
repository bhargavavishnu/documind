"""Tests for DocuMind FAISS RAG pipeline, prompt templates, and mocked OpenAI LLM calls."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from documind.config import Settings
from documind.prompts import get_qa_prompt
from documind.rag import DocuMindPipeline


class DeterministicHashEmbeddings(Embeddings):
    """Fast deterministic n-gram embeddings for offline unit testing."""

    dim: int = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = digest[0] % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_zero_and_few_shot_prompt_templates() -> None:
    zero = get_qa_prompt("zero_shot")
    few = get_qa_prompt("few_shot")
    zero_msgs = zero.format_messages(context="ctx", question="q")
    few_msgs = few.format_messages(context="ctx", question="q")
    assert len(few_msgs) > len(zero_msgs)
    assert "I cannot find enough information" in zero_msgs[0].content


def test_pipeline_with_mocked_llm_and_citations() -> None:
    captured_inputs: list[Any] = []

    def fake_llm(prompt_val: Any) -> AIMessage:
        captured_inputs.append(prompt_val)
        return AIMessage(
            content="Refresh tokens remain valid for 14 days [Source: auth.txt, p.1]."
        )

    settings = Settings(provider_mode="local", chunk_size=300, chunk_overlap=50, top_k=2)
    pipeline = DocuMindPipeline(
        settings=settings,
        embeddings=DeterministicHashEmbeddings(),
        chat_model=RunnableLambda(fake_llm),
    )
    n_chunks = pipeline.ingest_bytes(
        [
            (
                "auth.txt",
                b"OAuth 2.1 enforces PKCE S256 and refresh tokens remain valid for 14 days.",
            ),
            (
                "billing.txt",
                b"Idle GPU nodes scale to zero after 12 minutes of inactivity.",
            ),
        ]
    )
    assert n_chunks == 2

    resp = pipeline.ask("How long do refresh tokens remain valid?", prompt_mode="few_shot")
    assert "14 days" in resp.answer
    assert "[Source: auth.txt, p.1]" in resp.answer
    assert len(resp.citations) == 2
    assert len(captured_inputs) == 1

    summary = pipeline.summarize(chunks_per_batch=1)
    assert summary.total_pages == 2
    assert len(summary.section_summaries) == 2


def test_local_fallback_pipeline_refuses_unrelated_query() -> None:
    settings = Settings(provider_mode="local", chunk_size=300, chunk_overlap=50, top_k=2)
    pipeline = DocuMindPipeline(
        settings=settings,
        embeddings=DeterministicHashEmbeddings(),
    )
    pipeline.ingest_bytes(
        [("security.txt", b"All databases encrypt customer payloads with AES-256-GCM.")]
    )
    resp = pipeline.ask("What pizza toppings are served on Fridays?")
    assert "cannot find enough information" in resp.answer.lower()
