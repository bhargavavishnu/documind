"""Core DocuMind RAG pipeline: retrieval, grounded Q&A, and 50+ page summarization."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from documind.config import Settings
from documind.ingest import chunk_documents, load_document_bytes, load_document_path
from documind.prompts import MAP_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT, get_qa_prompt
from documind.vectorstore import VectorIndexManager, build_embeddings, tokenize

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "when", "where", "why", "how", "in", "on", "at", "to", "for", "of",
    "and", "or", "with", "by", "from", "as", "do", "does", "did", "can",
    "be", "this", "that", "it", "their", "our", "your",
}


class Citation(BaseModel):
    """Source citation pointing to a specific file, page, and chunk."""

    source: str
    page: int
    chunk_id: str
    score: float
    excerpt: str


class RAGResponse(BaseModel):
    """Structured response returned by DocuMind Q&A."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    provider_used: str
    prompt_mode: str
    latency_ms: float


class SummaryResult(BaseModel):
    """Structured response returned by 50+ page document summarization."""

    executive_summary: str
    section_summaries: list[str]
    total_pages: int
    total_chunks: int
    provider_used: str
    latency_ms: float


def format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks with explicit source/page headers for citation grounding."""
    blocks: list[str] = []
    for doc, _ in docs_with_scores:
        src = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", 1)
        blocks.append(f"[Source: {src}, p.{page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def _local_grounded_answer(
    question: str,
    docs_with_scores: list[tuple[Document, float]],
) -> str:
    """Synthesize a strictly grounded answer from top-ranked retrieved sentences (no LLM)."""
    q_tokens = {t for t in tokenize(question) if t not in _STOPWORDS}
    if not docs_with_scores:
        return "I cannot find enough information in the provided documents to answer this question."

    scored_sentences: list[tuple[float, str, str, int]] = []
    for rank, (doc, dist) in enumerate(docs_with_scores):
        src = str(doc.metadata.get("source", "document"))
        page = int(doc.metadata.get("page", 1))
        # Lower FAISS L2 distance means higher semantic closeness.
        rank_bonus = max(0.2, 1.0 - 0.15 * rank)
        for sent in _SENTENCE_SPLIT_RE.split(doc.page_content.strip()):
            clean = " ".join(sent.split())
            if len(clean) < 20 or clean[0].islower():
                continue
            s_tokens = {t for t in tokenize(clean) if t not in _STOPWORDS}
            overlap = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
            score = rank_bonus * 0.45 + overlap * 0.55 - min(float(dist), 2.0) * 0.05
            scored_sentences.append((score, clean, src, page))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    if not scored_sentences:
        return "I cannot find enough information in the provided documents to answer this question."

    # Guardrail: if query has specific terms and none of the top sentences match meaningfully
    # and FAISS distance is high, abstain rather than hallucinating.
    best_dist = float(docs_with_scores[0][1])
    top_overlap = max(
        len(q_tokens & set(tokenize(s[1]))) / max(len(q_tokens), 1)
        for s in scored_sentences[:3]
    )
    if best_dist > 1.45 and top_overlap < 0.15:
        return "I cannot find enough information in the provided documents to answer this question."

    selected: list[str] = []
    seen_texts: set[str] = set()
    for _, sentence, src, page in scored_sentences:
        if sentence in seen_texts:
            continue
        seen_texts.add(sentence)
        selected.append(f"{sentence} [Source: {src}, p.{page}]")
        if len(selected) >= 3:
            break
    return " ".join(selected)


def _local_batch_summary(chunks: list[Document]) -> str:
    """Extract the most informative sentence per page group without external API calls."""
    by_page: dict[tuple[str, int], str] = {}
    for chunk in chunks:
        key = (str(chunk.metadata.get("source", "doc")), int(chunk.metadata.get("page", 1)))
        if key in by_page:
            continue
        sentences = [
            " ".join(s.split())
            for s in _SENTENCE_SPLIT_RE.split(chunk.page_content.strip())
            if len(s.strip()) > 35
        ]
        if sentences:
            by_page[key] = f"- {sentences[0]} [Source: {key[0]}, p.{key[1]}]"
    return "\n".join(by_page.values())


class DocuMindPipeline:
    """End-to-end LangChain + FAISS RAG pipeline with OpenAI primary and local fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        chat_model: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.provider = self.settings.resolve_provider() if chat_model is None else "custom"
        self.embeddings = embeddings or build_embeddings(self.settings)
        self.index_manager = VectorIndexManager(self.embeddings)
        self.chat_model = chat_model
        self.raw_documents: list[Document] = []
        self.chunks: list[Document] = []

    def _get_chat_model(self) -> ChatOpenAI | Any:
        if self.chat_model is not None:
            return self.chat_model
        return ChatOpenAI(
            model=self.settings.openai_chat_model,
            api_key=self.settings.openai_api_key,
            temperature=0.0,
        )

    def ingest_paths(self, paths: list[Path]) -> int:
        """Load and index documents from file paths; returns indexed chunk count."""
        docs: list[Document] = []
        for path in paths:
            docs.extend(load_document_path(path))
        return self._build_index(docs)

    def ingest_bytes(self, files: list[tuple[str, bytes]]) -> int:
        """Load and index in-memory uploaded files; returns indexed chunk count."""
        docs: list[Document] = []
        for filename, content in files:
            docs.extend(load_document_bytes(filename, content))
        return self._build_index(docs)

    def _build_index(self, docs: list[Document]) -> int:
        if not docs:
            raise ValueError("No readable text found in the provided documents.")
        self.raw_documents = docs
        self.chunks = chunk_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.index_manager.build_from_documents(self.chunks)
        return len(self.chunks)

    def ask(
        self,
        question: str,
        k: int | None = None,
        prompt_mode: Literal["zero_shot", "few_shot"] = "few_shot",
    ) -> RAGResponse:
        """Retrieve relevant chunks from FAISS and produce a cited answer."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        t0 = time.perf_counter()
        top_k = k or self.settings.top_k
        docs_with_scores = self.index_manager.search(question, k=top_k)

        if self.provider in {"openai", "custom"}:
            prompt = get_qa_prompt(prompt_mode)
            chain = prompt | self._get_chat_model()
            context = format_context(docs_with_scores)
            ai_msg = chain.invoke({"context": context, "question": question})
            answer_text = str(getattr(ai_msg, "content", ai_msg)).strip()
        else:
            answer_text = _local_grounded_answer(question, docs_with_scores)

        citations = [
            Citation(
                source=str(doc.metadata.get("source", "document")),
                page=int(doc.metadata.get("page", 1)),
                chunk_id=str(doc.metadata.get("chunk_id", "")),
                score=round(float(score), 4),
                excerpt=doc.page_content[:280].strip(),
            )
            for doc, score in docs_with_scores
        ]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RAGResponse(
            answer=answer_text,
            citations=citations,
            provider_used=self.provider,
            prompt_mode=prompt_mode,
            latency_ms=latency_ms,
        )

    def summarize(self, chunks_per_batch: int = 15) -> SummaryResult:
        """Summarize indexed documents (including 50+ page PDFs) using batched map-reduce."""
        if not self.chunks:
            raise RuntimeError("No documents indexed yet.")
        t0 = time.perf_counter()
        batches = [
            self.chunks[i : i + chunks_per_batch]
            for i in range(0, len(self.chunks), chunks_per_batch)
        ]

        section_summaries: list[str] = []
        if self.provider in {"openai", "custom"}:
            llm = self._get_chat_model()
            map_chain = MAP_SUMMARY_PROMPT | llm
            reduce_chain = REDUCE_SUMMARY_PROMPT | llm
            map_inputs = [
                {"context": format_context([(c, 0.0) for c in batch])}
                for batch in batches
            ]
            # Batch map calls to keep 50+ page summarization fast.
            map_outputs = map_chain.batch(map_inputs)
            section_summaries = [
                str(getattr(msg, "content", msg)).strip() for msg in map_outputs
            ]
            reduced = reduce_chain.invoke(
                {"summaries": "\n\n".join(section_summaries)}
            )
            executive_summary = str(getattr(reduced, "content", reduced)).strip()
        else:
            for idx, batch in enumerate(batches, start=1):
                pages = sorted({int(c.metadata.get("page", 1)) for c in batch})
                bullets = _local_batch_summary(batch)
                section_summaries.append(
                    f"**Batch {idx} (pp. {pages[0]}–{pages[-1]}):**\n{bullets}"
                )
            highlights = []
            for sec in section_summaries[:6]:
                lines = [ln for ln in sec.splitlines() if ln.startswith("- ")]
                if lines:
                    highlights.append(lines[0])
            total_pages = len({(d.metadata.get("source"), d.metadata.get("page")) for d in self.raw_documents})
            executive_summary = (
                f"Document corpus spans {total_pages} pages across {len(self.chunks)} indexed chunks.\n\n"
                "Key section takeaways:\n" + "\n".join(highlights)
            )

        total_pages = len(
            {(d.metadata.get("source"), d.metadata.get("page")) for d in self.raw_documents}
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SummaryResult(
            executive_summary=executive_summary,
            section_summaries=section_summaries,
            total_pages=total_pages,
            total_chunks=len(self.chunks),
            provider_used=self.provider,
            latency_ms=latency_ms,
        )
