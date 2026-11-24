"""Reproducible RAG evaluation comparing FAISS semantic retrieval vs. naive keyword baseline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from documind.config import Settings  # noqa: E402
from documind.ingest import chunk_documents, load_document_path  # noqa: E402
from documind.rag import DocuMindPipeline  # noqa: E402
from documind.sample_data import generate_handbook_pdf  # noqa: E402
from documind.vectorstore import KeywordRetriever, VectorIndexManager, build_embeddings, tokenize  # noqa: E402

EVAL_QUERIES = [
    # Section 1 (pp. 2-5): Identity, OAuth 2.1, Session Security
    {"query": "How long do refresh credentials last before a user must log in again?", "expected_pages": [2, 3, 4, 5], "topic": "OAuth refresh lifetime"},
    {"query": "Can engineers use SMS text codes for admin console two-factor login?", "expected_pages": [2, 3, 4, 5], "topic": "MFA FIDO2 vs SMS"},
    # Section 2 (pp. 6-9): Vector Indexing & FAISS
    {"query": "Which FAISS index structure is chosen once embeddings grow past half a million vectors?", "expected_pages": [6, 7, 8, 9], "topic": "FAISS IndexIVFPQ"},
    {"query": "Why are vectors L2 normalized before placing them into the search index?", "expected_pages": [6, 7, 8, 9], "topic": "Cosine similarity normalization"},
    # Section 3 (pp. 10-13): Rate Limits & Backoff
    {"query": "What wait strategy should clients follow after receiving a 429 throttling error?", "expected_pages": [10, 11, 12, 13], "topic": "Retry backoff"},
    {"query": "How many API calls per minute are permitted for standard workspaces?", "expected_pages": [10, 11, 12, 13], "topic": "Rate quota"},
    # Section 4 (pp. 14-17): Incident Command
    {"query": "How quickly must an Incident Commander take ownership of a critical outage?", "expected_pages": [14, 15, 16, 17], "topic": "SEV-1 assignment"},
    {"query": "Within how many hours after resolving an incident must the postmortem be written?", "expected_pages": [14, 15, 16, 17], "topic": "Post-incident review"},
    # Section 5 (pp. 18-21): SLAs & Credits
    {"query": "What percentage invoice refund applies if monthly uptime drops beneath 99 percent?", "expected_pages": [18, 19, 20, 21], "topic": "SLA 30% credit"},
    {"query": "Are brief planned upgrades excluded from downtime penalties?", "expected_pages": [18, 19, 20, 21], "topic": "Maintenance exclusions"},
    # Section 6 (pp. 22-25): Data Retention & Erasure
    {"query": "How soon are uploaded customer PDFs wiped after deleting a workspace?", "expected_pages": [22, 23, 24, 25], "topic": "PDF purge window"},
    {"query": "How many years must immutable accounting records be preserved?", "expected_pages": [22, 23, 24, 25], "topic": "WORM ledger retention"},
    # Section 7 (pp. 26-29): Billing & Spend Guardrails
    {"query": "What happens to unused GPU endpoints when nobody sends requests?", "expected_pages": [26, 27, 28, 29], "topic": "Scale to zero GPU"},
    {"query": "At what budget thresholds are spending notifications dispatched?", "expected_pages": [26, 27, 28, 29], "topic": "Budget webhooks"},
    # Section 8 (pp. 30-33): Kubernetes Autoscaling
    {"query": "What CPU utilization percentage triggers horizontal pod scaling?", "expected_pages": [30, 31, 32, 33], "topic": "HPA target"},
    {"query": "How do stateful pods stay online if an entire cloud availability zone loses power?", "expected_pages": [30, 31, 32, 33], "topic": "Multi-AZ anti-affinity"},
    # Section 9 (pp. 34-37): Observability & SLOs
    {"query": "When do we record 100 percent of distributed traces instead of 10 percent?", "expected_pages": [34, 35, 36, 37], "topic": "Tail-based sampling"},
    {"query": "What happens to code merges if the error budget burns too rapidly?", "expected_pages": [34, 35, 36, 37], "topic": "Release freeze"},
    # Section 10 (pp. 38-41): Disaster Recovery
    {"query": "How much data can be lost during a regional failure versus a single-zone failure?", "expected_pages": [38, 39, 40, 41], "topic": "RPO targets"},
    {"query": "How frequently are backup snapshots verified by restoring into staging?", "expected_pages": [38, 39, 40, 41], "topic": "Quarterly game days"},
    # Section 11 (pp. 42-45): Key Rotation
    {"query": "How often do KMS data encryption keys rotate automatically?", "expected_pages": [42, 43, 44, 45], "topic": "90-day key rotation"},
    {"query": "How fast are compromised credentials invalidated across the edge?", "expected_pages": [42, 43, 44, 45], "topic": "3-second revocation"},
    # Section 12 (pp. 46-49): Canary Rollouts
    {"query": "What share of production requests enters a new canary deployment initially?", "expected_pages": [46, 47, 48, 49], "topic": "Canary 5% traffic"},
    {"query": "Why must database schema updates use expand-and-contract steps?", "expected_pages": [46, 47, 48, 49], "topic": "Zero-downtime rollbacks"},
    # Section 13 (pp. 50-53): Remote Work & Expenses
    {"query": "How much money can staff spend on a home desk, chair, or monitor each year?", "expected_pages": [50, 51, 52, 53], "topic": "Ergonomics stipend"},
    {"query": "When should employees pick trains instead of airplanes for company travel?", "expected_pages": [50, 51, 52, 53], "topic": "Rail travel under 4h"},
    # Section 14 (pp. 54-56): Compliance & Audits
    {"query": "Which external security certifications are renewed every year?", "expected_pages": [54, 55, 56], "topic": "SOC 2 & ISO 27001"},
    {"query": "How quickly are permissions removed when a contractor leaves?", "expected_pages": [54, 55, 56], "topic": "24-hour access revocation"},
]


class NaiveExactKeywordRetriever:
    """Simple unweighted exact-word-overlap baseline (no semantic embeddings or IDF weighting)."""

    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.chunk_tokens = [set(tokenize(c.page_content)) for c in self.chunks]

    def search(self, query: str, k: int = 4):
        q_tokens = set(tokenize(query))
        scored = []
        for chunk, doc_tokens in zip(self.chunks, self.chunk_tokens):
            overlap = len(q_tokens & doc_tokens)
            scored.append((chunk, float(overlap)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def score_retriever(retriever, queries: list[dict], k: int = 4) -> dict[str, float]:
    """Compute Recall@K, MRR@K, and Irrelevant-Context Rate across eval queries."""
    hits = 0
    mrr_sum = 0.0
    total_slots = 0
    irrelevant_slots = 0

    for item in queries:
        expected = set(item["expected_pages"])
        results = retriever.search(item["query"], k=k)
        pages = [int(doc.metadata.get("page", -1)) for doc, _ in results]

        if any(p in expected for p in pages):
            hits += 1
        for rank, p in enumerate(pages, start=1):
            if p in expected:
                mrr_sum += 1.0 / rank
                break

        total_slots += len(pages)
        irrelevant_slots += sum(1 for p in pages if p not in expected)

    n = len(queries)
    return {
        "recall_at_k": round(hits / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "irrelevant_context_rate": round(irrelevant_slots / max(total_slots, 1), 4),
    }


def main() -> None:
    pdf_path = ROOT_DIR / "data" / "samples" / "nova_cloud_handbook_56p.pdf"
    if not pdf_path.exists():
        generate_handbook_pdf(pdf_path, target_pages=56)

    docs = load_document_path(pdf_path)
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)

    settings = Settings.from_env()
    # Evaluate FAISS with active embeddings (OpenAI if configured, else local sentence-transformers).
    embeddings = build_embeddings(settings)
    faiss_mgr = VectorIndexManager(embeddings)
    t0 = time.perf_counter()
    faiss_mgr.build_from_documents(chunks)
    index_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    naive_kw = NaiveExactKeywordRetriever(chunks)
    bm25 = KeywordRetriever(chunks)

    naive_metrics = score_retriever(naive_kw, EVAL_QUERIES, k=4)
    bm25_metrics = score_retriever(bm25, EVAL_QUERIES, k=4)
    faiss_metrics = score_retriever(faiss_mgr, EVAL_QUERIES, k=4)

    # Measure 56-page summarization latency and sample grounded Q&A.
    pipeline = DocuMindPipeline(settings=settings, embeddings=embeddings)
    pipeline.raw_documents = docs
    pipeline.chunks = chunks
    pipeline.index_manager = faiss_mgr
    summary = pipeline.summarize()
    sample_qa = pipeline.ask(EVAL_QUERIES[0]["query"], k=4, prompt_mode="few_shot")
    abstain_qa = pipeline.ask("What is the CEO's favorite coffee roast?", k=4, prompt_mode="few_shot")

    rel_recall_gain = (
        (faiss_metrics["recall_at_k"] - naive_metrics["recall_at_k"])
        / max(naive_metrics["recall_at_k"], 1e-6)
    )
    irr_reduction = (
        (naive_metrics["irrelevant_context_rate"] - faiss_metrics["irrelevant_context_rate"])
        / max(naive_metrics["irrelevant_context_rate"], 1e-6)
    )

    report = {
        "corpus_pages": len(docs),
        "total_chunks": len(chunks),
        "num_eval_queries": len(EVAL_QUERIES),
        "top_k": 4,
        "provider_tested": settings.resolve_provider(),
        "embedding_model": (
            settings.openai_embedding_model
            if settings.resolve_provider() == "openai"
            else settings.local_embedding_model
        ),
        "faiss_index_build_ms": index_ms,
        "keyword": naive_metrics,
        "bm25": bm25_metrics,
        "faiss": faiss_metrics,
        "relative_recall_improvement_vs_keyword": round(rel_recall_gain, 4),
        "irrelevant_context_reduction_vs_keyword": round(irr_reduction, 4),
        "summarization_pages": summary.total_pages,
        "summarization_latency_ms": summary.latency_ms,
        "sample_grounded_answer": sample_qa.answer,
        "out_of_scope_guardrail_answer": abstain_qa.answer,
    }

    eval_dir = ROOT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "qa_dataset.json").write_text(json.dumps(EVAL_QUERIES, indent=2), encoding="utf-8")
    (eval_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# DocuMind Retrieval & Summarization Evaluation

- **Corpus:** `{pdf_path.name}` ({report['corpus_pages']} pages, {report['total_chunks']} chunks)
- **Queries:** {report['num_eval_queries']} paraphrased domain questions across all 14 handbook sections (`top_k=4`)
- **Provider / Embeddings:** `{report['provider_tested']}` (`{report['embedding_model']}`)
- **FAISS Index Build Time:** {report['faiss_index_build_ms']} ms

## Retrieval Benchmark (`k=4`)

| Retriever | Recall@4 | MRR@4 | Irrelevant-Context Rate |
| :--- | :---: | :---: | :---: |
| Naive Exact-Keyword Baseline | {naive_metrics['recall_at_k']:.1%} | {naive_metrics['mrr']:.3f} | {naive_metrics['irrelevant_context_rate']:.1%} |
| BM25 Lexical Baseline | {bm25_metrics['recall_at_k']:.1%} | {bm25_metrics['mrr']:.3f} | {bm25_metrics['irrelevant_context_rate']:.1%} |
| **FAISS Semantic RAG** | **{faiss_metrics['recall_at_k']:.1%}** | **{faiss_metrics['mrr']:.3f}** | **{faiss_metrics['irrelevant_context_rate']:.1%}** |

- **Relative Recall@4 Gain vs. Naive Keyword:** `{rel_recall_gain:+.1%}`
- **Irrelevant-Context Reduction vs. Naive Keyword:** `{irr_reduction:+.1%}`

## 56-Page Summarization & Grounding Checks

- **56-Page Summary Latency:** {summary.latency_ms} ms ({summary.total_pages} pages, {summary.total_chunks} chunks)
- **Sample Grounded Answer (`{EVAL_QUERIES[0]['topic']}`):**
  > {sample_qa.answer}
- **Out-of-Scope Hallucination Guardrail Check:**
  > {abstain_qa.answer}
"""
    (eval_dir / "results.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
