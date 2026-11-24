# DocuMind Retrieval & Summarization Evaluation

- **Corpus:** `nova_cloud_handbook_56p.pdf` (56 pages, 111 chunks)
- **Queries:** 28 paraphrased domain questions across all 14 handbook sections (`top_k=4`)
- **Provider / Embeddings:** `local` (`sentence-transformers/all-MiniLM-L6-v2`)
- **FAISS Index Build Time:** 763.8 ms

## Retrieval Benchmark (`k=4`)

| Retriever | Recall@4 | MRR@4 | Irrelevant-Context Rate |
| :--- | :---: | :---: | :---: |
| Naive Exact-Keyword Baseline | 75.0% | 0.750 | 25.0% |
| BM25 Lexical Baseline | 89.3% | 0.893 | 10.7% |
| **FAISS Semantic RAG** | **100.0%** | **0.955** | **14.3%** |

- **Relative Recall@4 Gain vs. Naive Keyword:** `+33.3%`
- **Irrelevant-Context Reduction vs. Naive Keyword:** `+42.8%`

## 56-Page Summarization & Grounding Checks

- **56-Page Summary Latency:** 0.66 ms (56 pages, 111 chunks)
- **Sample Grounded Answer (`OAuth refresh lifetime`):**
  > Clause 1.3.4: Short-lived JSON Web Tokens expire after 15 minutes, while rotating refresh tokens remain valid for 14 days before requiring fresh interactive sign-in. [Source: nova_cloud_handbook_56p.pdf, p.4] Operational Verification Note (Page 4): Platform owners review Identity, OAuth 2.1, and Session Security controls prior to every quarterly readiness sign-off. [Source: nova_cloud_handbook_56p.pdf, p.4] Clause 1.4.3: Short-lived JSON Web Tokens expire after 15 minutes, while rotating refresh tokens remain valid for 14 days before requiring fresh interactive sign-in. [Source: nova_cloud_handbook_56p.pdf, p.5]
- **Out-of-Scope Hallucination Guardrail Check:**
  > I cannot find enough information in the provided documents to answer this question.
