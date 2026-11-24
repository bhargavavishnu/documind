"""Interactive Streamlit UI for DocuMind RAG Q&A, 50+ page summarization, and eval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from documind.config import Settings
from documind.rag import DocuMindPipeline
from documind.sample_data import generate_handbook_pdf

ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_PDF_PATH = ROOT_DIR / "data" / "samples" / "nova_cloud_handbook_56p.pdf"
EVAL_RESULTS_PATH = ROOT_DIR / "eval" / "results.json"

st.set_page_config(page_title="DocuMind — RAG Document Intelligence", page_icon="📄", layout="wide")
st.title("DocuMind — RAG Document Intelligence Assistant")
st.caption("LangChain + FAISS vector search with grounded inline citations and 50+ page summarization.")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key (optional for session)",
        type="password",
        help="Leave blank to run with local sentence-transformer embeddings & grounded fallback.",
    )
    prompt_mode = st.selectbox("Prompt Template", ["few_shot", "zero_shot"], index=0)
    top_k = st.slider("Top-K Chunks (FAISS)", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk Size (chars)", min_value=400, max_value=1400, value=800, step=100)

    settings = Settings.from_env(openai_api_key_override=api_key_input or None)
    settings = Settings(
        openai_api_key=settings.openai_api_key,
        openai_chat_model=settings.openai_chat_model,
        openai_embedding_model=settings.openai_embedding_model,
        provider_mode=settings.provider_mode,
        local_embedding_model=settings.local_embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=150,
        top_k=top_k,
    )
    active_provider = settings.resolve_provider()
    st.info(f"Active Provider: **{active_provider.upper()}**")

    st.divider()
    uploaded = st.file_uploader(
        "Upload PDF / TXT / MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    load_sample = st.button("Load Sample 56-Page Handbook PDF", use_container_width=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.doc_label = None
    st.session_state.messages = []

if uploaded:
    file_payloads = [(f.name, f.read()) for f in uploaded]
    with st.spinner("Chunking and building FAISS index..."):
        pipe = DocuMindPipeline(settings=settings)
        n_chunks = pipe.ingest_bytes(file_payloads)
        st.session_state.pipeline = pipe
        st.session_state.doc_label = f"{len(file_payloads)} uploaded file(s) ({n_chunks} chunks)"
        st.session_state.messages = []
elif load_sample:
    if not SAMPLE_PDF_PATH.exists():
        generate_handbook_pdf(SAMPLE_PDF_PATH)
    with st.spinner("Indexing 56-page sample handbook into FAISS..."):
        pipe = DocuMindPipeline(settings=settings)
        n_chunks = pipe.ingest_paths([SAMPLE_PDF_PATH])
        st.session_state.pipeline = pipe
        st.session_state.doc_label = f"{SAMPLE_PDF_PATH.name} ({n_chunks} chunks)"
        st.session_state.messages = []

if st.session_state.doc_label:
    st.success(f"Indexed: {st.session_state.doc_label}")
else:
    st.warning("Upload documents or click **Load Sample 56-Page Handbook PDF** in the sidebar to begin.")

tab_qa, tab_summary, tab_eval = st.tabs(["Ask Documents", "50+ Page Summarizer", "Evaluation Benchmark"])

with tab_qa:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Retrieved Source Chunks & Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"- **{c['source']} (p.{c['page']})** | score `{c['score']}` — {c['excerpt']}...")

    question = st.chat_input("Ask a question about the indexed documents...")
    if question:
        if st.session_state.pipeline is None:
            st.error("Please index at least one document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching FAISS and generating grounded answer..."):
                    res = st.session_state.pipeline.ask(
                        question, k=top_k, prompt_mode=prompt_mode
                    )
                st.markdown(res.answer)
                st.caption(
                    f"Provider: `{res.provider_used}` | Prompt: `{res.prompt_mode}` | Latency: `{res.latency_ms} ms`"
                )
                citations_data = [c.model_dump() for c in res.citations]
                with st.expander("Retrieved Source Chunks & Citations"):
                    for c in citations_data:
                        st.markdown(f"- **{c['source']} (p.{c['page']})** | score `{c['score']}` — {c['excerpt']}...")
                st.session_state.messages.append(
                    {"role": "assistant", "content": res.answer, "citations": citations_data}
                )

with tab_summary:
    if st.button("Summarize Full Indexed Corpus", disabled=st.session_state.pipeline is None):
        with st.spinner("Running batched map-reduce summarization..."):
            summary = st.session_state.pipeline.summarize()
        st.metric("Pages Summarized", summary.total_pages)
        st.metric("Summarization Latency", f"{summary.latency_ms} ms")
        st.markdown("### Executive Summary")
        st.markdown(summary.executive_summary)
        with st.expander("Section-by-Section Batches"):
            for sec in summary.section_summaries:
                st.markdown(sec)
                st.divider()

with tab_eval:
    st.markdown("### FAISS Semantic RAG vs. Naive Keyword Baseline")
    if EVAL_RESULTS_PATH.exists():
        data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Recall@4 (FAISS vs Keyword)",
            f"{data['faiss']['recall_at_k']:.1%}",
            delta=f"{(data['faiss']['recall_at_k'] - data['keyword']['recall_at_k']):+.1%}",
        )
        col2.metric(
            "MRR@4 (FAISS vs Keyword)",
            f"{data['faiss']['mrr']:.3f}",
            delta=f"{(data['faiss']['mrr'] - data['keyword']['mrr']):+.3f}",
        )
        col3.metric(
            "Irrelevant-Context Rate",
            f"{data['faiss']['irrelevant_context_rate']:.1%}",
            delta=f"{(data['faiss']['irrelevant_context_rate'] - data['keyword']['irrelevant_context_rate']):+.1%}",
            delta_color="inverse",
        )
        st.json(data)
    else:
        st.info("Run `uv run python eval/evaluate_rag.py` to generate live evaluation results.")
