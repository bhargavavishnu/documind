"""Zero-shot, few-shot, and map-reduce summarization prompt templates."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GROUNDED_SYSTEM_RULES = (
    "You are DocuMind, a rigorous document intelligence assistant.\n"
    "Follow these rules strictly:\n"
    "1. Answer ONLY using facts explicitly supported by the provided context.\n"
    "2. Cite every claim inline using the exact tag format [Source: <filename>, p.<page>].\n"
    "3. If the context does not contain enough information to answer, reply exactly with: "
    "'I cannot find enough information in the provided documents to answer this question.'\n"
    "4. Never invent metrics, dates, filenames, or page numbers."
)

ZERO_SHOT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM_RULES),
        (
            "human",
            "Context passages:\n{context}\n\nQuestion: {question}\n\nGrounded answer with citations:",
        ),
    ]
)

FEW_SHOT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM_RULES),
        (
            "human",
            "Context passages:\n"
            "[Source: policy.pdf, p.4] Employees receive 20 days of paid annual leave.\n\n"
            "Question: How many days of paid annual leave do employees get?",
        ),
        (
            "ai",
            "Employees receive 20 days of paid annual leave [Source: policy.pdf, p.4].",
        ),
        (
            "human",
            "Context passages:\n"
            "[Source: policy.pdf, p.4] Employees receive 20 days of paid annual leave.\n\n"
            "Question: What is the company's reimbursement cap for home internet?",
        ),
        (
            "ai",
            "I cannot find enough information in the provided documents to answer this question.",
        ),
        (
            "human",
            "Context passages:\n{context}\n\nQuestion: {question}\n\nGrounded answer with citations:",
        ),
    ]
)

MAP_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize the key facts, numbers, and decisions from the following document section. "
            "Preserve [Source: <filename>, p.<page>] citations for each bullet.",
        ),
        ("human", "Section passages:\n{context}\n\nConcise bullet summary:"),
    ]
)

REDUCE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Combine the section summaries below into a cohesive executive summary with "
            "a 3-sentence overview followed by key takeaways retaining [Source: ..., p....] citations.",
        ),
        ("human", "Section summaries:\n{summaries}\n\nExecutive summary:"),
    ]
)


def get_qa_prompt(mode: str) -> ChatPromptTemplate:
    """Return the configured zero-shot or few-shot QA prompt template."""
    if mode == "zero_shot":
        return ZERO_SHOT_QA_PROMPT
    if mode == "few_shot":
        return FEW_SHOT_QA_PROMPT
    raise ValueError(f"Unknown prompt mode '{mode}'. Expected 'zero_shot' or 'few_shot'.")
