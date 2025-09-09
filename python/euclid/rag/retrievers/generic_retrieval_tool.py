# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Generic Retriever Tool for querying Euclid-Consortium documents."""

from __future__ import annotations

import string
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import torch
from langchain.agents import Tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from euclid.rag.extra_scripts.deduplication import SemanticSimilarityDeduplicator

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

BONUS_WEIGHTS = {
    "keywords": 0.5,
    "title": 0.3,
    "authors": 0.3,
    "year": 0.2,
}

TOP_K_FOR_METADATA_SCORING = {
    "similarity_k": 50,
    "top_metadata_k": 20,
    "top_reranked_k": 10,
}

punctuation_strip = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> set[str]:
    """Convert text into a set of lowercase tokens ≥3 characters."""
    return {w for w in text.lower().translate(punctuation_strip).split() if len(w) > 2}


def bonus_overlap(q: set[str], field: str | None, weight: float) -> float:
    """Compute weighted count of query tokens in a metadata field."""
    if not field:
        return 0.0
    return weight * sum(1 for w in str(field).replace(",", " ").split() if w.lower().translate(punctuation_strip) in q)


def semantic_rerank(query: str, docs: list) -> list:
    """Rerank a list of documents by semantic similarity to the query."""
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = _tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze()
    scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
    return [doc for _, doc in sorted(zip(scores, docs, strict=False), key=lambda x: -x[0])]


def format_source(m: dict) -> str:
    """Format a source line based on document metadata."""
    category = m.get("category", "").upper()
    if category == "DPDD":
        return f"- **{m.get('subtopic', 'Untitled')}** — {m.get('source', 'Unknown source')}"
    elif m.get("title"):
        return (
            f"- **{m.get('title', 'Untitled')}** "
            f"({m.get('year', 'n.d.')}) — "
            f"{m.get('authors', 'Euclid Collaboration')} — "
            f"{m.get('url', 'Unknown URL')}"
        )
    else:
        return f"- {m.get('source', 'Unknown source')}"


def normalize_url(url: str | None) -> str | None:
    """Remove URL fragments and query parameters for deduplication."""
    if not url:
        return url
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment="", query=""))


def get_generic_retrieval_tool(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Tool:
    """
    Return a generic Euclid retrieval tool that answers questions
    from any document type (publications, DPDD, etc.).

    Parameters
    ----------
    llm : BaseLanguageModel
        The language model used to generate answers.
    retriever : VectorStoreRetriever
        The retriever for accessing vectorstore documents.

    Returns
    -------
    Tool
        A callable tool that answers questions and formats sources
        for all Euclid document types.
    """
    # Prompt template remains the same
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are the Euclid Retrieval Agent.\n"
                "First look at the CONTEXT below.\n"
                "If it contains information that directly answers"
                "the user's question, quote or paraphrase it.\n"
                "If the CONTEXT is missing a full answer,"
                "rely on your own knowledge without inventing sources.\n\n"
                "<CONTEXT>\n{context}\n</CONTEXT>"
            ),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)

    dedup_semantic = SemanticSimilarityDeduplicator(
        vectorstore=cast("FAISS", retriever.vectorstore),
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        similarity_threshold=0.8,
        rerank_threshold=0.85,
        k_candidates=10,
    )

    def retrieve(query: str) -> list:
        """
        Multi-stage retrieval with robust metadata scoring.
        Missing metadata fields are ignored.
        """
        filtered_docs = []
        for doc, _ in retriever.vectorstore.similarity_search_with_score(
            query, k=TOP_K_FOR_METADATA_SCORING["similarity_k"]
        ):
            text = doc.page_content
            if not dedup_semantic.filter(text):
                filtered_docs.append(doc)

        query_tokens = tokenize(query)
        metadata_scored_docs = []

        for doc in filtered_docs:
            metadata = doc.metadata
            score = (
                bonus_overlap(
                    query_tokens,
                    metadata.get("keywords"),
                    BONUS_WEIGHTS["keywords"],
                )
                + bonus_overlap(query_tokens, metadata.get("title"), BONUS_WEIGHTS["title"])
                + bonus_overlap(
                    query_tokens,
                    metadata.get("authors"),
                    BONUS_WEIGHTS["authors"],
                )
                + bonus_overlap(
                    query_tokens,
                    str(metadata.get("year")),
                    BONUS_WEIGHTS["year"],
                )
            )
            metadata_scored_docs.append((score, doc))

        metadata_scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_scored_docs = [d for _, d in metadata_scored_docs[: TOP_K_FOR_METADATA_SCORING["top_metadata_k"]]]
        return semantic_rerank(query, top_scored_docs)[: TOP_K_FOR_METADATA_SCORING["top_reranked_k"]]

    class _Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **_: Any) -> list:
            return retrieve(query)

    chain = create_retrieval_chain(_Retriever(), doc_chain)

    def run(question: str, callbacks: list[Any] | None = None) -> str:
        """Return LLM answer plus formatted source list."""
        # Run the chain using the page_content only
        res = chain.invoke({"input": question, "chat_history": []}, {"callbacks": callbacks})

        # Deduplicate sources
        seen = set()
        lines = []

        for d in res["context"]:
            m = getattr(d, "metadata", d) or {}
            raw_url = m.get("url") or m.get("source")
            norm_url = normalize_url(raw_url)  # strip fragments/query

            # Deduplication key: combine the formatted source text and the URL
            display_text = (m.get("subtopic") or m.get("title") or m.get("source") or "").strip()

            key = (display_text.strip().lower(), norm_url)

            # Skip if already seen
            if key in seen:
                continue

            # Add the URL to metadata for the source
            seen.add(key)
            lines.append(format_source(m))

        # Build the final answer
        answer = res["answer"].strip().split("**Sources**")[0].strip()
        if lines:
            answer += "\n\nSources\n" + "\n".join(lines)

        return answer

    return Tool(
        name="euclid_retrieval_tool",
        func=run,
        description="Answer questions using Euclid publications and DPDD.",
    )
