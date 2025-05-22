# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Agent for all EC publications and metadata."""

from __future__ import annotations

import string
from typing import Any

import torch
from langchain.agents import Tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from euclid.rag.extra_scripts.deduplication import DeduplicationFilter

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
_model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base"
)


def semantic_rerank(query: str, docs: list) -> list:
    """Rerank a list of documents based on their similarity to a query."""
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = _tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze()
    scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
    return [
        doc
        for _, doc in sorted(
            zip(scores, docs, strict=False), key=lambda x: -x[0]
        )
    ]


punctuation_strip = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> set[str]:
    """Return lowercase words ≥3 chars, punctuation removed."""
    return {
        w
        for w in text.lower().translate(punctuation_strip).split()
        if len(w) > 2
    }


def bonus_overlap(q: set[str], field: str | None, weight: float) -> float:
    """Weighted count of query tokens found in a metadata field."""
    if not field:
        return 0.0
    return weight * sum(
        1
        for w in field.replace(",", " ").split()
        if w.lower().translate(punctuation_strip) in q
    )


def get_publication_agent(
    llm: BaseLanguageModel, retriever: VectorStoreRetriever
) -> Tool:
    """Return an Agent that answers based on EC publications."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are the Publications Agent for the Euclid AI Assistant;"
                "the AI assistant for the Euclid Space Telescope Mission by"
                "the Euclid Consortium and European Space Agency.\n"
                "First look at the CONTEXT below.\n"
                "If it contains information that directly answers"
                "the users question, quote or paraphrase that.\n"
                "If the CONTEXT is missing a full answer,"
                "rely on your own general knowledge of astronomy"
                "and the Euclid mission to provide a clear,"
                "accurate response **without inventing sources**.\n\n"
                "<CONTEXT>\n{context}\n</CONTEXT>"
            ),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)

    deduper = DeduplicationFilter(
        vectorstore=retriever.vectorstore,
        config={
            "similarity_threshold": 0.8,
            "rerank_threshold": 0.85,
            "k_candidates": 10,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    )

    def retrieve(query: str) -> list:
        """Similarity search for (k=25) + bonuses."""
        unique_hits = []
        for doc, _ in retriever.vectorstore.similarity_search_with_score(
            query, k=50
        ):
            if not deduper.is_duplicate(doc.page_content):
                unique_hits.append(doc)

        q_tok = tokenize(query)
        scored = []
        for doc in unique_hits:
            m = doc.metadata
            score = (
                bonus_overlap(q_tok, m.get("keywords"), 0.5)
                + bonus_overlap(q_tok, m.get("title"), 0.3)
                + bonus_overlap(q_tok, m.get("authors"), 0.3)
                + bonus_overlap(q_tok, str(m.get("year")), 0.2)
            )
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        rerank_candidates = [d for _, d in scored[:20]]
        return semantic_rerank(query, rerank_candidates)[:10]

    class _Retriever(BaseRetriever):
        """Class that lets LangChain call custom retrieve function."""

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any = None,
            **_: Any,
        ) -> list:
            return retrieve(query)

    chain = create_retrieval_chain(_Retriever(), doc_chain)

    def run(question: str, callbacks: list[Any] | None = None) -> str:
        """Return LLM answer plus formatted source list."""
        res = chain.invoke(
            {"input": question, "chat_history": []},
            {"callbacks": callbacks},
        )

        # Remove any model-generated sources if present
        answer = res["answer"].strip().split("**Sources**")[0].strip()

        # Build custom source list from metadata
        seen, lines = set(), []
        for d in res["context"]:
            m = d.metadata
            key = (m.get("title"), m.get("url"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- **{m.get('title', 'Untitled')}** "
                f"({m.get('year', 'n.d.')}) — "
                f"{m.get('authors', 'Euclid Collaboration')} — {m.get('url')}"
            )

        # Append formatted sources if any
        if lines:
            answer += "\n\n**Sources**\n" + "\n".join(lines)

        return answer

    return Tool(
        name="euclid_publication_agent",
        func=run,
        description="Answer questions about Euclid scientific papers.",
    )
