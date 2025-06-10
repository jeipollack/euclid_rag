# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Agent for all EC publications and metadata."""

from __future__ import annotations

import string
from typing import Any, cast

import torch
from langchain.agents import Tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from euclid.rag.extra_scripts.deduplication import (
    HashDeduplicator,
    SemanticSimilarityDeduplicator,
)

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
_model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base"
)

BONUS_WEIGHTS = {
    "keywords": 0.5,
    "title": 0.3,
    "authors": 0.3,
    "year": 0.2,
}


def semantic_rerank(query: str, docs: list) -> list:
    """
    Rerank a list of documents based on semantic similarity to a query.

    Parameters
    ----------
    query : str
        The input query string.
    docs : list
        A list of documents to rerank.

        Each document must have a `page_content` attribute.

    Returns
    -------
    list
        The input documents sorted by descending relevance to the query.
    """
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
    """
    Convert text into a set of lowercase tokens with ≥3 characters.

    Punctuation is removed before tokenization.

    Parameters
    ----------
    text : str
        The input text string.

    Returns
    -------
    set of str
        Set of cleaned, lowercase tokens at least 3 characters long.
    """
    return {
        w
        for w in text.lower().translate(punctuation_strip).split()
        if len(w) > 2
    }


def bonus_overlap(q: set[str], field: str | None, weight: float) -> float:
    """
    Compute a weighted count of query tokens found in a metadata field.

    Parameters
    ----------
    q : set of str
        Query tokens.
    field : str or None
        Metadata field to search for token matches.
    weight : float
        Weight to multiply the count by.

    Returns
    -------
    float
        Weighted token overlap score.
    """
    if not field:
        return 0.0
    return weight * sum(
        1
        for w in field.replace(",", " ").split()
        if w.lower().translate(punctuation_strip) in q
    )


def get_publication_tool(
    llm: BaseLanguageModel, retriever: VectorStoreRetriever
) -> Tool:
    """
    Return a tool that answers questions using Euclid Consortium publications.

    Uses a language model and vectorstore retriever,

    to find and summarize relevant papers.

    Parameters
    ----------
    llm : BaseLanguageModel
        The language model used to generate answers.
    retriever : VectorStoreRetriever
        The retriever for accessing publication documents.

    Returns
    -------
    Tool
        A callable tool that answers questions based on EC publications.
    """
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
                "rely on your own knowledge of cosmology and astrophysics"
                "and the Euclid mission to provide a clear,"
                "accurate response **without inventing sources**.\n\n"
                "<CONTEXT>\n{context}\n</CONTEXT>"
            ),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)

    dedup_hash = HashDeduplicator()

    dedup_semantic = SemanticSimilarityDeduplicator(
        vectorstore=cast("FAISS", retriever.vectorstore),
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        similarity_threshold=0.8,
        rerank_threshold=0.85,
        k_candidates=10,
    )

    def retrieve(
        query: str,
        similarity_threshold: int = 50,
        top_candidates_threshold: int = 20,
        top_reranked_threshold: int = 10,
    ) -> list:
        """
        Retrieve relevant documents for a query using multi-stage ranking.

        1. Perform FAISS similarity search (default: 50).
        2. Filter out exact and semantic duplicates.
        3. Score remaining documents using metadata keyword overlap.
        4. Select top N candidates (default: 20) for CrossEncoder reranking.
        5. Return top M reranked documents (default: 10).

        Returns
        -------
        list
            Ranked list of the most relevant documents for the query.
        """
        filtered_docs = []
        for doc, _ in retriever.vectorstore.similarity_search_with_score(
            query, k=similarity_threshold
        ):
            text = doc.page_content
            if not dedup_hash.filter(text) and not dedup_semantic.filter(text):
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
                + bonus_overlap(
                    query_tokens, metadata.get("title"), BONUS_WEIGHTS["title"]
                )
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
        rerank_candidates = [
            d for _, d in metadata_scored_docs[:top_candidates_threshold]
        ]
        return semantic_rerank(query, rerank_candidates)[
            :top_reranked_threshold
        ]

    class _Retriever(BaseRetriever):
        """
        A retriever for a custom `retrieve()` function.

        This class wraps the custom retrieval logic,

        so it can be used in retrieval chains.
        """

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any = None,
            **_: Any,
        ) -> list:
            """
            Retrieve relevant documents for a given query.

            Parameters
            ----------
            query : str
                The query string to retrieve documents for.
            run_manager : Any, optional
                Optional run manager used by LangChain (ignored here).
            **_ : Any
                Additional keyword arguments (ignored).

            Returns
            -------
            list
                List of relevant documents.
            """
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
