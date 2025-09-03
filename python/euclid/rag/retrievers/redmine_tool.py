# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Tool for querying Euclid-Consortium redmine and metadata."""

from __future__ import annotations

import json
import logging
import string
from datetime import datetime, timezone
from typing import Any, cast

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

from euclid import STATIC_DIR
from euclid.rag.extra_scripts.deduplication import HashDeduplicator, SemanticSimilarityDeduplicator
from euclid.rag.utils.acronym_handler import expand_acronyms_in_query

acronym_path = STATIC_DIR / "acronyms.json"
with acronym_path.open(encoding="utf-8") as f:
    ACRONYMS = json.load(f)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

BONUS_WEIGHTS = {
    "pages": 0.5,
    "category": 0.3,
    "year": 0.2,
    "recency": 0.5,
}

TOP_K_SCORING = {
    "similarity_k": 20,
    "top_metadata_k": 10,
    "top_reranked_k": 5,
}


def semantic_rerank(query: str, docs: list) -> list:
    """
    Rerank a list of documents based on semantic similarity to a query.

    Parameters
    ----------
    query : str
        The input query string.
    docs : list
        A list of (langchain) documents to rerank.

        Each document must have a `page_content` attribute.

    Returns
    -------
    list
        The input documents sorted by descending relevance to the query.
    """
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = _tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze()
    scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
    return [doc for _, doc in sorted(zip(scores, docs, strict=False), key=lambda x: -x[0])]


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
    return {w for w in text.lower().translate(punctuation_strip).split() if len(w) > 2}


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

    cleaned_field = field.replace(",", " ").replace("_", " ").replace("-", " ")

    return weight * sum(1 for w in cleaned_field.split() if w.lower().translate(punctuation_strip) in q)


def bonus_recency(updated_on: datetime | None, weight: float = 0.3, half_life: int = 365) -> float:
    """
    Compute a bonus based on how recent the update is.

    Parameters
    ----------
    updated_on : datetime.datetime
        The datetime the document was last updated.
    weight : float
        The max weight to apply for a very recent document.
    half_life : int
        The number of days after which the weight is halved.
        If not specified, defaults to 365 days.

    Returns
    -------
    float
        A score between 0 and weight depending on recency.
    """
    if updated_on is None:
        return 0.0

    # Always work with UTC-aware datetimes
    now = datetime.now(timezone.utc)  # noqa: UP017 - datetime.UTC is not compatible with mypy
    if updated_on.tzinfo is None:
        updated_on = updated_on.replace(tzinfo=timezone.utc)  # noqa: UP017

    days_old: float = float((now - updated_on).days)

    # Exponential decay: more recent → higher score
    decay: float = 0.5 ** (days_old / half_life)
    return decay


def bonus_scoring(query: str, docs: list, nb_retained_docs: int = 10) -> list[tuple]:
    """Score documents based on metadata keyword overlap and recency."""
    query_tokens = tokenize(query)
    metadata_scored_docs = []
    for doc in docs:
        logger.debug(f"[RAG] Scoring doc: {doc.metadata.get('project_path')}")
        metadata = doc.metadata
        updated_on = metadata.get("updated_on")
        score = (
            bonus_overlap(query_tokens, metadata.get("page_name"), BONUS_WEIGHTS["pages"])
            + bonus_overlap(query_tokens, str(metadata.get("category")), BONUS_WEIGHTS["category"])
            + bonus_overlap(query_tokens, str(updated_on.year) if updated_on else None, BONUS_WEIGHTS["year"])
            + bonus_recency(updated_on, weight=BONUS_WEIGHTS["recency"])
        )
        metadata_scored_docs.append((score, doc))
        logger.debug(
            f"[RAG] Bonus score {score:.3f} for {metadata.get('project_path')} "
            f"(category={metadata.get('category')}, "
            f"year={metadata.get('updated_on')},"
            f"hierarchy={metadata.get('hierarchy')})"
        )
    logger.info("[RAG] Metadata scoring completed.")
    metadata_scored_docs.sort(key=lambda x: x[0], reverse=True)
    return metadata_scored_docs[:nb_retained_docs]


def filter_retrieved(
    scored_and_docs: list[tuple], dedup_hash: HashDeduplicator, dedup_semantic: SemanticSimilarityDeduplicator
) -> list[tuple]:
    """Filter out exact and semantic duplicates from retrieved documents."""
    filtered_scores_and_docs = []
    for doc, score in scored_and_docs:
        text = doc.page_content
        if dedup_hash.filter(text):
            logger.debug("[RAG] Hash duplicate removed.")
            continue
        if dedup_semantic and dedup_semantic.filter(text):
            logger.debug("[RAG] Semantic duplicate removed.")
            continue
        filtered_scores_and_docs.append((score, doc))
    logger.info("[RAG] Duplicate filtering completed.")
    return sorted(filtered_scores_and_docs, key=lambda x: x[0], reverse=True)


def get_redmine_tool(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Tool:
    """
    Return a tool that answers questions using Euclid Consortium Redmine.
    Uses a language model and vector store retriever,
    to find and summarize relevant redmine wikis.

    Parameters
    ----------
    llm : BaseLanguageModel
        The language model used to generate answers.
    retriever : VectorStoreRetriever
        The retriever for accessing redmine documents.

    Returns
    -------
    Tool
        A callable tool that answers questions based on EC redmine.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are the Redmine Agent for the Euclid AI Assistant;"
                "the AI assistant for the Euclid Space Telescope Mission by"
                "the Euclid Consortium and European Space Agency.\n"
                "First look at the CONTEXT below.\n"
                "If it contains information that directly answers"
                "the user's question, quote or paraphrase it.\n"
                "If the CONTEXT is missing a full answer,"
                "answer 'I don't have that information yet.' and nothing else."
                "\n\n"
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

    def retrieve(query: str) -> list:
        """
        Retrieve relevant documents for a query using multi-stage ranking.

        Steps:
        1. Expand acronyms.
        2. Perform FAISS similarity search.
        3. Filter out exact and semantic duplicates.
        4. Score remaining documents using metadata keyword overlap.
        5. Select top metadata-matching documents for CrossEncoder reranking.
        . Return top reranked documents.

        Returns
        -------
        list
            Ranked list of the most relevant documents for the query.
        """
        # 1. Filter and expand acronyms in the query
        logger.info(f"[RAG] Query received: {query}")
        query = expand_acronyms_in_query(query, ACRONYMS)
        logger.debug(f"[RAG] Expanded query: {query}")
        # 2. Initial FAISS similarity search
        initial_results = retriever.vectorstore.similarity_search_with_score(query, k=TOP_K_SCORING["similarity_k"])
        logger.info(f"[RAG] Retrieved {len(initial_results)} documents from FAISS.")
        # 3. Filter out exact and semantic duplicates
        filtered_scores_and_docs = filter_retrieved(initial_results, dedup_hash, dedup_semantic)
        logger.info(f"[RAG] {len(filtered_scores_and_docs)} documents remaining.")
        logger.info(f"[RAG] Top corresponding scores: {[round(s, 3) for s, _ in filtered_scores_and_docs[:5]]}")
        # 4. Score documents using metadata keyword overlap and recency
        filtered_docs = [d for _, d in filtered_scores_and_docs]
        top_metadata_scores_docs = bonus_scoring(query, filtered_docs, TOP_K_SCORING["top_metadata_k"])
        logger.info(f"[RAG] Top metadata scores: {[round(s, 3) for s, _ in top_metadata_scores_docs]}")
        logger.info(f"[RAG] {len(top_metadata_scores_docs)} documents kept for reranking.")
        # 5. Rerank top metadata-matching documents using CrossEncoder
        top_metadata_docs = [d for _, d in top_metadata_scores_docs]
        reranked = semantic_rerank(query, top_metadata_docs)
        logger.info(f"[RAG] Final reranked document count: {len(reranked)}")
        return reranked[: TOP_K_SCORING["top_reranked_k"]]

    class _Retriever(BaseRetriever):
        """
        A retriever for a custom `retrieve()` function.
        This class wraps the custom retrieval logic,
        so it can be used in retrieval chains.
        """

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None, **_: Any) -> list:
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
            key = (m.get("wiki_path"), m.get("updated_on"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"{m.get('source')} "
                f"- {m.get('project_path', 'Untitled')} "
                f" (last update: {m.get('updated_on', 'n.d.')}) \n"
                f"{m.get('wiki_path', 'No URL available')}"
            )
        # Append formatted sources if any
        if lines:
            answer += "\n\n**Sources**\n\n" + "\n\n".join(lines)

        logger.info(f"[RAG] Returning final answer with {len(res['context'])} sources.")

        return answer

    return Tool(
        name="euclid_redmine_tool",
        func=run,
        description="Answer questions using Euclid Consortium Redmine.",
    )
