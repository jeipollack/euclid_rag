# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Deduplication filter using hash, FAISS similarity,
and cross-encoder re-ranking.
"""

import hashlib

import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sklearn.cluster import AgglomerativeClustering


class DeduplicationFilter:
    """Deduplication filter using hash and semantic similarity
    with re-ranking and batch clustering.
    """

    def __init__(
        self,
        vectorstore: FAISS | None = None,
        config: dict | None = None,
    ) -> None:
        config = config or {}

        self.vectorstore = vectorstore
        self.similarity_threshold = config.get("similarity_threshold")
        self.rerank_threshold = config.get("rerank_threshold")
        self.k_candidates = config.get("k_candidates")

        self._seen_hashes: set[str] = set()
        self._seen_texts: dict[str, str] = {}

        reranker_model = config.get("reranker_model")
        self.reranker = CrossEncoder(reranker_model)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        h = self._hash_text(text)

        if h in self._seen_hashes:
            return True

        if self._is_semantic_duplicate(text):
            return True

        self._seen_hashes.add(h)
        self._seen_texts[h] = text
        return False

    def _is_semantic_duplicate(self, text: str) -> bool:
        if self.vectorstore is None:
            return False

        results = self.vectorstore.similarity_search_with_score(
            text, k=self.k_candidates or 5
        )

        if not results:
            return False

        if self.similarity_threshold is None:
            return False

        top_docs = [
            doc for doc, score in results if score >= self.similarity_threshold
        ]
        if not top_docs:
            return False

        rerank_pairs = [(text, doc.page_content) for doc in top_docs]
        scores = self.reranker.predict(rerank_pairs)

        for _, score in enumerate(scores):
            if score >= self.rerank_threshold:
                return True

        return False

    def cluster_filter(
        self, texts: list[str], embeddings: np.ndarray
    ) -> list[str]:
        """Batch deduplicate a list of texts using
        clustering on their embeddings.
        """
        if len(texts) == 0:
            return []

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.1,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        cluster_map = {}
        for i, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = i
        return [texts[i] for i in sorted(cluster_map.values())]
