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


class HashDeduplicator:
    """
    Deduplicator using SHA256 hashes for exact match detection.

    Tracks seen inputs by their hash and filters out exact duplicates.
    """

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def filter(self, text: str) -> bool:
        """
        Check if the text has already been seen via hashing.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        bool
            True if the text is a duplicate, False otherwise.
        """
        h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False


class SemanticSimilarityDeduplicator:
    """
    Deduplicator using semantic similarity and optional reranking.

    Uses FAISS to find similar texts and CrossEncoder to refine scoring.
    Texts are considered duplicates if both thresholds are exceeded.
    """

    def __init__(
        self,
        vectorstore: FAISS | None,
        reranker_model: str,
        similarity_threshold: float,
        rerank_threshold: float,
        k_candidates: int = 5,
    ) -> None:
        self.vectorstore = vectorstore
        self.similarity_threshold = similarity_threshold
        self.rerank_threshold = rerank_threshold
        self.k_candidates = k_candidates
        self.reranker = CrossEncoder(reranker_model)

    def filter(self, text: str) -> bool:
        """
        Check if the text is semantically similar to existing vectors
        based on a similarity and reranking threshold.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        bool
            True if semantically duplicate, False otherwise.
        """
        if self.vectorstore is None or self.vectorstore.index is None:
            return False

        results = self.vectorstore.similarity_search_with_score(
            text, k=self.k_candidates
        )
        if not results:
            return False

        top_docs = [
            doc for doc, score in results if score >= self.similarity_threshold
        ]
        if not top_docs:
            return False

        rerank_pairs = [(text, doc.page_content) for doc in top_docs]
        scores = self.reranker.predict(rerank_pairs)

        return any(score >= self.rerank_threshold for score in scores)


class ChunkClusterer:
    """
    Cluster embedding vectors and return one representative text per cluster.

    This class uses clustering on embedding vectors to identify
    similar groups of text chunks.

    Parameters
    ----------
    distance_threshold : float, optional
        Maximum cosine distance between elements in a cluster.
        Lower values produce tighter, more conservative clusters.
        Default is 0.1.
    """

    def __init__(self, distance_threshold: float = 0.1) -> None:
        self.distance_threshold = distance_threshold

    def filter(self, texts: list[str], embeddings: np.ndarray) -> list[str]:
        """
        Filter semantically similar texts using clustering.

        Parameters
        ----------
        texts : list of str
            The input text chunks.
        embeddings : np.ndarray
            The embedding vectors.

        Returns
        -------
        list of str
            One text per cluster.
        """
        if len(texts) == 0:
            return []

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        cluster_map = {}
        for i, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = i
        return [texts[i] for i in sorted(cluster_map.values())]
