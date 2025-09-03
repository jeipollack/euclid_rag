"""Module to ingest JSON-exported pages into a FAISS vector store."""

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from euclid.rag.extra_scripts.deduplication import HashDeduplicator, SemanticSimilarityDeduplicator
from euclid.rag.extra_scripts.vectorstore_embedder import Embedder
from euclid.rag.utils.config import load_config
from euclid.rag.utils.redmine_cleaner import RedmineCleaner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


DEDUPLICATION_CONFIG: dict[str, str | float | int] = {
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "similarity_threshold": 0.8,
    "rerank_threshold": 0.85,
    "k_candidates": 5,
}


class JSONIngestor:
    """
    Ingest JSON-exported pages into a FAISS vector store.
    JSON structure should be as follows:
    {
    "content": "Full text of the page...",
    "metadata": {
        "field1": "",
        "field2": "",
        ...
        }
    },...
    """

    def __init__(self, index_dir: Path, json_dir: Path, cleaner: RedmineCleaner, data_config: dict) -> None:
        self._index_dir = index_dir
        self._json_dir = json_dir
        self._model_name = data_config.get("embedding_model_name", "intfloat/e5-small-v2")
        self._batch_size = data_config.get("embedding_batch_size", 16)
        self._key_fields = data_config.get("deduplication_key_fields", ["hierarchy", "title"])
        self._source = data_config.get("source", "redmine")
        self._embedder = Embedder(model_name=self._model_name, batch_size=self._batch_size)
        self._vectorstore = self._load_vectorstore()
        self._data_config = data_config
        self._cleaner = cleaner
        logger.info(
            "[INGEST] JSONIngestor initialized"
            f"(model={self._model_name}, index_dir={self._index_dir}, json_dir={self._json_dir})"
        )

    def _load_vectorstore(self) -> FAISS | None:
        """Load an existing FAISS vector store from the specified index directory."""
        index_file = self._index_dir / "index.faiss"
        if index_file.exists():
            logger.info(f"[INGEST] Loading existing FAISS index from {self._index_dir}")
            return FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )
        logger.info("[INGEST] No existing vector store found at %s. Initializing a new one.", self._index_dir)
        return None

    @staticmethod
    def _page_source_key(meta: dict[str, Any], content: str, key_fields: list[str]) -> str:
        """Compute a unique key to identify a chunk, using specific metadata fields and content."""
        key_parts = [str(meta.get(field, "")).strip() for field in key_fields]
        content_digest = hashlib.md5(content.encode("utf-8")).hexdigest()  # noqa: S324 - hash is not for security
        key_parts.append(content_digest)
        return "::".join(key_parts)

    def _get_existing_source_keys(self) -> set[str]:
        """Retrieve existing source keys from the vector store to avoid re-ingestion."""
        keys: set[str] = set()
        if self._vectorstore is None:
            return keys

        store = self._vectorstore.docstore
        for doc_id in self._vectorstore.index_to_docstore_id.values():
            doc = store.search(doc_id)
            if not isinstance(doc, Document):
                raise TypeError(f"Expected a Document from docstore, got {type(doc)}")
            k = doc.metadata.get("source_key")
            if k:
                keys.add(k)
        return keys

    def ingest_json_files(self) -> None:
        """Ingest documents from JSON files into the FAISS vector store."""
        logger.info("[INGEST] Starting ingestion of JSON contents...")

        dedup_filter_hash = HashDeduplicator()
        dedup_filter_semantic = SemanticSimilarityDeduplicator(
            vectorstore=self._vectorstore,
            reranker_model=str(DEDUPLICATION_CONFIG["reranker_model"]),
            similarity_threshold=float(DEDUPLICATION_CONFIG["similarity_threshold"]),
            rerank_threshold=float(DEDUPLICATION_CONFIG["rerank_threshold"]),
            k_candidates=int(DEDUPLICATION_CONFIG["k_candidates"]),
        )

        existing_keys = self._get_existing_source_keys()
        logger.info(f"[INGEST] Loaded {len(existing_keys)} existing source keys.")

        for json_path in self._json_dir.glob("*.json"):
            try:
                with json_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.exception(f"[INGEST] Failed to read/parse {json_path}")
                continue

            if not isinstance(data, list):
                raise TypeError(f"JSON file {json_path} does not contain a list of pages.")

            prepared_docs = self._cleaner.prepare_for_ingestion(data)

            for entry in prepared_docs:
                metadata: dict[str, Any] = entry.get("metadata", {}) or {}
                content = entry.get("content", "") or ""

                # Skip empty or already ingested contents
                if not content.strip():
                    logger.debug("[INGEST] Empty content, skipping chunk.")
                    continue
                source_key = JSONIngestor._page_source_key(meta=metadata, content=content, key_fields=self._key_fields)
                if source_key in existing_keys:
                    logger.debug(f"[INGEST] Already ingested, skipping. key={source_key}")
                    continue
                if dedup_filter_hash.filter(content):
                    logger.debug("[INGEST] Chunk filtered by hash deduplication.")
                    continue
                if self._vectorstore and dedup_filter_semantic.filter(content):
                    logger.debug("[INGEST] Chunk filtered by semantic deduplication.")
                    continue

                doc = Document(
                    page_content=content,
                    metadata={**metadata, "source": self._source, "source_key": source_key},
                )
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents([doc], self._embedder)
                    dedup_filter_semantic.vectorstore = self._vectorstore
                else:
                    self._vectorstore.add_documents([doc])

                self._vectorstore.save_local(str(self._index_dir))
                existing_keys.add(source_key)
                logger.info(f"[INGEST] Ingested page: {source_key}")

        logger.info("[INGEST] Ingestion completed.")


def run_redmine_ingestion(config: dict) -> None:
    """Run the ingestion process for Redmine pages using the provided configuration."""
    index_dir = Path(config["vector_store"]["redmine_index_dir"]).resolve()
    json_dir = Path(config["json_data"]["redmine_json_dir"]).resolve()
    data_config = config.get("json_data", {})
    ingestor = JSONIngestor(
        index_dir=index_dir,
        json_dir=json_dir,
        cleaner=RedmineCleaner(max_chunk_length=data_config.get("chunk_size", 800)),
        data_config=data_config,
    )
    ingestor.ingest_json_files()


if __name__ == "__main__":
    config = load_config(Path("python/euclid/rag/app_config.yaml"))
    run_redmine_ingestion(config)
