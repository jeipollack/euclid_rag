"""Module to ingest JSON-exported Redmine pages into a FAISS vectorstore."""

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from euclid.rag.extra_scripts.deduplication import (
    HashDeduplicator,
    SemanticSimilarityDeduplicator,
)
from euclid.rag.extra_scripts.vectorstore_embedder import Embedder
from euclid.rag.utils.config import load_config
from euclid.rag.utils.redmine_cleaner import RedmineCleaner

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


DEDUPLICATION_CONFIG: dict[str, str | float | int] = {
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "similarity_threshold": 0.8,
    "rerank_threshold": 0.85,
    "k_candidates": 5,
}


class EuclidJSONIngestor:
    """Ingest JSON-exported Redmine pages into a FAISS vectorstore."""

    def __init__(
        self, index_dir: Path, json_dir: Path, data_config: dict
    ) -> None:
        self._index_dir = index_dir
        self._json_dir = json_dir
        self._model_name = data_config.get(
            "embedding_model_name", "intfloat/e5-small-v2"
        )
        self._batch_size = data_config.get("embedding_batch_size", 16)
        self._embedder = Embedder(
            model_name=self._model_name, batch_size=self._batch_size
        )
        self._vectorstore = self._load_vectorstore()
        self._data_config = data_config
        self._cleaner = RedmineCleaner(
            max_chunk_length=self._data_config.get("chunk_size", 800)
        )
        logger.info(
            "[ING] EuclidJSONIngestor initialized"
            f"with model={self._model_name}",
            f"index_dir={self._index_dir}, json_dir={self._json_dir}",
        )

    def _load_vectorstore(self) -> FAISS | None:
        index_file = self._index_dir / "index.faiss"
        if index_file.exists():
            logger.info(
                f"[ING] Loading existing FAISS index from {self._index_dir}"
            )
            return FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )
        logger.info("[ING] No existing vectorstore found, starting fresh.")
        return None

    def ingest_redmine_pages(self) -> None:
        logger.info("[ING] Starting ingestion of Redmine pages...")
        dedup_filter_hash = HashDeduplicator()

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents([], self._embedder)

        dedup_filter_semantic = SemanticSimilarityDeduplicator(
            vectorstore=self._vectorstore,
            reranker_model=str(DEDUPLICATION_CONFIG["reranker_model"]),
            similarity_threshold=float(
                DEDUPLICATION_CONFIG["similarity_threshold"]
            ),
            rerank_threshold=float(DEDUPLICATION_CONFIG["rerank_threshold"]),
            k_candidates=int(DEDUPLICATION_CONFIG["k_candidates"]),
        )

        existing_sources = self._get_existing_sources()
        logger.info(f"[ING] Loaded {len(existing_sources)} existing sources.")

        for json_path in self._json_dir.glob("*.json"):
            with json_path.open(encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.exception(f"[ING] Decoding error for {json_path}")
                    continue

            pages = data if isinstance(data, list) else data.get("pages", [])
            prepared_docs = self._cleaner.prepare_for_ingestion(pages)

            for entry in prepared_docs:
                metadata: dict[str, Any] = entry.get("metadata", {})
                page_id = str(metadata.get("project_id", "unknown")).strip()

                if page_id in existing_sources:
                    logger.debug(
                        f"[ING] Page {page_id} already ingested, skipping."
                    )
                    continue

                doc = Document(
                    page_content=entry.get("content", ""),
                    metadata={**metadata, "source": "Redmine"},
                )

                if dedup_filter_hash.filter(doc.page_content):
                    logger.debug("[ING] Chunk filtered by hash deduplication.")
                    continue
                if dedup_filter_semantic.filter(doc.page_content):
                    logger.debug(
                        "[ING] Chunk filtered by semantic deduplication."
                    )
                    continue

                self._vectorstore.add_documents([doc])

                self._vectorstore.save_local(str(self._index_dir))
                self._vectorstore = FAISS.load_local(
                    str(self._index_dir),
                    self._embedder,
                    allow_dangerous_deserialization=True,
                )

                logger.info(
                    f"[ING] Ingested page: {metadata.get('hierarchy')}"
                )

        logger.info("[ING] Redmine ingestion completed.")

    def _get_existing_sources(self) -> set[str]:
        existing_sources: set[str] = set()
        if self._vectorstore is not None:
            store = self._vectorstore.docstore
            for doc_id in self._vectorstore.index_to_docstore_id.values():
                raw_result: Any = store.search(doc_id)
                if not isinstance(raw_result, list):
                    continue
                for doc in raw_result:
                    if isinstance(doc, Document):
                        source = doc.metadata.get("source")
                        if source:
                            existing_sources.add(source)
        return existing_sources


def run_redmine_ingestion(config: dict) -> None:
    """Run the Redmine JSON ingestion script."""
    index_dir = Path(config["vector_store"]["index_dir"]).resolve()
    json_dir = Path(config["data"]["redmine_json_dir"]).resolve()
    data_config = config.get("data", {})

    ingestor = EuclidJSONIngestor(
        index_dir=index_dir,
        json_dir=json_dir,
        data_config=data_config,
    )
    ingestor.ingest_redmine_pages()


if __name__ == "__main__":
    """
    Manual ingestion entrypoint for Redmine pages.
    """
    config = load_config(Path("rag/app_config.yaml"))
    run_redmine_ingestion(config)
