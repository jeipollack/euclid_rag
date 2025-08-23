# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
Ingest publications into a FAISS vectorstore from the official EC BibTeX.
Each paper is embedded immediately after download and deleted afterward.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import requests
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from euclid.rag.extra_scripts.deduplication import (
    HashDeduplicator,
    SemanticSimilarityDeduplicator,
)
from euclid.rag.extra_scripts.vectorstore_embedder import (
    Embedder,
    load_or_create_vectorstore,
)
from euclid.rag.utils.config import load_config

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


class EuclidBibIngestor:
    """Downloads and ingests new papers from the Euclid
    BibTeX file into the vectorstore.
    """

    def __init__(
        self,
        index_dir: Path,
        temp_dir: Path,
        data_config: dict,
    ) -> None:
        """Initiate the ingestor."""
        self._index_dir = index_dir
        self._temp_dir = temp_dir
        self._model_name = data_config.get(
            "embedding_model_name", "intfloat/e5-small-v2"
        )
        self._batch_size = data_config.get("embedding_batch_size", 16)
        self._bib_url = data_config.get("bibtex_url")
        self._arxiv_pdf_url = data_config.get("arxiv_pdf_base_url")
        self._embedder = Embedder(
            model_name=self._model_name, batch_size=self._batch_size
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = self._load_vectorstore()
        self._data_config = data_config

    def _load_vectorstore(self) -> FAISS | None:
        """Load the FAISS vectorstore if it exists."""
        if (self._index_dir / "index.faiss").exists():
            logger.info(
                "Loading existing vectorstore from %s", self._index_dir
            )
            return FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )
        else:
            pdf_paths = list(self._temp_dir.glob("*.pdf"))
            if pdf_paths:
                logger.info(
                    "Creating new vectorstore from PDFs in %s", self._temp_dir
                )
                return load_or_create_vectorstore(
                    self._index_dir, self._embedder, pdf_paths
                )
        return None

    def ingest_new_papers(self) -> None:
        """Ingests new papers into the vectorstore."""
        logger.info("Starting ingestion of new papers.")
        dedup_filter_hash = HashDeduplicator()

        bib_entries = self._fetch_bibtex_entries()
        logger.info("Fetched %d BibTeX entries.", len(bib_entries))
        existing_sources = self._get_existing_sources()
        logger.info(
            "Found %d existing sources in vectorstore.", len(existing_sources)
        )

        for entry in bib_entries:
            if not self._should_process(entry, existing_sources):
                continue

            filename = self._format_filename(entry["eprint"], entry["title"])
            filepath = self._temp_dir / filename

            if not self._download_pdf(entry["eprint"], filepath):
                continue

            chunks = self._load_and_split_pdf(filepath)

            # Init of vectorstore after first chunks are available
            if self._vectorstore is None:
                logger.info("Creating new vectorstore from first paper.")
                self._vectorstore = FAISS.from_documents(
                    chunks, self._embedder
                )
                self._vectorstore.save_local(str(self._index_dir))
                self._reload_vectorstore()

            dedup_filter_semantic = SemanticSimilarityDeduplicator(
                vectorstore=self._vectorstore,
                reranker_model=str(DEDUPLICATION_CONFIG["reranker_model"]),
                similarity_threshold=float(
                    DEDUPLICATION_CONFIG["similarity_threshold"]
                ),
                rerank_threshold=float(
                    DEDUPLICATION_CONFIG["rerank_threshold"]
                ),
                k_candidates=int(DEDUPLICATION_CONFIG["k_candidates"]),
            )

            filtered_chunks = self._filter_and_enrich_chunks(
                chunks,
                entry,
                filename,
                dedup_filter_hash,
                dedup_filter_semantic,
            )

            if filtered_chunks:
                self._add_to_vectorstore(filtered_chunks)
                self._log_sampled_chunks(filename)

            filepath.unlink(missing_ok=True)

        if self._vectorstore is None:
            logger.warning(
                "No valid papers were ingested,vectorstore was not created."
            )
            raise RuntimeError(
                "No valid papers were ingested,vectorstore was not created."
            )
        logger.info("Ingestion of new papers complete.")

    def _reload_vectorstore(self) -> None:
        if self._vectorstore is not None:
            logger.info("Reloading vectorstore from %s", self._index_dir)
            self._vectorstore = FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )

    def _get_existing_sources(self) -> set[str]:
        """Return set of existing 'source' values from the vectorstore."""
        existing_sources: set[str] = set()
        if self._vectorstore is not None:
            store = self._vectorstore.docstore
            for doc_id in self._vectorstore.index_to_docstore_id.values():
                docs: Any = store.search(doc_id)
                if not isinstance(docs, list):
                    continue
                docs_list: list[Document] = [
                    d for d in docs if isinstance(d, Document)
                ]
                for doc in docs_list:
                    source = doc.metadata.get("source")
                    if isinstance(source, str):
                        existing_sources.add(source)
        return existing_sources

    def _should_process(self, entry: dict, existing_sources: set[str]) -> bool:
        arxiv_id = entry.get("eprint")
        title = entry.get("title")
        if not arxiv_id or not title:
            logger.debug(
                "Skipping entry due to missing arxiv_id or title: %s", entry
            )
            return False
        filename = self._format_filename(arxiv_id, title)
        if filename in existing_sources:
            logger.debug("Skipping existing paper: %s", filename)
            return False
        return True

    def _load_and_split_pdf(self, filepath: Path) -> list[Document]:
        logger.info("Loading and splitting PDF: %s", filepath)
        loader = PyMuPDFLoader(str(filepath))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._data_config.get("chunk_size", 800),
            chunk_overlap=self._data_config.get("chunk_overlap", 100),
        )
        return splitter.split_documents(docs)

    def _filter_and_enrich_chunks(
        self,
        chunks: list[Document],
        entry: dict,
        filename: str,
        dedup_filter_hash: HashDeduplicator,
        dedup_filter_semantic: SemanticSimilarityDeduplicator,
    ) -> list[Document]:
        filtered_chunks = []
        paper_meta = self._entry_metadata(entry)
        hierarchy_root = f"{paper_meta['title']}"  # niveau 0 = le papier
        for idx, chunk in enumerate(chunks):
            if dedup_filter_hash.filter(chunk.page_content):
                logger.debug("Skipping chunk due to hash deduplication.")
                continue
            if dedup_filter_semantic.filter(chunk.page_content):
                logger.debug("Skipping chunk due to semantic similarity.")
                continue
            chunk.metadata.update(paper_meta)
            chunk.metadata.update(
                {
                    "category": "publication",
                    "source": filename,
                    "hierarchy": f"{hierarchy_root}/chunk_{idx}",
                }
            )
            filtered_chunks.append(chunk)
        logger.info(
            f"Filtered {len(chunks)} chunks, {len(filtered_chunks)} remaining."
        )
        return filtered_chunks

    def _add_to_vectorstore(self, chunks: list[Document]) -> None:
        logger.info("Adding %d chunks to vectorstore.", len(chunks))
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(chunks, self._embedder)
        else:
            self._vectorstore.add_documents(chunks)
        self._vectorstore.save_local(str(self._index_dir))
        self._vectorstore = FAISS.load_local(
            str(self._index_dir),
            self._embedder,
            allow_dangerous_deserialization=True,
        )

    def _log_sampled_chunks(self, filename: str) -> None:
        """Log (or inspect) up to 3 chunks for a given file from the
        vectorstore.
        """
        if self._vectorstore is None:
            return

        store = self._vectorstore.docstore
        shown = 0

        index_ids = getattr(self._vectorstore, "index_to_docstore_id", {})
        if not isinstance(index_ids, dict):
            return

        for doc_id in index_ids.values():
            docs: Any = store.search(doc_id)
            if not isinstance(docs, list):
                continue

            docs_list: list[Document] = [
                d for d in docs if isinstance(d, Document)
            ]
            for doc in docs_list:
                source = doc.metadata.get("source")
                if isinstance(source, str) and source == filename:
                    logger.debug(
                        "Sample chunk for %s: %s",
                        filename,
                        doc.page_content[:200],
                    )
                    shown += 1
                    if shown >= 3:
                        return

    def _fetch_bibtex_entries(self) -> list[dict]:
        """Fetch BibTeX entries."""
        if not isinstance(self._bib_url, str):
            raise TypeError("Missing or invalid 'bibtex_url' in config.")

        logger.info("Fetching BibTeX entries from %s", self._bib_url)
        response = requests.get(self._bib_url, timeout=60)
        response.raise_for_status()
        parser = BibTexParser(common_strings=True)
        parser.customization = convert_to_unicode
        return parser.parse(response.text).entries

    def _download_pdf(self, arxiv_id: str, target_path: Path) -> bool:
        """Download a PDF from arXiv."""
        if not isinstance(self._arxiv_pdf_url, str):
            raise TypeError("Missing or invalid 'arxiv pdf url' in config.")

        url = f"{self._arxiv_pdf_url}{arxiv_id}.pdf"
        logger.info("Downloading PDF from %s", url)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with target_path.open("wb") as f:
                f.write(response.content)
            logger.info("Successfully downloaded %s", target_path)
            return True  # noqa: TRY300
        except requests.exceptions.RequestException:
            logger.exception("Failed to download %s", url)
            return False

    def _entry_metadata(self, entry: dict) -> dict:
        """Metadata to keep from a BibTeX entry."""
        title = entry.get("title", "").strip("{}")
        authors = entry.get("author", "")
        year = entry.get("year")
        kw = entry.get("keywords", "")
        # prefer ADS URL; fall back to arXiv PDF link
        eprint = entry.get("eprint")
        if not isinstance(eprint, str):
            raise TypeError("Missing or invalid eprint in BibTeX entry.")

        url = entry.get("adsurl") or f"{self._arxiv_pdf_url}{eprint}.pdf"

        return {
            "title": title,
            "authors": authors,
            "year": int(year) if year else None,
            "keywords": kw,
            "url": url,
        }

    @staticmethod
    def _format_filename(arxiv_id: str, title: str) -> str:
        """Format a filename for Arxiv."""
        clean_title = re.sub(r"[^\w\s]", "", title.lower())
        short = "_".join(clean_title.strip().split()[:6])
        return f"{arxiv_id}_{short}.pdf"


def run_bibtex_ingestion(config: dict) -> None:
    """Run the bibtex ingestion script."""
    logger.info("Starting BibTeX ingestion process.")
    index_dir = Path(config["vector_store"]["publication_index_dir"]).resolve()
    temp_dir = Path("tmp").resolve()
    data_config = config.get("data", {})

    ingestor = EuclidBibIngestor(
        index_dir=index_dir,
        temp_dir=temp_dir,
        data_config=data_config,
    )
    ingestor.ingest_new_papers()
    logger.info("BibTeX ingestion process finished.")


def main() -> None:
    """Run the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest publications from the Euclid BibTeX file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="python/euclid/rag/app_config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_bibtex_ingestion(config)


if __name__ == "__main__":
    # Run the ingestion script if this file is executed directly
    # This allows for easy command-line execution
    # and testing without needing to import it in another script.
    main()
