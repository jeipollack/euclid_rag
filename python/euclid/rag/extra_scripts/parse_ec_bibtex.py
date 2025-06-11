# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
Ingest publications into a FAISS vectorstore from the official EC BibTeX.
Each paper is embedded immediately after download and deleted afterward.
"""

import re
from pathlib import Path

import requests
import yaml
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

DEDUPLICATION_CONFIG = {
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "similarity_threshold": 0.8,
    "rerank_threshold": 0.85,
    "k_candidates": 5,
}


def load_config(path: Path) -> dict:
    """Load YAML configuration from the given path."""
    with path.open("r") as f:
        return yaml.safe_load(f)


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
        self._embedder = Embedder()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = self._load_vectorstore()
        self._bib_url = data_config.get("bibtex_url")
        self._arxiv_pdf_url = data_config.get("arxiv_pdf_base_url")
        self._data_config = data_config

    def _load_vectorstore(self) -> FAISS | None:
        """Load the FAISS vectorstore if it exists."""
        if (self._index_dir / "index.faiss").exists():
            return FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )
        else:
            pdf_paths = list(self._temp_dir.glob("*.pdf"))
            if pdf_paths:
                return load_or_create_vectorstore(
                    self._index_dir, self._embedder, pdf_paths
                )
        return None

    def ingest_new_papers(self) -> None:
        """Ingests new papers into the vectorstore."""
        dedup_filter_hash = HashDeduplicator()

        dedup_filter_semantic = SemanticSimilarityDeduplicator(
            vectorstore=self._vectorstore,
            reranker_model=DEDUPLICATION_CONFIG["reranker_model"],
            similarity_threshold=DEDUPLICATION_CONFIG["similarity_threshold"],
            rerank_threshold=DEDUPLICATION_CONFIG["rerank_threshold"],
            k_candidates=DEDUPLICATION_CONFIG["k_candidates"],
        )

        bib_entries = self._fetch_bibtex_entries()
        self._reload_vectorstore()
        existing_sources = self._get_existing_sources()

        for entry in bib_entries:
            if not self._should_process(entry, existing_sources):
                continue

            filename = self._format_filename(entry["eprint"], entry["title"])
            filepath = self._temp_dir / filename

            if not self._download_pdf(entry["eprint"], filepath):
                continue

            chunks = self._load_and_split_pdf(filepath)
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

    def _reload_vectorstore(self) -> None:
        if self._vectorstore is not None:
            self._vectorstore = FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )

    def _get_existing_sources(self) -> set[str]:
        existing_sources: set[str] = set()
        if self._vectorstore is not None:
            store = self._vectorstore.docstore
            for doc_id in self._vectorstore.index_to_docstore_id.values():
                for doc in store.search(doc_id):
                    if isinstance(doc, Document):
                        source = doc.metadata.get("source")
                        if source:
                            existing_sources.add(source)
        return existing_sources

    def _should_process(self, entry: dict, existing_sources: set[str]) -> bool:
        arxiv_id = entry.get("eprint")
        title = entry.get("title")
        if not arxiv_id or not title:
            return False
        filename = self._format_filename(arxiv_id, title)
        return filename not in existing_sources

    def _load_and_split_pdf(self, filepath: Path) -> list[Document]:
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
        for chunk in chunks:
            if dedup_filter_hash.filter(chunk.page_content):
                continue
            if dedup_filter_semantic.filter(chunk.page_content):
                continue
            chunk.metadata.update(paper_meta)
            chunk.metadata["category"] = "publication"
            chunk.metadata["source"] = filename
            filtered_chunks.append(chunk)
        return filtered_chunks

    def _add_to_vectorstore(self, chunks: list[Document]) -> None:
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
        if self._vectorstore is None:
            return

        store = self._vectorstore.docstore
        shown = 0

        if not hasattr(store, "search"):
            return

        index_ids = getattr(self._vectorstore, "index_to_docstore_id", {})
        if not isinstance(index_ids, dict):
            return

        for doc_id in index_ids.values():
            for doc in store.search(doc_id):
                if not isinstance(doc, Document):
                    continue
                source = doc.metadata.get("source")
                if isinstance(source, str) and source == filename:
                    shown += 1
                    if shown >= 3:
                        return

    def _fetch_bibtex_entries(self) -> list[dict]:
        """Fetch BibTeX entries."""
        response = requests.get(self._bib_url, timeout=60)
        response.raise_for_status()
        parser = BibTexParser(common_strings=True)
        parser.customization = convert_to_unicode
        return parser.parse(response.text).entries

    def _download_pdf(self, arxiv_id: str, target_path: Path) -> bool:
        """Download a PDF from arXiv."""
        url = f"{self._arxiv_pdf_url}{arxiv_id}.pdf"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with target_path.open("wb") as f:
            f.write(response.content)
        return True

    def _entry_metadata(self, entry: dict) -> dict:
        """Metadata to keep from a BibTeX entry."""
        title = entry.get("title", "").strip("{}")
        authors = entry.get("author", "")
        year = entry.get("year")
        kw = entry.get("keywords", "")
        # prefer ADS URL; fall back to arXiv PDF link
        url = (
            entry.get("adsurl")
            or f"{self._arxiv_pdf_url}{entry.get('eprint')}.pdf"
        )

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


def run_bibtex_ingestion() -> None:
    """Run the bibtex ingestion script."""
    config = load_config(Path("rag/app_config.yaml"))

    index_dir = Path(config["vector_store"]["index_dir"]).resolve()
    temp_dir = Path("rag/downloaded").resolve()
    data_config = config.get("data", {})

    ingestor = EuclidBibIngestor(
        index_dir=index_dir,
        temp_dir=temp_dir,
        data_config=data_config,
    )
    ingestor.ingest_new_papers()


if __name__ == "__main__":
    """
    Manual ingestion call.
    """
    run_bibtex_ingestion()
