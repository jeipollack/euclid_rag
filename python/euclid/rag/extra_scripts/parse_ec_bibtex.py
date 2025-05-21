# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
Ingest publications into a FAISS vectorstore from the official EC BibTeX.
Each paper is embedded immediately after download and deleted afterward.
Includes a weekly ingestion scheduler using APScheduler.
"""

import re
from pathlib import Path

import requests
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

from euclid.rag.FAISS_vectorstore.VectorstoreEmbedder import (
    E5MpsEmbedder,
    load_or_create_vectorstore,
)


class EuclidBibIngestor:
    """Downloads and ingests new papers from the Euclid
    BibTeX file into the vectorstore.
    """

    BIB_URL = "https://eceb.astro.uni-bonn.de/public/Euclid.bib"

    def __init__(self, index_dir: Path, temp_dir: Path) -> None:
        """Initialize the ingestor."""
        self._index_dir = index_dir
        self._temp_dir = temp_dir
        self._embedder = E5MpsEmbedder()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> FAISS | None:
        """Load the FAISS vectorstore if it exists."""
        if (
            self._index_dir.exists()
            and (self._index_dir / "index.faiss").exists()
        ):
            return FAISS.load_local(
                str(self._index_dir),
                self._embedder,
                allow_dangerous_deserialization=True,
            )
        else:
            # Create vectorstore from any existing PDFs in temp_dir if present
            pdf_paths = list(self._temp_dir.glob("*.pdf"))
            if pdf_paths:
                return load_or_create_vectorstore(
                    self._index_dir, self._embedder, pdf_paths
                )
        return None

    def ingest_new_papers(self) -> None:
        """Ingests new papers into the vectorstore."""
        bib_entries = self._fetch_bibtex_entries()
        new_ingested = 0

        # Get list of already ingested paper filenames
        existing_sources = set()
        if self._vectorstore is not None:
            # Extract metadata 'source' from vectorstore's docs
            try:
                # No public method in LangChain's InMemoryDocstore
                # to access stored documents, accessing _dict directly
                # for vectorstore deduplication
                existing_sources = {
                    doc.metadata.get("source", "")
                    for doc in self._vectorstore.docstore._dict.values()  # noqa: SLF001
                    if "source" in doc.metadata
                }
            except Exception:
                # Fallback if docstore structure changes or is private
                existing_sources = set()

        for entry in bib_entries:
            arxiv_id = entry.get("eprint")
            title = entry.get("title")
            if not arxiv_id or not title:
                continue

            filename = self._format_filename(arxiv_id, title)

            if filename in existing_sources:
                continue

            filepath = self._temp_dir / filename

            if not self._download_pdf(arxiv_id, filepath):
                continue

            loader = PyMuPDFLoader(str(filepath))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(
                    chunks, self._embedder
                )
            else:
                self._vectorstore.add_documents(chunks)

            filepath.unlink(missing_ok=True)
            new_ingested += 1

            # Save after each paper
            self._vectorstore.save_local(str(self._index_dir))

    def _fetch_bibtex_entries(self) -> list[dict]:
        """Fetch BibTeX entries."""
        response = requests.get(self.BIB_URL, timeout=15)
        response.raise_for_status()
        parser = BibTexParser(common_strings=True)
        parser.customization = convert_to_unicode
        return parser.parse(response.text).entries

    def _download_pdf(self, arxiv_id: str, target_path: Path) -> bool:
        """Download a PDF from arXiv."""
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with target_path.open("wb") as f:
            f.write(response.content)
        return True

    @staticmethod
    def _format_filename(arxiv_id: str, title: str) -> str:
        """Format a filename for Arxiv."""
        clean_title = re.sub(r"[^\w\s]", "", title.lower())
        short = "_".join(clean_title.strip().split()[:6])
        return f"{arxiv_id}_{short}.pdf"


def run_bibtex_ingestion() -> None:
    """Run the BibTeX ingestion process."""
    index_dir = Path("rag/FAISS_vectorstore").resolve()
    temp_dir = Path("rag/downloaded").resolve()
    ingestor = EuclidBibIngestor(index_dir=index_dir, temp_dir=temp_dir)
    ingestor.ingest_new_papers()
