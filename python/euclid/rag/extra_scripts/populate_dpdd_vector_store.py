#!/usr/bin/env python
# ruff: noqa: ERA001, C901, PLR0912, E501

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Ingest DPDD Euclid information into a FAISS vectorstore."""

import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from euclid.rag.FAISS_vectorstore.vectorstore_embedder import E5MpsEmbedder

logger = logging.getLogger(__name__)

# Hard-coded limit of topics to ingest
# It's only for debug/testing purposes
# In production we will set it to 0 (no limit) to ingest all topics
TOPICS_NUMBER_LIMIT = 0  # 0 = no limit


class EuclidDPDDIngestor:
    """Downloads and ingests DPDD data into the vectorstore."""

    def __init__(self, index_dir: Path) -> None:
        """Initialize the ingestor."""
        self._index_dir = index_dir
        self._embedder = E5MpsEmbedder()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = self._load_vectorstore()
        # Load banned sections config
        config_path = Path(__file__).parent / "dpdd_config.yaml"
        with Path.open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.banned_sections = {
            name.lower() for name in cfg["banned_sections"]["names"]
        }
        # self.banned_full_links = set(cfg['banned_sections']['full_links'])
        self.topics = cfg["topics"]
        # TODO (paulz): for the moment it's hardcoded, but we should use config
        self.base_url = "https://euclid.esac.esa.int/dr/q1/dpdd/"
        # self.base_url = 'https://euclid.esac.esa.int/msp/dpdd/live/'
        self.scrape_all = True

    def _load_vectorstore(self) -> FAISS | None:
        """Load the FAISS vectorstore if it exists."""
        if (
            self._index_dir.exists()
            and (self._index_dir / "index.faiss").exists()
        ):
            try:
                return FAISS.load_local(
                    str(self._index_dir),
                    self._embedder,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.warning("Failed to load vectorstore, rebuilding: %s", e)

        return None

    def ingest_new_data(self) -> None:
        """Ingests new data into the vectorstore."""
        texts, metadatas = self._fetch_dpdd_entries()
        new_ingested = 0

        # Get list of already ingested paper filenames
        existing_sources = set()
        if self._vectorstore is not None:
            # Extract metadata 'source' from vectorstore's docs
            try:
                existing_sources = {
                    doc.metadata.get("source", "")
                    for doc in self._vectorstore.docstore._dict.values()  # noqa: SLF001
                    if "source" in doc.metadata
                }
            except Exception:
                # Fallback if docstore structure changes or is private
                existing_sources = set()

        for text, metadata in zip(texts, metadatas, strict=False):
            if metadata.get("source", "") in existing_sources:
                logger.warning(
                    "Skipping already ingested source: %s",
                    metadata.get("source"),
                )
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            docs = [Document(page_content=text, metadata=metadata)]
            chunks = splitter.split_documents(docs)

            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(
                    chunks, self._embedder
                )
            else:
                self._vectorstore.add_documents(chunks)

            new_ingested += 1

            self._vectorstore.save_local(str(self._index_dir))

    def _get_all_topics_for_baseurl(self) -> list[dict]:
        """Get all topics for baseurl."""
        results: list[dict] = []
        try:
            # Get the main page
            response = requests.get(self.base_url, timeout=15)
            try:
                response.raise_for_status()  # exception for bad status codes
            except requests.exceptions.HTTPError:
                if response.status_code == 404:
                    logger.warning(
                        f"DPDD page not found (404 code): {self.base_url}"
                    )
                    return []
                logger.exception(f"Failed to fetch DPDD page {self.base_url}")
                return []
            # Parse the main page
            soup = BeautifulSoup(response.text, "html.parser")
            # Find main content div
            main_div = soup.find("div", class_="body", role="main")
            if not main_div:
                logger.warning(
                    f"Main content div not found for {self.base_url}"
                )
                return results
            # Iterate over sections except indices-and-tables
            for section in main_div.find_all("section"):
                if section.get("id") == "indices-and-tables":
                    continue
                # Find links with class "reference internal"
                for link in section.find_all(
                    "a", class_="reference internal", href=True
                ):
                    name = link.text.strip()
                    href = link["href"]
                    results.append({"name": name, "link": href})
        except requests.exceptions.RequestException:
            logger.exception(f"Failed to fetch DPDD page {self.base_url}")
            return results

        return results

    def _fetch_dpdd_entries(self) -> list[dict]:
        """Fetch DPDD entries."""
        texts, metadatas = [], []
        # Scrape All
        if self.scrape_all:
            list_of_urls = self._get_all_topics_for_baseurl()
        else:
            list_of_urls = self.topics

        # Scrape topics
        for i, topic in enumerate(list_of_urls):
            if TOPICS_NUMBER_LIMIT > 0 and i >= TOPICS_NUMBER_LIMIT:
                break
            name = topic["name"]
            url = urljoin(self.base_url, topic["link"])
            results = self._get_dpdd_sections(url, name)
            if results:
                for item in results:
                    texts.append(item["content"])
                    metadatas.append(
                        {k: v for k, v in item.items() if k != "content"}
                    )
        return texts, metadatas

    def _get_dpdd_sections(self, url: str, name: str) -> list[dict]:
        """Get DPDD sections."""
        # get section names and and contents
        results = []
        try:
            # Get the main page
            response = requests.get(url, timeout=15)
            try:
                # Raise an exception for bad status codes
                response.raise_for_status()
            except requests.exceptions.HTTPError:
                if response.status_code == 404:
                    logger.warning(f"DPDD page not found (404 code): {url}")
                    return []
                logger.exception(f"Failed to fetch DPDD page {url}")
                return []
            # Parse the main page
            soup = BeautifulSoup(response.text, "html.parser")
            # Find all links that have class "reference internal"
            subtopic_links = soup.find_all(
                "a", class_="reference internal", href=True
            )

            # Process each subtopic
            for link in subtopic_links:
                subtopic_url = link["href"]
                subtopic_text = link.text.strip()

                # Handle relative URLs using urljoin
                if not subtopic_url.startswith(("http://", "https://")):
                    subtopic_url = urljoin(url, subtopic_url)

                try:
                    # Get the subtopic page
                    subtopic_response = requests.get(subtopic_url, timeout=15)
                    subtopic_response.raise_for_status()

                    # Parse the subtopic page
                    subtopic_soup = BeautifulSoup(
                        subtopic_response.text, "html.parser"
                    )

                    # Find all sections in the subtopic page
                    sections = subtopic_soup.find_all(["section"])
                    for section in sections:
                        # Get section name from first heading in the section
                        section_name = section.find(["h1", "h2", "h3"])
                        if section_name:
                            section_text = (
                                section_name.text.replace("¶", "")
                                .lower()
                                .strip()
                            )

                            if (
                                section_text is None
                                or section_text in self.banned_sections
                            ):
                                logger.warning(
                                    "Skipping banned section: %s in %s",
                                    section_text,
                                    subtopic_url,
                                )
                                continue

                            # remove the header so it's not included in content
                            section_name.extract()

                            content = section.get_text(
                                separator=" ", strip=True
                            )

                            # Skip placeholder content starting with $ and no spaces
                            # (like $PrintDataProductName or $PrintDataProductCustodian etc)
                            # Such placeholder are no good for us
                            if content.startswith("$") and " " not in content:
                                # Skipping wrong placeholder section: {content} in {subtopic_url}
                                continue

                            if content:
                                # Get direct link to the section using its id
                                section_id = section.get(
                                    "id"
                                ) or section_name.get("id")
                                if section_id:
                                    section_url_with_anchor = (
                                        subtopic_url + "#" + section_id
                                    )
                                else:
                                    section_url_with_anchor = subtopic_url
                                metadata = {
                                    "source": section_url_with_anchor,
                                    "section": section_text,
                                    "subtopic": subtopic_text,
                                    "source_name": name,
                                    "category": "DPDD",
                                }
                                results.append(
                                    {"content": content, **metadata}
                                )

                except requests.RequestException as e:
                    logger.warning(
                        "Error processing subtopic %s: %s", subtopic_url, e
                    )

        except requests.RequestException as e:
            logger.warning("Error accessing main page %s: %s", url, e)

        return results


def run_dpdd_ingestion() -> None:
    """Run the DPDD ingestion process."""
    # index_dir = Path("VectorStore_indexes").resolve()
    index_dir = Path("rag/FAISS_vectorstore").resolve()
    ingestor = EuclidDPDDIngestor(index_dir=index_dir)
    ingestor.ingest_new_data()


if __name__ == "__main__":
    run_dpdd_ingestion()
