#!/usr/bin/env python
# ruff: noqa: ERA001, C901, PLR0912, E501

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Ingest Euclid Science Ground Segment Data Product Description Document (DPDD).

This script downloads the DPDD from the Euclid website, processes it, and
ingests the data into a FAISS vectorstore for use in the Euclid RAG system.

"""

import argparse
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import AttributeValueList, Tag
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from euclid.rag.extra_scripts.vectorstore_embedder import Embedder
from euclid.rag.utils.config import load_config

logger = logging.getLogger(__name__)


class EuclidDPDDIngestor:
    """Downloads and ingests DPDD data into the vector store."""

    def __init__(
        self,
        vector_store_dir: Path,
        dpdd_config_path: Path,
    ) -> None:
        """
        Initialize the DPDD ingestor.

        Parameters
        ----------
        vector_store_dir : Path
            Directory where the vector store index will be stored.
        dpdd_config_path : Path
            Path to the DPDD ingestion YAML configuration file.
        """
        self._vector_store_dir = vector_store_dir
        self._embedder = Embedder()
        self._vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store = self._load_vector_store()
        # Load configuration for DPDD ingestion
        cfg = load_config(dpdd_config_path)

        self.banned_sections = {name.lower() for name in cfg["banned_sections"]["names"]}
        self.topics = cfg.get("topics", [])
        self.base_url = cfg["base_urls"][0]["base_url"]
        self.scrape_all = cfg.get("scrape_all", True)
        self.topics_number_limit = cfg.get("topics_number_limit", 0)

    def _load_vector_store(self) -> FAISS | None:
        """Load the FAISS vector store from the index directory."""
        if self._vector_store_dir.exists() and (self._vector_store_dir / "index.faiss").exists():
            try:
                return FAISS.load_local(
                    str(self._vector_store_dir),
                    self._embedder,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.warning("Failed to load vector store, rebuilding: %s", e)

        return None

    def ingest_new_data(self) -> None:
        """Ingest new data into the vector store.

        This method fetches DPDD entries, processes them, and adds them to the
        vector store, avoiding duplicates based on the 'source' metadata field.

        Raises
        ------
        RuntimeError
            If the vector store directory is missing or cannot be created.

        Returns
        -------
        None
            This function does not return anything; it performs the ingestion.
        """
        texts, metadatas = self._fetch_dpdd_entries()

        # Get list of already ingested paper filenames
        existing_sources: set[str] = set()
        if self._vector_store is not None:
            # Extract metadata 'source' from vector store's docs
            try:
                docstore = self._vector_store.docstore
                if hasattr(docstore, "_dict"):
                    # If docstore is a dict-like object
                    existing_sources = {
                        doc.metadata.get("source", "")
                        for doc in getattr(docstore, "_dict", {}).values()
                        if isinstance(doc.metadata.get("source", ""), str)
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

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            docs = [Document(page_content=text, metadata=metadata)]
            chunks = splitter.split_documents(docs)

            if self._vector_store is None:
                self._vector_store = FAISS.from_documents(chunks, self._embedder)
            else:
                self._vector_store.add_documents(chunks)

            self._vector_store.save_local(str(self._vector_store_dir))

    def _get_all_topics_for_baseurl(self) -> list[dict[str, str]]:
        """Get all topics for the base URL.

        Returns
        -------
        list[dict[str, str]]
            A list of dictionaries with topic names and links.
        """
        results: list[dict[str, str]] = []
        try:
            # Get the main page
            response = requests.get(self.base_url, timeout=15)
            try:
                response.raise_for_status()  # exception for bad status codes
            except requests.exceptions.HTTPError:
                if response.status_code == 404:
                    logger.warning("DPDD page not found (404 code) %s", self.base_url)
                    return []
                logger.exception("Failed to fetch DPDD page %s", self.base_url)
                return []

            # Parse the main page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find main content div
            main_div = soup.find("div", class_="body", role="main")
            if not isinstance(main_div, Tag):
                logger.warning("Main content div not found for %s", self.base_url)
                return results

            # Iterate over sections except indices-and-tables
            for section in main_div.find_all("section"):
                if not isinstance(section, Tag):
                    continue
                if section.get("id") == "indices-and-tables":
                    continue
                # Find links with class "reference internal"
                links = section.find_all("a", class_="reference internal")
                for link in links:
                    if not isinstance(link, Tag):
                        continue
                    name = link.text.strip()
                    href = link["href"]
                    # If href is a list, take the first element
                    if isinstance(href, list):
                        if href:  # not empty list
                            href = href[0]

                    # Ensure href is a string
                    if not isinstance(href, str):
                        continue
                    results.append({"name": name, "link": href})
        except requests.exceptions.RequestException:
            logger.exception("Failed to fetch DPDD page %s", self.base_url)
        return results

    def _fetch_dpdd_entries(self) -> tuple[list[str], list[dict[str, str]]]:
        """Fetch DPDD entries from the configured topics.

        Returns
        -------
        tuple[list[str], list[dict[str, str]]]
            A tuple containing a list of texts and a list of metadata dictionaries.
        """
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []

        # Choose topics to scrape
        list_of_urls = self._get_all_topics_for_baseurl() if self.scrape_all else self.topics

        # Apply topics_number_limit (0 = no limit)
        if self.topics_number_limit > 0:
            list_of_urls = list_of_urls[: self.topics_number_limit]

        # Scrape each topic
        for topic in list_of_urls:
            name = topic["name"]
            url = urljoin(self.base_url, str(topic["link"]))
            results = self._get_dpdd_sections(url, name)
            if results:
                for item in results:
                    texts.append(item["content"])
                    metadatas.append({k: v for k, v in item.items() if k != "content"})

        return texts, metadatas

    def _get_dpdd_sections(self, url: str, name: str) -> list[dict[str, str]]:
        """Get DPDD sections from a given URL.

        Parameters
        ----------
        url : str
            The URL of the DPDD topic page.
        name : str
            The name of the DPDD topic.

        Returns
        -------
        list[dict[str, str]]
            List of dictionaries containing the content and metadata of each section.
        """
        results: list[dict[str, str]] = []
        try:
            soup = self._fetch_and_parse(url)
            subtopic_links = self._extract_subtopic_links(soup)

            for link in subtopic_links:
                subtopic_url, subtopic_text = self._normalize_link(link, base_url=url)
                if subtopic_url is None:
                    continue

                try:
                    subtopic_soup = self._fetch_and_parse(subtopic_url)
                    self._process_subtopic_sections(
                        subtopic_soup,
                        subtopic_url,
                        subtopic_text,
                        name,
                        results,
                    )
                except requests.RequestException as e:
                    logger.warning("Error processing subtopic %s: %s", subtopic_url, e)
        except requests.RequestException as e:
            logger.warning("Error accessing main page %s: %s", url, e)
        return results

    def _fetch_and_parse(self, url: str) -> BeautifulSoup:
        """Fetch and parse a URL, returning a BeautifulSoup object.

        Parameters
        ----------
        url : str
            The URL to fetch and parse.

        Returns
        -------
        BeautifulSoup
            Parsed HTML content as a BeautifulSoup object.
        """
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _extract_subtopic_links(self, soup: BeautifulSoup) -> list[Tag]:
        """Extract subtopic links from the BeautifulSoup object.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object containing the parsed HTML.

        Returns
        -------
        list[Tag]
            List of subtopic links found in the soup.
        """
        return [el for el in soup.find_all("a", class_="reference internal", href=True) if isinstance(el, Tag)]

    def _normalize_link(self, link: Tag, base_url: str) -> tuple[str | None, str]:
        """Normalize a link to ensure it has a valid URL.

        Parameters
        ----------
        link : Tag
            The BeautifulSoup Tag containing the link.
        base_url : str
            The base URL to use for relative links.

        Returns
        -------
        tuple[str | None, str]
            A tuple containing the normalized URL and the link text.
            If the link is invalid, returns (None, "").
        """
        if not (isinstance(link, Tag) and "href" in link.attrs):
            return None, ""
        href = link["href"]
        text = link.text.strip()

        # Handle AttributeValueList
        if isinstance(href, AttributeValueList):
            if href:
                href = href[0]
            else:
                return None, text

        if isinstance(href, str) and not href.startswith(("http://", "https://")):
            href = urljoin(base_url, href)
        return href, text

    def _process_subtopic_sections(
        self,
        soup: BeautifulSoup,
        subtopic_url: str,
        subtopic_text: str,
        source_name: str,
        results: list[dict[str, str]],
    ) -> None:
        """Process sections of a subtopic and extract content.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object containing the parsed HTML of the subtopic.
        subtopic_url : str
            The URL of the subtopic page.
        subtopic_text : str
            The text of the subtopic.
        source_name : str
            The name of the source (e.g., "DPDD").
        results : list[dict[str, str]]
            List to append the extracted content and metadata.

        Returns
        -------
        None
            This function modifies the results list in place.
        """
        sections = soup.find_all("section")
        for section in sections:
            if not isinstance(section, Tag):
                continue

            section_name = section.find(["h1", "h2", "h3"])
            if not section_name:
                continue
            section_text = section_name.text.replace("¶", "").lower().strip()

            if section_text in self.banned_sections:
                logger.warning(
                    "Skipping banned section: %s in %s",
                    section_text,
                    subtopic_url,
                )
                continue

            section_name.extract()
            content = section.get_text(separator=" ", strip=True)

            if content.startswith("$") and " " not in content:
                continue  # skip placeholder sections

            if not content:
                continue

            section_id = None
            if isinstance(section, Tag):
                section_id = section.get("id")
            if section_id is None and isinstance(section_name, Tag):
                section_id = section_name.get("id")

            if isinstance(section_id, (list, tuple)):
                section_id_str = str(section_id[0]) if section_id else ""
            else:
                section_id_str = str(section_id) if section_id else ""

            section_url_with_anchor = f"{subtopic_url}#{section_id_str}" if section_id_str else subtopic_url

            metadata = {
                "source": section_url_with_anchor,
                "section": section_text,
                "subtopic": subtopic_text,
                "source_name": source_name,
                "category": "DPDD",
            }
            results.append({"content": content, **metadata})


def run_dpdd_ingestion(config: dict) -> None:
    """Run the DPDD ingestion process.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing paths and settings.

    Raises
    ------
    RuntimeError
        If the vector store directory is missing or cannot be created.

    Returns
    -------
    None
        This function does not return anything; it performs the ingestion.
    """
    index_dir = Path(config["vector_store"]["public_data_index_dir"])
    config_dir = Path(config["dpdd_data"]["config"])
    ingestor = EuclidDPDDIngestor(vector_store_dir=index_dir, dpdd_config_path=config_dir)
    ingestor.ingest_new_data()


def main() -> None:
    """Run the ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest publications from the Euclid BibTeX file.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="python/euclid/rag/app_config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_dpdd_ingestion(config)


if __name__ == "__main__":
    main()
