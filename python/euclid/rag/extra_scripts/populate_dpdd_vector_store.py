#!/usr/bin/env python
# ruff: noqa: ERA001, C901, PLR0912, E501

# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""Ingest DPDD Euclid information into a FAISS vectorstore."""

import argparse
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from bs4.element import AttributeValueList, Tag
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from euclid.rag.extra_scripts.vectorstore_embedder import Embedder
from euclid.rag.utils.config import load_config

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
        self._embedder = Embedder()
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
        self.base_url = cfg["base_urls"][0]["base_url"]
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

        existing_sources: set[str] = set()
        if self._vectorstore is not None:
            # Extract metadata 'source' from vectorstore's docs
            try:
                docstore = self._vectorstore.docstore
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

    def _get_all_topics_for_baseurl(self) -> list[dict[str, str]]:
        """Get all topics for baseurl."""
        results: list[dict[str, str]] = []
        try:
            # Get the main page
            response = requests.get(self.base_url, timeout=15)
            try:
                response.raise_for_status()  # exception for bad status codes
            except requests.exceptions.HTTPError:
                if response.status_code == 404:
                    logger.warning(
                        "DPDD page not found (404 code) %s", self.base_url
                    )
                    return []
                logger.exception("Failed to fetch DPDD page %s", self.base_url)
                return []
            # Parse the main page
            soup = BeautifulSoup(response.text, "html.parser")
            # Find main content div
            main_div = soup.find("div", class_="body", role="main")
            if not isinstance(main_div, Tag):
                logger.warning(
                    "Main content div not found for %s", self.base_url
                )
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
        """Fetch DPDD entries."""
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
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
            url = urljoin(self.base_url, str(topic["link"]))
            results = self._get_dpdd_sections(url, name)
            if results:
                for item in results:
                    texts.append(item["content"])
                    metadatas.append(
                        {k: v for k, v in item.items() if k != "content"}
                    )
        return texts, metadatas

    def _get_dpdd_sections(self, url: str, name: str) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        try:
            soup = self._fetch_and_parse(url)
            subtopic_links = self._extract_subtopic_links(soup)

            for link in subtopic_links:
                subtopic_url, subtopic_text = self._normalize_link(
                    link, base_url=url
                )
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
                    logger.warning(
                        "Error processing subtopic %s: %s", subtopic_url, e
                    )
        except requests.RequestException as e:
            logger.warning("Error accessing main page %s: %s", url, e)
        return results

    def _fetch_and_parse(self, url: str) -> BeautifulSoup:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _extract_subtopic_links(self, soup: BeautifulSoup) -> list[Tag]:
        return [
            el
            for el in soup.find_all(
                "a", class_="reference internal", href=True
            )
            if isinstance(el, Tag)
        ]

    def _normalize_link(
        self, link: Tag, base_url: str
    ) -> tuple[str | None, str]:
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

        if isinstance(href, str) and not href.startswith(
            ("http://", "https://")
        ):
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

            if section_id_str:
                section_url_with_anchor = f"{subtopic_url}#{section_id_str}"
            else:
                section_url_with_anchor = subtopic_url

            metadata = {
                "source": section_url_with_anchor,
                "section": section_text,
                "subtopic": subtopic_text,
                "source_name": source_name,
                "category": "DPDD",
            }
            results.append({"content": content, **metadata})


def run_dpdd_ingestion(config: dict) -> None:
    """Run the DPDD ingestion process."""
    index_dir = Path(config["vector_store"]["index_dir"])
    ingestor = EuclidDPDDIngestor(index_dir=index_dir)
    ingestor.ingest_new_data()


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
    run_dpdd_ingestion(config)


if __name__ == "__main__":
    main()
