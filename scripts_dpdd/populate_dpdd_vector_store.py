#!/usr/bin/env python
import csv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import yaml
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Hard-coded inputs
TOPICS_CSV = 'topics2get.csv'
TOPICS_NUMBER_LIMIT = 3  # 0 = no limit

from FAISS_vectorstore.vectorstore_embedder import (
    E5MpsEmbedder,
)


class EuclidDPDDIngestor:
    """Downloads and ingests DPDD data into the vectorstore.
    """

    def __init__(self, index_dir: Path) -> None:
        """Initialize the ingestor."""
        self._index_dir = index_dir
        self._embedder = E5MpsEmbedder()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = self._load_vectorstore()
        # Load banned sections config
        config_path = Path(__file__).parent / 'dpdd_ingestion_config' / 'dpdd_config.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.banned_sections = {name.lower() for name in cfg['banned_sections']['names']}
        # self.banned_full_links = set(cfg['banned_sections']['full_links'])

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

        for text, metadata in zip(texts, metadatas):
            if metadata.get("source", "") in existing_sources:
                print(f"Skipping already ingested source: {metadata.get('source')}")
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            # chunks = splitter.split_documents(text)
            # wrap text+metadata into a Document before splitting
            docs   = [Document(page_content=text, metadata=metadata)]
            chunks = splitter.split_documents(docs)

            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(
                    chunks, self._embedder
                )
            else:
                self._vectorstore.add_documents(chunks)

            new_ingested += 1

            # Save after each paper
            self._vectorstore.save_local(str(self._index_dir))

    def _fetch_dpdd_entries(self) -> list[dict]:
        """Fetch DPDD entries."""
        texts, metadatas = [], []
        with open(TOPICS_CSV, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for i, (name, url) in enumerate(reader):
                if TOPICS_NUMBER_LIMIT > 0 and i >= TOPICS_NUMBER_LIMIT:
                    break
                results = self._get_dpdd_sections(url, name)
                for item in results:
                    # print(item['content'])
                    # print({k: v for k, v in item.items() if k != 'content'})
                    # print('-' * 120)
                    texts.append(item['content'])
                    metadatas.append({k: v for k, v in item.items() if k != 'content'})
        return texts, metadatas

    def _get_dpdd_sections(self, url, name) -> list[dict]:
        """Get DPDD sections."""
        # get section names and and contents
        try:
            # Get the main page
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes        
            # Parse the main page
            soup = BeautifulSoup(response.text, 'html.parser')        
            # Find all links that have class "reference internal"
            subtopic_links = soup.find_all('a', class_='reference internal', href=True)

            results = []

            # Process each subtopic
            for link in subtopic_links:
                subtopic_url = link['href']
                subtopic_text = link.text.strip()
                
                # Handle relative URLs using urljoin
                if not subtopic_url.startswith(('http://', 'https://')):
                    subtopic_url = urljoin(url, subtopic_url)
                
                try:
                    # Get the subtopic page
                    subtopic_response = requests.get(subtopic_url)
                    subtopic_response.raise_for_status()
                    
                    # Parse the subtopic page
                    subtopic_soup = BeautifulSoup(subtopic_response.text, 'html.parser')
                    
                    # Find all sections in the subtopic page
                    sections = subtopic_soup.find_all(['section'])              
                    for section in sections:
                        # Get section name from the first heading in the section
                        section_name = section.find(['h1', 'h2', 'h3'])
                        if section_name:
                            section_text = section_name.text.replace('¶', '').lower().strip()

                            if section_text is None or section_text in self.banned_sections: 
                                print(f"Skipping banned section: {section_text} in {subtopic_url}")
                                continue

                            content = section.get_text(separator=" ", strip=True)
                            
                            if content:
                                # Compute direct link to the section using its id
                                section_id = section.get('id') or section_name.get('id')
                                if section_id:
                                    section_url_with_anchor = subtopic_url + '#' + section_id
                                else:
                                    section_url_with_anchor = subtopic_url
                                results.append({'content':content,'source':section_url_with_anchor,'section':section_text,'subtopic':subtopic_text,'source_name':name})                            
                        
                except requests.RequestException as e:
                    print(f"Error processing subtopic {subtopic_url}: {str(e)}")
                # print("____________________________")
                    
        except requests.RequestException as e:
            print(f"Error accessing main page {url}: {str(e)}")

        return results    


def run_dpdd_ingestion() -> None:
    """Run the DPDD ingestion process."""
    ingestor = EuclidDPDDIngestor(index_dir=Path("VectorStore_indexes").resolve())
    ingestor.ingest_new_data()


if __name__ == "__main__":
    run_dpdd_ingestion()