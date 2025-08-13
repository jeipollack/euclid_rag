# populate_dpdd_vector_store.py

## Overview
A Python script to scrape Euclid DPDD (Data Product Definition Documents) and ingest their content into a FAISS vector store for retrieval-augmented generation (RAG) workflows.

## Key Components
- **EuclidDPDDIngestor**: Orchestrates scraping, processing, embedding, and storage.
- **E5MpsEmbedder**: Embedder implementation for generating vector representations.
- **FAISS Vector Store**: Local storage of document embeddings via `langchain_community.vectorstores.FAISS`.

## Workflow
1. **Initialization**
   - Create or load index directory (`rag/FAISS_vectorstore`).
   - Load existing FAISS index if present.
   - Read `dpdd_config.yaml` to obtain:
     - `banned_sections`: section names to skip.
     - `topics`: optional list of specific topic links.
   - Set `base_url` for live DPDD site.

2. **Data Ingestion**
   - `ingest_new_data()`:
     - Fetch texts and metadata with `_fetch_dpdd_entries()`.
     - Identify already ingested sources to avoid duplicates.
     - For each new entry:
       - Split content into chunks (`RecursiveCharacterTextSplitter`, chunk_size=800, overlap=100).
       - Generate embeddings via `E5MpsEmbedder`.
       - Add chunks to FAISS index (create or update).
       - Save index locally.

3. **Scraping Logic**
   - `_get_all_topics_for_baseurl()`: Fetches and aggregates DPDD topic entries from the base URL.
       - Sends `GET` request to `base_url` with a timeout (15s).
       - Locates the `<div class="body" role="main">`; warns and returns `[]` if not found.
       - Iterates over `<section>` elements, skipping those with `id="indices-and-tables"`.
       - Within each section, finds `<a class="reference internal" href>` links.
       - Appends dictionaries `{'name': name, 'link': href}` to the results list.
       - Returns a list of topic dictionaries.
   - `_fetch_dpdd_entries()`: Orchestrates topic scraping and aggregates content and metadata.
      - Determines `list_of_urls` based on `scrape_all` flag (all topics) or config-specified `topics`.
      - Applies `TOPICS_NUMBER_LIMIT` if non-zero to limit processed topics.
      - Constructs full topic URLs via `urljoin` with `base_url`.
      - For each topic entry, invokes `_get_dpdd_sections(url, name)`.
      - Collects each section’s `content` and metadata into parallel `texts` and `metadatas` lists.
      - Returns `(texts, metadatas)`.
   - `_get_dpdd_sections(url, name)`: For each topic URL:
     - Parse subtopic links.
     - Fetch and parse each subtopic page.
     - Extract sections, filter banned or placeholder content.
     - Construct metadata (source URL, section name, subtopic, source_name, category).

## Configuration
- **TOPICS_NUMBER_LIMIT**: Debug limit on number of topics (0 = ingest all).
- **dpdd_config.yaml**: Configuration file defining scraping and filtering parameters:
    - **base_urls** (list, optional): configurations for DPDD endpoints (currently not used in code):
        - `type` (str): identifier (e.g., `msp`).
        - `url` (str): DPDD site URL.
        - `version` (str): version label (e.g., `live`).
        - `scrape_all` (bool): if true, override `topics` and scrape entire site.
    - **topics** (list): explicit topic entries to ingest in absence of full-site scrape:
        - Each item:
            - `name` (str): topic title.
            - `link` (str): relative or absolute link to topic index.
    - **banned_sections** (map): sections to skip during scraping:
        - `names` (list of str): section headings to exclude.
        - `full_links` (list of str): specific URLs to exclude entirely.

## Usage
```bash
python populate_dpdd_vector_store.py
```
This populates or updates the FAISS index at `rag/FAISS_vectorstore`.

## Dependencies
- `requests`, `beautifulsoup4`, `PyYAML`
- `langchain`, `langchain_community`
- Local module: `euclid.rag.FAISS_vectorstore.vectorstore_embedder`
