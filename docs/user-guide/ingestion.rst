##################
Document Ingestion
##################

Before using the chatbot, you must ingest documents into the vector store. This process downloads, processes, and indexes documents for semantic search.

Prerequisites
=============

* :doc:`installation` completed
* :doc:`configuration` file set up
* Internet connection for downloading documents
* Sufficient disk space for vector stores

.. warning::
   If the vector store is missing, the app will fail to start with:

   .. code-block:: text

      RuntimeError: Vector store missing. Please run ingestion before launching the app.

Publications Ingestion
======================


Ingest Euclid collaboration publications from the official BibTeX bibliography.

The EuclidBibIngestor is specifically designed to process publications from the Euclid collaboration's official bibliography file, which contains peer-reviewed papers, conference proceedings, and preprints related to the Euclid mission.

Basic Usage
-----------

.. code-block:: bash

   python python/euclid/rag/ingestion/ingest_publications.py --config path/to/app_config.yaml

Configuration Requirements
--------------------------

Ensure your app_config.yaml includes:

.. code-block:: yaml

   pdf_data:
     bibtex_url: "https://eceb.astro.uni-bonn.de/public/Euclid.bib"
     arxiv_pdf_base_url: "https://arxiv.org/pdf/"
     chunk_size: 800
     chunk_overlap: 100

Data Source
-----------

The ingestion process downloads and parses the official Euclid collaboration BibTeX file, which includes:

* **Key project papers** - Foundational Euclid mission publications
* **Data release papers** - Documentation for specific data releases (e.g., Q1 Special Issue)
* **Technical papers** - Instrument descriptions, data processing methods
* **Scientific results** - Analysis and findings from Euclid observations

Key Features
------------

**BibTeX Processing**
  * Parses official Euclid collaboration bibliography
  * Extracts publication metadata (authors, titles, abstracts)
  * Downloads full-text PDFs from arXiv when available

**Content Extraction**
  * Processes PDF content for full-text search
  * Maintains publication metadata for proper attribution
  * Handles various publication formats (journal articles, preprints, conference papers)

**Semantic Deduplication**
  * Prevents ingestion of semantically similar content across publications
  * Particularly effective for reducing redundant introductory material
  * Uses similarity thresholds to identify and filter duplicate content sections

**Vector Store Integration**
  * Chunks document content for semantic search
  * Configurable chunk sizes and overlap for optimal retrieval
  * Prevents duplicate ingestion of existing publications

Custom Configuration
--------------------

To use a different configuration file:

.. code-block:: bash

   python python/euclid/rag/ingestion/ingest_publications.py -c /path/to/custom_config.yaml

   # Or using the long form
   python python/euclid/rag/ingestion/ingest_publications.py --config /path/to/custom_config.yaml

JSON Document Ingestion
=======================

Ingest documents from JSON files. While commonly used for Redmine wiki exports, this ingestion method works with any JSON document structure.

Basic Usage
-----------

.. code-block:: bash

   python python/euclid/rag/ingestion/ingest_json.py -c /path/to/custom_config.yaml

   # Or using the long form
   python python/euclid/rag/ingestion/ingest_json.py --config /path/to/custom_config.yaml


This processes JSON files from the directory specified in json_data.redmine_json_dir configuration.

Configuration Requirements
--------------------------

Ensure your `app_config.yaml` includes:

.. code-block:: yaml

   json_data:
     redmine_json_dir: "/path/to/json/files"
     chunk_size: 800
     json_root_key: pages

JSON File Format
----------------

The expected JSON structure (example using Redmine export format):

.. code-block:: json

   {
     "pages": [
       {
         "project_id": "project-name",
         "page_name": "Wiki Page Title",
         "content": "Page content text...",
         "updated_on": "2024-01-15T10:30:00Z",
         "url": "https://redmine.example.com/page",
         "metadata": {
           "author": "username",
           "version": 5
         }
       }
     ]
   }

Key Features
------------

**Deduplication**
  Uses hardcoded key fields to prevent ingesting the same content multiple times

**Content Processing**
  * Extracts text from JSON structure
  * Preserves metadata for source attribution
  * Handles nested JSON structures

**Chunking Strategy**
  * Respects document boundaries
  * Maintains context between related sections
  * Configurable chunk sizes for different content types

Custom JSON Structures
----------------------

For different JSON document formats, modify the configuration:

.. code-block:: yaml

   json_data:
     json_root_key: "documents" # Change root key for your JSON structure
     redmine_json_dir: "/path/to/your/json/files" # Any JSON documents


DPDD Ingestion
==============

The DPDD (Data Product Description Document) ingestion downloads and processes Euclid DPDD pages from the official website.

Basic Usage
-----------

.. code-block:: bash

   python python/euclid/rag/ingestion/ingest_dpdd.py --config path/to/app_config.yaml

Features
--------

The DPDD ingestion process:

* **Downloads DPDD pages** from the Euclid website
* **Extracts subtopics and sections**, skipping banned sections
* **Splits text into chunks** using RecursiveCharacterTextSplitter
* **Stores chunks** in a FAISS vector store for semantic search
* **Prevents duplicate ingestion** by checking existing sources

Configuration Options
---------------------

The DPDD ingestion behavior is controlled by the ``dpdd_ingest_config.yaml`` file:

**Selective Ingestion**
   Specify particular topics to ingest:

   .. code-block:: yaml

      scrape_all: false
      topics:
        - name: Purpose and Scope
          link: purpose.html
        - name: LE1 Data Products
          link: le1dpd/le1index.html

**Complete Ingestion**
   Ingest all available content:

   .. code-block:: yaml

      scrape_all: true
      topics_number_limit: 0  # No limit

**Limited Ingestion**
   Limit the number of topics:

   .. code-block:: yaml

      scrape_all: true
      topics_number_limit: 5  # Only first 5 topics

Ingestion Process Details
=========================

Text Processing Pipeline
------------------------

1. **Document Download**: Fetch content from configured URLs
2. **Content Extraction**: Parse HTML and extract relevant text
3. **Section Filtering**: Skip banned sections (headers, navigation, etc.)
4. **Text Chunking**: Split long documents into manageable chunks
5. **Embedding Generation**: Create vector embeddings for each chunk
6. **Vector Storage**: Store embeddings in FAISS index
7. **Metadata Storage**: Save document metadata for source attribution

Chunk Size and Overlap
-----------------------

The system uses ``RecursiveCharacterTextSplitter`` with optimized settings:

* **Chunk size**: Balanced for context and performance
* **Chunk overlap**: Ensures continuity between chunks
* **Separator handling**: Respects document structure (paragraphs, sentences)

Duplicate Prevention
--------------------

The ingestion process automatically:

* **Checks existing sources** in the vector store
* **Skips already processed documents** to avoid duplicates
* **Updates metadata** for modified documents
* **Maintains consistency** across ingestion runs

Monitoring Ingestion Progress
=============================

Command Line Output
-------------------

The ingestion scripts provide progress information:

.. code-block:: text

   Processing topic: Purpose and Scope
   Extracting sections from: purpose.html
   Skipping banned section: Header
   Creating 15 text chunks
   Storing embeddings in vector store
   ✓ Completed: Purpose and Scope (15 chunks)

Logging
-------

Detailed logs are available for troubleshooting:

.. code-block:: bash

   # Enable verbose logging
   python python/euclid/rag/ingestion/ingest_dpdd.py --config config.yaml --verbose

Storage Requirements
====================

Estimate disk space needed for your vector stores:

**Small Dataset** (< 100 documents)
   * Vector store: ~50-100 MB
   * Metadata: ~10-20 MB

**Medium Dataset** (100-1000 documents)
   * Vector store: ~500 MB - 1 GB
   * Metadata: ~50-100 MB

**Large Dataset** (> 1000 documents)
   * Vector store: > 1 GB
   * Metadata: > 100 MB

.. note::
   Actual sizes depend on document length, embedding dimensions, and chunk sizes.

Batch Processing
================

For large document collections, consider batch processing:

.. code-block:: bash

   # Process publications first
   python python/euclid/rag/ingestion/ingest_publications.py -c config.yaml

   # Then process DPDD documents
   python python/euclid/rag/ingestion/ingest_dpdd.py --config config.yaml

   # Verify vector stores were created
   ls -la *_vector_store/

Ingestion Validation
====================

After ingestion, verify the vector stores:

.. code-block:: python

   # Check vector store contents
   import os
   from euclid.rag import chatbot

   # Load configuration
   config_path = "python/euclid/rag/app_config.yaml"

   # Verify vector store files exist
   vector_store_dirs = ["redmine_vector_store", "public_data_vector_store"]

   for dir_name in vector_store_dirs:
       if os.path.exists(dir_name):
           files = os.listdir(dir_name)
           print(f"{dir_name}: {files}")
       else:
           print(f"⚠️  Missing: {dir_name}")

Performance Optimization
========================

For faster ingestion:

**Parallel Processing**
   The ingestion scripts support concurrent processing where possible.

**Network Optimization**
   Use a stable, fast internet connection for downloading documents.

**Storage Optimization**
   Use SSD storage for better I/O performance during indexing.

**Memory Management**
   Ensure sufficient RAM for large document processing.

Next Steps
==========

After successful ingestion:

* :doc:`usage` - Run the chatbot interface
* :doc:`troubleshooting` - Resolve any ingestion issues
* Verify that documents are searchable through the interface

Common ingestion issues and solutions are covered in :doc:`troubleshooting`.