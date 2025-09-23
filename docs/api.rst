:og:description: Comprehensive API documentation for euclid.rag.

**********************
Python API Reference
**********************

.. currentmodule:: euclid.rag

Core Application
================

The main chatbot and user interface functionality.

Key Functions
-------------

The core application provides functions for:

* Configuring retrievers for document search
* Handling user input and responses
* Setting up the Streamlit interface

.. Note:
   The main application module (app.py) is excluded from documentation
   as it contains runtime initialization code for Streamlit.

.. automodule:: euclid.rag.chatbot
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.layout
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.streamlit_callback
   :members:
   :show-inheritance:

Retrieval System
================


Tools for retrieving and formatting information from document stores.


Overview
--------

The retrieval system provides specialized tools for querying Euclid mission documents. The system retrieves relevant documents from vector stores, ranks them using similarity and metadata scoring, and provides the top-ranked sources as context for response generation.

Source Attribution
~~~~~~~~~~~~~~~~~~

The chatbot responses include the top-ranked documents that were provided as context to the language model. These sources represent the retrieved, reranked, and deduplicated documents used to generate the response, not necessarily a selection made by the language model itself.

.. automodule:: euclid.rag.retrievers.generic_retrieval_tool
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: euclid.rag.retrievers.publication_tool
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.retrievers.redmine_tool
   :members:
   :show-inheritance:

Data Ingestion
==============

.. automodule:: euclid.rag.ingestion.ingest_dpdd
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.ingestion.ingest_publications
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.ingestion.ingest_redmine
   :members:
   :show-inheritance:

Utilities
=========

.. automodule:: euclid.rag.utils.acronym_handler
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.utils.config
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.utils.device
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.utils.redmine_cleaner
   :members:
   :show-inheritance:

Extra Tools
===========

.. automodule:: euclid.rag.extra_scripts.deduplication
   :members:
   :show-inheritance:

.. automodule:: euclid.rag.extra_scripts.vectorstore_embedder
   :members:
   :show-inheritance:


