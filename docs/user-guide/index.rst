:og:description: Learn how to use euclid_rag.

##########
User Guide
##########

Learn how to use **euclid_rag** for querying Euclid space mission documents.

Overview
========

Euclid RAG is an open-source Retrieval-Augmented Generation (RAG) system designed to provide **efficient document retrieval and knowledge augmentation** for the Euclid scientific community. The project integrates local Large Language Models (LLMs) with a vector database to **retrieve, process, and generate** relevant scientific information.

Key Features
------------

* **Local deployment** without API-based LLM dependencies
* **Document retrieval strategies** tailored to Euclid's scientific workflows
* **Streamlit-based user interface** for easy interaction
* **FAISS vector store** for efficient semantic search
* **Multiple document types** including publications and DPDD documents
* **Docker support** for containerized deployment
* **Potential agentic capabilities** for automated knowledge retrieval

Quick Start
===========

Get up and running with euclid_rag in four steps:

1. :doc:`Install <installation>` euclid_rag using your preferred method
2. :doc:`Configure <configuration>` your system and vector store settings
3. :doc:`Ingest <ingestion>` documents into the vector store
4. :doc:`Run <usage>` the chatbot interface

Project Origins
===============

This project was initially forked from the `Rubin Observatory's Rubin RAG system <https://github.com/lsst-dm/rubin_rag>`_. While developed in consultation with Rubin developers, **Euclid RAG is evolving in a different direction** to meet the specific needs of the Euclid collaboration.

**Key Differences:**

* Focus on local deployment without API dependencies
* Different document retrieval strategies for Euclid workflows
* Potential agentic capabilities for automated processing

Next Steps
==========

* Explore the :doc:`../api` for detailed function documentation
* Check the :doc:`../developer-guide/index` for contributing guidelines
* Review the :doc:`../changelog` for recent updates

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   configuration
   ingestion
   usage
   troubleshooting