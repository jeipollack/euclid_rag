#############
Configuration
#############

Configure euclid_rag for your environment and document sources.

Overview
========

The system uses a YAML configuration file to define vector store settings, document sources, and ingestion parameters. The main configuration file is typically located at ``python/euclid/rag/app_config.yaml``.

Main Configuration File
=======================

Basic Structure
---------------

.. code-block:: yaml

   vector_store:
     type: "faiss"
     redmine_index_dir: "redmine_vector_store"
     public_data_index_dir: "public_data_vector_store"

   data:
     dpdd:
       config: "path/to/dpdd_ingest_config.yaml"

Vector Store Settings
---------------------

**type**
   Currently only "faiss" is supported for the vector store backend.

**{prefix}_index_dir**
   Path where FAISS index files (``index.faiss``, ``index.pkl``) will be stored.
   This can be:

   * **Relative path**: Within the repository (e.g., ``"vector_store"``)
   * **Absolute path**: Full system path (e.g., ``"/data/vector_stores/euclid"``)

Example Configurations
======================

Development Configuration
-------------------------

For local development:

.. code-block:: yaml

   llm:
     model: "granite3.2:latest"
     temperature: 0.1
     base_url: "http://localhost:11434"

   vector_store:
     type: "faiss"
     redmine_index_dir: "./data/redmine_vector_store"
     public_data_index_dir: "./data/public_data_vector_store"

   json_data:
     redmine_json_dir: "./data/redmine_exports/"
     chunk_size: 500  # Smaller for development


Production Configuration
------------------------

For production deployment:

.. code-block:: yaml

   llm:
     model: "granite3.2:latest"
     temperature: 0
     base_url: "http://ollama:11434"

   embeddings:
     class: "E5MpsEmbedder"
     model_name: "intfloat/e5-large-v2"  # More accurate
     batch_size: 32  # Larger batches

   vector_store:
     type: "faiss"
     redmine_index_dir: "/data/euclid_rag/redmine_vector_store"
     public_data_index_dir: "/data/euclid_rag/public_data_vector_store"

   json_data:
     redmine_json_dir: "/data/redmine_exports/"
     chunk_size: 800


High Performance Configuration
------------------------------

For faster responses:

.. code-block:: yaml

   llm:
     model: "mistral:7b"  # Smaller, faster model
     temperature: 0
     base_url: "http://ollama:11434"

   embeddings:
     class: "E5MpsEmbedder"
     model_name: "intfloat/e5-small-v2"  # Faster embedding
     batch_size: 64


DPDD Ingestion Configuration
============================

The DPDD (Data Product Description Document) ingestion requires a separate configuration file, typically ``dpdd_ingest_config.yaml``:

.. code-block:: yaml

   # Base URLs for DPDD content
   base_urls:
     - type: msp
       base_url: https://euclid.esac.esa.int/dr/q1/dpdd/
       version: dm10

   # Topics to ingest
   topics:
     - name: Purpose and Scope
       link: purpose.html
     - name: LE1 Data Products
       link: le1dpd/le1index.html
     - name: SIM Data Products
       link: simdpd/simindex.html
     - name: VIS Data Products
       link: visdpd/visindex.html

   # Ingestion limits and options
   topics_number_limit: 0  # 0 = no limit (ingest all topics)
   scrape_all: true        # If true, ignores topics list and scrapes all sections

   # Sections to skip during ingestion
   banned_sections:
     names:
       - Header
       - Data Header
     full_links: []

Configuration Parameters
========================

DPDD Parameters
---------------

**base_urls**
   List of base URLs to scrape DPDD content from.

**topics**
   Specific topics to ingest. Each topic has a ``name`` and ``link``.

**topics_number_limit**
   Maximum number of topics to ingest. Set to ``0`` for no limit.

**scrape_all**
   If ``true``, ignores the topics list and scrapes all available sections.

**banned_sections**
   Sections to skip during ingestion:

   * **names**: Section names to skip (e.g., "Header", "Data Header")
   * **full_links**: Complete URLs to skip

.. note::
   Banned sections help prevent ingesting content that might confuse the LLM, such as repetitive headers or navigation elements.

Custom Configuration Paths
===========================

Using Custom Config Files
--------------------------

You can specify custom configuration files when running ingestion:

.. code-block:: bash

   # For publications
   python python/euclid/rag/ingestion/ingest_publications.py -c /path/to/custom_config.yaml

   # For Redmine (or other JSON sources)
   python python/euclid/rag/ingestion/ingest_redmine.py -c /path/to/custom_config.yaml

   # For DPDD
   python python/euclid/rag/ingestion/ingest_dpdd.py --config /path/to/custom_config.yaml

Note: You can ingest multiple sources into the same vector store by using the same ``redmine_index_dir`` or ``public_data_index_dir`` in the main configuration file.

Environment Variables
=====================

Some configurations can be overridden with environment variables:

.. code-block:: bash

   # Set custom vector store path
   export EUCLID_RAG_VECTOR_STORE_PATH="/custom/path/to/vector_store"

   # Set custom config file
   export EUCLID_RAG_CONFIG_PATH="/path/to/config.yaml"

Troubleshooting Configuration
=============================

Common Issues
-------------

**Permission Errors**
   Ensure the specified directories are writable by the user running the application.

**Path Not Found**
   Verify that relative paths are correct relative to your working directory.

**YAML Syntax Errors**
   Use a YAML validator to check your configuration files for syntax issues.

Validation
----------

Test your configuration by running:

.. code-block:: bash

   python -c "
   import yaml
   with open('python/euclid/rag/app_config.yaml', 'r') as f:
       config = yaml.safe_load(f)
       print('Configuration loaded successfully!')
       print(f'Vector store type: {config[\"vector_store\"][\"type\"]}')
   "

Next Steps
==========

After configuring your system:

* :doc:`ingestion` - Ingest documents into your configured vector stores
* :doc:`usage` - Run the chatbot with your configuration
* :doc:`troubleshooting` - Resolve common configuration issues