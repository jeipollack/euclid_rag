############
Installation
############

Choose the installation method that best fits your needs and environment.

PyPI Installation
=================

**Recommended for:** Basic usage and getting started quickly.

Install euclid_rag from PyPI:

.. code-block:: bash

   pip install euclid_rag

This installs the package and its dependencies, allowing you to use euclid_rag in your Python environment.

Development Installation
========================

**Recommended for:** Contributing to the project or customizing the code.

Clone the repository and set up a development environment:

.. code-block:: bash

   git clone https://github.com/jeipollack/euclid_rag
   cd euclid_rag
   python3 -m venv .venv
   source .venv/bin/activate
   make init

This setup includes:

* **Virtual environment** for isolated dependencies
* **Development dependencies** for testing and building
* **Editable installation** so changes are immediately available

Docker Installation
===================

**Recommended for:** Production deployment and containerized environments.

For containerized deployment with Docker Compose:

.. code-block:: bash

   git clone https://github.com/jeipollack/euclid_rag
   cd euclid_rag
   docker compose up --build

Docker Features
---------------

This setup includes:

* **Parallelized build stages** for faster container building
* **Optimized image size** for efficient deployment
* **Separate Ollama container** managed by Docker Compose
* **Dynamic package versioning** for correct version tracking
* **Isolated services** with automatic orchestration

Setting up the LLM Model
-------------------------

After the containers are running, you need to pull the desired model:

.. code-block:: bash

   docker exec -it euclid_rag-ollama-1 ollama pull mistral:latest

.. note::
   The model must be explicitly requested with the ``docker exec`` command after the containers are started.

Verification
============

Test your installation by importing the package:

.. code-block:: python

   import euclid.rag.chatbot

   print("euclid_rag installed successfully!")

Next Steps
==========

After installation, proceed to:

* :doc:`configuration` - Set up your system configuration
* :doc:`ingestion` - Ingest documents into the vector store
* :doc:`usage` - Run the chatbot interface