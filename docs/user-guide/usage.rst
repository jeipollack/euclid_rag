###################
Running the Chatbot
###################

Once you have completed :doc:`installation`, :doc:`configuration`, and :doc:`ingestion`, you can run the euclid_rag chatbot interface.

Local Deployment
================

Standard Streamlit Launch
-------------------------

For local development and testing:

.. code-block:: bash

   cd python/euclid
   streamlit run rag/app.py

This starts the Streamlit web interface, typically accessible at ``http://localhost:8501``.

Custom Port and Host
--------------------

To run on a different port or host:

.. code-block:: bash

   cd python/euclid
   streamlit run rag/app.py --port 8080 --server.address 0.0.0.0

Production Deployment Options
-----------------------------

For production environments:

.. code-block:: bash

   # With specific configuration
   streamlit run rag/app.py --server.port 80 --server.address 0.0.0.0

   # With custom config file
   EUCLID_RAG_CONFIG_PATH=/path/to/prod_config.yaml streamlit run rag/app.py

Docker Deployment
=================

Container-based deployment with Docker Compose provides isolation and easier management.

Starting Services
-----------------

.. code-block:: bash

   # Start all services
   docker compose up --build

This launches:

* **Streamlit application** - The main web interface
* **Ollama LLM server** - Local language model service
* **Supporting services** - Database, networking, etc.

Setting up the LLM Model
------------------------

After containers are running, pull the desired model:

.. code-block:: bash

   # Pull the default model (Mistral)
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:latest

   # Or pull alternative models
   docker exec -it euclid_rag-ollama-1 ollama pull llama2:latest
   docker exec -it euclid_rag-ollama-1 ollama pull codellama:latest

Available Models
----------------

Check which models are available:

.. code-block:: bash

   # List available models to pull
   docker exec -it euclid_rag-ollama-1 ollama list

   # Check model details
   docker exec -it euclid_rag-ollama-1 ollama show mistral:latest

Docker Benefits
---------------

The Docker setup provides:

* **Isolated environment** with all dependencies
* **Ollama LLM server** running in a separate container
* **Streamlit application** accessible via web browser
* **Automatic service orchestration** with Docker Compose
* **Easy scaling** and deployment management

Using the Interface
===================

Web Interface Features
----------------------

The Streamlit interface provides:

**Chat Interface**
   * Natural language input for questions
   * Real-time responses from the LLM
   * Conversation history and context

**Document Querying**
   * Search across multiple document types
   * Publications, DPDD, and other ingested content
   * Semantic search with vector similarity

**Source Attribution**
   * View source documents for each answer
   * Direct links to original content where available
   * Confidence scores and relevance rankings

**Interactive Elements**
   * File upload for additional documents
   * Configuration adjustments
   * Export conversation history

Query Examples
--------------

Try these example queries to test your system:

**General Questions**
   * "What is the purpose of the Euclid mission?"
   * "How are VIS data products structured?"
   * "Explain the LE1 data processing pipeline"

**Technical Queries**
   * "What file formats are used for SIM data products?"
   * "How is astrometric calibration performed?"
   * "What are the quality requirements for photometry?"

**Document-Specific**
   * "Find information about header keywords"
   * "Show me examples of DPDD data structures"
   * "What are the validation procedures?"

Configuration During Runtime
============================

LLM Model Switching
--------------------

To change models without rebuilding containers:

.. code-block:: bash

   # Pull new model
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:7b-instruct

   # Update app_config.yaml
   # Change: model: "granite3.2:latest"
   # To:     model: "mistral:7b-instruct"

   # Restart the application
   docker compose restart euclid_rag

Temperature and Behavior Tuning
-------------------------------

Adjust response characteristics in ``app_config.yaml``:

.. code-block:: yaml

   llm:
     model: "granite3.2:latest"
     temperature: 0.2  # Adjust between 0.0-1.0
     base_url: "http://ollama:11434"

**Temperature Effects:**
   * ``0.0`` - Always chooses most likely response (deterministic)
   * ``0.1`` - Very consistent, minimal creativity
   * ``0.3`` - Balanced factual accuracy with slight variation
   * ``0.7`` - More creative, conversational responses
   * ``1.0`` - Highly creative, unpredictable responses

Embedding Model Configuration
-----------------------------

Optimize embedding performance:

.. code-block:: yaml

   embeddings:
     class: "E5MpsEmbedder"  # Optimized for Apple Silicon
     model_name: "intfloat/e5-large-v2"  # Higher accuracy
     batch_size: 32  # Adjust based on available memory

**Embedding Model Options:**
   * ``"intfloat/e5-small-v2"`` - Fast, lower memory usage
   * ``"intfloat/e5-base-v2"`` - Balanced performance
   * ``"intfloat/e5-large-v2"`` - Best accuracy, higher memory usage

Environment Variables
---------------------

Override settings without modifying config files:

.. code-block:: bash

   # Set custom model
   export EUCLID_RAG_LLM_MODEL="mistral:latest"
   export EUCLID_RAG_TEMPERATURE="0.3"

   # Set custom vector store path
   export EUCLID_RAG_VECTOR_STORE_PATH="/custom/path"

   # Enable debug mode
   export EUCLID_RAG_DEBUG=true

   # Then run the application
   streamlit run rag/app.py

Streamlit Configuration
-----------------------

Create a ``.streamlit/config.toml`` file for UI customization:

.. code-block:: toml

   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#262730"

   [server]
   port = 8501
   address = "localhost"

Performance Optimization
========================

Memory Management
-----------------

For large document collections:

* **Monitor memory usage** during queries
* **Restart services periodically** to clear cache
* **Adjust chunk sizes** in configuration if needed

Response Time Optimization
--------------------------

To improve query response times:

* **Use SSD storage** for vector stores
* **Increase available RAM** for embedding operations
* **Choose appropriate LLM models** (smaller models = faster responses)
* **Optimize vector store configuration** for your use case

Monitoring and Logging
======================

Application Logs
----------------

View application logs:

.. code-block:: bash

   # For local deployment
   tail -f logs/euclid_rag.log

   # For Docker deployment
   docker compose logs -f euclid_rag
   docker compose logs -f ollama

Performance Metrics
-------------------

Monitor key metrics:

* **Query response time**
* **Memory usage**
* **Vector store size**
* **Document retrieval accuracy**

Health Checks
-------------

Verify services are running correctly:

.. code-block:: bash

   # Check Streamlit is accessible
   curl http://localhost:8501/healthz

   # Check Ollama service (Docker)
   docker exec -it euclid_rag-ollama-1 ollama list

Troubleshooting Runtime Issues
==============================

Common issues during runtime:

**Slow Responses**
   * Check available memory and CPU
   * Verify vector store isn't corrupted
   * Consider using a smaller LLM model

**Connection Errors**
   * Ensure all services are running
   * Check firewall and port configurations
   * Verify Docker containers are healthy

**Inaccurate Results**
   * Review ingested document quality
   * Adjust similarity thresholds
   * Re-run ingestion if needed

For detailed troubleshooting, see :doc:`troubleshooting`.

Advanced Usage
==============

Custom Integration
------------------

Integrate euclid_rag into your own applications:

.. code-block:: python

   from euclid.rag import chatbot

   # Configure retriever
   retriever = chatbot.configure_retriever()

   # Create router with custom callback
   router = chatbot.create_euclid_router(callback_handler=my_callback)

   # Process queries programmatically
   response = router.invoke({"query": "What is Euclid?"})

API Usage
---------

Use the underlying functions directly:

.. code-block:: python

   from euclid.rag.retrievers.generic_retrieval_tool import get_generic_retrieval_tool

   # Get retrieval tool
   tool = get_generic_retrieval_tool(llm, retriever)

   # Use for custom applications
   result = tool.invoke("your question here")

Next Steps
==========

* Explore the :doc:`../api` for programmatic usage
* See :doc:`troubleshooting` for common issues
* Check :doc:`../developer-guide/index` for customization options