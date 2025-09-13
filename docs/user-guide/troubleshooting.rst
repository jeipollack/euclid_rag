###############
Troubleshooting
###############

Common issues and solutions for euclid_rag installation, configuration, and usage.

Installation Issues
===================

Package Installation Failures
------------------------------

**Problem**: ``pip install euclid_rag`` fails with dependency errors.

**Solutions**:

.. code-block:: bash

   # Update pip and try again
   pip install --upgrade pip
   pip install euclid_rag

   # Install with specific dependency versions
   pip install euclid_rag --no-deps
   pip install -r requirements.txt

   # Use conda for complex dependencies
   conda install -c conda-forge faiss-cpu
   pip install euclid_rag

**Problem**: Import errors after installation.

**Solutions**:

.. code-block:: python

   # Check if package is properly installed
   import sys

   print(sys.path)

   # Verify installation location
   import euclid

   print(euclid.__file__)

.. code-block:: bash

   # Reinstall in development mode
   pip install -e .

Docker Issues
-------------

**Problem**: ``docker compose up`` fails to start services.

**Solutions**:

.. code-block:: bash

   # Check Docker daemon is running
   docker --version
   docker-compose --version

   # Rebuild containers completely
   docker compose down --volumes
   docker compose build --no-cache
   docker compose up

   # Check container logs
   docker compose logs euclid_rag
   docker compose logs ollama

**Problem**: Cannot pull Ollama models.

**Solutions**:

.. code-block:: bash

   # Check if Ollama container is running
   docker ps | grep ollama

   # Try different model names
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:7b
   docker exec -it euclid_rag-ollama-1 ollama pull llama2:7b-chat

   # Check available space
   docker system df

Configuration Issues
====================

Vector Store Missing Error
---------------------------

**Problem**:
.. code-block:: text

   RuntimeError: Vector store missing. Please run ingestion before launching the app.

**Solutions**:

1. **Run ingestion first**:

   .. code-block:: bash

      python python/euclid/rag/ingestion/ingest_publications.py
      python python/euclid/rag/ingestion/ingest_dpdd.py --config python/euclid/rag/app_config.yaml

2. **Check vector store paths**:

   .. code-block:: python

      import os
      import yaml

      with open("python/euclid/rag/app_config.yaml", "r") as f:
          config = yaml.safe_load(f)

      for key, path in config["vector_store"].items():
          if "index_dir" in key:
              print(f"{key}: {path} - Exists: {os.path.exists(path)}")

3. **Create missing directories**:

   .. code-block:: bash

      mkdir -p redmine_vector_store public_data_vector_store

YAML Configuration Errors
--------------------------

**Problem**: ``yaml.parser.ParserError`` or similar YAML parsing errors.

**Solutions**:

.. code-block:: bash

   # Validate YAML syntax
   python -c "
   import yaml
   with open('python/euclid/rag/app_config.yaml', 'r') as f:
       try:
           config = yaml.safe_load(f)
           print('✓ YAML is valid')
       except yaml.YAMLError as e:
           print(f'✗ YAML error: {e}')
   "

   # Check indentation (use spaces, not tabs)
   cat -A python/euclid/rag/app_config.yaml

Path Resolution Issues
----------------------

**Problem**: Configuration files or vector stores not found.

**Solutions**:

.. code-block:: bash

   # Use absolute paths in configuration
   pwd  # Note current directory

   # Update config with full paths
   /full/path/to/vector_store

   # Or ensure you're running from correct directory
   cd /path/to/euclid_rag
   python python/euclid/rag/app.py

Ingestion Issues
================

Network Connection Problems
---------------------------

**Problem**: DPDD ingestion fails with connection timeouts or HTTP errors.

**Solutions**:

.. code-block:: bash

   # Test network connectivity
   curl -I https://euclid.esac.esa.int/dr/q1/dpdd/

   # Use VPN if behind corporate firewall
   # Check proxy settings
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port

   # Retry with longer timeout
   python python/euclid/rag/ingestion/ingest_dpdd.py --config config.yaml --timeout 60

Memory Issues During Ingestion
-------------------------------

**Problem**: Out of memory errors during document processing.

**Solutions**:

.. code-block:: bash

   # Monitor memory usage
   top -p $(pgrep -f ingest_dpdd)

   # Reduce batch size in configuration
   # Process documents in smaller chunks
   # Close other applications to free memory

   # For very large datasets, use streaming processing
   ulimit -v 4000000  # Limit virtual memory

**Problem**: Ingestion takes extremely long time.

**Solutions**:

.. code-block:: yaml

   # Limit topics in dpdd_ingest_config.yaml
   topics_number_limit: 5  # Start small

   # Or process specific topics only
   scrape_all: false
   topics:
     - name: Purpose and Scope
       link: purpose.html

Permission and Storage Issues
-----------------------------

**Problem**: Permission denied when writing vector stores.

**Solutions**:

.. code-block:: bash

   # Check directory permissions
   ls -la vector_store_directory/

   # Fix permissions
   chmod 755 vector_store_directory/
   chown $USER:$USER vector_store_directory/

   # Use a directory you own
   mkdir ~/euclid_vector_stores
   # Update config to point to this directory

**Problem**: Insufficient disk space for vector stores.

**Solutions**:

.. code-block:: bash

   # Check available space
   df -h

   # Clean up old vector stores
   rm -rf old_vector_store_*

   # Use external storage
   ln -s /external/storage/path vector_store_directory

Runtime Issues
==============

Slow Query Responses
--------------------

**Problem**: Chatbot responses take very long time.

**Solutions**:

1. **Check system resources**:

   .. code-block:: bash

      # Monitor CPU and memory
      htop

      # Check disk I/O
      iotop

2. **Optimize vector store**:

   .. code-block:: python

      # Re-index with better parameters
      # Reduce chunk size in ingestion
      # Use faster embedding models

3. **Use lighter LLM models**:

   .. code-block:: bash

      # Switch to smaller model
      docker exec -it euclid_rag-ollama-1 ollama pull mistral:7b

Inaccurate or Irrelevant Responses
----------------------------------

**Problem**: Chatbot provides poor quality answers.

**Solutions**:

1. **Check ingested content quality**:

   .. code-block:: python

      # Inspect vector store contents
      from euclid.rag import chatbot

      retriever = chatbot.configure_retriever()

      # Test with known queries
      results = retriever.get_relevant_documents("test query")
      for doc in results[:3]:
          print(f"Content: {doc.page_content[:200]}...")
          print(f"Source: {doc.metadata}")

2. **Adjust similarity thresholds**:

   .. code-block:: yaml

      # In configuration, adjust retrieval parameters
      retrieval:
        similarity_threshold: 0.7  # Higher = more strict
        max_results: 5

3. **Re-run ingestion with better filtering**:

   .. code-block:: yaml

      # Add more banned sections
      banned_sections:
        names:
          - Header
          - Navigation
          - Footer
          - Table of Contents

Streamlit Interface Issues
--------------------------

**Problem**: Web interface not accessible or loads incorrectly.

**Solutions**:

.. code-block:: bash

   # Check if Streamlit is running
   ps aux | grep streamlit

   # Check port availability
   netstat -tlnp | grep 8501

   # Try different port
   streamlit run rag/app.py --server.port 8080

   # Clear browser cache and cookies
   # Try in incognito/private mode

**Problem**: Session state issues or interface behaves unexpectedly.

**Solutions**:

.. code-block:: bash

   # Restart Streamlit
   pkill -f streamlit
   streamlit run rag/app.py

   # Clear Streamlit cache
   rm -rf ~/.streamlit/

   # Check for conflicting browser extensions

LLM Model Issues
================

Model Loading Failures
-----------------------

**Problem**: Ollama fails to load or run models.

**Solutions**:

.. code-block:: bash

   # Check available models
   docker exec -it euclid_rag-ollama-1 ollama list

   # Check model file integrity
   docker exec -it euclid_rag-ollama-1 ollama show mistral:latest

   # Re-pull corrupted models
   docker exec -it euclid_rag-ollama-1 ollama rm mistral:latest
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:latest

**Problem**: Model responses are gibberish or inappropriate.

**Solutions**:

.. code-block:: bash

   # Try different model versions
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:7b-instruct-v0.2

   # Check model temperature settings
   # Verify prompt templates are correct

Memory Issues with LLM
----------------------

**Problem**: Out of memory errors when running LLM inference.

**Solutions**:

.. code-block:: bash

   # Use smaller models
   docker exec -it euclid_rag-ollama-1 ollama pull mistral:7b  # Instead of larger variants

   # Increase Docker memory limits
   # In Docker Desktop: Settings > Resources > Memory

   # Monitor memory usage
   docker stats euclid_rag-ollama-1

Development and Debug Issues
============================

Import Errors in Development
-----------------------------

**Problem**: Cannot import euclid modules during development.

**Solutions**:

.. code-block:: bash

   # Ensure you're in the right directory
   pwd
   ls python/euclid/

   # Install in development mode
   pip install -e .

   # Add to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

**Problem**: Changes not reflected when testing.

**Solutions**:

.. code-block:: bash

   # For Python code changes
   pip install -e .  # Ensure editable install

   # For Streamlit, restart the server
   # Streamlit should auto-reload, but may need manual restart

Debug Mode and Logging
----------------------

Enable detailed logging for troubleshooting:

.. code-block:: bash

   # Set debug environment variables
   export EUCLID_RAG_DEBUG=true
   export STREAMLIT_LOGGER_LEVEL=debug

   # Run with verbose output
   streamlit run rag/app.py --logger.level debug

.. code-block:: python

   # Add debug prints in Python code
   import logging

   logging.basicConfig(level=logging.DEBUG)

   # In your code
   logging.debug(f"Variable value: {variable}")

Getting Help
============

Log Collection
--------------

When reporting issues, collect relevant logs:

.. code-block:: bash

   # Application logs
   tail -n 100 ~/.streamlit/logs/streamlit.log

   # Docker logs
   docker compose logs --tail=100 euclid_rag
   docker compose logs --tail=100 ollama

   # System information
   python --version
   pip list | grep -E "(streamlit|langchain|faiss)"

Documentation and Support
-------------------------

* **API Documentation**: :doc:`../api`
* **Developer Guide**: :doc:`../developer-guide/index`
* **GitHub Issues**: https://github.com/jeipollack/euclid_rag/issues
* **Configuration Examples**: Check the repository's ``examples/`` directory

Performance Benchmarking
========================

Test System Performance
-----------------------

.. code-block:: python

   # Simple performance test
   import time
   from euclid.rag import chatbot

   start_time = time.time()
   retriever = chatbot.configure_retriever()
   setup_time = time.time() - start_time

   start_time = time.time()
   results = retriever.get_relevant_documents("test query")
   query_time = time.time() - start_time

   print(f"Setup time: {setup_time:.2f}s")
   print(f"Query time: {query_time:.2f}s")
   print(f"Results found: {len(results)}")

Expected Performance
--------------------

Typical performance benchmarks:

* **Vector store loading**: < 5 seconds
* **Simple queries**: < 2 seconds
* **Complex queries**: < 10 seconds
* **Memory usage**: 500MB - 2GB depending on data size

If performance is significantly worse, check:

* Available system memory
* Storage type (SSD vs HDD)
* Network connectivity
* Model size and complexity

Recovery Procedures
===================

Complete Reset
--------------

If all else fails, perform a complete reset:

.. code-block:: bash

   # Stop all services
   docker compose down --volumes
   pkill -f streamlit

   # Remove vector stores
   rm -rf *_vector_store/

   # Clean Python cache
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -delete

   # Reinstall dependencies
   pip uninstall euclid_rag -y
   pip install euclid_rag

   # Re-run ingestion
   python python/euclid/rag/ingestion/ingest_publications.py
   python python/euclid/rag/ingestion/ingest_dpdd.py --config python/euclid/rag/app_config.yaml

Backup and Restore
------------------

Backup your vector stores:

.. code-block:: bash

   # Backup vector stores
   tar -czf euclid_vector_stores_backup.tar.gz *_vector_store/

   # Backup configuration
   cp python/euclid/rag/app_config.yaml app_config_backup.yaml

Restore from backup:

.. code-block:: bash

   # Restore vector stores
   tar -xzf euclid_vector_stores_backup.tar.gz

   # Restore configuration
   cp app_config_backup.yaml python/euclid/rag/app_config.yaml

Still Having Issues?
====================

If you're still experiencing problems after trying these solutions:

1. **Check the GitHub issues**: https://github.com/jeipollack/euclid_rag/issues
2. **Create a new issue** with:
   - Your operating system and Python version
   - Complete error messages
   - Steps to reproduce the problem
   - Relevant log files
3. **Include system information**:

   .. code-block:: bash

      # System info script
      echo "OS: $(uname -a)"
      echo "Python: $(python --version)"
      echo "Pip packages:"
      pip list | grep -E "(euclid|streamlit|langchain|faiss)"
      echo "Docker: $(docker --version)"