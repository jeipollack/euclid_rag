# euclid_rag
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://jeipollack.github.io/euclid_rag/)
[![Release](https://img.shields.io/github/v/release/jeipollack/euclid_rag?include_prereleases)](https://github.com/jeipollack/euclid_rag/releases)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](LICENSE)

RAG-powered chatbot for querying Euclid mission documents...

## Euclid RAG: A Local RAG System for Scientific Research

Euclid RAG is an open-source Retrieval-Augmented Generation (RAG) system designed to provide **efficient document retrieval and knowledge augmentation** for the Euclid scientific community. The project aims to integrate local Large Language Models (LLMs) with a vector database to **retrieve, process, and generate** relevant scientific information.

## Origin & Development

This project was initially forked from the [**Rubin Observatory's Rubin RAG system**](https://github.com/lsst-dm/rubin_rag). While we are working in consultation and knowledge-sharing with Rubin developers, **Euclid RAG is evolving in a different direction** to meet the specific needs of the Euclid collaboration. Key differences will include:

- **A focus on local deployment** without API-based LLM dependencies.
- **Different document retrieval strategies** tailored to Euclid's scientific workflows.
- **Potential agentic capabilities** to enhance automated knowledge retrieval and processing.

## Installation

Install euclid_rag in development mode:

```
   git clone https://github.com/yourusername/euclid_rag.git
   cd euclid_rag
   pip install -e .
```
`euclid_rag` is developed by Euclid Consortium Science Ground Segment members at https://github.com/jeipollack/euclid_rag.

## Features

<!-- A bullet list with things that this package does -->

## Developing euclid_rag

The best way to start contributing to rubin_rag is by cloning this repository, creating a virtual environment, and running the `make init` command:

```sh
git clone https://github.com/jeipollack/euclid_rag
cd euclid_rag

python3 -m venv .venv
source .venv/bin/activate

make init

```
### Build the Vector Store
Before running the chatbot, you must ingest data and build the vector store.
The location and type of the vector store(s) are defined in `app_config.yaml`, for example:

```
vector_store:
  type: "faiss"
  redmine_index_dir: "redmine_vector_store"
  public_data_index_dir: "public_data_vector_store"
```

- type — currently only "faiss" is supported.
- {prefix}_index_dir — path where the FAISS index files (index.faiss, index.pkl) will be stored.

This can be a relative path (within the repo) or an absolute path.

If the vector store is missing, the app will fail to start with:

```
RuntimeError: Vector store missing. Please run ingestion before launching the app.
```

### Run ingestion:
```
python python/euclid/rag/ingestion/ingest_publications.py -c /path/to/config_file
```

By default, this uses the `python/euclid/rag/app_config.yaml` file in the repository.

To use a different config file, pass the `-c` / `--config` option:

```
python python/euclid/rag/ingestion/ingest_publications.py -c /path/to/config_file
```

### Run the chatbot:

Once the vector store has been built, you can launch the chatbot:

```sh
cd python/euclid
streamlit run rag/app.py
```

You can run tests and build documentation with [tox](https://tox.wiki/en/latest/):

```sh
tox
```

To learn more about the individual environments:

```sh
tox -av
```


