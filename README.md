# euclid_rag

**Euclid RAG: A Local RAG System for Scientific Research**

Euclid RAG is an open-source Retrieval-Augmented Generation (RAG) system designed to provide **efficient document retrieval and knowledge augmentation** for the Euclid scientific community. The project aims to integrate local Large Language Models (LLMs) with a vector database to **retrieve, process, and generate** relevant scientific information.

## Origin & Development

This project was initially forked from the [**Rubin Observatory's Rubin RAG system**](https://github.com/lsst-dm/rubin_rag). While we are working in consultation and knowledge-sharing with Rubin developers, **Euclid RAG is evolving in a different direction** to meet the specific needs of the Euclid collaboration. Key differences will include:

- **A focus on local deployment** without API-based LLM dependencies.
- **Different document retrieval strategies** tailored to Euclid's scientific workflows.
- **Potential agentic capabilities** to enhance automated knowledge retrieval and processing.

## Installation

Install from PyPI:

```sh
pip install euclid_rag
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

### Run the chatbot:
```sh
cd python/euclid
streamlit run rag/app.py
```

### Run the chatbot:
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


