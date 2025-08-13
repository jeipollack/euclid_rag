# Euclid DPDD Ingestion

This script downloads the **Euclid Science Ground Segment Data Product Description Document (DPDD)**, processes it, and ingests the data into a **FAISS vectorstore** for use in the Euclid RAG system.


## Features

- Downloads DPDD pages from the Euclid website.
- Extracts subtopics and sections, skipping banned sections.
- Splits text into chunks using `langchain.text_splitter.RecursiveCharacterTextSplitter`.
- Stores chunks in a FAISS vectorstore for semantic search.
- Prevents duplicate ingestion by checking existing sources in the vectorstore.


## Requirements

- Python 3.10+
- Dependencies (install via `pip`):

```bash
pip install requests beautifulsoup4 pyyaml langchain langchain-community
```

Note: FAISS must be installed, e.g., `pip install faiss-cpu` or `faiss-gpu` if using GPU.

## Configuration
The script requires a YAML configuration file, typically located at:

```
python/euclid/rag/app_config.yaml
```

Example configuration structure:

```
vector_store:
  index_dir: "path/to/faiss_index"

data:
  dpdd:
    config: "path/to/dpdd_ingest_config.yaml"
```

The DPDD ingestion config (`dpdd_ingest_config.yaml`) should contain:

```yaml
# config/base_urls
base_urls:
  - type: msp
    base_url: https://euclid.esac.esa.int/dr/q1/dpdd/
    version: dm10

# config/topics_to_get
topics:
  - name: Purpose and Scope
    link: purpose.html
  - name: LE1 Data Products
    link: le1dpd/le1index.html
  - name: SIM Data Products
    link: simdpd/simindex.html
  - name: VIS Data Products
    link: visdpd/visindex.html
  # Additional topics can be commented out or added

# Optional limit of topics to ingest
# 0 or missing = no limit (ingest all topics)
topics_number_limit: 0

# Option to scrape all sections
# If true, ignores the topics list and scrapes all available sections
scrape_all: true  # or false

# List of banned sections to skip
banned_sections:
  names:
    - Header
    - Data Header
  full_links:

```
Note: `banned_section` are sections to skip as they might confuse a LLM.

## Usage

Run the script from the command line:

```
python ingest_dpdd.py --config path/to/app_config.yaml
```

Optional arguments:
-`-c, --config`: Path to the YAML configuration file (default: `python/euclid/rag/app_config.yaml`).


## Output

- Vectorstore saved in the directory specified by vector_store.index_dir.
- FAISS index can be loaded later for semantic search in the Euclid RAG system.


## License

This module is licensed under GNU LGPL v3.0. See https://www.gnu.org/licenses/.