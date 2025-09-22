# Change log

`euclid_rag` is versioned according to [Semantic Versioning](https://semver.org/).

Planned and unreleased changes are tracked via individual changelog fragment files located in the [`changelog.d` directory](https://github.com/jeipollack/euclid_rag/tree/main/changelog.d/).

<!-- scriv-insert-here -->

<a id='changelog-0.1.0'></a>
## 0.1.0 (2025-09-22)

### New features

- RAG-powered chatbot for querying Euclid mission documents
- Document ingestion for DPDD (Data Product Description Documents)
- Publication ingestion from Euclid collaboration BibTeX bibliography
- JSON document ingestion (supporting Redmine wiki exports)
- Streamlit web interface for interactive document querying
- FAISS vector store integration for semantic search
- LangChain-based LLM integration with configurable models
- Docker deployment support with Ollama integration

### Other changes

- Initial release consolidation of hackathon features and main development branches

### Documentation

- Complete documentation and user guides
- CI/CD workflow for building, link-checking, and deploying Sphinx docs to GitHub Pages
