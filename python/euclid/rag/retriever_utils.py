# This program is licensed under the GNU Lesser General Public License
# (LGPL) v3.0, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""retriever_utils.py.

Utility functions for constructing and caching retrievers from
local vector stores.
This module works with FAISS indexes and supports dynamic loading of multiple
vector stores based on user selection or configuration.

"""

from pathlib import Path
from typing import Any

import streamlit as st
from config_utils import RAGConfig
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

from euclid.rag.embedding_utils import E5MpsEmbedder


def load_documents(format_type: str, path: str) -> list[Document]:
    """
    Load documents from a file in the specified format.

    Parameters
    ----------
    format_type : str
        The format_type of the input file. Supported formats are:
        - "pdf": loads PDF files using PyMuPDFLoader.
        - "markdown": loads Markdown files using TextLoader.
        - "json": currently not implemented.
    path : str
        Path to the file to load.

    Returns
    -------
    list[Document]
        A list of loaded documents compatible with LangChain's
        document format type.

    Raises
    ------
    NotImplementedError
        If the specified format is "json".
    ValueError
        If the specified format is not supported.
    """
    if format_type == "pdf":
        return PyMuPDFLoader(path).load()
    elif format_type == "markdown":
        return TextLoader(path, encoding="utf-8").load()
    elif format_type == "json":
        # Implement custom JSON loader for issues
        raise NotImplementedError("Add your JSON loader logic here.")
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def build_vectorstore_from_sources(
    sources_config: list[str],
    embedder: E5MpsEmbedder,
    index_dir: Path,
    config: RAGConfig,
) -> FAISS:
    """
    Build and save a FAISS vector store from one or more document sources.

    This function loads and processes documents from configured sources,
    splits them into chunks, embeds them using the specified embedder,
    and saves the resulting FAISS vector store to a local directory.

    Parameters
    ----------
    sources_config : list of str
        A list of source names defined in the config under `data.sources`.
    embedder : E5MpsEmbedder
        An instance of the embedding model used to convert text chunks into
        vectors.
    index_dir : Path
        The directory path where the resulting FAISS index will be saved.
    config : RAGConfig
        The configuration object containing `project_root`, `data_root`,
        and `data_sources`.

    Returns
    -------
    FAISS
        A FAISS vector store containing the embedded document chunks.

    Raises
    ------
    ValueError
        If a source is not found in the configuration or if the file path does
        not exist.
    """
    all_chunks = []

    for source_name in sources_config:
        if config.data_sources.get(source_name) is None:
            raise ValueError(
                f"Source '{source_name}' not found in config.data.sources."
            )

        source_config = config.data_sources[source_name]

        pdf_path = config.data_root / source_config["path"]

        if not pdf_path.exists():
            raise ValueError(f"File path {pdf_path} is not a valid file")

        raw_docs = load_documents(source_config["format"], pdf_path)

        for doc in raw_docs:
            doc.metadata["source"] = source_name
            doc.metadata["source_key"] = pdf_path.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=source_config["chunk_size"],
            chunk_overlap=source_config["chunk_overlap"],
        )
        chunks = splitter.split_documents(raw_docs)
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_documents(all_chunks, embedder)
    vectorstore.save_local(str(index_dir))

    return vectorstore


@st.cache_resource(ttl="1h")
def configure_retrievers(_config: RAGConfig) -> dict[str, Any]:
    """Load all retrievers defined in config and return as a dict."""
    embedder = E5MpsEmbedder(
        model_name=_config.embeddings["model_name"],
        batch_size=_config.embeddings["batch_size"],
    )

    retrievers = {}

    for store_name, store_config in _config.vector_stores.items():
        index_dir = store_config.get("index_dir")
        if not index_dir:
            raise (
                f"Skipping vector store '{store_name}': No index_dir defined."
            )
            continue

        index_path = Path("vector_stores") / index_dir

        if index_path.exists():
            vectorstore = FAISS.load_local(
                str(index_path),
                embedder,
                allow_dangerous_deserialization=True,
            )
        else:
            vectorstore = build_vectorstore_from_sources(
                sources_config=store_config["sources"],
                embedder=embedder,
                data_source_name=store_name,
                index_dir=index_dir,
                config=_config,  # optionally pass full config
            )

        retrievers[store_name] = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, "return_metadata": ["score"]},
        )

    return retrievers
