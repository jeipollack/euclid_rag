# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
Embedding and vectorstore management utilities for Euclid document ingestion.

This module provides:
- An E5 embedding class with support for MPS/CUDA/CPU.
- A function to load or create a FAISS vectorstore from PDFs.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

from euclid.rag.utils.device import get_device

logger = logging.getLogger(__name__)


class Embedder(Embeddings):
    """
    Embeds text into dense vectors using a HuggingFace model.

    Supports MPS, CUDA, or CPU.

    Pooling strategy (CLS or mean) is inferred automatically.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model to use. Default is "intfloat/e5-small-v2".
    batch_size : int, optional
        Number of texts per batch. Default is 16.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        batch_size: int = 16,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(get_device())
        self._batch_size = batch_size
        self._device = self._model.device
        self._pooling_strategy = self._detect_pooling()

    def _detect_pooling(self) -> str:
        """
        Infer whether CLS or mean pooling should be used based on output shape.

        Enables integration with different types of embedding models.
        """
        sample = ["Test input for pooling detection."]
        tokens = self._tokenizer(
            sample,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            output = self._model(**tokens).last_hidden_state

        if output.shape[1] == 1:
            return "cls"
        return "mean"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using dynamic pooling."""
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                output = self._model(**tokens).last_hidden_state
                if self._pooling_strategy == "cls":
                    embeddings = output[:, 0]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    summed = torch.sum(output * mask, dim=1)
                    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = summed / counts

            vectors = embeddings.cpu().numpy()
            vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
            vectors = normalize(vectors, axis=1)
            results.extend(vectors.tolist())
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into dense vectors."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a dense vector."""
        return self._embed([text])[0]

    @property
    def device(self) -> torch.device:
        """Return the torch device used by the model."""
        return self._device


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    """
    Load documents from a list of PDF files using PyMuPDF.

    Parameters
    ----------
    pdf_paths : List[Path]
        List of paths to PDF files.

    Returns
    -------
    List[Document]
        A list of LangChain Document objects.
    """
    docs = []
    for pdf_path in pdf_paths:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = pdf_path.name
            docs.extend(loaded)
        except Exception as e:
            logger.warning(f"Failed to load PDF '{pdf_path}': {e}")
    return docs


def load_json_documents(json_paths: list[Path]) -> list[Document]:
    """
    Load documents from a list of JSON files.

    Each JSON file should contain a list of dicts
    with at least a "content" field
    and optionally a "metadata" field.

    Parameters
    ----------
    json_paths : List[Path]
        List of paths to JSON files.

    Returns
    -------
    List[Document]
        A list of LangChain Document objects.
    """
    docs = []
    for json_path in json_paths:
        try:
            with Path.open(json_path, encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    if content.strip():
                        docs.append(
                            Document(page_content=content, metadata=metadata)
                        )
        except Exception as e:
            logger.warning(f"Failed to load JSON '{json_path}': {e}")
    return docs


def load_or_create_vectorstore(
    index_dir: Path,
    embedder: Embeddings,
    pdf_paths: list[Path] = list | None,
    json_paths: list[Path] = list | None,
) -> FAISS:
    """
    Load a FAISS vectorstore from disk,
    or create it from PDF and JSON documents.

    Parameters
    ----------
    index_dir : Path
        Directory where the FAISS index is stored (or will be created).
    embedder : Embeddings
        Embedding model implementing the LangChain Embeddings interface.
    pdf_paths : List[Path], optional
        List of PDF paths to load and embed.
    json_paths : List[Path], optional
        List of JSON paths to load and embed.

    Returns
    -------
    FAISS
        The FAISS vectorstore.
    """
    if pdf_paths is None:
        pdf_paths = []
    if json_paths is None:
        json_paths = []
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = load_pdf_documents(pdf_paths) + load_json_documents(json_paths)
    if not docs:
        raise ValueError(
            "No documents found (PDFs and JSONs are empty or invalid)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    index_file = index_dir / "index.faiss"
    if index_file.exists():
        try:
            vectorstore = FAISS.load_local(
                str(index_dir),
                embedder,
                allow_dangerous_deserialization=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load existing FAISS vectorstore."
            ) from exc

        if chunks:
            vectorstore.add_documents(chunks)
            vectorstore.save_local(str(index_dir))
        return vectorstore

    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(str(index_dir))
    return vectorstore
