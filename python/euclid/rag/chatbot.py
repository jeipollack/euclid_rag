#
# This file was originally part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project (https://www.lsst.org).
#
# Originally licensed under the MIT License.
# Modifications for the Euclid RAG application were made by members of the
# Euclid Science Ground Segment.
#
# This program is now licensed under the GNU Lesser General Public License
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
# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
Streamlit chatbot interface for the Euclid AI Assistant.
Uses an existing FAISS vectorstore and E5 embeddings for retrieval.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from langchain.agents import Tool
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM

from .extra_scripts.vectorstore_embedder import Embedder
from .retrievers.publication_tool import get_publication_tool
from .streamlit_callback import get_streamlit_cb


def start_ollama_server(model: str) -> None:
    """
    Ensure an Ollama server is running for the requested model.

    Parameters
    ----------
    model : str
        Name of the Ollama model to launch (e.g., "gemma:2b").
    """
    subprocess.Popen(["ollama", "run", model])


@st.cache_resource(ttl="1h")
def configure_retriever(config: dict) -> VectorStoreRetriever:
    """
    Build and cache a FAISS based retriever for Euclid publications.

    Returns
    -------
    VectorStoreRetriever
        Retriever with ``search_type="similarity"`` and *k*=6.
    """
    embedder = Embedder(
        model_name=config["embeddings"]["model_name"],
        batch_size=config["embeddings"]["batch_size"],
    )
    index_dir = Path(config["vector_store"]["index_dir"])
    vectorstore = FAISS.load_local(
        str(index_dir), embedder, allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "return_metadata": ["score"]},
    )


def submit_text() -> None:
    """Flag that the user pressed <enter> in
    the chat box (Streamlit callback).
    """
    st.session_state.message_sent = True


def _build_tools(llm: BaseLanguageModel, config: dict) -> list[Tool]:
    """
    Assemble all domain-specific RAG tools.

    Currently, returns only the publications tool, but additional
    tools (DPDD, Redmine, etc.) can be appended here.

    Parameters
    ----------
    llm : BaseLanguageModel
        Shared language model used by all tools.

    Returns
    -------
    list of Tool
        Tools ready for routing.
    """
    retriever = configure_retriever(config)
    return [
        get_publication_tool(llm, retriever),
    ]


def create_euclid_router(
    config: dict,
) -> Callable[[dict, list[BaseCallbackHandler] | None], dict]:
    """Return Euclid-AI that **always** delegates to at least one sub-agent."""
    start_ollama_server(config["llm"]["model"])
    llm = OllamaLLM(**config["llm"])

    tools = _build_tools(llm, config)

    def router(
        inputs: dict,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> dict:
        """
        Build a router for Euclid RAG tools.

        Returns
        -------
        Callable
            Callable that takes a query dict and optional callbacks,
            and returns an answer with context.
        """
        question = inputs["input"]
        for tool in tools:
            answer = tool.run(question, callbacks=callbacks)
            if not answer.lower().startswith("i don't have"):
                return {"answer": answer, "context": []}

        # Nothing found in any agent
        return {"answer": "I don't have that information yet.", "context": []}

    return router


def handle_user_input(
    router: Callable[[dict, list[BaseCallbackHandler] | None], dict],
    msgs: StreamlitChatMessageHistory,
) -> None:
    """
    Display chat history and handle new user input in Streamlit.

    Parameters
    ----------
    router : Callable
        Callable that routs to correct tools for the response.
    msgs : StreamlitChatMessageHistory
        Chat history object for managing messages.
    """
    if len(msgs.messages) == 0:
        msgs.clear()

    avatars = {"human": "user", "ai": "assistant"}
    avatar_images = {
        "human": Path(__file__).resolve().parents[3]
        / "static"
        / "user_avatar.png",
        "ai": Path(__file__).resolve().parents[3]
        / "static"
        / "euclid_cartoon.png",
    }

    for msg in msgs.messages:
        st.chat_message(
            avatars[msg.type], avatar=avatar_images[msg.type]
        ).markdown(msg.content)

    if user_query := st.chat_input(
        placeholder="Message Euclid AI",
        key="euclid_chat_input",
        on_submit=submit_text,
    ):
        with st.chat_message("user", avatar=avatar_images["human"]):
            st.markdown(user_query)
        msgs.add_user_message(user_query)

        with st.chat_message("assistant", avatar=avatar_images["ai"]):
            placeholder = st.empty()
            stream_handler = get_streamlit_cb(placeholder)

            result = router(
                {"input": user_query, "chat_history": msgs.messages},
                [stream_handler],
            )

            answer = result["answer"]
            placeholder.markdown(answer, unsafe_allow_html=False)

        msgs.add_ai_message(answer)
