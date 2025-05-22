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
from typing import Any

import streamlit as st
import yaml
from langchain.agents.agent import AgentExecutor
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM

from .Agents.publication_agent import get_publication_agent
from .extra_scripts.vectorstore_embedder import E5MpsEmbedder
from .streamlit_callback import get_streamlit_cb


@st.cache_resource
def load_cfg(config_path: str | None = None) -> dict:
    """Load chatbot configuration from app_config.yaml."""
    cfg_path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent / "app_config.yaml"
    )
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


@st.cache_resource(ttl="1h")
def configure_retriever() -> VectorStoreRetriever:
    """Load retriever based on config.yaml."""
    cfg = load_cfg()

    embedder = E5MpsEmbedder(
        model_name=cfg["embeddings"]["model_name"],
        batch_size=cfg["embeddings"]["batch_size"],
    )
    index_dir = Path(cfg["vector_store"]["index_dir"])
    vectorstore = FAISS.load_local(
        str(index_dir), embedder, allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "return_metadata": ["score"]},
    )


def submit_text() -> None:
    """Submit the user input."""
    st.session_state.message_sent = True


def create_agent() -> Callable[[dict], dict]:
    """Return Euclid-AI that **always** delegates to at least one sub-agent."""
    cfg = load_cfg()
    llm = OllamaLLM(**cfg["llm"])

    retriever = configure_retriever()

    # Sub-agents: later add get_dpdd_agent(), get_redmine_agent(), etc ..
    agents = [
        get_publication_agent(llm, retriever),
    ]

    def euclid_ai(inputs: dict, callbacks: list[Any] | None = None) -> dict:
        """
        Loop through sub-agents until one gives a non-empty answer.
        Ensures at least one agent is always used.
        """
        question = inputs["input"]
        for ag in agents:
            answer = ag.run(question, callbacks=callbacks)
            if not answer.lower().startswith("i don't have"):
                return {"answer": answer, "context": []}

        # Nothing found in any agent
        return {"answer": "I don't have that information yet.", "context": []}

    return euclid_ai


def handle_user_input(
    agent: AgentExecutor, msgs: StreamlitChatMessageHistory
) -> None:
    """Manage input from user."""
    if len(msgs.messages) == 0:
        msgs.clear()

    avatars = {"human": "user", "ai": "assistant"}
    avatar_images = {
        "human": STATIC_DIR / "user_avatar.png",
        "ai": STATIC_DIR / "euclid_cartoon.png",
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

            result = agent(
                {"input": user_query, "chat_history": msgs.messages},
                callbacks=[stream_handler],
            )

            answer = result["answer"]
            placeholder.markdown(answer, unsafe_allow_html=False)

        msgs.add_ai_message(answer)
