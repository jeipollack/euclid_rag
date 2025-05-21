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

from pathlib import Path

import streamlit as st
import yaml
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM

from .FAISS_vectorstore.vectorstore_embedder import E5MpsEmbedder
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


def create_qa_chain(retriever: VectorStoreRetriever) -> Runnable:
    """Create a QA chain for the chatbot."""
    cfg = load_cfg()
    llm = OllamaLLM(**cfg["llm"])

    system_template = """You are Euclid AI Assistant, a helpful assistant
    within the Euclid Consortium Science Ground Segment.
    Use only the provided CONTEXT to answer questions.
    If the answer is not in CONTEXT, reply: I don't have that info.

    ----------------
    {context}
    ----------------"""

    qa_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Question:```{input}```"),
        ],
        input_variables=["input", "chat_history", "context"],
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def handle_user_input(
    qa_chain: Runnable, msgs: StreamlitChatMessageHistory
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
        ).write(msg.content)

    if user_query := st.chat_input(
        placeholder="Message Euclid AI", on_submit=submit_text
    ):
        with st.chat_message("user", avatar=avatar_images["human"]):
            st.write(user_query)
        msgs.add_user_message(user_query)

        with st.chat_message("assistant", avatar=avatar_images["ai"]):
            stream_handler = get_streamlit_cb(st.empty())
            result = qa_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": msgs.messages,
                },
                {"callbacks": [stream_handler]},
            )
            answer = result["answer"]
            msgs.add_ai_message(answer)

            with st.expander("See sources"):
                scores = [
                    chunk.metadata["score"]
                    for chunk in result["context"]
                    if "score" in chunk.metadata
                ]
                if scores:
                    max_score = max(scores)
                    threshold = max_score * 0.9
                    for chunk in result["context"]:
                        score = chunk.metadata.get("score", 0)
                        if score >= threshold:
                            st.info(f"Source: {chunk.metadata['source']}")
