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


"""Set up of base chatbot based on settings in app_config.yaml file."""

import subprocess
from pathlib import Path

import streamlit as st
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
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM

from .config_utils import RAGConfig
from .streamlit_callback import get_streamlit_cb


def start_ollama_server(model: str) -> None:
    """Ensure the Ollama server is running with the specified model."""
    subprocess.Popen(["ollama", "run", model])


def submit_text() -> None:
    """Submit the user input."""
    st.session_state.message_sent = True


def create_qa_chain(
    config: RAGConfig,
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Create a QA chain for the chatbot."""
    start_ollama_server(config.llm.get("model"))
    llm = OllamaLLM(**config.llm, streaming=True)

    # Define the system message template
    system_template = """You are Euclid AI Assistant, a helpful assistant
    within the Euclid Consortium Science Ground Segment.
    Do your best to answer the questions in as much detail as possible.
    Do not attempt to provide an answer if you do not know the answer.
    In your response, do not recommend reading elsewhere.
    Use the following pieces of context to answer the user's
    question at the end.
    You must only use the provided CONTEXT.
    Never guess, elaborate, or use prior knowledge.
    If the answer is in CONTEXT, quote it directly or summarize precisely.
    Cite PDF file names like [paper.pdf].
    If the answer is not in CONTEXT, reply exactly: I don't have that info.

    ----------------
    {context}
    ----------------"""

    # Create a ChatPromptTemplate for the QA conversation
    qa_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Question:```{input}```"),
        ],
        input_variables=["input", "chat_history", "context"],
    )

    # Create the QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def handle_user_input(
    qa_chain: Runnable, msgs: StreamlitChatMessageHistory
) -> None:
    """Handle user input and chat history."""
    # Check if the message history is empty or the user
    # clicked the "Clear message history" button
    if len(msgs.messages) == 0:
        msgs.clear()

    # Define avatars for user and assistant messages
    avatars = {"human": "user", "ai": "assistant"}
    avatar_images = {
        "human": Path(__file__).resolve().parents[3]
        / "static"
        / "user_avatar.png",
        "ai": Path(__file__).resolve().parents[3]
        / "static"
        / "rubin_telescope.png",
    }

    for msg in msgs.messages:
        st.chat_message(
            avatars[msg.type], avatar=avatar_images[msg.type]
        ).write(msg.content)

    # Handle new user input
    if user_query := st.chat_input(
        placeholder="Message Euclid AI", on_submit=submit_text
    ):
        with st.chat_message("user", avatar=avatar_images["human"]):
            st.write(user_query)
        msgs.add_user_message(user_query)

        with st.chat_message("assistant", avatar=avatar_images["ai"]):
            stream_handler = get_streamlit_cb(st.empty())

            # Invoke retriever logic
            result = qa_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": msgs.messages,
                },
                {"callbacks": [stream_handler]},
            )
            answer = result["answer"]
            msgs.add_ai_message(answer)

            # Optional: Display source documents to user
            with st.expander("See sources"):
                scores = [
                    chunk.metadata["score"]
                    for chunk in result["context"]
                    if "score" in chunk.metadata
                ]

                if scores:
                    max_score = max(scores)
                    threshold = (
                        max_score * 0.9
                    )  # Set threshold to 90% of the highest score

                    for chunk in result["context"]:
                        score = chunk.metadata.get("score", 0)

                        # Only show sources with scores above threshold
                        if score >= threshold:
                            st.info(f"Source: {chunk.metadata['source']}")
