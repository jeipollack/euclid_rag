#
# This file was originally part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project (https://www.lsst.org).
#
# Originally licensed under the MIT License.
# Modifications for the Euclid RAG application were made by members of the Euclid Science Ground Segment.
#
# This program is now licensed under the GNU Lesser General Public License (LGPL) v3.0,
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#


"""Set up a Streamlit-based chatbot using Weaviate for vector search and
GPT-4o-mini for answering user queries.
"""

import os

import streamlit as st
import weaviate
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
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from weaviate.classes.init import Auth

from .custom_weaviate_vector_store import CustomWeaviateVectorStore
from .streamlit_callback import get_streamlit_cb


def submit_text() -> None:
    """Submit the user input."""
    st.session_state.message_sent = True


@st.cache_resource(ttl="1h")
def configure_retriever() -> VectorStoreRetriever:
    """Configure the Weaviate retriever."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    http_host = os.getenv("HTTP_HOST")
    grpc_host = os.getenv("GRPC_HOST")

    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if weaviate_api_key is None:
        raise ValueError("WEAVIATE_API_KEY environment variable is not set")
    if http_host is None:
        raise ValueError("HTTP_HOST environment variable is not set")
    if grpc_host is None:
        raise ValueError("GRPC_HOST environment variable is not set")

    client = weaviate.connect_to_custom(
        http_host=http_host,  # Hostname for the HTTP API connection
        http_port=80,  # Default is 80, WCD uses 443
        http_secure=False,  # Whether to use https (secure) for HTTP
        grpc_host=grpc_host,  # Hostname for the gRPC API connection
        grpc_port=50051,  # Default is 50051, WCD uses 443
        grpc_secure=False,  # Whether to use a secure channel for gRPC
        auth_credentials=Auth.api_key(
            weaviate_api_key
        ),  # The API key to use for authentication
        headers={"X-OpenAI-Api-Key": openai_api_key},
        skip_init_checks=True,
    )

    return CustomWeaviateVectorStore(
        client=client,
        index_name="LangChain_9787ec4b92d3438a8de3ff04ead7ead6",
        text_key="text",
        embedding=OpenAIEmbeddings(),
        attributes=["source", "source_key"],
    ).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "return_metadata": ["score"]},
    )


def create_qa_chain(
    retriever: VectorStoreRetriever,
) -> ChatPromptTemplate:
    """Create a QA chain for the chatbot."""
    # Setup ChatOpenAI (Language Model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # Define the system message template
    system_template = """You are Rubin AI Assistant, a helpful assistant at
    Vera C Rubin Observatory.
    Do your best to answer the questions in as much detail as possible.
    Do not attempt to provide an answer if you do not know the answer.
    In your response, do not recommend reading elsewhere.
    Use the following pieces of context to answer the user's
    question at the end.
    ----------------
    {context}
    ----------------"""

    # Create a ChatPromptTemplate for the QA conversation
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Question:```{input}```"),
        ]
    )

    # Create the QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def handle_user_input(
    qa_chain: ChatPromptTemplate, msgs: StreamlitChatMessageHistory
) -> None:
    """Handle user input and chat history."""
    # Check if the message history is empty or the user
    # clicked the "Clear message history" button
    if len(msgs.messages) == 0:
        msgs.clear()

    # Define avatars for user and assistant messages
    avatars = {"human": "user", "ai": "assistant"}
    avatar_images = {
        "human": "../../../static/user_avatar.png",
        "ai": "../../../static/rubin_telescope.png",
    }

    for msg in msgs.messages:
        st.chat_message(
            avatars[msg.type], avatar=avatar_images[msg.type]
        ).write(msg.content)

    # Handle new user input
    if user_query := st.chat_input(
        placeholder="Message Rubin AI", on_submit=submit_text
    ):
        with st.chat_message("user", avatar=avatar_images["human"]):
            st.write(user_query)
        msgs.add_user_message(user_query)

        with st.chat_message("assistant", avatar=avatar_images["ai"]):
            stream_handler = get_streamlit_cb(st.empty())

            filters = [
                source.lower()
                for source in st.session_state["required_sources"]
            ]
            where_filter = {
                "operator": "Or",
                "operands": [
                    {
                        "path": ["source_key"],
                        "operator": "Equal",
                        "valueText": source,
                    }
                    for source in filters
                ],
            }

            # Invoke retriever logic
            result = qa_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": msgs.messages,
                    "filter": where_filter,
                },
                {"callbacks": [stream_handler]},
            )
            msgs.add_ai_message(result["answer"])  # type: ignore[index]

            # Display source documents in an expander
            with st.expander("See sources"):
                scores = [
                    chunk.metadata["score"]
                    for chunk in result["context"]  # type: ignore[index]
                ]

                if scores:
                    max_score = max(scores)
                    threshold = (
                        max_score * 0.9
                    )  # Set threshold to 90% of the highest score

                    for chunk in result["context"]:  # type: ignore[index]
                        score = chunk.metadata["score"]

                        # Only show sources with scores
                        # significantly higher (above the threshold)
                        if score >= threshold:
                            st.info(f"Source: {chunk.metadata['source']}")
