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


"""Set up a Streamlit-based chatbot using Weaviate for vector search and
GPT-4o-mini for answering user queries.
"""

import subprocess
from pathlib import Path

import streamlit as st
import torch
import yaml
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaLLM
from transformers import AutoModel, AutoTokenizer

from .streamlit_callback import get_streamlit_cb


def run_ollama(model: str) -> None:
    """Set up the Ollama model."""
    subprocess.Popen(["ollama", "run", model])


@st.cache_resource
def load_cfg() -> dict:
    """Load chatbot configuration from config.yaml."""
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def submit_text() -> None:
    """Submit the user input."""
    st.session_state.message_sent = True


@st.cache_resource(ttl="1h")
def configure_retriever() -> VectorStoreRetriever:
    """Load FAISS retriever based on config.yaml."""
    cfg = load_cfg()

    embedder = E5MpsEmbedder(
        model_name=cfg["embeddings"]["model_name"],
        batch_size=cfg["embeddings"]["batch_size"],
    )
    index_dir = Path(cfg["vector_store"]["index_dir"])

    if index_dir.exists():
        vectorstore = FAISS.load_local(
            str(index_dir), embedder, allow_dangerous_deserialization=True
        )
    else:
        pdf_path = (
            Path(__file__).resolve().parents[1] / cfg["data"]["pdf_path"]
        )

        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = pdf_path.name
            d.metadata["source_key"] = pdf_path.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["data"]["chunk_size"],
            chunk_overlap=cfg["data"]["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embedder)
        vectorstore.save_local(str(index_dir))

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6, "return_metadata": ["score"]},
    )


# Embeddings
def get_device() -> str:
    """Return the best available device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class E5MpsEmbedder(Embeddings):
    """E5 embedding model running on GPU, MPS or CPU."""

    def __init__(
        self, model_name: str = "intfloat/e5-small-v2", batch_size: int = 16
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = get_device()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vector representations."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            toks = self.tokenizer(
                texts[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                out.extend(
                    self.model(**toks)
                    .last_hidden_state[:, 0, :]
                    .cpu()
                    .numpy()
                    .tolist()
                )
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed([text])[0]


def create_qa_chain(
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Create a QA chain for the chatbot."""
    cfg = load_cfg()
    run_ollama(cfg["llm"]["model"])
    llm = OllamaLLM(**cfg["llm"], streaming=True)

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

            # Display source documents in an expander
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

                    for chunk in result["context"]:  # type: ignore[index]
                        score = chunk.metadata.get("score", 0)

                        # Only show sources with scores
                        # significantly higher (above the threshold)
                        if score >= threshold:
                            st.info(f"Source: {chunk.metadata['source']}")
