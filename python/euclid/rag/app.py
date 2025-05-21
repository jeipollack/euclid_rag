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


"""Set up the Streamlit interface for the chatbot, configuring the
retriever, QA chain, session state, UI elements, handling user of
interactions, and scheduled ingestion of data.
"""

import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from euclid import STATIC_DIR
from euclid.rag.chatbot import (
    configure_retriever,
    create_qa_chain,
    handle_user_input,
)
from euclid.rag.extra_scripts.parse_EC_BibTeX import run_bibtex_ingestion
from euclid.rag.layout import (
    setup_header_and_footer,
    setup_landing_page,
    setup_sidebar,
)

# Load environment variables from .env file
load_dotenv()
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

# Automated data ingestion (for now: BibTeX only)
threading.Thread(target=run_bibtex_ingestion, daemon=True).start()

# Set page configuration and design
icon_path = str(STATIC_DIR / "rubin_telescope.png")
st.set_page_config(
    page_title="Euclid Bot",
    initial_sidebar_state="collapsed",
    page_icon=icon_path,
)

# Load the CSS file
file_path = STATIC_DIR / "style.css"

with Path.open(file_path) as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Set up the session state
if "message_sent" not in st.session_state:
    st.session_state.message_sent = False


vectorstore_path = Path("rag/FAISS_vectorstore/index.faiss")
while not vectorstore_path.exists():
    with st.spinner("Preparing Euclid knowledge base..."):
        while not vectorstore_path.exists():
            time.sleep(1)

# Configure the vectorstore retriever and QA chain
retriever = configure_retriever()
qa_chain = create_qa_chain(retriever)

# Enable dynamic filtering based on user input
setup_sidebar()

# Set up the landing page
setup_landing_page()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

# Set up the header and footer
setup_header_and_footer(msgs)

# Handle user input and chat history
handle_user_input(qa_chain, msgs)
