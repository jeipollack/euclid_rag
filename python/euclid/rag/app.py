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

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from euclid import STATIC_DIR
from euclid.rag.chatbot import create_euclid_router, handle_user_input
from euclid.rag.layout import setup_header_and_footer, setup_landing_page, setup_sidebar
from euclid.rag.utils.config import load_config

# Load environment variables from .env file
load_dotenv()

# Load configuration file
CONFIG = load_config(Path("python/euclid/rag/app_config.yaml"))

# Set page configuration and design
icon_path = str(STATIC_DIR / "euclid_cartoon.png")
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


# Enable dynamic filtering based on user input
setup_sidebar()

# Configure the vector store retriever and QA chain
if "selected_tool" not in st.session_state:
    st.session_state["selected_tool"] = "redmine"

router = create_euclid_router(CONFIG)

# Set up the landing page
setup_landing_page()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

# Set up the header and footer
setup_header_and_footer(msgs)

# Handle user input and chat history
handle_user_input(router, msgs)
