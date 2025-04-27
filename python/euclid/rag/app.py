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


"""Set up the Streamlit interface for the chatbot, configuring the
retriever, QA chain, session state, UI elements, and handling user of
interactions.
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from euclid.rag.chatbot import (
    configure_retriever,
    create_qa_chain,
    handle_user_input,
)
from euclid.rag.layout import (
    setup_header_and_footer,
    setup_landing_page,
    setup_sidebar,
)

# Load environment variables from .env file
load_dotenv()

# Set page configuration and design
st.set_page_config(
    page_title="Euclid Bot",
    initial_sidebar_state="collapsed",
    page_icon="../../../static/rubin_telescope.png",
)
st.logo("../../../static/rubin_telescope.png")

# Load the CSS file
file_path = Path("../../../static/style.css")
with Path.open(file_path) as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Set up the session state
if "message_sent" not in st.session_state:
    st.session_state.message_sent = False

# Configure the Weaviate retriever and QA chain
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
