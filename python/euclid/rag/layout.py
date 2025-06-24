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


"""Set up the sidebar, landing page, and header/footer for a Streamlit
app that interacts with the chatbot.
"""

import streamlit as st
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from euclid import STATIC_DIR


# def setup_sidebar() -> None:
#     """Set up the sidebar for the Streamlit app."""
#     st.sidebar.markdown("Select sources to search:")
#     st.session_state["required_sources"] = []
#     if st.sidebar.checkbox("Redmine", value=True):
#         st.session_state["required_sources"].append("redmine")
#     if st.sidebar.checkbox("Data Products Descriptions", value=True):
#         st.session_state["required_sources"].append("dpdd")
#     if st.sidebar.checkbox("Euclid SGS Developers", value=True):
#         st.session_state["required_sources"].append("sgsdev")


def setup_sidebar() -> None:
    """Set up the sidebar for the Streamlit app."""

    st.sidebar.title("Settings")

    tabs = st.sidebar.tabs(["Configuration"])

    with tabs[0]:
        st.subheader("Metadata bonus weights")

        if "BONUS_WEIGHTS" not in st.session_state:
            st.session_state.BONUS_WEIGHTS = {
                "pages": 0.5,
                "category": 0.3,
                "year": 0.2,
                "recency": 0.5,
            }

        st.session_state.BONUS_WEIGHTS["pages"] = st.slider(
            "Keyword in page title", 0.0, 1.0, st.session_state.BONUS_WEIGHTS["pages"], 0.05
        )
        st.session_state.BONUS_WEIGHTS["category"] = st.slider(
            "Keyword in page category", 0.0, 1.0, st.session_state.BONUS_WEIGHTS["category"], 0.05
        )
        st.session_state.BONUS_WEIGHTS["year"] = st.slider(
            "Last update corresponding to a given year", 0.0, 1.0, st.session_state.BONUS_WEIGHTS["year"], 0.05
        )
        st.session_state.BONUS_WEIGHTS["recency"] = st.slider(
            "Page recency", 0.0, 1.0, st.session_state.BONUS_WEIGHTS["recency"], 0.05
        )

        st.divider()
        st.subheader("Number of pages selected")

        if "TOP_K_FOR_METADATA_SCORING" not in st.session_state:
            st.session_state.TOP_K_FOR_METADATA_SCORING = {
                "similarity_k": 20,
                "top_metadata_k": 10,
                "top_reranked_k": 5,
            }

        st.session_state.TOP_K_FOR_METADATA_SCORING["similarity_k"] = st.number_input(
            "1. Top K - Semantic similarity", min_value=1, max_value=100, value=st.session_state.TOP_K_FOR_METADATA_SCORING["similarity_k"]
        )
        st.session_state.TOP_K_FOR_METADATA_SCORING["top_metadata_k"] = st.number_input(
            "2. Top K - Metadata bonus weights", min_value=1, max_value=100, value=st.session_state.TOP_K_FOR_METADATA_SCORING["top_metadata_k"]
        )
        st.session_state.TOP_K_FOR_METADATA_SCORING["top_reranked_k"] = st.number_input(
            "3. Top K - Semantic similarity reranked", min_value=1, max_value=100, value=st.session_state.TOP_K_FOR_METADATA_SCORING["top_reranked_k"]
        )


def setup_landing_page() -> None:
    """Set up the landing page for the Streamlit app."""
    if not st.session_state.message_sent:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(" ")
        with col2:
            logo_path = STATIC_DIR / "euclid_cartoon.png"
            st.image(str(logo_path))

            st.markdown(
                "<h2 class='h2-landing-page'>Hello!</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h4 class='h2-landing-page'>I am the "
                "Euclid AI Assistant.</h4>",
                unsafe_allow_html=True,
            )
        with col3:
            st.write(" ")


def setup_header_and_footer(msgs: StreamlitChatMessageHistory) -> None:
    """Set up the header and footer for the Streamlit app."""

    def clear_text() -> None:
        """Clear the text area."""
        msgs.clear()
        st.session_state.message_sent = False

    st.button(":material/edit_square:", on_click=clear_text)
    st.markdown(
        (
            "<footer class='footer-fixed'>Euclid AI Assistant aims for "
            "accuracy, but can make mistakes. \n"
            "If the sources are outdated, please update the tables at "
            "<a href='https://euclid.roe.ac.uk/projects/ops/wiki/Selected_Redmine_Projects' target='_blank'>Redmine Projects</a> & "
            "<a href='https://euclid.roe.ac.uk/projects/ops/wiki/Rag_wiki_pages' target='_blank'>RAG Wiki Pages</a>."
            "</footer>"
        ),
        # (
        #     "<footer class='footer-fixed'>Euclid AI Assistant aims for "
        #     "accuracy, but can make mistakes.</footer>"
        # ),
        unsafe_allow_html=True,
    )