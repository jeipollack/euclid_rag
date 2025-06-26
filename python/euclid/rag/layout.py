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
    st.sidebar.markdown("Select sources to search:")
    st.session_state["required_sources"] = []
    if st.sidebar.checkbox("Redmine", value=True):
        st.session_state["required_sources"].append("redmine")
    if st.sidebar.checkbox("Data Products Descriptions", value=True):
        st.session_state["required_sources"].append("dpdd")
    if st.sidebar.checkbox("Euclid SGS Developers", value=True):
        st.session_state["required_sources"].append("sgsdev")


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
            "accuracy, but can make mistakes.</footer>"
        ),
        unsafe_allow_html=True,
    )
