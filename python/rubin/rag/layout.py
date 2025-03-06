#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Set up the sidebar, landing page, and header/footer for a Streamlit
app that interacts with the chatbot.
"""

import streamlit as st
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


def setup_sidebar() -> None:
    """Set up the sidebar for the Streamlit app."""
    st.sidebar.markdown("Select sources to search:")
    st.session_state["required_sources"] = []
    if st.sidebar.checkbox("Confluence", value=True):
        st.session_state["required_sources"].append("confluence")
    if st.sidebar.checkbox("Jira", value=True):
        st.session_state["required_sources"].append("jira")
    if st.sidebar.checkbox("LSST Forum Docs", value=True):
        st.session_state["required_sources"].append("lsstforum")
    if st.sidebar.checkbox("Local Docs", value=True):
        st.session_state["required_sources"].append("localdocs")


def setup_landing_page() -> None:
    """Set up the landing page for the Streamlit app."""
    # Display the landing page until the first message is sent
    if not st.session_state.message_sent:
        # Create three columns to center image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(" ")
        with col2:
            # Add logo (Make sure the logo is in your
            # working directory or provide the full path)
            st.image(
                "../../../static/rubin_avatar_bw.png", clamp=True, width=400
            )
        with col3:
            st.write(" ")
        # Centered title and message
        st.markdown(
            "<h2 class='h2-landing-page'>Hello, I'm Vera!</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<h4 class='h4-landing-page'>Your dedicated "
                "Rubin Observatory bot.</h4>"
            ),
            unsafe_allow_html=True,
        )


def setup_header_and_footer(msgs: StreamlitChatMessageHistory) -> None:
    """Set up the header and footer for the Streamlit app."""

    def clear_text() -> None:
        """Clear the text area."""
        msgs.clear()
        st.session_state.message_sent = False

    st.button(":material/edit_square:", on_click=clear_text)
    st.markdown(
        (
            "<footer class='footer-fixed'>Vera aims for clarity, "
            "but can make mistakes.</footer>"
        ),
        unsafe_allow_html=True,
    )
