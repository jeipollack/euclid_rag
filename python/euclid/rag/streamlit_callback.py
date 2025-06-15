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


"""Create a Streamlit callback handler that dynamically updates a UI
container with new tokens from a language model.
"""

import inspect
import os
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import (
    add_script_run_ctx,
    get_script_run_ctx,
)


# Define a function to create a callback handler
# for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Create a Streamlit callback handler that updates the provided
    Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container
        where the text will be rendered.

    Returns
    -------
        BaseCallbackHandler: An instance of a callback handler
        configured for Streamlit.
    """

    # Define a custom callback handler class for managing and
    # displaying stream events in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a
        Streamlit container with new tokens.
        """

        def __init__(
            self,
            container: st.delta_generator.DeltaGenerator,
            initial_text: str = "",
        ) -> None:
            """
            Initialize the StreamHandler with a Streamlit container and
            optional initial text.
            Args:
                container (st.delta_generator.DeltaGenerator):
                    The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to
                start within the container.
            """
            self.container = container
            self.text = initial_text
            self.run_id_ignore_token = None
            self.instance_id = os.urandom(4).hex()

        def on_llm_start(
            self, serialized: dict, prompts: list, **kwargs: Any
        ) -> None:
            """Trigger when the language model starts generating tokens."""
            # Workaround to prevent showing the rephrased question as output
            if prompts[0].startswith("Human"):
                self.run_id_ignore_token = kwargs.get("run_id")

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            """
            Trigger when a new token is
            received (e.g., from a language model).
            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            st.session_state.spinner_active = False
            if self.run_id_ignore_token == kwargs.get("run_id"):
                return
            self.text += token
            self.container.markdown(self.text)

    # Define a type variable for generic type hinting in
    # the decorator, to maintain
    # input function and wrapped function return type
    fn_return_type = TypeVar("fn_return_type")

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(
        fn: Callable[..., fn_return_type],
    ) -> Callable[..., fn_return_type]:
        """
        Ensure that the decorated function runs within
        the Streamlit execution context.
        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.

        Returns
        -------
            Callable[..., fn_return_type]: The decorated function
            that includes the Streamlit context setup.
        """
        ctx = (
            get_script_run_ctx()
        )  # Retrieve the current Streamlit script execution context

        def wrapper(*args: Any, **kwargs: Any) -> fn_return_type:
            """
            Add the Streamlit context and then calls the original function.

            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.

            Returns
            -------
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(
                ctx=ctx
            )  # Add the Streamlit context to the current execution
            return fn(
                *args, **kwargs
            )  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with
    # the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith("on_"):  # Identify callback methods
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Wrap and replace the method

    # Return the fully configured StreamHandler instance with
    # the context-aware callback methods
    return st_cb
