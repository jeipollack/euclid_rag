# This program is licensed under the GNU Lesser General Public License
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
"""config_utils.py.

Utility functions and classes for loading, parsing,
and validating application configuration from YAML
files.

Intended for use at app startup to load and cache configuration data
globally.
"""

from pathlib import Path
from typing import Any

import yaml


class RAGConfig:
    """
    Class for loading and accessing application configuration from a YAML file.

    This class reads a configuration file at startup, resolves any relative
    paths, and provides convenient access to configuration sections such as
    LLMs, embeddings, data sources, and vector stores. It is intended to be
    used as a singleton or globally cached configuration object in RAG-based
    applications.

    Parameters
    ----------
    path : str, optional
        Path to the YAML configuration file.
        Defaults to "config/app_config.yaml".

    Attributes
    ----------
    config_path : pathlib.Path
        Absolute path to the configuration file.
    project_root : pathlib.Path
        Root directory of the project (assumed to be the parent of the config
        directory).
    data_root : pathlib.Path
        Path to the data directory inside the project root.
    _config : dict
        Parsed configuration data loaded from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the specified path.

    Methods
    -------
    get_data_source_config(name)
        Returns the configuration dictionary for a specific data source.
    get_vector_store_config(name)
        Returns the configuration dictionary for a specific vector store.

    Properties
    ----------
    llm : dict
        The configuration dictionary for the LLM (large language model)
        section.
    embeddings : dict
        The configuration dictionary for the embeddings section.
    data_sources : dict
        The dictionary of all configured data sources.
    vector_stores : dict
        The dictionary of all configured vector stores.
    """

    def __init__(self, path: str = "config/app_config.yaml") -> None:
        # Store the absolute path to the config file
        self.config_path = Path(path).expanduser().resolve()

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )

        self.project_root = self.config_path.parent.parent
        self.data_root = self.project_root / "data"

        # Load and resolve the config
        self._config = self._load_yaml()
        self._resolve_paths()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RAGConfig)
            and self.config_path == other.config_path
        )

    def __hash__(self) -> int:
        return hash(self.config_path)

    def _load_yaml(self) -> dict[str, Any]:
        with self.config_path.open("r") as f:
            return yaml.safe_load(f)

    def _resolve_paths(self) -> None:
        """Resolve relative paths in data_sources."""
        data_sources = self._config.get("data_sources", {})
        for source in data_sources.values():
            if "path" in source:
                original_path = Path(source["path"])
                if not original_path.is_absolute():
                    resolved_path = (
                        self.project_root / original_path
                    ).resolve()
                    source["path"] = str(resolved_path)

    @property
    def llm(self) -> dict[str, Any]:
        return self._config.get("llm", {})

    @property
    def embeddings(self) -> dict[str, Any]:
        return self._config.get("embeddings", {})

    @property
    def data_sources(self) -> dict[str, Any]:
        return self._config.get("data", {}).get("sources", {})

    @property
    def vector_stores(self) -> dict[str, Any]:
        return self._config.get("vector_store", {}).get("stores", {})

    def get_data_source_config(self, name: str) -> dict[str, Any]:
        return self.data_sources.get(name, {})

    def get_vector_store_config(self, name: str) -> dict[str, Any]:
        return self.vector_stores.get(name, {})
