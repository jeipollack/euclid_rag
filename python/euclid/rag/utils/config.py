"""Utility for loading and parsing config files."""

from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration from a file.

    Parameters
    ----------
    config_path : Path
        Path to the config YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with config_path.open("r") as f:
        return yaml.safe_load(f)
