"""
Module providing functions to load, extract, and expand acronyms
from a JSON file and within a given text.
"""

import json
import re
from pathlib import Path


def load_acronyms(path: str | Path) -> dict[str, str]:
    """
    Load acronyms from a JSON file.

    Args:
        path (str | Path): The file path to the JSON file containing acronyms.

    Returns
    -------
        dict[str, str]: A dictionary where keys are acronyms
                        and values are their corresponding definitions.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)
    return data


def extract_acronyms(text: str) -> set[str]:
    """
    Extract acronyms from a string.

    Args:
        text (str): The string containing acronyms.

    Returns
    -------
        set[str]: A set of the acronyms.
    """
    return set(re.findall(r"\b[A-Z]{2,6}\b", text))


def match_acronyms(
    question: str, acronym_dict: dict[str, str]
) -> dict[str, str]:
    """
    Match acronyms between a string and a dictionary of acronyms.

    Args:
        question (str): The string containing acronyms.
        acronym_dict (dict[str, str]): A dictionary mapping acronyms (keys)
            to their definitions (values).

    Returns
    -------
        dict[str]: A susbset of acronym_dict
    """
    acronyms = extract_acronyms(question)
    return {ac: acronym_dict[ac] for ac in acronyms if ac in acronym_dict}


def expand_acronyms_in_query(query: str, acronyms: dict) -> str:
    """
    Expand acronyms found in a given query string
    by replacing them with their definitions.

    Args:
        query (str): The input string containing potential acronyms.
            acronyms (dict): A dictionary mapping acronyms (keys)
            to their definitions (values)
            based on http://ycopin.pages.euclid-sgs.uk/euclidator/ by Y. Copin

    Returns
    -------
        str: The modified query string with acronyms expanded
            to include their definitions.

    Example:
        >>> acro = {"DSS": "Data Storage System | Distributed Storage System"}
        >>> expand_acronyms_in_query("What is DSS?", acro)
        'What is DSS (Data Storage System | Distributed Storage System)?'
    """

    def replace_acronym(match: re.Match) -> str:
        word = match.group(0)
        definition = acronyms.get(word.upper())
        return f"{word} ({definition})" if definition else word

    pattern = r"\b[A-Z]{2,}(?:/[A-Z]+)?\b"
    return re.sub(pattern, replace_acronym, query)
