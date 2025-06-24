import json
import re
from pathlib import Path
import logging

def load_acronyms(path: str | Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_acronyms(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]{2,6}\b", text))

def match_acronyms(question: str, acronym_dict: dict[str, str]) -> dict[str, str]:
    acronyms = extract_acronyms(question)
    return {ac: acronym_dict[ac] for ac in acronyms if ac in acronym_dict}

def expand_acronyms_in_query(query: str, acronyms: dict) -> str:
    """
    e.g.: "What is DSS?" -> "What is DSS (Data Storage System | Distributed Storage System)?"
    Acronyms are based on http://ycopin.pages.euclid-sgs.uk/euclidator/ by Y. Copin
    """
    def replace_acronym(match):
        word = match.group(0)
        definition = acronyms.get(word.upper())
        return f"{word} ({definition})" if definition else word
    
    pattern = r"\b[A-Z]{2,}(?:/[A-Z]+)?\b"
    return re.sub(pattern, replace_acronym, query)