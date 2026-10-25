import re
import unicodedata
from typing import List, Set

# Regex to identify word tokens (including hyphenated compound words like among-among)
WORD_REGEX = re.compile(r"[\w]+(?:-[\w]+)*", re.UNICODE)


def strip_accents(text: str) -> str:
    """Normalize accented characters: 'Borèh' -> 'Boreh', 'pètro' -> 'petro'."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and normalize whitespace."""
    if not text:
        return ""
    text = strip_accents(text)
    return text.lower().strip()


def tokenize(text: str, expand_compounds: bool = True) -> List[str]:
    """
    Tokenize text into a list of normalized tokens.
    If expand_compounds is True, hyphenated words like 'among-among'
    emit both 'among-among' and the subparts 'among'.
    """
    if not text:
        return []

    norm = normalize_text(text)
    raw_tokens = WORD_REGEX.findall(norm)

    tokens: List[str] = []
    for token in raw_tokens:
        token = token.strip("-_")
        if not token:
            continue
        tokens.append(token)
        if expand_compounds and "-" in token:
            subparts = [p for p in token.split("-") if p]
            tokens.extend(subparts)

    return tokens
