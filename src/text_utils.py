"""Shared free-text matching helpers used by both the retrieval layer
(src/rag/index.py) and the agent tools (src/agents/tools.py), so district
matching stays consistent between metadata filters and result post-checks."""
from __future__ import annotations

_ACCENT_REPLACEMENTS = {"ö": "o", "ü": "u", "ä": "a", "ß": "ss"}


def normalize_district(text: str) -> str:
    """Lowercase and strip umlauts/diacritics so 'neukolln', 'Neukölln', and
    'neukoelln' all compare equal."""
    lowered = text.strip().lower()
    for accented, plain in _ACCENT_REPLACEMENTS.items():
        lowered = lowered.replace(accented, plain)
    return lowered


def district_matches(query: str, district: str | None) -> bool:
    if not district:
        return False
    normalized_query = normalize_district(query)
    normalized_district = normalize_district(district)
    return normalized_query in normalized_district or normalized_district in normalized_query
