from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote


def canonical_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    candidate = unquote(value.strip())
    candidate = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", candidate, flags=re.I)
    candidate = re.sub(r"^arxiv:", "", candidate, flags=re.I)
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.I)
    candidate = candidate.split("?", 1)[0].strip("/")
    candidate = re.sub(r"v\d+$", "", candidate)
    if not candidate:
        return None
    if re.fullmatch(r"\d{4}\.\d{4,5}", candidate):
        return candidate
    if re.fullmatch(r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}", candidate):
        return candidate
    return None


def normalized_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def canonical_key(arxiv_id: str | None, title: str) -> str:
    normalized_id = canonical_arxiv_id(arxiv_id)
    if normalized_id:
        return f"arxiv:{normalized_id}"
    title_key = normalized_title(title)
    return f"title:{title_key[:240]}"
