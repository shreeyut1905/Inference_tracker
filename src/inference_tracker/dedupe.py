from __future__ import annotations

from collections.abc import Iterable

from .models import Paper


def merge_papers(papers: Iterable[Paper]) -> list[Paper]:
    merged: dict[str, Paper] = {}
    for paper in papers:
        existing = merged.get(paper.canonical_id)
        if existing is None:
            merged[paper.canonical_id] = paper
        else:
            existing.merge(paper)
    return list(merged.values())
