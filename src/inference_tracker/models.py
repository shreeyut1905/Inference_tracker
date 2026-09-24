from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Paper:
    canonical_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    source_dates: dict[str, datetime] = field(default_factory=dict)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    categories: set[str] = field(default_factory=set)
    urls: dict[str, str] = field(default_factory=dict)
    upvotes: int | None = None
    github_url: str | None = None
    heuristic_score: float = 0.0
    heuristic_signals: list[str] = field(default_factory=list)
    heuristic_category: str = "unclassified"
    useful: bool = False
    useful_category: str = "not_relevant"
    useful_confidence: float = 0.0
    useful_reason: str = ""
    matched_topics: list[str] = field(default_factory=list)

    def merge(self, other: Paper) -> None:
        if len(other.title) > len(self.title):
            self.title = other.title
        if len(other.abstract) > len(self.abstract):
            self.abstract = other.abstract
        self.authors = list(dict.fromkeys([*self.authors, *other.authors]))
        self.sources.update(other.sources)
        self.categories.update(other.categories)
        self.source_dates.update(other.source_dates)
        if self.published_at is None or (
            other.published_at is not None and other.published_at < self.published_at
        ):
            self.published_at = other.published_at
        if self.updated_at is None or (
            other.updated_at is not None and other.updated_at > self.updated_at
        ):
            self.updated_at = other.updated_at
        for key, value in other.urls.items():
            self.urls.setdefault(key, value)
        if other.upvotes is not None:
            self.upvotes = max(self.upvotes or 0, other.upvotes)
        if self.github_url is None and other.github_url is not None:
            self.github_url = other.github_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "sources": sorted(self.sources),
            "source_dates": {
                source: value.isoformat() for source, value in self.source_dates.items()
            },
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "categories": sorted(self.categories),
            "urls": self.urls,
            "upvotes": self.upvotes,
            "github_url": self.github_url,
            "heuristic_score": self.heuristic_score,
            "heuristic_signals": self.heuristic_signals,
            "heuristic_category": self.heuristic_category,
            "useful": self.useful,
            "useful_category": self.useful_category,
            "useful_confidence": self.useful_confidence,
            "useful_reason": self.useful_reason,
            "matched_topics": self.matched_topics,
        }
