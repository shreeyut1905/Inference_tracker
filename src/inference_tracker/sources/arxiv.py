from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import feedparser

from ..config import Settings
from ..http_client import HttpClient
from ..models import Paper
from ..time_utils import as_utc, in_window, parse_datetime
from .common import canonical_arxiv_id, canonical_key

ARXIV_CATEGORY_QUERY = (
    "(cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:cs.SD OR "
    "cat:cs.RO OR cat:stat.ML OR cat:eess.IV)"
)

ARXIV_QUERIES = (
    "all:diffusion",
    'all:"diffusion transformer"',
    "all:denoising",
    'all:"flow matching"',
    'all:"rectified flow"',
)


class ArxivSource:
    def __init__(
        self,
        settings: Settings,
        client: HttpClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.client = client or HttpClient(
            timeout=30.0,
            user_agent=settings.user_agent,
            retries=3,
        )
        self._owns_client = client is None
        self.sleep = sleep

    def fetch(self, start: datetime, end: datetime) -> list[Paper]:
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        papers: dict[str, Paper] = {}
        for query_index, query in enumerate(ARXIV_QUERIES):
            if query_index and self.settings.arxiv_request_delay_seconds > 0:
                self.sleep(self.settings.arxiv_request_delay_seconds)
            date_query = (
                f"({query}) AND {ARXIV_CATEGORY_QUERY} AND submittedDate:"
                f"[{_arxiv_date(start_utc)} TO {_arxiv_date(end_utc)}]"
            )
            response = self.client.get(
                self.settings.arxiv_api_url,
                params={
                    "search_query": date_query,
                    "start": 0,
                    "max_results": 100,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            parsed = feedparser.parse(response.content)
            for entry in parsed.entries:
                paper = _entry_to_paper(entry)
                if paper is None or not in_window(paper.published_at, start_utc, end_utc):
                    continue
                existing = papers.get(paper.canonical_id)
                if existing is None:
                    papers[paper.canonical_id] = paper
                else:
                    existing.merge(paper)
        return list(papers.values())

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _arxiv_date(value: datetime) -> str:
    return as_utc(value).strftime("%Y%m%d%H%M")


def _entry_to_paper(entry: Any) -> Paper | None:
    entry_id = str(entry.get("id", ""))
    arxiv_id = canonical_arxiv_id(entry_id)
    title = str(entry.get("title", "")).strip()
    abstract = str(entry.get("summary", "")).strip()
    if not arxiv_id or not title:
        return None
    published_at = parse_datetime(entry.get("published"))
    updated_at = parse_datetime(entry.get("updated"))
    if published_at is None:
        published_at = updated_at
    if published_at is None:
        return None
    authors = [
        str(author.get("name", "")).strip()
        for author in entry.get("authors", [])
        if str(author.get("name", "")).strip()
    ]
    categories = {
        str(tag.get("term", "")).strip()
        for tag in entry.get("tags", [])
        if str(tag.get("term", "")).strip()
    }
    primary_category = entry.get("arxiv_primary_category")
    if isinstance(primary_category, dict) and primary_category.get("term"):
        categories.add(str(primary_category["term"]).strip())
    urls: dict[str, str] = {}
    for link in entry.get("links", []):
        href = str(link.get("href", "")).strip()
        rel = str(link.get("rel", ""))
        if not href:
            continue
        if rel == "alternate":
            urls.setdefault("arxiv", href)
        elif link.get("type") == "application/pdf" or rel == "related":
            urls.setdefault("pdf", href)
    urls.setdefault("arxiv", f"https://arxiv.org/abs/{arxiv_id}")
    urls.setdefault("pdf", f"https://arxiv.org/pdf/{arxiv_id}")
    paper = Paper(
        canonical_id=canonical_key(arxiv_id, title),
        title=title,
        abstract=abstract,
        authors=authors,
        sources={"arxiv"},
        source_dates={"arxiv": published_at},
        published_at=published_at,
        updated_at=updated_at,
        categories=categories,
        urls=urls,
    )
    return paper
