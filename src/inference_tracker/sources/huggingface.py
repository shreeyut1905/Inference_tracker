from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from ..config import Settings
from ..http_client import HttpClient
from ..models import Paper
from ..time_utils import as_utc, listing_dates_for_window, parse_datetime
from .common import canonical_arxiv_id, canonical_key


class HuggingFaceSource:
    def __init__(self, settings: Settings, client: HttpClient | None = None) -> None:
        self.settings = settings
        self.client = client or HttpClient(
            timeout=30.0,
            user_agent=settings.user_agent,
            retries=3,
        )
        self._owns_client = client is None

    def fetch(self, start: datetime, end: datetime) -> list[Paper]:
        dates = listing_dates_for_window(as_utc(start), as_utc(end))
        if not self.settings.hf_include_previous_day and len(dates) == 2:
            dates = dates[1:]
        papers: dict[str, Paper] = {}
        for listing_date in dates:
            for paper in self._fetch_date(listing_date):
                existing = papers.get(paper.canonical_id)
                if existing is None:
                    papers[paper.canonical_id] = paper
                else:
                    existing.merge(paper)
        return list(papers.values())

    def _fetch_date(self, listing_date: date) -> list[Paper]:
        papers: list[Paper] = []
        page = 1
        limit = 100
        while page <= 50:
            response = self.client.get(
                f"{self.settings.hf_api_url.rstrip('/')}/api/daily_papers",
                params={"date": listing_date.isoformat(), "page": page, "limit": limit},
            )
            payload: Any = response.json()
            entries = payload if isinstance(payload, list) else payload.get("results", [])
            if not isinstance(entries, list) or not entries:
                break
            for entry in entries:
                paper = _entry_to_paper(entry, listing_date, self.settings.hf_api_url)
                if paper is not None:
                    papers.append(paper)
            if len(entries) < limit:
                break
            page += 1
        return papers

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


def _entry_to_paper(
    entry: dict[str, Any], listing_date: date, base_url: str
) -> Paper | None:
    data = entry.get("paper", entry)
    if not isinstance(data, dict):
        return None
    paper_id = str(data.get("id", "")).strip()
    arxiv_id = canonical_arxiv_id(paper_id)
    title = str(data.get("title", "")).strip()
    if not arxiv_id or not title:
        return None
    published_at = parse_datetime(data.get("publishedAt"))
    submitted_at = parse_datetime(data.get("submittedOnDailyAt"))
    if submitted_at is None:
        submitted_at = datetime.combine(listing_date, time.min, tzinfo=UTC)
    authors = [
        str(author.get("name", "")).strip()
        for author in data.get("authors", [])
        if isinstance(author, dict) and str(author.get("name", "")).strip()
    ]
    urls = {
        "huggingface": f"{base_url.rstrip('/')}/papers/{paper_id}",
        "arxiv": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
    }
    github_url = data.get("githubRepo")
    paper = Paper(
        canonical_id=canonical_key(arxiv_id, title),
        title=title,
        abstract=str(data.get("summary", "")).strip(),
        authors=authors,
        sources={"huggingface"},
        source_dates={"huggingface": submitted_at},
        published_at=published_at,
        categories=set(),
        urls=urls,
        upvotes=_optional_int(data.get("upvotes")),
        github_url=str(github_url) if github_url else None,
    )
    return paper


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
