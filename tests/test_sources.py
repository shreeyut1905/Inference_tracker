from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from inference_tracker.config import Settings
from inference_tracker.http_client import HttpClient
from inference_tracker.sources.arxiv import ArxivSource
from inference_tracker.sources.huggingface import HuggingFaceSource

FIXTURES = Path(__file__).parent / "fixtures"


def mock_client(handler):
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_arxiv_source_parses_atom_and_filters_window(settings: Settings):
    payload = (FIXTURES / "arxiv.xml").read_bytes()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=payload, request=request)

    source = ArxivSource(settings, mock_client(handler))
    papers = source.fetch(
        datetime(2026, 9, 23, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 24, 0, tzinfo=timezone.utc),
    )
    assert len(papers) == 1
    assert papers[0].canonical_id == "arxiv:2609.12345"
    assert papers[0].sources == {"arxiv"}
    assert len(calls) == 5
    assert "submittedDate" in calls[0].url.params["search_query"]


def test_huggingface_source_parses_daily_api(settings: Settings):
    payload = json.loads((FIXTURES / "hf.json").read_text())
    requested_dates = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_dates.append(request.url.params["date"])
        return httpx.Response(200, json=payload, request=request)

    source = HuggingFaceSource(settings, mock_client(handler))
    papers = source.fetch(
        datetime(2026, 9, 23, 2, 30, tzinfo=timezone.utc),
        datetime(2026, 9, 24, 2, 30, tzinfo=timezone.utc),
    )
    assert len(papers) == 1
    assert papers[0].canonical_id == "arxiv:2609.12345"
    assert requested_dates == ["2026-09-23", "2026-09-24"]
    assert papers[0].source_dates["huggingface"].date().isoformat() == "2026-09-24"
