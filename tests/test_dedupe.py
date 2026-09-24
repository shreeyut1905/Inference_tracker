from __future__ import annotations

from inference_tracker.dedupe import merge_papers
from inference_tracker.models import Paper


def test_merge_papers_deduplicates_arxiv_sources():
    first = Paper(
        canonical_id="arxiv:2609.12345",
        title="Fast Diffusion",
        abstract="Short abstract",
        sources={"arxiv"},
    )
    second = Paper(
        canonical_id="arxiv:2609.12345",
        title="Fast Diffusion",
        abstract="A much longer abstract with implementation details.",
        sources={"huggingface"},
        urls={"huggingface": "https://huggingface.co/papers/2609.12345"},
    )
    merged = merge_papers([first, second])
    assert len(merged) == 1
    assert merged[0].sources == {"arxiv", "huggingface"}
    assert merged[0].abstract.startswith("A much longer")
    assert "huggingface" in merged[0].urls
