from __future__ import annotations

from datetime import datetime, timezone

from inference_tracker.config import Settings
from inference_tracker.emailer import EmailSender
from inference_tracker.models import Paper


def test_email_message_contains_links_and_escapes_title(settings: Settings):
    configured = settings.with_overrides(
        mail_from="tracker@example.com",
        mail_to=("reader@example.com",),
        timezone="Asia/Kolkata",
    )
    paper = Paper(
        canonical_id="arxiv:2609.12345",
        title="Fast <diffusion> sampling",
        abstract="A useful abstract.",
        useful_category="inference_optimization",
        useful_confidence=0.9,
        useful_reason="Uses feature caching.",
        sources={"arxiv", "huggingface"},
        urls={
            "arxiv": "https://arxiv.org/abs/2609.12345",
            "huggingface": "https://huggingface.co/papers/2609.12345",
        },
    )
    message = EmailSender(configured).build_message(
        [paper], datetime(2026, 9, 24, 2, 30, tzinfo=timezone.utc)
    )
    assert message["To"] == "reader@example.com"
    assert "2026-09-24" in message["Subject"]
    html_body = message.get_body(("html",)).get_content()
    plain_body = message.get_body(("plain",)).get_content()
    assert "Fast &lt;diffusion&gt; sampling" in html_body
    assert "https://arxiv.org/abs/2609.12345" in html_body
    assert "https://huggingface.co/papers/2609.12345" in html_body
    assert "arXiv papers" in html_body
    assert "Hugging Face Hub papers" in html_body
    assert "arXiv paper" in plain_body
    assert "Hugging Face Hub" in plain_body
