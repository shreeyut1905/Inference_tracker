from __future__ import annotations

from datetime import datetime, timezone

from inference_tracker.llm import ClassificationResult
from inference_tracker.models import Paper
from inference_tracker.pipeline import TrackerPipeline
from inference_tracker.storage import StateStore


class Source:
    def fetch(self, start, end):
        return [
            Paper(
                canonical_id="arxiv:2609.12345",
                title="Fast Diffusion Inference with KV Cache",
                abstract="A diffusion model with feature caching.",
            )
        ]


class Classifier:
    def classify(self, paper):
        return ClassificationResult(
            useful=True,
            category="inference_optimization",
            confidence=0.9,
            reason="Uses feature caching.",
            matched_topics=["cache"],
        )


class Sender:
    def __init__(self):
        self.sent = []

    def send(self, papers, now):
        self.sent.extend(papers)


def test_pipeline_uses_llm_result_and_respects_confidence(settings):
    configured = settings.with_overrides(llm_enabled=True, llm_min_confidence=0.8)
    sender = Sender()
    with StateStore(str(settings.state_db_path)) as store:
        pipeline = TrackerPipeline(
            settings=configured,
            arxiv_source=Source(),
            huggingface_source=Source(),
            classifier=Classifier(),
            store=store,
            email_sender=sender,
        )
        report = pipeline.run(
            send=False,
            now=datetime(2026, 9, 24, 2, 30, tzinfo=timezone.utc),
        )
    assert report.useful_count == 1
    assert report.selected[0].useful_category == "inference_optimization"
    assert sender.sent == []
