from __future__ import annotations

import json

import httpx

from inference_tracker.config import Settings
from inference_tracker.http_client import HttpClient
from inference_tracker.llm import OpenRouterClassifier, _parse_classification
from inference_tracker.models import Paper


def test_parse_classification_accepts_fenced_json():
    result = _parse_classification(
        '```json\n{"useful": true, "category": "distillation", '
        '"confidence": 0.91, "reason": "Distills a diffusion sampler.", '
        '"matched_topics": ["distillation"]}\n```'
    )
    assert result.useful
    assert result.category == "distillation"
    assert result.confidence == 0.91


def test_openrouter_classifier_uses_configured_model(settings: Settings):
    configured = settings.with_overrides(
        openrouter_api_key="test-key",
        openrouter_model="google/gemma-4-26b-a4b-it:free",
        openrouter_fallback_models=(),
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"useful": true, "category": "inference_optimization", '
                            '"confidence": 0.88, "reason": "Uses caching.", '
                            '"matched_topics": ["cache"]}'
                        }
                    }
                ]
            },
            request=request,
        )

    client = HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    classifier = OpenRouterClassifier(configured, client)
    result = classifier.classify(
        Paper(
            canonical_id="arxiv:2609.12345",
            title="Cached diffusion sampling",
            abstract="A diffusion model with feature caching.",
        )
    )
    assert result.useful
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["model"] == "google/gemma-4-26b-a4b-it:free"
