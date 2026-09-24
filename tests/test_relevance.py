from __future__ import annotations

from inference_tracker.models import Paper
from inference_tracker.relevance import assess_paper


def paper(title: str, abstract: str) -> Paper:
    return Paper(
        canonical_id=f"title:{title}",
        title=title,
        abstract=abstract,
    )


def test_cache_and_attention_paper_is_a_candidate():
    result = assess_paper(
        paper(
            "Fast Diffusion Inference with KV Cache",
            "We reduce diffusion sampling latency using a reusable feature cache.",
        )
    )
    assert result.candidate
    assert result.category == "inference_optimization"
    assert result.score >= 35


def test_rl_training_paper_is_a_candidate():
    result = assess_paper(
        paper(
            "Reinforcement Learning for Diffusion Models",
            "Policy optimization improves the reward of a diffusion image generator.",
        )
    )
    assert result.candidate
    assert result.category == "rl_training"


def test_quality_only_diffusion_paper_is_not_a_candidate():
    result = assess_paper(
        paper(
            "A High Quality Diffusion Image Generator",
            "We introduce a new benchmark and improve visual quality and diversity.",
        )
    )
    assert not result.candidate


def test_general_llm_efficiency_paper_is_not_a_candidate():
    result = assess_paper(
        paper(
            "Efficient Attention for Language Models",
            "KV cache compression speeds up autoregressive language model inference.",
        )
    )
    assert not result.candidate
