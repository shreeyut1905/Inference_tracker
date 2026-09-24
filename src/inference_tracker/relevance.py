from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import Paper

DOMAIN_TERMS = (
    "diffusion model",
    "diffusion models",
    "diffusion transformer",
    "diffusion transformers",
    "denoising diffusion",
    "score based diffusion",
    "latent diffusion",
    "rectified flow",
    "consistency model",
    "consistency models",
    "diffusion sampler",
    "diffusion sampling",
    "diffusion inference",
    "text to image diffusion",
    "text to video diffusion",
    "dit",
)

GENERIC_DOMAIN_TERMS = (
    "diffusion",
    "flow matching",
)

GENERATIVE_CONTEXT_TERMS = (
    "image",
    "video",
    "audio",
    "visual",
    "pixel",
    "3d",
    "text to image",
    "text to video",
    "image generation",
    "video generation",
    "generative model",
    "generative modeling",
    "latent",
    "sampler",
    "sampling",
    "inference",
    "denoising",
    "token",
)

TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "rl_training": (
        "reinforcement learning",
        "rl",
        "rlhf",
        "grpo",
        "ppo",
        "dpo",
        "policy optimization",
        "reward model",
        "reward modeling",
        "on policy",
        "preference optimization",
    ),
    "distillation": (
        "distillation",
        "teacher student",
        "student model",
        "progressive distillation",
        "consistency distillation",
        "score distillation",
        "step distillation",
    ),
    "inference_optimization": (
        "inference",
        "sampling",
        "fast sampling",
        "few step",
        "one step",
        "zero step",
        "solver",
        "scheduler",
        "latency",
        "throughput",
        "real time",
        "cache",
        "caching",
        "kv cache",
        "attention",
        "linear attention",
        "sparse attention",
        "token merging",
        "pruning",
        "quantization",
        "sparsity",
        "speculative decoding",
        "parallel decoding",
        "memory efficient",
        "efficient inference",
        "speedup",
        "speed up",
    ),
    "other_efficient_diffusion": (
        "efficient",
        "efficiency",
        "fast",
        "accelerat",
        "low bit",
        "low precision",
        "compression",
        "memory",
        "edge device",
        "on device",
        "mobile",
        "distribut",
    ),
}


@dataclass(frozen=True)
class HeuristicAssessment:
    candidate: bool
    score: float
    category: str
    signals: list[str]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _matches(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    escaped_words = r"\s+".join(re.escape(word) for word in normalized_term.split())
    pattern = rf"(?<!\w){escaped_words}(?!\w)"
    return re.search(pattern, text) is not None


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _matches(text, term)]


def assess_paper(paper: Paper) -> HeuristicAssessment:
    title = normalize_text(paper.title)
    abstract = normalize_text(paper.abstract)
    domain_title = _matched_terms(title, DOMAIN_TERMS)
    domain_abstract = _matched_terms(abstract, DOMAIN_TERMS)
    domain_signals = [*domain_title, *domain_abstract]
    if not domain_signals:
        generic_title = _matched_terms(title, GENERIC_DOMAIN_TERMS)
        generic_abstract = _matched_terms(abstract, GENERIC_DOMAIN_TERMS)
        context_title = _matched_terms(title, GENERATIVE_CONTEXT_TERMS)
        context_abstract = _matched_terms(abstract, GENERATIVE_CONTEXT_TERMS)
        if (generic_title or generic_abstract) and (context_title or context_abstract):
            domain_signals = [
                *generic_title,
                *generic_abstract,
                *context_title,
                *context_abstract,
            ]
    if not domain_signals:
        return HeuristicAssessment(False, 0.0, "not_relevant", [])

    group_matches: dict[str, list[str]] = {}
    for category, terms in TARGET_GROUPS.items():
        matches = [*_matched_terms(title, terms), *_matched_terms(abstract, terms)]
        if matches:
            group_matches[category] = list(dict.fromkeys(matches))

    if not group_matches:
        return HeuristicAssessment(False, 35.0, "not_relevant", domain_signals)

    score = 35.0 if domain_title else 20.0
    if domain_abstract:
        score += 10.0
    for matches in group_matches.values():
        score += min(25.0, 10.0 + 3.0 * len(matches))
    category = max(
        group_matches,
        key=lambda name: (len(group_matches[name]), -list(TARGET_GROUPS).index(name)),
    )
    signals = list(dict.fromkeys([*domain_signals, *sum(group_matches.values(), [])]))
    return HeuristicAssessment(True, min(100.0, score), category, signals)
