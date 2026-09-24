from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from .config import Settings
from .dedupe import merge_papers
from .emailer import EmailSender
from .llm import ClassificationResult, ClassifierError
from .models import Paper
from .relevance import assess_paper
from .storage import StateStore
from .time_utils import as_utc, utc_now

logger = logging.getLogger(__name__)


class PaperClassifier(Protocol):
    def classify(self, paper: Paper) -> ClassificationResult:
        pass


class PaperSource(Protocol):
    def fetch(self, start: datetime, end: datetime) -> list[Paper]:
        pass


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineReport:
    window_start: datetime
    window_end: datetime
    fetched_count: int
    candidate_count: int
    useful_count: int
    llm_skipped_count: int
    sent_count: int
    selected: list[Paper]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "fetched_count": self.fetched_count,
            "candidate_count": self.candidate_count,
            "useful_count": self.useful_count,
            "llm_skipped_count": self.llm_skipped_count,
            "sent_count": self.sent_count,
            "selected": [paper.to_dict() for paper in self.selected],
            "errors": self.errors,
        }


class TrackerPipeline:
    def __init__(
        self,
        settings: Settings,
        arxiv_source: PaperSource,
        huggingface_source: PaperSource,
        classifier: PaperClassifier | None,
        store: StateStore,
        email_sender: EmailSender,
    ) -> None:
        self.settings = settings
        self.arxiv_source = arxiv_source
        self.huggingface_source = huggingface_source
        self.classifier = classifier
        self.store = store
        self.email_sender = email_sender

    def run(
        self,
        send: bool = False,
        resend: bool = False,
        now: datetime | None = None,
        source_names: Iterable[str] = ("arxiv", "huggingface"),
    ) -> PipelineReport:
        end = as_utc(now or utc_now())
        start = end - timedelta(hours=self.settings.lookback_hours)
        if send:
            self.settings.require_smtp()
            if self.settings.llm_enabled:
                self.settings.require_llm()
        papers, errors = self._fetch_sources(start, end, tuple(source_names))
        if not papers and errors:
            raise PipelineError("; ".join(errors))
        merged = merge_papers(papers)
        unreviewed = merged if resend else self.store.filter_unreviewed(merged)
        candidates: list[Paper] = []
        for paper in unreviewed:
            assessment = assess_paper(paper)
            paper.heuristic_score = assessment.score
            paper.heuristic_signals = assessment.signals
            paper.heuristic_category = assessment.category
            if assessment.candidate:
                candidates.append(paper)
        candidates.sort(key=lambda paper: (-paper.heuristic_score, paper.title.lower()))
        candidate_count = len(candidates)
        llm_skipped = 0
        if self.settings.llm_enabled and len(candidates) > self.settings.llm_max_papers:
            llm_skipped = len(candidates) - self.settings.llm_max_papers
            candidates = candidates[: self.settings.llm_max_papers]
        useful_candidates = self._classify_candidates(candidates, errors)
        useful_ids = {paper.canonical_id for paper in useful_candidates}
        useful_candidates.sort(
            key=lambda paper: (
                -paper.useful_confidence,
                -paper.heuristic_score,
                paper.title.lower(),
            )
        )
        useful = useful_candidates
        if self.settings.max_papers is not None:
            useful = useful[: self.settings.max_papers]
        rejected = [
            paper for paper in candidates if paper.canonical_id not in useful_ids
        ]
        sent_count = 0
        if send:
            if useful or self.settings.send_empty_report:
                self.email_sender.send(useful, end)
                sent_count = len(useful)
            self.store.mark_sent(useful, end)
            self.store.mark_reviewed(rejected, useful=False, reviewed_at=end)
        return PipelineReport(
            window_start=start,
            window_end=end,
            fetched_count=len(merged),
            candidate_count=candidate_count,
            useful_count=len(useful),
            llm_skipped_count=llm_skipped,
            sent_count=sent_count,
            selected=useful,
            errors=errors,
        )

    def _fetch_sources(
        self, start: datetime, end: datetime, source_names: tuple[str, ...]
    ) -> tuple[list[Paper], list[str]]:
        sources = {
            "arxiv": self.arxiv_source,
            "huggingface": self.huggingface_source,
        }
        papers: list[Paper] = []
        errors: list[str] = []
        for name in source_names:
            source = sources.get(name)
            if source is None:
                continue
            try:
                papers.extend(source.fetch(start, end))
            except Exception as error:
                message = f"{name}: {error}"
                errors.append(message)
                logger.exception(message)
        return papers, errors

    def _classify_candidates(
        self, candidates: list[Paper], errors: list[str]
    ) -> list[Paper]:
        if not self.settings.llm_enabled:
            return [
                paper
                for paper in candidates
                if self._apply_heuristic_result(paper)
            ]
        if self.classifier is None:
            raise PipelineError("LLM classification is enabled but no classifier is configured")
        batch_method = getattr(self.classifier, "classify_many", None)
        if callable(batch_method):
            try:
                results = batch_method(candidates)
                if not isinstance(results, list) or len(results) != len(candidates):
                    raise ClassifierError("LLM returned an invalid batch result")
                for paper, result in zip(candidates, results, strict=True):
                    _apply_result(paper, result)
            except (ClassifierError, RuntimeError, TypeError, ValueError) as error:
                if not self.settings.llm_fail_open:
                    raise PipelineError(f"LLM classification failed: {error}") from error
                errors.append(f"LLM fallback for batch: {error}")
                for paper in candidates:
                    _apply_result(paper, self._heuristic_result(paper, fallback=True))
        else:
            for paper in candidates:
                try:
                    result = self.classifier.classify(paper)
                except (ClassifierError, RuntimeError) as error:
                    if not self.settings.llm_fail_open:
                        raise PipelineError(f"LLM classification failed: {error}") from error
                    errors.append(f"LLM fallback for {paper.canonical_id}: {error}")
                    result = self._heuristic_result(paper, fallback=True)
                _apply_result(paper, result)
        return [
            paper
            for paper in candidates
            if paper.useful and paper.useful_confidence >= self.settings.llm_min_confidence
        ]

    def _apply_heuristic_result(self, paper: Paper) -> bool:
        result = self._heuristic_result(paper, fallback=False)
        _apply_result(paper, result)
        return paper.useful

    def _heuristic_result(self, paper: Paper, fallback: bool) -> ClassificationResult:
        useful = paper.heuristic_score >= self.settings.relevance_threshold
        category = paper.heuristic_category
        reason = (
            "LLM evaluation was unavailable; retained by the deterministic heuristic fallback."
            if fallback
            else "Included by the deterministic relevance heuristic."
        )
        return ClassificationResult(
            useful=useful,
            category=category,
            confidence=min(1.0, paper.heuristic_score / 100.0),
            reason=reason,
            matched_topics=paper.heuristic_signals,
        )


def _apply_result(paper: Paper, result: ClassificationResult) -> None:
    paper.useful = result.useful
    paper.useful_category = result.category
    paper.useful_confidence = result.confidence
    paper.useful_reason = result.reason
    paper.matched_topics = result.matched_topics
