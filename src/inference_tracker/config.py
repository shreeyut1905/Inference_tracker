from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, cast


class ConfigurationError(ValueError):
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _bool_env(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _int_env(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value else default
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error


def _float_env(name: str, default: float) -> float:
    value = _env(name)
    try:
        return float(value) if value else default
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error


def _csv_env(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = _env(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_fallback_models: tuple[str, ...]
    llm_enabled: bool
    llm_fail_open: bool
    llm_max_papers: int
    llm_batch_size: int
    llm_max_tokens: int
    llm_min_confidence: float
    llm_timeout_seconds: float
    llm_max_retries: int
    arxiv_api_url: str
    openrouter_api_url: str
    hf_api_url: str
    arxiv_request_delay_seconds: float
    hf_include_previous_day: bool
    lookback_hours: int
    timezone: str
    relevance_threshold: float
    max_papers: int | None
    smtp_host: str | None
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    smtp_security: str
    mail_from: str | None
    mail_to: tuple[str, ...]
    send_empty_report: bool
    state_db_path: str
    user_agent: str

    @classmethod
    def from_env(cls) -> Settings:
        security = _env("SMTP_SECURITY", "starttls").lower()
        if security not in {"starttls", "ssl", "none"}:
            raise ConfigurationError("SMTP_SECURITY must be starttls, ssl, or none")
        max_papers_value = _env("MAX_PAPERS")
        max_papers = int(max_papers_value) if max_papers_value else None
        if max_papers is not None and max_papers <= 0:
            raise ConfigurationError("MAX_PAPERS must be positive")
        lookback_hours = _int_env("LOOKBACK_HOURS", 24)
        if lookback_hours <= 0:
            raise ConfigurationError("LOOKBACK_HOURS must be positive")
        relevance_threshold = _float_env("RELEVANCE_THRESHOLD", 35.0)
        if not 0 <= relevance_threshold <= 100:
            raise ConfigurationError("RELEVANCE_THRESHOLD must be between 0 and 100")
        llm_max_papers = _int_env("LLM_MAX_PAPERS", 50)
        if llm_max_papers <= 0:
            raise ConfigurationError("LLM_MAX_PAPERS must be positive")
        llm_batch_size = _int_env("LLM_BATCH_SIZE", 5)
        if not 1 <= llm_batch_size <= 20:
            raise ConfigurationError("LLM_BATCH_SIZE must be between 1 and 20")
        llm_max_tokens = _int_env("LLM_MAX_TOKENS", 2000)
        if llm_max_tokens <= 0:
            raise ConfigurationError("LLM_MAX_TOKENS must be positive")
        llm_min_confidence = _float_env("LLM_MIN_CONFIDENCE", 0.5)
        if not 0 <= llm_min_confidence <= 1:
            raise ConfigurationError("LLM_MIN_CONFIDENCE must be between 0 and 1")
        smtp_port = _int_env("SMTP_PORT", 587)
        if not 1 <= smtp_port <= 65535:
            raise ConfigurationError("SMTP_PORT must be between 1 and 65535")
        llm_max_retries = _int_env("LLM_MAX_RETRIES", 3)
        if llm_max_retries < 0:
            raise ConfigurationError("LLM_MAX_RETRIES cannot be negative")
        return cls(
            openrouter_api_key=_env("OPENROUTER_API_KEY") or None,
            openrouter_model=_env(
                "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"
            ),
            openrouter_fallback_models=_csv_env(
                "OPENROUTER_FALLBACK_MODELS",
                (
                    "liquid/lfm-2.5-2.6b:free",
                    "nvidia/nemotron-3.5-lightning:free",
                    "qwen/qwen3.8-27b:free",
                ),
            ),
            llm_enabled=_bool_env("LLM_ENABLED", True),
            llm_fail_open=_bool_env("LLM_FAIL_OPEN", False),
            llm_max_papers=llm_max_papers,
            llm_batch_size=llm_batch_size,
            llm_max_tokens=llm_max_tokens,
            llm_min_confidence=llm_min_confidence,
            llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 45.0),
            llm_max_retries=llm_max_retries,
            arxiv_api_url=_env("ARXIV_API_URL", "https://export.arxiv.org/api/query"),
            openrouter_api_url=_env(
                "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
            ),
            hf_api_url=_env("HF_API_URL", "https://huggingface.co"),
            arxiv_request_delay_seconds=_float_env("ARXIV_REQUEST_DELAY_SECONDS", 3.0),
            hf_include_previous_day=_bool_env("HF_INCLUDE_PREVIOUS_DAY", True),
            lookback_hours=lookback_hours,
            timezone=_env("TRACKER_TIMEZONE", "Asia/Kolkata"),
            relevance_threshold=relevance_threshold,
            max_papers=max_papers,
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com") or None,
            smtp_port=smtp_port,
            smtp_username=_env("SMTP_USERNAME") or None,
            smtp_password=_env("SMTP_PASSWORD") or None,
            smtp_security=security,
            mail_from=_env("MAIL_FROM") or None,
            mail_to=_csv_env("MAIL_TO"),
            send_empty_report=_bool_env("SEND_EMPTY_REPORT", False),
            state_db_path=_env("STATE_DB_PATH", ".tracker-state.sqlite3"),
            user_agent=_env(
                "USER_AGENT", "InferenceTracker/0.1 (personal research paper tracker)"
            ),
        )

    def require_llm(self) -> None:
        if self.llm_enabled and not self.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required when LLM_ENABLED=true"
            )

    def require_smtp(self) -> None:
        missing = [
            name
            for name, value in (
                ("SMTP_HOST", self.smtp_host),
                ("MAIL_FROM", self.mail_from),
                ("MAIL_TO", self.mail_to),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing email configuration: {', '.join(missing)}")
        if self.smtp_security != "none" and not (
            self.smtp_username and self.smtp_password
        ):
            raise ConfigurationError(
                "SMTP_USERNAME and SMTP_PASSWORD are required unless SMTP_SECURITY=none"
            )

    def with_overrides(self, **overrides: object) -> Settings:
        return cast(Settings, replace(self, **cast(dict[str, Any], overrides)))
