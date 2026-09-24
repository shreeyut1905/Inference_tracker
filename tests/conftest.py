from __future__ import annotations

import pytest

from inference_tracker.config import Settings


@pytest.fixture
def settings(monkeypatch, tmp_path):
    names = (
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_URL",
        "OPENROUTER_MODEL",
        "OPENROUTER_FALLBACK_MODELS",
        "LLM_ENABLED",
        "LLM_FAIL_OPEN",
        "LLM_MAX_PAPERS",
        "LLM_BATCH_SIZE",
        "LLM_MAX_TOKENS",
        "LLM_MIN_CONFIDENCE",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
        "ARXIV_API_URL",
        "HF_API_URL",
        "ARXIV_REQUEST_DELAY_SECONDS",
        "HF_INCLUDE_PREVIOUS_DAY",
        "LOOKBACK_HOURS",
        "TRACKER_TIMEZONE",
        "RELEVANCE_THRESHOLD",
        "MAX_PAPERS",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_SECURITY",
        "MAIL_FROM",
        "MAIL_TO",
        "SEND_EMPTY_REPORT",
        "STATE_DB_PATH",
        "USER_AGENT",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LLM_MAX_PAPERS", "50")
    monkeypatch.setenv("ARXIV_REQUEST_DELAY_SECONDS", "0")
    monkeypatch.setenv("LOOKBACK_HOURS", "24")
    monkeypatch.setenv("TRACKER_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "35")
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.sqlite3"))
    return Settings.from_env()
