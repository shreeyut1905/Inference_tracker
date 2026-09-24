from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .models import Paper
from .time_utils import format_datetime, utc_now


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_papers (
                canonical_id TEXT PRIMARY KEY,
                sent_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviewed_papers (
                canonical_id TEXT PRIMARY KEY,
                reviewed_at TEXT NOT NULL,
                useful INTEGER NOT NULL
            )
            """
        )
        self.connection.commit()

    def sent_ids(self) -> set[str]:
        rows = self.connection.execute("SELECT canonical_id FROM sent_papers").fetchall()
        return {str(row[0]) for row in rows}

    def reviewed_ids(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT canonical_id FROM reviewed_papers"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def filter_unsent(self, papers: Iterable[Paper]) -> list[Paper]:
        sent_ids = self.sent_ids()
        return [paper for paper in papers if paper.canonical_id not in sent_ids]

    def filter_unreviewed(self, papers: Iterable[Paper]) -> list[Paper]:
        excluded_ids = self.sent_ids() | self.reviewed_ids()
        return [paper for paper in papers if paper.canonical_id not in excluded_ids]

    def mark_sent(self, papers: Iterable[Paper], sent_at: datetime | None = None) -> None:
        timestamp = format_datetime(sent_at or utc_now())
        rows = [(paper.canonical_id, timestamp) for paper in papers]
        if not rows:
            return
        self.connection.executemany(
            "INSERT OR IGNORE INTO sent_papers (canonical_id, sent_at) VALUES (?, ?)",
            rows,
        )
        self.connection.commit()

    def mark_reviewed(
        self,
        papers: Iterable[Paper],
        useful: bool = False,
        reviewed_at: datetime | None = None,
    ) -> None:
        timestamp = format_datetime(reviewed_at or utc_now())
        rows = [(paper.canonical_id, timestamp, int(useful)) for paper in papers]
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO reviewed_papers
                (canonical_id, reviewed_at, useful)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
