from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return as_utc(datetime.fromisoformat(normalized))
    except ValueError:
        try:
            return as_utc(parsedate_to_datetime(value))
        except (TypeError, ValueError, OverflowError):
            return None


def format_datetime(value: datetime) -> str:
    return as_utc(value).isoformat()


def in_window(value: datetime | None, start: datetime, end: datetime) -> bool:
    if value is None:
        return False
    normalized = as_utc(value)
    return start <= normalized <= end


def listing_dates_for_window(start: datetime, end: datetime) -> tuple[date, ...]:
    first = as_utc(start).date()
    last = as_utc(end).date()
    if first == last:
        return first - timedelta(days=1), first
    dates: list[date] = []
    current = first
    while current <= last:
        dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)
