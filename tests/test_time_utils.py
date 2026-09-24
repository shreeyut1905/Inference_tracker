from __future__ import annotations

from datetime import datetime, timezone

from inference_tracker.time_utils import in_window, listing_dates_for_window, parse_datetime


def test_parse_datetime_supports_iso_and_rfc_dates():
    assert parse_datetime("2026-09-23T12:00:00Z") == datetime(
        2026, 9, 23, 12, tzinfo=timezone.utc
    )
    assert parse_datetime("Wed, 23 Sep 2026 12:00:00 GMT") == datetime(
        2026, 9, 23, 12, tzinfo=timezone.utc
    )


def test_in_window_is_inclusive_at_both_boundaries():
    start = datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
    end = datetime(2026, 9, 23, 8, tzinfo=timezone.utc)
    assert in_window(start, start, end)
    assert in_window(end, start, end)
    assert not in_window(datetime(2026, 9, 22, 7, 59, tzinfo=timezone.utc), start, end)


def test_listing_dates_include_previous_day_for_same_day_window():
    value = datetime(2026, 9, 23, 2, 30, tzinfo=timezone.utc)
    assert listing_dates_for_window(value, value) == (
        datetime(2026, 9, 22, tzinfo=timezone.utc).date(),
        datetime(2026, 9, 23, tzinfo=timezone.utc).date(),
    )
