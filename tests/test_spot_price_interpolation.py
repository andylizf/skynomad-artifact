import datetime as dt

from scripts_multi.fetch_spot_prices.fetch_spot_price_history import _interpolate_prices


def test_interpolate_step_function_basic():
    # Events include a seed event before start and one change after
    start = dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    events = [
        (start - dt.timedelta(seconds=60), 0.50),  # seed
        (start + dt.timedelta(seconds=90), 0.60),  # change
    ]
    prices = _interpolate_prices(
        events=events,
        start_time=start,
        gap_seconds=30,
        num_ticks=5,
        default_price=None,
    )
    # Ticks at 0,30,60,90,120 seconds from start
    assert prices == [0.50, 0.50, 0.50, 0.60, 0.60]


def test_interpolate_requires_event_at_or_before_start():
    # If events list is empty, it must raise (no synthetic defaults)
    start = dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    try:
        _interpolate_prices(
            events=[],
            start_time=start,
            gap_seconds=60,
            num_ticks=3,
            default_price=None,
        )
    except ValueError as e:
        assert "No spot price events" in str(e)
    else:
        raise AssertionError("Expected ValueError for empty events without default_price")


def test_interpolate_with_default_price_seed():
    # If no events exist but default is provided, fill with default
    start = dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    prices = _interpolate_prices(
        events=[],
        start_time=start,
        gap_seconds=60,
        num_ticks=4,
        default_price=0.777,
    )
    assert prices == [0.777, 0.777, 0.777, 0.777]


