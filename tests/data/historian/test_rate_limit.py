import pytest

from unittest.mock import patch


@pytest.mark.asyncio
async def test_rate_limit_no_sleep_when_interval_elapsed(historian):
    import time
    from src.data.historian import _REQUEST_INTERVAL_S

    # Simulate a request that happened 10 seconds ago (past the interval)
    historian._last_request_time = time.monotonic() - (_REQUEST_INTERVAL_S + 2)

    with patch("src.data.historian.asyncio.sleep") as mock_sleep:
        await historian._enforce_rate_limit()
        # sleep should not be called, or called with 0 or negative value
        if mock_sleep.called:
            args = mock_sleep.call_args[0]
            assert args[0] <= 0


@pytest.mark.asyncio
async def test_rate_limit_sleeps_remaining_duration(historian):
    import time
    from src.data.historian import _REQUEST_INTERVAL_S
    from unittest.mock import patch, AsyncMock

    historian._last_request_time = time.monotonic() - 2.0  # 2s ago

    with patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        await historian._enforce_rate_limit()

    mock_sleep.assert_called_once()
    sleep_duration = mock_sleep.call_args[0][0]
    # Remaining â‰ˆ 6s (8 - 2). Allow Â±0.5s tolerance for test execution time.
    assert 5.5 <= sleep_duration <= 6.5


@pytest.mark.asyncio
async def test_rate_limit_no_sleep_on_first_call(historian):
    from unittest.mock import patch, AsyncMock

    # Default: _last_request_time = 0.0
    assert historian._last_request_time == 0.0

    with patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        await historian._enforce_rate_limit()

    # No sleep on the very first call â€” elapsed time since epoch is huge
    mock_sleep.assert_not_called()
