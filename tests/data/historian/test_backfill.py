import pytest


@pytest.mark.asyncio
async def test_backfill_returns_zero_when_up_to_date(historian):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch

    # Simulate: last bar is in the future relative to now
    future_start = datetime.now(timezone.utc) + timedelta(minutes=5)

    with patch.object(historian, "_determine_start", return_value=future_start), \
         patch.object(historian, "_fetch_and_save") as mock_fetch:

        result = await historian.backfill("EUR_USD")

    assert result == 0
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_backfill_calls_fetch_and_save_with_gap(historian):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch, AsyncMock

    past_start = datetime.now(timezone.utc) - timedelta(days=1)

    with patch.object(historian, "_determine_start", return_value=past_start), \
         patch.object(
             historian, "_fetch_and_save", new_callable=AsyncMock, return_value=150
         ) as mock_fetch:

        result = await historian.backfill("EUR_USD")

    assert result == 150
    mock_fetch.assert_called_once()
    call_args = mock_fetch.call_args
    assert call_args[0][0] == "EUR_USD"   # symbol
    assert call_args[0][1] == past_start  # start_dt


@pytest.mark.asyncio
async def test_backfill_returns_total_bars_from_fetch(historian):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch, AsyncMock

    past_start = datetime.now(timezone.utc) - timedelta(hours=2)

    with patch.object(historian, "_determine_start", return_value=past_start), \
         patch.object(
             historian, "_fetch_and_save", new_callable=AsyncMock, return_value=42
         ):
        result = await historian.backfill("GBP_USD")

    assert result == 42


@pytest.mark.asyncio
async def test_backfill_propagates_historian_error(historian):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch, AsyncMock
    from src.data.historian import HistorianError

    past_start = datetime.now(timezone.utc) - timedelta(hours=1)

    with patch.object(historian, "_determine_start", return_value=past_start), \
         patch.object(
             historian, "_fetch_and_save",
             new_callable=AsyncMock,
             side_effect=HistorianError("all retries failed", symbol="EUR_USD"),
         ):
        with pytest.raises(HistorianError) as exc_info:
            await historian.backfill("EUR_USD")

    assert exc_info.value.symbol == "EUR_USD"


@pytest.mark.asyncio
async def test_backfill_logs_start_and_complete(historian, caplog):
    import logging
    from datetime import datetime, timezone, timedelta
    from unittest.mock import patch, AsyncMock

    past_start = datetime.now(timezone.utc) - timedelta(days=1)

    with patch.object(historian, "_determine_start", return_value=past_start), \
         patch.object(
             historian, "_fetch_and_save", new_callable=AsyncMock, return_value=100
         ):
        with caplog.at_level(logging.INFO, logger="src.data.historian"):
            await historian.backfill("EUR_USD")

    messages = " ".join(r.message for r in caplog.records)
    assert "EUR_USD" in messages