import re
import pytest
import aiohttp


@pytest.fixture
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.mark.asyncio
async def test_fetch_chunk_success_returns_bars(historian, http_session):
    from aioresponses import aioresponses
    from datetime import datetime, timezone

    payload = {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850",
                "high": "1.0860",
                "low": "1.0845",
                "close": "1.0855",
                "volume": "200",
            }
        ],
    }

    with aioresponses() as m:
        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), payload=payload)
        bars = await historian._fetch_chunk(
            session=http_session,
            api_symbol="EUR/USD",
            symbol="EUR_USD",
            start_dt=datetime(2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2026, 4, 12, 23, 59, 0, tzinfo=timezone.utc),
        )

    assert len(bars) == 1
    assert bars[0].symbol == "EUR_USD"


@pytest.mark.asyncio
async def test_fetch_chunk_retries_on_429(historian, http_session):
    from aioresponses import aioresponses
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock

    success_payload = {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850",
                "high": "1.0860",
                "low": "1.0845",
                "close": "1.0855",
                "volume": "200",
            }
        ],
    }

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ):

        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), status=429)
        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), payload=success_payload)

        bars = await historian._fetch_chunk(
            session=http_session,
            api_symbol="EUR/USD",
            symbol="EUR_USD",
            start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
        )

    assert len(bars) == 1


@pytest.mark.asyncio
async def test_fetch_chunk_retries_on_http_500(historian, http_session):
    from aioresponses import aioresponses
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock

    success_payload = {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-04-12 10:00:00",
                "open": "1.0850",
                "high": "1.0860",
                "low": "1.0845",
                "close": "1.0855",
                "volume": "200",
            }
        ],
    }

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ):

        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), status=500, body="error")
        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), payload=success_payload)

        bars = await historian._fetch_chunk(
            session=http_session,
            api_symbol="EUR/USD",
            symbol="EUR_USD",
            start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
        )

    assert len(bars) == 1


@pytest.mark.asyncio
async def test_fetch_chunk_raises_historian_error_after_all_retries(
    historian, http_session
):
    from aioresponses import aioresponses
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock
    from src.data.historian import HistorianError, _MAX_RETRIES

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ):

        for _ in range(_MAX_RETRIES):
            m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), status=503, body="down")

        with pytest.raises(HistorianError) as exc_info:
            await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )

    assert exc_info.value.symbol == "EUR_USD"


@pytest.mark.asyncio
async def test_fetch_chunk_returns_empty_on_api_error_body(historian, http_session):
    from aioresponses import aioresponses
    from datetime import datetime, timezone

    error_payload = {
        "status": "error",
        "message": "You have run out of API credits.",
        "code": 429,
    }

    with aioresponses() as m:
        m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), payload=error_payload)

        bars = await historian._fetch_chunk(
            session=http_session,
            api_symbol="EUR/USD",
            symbol="EUR_USD",
            start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
            end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
        )

    assert bars == []


@pytest.mark.asyncio
async def test_fetch_chunk_backoff_doubles_on_repeated_429(historian, http_session):
    """asyncio.sleep must be called with 15s on the first retry and 30s on the
    second — the backoff must double, not stay flat."""
    from aioresponses import aioresponses
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock, call
    from src.data.historian import _RETRY_BACKOFF_S, _MAX_RETRIES

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        # All 3 attempts return 429 so we exhaust retries and collect all sleeps
        for _ in range(_MAX_RETRIES):
            m.get(re.compile(r"https://api\.twelvedata\.com/time_series"), status=429)

        with pytest.raises(Exception):  # HistorianError after exhaustion
            await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )

    sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
    # The rate-limit sleep (no-op on first call) may appear; filter to backoff sleeps
    backoff_sleeps = [s for s in sleep_calls if s >= _RETRY_BACKOFF_S]
    assert len(backoff_sleeps) >= 2, "Expected at least two backoff sleeps"
    assert backoff_sleeps[1] == backoff_sleeps[0] * 2, "Second backoff must double the first"


@pytest.mark.asyncio
async def test_fetch_chunk_retries_on_server_timeout(historian, http_session):
    """aiohttp.ServerTimeoutError inherits from ClientError and must trigger
    the same retry path as a connection error."""
    from aioresponses import aioresponses
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock
    from src.data.historian import HistorianError, _MAX_RETRIES
    import aiohttp

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ):
        for _ in range(_MAX_RETRIES):
            m.get(
                re.compile(r"https://api\.twelvedata\.com/time_series"),
                exception=aiohttp.ServerTimeoutError(),
            )

        with pytest.raises(HistorianError):
            await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )


@pytest.mark.asyncio
async def test_fetch_chunk_retries_on_client_error(historian, http_session):
    from aioresponses import aioresponses, CallbackResult
    from datetime import datetime, timezone
    from unittest.mock import patch, AsyncMock
    from src.data.historian import HistorianError, _MAX_RETRIES
    import aiohttp

    with aioresponses() as m, patch(
        "src.data.historian.asyncio.sleep", new_callable=AsyncMock
    ):

        for _ in range(_MAX_RETRIES):
            m.get(
                re.compile(r"https://api\.twelvedata\.com/time_series"),
                exception=aiohttp.ClientConnectionError("refused"),
            )

        with pytest.raises(HistorianError):
            await historian._fetch_chunk(
                session=http_session,
                api_symbol="EUR/USD",
                symbol="EUR_USD",
                start_dt=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 4, 12, 23, 59, tzinfo=timezone.utc),
            )
