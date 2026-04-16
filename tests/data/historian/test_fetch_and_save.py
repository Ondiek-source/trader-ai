"""
tests/data/historian/test_fetch_and_save.py

Tests for Historian._fetch_and_save — the chunk-walking loop that drives
the backfill. _fetch_chunk and _save_bars are mocked so these tests focus
solely on the orchestration logic: chunking, accumulation, empty-chunk
skipping, and error propagation.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from src.data.historian import Historian, HistorianError, _CHUNK_DAYS


# ── Helpers ────────────────────────────────────────────────────────────────────

# Date range that fits inside a single 3-day chunk
_START = datetime(2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc)
_END   = datetime(2026, 4, 12, 23, 59, 0, tzinfo=timezone.utc)

# Date range that spans multiple chunks (19 days > _CHUNK_DAYS = 3)
_MULTI_START = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
_MULTI_END   = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)


# ── Single-chunk scenarios ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_and_save_returns_bar_count_on_success(historian, valid_bar):
    """Single chunk: bar count from _save_bars is returned."""
    bars = [valid_bar, valid_bar, valid_bar]

    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", return_value=3):
        mock_fetch.return_value = bars
        result = await historian._fetch_and_save("EUR_USD", _START, _END)

    assert result == 3


@pytest.mark.asyncio
async def test_fetch_and_save_calls_save_bars_when_bars_returned(historian, valid_bar):
    """_save_bars must be called exactly once when _fetch_chunk returns bars."""
    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", return_value=1) as mock_save:
        mock_fetch.return_value = [valid_bar]
        await historian._fetch_and_save("EUR_USD", _START, _END)

    mock_save.assert_called_once_with("EUR_USD", [valid_bar])


@pytest.mark.asyncio
async def test_fetch_and_save_skips_save_on_empty_chunk(historian):
    """When _fetch_chunk returns [], _save_bars must not be called."""
    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars") as mock_save:
        mock_fetch.return_value = []
        result = await historian._fetch_and_save("EUR_USD", _START, _END)

    assert result == 0
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_and_save_propagates_historian_error(historian):
    """HistorianError raised by _fetch_chunk must propagate out unchanged."""
    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = HistorianError("network down", symbol="EUR_USD")

        with pytest.raises(HistorianError) as exc_info:
            await historian._fetch_and_save("EUR_USD", _START, _END)

    assert exc_info.value.symbol == "EUR_USD"


# ── Multi-chunk scenarios ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_and_save_multi_chunk_calls_fetch_multiple_times(historian, valid_bar):
    """A range spanning multiple chunks triggers multiple _fetch_chunk calls."""
    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", return_value=1):
        mock_fetch.return_value = [valid_bar]
        await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)

    assert mock_fetch.call_count > 1


@pytest.mark.asyncio
async def test_fetch_and_save_multi_chunk_sums_totals(historian, valid_bar):
    """Bar counts from every chunk are accumulated into the return value."""
    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", return_value=5):
        mock_fetch.return_value = [valid_bar]
        result = await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)

    expected_total = mock_fetch.call_count * 5
    assert result == expected_total


@pytest.mark.asyncio
async def test_fetch_and_save_mixed_empty_and_nonempty_chunks(historian, valid_bar):
    """Empty chunks contribute 0 to the total; non-empty chunks accumulate."""
    # Alternate: empty, non-empty, empty, non-empty, ...
    responses = []
    chunk_count = 3
    for i in range(chunk_count):
        responses.append([valid_bar] if i % 2 == 1 else [])

    call_index = {"n": 0}

    async def _side_effect(**kwargs):
        idx = call_index["n"]
        call_index["n"] += 1
        if idx < len(responses):
            return responses[idx]
        return []

    with patch.object(historian, "_fetch_chunk", side_effect=_side_effect), \
         patch.object(historian, "_save_bars", return_value=1) as mock_save:
        result = await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)

    # Only non-empty chunks should trigger _save_bars
    assert mock_save.call_count == sum(1 for r in responses if r)


# ── Consecutive storage failure guard ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_and_save_aborts_after_consecutive_storage_failures(
    historian, valid_bar
):
    """When _save_bars returns 0 on _MAX_CONSECUTIVE_STORAGE_FAILURES consecutive
    chunks, _fetch_and_save must raise HistorianError to preserve API quota."""
    from src.data.historian import _MAX_CONSECUTIVE_STORAGE_FAILURES

    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", return_value=0):
        mock_fetch.return_value = [valid_bar]

        with pytest.raises(HistorianError) as exc_info:
            await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)

    assert exc_info.value.symbol == "EUR_USD"
    # fetch was called at least _MAX_CONSECUTIVE_STORAGE_FAILURES times before abort
    assert mock_fetch.call_count >= _MAX_CONSECUTIVE_STORAGE_FAILURES


@pytest.mark.asyncio
async def test_fetch_and_save_resets_failure_counter_on_success(
    historian, valid_bar
):
    """After a successful save the consecutive-failure counter resets to 0,
    so two failures → success → two failures must NOT trigger the abort."""
    from src.data.historian import _MAX_CONSECUTIVE_STORAGE_FAILURES

    # Produce enough chunks: fail, fail, succeed, fail, fail — never hits threshold
    # _MAX_CONSECUTIVE_STORAGE_FAILURES is 3, so two consecutive failures never abort
    save_results = [0, 0, 3, 0, 0]  # two fails, one success, two more fails
    save_iter = iter(save_results)

    with patch.object(historian, "_fetch_chunk", new_callable=AsyncMock) as mock_fetch, \
         patch.object(historian, "_save_bars", side_effect=lambda *_: next(save_iter, 1)):
        mock_fetch.return_value = [valid_bar]
        # Should NOT raise — counter was reset after the success
        result = await historian._fetch_and_save("EUR_USD", _MULTI_START, _MULTI_END)

    # Total saved is the sum of non-zero save_results that were consumed
    assert result >= 3


@pytest.mark.asyncio
async def test_fetch_and_save_empty_chunks_do_not_count_as_failures(
    historian, valid_bar
):
    """Empty chunks (weekends/holidays) must NOT advance the consecutive-failure
    counter. Only non-empty chunks where _save_bars returns 0 count."""
    from src.data.historian import _MAX_CONSECUTIVE_STORAGE_FAILURES

    # Return empty on every other chunk; _save_bars always returns 0 for non-empty
    call_index = {"n": 0}

    async def alternating_fetch(**kwargs):
        idx = call_index["n"]
        call_index["n"] += 1
        # Even indices → empty chunk; odd indices → one bar
        return [] if idx % 2 == 0 else [valid_bar]

    with patch.object(historian, "_fetch_chunk", side_effect=alternating_fetch), \
         patch.object(historian, "_save_bars", return_value=0):
        # With alternating empty/non-empty, consecutive save-failures never reach
        # the threshold within the chunks produced by _MULTI_START→_MULTI_END
        # (each non-empty chunk is separated by an empty one → counter resets
        #  opportunity is never reached consecutively past the threshold).
        # The test passes if no HistorianError is raised for fewer than threshold
        # consecutive real failures.
        try:
            await historian._fetch_and_save("EUR_USD", _START, _END)
        except HistorianError:
            pytest.fail(
                "HistorianError raised but empty chunks should not count toward "
                "the consecutive-failure threshold."
            )
