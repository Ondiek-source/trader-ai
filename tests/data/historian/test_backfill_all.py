import pytest


@pytest.mark.asyncio
async def test_backfill_all_processes_all_pairs(historian, mock_settings):
    from unittest.mock import patch, AsyncMock

    mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]

    with patch.object(
        historian, "backfill", new_callable=AsyncMock, return_value=100
    ) as mock_backfill:
        await historian.backfill_all()

    assert mock_backfill.call_count == 3
    called_symbols = [c[0][0] for c in mock_backfill.call_args_list]
    assert "EUR_USD" in called_symbols
    assert "GBP_USD" in called_symbols
    assert "USD_JPY" in called_symbols


@pytest.mark.asyncio
async def test_backfill_all_returns_correct_results_dict(historian, mock_settings):
    from unittest.mock import patch, AsyncMock

    mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]

    side_effects = [200, 150]
    with patch.object(
        historian, "backfill",
        new_callable=AsyncMock,
        side_effect=side_effects,
    ):
        results = await historian.backfill_all()

    assert results == {"EUR_USD": 200, "GBP_USD": 150}


@pytest.mark.asyncio
async def test_backfill_all_continues_after_historian_error(
    historian, mock_settings
):
    from unittest.mock import patch, AsyncMock
    from src.data.historian import HistorianError

    mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]

    def backfill_side_effect(symbol):
        if symbol == "EUR_USD":
            raise HistorianError("timeout", symbol=symbol)
        return 180

    with patch.object(historian, "backfill", side_effect=backfill_side_effect):
        results = await historian.backfill_all()

    assert results["EUR_USD"] == 0    # failed pair
    assert results["GBP_USD"] == 180  # succeeded despite EUR_USD failure


@pytest.mark.asyncio
async def test_backfill_all_empty_pairs_returns_empty_dict(
    historian, mock_settings
):
    """When backfill_pairs is empty, backfill_all() must return {} immediately
    without calling backfill() at all."""
    from unittest.mock import patch, AsyncMock

    mock_settings.backfill_pairs = []

    with patch.object(
        historian, "backfill", new_callable=AsyncMock
    ) as mock_backfill:
        results = await historian.backfill_all()

    assert results == {}
    mock_backfill.assert_not_called()


@pytest.mark.asyncio
async def test_backfill_all_processes_pairs_sequentially(
    historian, mock_settings
):
    import asyncio
    from unittest.mock import patch, AsyncMock

    mock_settings.backfill_pairs = ["EUR_USD", "GBP_USD"]
    call_order = []

    async def ordered_backfill(symbol):
        call_order.append(symbol)
        await asyncio.sleep(0)   # yield to event loop
        return 0

    with patch.object(historian, "backfill", side_effect=ordered_backfill):
        await historian.backfill_all()

    # Sequential means the order matches the config list exactly
    assert call_order == ["EUR_USD", "GBP_USD"]