def test_save_bars_calls_save_bar_batch(historian, mock_storage, valid_bar):
    bars = [valid_bar, valid_bar]
    historian._save_bars("EUR_USD", bars)

    mock_storage.save_bar_batch.assert_called_once_with(bars)
    mock_storage.save_bar.assert_not_called()


def test_save_bars_returns_bar_count_on_success(historian, mock_storage, valid_bar):
    bars = [valid_bar, valid_bar, valid_bar]
    result = historian._save_bars("EUR_USD", bars)
    assert result == 3


def test_save_bars_returns_zero_on_storage_error(
    historian, mock_storage, valid_bar
):
    from data.storage import StorageError
    mock_storage.save_bar_batch.side_effect = StorageError("disk full")

    result = historian._save_bars("EUR_USD", [valid_bar])

    assert result == 0
    # Must not raise â€” the StorageError is caught


def test_save_bars_returns_zero_on_empty_list(historian, mock_storage):
    result = historian._save_bars("EUR_USD", [])
    assert result == 0
    mock_storage.save_bar_batch.assert_not_called()