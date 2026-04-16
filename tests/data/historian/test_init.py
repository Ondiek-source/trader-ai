def test_historian_constructs(historian):
    assert historian is not None


def test_historian_settings_wired(historian, mock_settings):
    assert historian._settings is mock_settings


def test_historian_storage_wired(historian, mock_storage):
    assert historian._storage is mock_storage


def test_historian_last_request_time_initialises_to_zero(historian):
    assert historian._last_request_time == 0.0


def test_get_historian_returns_same_instance():
    """get_historian() must be a true singleton — repeated calls return the
    exact same object, not a new Historian on every call."""
    import src.data.historian as historian_module
    from unittest.mock import MagicMock, patch

    # Reset the module-level singleton so this test is isolated
    original = historian_module._historian
    historian_module._historian = None

    try:
        mock_settings = MagicMock()
        mock_settings.data_mode = "LOCAL"

        with patch("src.data.historian.get_settings", return_value=mock_settings), \
             patch("src.data.historian.Storage"):
            h1 = historian_module.get_historian()
            h2 = historian_module.get_historian()

        assert h1 is h2
    finally:
        # Restore the original singleton so other tests are unaffected
        historian_module._historian = original


def test_get_historian_shares_rate_limit_state():
    """Both callers of get_historian() must operate on the same
    _last_request_time — a second independent Historian() would break the
    rate-limit guarantee."""
    import src.data.historian as historian_module
    from unittest.mock import MagicMock, patch

    original = historian_module._historian
    historian_module._historian = None

    try:
        mock_settings = MagicMock()
        mock_settings.data_mode = "LOCAL"

        with patch("src.data.historian.get_settings", return_value=mock_settings), \
             patch("src.data.historian.Storage"):
            h1 = historian_module.get_historian()
            h1._last_request_time = 999.0
            h2 = historian_module.get_historian()

        assert h2._last_request_time == 999.0
    finally:
        historian_module._historian = original