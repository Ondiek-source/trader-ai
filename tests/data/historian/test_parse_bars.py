def test_parse_bars_valid_values(historian, api_values):
    bars = historian._parse_bars("EUR_USD", api_values)
    assert len(bars) == 3
    for bar in bars:
        assert bar.symbol == "EUR_USD"
        assert bar.timeframe.value == "M1"
        assert bar.is_complete is True


def test_parse_bars_output_is_chronological(historian, api_values):
    # api_values is newest-first (descending timestamps)
    bars = historian._parse_bars("EUR_USD", api_values)
    timestamps = [b.timestamp for b in bars]
    assert timestamps == sorted(timestamps)


def test_parse_bars_volume_floor_applied_on_zero(historian):
    values = [
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852",
            "volume": "0",
        }
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert len(bars) == 1
    assert bars[0].volume >= 1.0


def test_parse_bars_volume_floor_applied_on_missing_field(historian):
    values = [
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852",
            # "volume" key deliberately absent
        }
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert len(bars) == 1
    assert bars[0].volume >= 1.0


def test_parse_bars_skips_bad_datetime(historian):
    values = [
        {
            "datetime": "NOT-A-DATE",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852", "volume": "100",
        },
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852", "volume": "100",
        },
    ]
    bars = historian._parse_bars("EUR_USD", values)
    # Only the valid bar is returned
    assert len(bars) == 1


def test_parse_bars_skips_non_numeric_price(historian):
    values = [
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "N/A", "high": "1.0855",   # N/A is a Twelve Data sentinel
            "low": "1.0848", "close": "1.0852", "volume": "100",
        }
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert len(bars) == 0


def test_parse_bars_skips_ohlc_violation(historian):
    values = [
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "1.0850",
            "high": "1.0840",   # high < open and high < low â€” invalid
            "low":  "1.0860",   # low > high â€” OHLC violation
            "close": "1.0852",
            "volume": "100",
        }
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert len(bars) == 0


def test_parse_bars_empty_values_returns_empty_list(historian):
    bars = historian._parse_bars("EUR_USD", [])
    assert bars == []


def test_parse_bars_logs_skip_count(historian, caplog):
    import logging
    values = [
        {"datetime": "BAD", "open": "1.0", "high": "1.1",
         "low": "1.0", "close": "1.05", "volume": "10"},
    ]
    with caplog.at_level(logging.WARNING, logger="src.data.historian"):
        historian._parse_bars("EUR_USD", values)

    assert any("Skipped" in r.message for r in caplog.records)


def test_parse_bars_all_malformed_returns_empty_list(historian):
    """When every bar in the chunk is malformed, the result is [] — not a crash."""
    values = [
        {
            "datetime": "NOT-A-DATE",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852", "volume": "100",
        },
        {
            "datetime": "2026-04-12 10:01:00",
            "open": "N/A", "high": "1.0855",    # non-numeric sentinel
            "low": "1.0848", "close": "1.0852", "volume": "100",
        },
        {
            "datetime": "2026-04-12 10:02:00",
            "open": "1.0850",
            "high": "1.0840",   # OHLC violation: high < low
            "low":  "1.0860",
            "close": "1.0852", "volume": "100",
        },
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert bars == []


def test_parse_bars_negative_volume_floored_to_min_volume(historian):
    """Negative volume (data anomaly) must be raised to _MIN_VOLUME, not stored
    as-is. max(negative, 1.0) == 1.0."""
    from src.data.historian import _MIN_VOLUME

    values = [
        {
            "datetime": "2026-04-12 10:00:00",
            "open": "1.0850", "high": "1.0855",
            "low": "1.0848", "close": "1.0852",
            "volume": "-50",
        }
    ]
    bars = historian._parse_bars("EUR_USD", values)
    assert len(bars) == 1
    assert bars[0].volume == _MIN_VOLUME