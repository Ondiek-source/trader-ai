def test_determine_start_first_run_midnight(historian, mock_storage):
    from datetime import datetime, timezone, timedelta
    mock_storage.get_bars.return_value = None
    now = datetime(2026, 4, 13, 14, 30, 0, tzinfo=timezone.utc)

    start = historian._determine_start("EUR_USD", now)

    expected = now - timedelta(days=365 * 2)
    expected = expected.replace(hour=0, minute=0, second=0, microsecond=0)
    assert start == expected


def test_determine_start_resume_one_minute_after_last_bar(
    historian, mock_storage
):
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    last_ts = datetime(2026, 4, 12, 23, 59, 0)
    df = pd.DataFrame({"timestamp": [last_ts]})
    mock_storage.get_bars.return_value = df

    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    start = historian._determine_start("EUR_USD", now)

    expected = last_ts.replace(tzinfo=timezone.utc) + timedelta(minutes=1)
    assert start == expected


def test_determine_start_attaches_utc_to_naive_timestamp(
    historian, mock_storage
):
    import pandas as pd
    from datetime import datetime, timezone

    naive_ts = datetime(2026, 4, 10, 8, 0, 0)   # no tzinfo
    df = pd.DataFrame({"timestamp": [naive_ts]})
    mock_storage.get_bars.return_value = df

    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    start = historian._determine_start("EUR_USD", now)

    # Must not raise; result must be timezone-aware
    assert start.tzinfo is not None


def test_determine_start_respects_backfill_years(
    historian, mock_settings, mock_storage
):
    from datetime import datetime, timezone, timedelta
    mock_settings.backfill_years = 1
    mock_storage.get_bars.return_value = None

    now = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
    start = historian._determine_start("EUR_USD", now)

    expected_date = (now - timedelta(days=365)).date()
    assert start.date() == expected_date


def test_determine_start_empty_dataframe_treated_as_first_run(
    historian, mock_storage
):
    """get_bars() returning an empty DataFrame must trigger full backfill,
    not a resume — the code guards `if df is None or df.empty`."""
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    mock_storage.get_bars.return_value = pd.DataFrame()   # empty, not None

    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    start = historian._determine_start("EUR_USD", now)

    # Must behave identically to the None case: midnight-normalised cutoff
    expected = (now - timedelta(days=365 * 2)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    assert start == expected


def test_determine_start_pandas_timestamp_converted_correctly(
    historian, mock_storage
):
    """Parquet reads return pd.Timestamp objects, not plain datetimes.
    The hasattr(raw_ts, 'to_pydatetime') branch must handle them correctly."""
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    # pd.Timestamp is what pandas returns when reading a Parquet timestamp column
    pd_ts = pd.Timestamp("2026-04-10 08:00:00")        # naive pd.Timestamp
    df = pd.DataFrame({"timestamp": [pd_ts]})
    mock_storage.get_bars.return_value = df

    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    start = historian._determine_start("EUR_USD", now)

    expected = pd_ts.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=1)
    assert start == expected
    assert start.tzinfo is not None