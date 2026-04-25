def test_historian_error_stores_symbol():
    from src.data.historian import HistorianError
    err = HistorianError("fetch failed", symbol="EUR_USD")
    assert err.symbol == "EUR_USD"
    assert "fetch failed" in str(err)


def test_historian_error_default_symbol():
    from src.data.historian import HistorianError
    err = HistorianError("generic failure")
    assert err.symbol == ""
    assert isinstance(err, Exception)