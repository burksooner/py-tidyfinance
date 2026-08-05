"""Tests for download_data_stock_prices."""

import os
import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.download_open_source import (  # noqa: E402
    _download_data_stock_prices,
)


def test_symbols_must_be_a_character_vector_without_missing_values():
    """Test symbols must be a character vector without missing values."""
    with pytest.raises((ValueError, TypeError)):
        _download_data_stock_prices(1)

    with pytest.raises((ValueError, TypeError)):
        _download_data_stock_prices(["AAPL", None])


def test_downloads_data_replaces_null_and_warns_on_failures():
    """Test downloads data, replaces NULL, and warns on failures."""
    success_body = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577836800, 1577923200],
                    "indicators": {
                        "quote": [
                            {
                                "volume": [100, None],
                                "open": [10, 11],
                                "low": [9, 10],
                                "high": [12, 13],
                                "close": [11, 12],
                            }
                        ],
                        "adjclose": [{"adjclose": [11, None]}],
                    },
                }
            ]
        }
    }
    fail_body = {
        "chart": {
            "error": {
                "code": "Not Found",
                "description": "No data found",
            }
        }
    }

    def fake_get(url, *a, **kw):
        resp = MagicMock()
        if "FAIL" in url:
            resp.status_code = 404
            resp.json.return_value = fail_body
        else:
            resp.status_code = 200
            resp.json.return_value = success_body
        return resp

    with patch(
        "tidyfinance.download_open_source.requests.get", side_effect=fake_get
    ):
        with pytest.warns(
            UserWarning, match="Failed to retrieve data for symbol FAIL"
        ):
            out = _download_data_stock_prices(
                ["AAPL", "FAIL"], "2020-01-01", "2020-01-03"
            )

    assert isinstance(out, pl.DataFrame)
    # Two AAPL rows are kept; the row with a NULL upstream value
    # retains a null rather than being dropped. FAIL is silently
    # skipped.
    assert out["symbol"].to_list() == ["AAPL", "AAPL"]
    assert out["open"].to_list() == [10, 11]
    assert out["low"].to_list() == [9, 10]
    assert out["high"].to_list() == [12, 13]
    assert out["close"].to_list() == [11, 12]
    assert out["volume"][0] == 100
    assert out["volume"][1] is None
    assert out["adjusted_close"][0] == 11
    assert out["adjusted_close"][1] is None


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
