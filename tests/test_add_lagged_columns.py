"""Tests for add_lagged_columns."""

import datetime as dt
import os
import sys

import polars as pl
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from tidyfinance.lagging import add_lagged_columns  # noqa: E402


def test_exact_lag_with_by_returns_correct_lagged_values():
    """Test exact lag with by returns correct lagged values."""
    dates = pl.date_range(
        dt.date(2023, 1, 1), dt.date(2023, 4, 1), "1mo", eager=True
    )
    data = pl.DataFrame(
        {
            "permno": [1] * 4 + [2] * 4,
            "date": dates.to_list() * 2,
            "size": [float(i) for i in range(1, 9)],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo", by="permno")
    g1 = result.filter(pl.col("permno") == 1)
    assert g1["size_lag"][0] is None
    assert g1["size_lag"][1] == 1
    assert g1["size_lag"][2] == 2


def test_exact_lag_without_by_returns_correct_values():
    """Test exact lag without by (by=None) returns correct values."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo")
    assert result["size_lag"][0] is None
    assert result["size_lag"][1] == 1
    assert result["size_lag"][2] == 2


def test_window_lag_handles_all_src_date_conditions():
    """Test window lag: NA, in-window, and below-lower-bound cases."""
    data = pl.DataFrame(
        {
            "date": [
                dt.date(2023, 1, 1),
                dt.date(2023, 2, 1),
                dt.date(2023, 6, 1),
            ],
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(data, cols="size", lag="1mo", max_lag="2mo")
    # Jan: no source at all -> null
    assert result["size_lag"][0] is None
    # Feb: src_date Jan within window [Dec, Jan] -> 1
    assert result["size_lag"][1] == 1
    # Jun: closest src is Feb which is below the [Apr, May] window -> null
    assert result["size_lag"][2] is None


def test_drop_na_skips_na_source_rows_in_window_lag():
    """Test drop_na: missing source rows are skipped when drop_na=True."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 5, 1), "1mo", eager=True
            ),
            "size": [1.0, None, None, 4.0, 5.0],
        }
    )
    r_keep = add_lagged_columns(data, cols="size", lag="1mo", max_lag="3mo")
    r_drop = add_lagged_columns(
        data, cols="size", lag="1mo", max_lag="3mo", drop_na=True
    )
    # Apr (index 3): window [Jan, Mar]; without drop_na, closest = Mar
    # (missing)
    assert r_keep["size_lag"][3] is None
    # Apr with drop_na: missing sources skipped, closest valid = Jan -> 1
    assert r_drop["size_lag"][3] == 1


def test_ff_adjustment_without_by_uses_year_grouping_only():
    """Test ff_adjustment without by uses year grouping only."""
    data = pl.DataFrame(
        {
            "date": [
                dt.date(2022, 6, 1),
                dt.date(2022, 12, 1),
                dt.date(2023, 6, 1),
            ],
            "size": [10.0, 20.0, 30.0],
        }
    )
    result = add_lagged_columns(
        data, cols="size", lag="6mo", ff_adjustment=True
    )
    # ff: 2022 -> Dec kept (20); 2023 -> Jun kept (30).
    # Shifted +6m: Jun-2023 (20), Dec-2023 (30).
    jun23 = result.filter(pl.col("date") == dt.date(2023, 6, 1))
    assert jun23["size_lag"][0] == 20


def test_non_default_date_col_is_respected():
    """Test non-default date_col uses the specified column name."""
    data = pl.DataFrame(
        {
            "my_date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
        }
    )
    result = add_lagged_columns(
        data, cols="size", lag="1mo", date_col="my_date"
    )
    assert result["size_lag"][0] is None
    assert result["size_lag"][1] == 1


def test_error_when_date_column_is_absent_from_data():
    """Test error when date column is absent from data."""
    with pytest.raises(ValueError, match="date"):
        add_lagged_columns(pl.DataFrame({"x": [1]}), cols="x", lag="1mo")


def test_error_when_lag_is_negative():
    """Test error when lag is negative."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="non-negative"):
        add_lagged_columns(data, cols="size", lag=-1)


def test_error_when_max_lag_is_less_than_lag():
    """Test error when max_lag is less than lag."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="max_lag"):
        add_lagged_columns(data, cols="size", lag="3mo", max_lag="1mo")


def test_error_when_requested_column_is_absent_from_data():
    """Test error when a requested column is absent from data."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(data, cols="no_such_col", lag="1mo")


def test_error_when_by_column_is_absent_from_data():
    """Test error when a by column is absent from data."""
    data = pl.DataFrame({"date": [dt.date(2023, 1, 1)], "size": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        add_lagged_columns(data, cols="size", lag="1mo", by="no_such_grp")


def test_error_when_join_key_is_not_unique():
    """Test error when join key is not unique."""
    data = pl.DataFrame(
        {
            "date": [dt.date(2023, 1, 1)] * 2,
            "size": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        add_lagged_columns(data, cols="size", lag="1mo")


def test_error_when_upper_helper_column_already_exists():
    """Test error when '_upper' helper column already exists in data."""
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
            "_upper": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="_upper"):
        add_lagged_columns(data, cols="size", lag="1mo", max_lag="2mo")


def test_data_options_specifies_date_column_name():
    """Test data_options dict specifies the date column name."""
    from tidyfinance.portfolios import data_options

    data = pl.DataFrame(
        {
            "my_date": pl.date_range(
                dt.date(2023, 1, 1), dt.date(2023, 3, 1), "1mo", eager=True
            ),
            "size": [1.0, 2.0, 3.0],
        }
    )
    opts = data_options(date="my_date")
    result = add_lagged_columns(data, cols="size", lag="1mo", data_options=opts)
    assert result["size_lag"][0] is None
    assert result["size_lag"][1] == 1


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
